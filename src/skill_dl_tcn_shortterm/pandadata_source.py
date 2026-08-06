"""PandaData ingestion boundary for bounded, point-in-time research datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol, Sequence, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .experiment import ContractError


_VENDOR_MINUTE_COLUMNS = {
    "symbol",
    "date",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}
_VENDOR_WEIGHT_COLUMNS = {"index_symbol", "stock_symbol", "date", "weight"}
_VENDOR_DAILY_COLUMNS = {
    "symbol",
    "date",
    "name",
    "close",
    "amount",
    "trade_status",
}
_VENDOR_SHARE_COLUMNS = {
    "symbol",
    "date",
    "total",
    "circulation_a",
    "free_circulation",
}
_VENDOR_ADJUSTMENT_COLUMNS = {
    "symbol",
    "ex_date",
    "announcement_date",
    "ex_cum_factor",
    "ex_factor",
    "ex_end_date",
}
_SUFFIXES = {"SZ": "XSHE", "SH": "XSHG"}


@dataclass(frozen=True)
class PandaDataFetchChunk:
    """One bounded, independently resumable provider request."""

    start_date: str
    end_date: str
    symbols: tuple[str, ...]
    chunk_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "symbols": list(self.symbols),
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class PandaDataCoverageAudit:
    """Receipt-ready partition coverage needed by contiguous TCN windows."""

    daily_coverage: pd.DataFrame
    candidate_signal_days: pd.DataFrame
    status: str
    blockers: tuple[str, ...]
    parameters: dict[str, object]
    source_identity: dict[str, object]


class PandaDataMinuteAPI(Protocol):
    """Narrow external seam needed by the partitioned downloader."""

    def get_stock_min(self, **kwargs: object) -> pd.DataFrame: ...


class PandaDataWeightAPI(Protocol):
    """Narrow external seam needed to build contemporaneous index membership."""

    def get_index_weights(self, **kwargs: object) -> pd.DataFrame: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _instrument_id(vendor_symbol: object) -> str:
    text = str(vendor_symbol)
    parts = text.split(".")
    if len(parts) != 2 or len(parts[0]) != 6 or parts[1] not in _SUFFIXES:
        raise ContractError(f"unsupported PandaData A-share symbol: {text}")
    return f"{parts[0]}.{_SUFFIXES[parts[1]]}"


def _iso_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be an ISO date") from exc


def _vendor_dates(values: pd.Series, field: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, format="%Y%m%d", errors="raise")
    except Exception as exc:
        raise ContractError(f"invalid PandaData {field}: {exc}") from exc
    return parsed.dt.date


def canonicalize_pandadata_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize historical daily names, suspension state, and turnover."""

    missing = sorted(_VENDOR_DAILY_COLUMNS.difference(frame.columns))
    if missing:
        raise ContractError(f"PandaData daily frame missing columns: {', '.join(missing)}")
    columns = [
        "instrument_id",
        "trade_date",
        "name",
        "close",
        "amount",
        "is_st",
        "is_suspended",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame()
    result["instrument_id"] = frame["symbol"].map(_instrument_id)
    result["trade_date"] = _vendor_dates(frame["date"], "daily date")
    result["name"] = frame["name"].map(lambda value: str(value).strip())
    result["close"] = pd.to_numeric(frame["close"], errors="raise").astype("float64")
    result["amount"] = pd.to_numeric(frame["amount"], errors="raise").astype("float64")
    result["is_st"] = result["name"].str.match(
        r"^(?:S\*ST|SST|\*ST|ST)", case=False, na=False
    )
    status = pd.to_numeric(frame["trade_status"], errors="raise")
    result["is_suspended"] = status.ne(0)
    if (result[["close", "amount"]].lt(0)).any(axis=None):
        raise ContractError("PandaData daily prices and amount cannot be negative")
    if result.duplicated(["instrument_id", "trade_date"]).any():
        raise ContractError("PandaData daily keys must be unique")
    return result.sort_values(["instrument_id", "trade_date"], kind="mergesort").reset_index(drop=True)


def canonicalize_pandadata_share_float(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize announcement-dated share capital without future backfill."""

    missing = sorted(_VENDOR_SHARE_COLUMNS.difference(frame.columns))
    if missing:
        raise ContractError(
            f"PandaData share-float frame missing columns: {', '.join(missing)}"
        )
    columns = [
        "instrument_id",
        "known_date",
        "total_shares",
        "circulation_a",
        "free_circulation",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame()
    result["instrument_id"] = frame["symbol"].map(_instrument_id)
    result["known_date"] = _vendor_dates(frame["date"], "share-float date")
    for source, target in [
        ("total", "total_shares"),
        ("circulation_a", "circulation_a"),
        ("free_circulation", "free_circulation"),
    ]:
        result[target] = pd.to_numeric(frame[source], errors="coerce").astype("float64")
    if result.duplicated(["instrument_id", "known_date"]).any():
        raise ContractError("PandaData share-float keys must be unique")
    return result.sort_values(["instrument_id", "known_date"], kind="mergesort").reset_index(drop=True)


def canonicalize_pandadata_adjustments(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a fail-closed corporate-action ledger from sparse factor events."""

    missing = sorted(_VENDOR_ADJUSTMENT_COLUMNS.difference(frame.columns))
    if missing:
        raise ContractError(
            f"PandaData adjustment frame missing columns: {', '.join(missing)}"
        )
    columns = [
        "instrument_id",
        "effective_date",
        "known_date",
        "ex_cum_factor",
        "ex_factor",
        "ex_end_date",
        "pit_reliable",
        "adjustment_policy",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame()
    result["instrument_id"] = frame["symbol"].map(_instrument_id)
    result["effective_date"] = _vendor_dates(frame["ex_date"], "ex-date")
    result["known_date"] = _vendor_dates(
        frame["announcement_date"], "adjustment announcement date"
    )
    result["ex_cum_factor"] = pd.to_numeric(
        frame["ex_cum_factor"], errors="raise"
    ).astype("float64")
    result["ex_factor"] = pd.to_numeric(frame["ex_factor"], errors="raise").astype(
        "float64"
    )
    result["ex_end_date"] = pd.to_datetime(
        frame["ex_end_date"], format="%Y%m%d", errors="coerce"
    ).dt.date
    result["pit_reliable"] = False
    result["adjustment_policy"] = "unadjusted-minute-label-invalidation-v1"
    if result.duplicated(["instrument_id", "effective_date"]).any():
        raise ContractError("PandaData adjustment keys must be unique")
    return result.sort_values(
        ["instrument_id", "effective_date"], kind="mergesort"
    ).reset_index(drop=True)


def merge_pit_enrichment_frames(
    reused: pd.DataFrame,
    fetched: pd.DataFrame,
    *,
    keys: list[str],
    name: str,
) -> pd.DataFrame:
    """Merge exact PIT rows while failing closed on conflicting business keys."""

    if reused.empty:
        return fetched.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if fetched.empty:
        return reused.sort_values(keys, kind="mergesort").reset_index(drop=True)
    missing = sorted(
        set(keys).difference(reused.columns) | set(keys).difference(fetched.columns)
    )
    if missing:
        raise ContractError(f"{name} merge keys missing: {', '.join(missing)}")
    columns = sorted(set(reused.columns) | set(fetched.columns))
    combined = pd.concat(
        [reused.reindex(columns=columns), fetched.reindex(columns=columns)],
        ignore_index=True,
    )
    exact = combined.drop_duplicates(ignore_index=True)
    if exact.duplicated(keys, keep=False).any():
        raise ContractError(f"{name} contains conflicting reused and fetched PIT keys")
    return exact.sort_values(keys, kind="mergesort").reset_index(drop=True)


def build_pandadata_causal_states(
    daily: pd.DataFrame,
    *,
    membership: pd.DataFrame,
    share_float: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build close-of-day PIT states with a strictly trailing ADV20."""

    required_daily = {
        "instrument_id",
        "trade_date",
        "close",
        "amount",
        "is_st",
        "is_suspended",
    }
    if missing := sorted(required_daily.difference(daily.columns)):
        raise ContractError(f"canonical daily frame missing columns: {', '.join(missing)}")
    required_membership = {"instrument_id", "trade_date"}
    if missing := sorted(required_membership.difference(membership.columns)):
        raise ContractError(f"membership frame missing columns: {', '.join(missing)}")
    selected = daily.merge(
        membership[list(required_membership)].drop_duplicates(),
        on=["instrument_id", "trade_date"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["instrument_id", "trade_date"], kind="mergesort")
    if selected.empty:
        raise ContractError("causal state input has no PIT member dates")
    shares = share_float if share_float is not None else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for instrument_id, group in selected.groupby("instrument_id", observed=True):
        history = group.sort_values("trade_date", kind="mergesort").copy()
        history["adv20"] = history["amount"].rolling(20, min_periods=20).mean()
        first_observed = cast(date, history["trade_date"].iloc[0])
        share_history = shares.loc[
            shares.get("instrument_id", pd.Series(dtype=str)).eq(instrument_id)
        ].sort_values("known_date") if not shares.empty else pd.DataFrame()
        for raw_item in history.itertuples(index=False):
            item = cast(Any, raw_item)
            trade_date = cast(date, getattr(item, "trade_date"))
            total_shares = float("nan")
            if not share_history.empty:
                known = share_history.loc[share_history["known_date"] <= trade_date]
                if not known.empty:
                    total_shares = float(known.iloc[-1]["total_shares"])
            close = float(getattr(item, "close"))
            market_cap = close * total_shares if pd.notna(total_shares) else float("nan")
            exchange = str(instrument_id).rsplit(".", 1)[1]
            suspended = bool(getattr(item, "is_suspended"))
            rows.append(
                {
                    "instrument_id": str(instrument_id),
                    "effective_at": pd.Timestamp(
                        f"{trade_date} 15:00:00", tz="Asia/Shanghai"
                    ),
                    "exchange": exchange,
                    "security_type": "A_SHARE",
                    "listed_date": first_observed,
                    "delisted_date": None,
                    "is_st": bool(getattr(item, "is_st")),
                    "is_delisting": False,
                    "is_suspended": suspended,
                    "industry": "unavailable",
                    "market_cap": market_cap,
                    "adv20": float(getattr(item, "adv20")),
                    "market_state": "suspended" if suspended else "normal",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["instrument_id", "effective_at"], kind="mergesort"
    ).reset_index(drop=True)


def canonicalize_pandadata_minutes(
    frame: pd.DataFrame, *, source_version: str
) -> pd.DataFrame:
    """Convert PandaData's local, reverse-ordered minute frame to raw_1m."""

    if not source_version:
        raise ContractError("source_version must be a non-empty string")
    missing = sorted(_VENDOR_MINUTE_COLUMNS.difference(frame.columns))
    if missing:
        raise ContractError(
            f"PandaData minute frame missing columns: {', '.join(missing)}"
        )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "instrument_id",
                "bar_end_at",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "quality_flag",
            ]
        )

    bars = frame.copy()
    bars["instrument_id"] = bars["symbol"].map(_instrument_id)
    try:
        timestamps = pd.to_datetime(bars["datetime"], errors="raise")
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(
                "Asia/Shanghai", ambiguous="raise", nonexistent="raise"
            )
        else:
            timestamps = timestamps.dt.tz_convert("Asia/Shanghai")
    except Exception as exc:
        raise ContractError(f"invalid PandaData datetime values: {exc}") from exc
    bars["bar_end_at"] = timestamps

    provider_dates = pd.to_datetime(
        bars["date"].astype(str), format="%Y%m%d", errors="raise"
    ).dt.date
    if not provider_dates.eq(bars["bar_end_at"].dt.date).all():
        raise ContractError("PandaData date and datetime fields disagree")

    numeric = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    if bars[numeric].isna().any(axis=None):
        raise ContractError("PandaData minute numeric fields cannot be null")
    if (bars[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ContractError("PandaData minute prices must be positive")
    if (bars[["volume", "amount"]] < 0).any(axis=None):
        raise ContractError("PandaData minute volume and amount cannot be negative")

    bars["quality_flag"] = "ok"
    result = bars[
        [
            "instrument_id",
            "bar_end_at",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "quality_flag",
        ]
    ].sort_values(["instrument_id", "bar_end_at"], kind="mergesort")
    if result.duplicated(["instrument_id", "bar_end_at"]).any():
        raise ContractError("PandaData minute frame contains duplicate bar keys")
    return result.reset_index(drop=True)


def select_daily_top_weight_universe(
    weights: pd.DataFrame, *, top_n: int
) -> pd.DataFrame:
    """Select a stable top-N independently on every PIT index-weight date."""

    if top_n < 1:
        raise ContractError("top_n must be positive")
    missing = sorted(_VENDOR_WEIGHT_COLUMNS.difference(weights.columns))
    if missing:
        raise ContractError(
            f"PandaData weight frame missing columns: {', '.join(missing)}"
        )
    if weights.empty:
        raise ContractError("PandaData weight frame must not be empty")

    universe = weights.copy()
    universe["trade_date"] = pd.to_datetime(
        universe["date"].astype(str), format="%Y%m%d", errors="raise"
    ).dt.date
    universe["weight"] = pd.to_numeric(universe["weight"], errors="coerce")
    if universe["weight"].isna().any() or (universe["weight"] < 0).any():
        raise ContractError("PandaData index weights must be finite and non-negative")
    if universe.duplicated(["trade_date", "stock_symbol"]).any():
        raise ContractError("PandaData weights contain duplicate date-symbol keys")

    universe = universe.sort_values(
        ["trade_date", "weight", "stock_symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    universe = universe.groupby("trade_date", sort=True, observed=True).head(top_n)
    universe["vendor_symbol"] = universe["stock_symbol"].astype(str)
    universe["instrument_id"] = universe["vendor_symbol"].map(_instrument_id)
    return universe[
        [
            "trade_date",
            "vendor_symbol",
            "instrument_id",
            "index_symbol",
            "weight",
        ]
    ].reset_index(drop=True)


def fetch_pandadata_pit_universe(
    api: PandaDataWeightAPI,
    *,
    index_symbol: str,
    start_date: str,
    end_date: str,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch index weights in bounded calendar-month calls and select daily PIT top-N."""

    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date")
    if start > end:
        raise ContractError("start_date cannot be after end_date")
    frames: list[pd.DataFrame] = []
    for period in pd.period_range(start=start, end=end, freq="M"):
        window_start = max(start, period.start_time.date())
        window_end = min(end, period.end_time.date())
        frame = api.get_index_weights(
            index_symbol=index_symbol,
            stock_symbol=None,
            start_date=window_start.strftime("%Y%m%d"),
            end_date=window_end.strftime("%Y%m%d"),
            fields=[],
        )
        frames.append(frame)
    weights = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    universe = select_daily_top_weight_universe(weights, top_n=top_n)
    weights = weights.sort_values(
        ["date", "weight", "stock_symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return weights, universe


def build_monthly_fetch_plan(
    universe: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    symbol_batch_size: int,
) -> tuple[PandaDataFetchChunk, ...]:
    """Create deterministic month-and-symbol chunks from daily PIT membership."""

    if symbol_batch_size < 1:
        raise ContractError("symbol_batch_size must be positive")
    missing = {"trade_date", "vendor_symbol"}.difference(universe.columns)
    if missing:
        raise ContractError(
            f"PIT universe missing columns: {', '.join(sorted(missing))}"
        )
    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date")
    if start > end:
        raise ContractError("start_date cannot be after end_date")

    local = universe.copy()
    local["trade_date"] = pd.to_datetime(local["trade_date"], errors="raise").dt.date
    local = local.loc[local["trade_date"].between(start, end)]
    chunks: list[PandaDataFetchChunk] = []
    for period in pd.period_range(start=start, end=end, freq="M"):
        month_start = max(period.start_time.date(), start)
        month_end = min(period.end_time.date(), end)
        symbols = sorted(
            local.loc[
                local["trade_date"].between(month_start, month_end),
                "vendor_symbol",
            ]
            .astype(str)
            .unique()
        )
        for batch_index, offset in enumerate(range(0, len(symbols), symbol_batch_size)):
            batch: Sequence[str] = symbols[offset : offset + symbol_batch_size]
            chunks.append(
                PandaDataFetchChunk(
                    start_date=month_start.strftime("%Y%m%d"),
                    end_date=month_end.strftime("%Y%m%d"),
                    symbols=tuple(batch),
                    chunk_id=f"{period.strftime('%Y-%m')}-{batch_index:03d}",
                )
            )
    return tuple(chunks)


def _canonical_universe(universe: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "vendor_symbol",
        "instrument_id",
        "index_symbol",
        "weight",
    }
    missing = sorted(required.difference(universe.columns))
    if missing:
        raise ContractError(f"PIT universe missing columns: {', '.join(missing)}")
    local = universe[list(sorted(required))].copy()
    local["trade_date"] = pd.to_datetime(local["trade_date"], errors="raise").dt.date
    if local.duplicated(["trade_date", "instrument_id"]).any():
        raise ContractError("PIT universe contains duplicate date-instrument keys")
    return local.sort_values(
        ["trade_date", "weight", "vendor_symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _validated_existing_manifest(
    manifest_path: Path,
    *,
    source_version: str,
    plan: Sequence[PandaDataFetchChunk],
) -> Path:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read PandaData manifest: {exc}") from exc
    if payload.get("source_version") != source_version:
        raise ContractError("existing PandaData manifest source_version conflicts")
    if payload.get("parameters", {}).get("plan") != [item.to_dict() for item in plan]:
        raise ContractError("existing PandaData manifest fetch plan conflicts")
    for partition in payload.get("partitions", []):
        part_path = manifest_path.parent / str(partition.get("path", ""))
        if not part_path.is_file() or _sha256(part_path) != partition.get("sha256"):
            raise ContractError("existing PandaData partition fingerprint mismatch")
    return manifest_path


def download_pandadata_pilot(
    api: PandaDataMinuteAPI,
    *,
    universe: pd.DataFrame,
    plan: Sequence[PandaDataFetchChunk],
    output_dir: str | Path,
    source_version: str,
    index_weights_path: str | Path | None = None,
) -> Path:
    """Download immutable PIT-filtered raw_1m chunks with hash-verified resume."""

    if not source_version:
        raise ContractError("source_version must be a non-empty string")
    if not plan:
        raise ContractError("PandaData fetch plan must contain at least one chunk")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    parts_dir = destination / "parts"
    parts_dir.mkdir(exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        return _validated_existing_manifest(
            manifest_path, source_version=source_version, plan=plan
        )

    pit = _canonical_universe(universe)
    universe_path = destination / "pit-universe.parquet"
    if universe_path.exists():
        existing = pd.read_parquet(universe_path)
        existing["trade_date"] = pd.to_datetime(existing["trade_date"]).dt.date
        if not existing.equals(pit):
            raise ContractError("existing PIT universe conflicts with requested universe")
    else:
        temporary = universe_path.with_suffix(".parquet.tmp")
        pit.to_parquet(temporary, index=False)
        temporary.replace(universe_path)

    partitions: list[dict[str, object]] = []
    membership = pit[["trade_date", "instrument_id"]].drop_duplicates()
    for chunk in plan:
        part_path = parts_dir / f"{chunk.chunk_id}.parquet"
        receipt_path = parts_dir / f"{chunk.chunk_id}.json"
        if part_path.exists() != receipt_path.exists():
            raise ContractError(f"incomplete existing PandaData chunk: {chunk.chunk_id}")
        if part_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("request") != chunk.to_dict():
                raise ContractError(f"PandaData chunk request conflict: {chunk.chunk_id}")
            if receipt.get("sha256") != _sha256(part_path):
                raise ContractError(f"PandaData chunk fingerprint mismatch: {chunk.chunk_id}")
            partitions.append(receipt)
            continue

        vendor = api.get_stock_min(
            symbol=list(chunk.symbols),
            start_date=chunk.start_date,
            end_date=chunk.end_date,
            fields=[],
            frequency="1m",
        )
        bars = canonicalize_pandadata_minutes(
            vendor, source_version=source_version
        )
        if not bars.empty:
            bars["trade_date"] = bars["bar_end_at"].dt.date
            bars = bars.merge(
                membership,
                on=["trade_date", "instrument_id"],
                how="inner",
                validate="many_to_one",
            ).drop(columns="trade_date")
            bars = bars.sort_values(
                ["instrument_id", "bar_end_at"], kind="mergesort"
            ).reset_index(drop=True)

        temporary = part_path.with_suffix(".parquet.tmp")
        bars.to_parquet(temporary, index=False)
        temporary.replace(part_path)
        receipt = {
            "chunk_id": chunk.chunk_id,
            "path": part_path.relative_to(destination).as_posix(),
            "request": chunk.to_dict(),
            "row_count": len(bars),
            "instrument_count": int(bars["instrument_id"].nunique()),
            "sha256": _sha256(part_path),
        }
        _write_json_atomic(receipt_path, receipt)
        partitions.append(receipt)

    manifest = {
        "schema_version": 1,
        "dataset_kind": "raw_1m_partitioned",
        "timezone": "Asia/Shanghai",
        "price_unit": "CNY",
        "volume_unit": "share",
        "amount_unit": "CNY",
        "source_version": source_version,
        "universe_path": universe_path.name,
        "universe_sha256": _sha256(universe_path),
        "parameters": {
            "frequency": "1m",
            "membership_filter": "trade_date+instrument_id",
            "plan": [item.to_dict() for item in plan],
        },
        "summary": {
            "partition_count": len(partitions),
            "row_count": sum(cast(int, item["row_count"]) for item in partitions),
            "pit_date_count": int(pit["trade_date"].nunique()),
            "pit_instrument_count": int(pit["instrument_id"].nunique()),
        },
        "partitions": partitions,
    }
    if index_weights_path is not None:
        weights_path = Path(index_weights_path).expanduser().resolve()
        if not weights_path.is_file() or weights_path.parent != destination:
            raise ContractError(
                "index_weights_path must be an existing file inside output_dir"
            )
        manifest["index_weights_path"] = weights_path.name
        manifest["index_weights_sha256"] = _sha256(weights_path)
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def audit_pandadata_coverage(
    manifest_path: str | Path,
    *,
    min_complete_instruments: int,
    lookback_days: int,
    max_horizon_days: int,
    required_signal_days: int,
) -> PandaDataCoverageAudit:
    """Stream partition keys and audit complete stock-days and contiguous windows."""

    positive = {
        "min_complete_instruments": min_complete_instruments,
        "lookback_days": lookback_days,
        "max_horizon_days": max_horizon_days,
        "required_signal_days": required_signal_days,
    }
    if any(value < 1 for value in positive.values()):
        raise ContractError("all PandaData coverage thresholds must be positive")
    source = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read PandaData manifest: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ContractError("PandaData manifest schema_version must equal 1")
    if manifest.get("dataset_kind") != "raw_1m_partitioned":
        raise ContractError("PandaData manifest must describe raw_1m_partitioned")

    universe_path = source.parent / str(manifest.get("universe_path", ""))
    if not universe_path.is_file() or _sha256(universe_path) != manifest.get(
        "universe_sha256"
    ):
        raise ContractError("PandaData PIT universe fingerprint mismatch")
    universe = pd.read_parquet(universe_path)
    universe["trade_date"] = pd.to_datetime(universe["trade_date"]).dt.date
    selected = (
        universe.groupby("trade_date", observed=True)["instrument_id"]
        .nunique()
        .rename("selected_instruments")
    )

    stock_day_frames: list[pd.DataFrame] = []
    total_rows = 0
    partition_fingerprints: list[str] = []
    for partition in manifest.get("partitions", []):
        part_path = source.parent / str(partition.get("path", ""))
        fingerprint = _sha256(part_path) if part_path.is_file() else ""
        if fingerprint != partition.get("sha256"):
            raise ContractError("PandaData partition fingerprint mismatch")
        bars = pd.read_parquet(part_path, columns=["instrument_id", "bar_end_at"])
        if len(bars) != int(partition.get("row_count", -1)):
            raise ContractError("PandaData partition row count mismatch")
        if bars.duplicated(["instrument_id", "bar_end_at"]).any():
            raise ContractError("PandaData partition contains duplicate minute keys")
        bars["trade_date"] = pd.to_datetime(bars["bar_end_at"]).dt.date
        counts = (
            bars.groupby(["trade_date", "instrument_id"], observed=True)
            .size()
            .rename("minute_count")
            .reset_index()
        )
        stock_day_frames.append(counts)
        total_rows += len(bars)
        partition_fingerprints.append(fingerprint)

    stock_days = pd.concat(stock_day_frames, ignore_index=True)
    if stock_days.duplicated(["trade_date", "instrument_id"]).any():
        raise ContractError("PandaData partitions overlap on stock-day keys")
    expected = universe[["trade_date", "instrument_id"]].drop_duplicates()
    stock_days = expected.merge(
        stock_days,
        on=["trade_date", "instrument_id"],
        how="left",
        validate="one_to_one",
    )
    stock_days["minute_count"] = stock_days["minute_count"].fillna(0).astype(int)
    stock_days["complete"] = stock_days["minute_count"] == 240
    daily = stock_days.groupby("trade_date", observed=True).agg(
        complete_instruments=("complete", "sum"),
        incomplete_instruments=("complete", lambda values: int((~values).sum())),
        raw_row_count=("minute_count", "sum"),
    )
    daily = daily.join(selected).reset_index().sort_values("trade_date")
    daily["eligible"] = daily["complete_instruments"] >= min_complete_instruments
    daily["rejection_reasons"] = daily["eligible"].map(
        {True: "", False: "too_few_complete_instruments"}
    )

    candidate_rows: list[dict[str, object]] = []
    run: list[date] = []

    def finish_run() -> None:
        if not run:
            return
        candidate_count = max(
            len(run) - lookback_days - max_horizon_days + 1,
            0,
        )
        first_index = lookback_days - 1
        for index in range(first_index, first_index + candidate_count):
            candidate_rows.append(
                {
                    "signal_date": run[index],
                    "lookback_start_date": run[index - lookback_days + 1],
                    "label_end_date": run[index + max_horizon_days],
                }
            )
        run.clear()

    for row in daily.itertuples(index=False):
        if bool(row.eligible):
            run.append(cast(date, row.trade_date))
        else:
            finish_run()
    finish_run()
    candidate_signal_days = pd.DataFrame(
        candidate_rows,
        columns=["signal_date", "lookback_start_date", "label_end_date"],
    )
    blockers = (
        ("insufficient_candidate_signal_days",)
        if len(candidate_signal_days) < required_signal_days
        else ()
    )
    parameters: dict[str, object] = dict(positive)
    return PandaDataCoverageAudit(
        daily_coverage=daily.reset_index(drop=True),
        candidate_signal_days=candidate_signal_days,
        status="blocked" if blockers else "ready",
        blockers=blockers,
        parameters=parameters,
        source_identity={
            "manifest_name": source.name,
            "manifest_sha256": _sha256(source),
            "partition_count": len(partition_fingerprints),
            "partition_set_sha256": hashlib.sha256(
                "".join(partition_fingerprints).encode("ascii")
            ).hexdigest(),
            "total_row_count": total_rows,
        },
    )


def write_pandadata_coverage_receipt(
    audit: PandaDataCoverageAudit, *, output_dir: str | Path
) -> Path:
    """Write immutable, fingerprinted coverage tables and a compact receipt."""

    destination = Path(output_dir).expanduser().resolve()
    daily_path = destination / "daily-coverage.csv"
    candidates_path = destination / "candidate-signal-days.csv"
    receipt_path = destination / "coverage-receipt.json"
    if any(path.exists() for path in (daily_path, candidates_path, receipt_path)):
        raise ContractError("PandaData coverage receipt refuses to overwrite artifacts")
    destination.mkdir(parents=True, exist_ok=True)
    for path, frame in (
        (daily_path, audit.daily_coverage),
        (candidates_path, audit.candidate_signal_days),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(path)
    artifacts = {
        daily_path.name: _sha256(daily_path),
        candidates_path.name: _sha256(candidates_path),
    }
    summary = {
        "total_date_count": len(audit.daily_coverage),
        "eligible_date_count": int(audit.daily_coverage["eligible"].sum()),
        "complete_stock_day_count": int(
            audit.daily_coverage["complete_instruments"].sum()
        ),
        "incomplete_stock_day_count": int(
            audit.daily_coverage["incomplete_instruments"].sum()
        ),
        "candidate_signal_day_count": len(audit.candidate_signal_days),
        "total_row_count": cast(int, audit.source_identity["total_row_count"]),
        "minimum_daily_complete_instruments": int(
            audit.daily_coverage["complete_instruments"].min()
        ),
        "maximum_daily_complete_instruments": int(
            audit.daily_coverage["complete_instruments"].max()
        ),
    }
    identity = {
        "artifacts": artifacts,
        "blockers": list(audit.blockers),
        "parameters": audit.parameters,
        "source": audit.source_identity,
        "status": audit.status,
        "summary": summary,
    }
    audit_id = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    _write_json_atomic(
        receipt_path,
        {"schema_version": 1, "audit_id": audit_id, **identity},
    )
    return receipt_path


def materialize_pandadata_runtime_slice(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    top_n: int,
) -> Path:
    """Stream a daily PIT top-N subset into the existing single-file runtime contract."""

    if top_n < 1:
        raise ContractError("top_n must be positive")
    source = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read PandaData manifest: {exc}") from exc
    if manifest.get("dataset_kind") != "raw_1m_partitioned":
        raise ContractError("runtime materialization requires raw_1m_partitioned")
    universe_path = source.parent / str(manifest.get("universe_path", ""))
    if _sha256(universe_path) != manifest.get("universe_sha256"):
        raise ContractError("PandaData PIT universe fingerprint mismatch")
    universe = pd.read_parquet(universe_path)
    universe["trade_date"] = pd.to_datetime(universe["trade_date"]).dt.date
    universe = universe.sort_values(
        ["trade_date", "weight", "vendor_symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = universe.groupby("trade_date", sort=True, observed=True).head(top_n)
    if selected.groupby("trade_date", observed=True).size().min() < top_n:
        raise ContractError("PandaData PIT universe has fewer members than top_n")
    membership = selected[["trade_date", "instrument_id"]].drop_duplicates()

    destination = Path(output_dir).expanduser().resolve()
    bars_path = destination / "bars-1m.parquet"
    states_path = destination / "instrument-state.parquet"
    runtime_manifest_path = destination / "manifest.json"
    if any(path.exists() for path in (bars_path, states_path, runtime_manifest_path)):
        raise ContractError("PandaData runtime slice refuses to overwrite artifacts")
    destination.mkdir(parents=True, exist_ok=True)
    temporary_bars = bars_path.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    complete_stock_days = 0
    first_observed: dict[str, date] = {}
    try:
        for partition in manifest.get("partitions", []):
            part_path = source.parent / str(partition.get("path", ""))
            if _sha256(part_path) != partition.get("sha256"):
                raise ContractError("PandaData partition fingerprint mismatch")
            bars = pd.read_parquet(part_path)
            if bars.empty:
                continue
            bars["trade_date"] = bars["bar_end_at"].dt.date
            bars = bars.merge(
                membership,
                on=["trade_date", "instrument_id"],
                how="inner",
                validate="many_to_one",
            )
            if bars.empty:
                continue
            counts = bars.groupby(
                ["trade_date", "instrument_id"], observed=True
            ).size()
            complete_keys = counts.loc[counts == 240].reset_index()[
                ["trade_date", "instrument_id"]
            ]
            bars = bars.merge(
                complete_keys,
                on=["trade_date", "instrument_id"],
                how="inner",
                validate="many_to_one",
            ).drop(columns="trade_date")
            if bars.empty:
                continue
            bars = bars.sort_values(
                ["instrument_id", "bar_end_at"], kind="mergesort"
            ).reset_index(drop=True)
            for instrument_id, observed_at in zip(
                complete_keys["instrument_id"], complete_keys["trade_date"]
            ):
                previous = first_observed.get(str(instrument_id))
                if previous is None or observed_at < previous:
                    first_observed[str(instrument_id)] = observed_at
            table = pa.Table.from_pandas(bars, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_bars,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            total_rows += len(bars)
            complete_stock_days += len(complete_keys)
    finally:
        if writer is not None:
            writer.close()
    if writer is None or total_rows == 0:
        temporary_bars.unlink(missing_ok=True)
        raise ContractError("PandaData runtime slice contains no complete stock-days")
    temporary_bars.replace(bars_path)

    state_rows = []
    for instrument_id, observed_at in sorted(first_observed.items()):
        exchange = instrument_id.rsplit(".", 1)[1]
        state_rows.append(
            {
                "instrument_id": instrument_id,
                "effective_at": pd.Timestamp(
                    f"{observed_at} 00:00:00", tz="Asia/Shanghai"
                ),
                "exchange": exchange,
                "security_type": "A_SHARE",
                "listed_date": observed_at,
                "delisted_date": None,
                "is_st": False,
                "is_delisting": False,
                "is_suspended": False,
                "industry": "unavailable",
                "market_cap": float("nan"),
                "adv20": float("nan"),
                "market_state": "index_member_with_complete_session",
            }
        )
    states = pd.DataFrame(state_rows)
    temporary_states = states_path.with_suffix(".parquet.tmp")
    states.to_parquet(temporary_states, index=False)
    temporary_states.replace(states_path)

    runtime_manifest = {
        "schema_version": 1,
        "dataset_kind": "raw_1m",
        "data_path": bars_path.name,
        "data_sha256": _sha256(bars_path),
        "instrument_state_path": states_path.name,
        "instrument_state_sha256": _sha256(states_path),
        "timezone": "Asia/Shanghai",
        "price_unit": "CNY",
        "volume_unit": "share",
        "amount_unit": "CNY",
        "source_version": f"{manifest['source_version']}:runtime-top{top_n}",
        "source_partitioned_manifest_sha256": _sha256(source),
        "materialization": {
            "top_n": top_n,
            "membership": "daily PIT index weight rank",
            "complete_minute_count": 240,
            "complete_stock_day_count": complete_stock_days,
            "row_count": total_rows,
            "state_limitations": (
                "eligibility is derived from PIT index membership and complete sessions; "
                "historical ST and corporate-action states are not supplied"
            ),
        },
    }
    _write_json_atomic(runtime_manifest_path, runtime_manifest)
    return runtime_manifest_path


def materialize_pandadata_enriched_runtime(
    runtime_manifest_path: str | Path,
    *,
    states: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    output_dir: str | Path,
    enrichment_identity: dict[str, object],
) -> Path:
    """Attach PIT state and action ledgers while reusing immutable minute bars."""

    source = Path(runtime_manifest_path).expanduser().resolve()
    try:
        base = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read PandaData runtime manifest: {exc}") from exc
    if base.get("dataset_kind") != "raw_1m":
        raise ContractError("enrichment requires a raw_1m runtime manifest")
    bars_path = (source.parent / str(base.get("data_path", ""))).resolve()
    if not bars_path.is_file() or _sha256(bars_path) != base.get("data_sha256"):
        raise ContractError("PandaData runtime bars fingerprint mismatch")
    required_states = {"instrument_id", "effective_at"}
    if missing := sorted(required_states.difference(states.columns)):
        raise ContractError(f"enriched states missing columns: {', '.join(missing)}")
    required_actions = {"instrument_id", "effective_date", "pit_reliable"}
    if missing := sorted(required_actions.difference(corporate_actions.columns)):
        raise ContractError(
            f"corporate actions missing columns: {', '.join(missing)}"
        )

    destination = Path(output_dir).expanduser().resolve()
    states_path = destination / "instrument-state.parquet"
    actions_path = destination / "corporate-actions.parquet"
    manifest_path = destination / "manifest.json"
    if any(path.exists() for path in (states_path, actions_path, manifest_path)):
        raise ContractError("PandaData enriched runtime refuses to overwrite artifacts")
    destination.mkdir(parents=True, exist_ok=True)
    temporary_states = states_path.with_suffix(".parquet.tmp")
    temporary_actions = actions_path.with_suffix(".parquet.tmp")
    states.sort_values(["instrument_id", "effective_at"], kind="mergesort").to_parquet(
        temporary_states, index=False
    )
    corporate_actions.sort_values(
        ["instrument_id", "effective_date"], kind="mergesort"
    ).to_parquet(temporary_actions, index=False)
    temporary_states.replace(states_path)
    temporary_actions.replace(actions_path)

    relative_bars = os.path.relpath(bars_path, destination)
    enriched = {
        **base,
        "schema_version": 1,
        "data_path": relative_bars,
        "data_sha256": _sha256(bars_path),
        "instrument_state_path": states_path.name,
        "instrument_state_sha256": _sha256(states_path),
        "corporate_action_path": actions_path.name,
        "corporate_action_sha256": _sha256(actions_path),
        "source_version": f"{base['source_version']}:pit-enriched-v2",
        "source_runtime_manifest_sha256": _sha256(source),
        "enrichment": {
            "schema_version": 2,
            "adv20": "current-and-prior-19-completed-sessions",
            "market_cap": "close-times-latest-announced-total-shares",
            "industry_history": "unavailable",
            "listed_date": "first-observed-in-bounded-dataset",
            "corporate_action_policy": "invalidate-unadjusted-crossing-labels",
            "identity": enrichment_identity,
            "state_row_count": len(states),
            "corporate_action_row_count": len(corporate_actions),
        },
    }
    _write_json_atomic(manifest_path, enriched)
    return manifest_path
