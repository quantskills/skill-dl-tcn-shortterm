"""Read-only adapter for the external Hermes A-share DuckDB dataset."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd

from .experiment import ContractError


_A_SHARE_FILTER = """
(
    (exchange = 'SHSE' AND regexp_full_match(symbol, '(600|601|603|605|688|689)[0-9]{3}'))
    OR
    (exchange = 'SZSE' AND regexp_full_match(symbol, '(000|001|002|003|300|301|302)[0-9]{3}'))
)
"""

_CANONICAL_INSTRUMENT = re.compile(r"^(?P<symbol>[0-9]{6})\.(?P<exchange>XSHG|XSHE)$")
_A_SHARE_PREFIXES = {
    "XSHG": ("600", "601", "603", "605", "688", "689"),
    "XSHE": ("000", "001", "002", "003", "300", "301", "302"),
}
_VENDOR_EXCHANGE = {"XSHG": "SHSE", "XSHE": "SZSE"}
_AUCTION_FILL_TIMES = {"14:58", "14:59"}


@dataclass(frozen=True)
class TrainingCoverageAudit:
    """Observable result of a read-only training-coverage audit."""

    daily_coverage: pd.DataFrame
    eligible_runs: pd.DataFrame
    candidate_signal_days: pd.DataFrame
    status: str
    blockers: tuple[str, ...]
    parameters: dict[str, Any]
    source_identity: dict[str, int | str]


_COVERAGE_RECEIPT_SCHEMA_VERSION = 1
_COVERAGE_ALGORITHM_VERSION = "duckdb-training-coverage-v1"


def _iso_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be an ISO date") from exc


def _database_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ContractError(f"DuckDB source does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instrument_pair(instrument_id: str) -> tuple[str, str]:
    match = _CANONICAL_INSTRUMENT.fullmatch(instrument_id)
    if match is None:
        raise ContractError(
            "instrument_ids must use canonical values such as 600000.XSHG"
        )
    symbol = match.group("symbol")
    exchange = match.group("exchange")
    if not symbol.startswith(_A_SHARE_PREFIXES[exchange]):
        raise ContractError(f"instrument is outside the A-share scope: {instrument_id}")
    return _VENDOR_EXCHANGE[exchange], symbol


def _expected_session_ends(trade_date: date) -> pd.DatetimeIndex:
    day = trade_date.isoformat()
    morning = pd.date_range(
        f"{day} 09:31", f"{day} 11:30", freq="1min", tz="Asia/Shanghai"
    )
    afternoon = pd.date_range(
        f"{day} 13:01", f"{day} 15:00", freq="1min", tz="Asia/Shanghai"
    )
    return pd.DatetimeIndex(morning.append(afternoon))


def _normalize_vendor_session(
    bars: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    queried_rows = len(bars)
    local = bars["bar_end_at"]
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    in_session = (
        ((minute_of_day > 9 * 60 + 30) & (minute_of_day <= 11 * 60 + 30))
        | ((minute_of_day > 13 * 60) & (minute_of_day <= 15 * 60))
    )
    normalized = bars.loc[in_session].copy()
    dropped_rows = queried_rows - len(normalized)
    if normalized.empty:
        raise ContractError("DuckDB selection has no rows inside A-share sessions")
    if normalized.duplicated(["instrument_id", "bar_end_at"]).any():
        raise ContractError("DuckDB selection contains duplicate minute bar keys")

    normalized["trade_date"] = normalized["bar_end_at"].dt.date
    fill_rows: list[dict[str, object]] = []
    unresolved_missing_rows = 0
    for (instrument_id, trade_date), group in normalized.groupby(
        ["instrument_id", "trade_date"], observed=True, sort=True
    ):
        expected = _expected_session_ends(trade_date)
        observed = pd.DatetimeIndex(group["bar_end_at"])
        missing = expected.difference(observed)
        close_at = expected[-1]
        has_close = close_at in observed
        for missing_at in missing:
            missing_text = missing_at.strftime("%H:%M")
            history = group.loc[group["bar_end_at"] < missing_at]
            if (
                missing_text not in _AUCTION_FILL_TIMES
                or not has_close
                or history.empty
            ):
                unresolved_missing_rows += 1
                continue
            prior_close = float(history.iloc[-1]["close"])
            fill_rows.append(
                {
                    "instrument_id": instrument_id,
                    "bar_end_at": missing_at,
                    "open": prior_close,
                    "high": prior_close,
                    "low": prior_close,
                    "close": prior_close,
                    "volume": 0.0,
                    "amount": 0.0,
                    "quality_flag": "auction_no_trade_fill",
                    "trade_date": trade_date,
                }
            )
    if fill_rows:
        normalized = pd.concat(
            [normalized, pd.DataFrame(fill_rows)], ignore_index=True, sort=False
        )
    normalized = normalized.sort_values(
        ["instrument_id", "bar_end_at"], kind="mergesort"
    ).reset_index(drop=True)
    stats = {
        "queried_row_count": queried_rows,
        "dropped_out_of_session_rows": dropped_rows,
        "auction_no_trade_fill_rows": len(fill_rows),
        "unresolved_missing_rows": unresolved_missing_rows,
        "exported_row_count": len(normalized),
    }
    return normalized.drop(columns="trade_date"), stats


def audit_duckdb_trade_dates(
    database_path: str | Path,
    *,
    start_date: str,
    end_date: str,
    min_instruments: int,
    min_average_bars: float,
) -> pd.DataFrame:
    """Return bounded, A-share-only date coverage from a read-only database."""

    path = _database_path(database_path)
    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date")
    if start > end:
        raise ContractError("start_date cannot be after end_date")
    if min_instruments < 1:
        raise ContractError("min_instruments must be positive")
    if min_average_bars <= 0:
        raise ContractError("min_average_bars must be positive")

    connection = duckdb.connect(str(path), read_only=True)
    try:
        audit = connection.execute(
            f"""
            SELECT
                trade_date,
                count(*)::BIGINT AS row_count,
                count(DISTINCT exchange || ':' || symbol)::BIGINT
                    AS instrument_count
            FROM stocks_1m
            WHERE trade_date BETWEEN ? AND ?
              AND {_A_SHARE_FILTER}
            GROUP BY trade_date
            ORDER BY trade_date
            """,
            [start, end],
        ).fetchdf()
    except duckdb.Error as exc:
        raise ContractError(f"cannot audit DuckDB minute source: {exc}") from exc
    finally:
        connection.close()

    if audit.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "row_count",
                "instrument_count",
                "average_bars",
                "eligible",
                "rejection_reasons",
            ]
        )

    audit["trade_date"] = pd.to_datetime(audit["trade_date"])
    audit["row_count"] = audit["row_count"].astype("int64")
    audit["instrument_count"] = audit["instrument_count"].astype("int64")
    audit["average_bars"] = audit["row_count"] / audit["instrument_count"]

    def rejection_reasons(row: pd.Series) -> str:
        reasons = []
        if int(row["instrument_count"]) < min_instruments:
            reasons.append("too_few_instruments")
        if float(row["average_bars"]) < min_average_bars:
            reasons.append("too_few_average_bars")
        return "|".join(reasons)

    audit["rejection_reasons"] = audit.apply(rejection_reasons, axis=1)
    audit["eligible"] = audit["rejection_reasons"] == ""
    return audit[
        [
            "trade_date",
            "row_count",
            "instrument_count",
            "average_bars",
            "eligible",
            "rejection_reasons",
        ]
    ]


def audit_duckdb_training_coverage(
    database_path: str | Path,
    *,
    start_date: str,
    end_date: str,
    min_instruments: int,
    min_average_bars: float,
    min_primary_source_ratio: float,
    lookback_days: int,
    max_horizon_days: int,
    train_signal_days: int,
    validation_signal_days: int,
    ordinary_test_signal_days: int,
) -> TrainingCoverageAudit:
    """Audit whether read-only minute coverage can form contiguous model samples."""

    path = _database_path(database_path)
    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date")
    if start > end:
        raise ContractError("start_date cannot be after end_date")
    if min_instruments < 1:
        raise ContractError("min_instruments must be positive")
    if min_average_bars <= 0:
        raise ContractError("min_average_bars must be positive")
    if not 0.0 <= min_primary_source_ratio <= 1.0:
        raise ContractError("min_primary_source_ratio must be between 0 and 1")
    day_budgets = {
        "lookback_days": lookback_days,
        "max_horizon_days": max_horizon_days,
        "train_signal_days": train_signal_days,
        "validation_signal_days": validation_signal_days,
        "ordinary_test_signal_days": ordinary_test_signal_days,
    }
    if any(value < 1 for value in day_budgets.values()):
        raise ContractError("all lookback, horizon and signal-day budgets must be positive")

    connection = duckdb.connect(str(path), read_only=True)
    try:
        daily = connection.execute(
            f"""
            SELECT
                trade_date,
                count(*)::BIGINT AS row_count,
                count(DISTINCT exchange || ':' || symbol)::BIGINT
                    AS instrument_count,
                count(*) FILTER (WHERE _source_file IS NULL)::BIGINT
                    AS primary_source_rows,
                count(*) FILTER (WHERE _source_file IS NOT NULL)::BIGINT
                    AS supplemental_source_rows
            FROM stocks_1m
            WHERE trade_date BETWEEN ? AND ?
              AND {_A_SHARE_FILTER}
            GROUP BY trade_date
            ORDER BY trade_date
            """,
            [start, end],
        ).fetchdf()
    except duckdb.Error as exc:
        raise ContractError(f"cannot audit DuckDB training coverage: {exc}") from exc
    finally:
        connection.close()

    daily_columns = [
        "trade_date",
        "row_count",
        "instrument_count",
        "average_bars",
        "primary_source_rows",
        "supplemental_source_rows",
        "primary_source_ratio",
        "eligible",
        "rejection_reasons",
    ]
    if daily.empty:
        daily = pd.DataFrame(columns=daily_columns)
    else:
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        count_columns = [
            "row_count",
            "instrument_count",
            "primary_source_rows",
            "supplemental_source_rows",
        ]
        daily[count_columns] = daily[count_columns].astype("int64")
        daily["average_bars"] = daily["row_count"] / daily["instrument_count"]
        daily["primary_source_ratio"] = (
            daily["primary_source_rows"] / daily["row_count"]
        )

        def coverage_rejections(row: pd.Series) -> str:
            reasons = []
            if int(row["instrument_count"]) < min_instruments:
                reasons.append("too_few_instruments")
            if float(row["average_bars"]) < min_average_bars:
                reasons.append("too_few_average_bars")
            if float(row["primary_source_ratio"]) < min_primary_source_ratio:
                reasons.append("too_little_primary_source")
            return "|".join(reasons)

        daily["rejection_reasons"] = daily.apply(coverage_rejections, axis=1)
        daily["eligible"] = daily["rejection_reasons"] == ""
        daily = daily[daily_columns]

    run_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    current_dates: list[pd.Timestamp] = []

    def finish_run() -> None:
        if not current_dates:
            return
        run_id = len(run_rows) + 1
        candidate_count = max(
            len(current_dates) - lookback_days - max_horizon_days + 1,
            0,
        )
        run_rows.append(
            {
                "run_id": run_id,
                "start_date": current_dates[0],
                "end_date": current_dates[-1],
                "day_count": len(current_dates),
                "candidate_signal_day_count": candidate_count,
            }
        )
        first_signal_index = lookback_days - 1
        for signal_index in range(
            first_signal_index,
            first_signal_index + candidate_count,
        ):
            signal_rows.append(
                {
                    "signal_date": current_dates[signal_index],
                    "run_id": run_id,
                    "lookback_start_date": current_dates[
                        signal_index - lookback_days + 1
                    ],
                    "label_end_date": current_dates[
                        signal_index + max_horizon_days
                    ],
                }
            )
        current_dates.clear()

    for row in daily.itertuples(index=False):
        if bool(row.eligible):
            current_dates.append(pd.Timestamp(str(row.trade_date)))
        else:
            finish_run()
    finish_run()

    eligible_runs = pd.DataFrame(
        run_rows,
        columns=[
            "run_id",
            "start_date",
            "end_date",
            "day_count",
            "candidate_signal_day_count",
        ],
    )
    candidate_signal_days = pd.DataFrame(
        signal_rows,
        columns=[
            "signal_date",
            "run_id",
            "lookback_start_date",
            "label_end_date",
        ],
    )
    required_signal_days = (
        train_signal_days + validation_signal_days + ordinary_test_signal_days
    )
    blockers = (
        ("insufficient_candidate_signal_days",)
        if len(candidate_signal_days) < required_signal_days
        else ()
    )
    source_stat = path.stat()
    parameters: dict[str, Any] = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "min_instruments": min_instruments,
        "min_average_bars": min_average_bars,
        "min_primary_source_ratio": min_primary_source_ratio,
        **day_budgets,
        "required_signal_days": required_signal_days,
    }
    return TrainingCoverageAudit(
        daily_coverage=daily,
        eligible_runs=eligible_runs,
        candidate_signal_days=candidate_signal_days,
        status="blocked" if blockers else "ready",
        blockers=blockers,
        parameters=parameters,
        source_identity={
            "database_name": path.name,
            "database_size": source_stat.st_size,
            "database_mtime_ns": source_stat.st_mtime_ns,
        },
    )


def write_training_coverage_receipt(
    audit: TrainingCoverageAudit,
    *,
    output_dir: str | Path,
) -> Path:
    """Write immutable, fingerprinted coverage evidence without copying source data."""

    destination = Path(output_dir).expanduser().resolve()
    artifact_frames = {
        "daily-coverage.csv": audit.daily_coverage,
        "eligible-runs.csv": audit.eligible_runs,
        "candidate-signal-days.csv": audit.candidate_signal_days,
    }
    receipt_path = destination / "coverage-receipt.json"
    targets = [destination / name for name in artifact_frames] + [receipt_path]
    if any(path.exists() for path in targets):
        raise ContractError("training coverage receipt refuses to overwrite artifacts")
    destination.mkdir(parents=True, exist_ok=True)

    artifact_hashes: dict[str, str] = {}
    for name, frame in artifact_frames.items():
        artifact_path = destination / name
        frame.to_csv(artifact_path, index=False, lineterminator="\n")
        artifact_hashes[name] = _sha256(artifact_path)

    longest_run = (
        int(audit.eligible_runs["day_count"].max())
        if not audit.eligible_runs.empty
        else 0
    )
    peak_instruments = (
        int(audit.daily_coverage["instrument_count"].max())
        if not audit.daily_coverage.empty
        else 0
    )
    summary = {
        "candidate_signal_day_count": len(audit.candidate_signal_days),
        "eligible_date_count": int(audit.daily_coverage["eligible"].sum()),
        "eligible_run_count": len(audit.eligible_runs),
        "longest_eligible_run_days": longest_run,
        "required_signal_day_count": int(
            audit.parameters["required_signal_days"]
        ),
        "total_date_count": len(audit.daily_coverage),
        "total_row_count": int(audit.daily_coverage["row_count"].sum()),
        "unique_instrument_count_peak": peak_instruments,
    }
    identity_payload = {
        "algorithm_version": _COVERAGE_ALGORITHM_VERSION,
        "artifacts": artifact_hashes,
        "blockers": list(audit.blockers),
        "parameters": audit.parameters,
        "source": audit.source_identity,
        "status": audit.status,
        "summary": summary,
    }
    audit_id = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    receipt = {
        "schema_version": _COVERAGE_RECEIPT_SCHEMA_VERSION,
        "audit_id": audit_id,
        **identity_payload,
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path


def export_duckdb_minute_slice(
    database_path: str | Path,
    *,
    output_dir: str | Path,
    trade_dates: Sequence[str],
    instrument_ids: Sequence[str],
    max_rows: int,
) -> Path:
    """Export an explicit, bounded DuckDB selection to the canonical raw_1m contract."""

    path = _database_path(database_path)
    if not trade_dates:
        raise ContractError("trade_dates must contain at least one explicit date")
    if len(trade_dates) > 31:
        raise ContractError("trade_dates cannot contain more than 31 dates per export")
    if not instrument_ids:
        raise ContractError("instrument_ids must contain at least one explicit instrument")
    if len(instrument_ids) > 512:
        raise ContractError("instrument_ids cannot contain more than 512 instruments")
    if max_rows < 1:
        raise ContractError("max_rows must be positive")

    dates = [_iso_date(value, "trade_dates") for value in trade_dates]
    if len(set(dates)) != len(dates):
        raise ContractError("trade_dates cannot contain duplicates")
    if len(set(instrument_ids)) != len(instrument_ids):
        raise ContractError("instrument_ids cannot contain duplicates")
    pairs = [_instrument_pair(value) for value in instrument_ids]

    date_placeholders = ", ".join("?" for _ in dates)
    instrument_predicate = " OR ".join(
        "(exchange = ? AND symbol = ?)" for _ in pairs
    )
    parameters: list[object] = list(dates)
    for exchange, symbol in pairs:
        parameters.extend([exchange, symbol])
    parameters.append(max_rows + 1)

    connection = duckdb.connect(str(path), read_only=True)
    try:
        bars = connection.execute(
            f"""
            SELECT
                exchange,
                symbol,
                eob AS bar_end_at,
                open,
                high,
                low,
                close,
                volume,
                amount
            FROM stocks_1m
            WHERE trade_date IN ({date_placeholders})
              AND ({instrument_predicate})
              AND {_A_SHARE_FILTER}
            ORDER BY symbol, exchange, eob
            LIMIT ?
            """,
            parameters,
        ).fetchdf()
    except duckdb.Error as exc:
        raise ContractError(f"cannot export DuckDB minute source: {exc}") from exc
    finally:
        connection.close()

    if len(bars) > max_rows:
        raise ContractError(f"DuckDB selection exceeds max_rows={max_rows}")
    if bars.empty:
        raise ContractError("DuckDB selection returned no A-share minute bars")

    suffix = {"SHSE": "XSHG", "SZSE": "XSHE"}
    bars["instrument_id"] = bars.apply(
        lambda row: f"{row['symbol']}.{suffix[str(row['exchange'])]}", axis=1
    )
    bars["bar_end_at"] = pd.to_datetime(
        bars["bar_end_at"], errors="raise", utc=True
    ).dt.tz_convert("Asia/Shanghai")
    bars["quality_flag"] = "ok"
    bars = bars[
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
    ]
    bars, normalization = _normalize_vendor_session(bars)

    destination = Path(output_dir).expanduser().resolve()
    data_path = destination / "bars-1m.parquet"
    manifest_path = destination / "manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise ContractError("DuckDB export refuses to overwrite existing artifacts")
    destination.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(data_path, index=False)

    source_stat = path.stat()
    manifest = {
        "schema_version": 1,
        "dataset_kind": "raw_1m",
        "data_path": data_path.name,
        "data_sha256": _sha256(data_path),
        "timezone": "Asia/Shanghai",
        "price_unit": "CNY",
        "volume_unit": "share",
        "amount_unit": "CNY",
        "source_version": (
            f"hermes-a-stock-duckdb-v1:{source_stat.st_size}:"
            f"{source_stat.st_mtime_ns}"
        ),
        "source_table": "stocks_1m",
        "source_database_name": path.name,
        "source_database_size": source_stat.st_size,
        "source_database_mtime_ns": source_stat.st_mtime_ns,
        "source_selection": {
            "trade_dates": [value.isoformat() for value in dates],
            "instrument_ids": list(instrument_ids),
            "max_rows": max_rows,
        },
        "normalization": normalization,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
