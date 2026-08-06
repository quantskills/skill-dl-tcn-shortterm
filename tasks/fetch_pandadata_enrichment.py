"""Fetch bounded PandaData PIT enrichment without persisting credentials."""

from __future__ import annotations

import argparse
from calendar import monthrange
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Callable, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.pandadata_source import (  # noqa: E402
    canonicalize_pandadata_adjustments,
    canonicalize_pandadata_daily,
    canonicalize_pandadata_share_float,
    merge_pit_enrichment_frames,
)


_PERMANENT_ERROR_MARKERS = (
    "200001",
    "200002",
    "200003",
    "200004",
    "200005",
    "200006",
    "200007",
    "200101",
    "200102",
    "200103",
    "200104",
    "600002",
    "600003",
    "请求参数",
)


class RetryingPandaDataAPI:
    """Retry the read-only enrichment endpoints on transient failures only."""

    def __init__(self, module: ModuleType, *, attempts: int = 4) -> None:
        self._module = module
        self._attempts = attempts

    def _call(self, name: str, **kwargs: object) -> pd.DataFrame:
        operation: Callable[..., pd.DataFrame] = getattr(self._module, name)
        for attempt in range(1, self._attempts + 1):
            try:
                return operation(**kwargs)
            except Exception as exc:
                permanent = any(
                    marker in str(exc) for marker in _PERMANENT_ERROR_MARKERS
                )
                if permanent or attempt == self._attempts:
                    raise
                time.sleep(2 ** (attempt - 1))
        raise RuntimeError("unreachable retry state")

    def get_stock_daily(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_stock_daily", **kwargs)

    def get_stock_status_change(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_stock_status_change", **kwargs)

    def get_adj_factor(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_adj_factor", **kwargs)

    def get_share_float(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_share_float", **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_top_n(source: Path, top_n: int) -> tuple[pd.DataFrame, list[str], str]:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    universe_path = source.parent / str(manifest.get("universe_path", ""))
    expected = manifest.get("universe_sha256")
    if not universe_path.is_file() or _sha256(universe_path) != expected:
        raise ContractError("PandaData PIT universe fingerprint mismatch")
    universe = pd.read_parquet(universe_path)
    universe["trade_date"] = pd.to_datetime(universe["trade_date"]).dt.date
    selected = (
        universe.sort_values(
            ["trade_date", "weight", "vendor_symbol"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("trade_date", sort=True, observed=True)
        .head(top_n)
        .reset_index(drop=True)
    )
    if selected.empty or selected.groupby("trade_date", observed=True).size().min() < top_n:
        raise ContractError("PandaData PIT universe has fewer members than top_n")
    return selected, sorted(selected["vendor_symbol"].unique()), _sha256(source)


def _plans(symbols: list[str], start_year: int, end_year: int) -> list[dict[str, object]]:
    plans: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start = f"{year:04d}{month:02d}01"
            end = f"{year:04d}{month:02d}{monthrange(year, month)[1]:02d}"
            for offset in range(0, len(symbols), 25):
                batch = symbols[offset : offset + 25]
                plans.append(
                    {
                        "chunk_id": f"{year:04d}-{month:02d}-{offset // 25:03d}",
                        "start_date": start,
                        "end_date": end,
                        "symbols": batch,
                    }
                )
    return plans


def _fetch_chunks(
    api: RetryingPandaDataAPI,
    *,
    endpoint: str,
    plans: list[dict[str, object]],
    output_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    operation: Callable[..., pd.DataFrame] = getattr(api, endpoint)
    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, object]] = []
    endpoint_dir = output_dir / "chunks" / endpoint
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    for position, plan in enumerate(plans, start=1):
        symbols = cast(list[str], plan["symbols"])
        path = endpoint_dir / f"{plan['chunk_id']}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            frame = operation(
                symbol=symbols,
                start_date=plan["start_date"],
                end_date=plan["end_date"],
                fields=[],
            )
            if frame.empty and len(frame.columns) == 0:
                frame = pd.DataFrame({"_empty_response": pd.Series(dtype="bool")})
            _atomic_parquet(path, frame)
        frames.append(frame.drop(columns=["_empty_response"], errors="ignore"))
        receipts.append(
            {
                "chunk_id": plan["chunk_id"],
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
                "row_count": len(frame),
                "start_date": plan["start_date"],
                "end_date": plan["end_date"],
                "symbol_count": len(symbols),
            }
        )
        if position % 20 == 0 or position == len(plans):
            print(
                f"{endpoint}: {position}/{len(plans)} chunks",
                file=sys.stderr,
                flush=True,
            )
    nonempty = [frame for frame in frames if len(frame.columns) > 0]
    return (pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()), receipts


def _load_enrichment_artifacts(
    manifest_path: Path,
    *,
    source_sha256: str,
    start_year: int,
    end_year: int,
    maximum_top_n: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Load a verified narrower enrichment for incremental universe expansion."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_kind") != "pandadata_pit_enrichment":
        raise ContractError("reused artifact is not PandaData PIT enrichment")
    if manifest.get("source_manifest_sha256") != source_sha256:
        raise ContractError("reused enrichment source manifest drifted")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ContractError("reused enrichment parameters are missing")
    reused_top_n = int(parameters.get("top_n", 0))
    if reused_top_n <= 0 or reused_top_n >= maximum_top_n:
        raise ContractError("reused enrichment must have a narrower positive top_n")
    if int(parameters.get("start_year", 0)) != start_year or int(
        parameters.get("end_year", 0)
    ) != end_year:
        raise ContractError("reused enrichment year bounds drifted")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractError("reused enrichment artifacts are missing")
    expected_names = {
        "daily",
        "share_float",
        "corporate_actions",
        "status_changes",
        "membership",
    }
    if set(artifacts) != expected_names:
        raise ContractError("reused enrichment artifact set drifted")
    frames: dict[str, pd.DataFrame] = {}
    for name in sorted(expected_names):
        artifact = artifacts[name]
        if not isinstance(artifact, dict):
            raise ContractError(f"reused enrichment artifact is invalid: {name}")
        path = manifest_path.parent / str(artifact.get("path", ""))
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise ContractError(f"reused enrichment fingerprint mismatch: {name}")
        frames[name] = pd.read_parquet(path).drop(
            columns=["_empty_response"], errors="ignore"
        )
    return manifest, frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch PandaData PIT enrichment")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", default=20, type=int)
    parser.add_argument("--start-year", default=2021, type=int)
    parser.add_argument("--end-year", default=2025, type=int)
    parser.add_argument("--reuse-enrichment-manifest", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        try:
            _, symbols, source_sha = _load_top_n(
                arguments.source_manifest.resolve(), arguments.top_n
            )
            planned_reused_symbols: list[str] = []
            if arguments.reuse_enrichment_manifest is not None:
                planned_reuse_manifest, _ = _load_enrichment_artifacts(
                    arguments.reuse_enrichment_manifest.resolve(),
                    source_sha256=source_sha,
                    start_year=arguments.start_year,
                    end_year=arguments.end_year,
                    maximum_top_n=arguments.top_n,
                )
                planned_reused_top_n = int(
                    planned_reuse_manifest["parameters"]["top_n"]
                )
                _, planned_reused_symbols, _ = _load_top_n(
                    arguments.source_manifest.resolve(), planned_reused_top_n
                )
            fetch_symbols = sorted(set(symbols).difference(planned_reused_symbols))
            plans = _plans(fetch_symbols, arguments.start_year, arguments.end_year)
            print(
                json.dumps(
                    {
                        "status": "planned",
                        "top_n": arguments.top_n,
                        "symbol_count": len(symbols),
                        "reused_symbol_count": len(planned_reused_symbols),
                        "fetched_symbol_count": len(fetch_symbols),
                        "chunk_count_per_endpoint": len(plans),
                        "request_count": len(plans) * 4,
                        "credentials_accessed": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        except (OSError, ValueError, KeyError, TypeError, ContractError) as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
            return 2
    username = os.environ.get("PANDADATA_USERNAME", "")
    password = os.environ.get("PANDADATA_PASSWORD", "")
    if not username or not password:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "PANDADATA_USERNAME and PANDADATA_PASSWORD environment variables are required",
                },
                ensure_ascii=False,
            )
        )
        return 2
    try:
        output_dir = arguments.output_dir.resolve()
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            for artifact in existing["artifacts"].values():
                path = output_dir / artifact["path"]
                if _sha256(path) != artifact["sha256"]:
                    raise ContractError("existing enrichment fingerprint mismatch")
            payload: dict[str, Any] = {
                "status": "success",
                "manifest_path": str(manifest_path),
                "resumed": True,
            }
        else:
            for name in list(os.environ):
                if name.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                    os.environ.pop(name, None)
            import panda_data
            import panda_data.auth_manager as auth_manager

            credential_path = Path(auth_manager._get_user_json_path()).resolve()
            if credential_path.exists():
                raise ContractError("refusing run because SDK credential state exists")
            auth_manager._persist_credentials = lambda *args, **kwargs: None
            panda_data.init_token(username=username, password=password)
            api = RetryingPandaDataAPI(panda_data)
            selected, symbols, source_sha = _load_top_n(
                arguments.source_manifest.resolve(), arguments.top_n
            )
            reused_manifest: dict[str, Any] | None = None
            reused_frames: dict[str, pd.DataFrame] = {}
            reused_symbols: list[str] = []
            if arguments.reuse_enrichment_manifest is not None:
                reused_path = arguments.reuse_enrichment_manifest.resolve()
                reused_manifest, reused_frames = _load_enrichment_artifacts(
                    reused_path,
                    source_sha256=source_sha,
                    start_year=arguments.start_year,
                    end_year=arguments.end_year,
                    maximum_top_n=arguments.top_n,
                )
                reused_top_n = int(reused_manifest["parameters"]["top_n"])
                _, reused_symbols, reused_source_sha = _load_top_n(
                    arguments.source_manifest.resolve(), reused_top_n
                )
                if reused_source_sha != source_sha:
                    raise ContractError("reused universe source identity drifted")
            fetch_symbols = sorted(set(symbols).difference(reused_symbols))
            if not fetch_symbols:
                raise ContractError("incremental enrichment has no new symbols to fetch")
            plans = _plans(fetch_symbols, arguments.start_year, arguments.end_year)
            output_dir.mkdir(parents=True, exist_ok=True)
            endpoint_receipts = {}
            raw_frames = {}
            for endpoint in [
                "get_stock_daily",
                "get_stock_status_change",
                "get_adj_factor",
                "get_share_float",
            ]:
                raw_frames[endpoint], endpoint_receipts[endpoint] = _fetch_chunks(
                    api, endpoint=endpoint, plans=plans, output_dir=output_dir
                )
            fetched_daily = canonicalize_pandadata_daily(raw_frames["get_stock_daily"])
            fetched_shares = canonicalize_pandadata_share_float(
                raw_frames["get_share_float"]
            )
            fetched_actions = canonicalize_pandadata_adjustments(
                raw_frames["get_adj_factor"]
            )
            fetched_status = raw_frames["get_stock_status_change"].drop_duplicates()
            daily = merge_pit_enrichment_frames(
                reused_frames.get("daily", pd.DataFrame()),
                fetched_daily,
                keys=["instrument_id", "trade_date"],
                name="daily",
            )
            shares = merge_pit_enrichment_frames(
                reused_frames.get("share_float", pd.DataFrame()),
                fetched_shares,
                keys=["instrument_id", "known_date"],
                name="share_float",
            )
            actions = merge_pit_enrichment_frames(
                reused_frames.get("corporate_actions", pd.DataFrame()),
                fetched_actions,
                keys=["instrument_id", "effective_date"],
                name="corporate_actions",
            )
            reused_status = reused_frames.get("status_changes", pd.DataFrame())
            status = pd.concat([reused_status, fetched_status], ignore_index=True).drop_duplicates()
            membership = selected[["trade_date", "instrument_id"]].drop_duplicates()
            frames = {
                "daily": daily,
                "share_float": shares,
                "corporate_actions": actions,
                "status_changes": status,
                "membership": membership,
            }
            artifacts = {}
            for name, frame in frames.items():
                path = output_dir / f"{name.replace('_', '-')}.parquet"
                if frame.empty and len(frame.columns) == 0:
                    frame = pd.DataFrame({"_empty_response": pd.Series(dtype="bool")})
                _atomic_parquet(path, frame)
                artifacts[name] = {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "row_count": len(frame),
                }
            manifest = {
                "schema_version": 1,
                "dataset_kind": "pandadata_pit_enrichment",
                "source_manifest_sha256": source_sha,
                "source_version": f"panda_data-{importlib.metadata.version('panda_data')}",
                "parameters": {
                    "top_n": arguments.top_n,
                    "start_year": arguments.start_year,
                    "end_year": arguments.end_year,
                    "symbol_count": len(symbols),
                    "reused_symbol_count": len(reused_symbols),
                    "fetched_symbol_count": len(fetch_symbols),
                    "chunking": "calendar-month-by-up-to-25-symbols",
                },
                "lineage": (
                    {
                        "mode": "incremental-universe-expansion",
                        "reused_enrichment_manifest": str(
                            arguments.reuse_enrichment_manifest.resolve()
                        ),
                        "reused_enrichment_sha256": _sha256(
                            arguments.reuse_enrichment_manifest.resolve()
                        ),
                    }
                    if arguments.reuse_enrichment_manifest is not None
                    else {"mode": "full-fetch"}
                ),
                "semantics": {
                    "adv20": "current-and-prior-19-completed-sessions",
                    "share_float": "known-from-provider-information-date",
                    "industry_history": "unavailable",
                    "adjustment": "unadjusted-minute-label-invalidation-v1",
                },
                "artifacts": artifacts,
                "chunks": endpoint_receipts,
            }
            _atomic_json(manifest_path, manifest)
            if credential_path.exists():
                raise ContractError("PandaData SDK created forbidden credential state")
            payload = {
                "status": "success",
                "manifest_path": str(manifest_path),
                "resumed": False,
                "symbol_count": len(symbols),
                "reused_symbol_count": len(reused_symbols),
                "fetched_symbol_count": len(fetch_symbols),
                "request_count": len(plans) * 4,
            }
    except Exception as exc:
        message = str(exc).replace(username, "<redacted>").replace(password, "<redacted>")
        payload = {"status": "error", "error": message}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
