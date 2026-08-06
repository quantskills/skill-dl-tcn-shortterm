"""Fetch a bounded PIT PandaData minute pilot without persisting credentials."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.pandadata_source import (  # noqa: E402
    build_monthly_fetch_plan,
    download_pandadata_pilot,
    fetch_pandadata_pit_universe,
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
    """Retry only the two read-only provider calls used by this workflow."""

    def __init__(self, module: ModuleType, *, attempts: int = 4) -> None:
        self._module = module
        self._attempts = attempts

    def _call(self, name: str, **kwargs: object) -> pd.DataFrame:
        operation: Callable[..., pd.DataFrame] = getattr(self._module, name)
        for attempt in range(1, self._attempts + 1):
            try:
                return operation(**kwargs)
            except Exception as exc:
                message = str(exc)
                permanent = any(marker in message for marker in _PERMANENT_ERROR_MARKERS)
                if permanent or attempt == self._attempts:
                    raise
                delay = 2 ** (attempt - 1)
                print(
                    f"PandaData transient {name} failure; retrying in {delay}s "
                    f"({attempt}/{self._attempts})",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable retry state")

    def get_index_weights(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_index_weights", **kwargs)

    def get_stock_min(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_stock_min", **kwargs)

    def get_stock_daily(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_stock_daily", **kwargs)

    def get_stock_status_change(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_stock_status_change", **kwargs)

    def get_adj_factor(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_adj_factor", **kwargs)

    def get_share_float(self, **kwargs: object) -> pd.DataFrame:
        return self._call("get_share_float", **kwargs)


def _write_weights(path: Path, weights: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.equals(weights):
            raise ContractError("existing index weights conflict with live response")
        return
    temporary = path.with_suffix(".parquet.tmp")
    weights.to_parquet(temporary, index=False)
    temporary.replace(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a resumable PandaData CSI300 PIT minute pilot"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--index-symbol", default="000300.SH")
    parser.add_argument("--top-n", default=100, type=int)
    parser.add_argument("--symbol-batch-size", default=25, type=int)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    username = os.environ.get("PANDADATA_USERNAME", "")
    password = os.environ.get("PANDADATA_PASSWORD", "")
    if not username or not password:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": (
                        "PANDADATA_USERNAME and PANDADATA_PASSWORD environment "
                        "variables are required"
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2

    try:
        for name in list(os.environ):
            if name.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                os.environ.pop(name, None)

        import panda_data
        import panda_data.auth_manager as auth_manager

        credential_path = Path(auth_manager._get_user_json_path()).resolve()
        if credential_path.exists():
            raise ContractError(
                "refusing PandaData run because SDK credential state already exists"
            )
        auth_manager._persist_credentials = lambda *args, **kwargs: None
        panda_data.init_token(username=username, password=password)
        api = RetryingPandaDataAPI(panda_data)

        print("Fetching bounded monthly CSI300 PIT weights", file=sys.stderr, flush=True)
        weights, universe = fetch_pandadata_pit_universe(
            api,
            index_symbol=arguments.index_symbol,
            start_date=arguments.start_date,
            end_date=arguments.end_date,
            top_n=arguments.top_n,
        )
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = arguments.output_dir.resolve() / "index-weights.parquet"
        _write_weights(weights_path, weights)
        plan = build_monthly_fetch_plan(
            universe,
            start_date=arguments.start_date,
            end_date=arguments.end_date,
            symbol_batch_size=arguments.symbol_batch_size,
        )
        print(
            f"Downloading {len(plan)} immutable month-symbol chunks",
            file=sys.stderr,
            flush=True,
        )
        source_version = (
            f"pandadata-{importlib.metadata.version('panda_data')}:"
            f"{arguments.index_symbol}:pit-top{arguments.top_n}"
        )
        manifest_path = download_pandadata_pilot(
            api,
            universe=universe,
            plan=plan,
            output_dir=arguments.output_dir,
            source_version=source_version,
            index_weights_path=weights_path,
        )
        if credential_path.exists():
            raise ContractError("PandaData SDK created forbidden credential state")
        payload: dict[str, Any] = {
            "status": "success",
            "manifest_path": str(manifest_path),
            "index_symbol": arguments.index_symbol,
            "top_n": arguments.top_n,
            "start_date": arguments.start_date,
            "end_date": arguments.end_date,
            "chunk_count": len(plan),
        }
    except Exception as exc:
        message = str(exc).replace(username, "<redacted>").replace(
            password, "<redacted>"
        )
        payload = {"status": "error", "error": message}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
