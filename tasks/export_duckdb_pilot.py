"""Audit and export a bounded real-data slice from the Hermes DuckDB source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.duckdb_source import (  # noqa: E402
    audit_duckdb_trade_dates,
    export_duckdb_minute_slice,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and export a bounded A-share minute slice"
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--instrument", required=True, action="append")
    parser.add_argument("--min-instruments", required=True, type=int)
    parser.add_argument("--min-average-bars", required=True, type=float)
    parser.add_argument("--max-rows", required=True, type=int)
    arguments = parser.parse_args()

    try:
        audit = audit_duckdb_trade_dates(
            arguments.database,
            start_date=arguments.start_date,
            end_date=arguments.end_date,
            min_instruments=arguments.min_instruments,
            min_average_bars=arguments.min_average_bars,
        )
        accepted = audit.loc[audit["eligible"], "trade_date"].dt.strftime(
            "%Y-%m-%d"
        ).tolist()
        rejected = audit.loc[~audit["eligible"], "trade_date"].dt.strftime(
            "%Y-%m-%d"
        ).tolist()
        if not accepted:
            raise ContractError("no trade dates passed the DuckDB quality gate")
        manifest_path = export_duckdb_minute_slice(
            arguments.database,
            output_dir=arguments.output_dir,
            trade_dates=accepted,
            instrument_ids=arguments.instrument,
            max_rows=arguments.max_rows,
        )
        payload = {
            "status": "success",
            "manifest_path": str(manifest_path),
            "accepted_dates": accepted,
            "rejected_dates": rejected,
        }
    except (ContractError, OSError) as exc:
        payload = {
            "status": "error",
            "manifest_path": None,
            "accepted_dates": [],
            "rejected_dates": [],
            "error": str(exc),
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
