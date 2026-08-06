"""Audit whether the external minute database can form real training samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.duckdb_source import (  # noqa: E402
    audit_duckdb_training_coverage,
    write_training_coverage_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit contiguous A-share minute coverage for a TCN pilot"
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--min-instruments", required=True, type=int)
    parser.add_argument("--min-average-bars", required=True, type=float)
    parser.add_argument("--min-primary-source-ratio", required=True, type=float)
    parser.add_argument("--lookback-days", required=True, type=int)
    parser.add_argument("--max-horizon-days", required=True, type=int)
    parser.add_argument("--train-signal-days", required=True, type=int)
    parser.add_argument("--validation-signal-days", required=True, type=int)
    parser.add_argument("--ordinary-test-signal-days", required=True, type=int)
    arguments = parser.parse_args()

    try:
        audit = audit_duckdb_training_coverage(
            arguments.database,
            start_date=arguments.start_date,
            end_date=arguments.end_date,
            min_instruments=arguments.min_instruments,
            min_average_bars=arguments.min_average_bars,
            min_primary_source_ratio=arguments.min_primary_source_ratio,
            lookback_days=arguments.lookback_days,
            max_horizon_days=arguments.max_horizon_days,
            train_signal_days=arguments.train_signal_days,
            validation_signal_days=arguments.validation_signal_days,
            ordinary_test_signal_days=arguments.ordinary_test_signal_days,
        )
        receipt_path = write_training_coverage_receipt(
            audit,
            output_dir=arguments.output_dir,
        )
        payload = {
            "status": audit.status,
            "blockers": list(audit.blockers),
            "candidate_signal_day_count": len(audit.candidate_signal_days),
            "receipt_path": str(receipt_path),
        }
        return_code = 0
    except (ContractError, OSError) as exc:
        payload = {
            "status": "error",
            "blockers": [],
            "candidate_signal_day_count": 0,
            "receipt_path": None,
            "error": str(exc),
        }
        return_code = 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
