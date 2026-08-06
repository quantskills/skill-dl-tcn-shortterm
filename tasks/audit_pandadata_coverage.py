"""Audit a partitioned PandaData raw_1m manifest without loading all bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.pandadata_source import (  # noqa: E402
    audit_pandadata_coverage,
    write_pandadata_coverage_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PandaData TCN coverage")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-complete-instruments", required=True, type=int)
    parser.add_argument("--lookback-days", default=10, type=int)
    parser.add_argument("--max-horizon-days", default=5, type=int)
    parser.add_argument("--required-signal-days", default=600, type=int)
    arguments = parser.parse_args()
    try:
        audit = audit_pandadata_coverage(
            arguments.manifest,
            min_complete_instruments=arguments.min_complete_instruments,
            lookback_days=arguments.lookback_days,
            max_horizon_days=arguments.max_horizon_days,
            required_signal_days=arguments.required_signal_days,
        )
        receipt_path = write_pandadata_coverage_receipt(
            audit, output_dir=arguments.output_dir
        )
        payload = {
            "status": audit.status,
            "blockers": list(audit.blockers),
            "receipt_path": str(receipt_path),
            "candidate_signal_day_count": len(audit.candidate_signal_days),
        }
    except (ContractError, OSError, ValueError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
