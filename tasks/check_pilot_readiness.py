"""Check whether an owner-supplied real-data pilot is ready to run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError, check_pilot_readiness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a TCN real-data pilot readiness descriptor"
    )
    parser.add_argument("--descriptor", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        report = check_pilot_readiness(arguments.descriptor)
        payload = report.to_dict()
    except (ContractError, OSError) as exc:
        payload = {
            "ready": False,
            "checks": [],
            "errors": [str(exc)],
            "warnings": ["No experiment or sealed evaluation was started."],
            "descriptor_sha256": None,
            "runtime_manifest_sha256": None,
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
