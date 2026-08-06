"""Freeze v35 for a future once-only sealed evaluation without consuming it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.sealed_readiness import (  # noqa: E402
    freeze_v35_sealed_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze v35 and build metadata-only sealed readiness"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        config = json.loads(arguments.config.resolve().read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ContractError("v36 config must be a JSON object")
        result = freeze_v35_sealed_readiness(ROOT, config, arguments.output_dir)
        payload = {
            "status": "success",
            "result": "awaiting_explicit_sealed_authorization_v36",
            "freeze_id": result.freeze_id,
            "output_dir": str(result.output_dir),
            "receipt": str(result.receipt_path),
            "sealed_test_accessed": False,
        }
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
