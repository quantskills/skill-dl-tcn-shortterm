"""Validate and publish an immutable TCN-v9 ordinary-validation plan receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.v9_protocol import (  # noqa: E402
    execute_v9_plan,
    parse_v9_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a bounded TCN-v9 plan")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        raw = json.loads(arguments.config.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ContractError("v9 config must contain an object")
        plan = parse_v9_plan(raw)
        split_manifest = pd.read_parquet(arguments.split_manifest)
        receipt_path = execute_v9_plan(
            plan,
            split_manifest,
            output_dir=arguments.output_dir,
            project_root=ROOT,
        )
        payload = {"status": "success", "receipt_path": str(receipt_path)}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, ContractError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
