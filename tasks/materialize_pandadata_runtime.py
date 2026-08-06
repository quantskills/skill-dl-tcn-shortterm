"""Materialize a bounded single-file PandaData runtime slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.pandadata_source import (  # noqa: E402
    materialize_pandadata_runtime_slice,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize PandaData runtime data")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", required=True, type=int)
    arguments = parser.parse_args()
    try:
        manifest_path = materialize_pandadata_runtime_slice(
            arguments.manifest,
            output_dir=arguments.output_dir,
            top_n=arguments.top_n,
        )
        payload = {"status": "success", "manifest_path": str(manifest_path)}
    except (ContractError, OSError, ValueError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
