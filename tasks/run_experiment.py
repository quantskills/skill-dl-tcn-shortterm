"""Command-line entry point for one offline experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError, run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an immutable offline TCN experiment"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ContractError("config file must contain a JSON object")
        result = run_experiment(
            config=config,
            manifest_path=arguments.manifest,
            output_root=arguments.output_root,
        )
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        print(
            json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "success",
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
