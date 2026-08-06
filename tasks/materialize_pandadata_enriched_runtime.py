"""Build an enriched runtime manifest from downloaded PandaData state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.pandadata_source import (  # noqa: E402
    build_pandadata_causal_states,
    materialize_pandadata_enriched_runtime,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize enriched PandaData runtime")
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--enrichment-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        source = arguments.enrichment_manifest.resolve()
        manifest = json.loads(source.read_text(encoding="utf-8"))
        frames = {}
        for name, artifact in manifest["artifacts"].items():
            path = source.parent / artifact["path"]
            if _sha256(path) != artifact["sha256"]:
                raise ContractError(f"enrichment artifact fingerprint mismatch: {name}")
            frames[name] = pd.read_parquet(path).drop(
                columns=["_empty_response"], errors="ignore"
            )
        states = build_pandadata_causal_states(
            frames["daily"],
            membership=frames["membership"],
            share_float=frames["share_float"],
        )
        result = materialize_pandadata_enriched_runtime(
            arguments.runtime_manifest,
            states=states,
            corporate_actions=frames["corporate_actions"],
            output_dir=arguments.output_dir,
            enrichment_identity={
                "manifest_sha256": _sha256(source),
                "source_version": manifest["source_version"],
            },
        )
        payload = {"status": "success", "manifest_path": str(result)}
    except (OSError, ValueError, KeyError, ContractError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
