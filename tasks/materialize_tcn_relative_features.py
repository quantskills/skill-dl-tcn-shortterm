"""Materialize the immutable v37 relative-feature candidate and top50 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.relative_features import (  # noqa: E402
    FEATURE_VERSION,
    audit_top_n_state_readiness,
    materialize_causal_relative_features,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize causal relative feature windows and audit top50 readiness"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--top100-universe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=512)
    arguments = parser.parse_args()

    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v37 relative feature materialization refuses to overwrite")
        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "universe": run_dir / "universe.parquet",
            "input_manifest": run_dir / "input-manifest.json",
            "top100_universe": arguments.top100_universe.resolve(),
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v37 sources missing: " + ", ".join(missing))
        source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        input_manifest = json.loads(
            source_paths["input_manifest"].read_text(encoding="utf-8")
        )
        if not isinstance(input_manifest, dict):
            raise ContractError("v37 input manifest must contain an object")
        if input_manifest.get("materialization", {}).get("top_n") != 20:
            raise ContractError("v37 source run must be the immutable PIT top20 materialization")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        universe = pd.read_parquet(source_paths["universe"])
        top100 = pd.read_parquet(source_paths["top100_universe"])
        readiness = audit_top_n_state_readiness(top100, universe, top_n=50)

        temporary.mkdir(parents=True)
        result = materialize_causal_relative_features(
            features,
            window_index,
            universe,
            output_path=temporary / "feature-windows.npy",
            bars_per_day=48,
            chunk_size=arguments.chunk_size,
            min_cross_section=10,
        )
        result.window_index.to_parquet(
            temporary / "window-index.parquet", index=False
        )
        result.audit.to_parquet(temporary / "feature-audit.parquet", index=False)
        _write_json(temporary / "feature-quality.json", result.quality)
        readiness_payload = {
            "status": readiness.status,
            "ready": readiness.ready,
            **readiness.evidence,
        }
        _write_json(temporary / "top50-readiness.json", readiness_payload)
        manifest: dict[str, Any] = {
            "schema_version": "tcn-relative-features-v37/v1",
            "feature_version": FEATURE_VERSION,
            "source_artifacts": {
                name: {"path": str(path), "sha256": source_hashes[name]}
                for name, path in source_paths.items()
            },
            "quality": result.quality,
            "top50_readiness": readiness_payload,
            "sealed_test_accessed": False,
        }
        _write_json(temporary / "manifest.json", manifest)

        mapped = getattr(result.features, "_mmap", None)
        if mapped is not None:
            mapped.close()
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-relative-feature-materialization-v37/v1",
            "run_id": "pandadata-tcn-relative-features-top20-v37",
            "feature_version": FEATURE_VERSION,
            "source_artifacts": manifest["source_artifacts"],
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "storage": "read_only_npy_memmap",
                "chunk_size": arguments.chunk_size,
            },
            "quality": result.quality,
            "top50_readiness": readiness_payload,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "output_dir": str(output_dir),
            "feature_version": FEATURE_VERSION,
            "top50_status": readiness.status,
            "receipt_id": receipt["receipt_id"],
        }
    except (
        ContractError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
