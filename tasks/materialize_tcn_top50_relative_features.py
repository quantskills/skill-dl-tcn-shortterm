"""Materialize the immutable v39 top50 base8-plus-relative2 tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
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
    TOP50_APPENDED_SEQUENCE_FEATURE_VERSION,
    materialize_top50_appended_relative_sequence_features,
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
        description="Materialize audited v39 top50 append-only relative features"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--min-cross-section", type=int, default=31)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v39 top50 feature materialization refuses to overwrite")
        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "universe": run_dir / "universe.parquet",
            "input_manifest": run_dir / "input-manifest.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v39 top50 sources missing: " + ", ".join(missing))
        source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        input_manifest = json.loads(
            source_paths["input_manifest"].read_text(encoding="utf-8")
        )
        if not isinstance(input_manifest, dict):
            raise ContractError("v39 input manifest must contain an object")
        materialization = input_manifest.get("materialization")
        enrichment = input_manifest.get("enrichment")
        if not isinstance(materialization, dict) or materialization.get("top_n") != 50:
            raise ContractError("v39 requires an immutable PIT top50 runtime")
        if not isinstance(enrichment, dict) or enrichment.get("schema_version") != 2:
            raise ContractError("v39 requires PIT enrichment schema v2")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        universe = pd.read_parquet(source_paths["universe"])
        temporary.mkdir(parents=True)
        result = materialize_top50_appended_relative_sequence_features(
            features,
            window_index,
            universe,
            output_path=temporary / "feature-windows.npy",
            bars_per_day=48,
            chunk_size=arguments.chunk_size,
            min_cross_section=arguments.min_cross_section,
        )
        minimum_width = int(result.audit["cross_section_count"].min())
        minimum_top_count = int(ceil(minimum_width * 0.1))
        quality = {
            **result.quality,
            "minimum_top10pct_count": minimum_top_count,
            "effective_breadth_gate_passed": minimum_top_count >= 4,
        }
        if minimum_top_count < 4:
            raise ContractError("v39 effective breadth cannot support four top holdings")
        result.window_index.to_parquet(
            temporary / "window-index.parquet", index=False
        )
        result.audit.to_parquet(temporary / "feature-audit.parquet", index=False)
        _write_json(temporary / "feature-quality.json", quality)
        manifest: dict[str, Any] = {
            "schema_version": "tcn-top50-appended-relative-sequence-v39/v1",
            "feature_version": TOP50_APPENDED_SEQUENCE_FEATURE_VERSION,
            "source_artifacts": {
                name: {"path": str(path), "sha256": source_hashes[name]}
                for name, path in source_paths.items()
            },
            "quality": quality,
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
            "schema_version": "tcn-top50-relative-feature-materialization-v39/v1",
            "run_id": "pandadata-tcn-top50-appended-relative-sequence-v39",
            "feature_version": TOP50_APPENDED_SEQUENCE_FEATURE_VERSION,
            "source_artifacts": manifest["source_artifacts"],
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "storage": "read_only_npy_memmap",
                "chunk_size": arguments.chunk_size,
            },
            "quality": quality,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "output_dir": str(output_dir),
            "feature_version": TOP50_APPENDED_SEQUENCE_FEATURE_VERSION,
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
