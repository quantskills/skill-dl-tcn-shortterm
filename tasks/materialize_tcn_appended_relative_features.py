"""Materialize the immutable v38 base8-plus-relative2 feature candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.relative_features import (  # noqa: E402
    APPENDED_SEQUENCE_FEATURE_VERSION,
    materialize_appended_relative_sequence_features,
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
        description="Materialize the v38 append-only relative sequence feature tensor"
    )
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--relative-feature-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=512)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v38 feature materialization refuses to overwrite")
        base = arguments.base_run_dir.resolve()
        relative = arguments.relative_feature_dir.resolve()
        source_paths = {
            "base_features": base / "feature-windows.npy",
            "base_window_index": base / "window-index.parquet",
            "relative_features": relative / "feature-windows.npy",
            "relative_window_index": relative / "window-index.parquet",
            "relative_manifest": relative / "manifest.json",
            "relative_receipt": relative / "receipt.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v38 feature sources missing: " + ", ".join(missing))
        source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        manifest = cast(
            dict[str, object],
            json.loads(source_paths["relative_manifest"].read_text(encoding="utf-8")),
        )
        receipt = cast(
            dict[str, object],
            json.loads(source_paths["relative_receipt"].read_text(encoding="utf-8")),
        )
        if manifest.get("feature_version") != "causal-relative-cross-sectional-v37":
            raise ContractError("v38 relative source feature identity drifted")
        if manifest.get("sealed_test_accessed") is not False or receipt.get(
            "sealed_test_accessed"
        ) is not False:
            raise ContractError("v38 relative source is not sealed-test fail-closed")

        base_features = np.load(
            source_paths["base_features"], mmap_mode="r", allow_pickle=False
        )
        relative_features = np.load(
            source_paths["relative_features"], mmap_mode="r", allow_pickle=False
        )
        base_index = pd.read_parquet(source_paths["base_window_index"])
        relative_index = pd.read_parquet(source_paths["relative_window_index"])
        identity_columns = ["sample_position", "sample_id", "instrument_id", "signal_date"]
        if any(
            not np.array_equal(
                base_index[column].astype(str).to_numpy(),
                relative_index[column].astype(str).to_numpy(),
            )
            for column in identity_columns
        ):
            raise ContractError("v38 base and relative sample identities drifted")

        temporary.mkdir(parents=True)
        result = materialize_appended_relative_sequence_features(
            base_features,
            relative_features,
            base_index,
            output_path=temporary / "feature-windows.npy",
            chunk_size=arguments.chunk_size,
        )
        result.window_index.to_parquet(
            temporary / "window-index.parquet", index=False
        )
        result.audit.to_parquet(temporary / "feature-audit.parquet", index=False)
        _write_json(temporary / "feature-quality.json", result.quality)
        output_manifest: dict[str, Any] = {
            "schema_version": "tcn-appended-relative-sequence-v38/v1",
            "feature_version": APPENDED_SEQUENCE_FEATURE_VERSION,
            "source_artifacts": {
                name: {"path": str(path), "sha256": source_hashes[name]}
                for name, path in source_paths.items()
            },
            "quality": result.quality,
            "sealed_test_accessed": False,
        }
        _write_json(temporary / "manifest.json", output_manifest)
        mapped = getattr(result.features, "_mmap", None)
        if mapped is not None:
            mapped.close()
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        output_receipt: dict[str, Any] = {
            "schema_version": "tcn-appended-relative-feature-materialization-v38/v1",
            "run_id": "pandadata-tcn-appended-relative-sequence-top20-v38",
            "feature_version": APPENDED_SEQUENCE_FEATURE_VERSION,
            "source_artifacts": output_manifest["source_artifacts"],
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "storage": "read_only_npy_memmap",
                "chunk_size": arguments.chunk_size,
            },
            "quality": result.quality,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        output_receipt["receipt_id"] = hashlib.sha256(
            canonical_bytes(output_receipt)
        ).hexdigest()
        _write_json(temporary / "receipt.json", output_receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "output_dir": str(output_dir),
            "feature_version": APPENDED_SEQUENCE_FEATURE_VERSION,
            "receipt_id": output_receipt["receipt_id"],
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
