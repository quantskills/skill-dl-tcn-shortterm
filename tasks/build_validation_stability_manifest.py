"""Build immutable ordinary-validation stability folds from a source run."""

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
from skill_dl_tcn_shortterm.stability import (  # noqa: E402
    build_validation_stability_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe ordinary-validation stability folds"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-fold", default=1, type=int)
    parser.add_argument("--train-days", default=400, type=int)
    parser.add_argument("--validation-days", default=80, type=int)
    parser.add_argument("--fold-count", default=5, type=int)
    parser.add_argument(
        "--window-kind", choices=["expanding", "sliding"], required=True
    )
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("stability manifest task refuses to overwrite artifacts")
        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "labels": run_dir / "labels.parquet",
            "split_manifest": run_dir / "split-manifest.parquet",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError(
                f"stability manifest source artifacts missing: {', '.join(missing)}"
            )
        result = build_validation_stability_manifest(
            pd.read_parquet(source_paths["window_index"]),
            pd.read_parquet(source_paths["labels"]),
            pd.read_parquet(source_paths["split_manifest"]),
            source_fold=arguments.source_fold,
            train_days=arguments.train_days,
            validation_days=arguments.validation_days,
            fold_count=arguments.fold_count,
            window_kind=arguments.window_kind,
        )
        temporary.mkdir(parents=True)
        manifest_path = temporary / "validation-stability-manifest.parquet"
        summary_path = temporary / "validation-stability-summary.parquet"
        result.manifest.to_parquet(manifest_path, index=False)
        result.summary.to_parquet(summary_path, index=False)
        receipt = {
            "schema_version": 1,
            "protocol": {
                "source_fold": arguments.source_fold,
                "train_days": arguments.train_days,
                "validation_days": arguments.validation_days,
                "fold_count": arguments.fold_count,
                "window_kind": arguments.window_kind,
                "data_fingerprint": result.fingerprint,
            },
            "source_artifacts": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in source_paths.items()
            },
            "outputs": {
                manifest_path.name: _sha256(manifest_path),
                summary_path.name: _sha256(summary_path),
            },
            "sealed_test_accessed": False,
        }
        (temporary / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_dir)
        payload = {
            "status": "success",
            "output_dir": str(output_dir),
            "data_fingerprint": result.fingerprint,
        }
    except (OSError, ValueError, KeyError, TypeError, ContractError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
