"""Re-benchmark sequence models from immutable preprocessing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.performance import benchmark_sequence_models  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark an immutable sequence run")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--hidden-size", default=8, type=int)
    parser.add_argument("--tcn-channels", default=3, type=int)
    parser.add_argument("--tcn-lite-channels", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--include-tcn-lite", action="store_true")
    parser.add_argument(
        "--models",
        help="comma-separated model subset: lstm,gru,bai-tcn,tcn-lite",
    )
    arguments = parser.parse_args()
    try:
        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "labels": run_dir / "labels.parquet",
            "split_manifest": (
                arguments.split_manifest.resolve()
                if arguments.split_manifest is not None
                else run_dir / "split-manifest.parquet"
            ),
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError(f"benchmark source artifacts missing: {', '.join(missing)}")
        output_dir = arguments.output_dir.resolve()
        metrics_path = output_dir / "performance-metrics.parquet"
        environment_path = output_dir / "performance-environment.json"
        receipt_path = output_dir / "receipt.json"
        if any(path.exists() for path in (metrics_path, environment_path, receipt_path)):
            raise ContractError("sequence benchmark refuses to overwrite artifacts")
        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        seeds = tuple(
            int(value.strip())
            for value in arguments.seeds.split(",")
            if value.strip()
        )
        models = (
            tuple(
                value.strip()
                for value in arguments.models.split(",")
                if value.strip()
            )
            if arguments.models
            else None
        )
        result = benchmark_sequence_models(
            features,
            pd.read_parquet(source_paths["window_index"]),
            pd.read_parquet(source_paths["labels"]),
            pd.read_parquet(source_paths["split_manifest"]),
            seed=7,
            seeds=seeds,
            hidden_size=arguments.hidden_size,
            tcn_channels=arguments.tcn_channels,
            tcn_kernel_size=3,
            tcn_dilations=(1, 2, 4, 8, 16, 32, 64),
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            device="cpu",
            torch_threads=arguments.torch_threads,
            num_workers=arguments.num_workers,
            include_tcn_lite=arguments.include_tcn_lite,
            tcn_lite_channels=arguments.tcn_lite_channels,
            learning_rate=arguments.learning_rate,
            models=models,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        result.measurements.to_parquet(metrics_path, index=False)
        environment_path.write_text(
            json.dumps(result.environment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt = {
            "schema_version": 2,
            "source_artifacts": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in source_paths.items()
            },
            "protocol": {
                "hidden_size": arguments.hidden_size,
                "tcn_channels": arguments.tcn_channels,
                "epochs": arguments.epochs,
                "batch_size": arguments.batch_size,
                "seeds": list(seeds),
                "torch_threads": arguments.torch_threads,
                "num_workers": arguments.num_workers,
                "learning_rate": arguments.learning_rate,
                "include_tcn_lite": arguments.include_tcn_lite,
                "tcn_lite_channels": arguments.tcn_lite_channels,
                "models": list(models) if models is not None else None,
                "precision": "float32",
                "device": "cpu",
            },
            "outputs": {
                metrics_path.name: _sha256(metrics_path),
                environment_path.name: _sha256(environment_path),
            },
            "sealed_test_accessed": False,
        }
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        payload = {"status": "success", "receipt": str(receipt_path)}
    except (OSError, ValueError, KeyError, ContractError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
