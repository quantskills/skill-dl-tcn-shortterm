"""Materialize v39 top50 training tensors without running unrelated baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.experiment import _canonical_json  # noqa: E402
from skill_dl_tcn_shortterm.features import (  # noqa: E402
    build_feature_windows_with_quality,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.labels import build_labels  # noqa: E402
from skill_dl_tcn_shortterm.market_data import (  # noqa: E402
    aggregate_five_minute_bars,
)
from skill_dl_tcn_shortterm.splits import build_walk_forward_splits  # noqa: E402
from skill_dl_tcn_shortterm.universe import build_pit_universe  # noqa: E402
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_optional_frame(
    manifest: dict[str, Any], manifest_path: Path, path_key: str, sha_key: str
) -> pd.DataFrame:
    path = (manifest_path.parent / str(manifest.get(path_key, ""))).resolve()
    if not path.is_file() or _sha256(path) != manifest.get(sha_key):
        raise ContractError(f"v39 manifest fingerprint mismatch: {path_key}")
    return pd.read_parquet(path)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize immutable PandaData top50 v39 training data"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", default=512, type=int)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v39 top50 training materialization refuses to overwrite")
        if arguments.chunk_size <= 0:
            raise ContractError("v39 output chunk size must be positive")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ContractError("v39 preprocessing config must contain an object")
        manifest_path = arguments.manifest.resolve()
        manifest = cast(
            dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        materialization = manifest.get("materialization")
        enrichment = manifest.get("enrichment")
        if manifest.get("dataset_kind") != "raw_1m":
            raise ContractError("v39 requires raw_1m PandaData runtime")
        if not isinstance(materialization, dict) or materialization.get("top_n") != 50:
            raise ContractError("v39 requires PIT top50 materialization")
        if not isinstance(enrichment, dict) or enrichment.get("schema_version") != 2:
            raise ContractError("v39 requires PIT enrichment schema v2")
        data_path = (manifest_path.parent / str(manifest.get("data_path", ""))).resolve()
        if not data_path.is_file() or _sha256(data_path) != manifest.get("data_sha256"):
            raise ContractError("v39 runtime bars fingerprint mismatch")
        started = perf_counter()
        stage_seconds: dict[str, float] = {}

        dataset = pd.read_parquet(data_path)
        stage_seconds["load_raw_1m"] = perf_counter() - started
        stage_started = perf_counter()
        canonical_bars, quality = aggregate_five_minute_bars(dataset, manifest)
        del dataset
        stage_seconds["aggregate_5m"] = perf_counter() - stage_started

        stage_started = perf_counter()
        states = _verified_optional_frame(
            manifest,
            manifest_path,
            "instrument_state_path",
            "instrument_state_sha256",
        )
        universe = build_pit_universe(
            canonical_bars, states, cast(dict[str, Any], config.get("universe", {}))
        )
        universe["universe_version"] = "pit-a-share-v1"
        universe["state_fingerprint"] = manifest["instrument_state_sha256"]
        universe["admission_config"] = _canonical_json(
            cast(dict[str, Any], config.get("universe", {}))
        )
        stage_seconds["build_pit_universe"] = perf_counter() - stage_started

        stage_started = perf_counter()
        feature_windows, window_index, window_rejections = (
            build_feature_windows_with_quality(
                canonical_bars,
                universe,
                lookback_days=int(cast(Any, config.get("lookback_days", 10))),
                source_fingerprint=str(manifest["data_sha256"]),
            )
        )
        quality["training_window_storage"] = "read_only_memmap"
        quality["valid_window_count"] = len(window_index)
        quality["rejected_window_count"] = len(window_rejections)
        quality["window_rejection_reasons"] = (
            window_rejections["rejection_reason"].value_counts().to_dict()
            if not window_rejections.empty
            else {}
        )
        stage_seconds["build_feature_windows"] = perf_counter() - stage_started

        stage_started = perf_counter()
        corporate_actions = _verified_optional_frame(
            manifest,
            manifest_path,
            "corporate_action_path",
            "corporate_action_sha256",
        )
        horizons = [
            int(cast(Any, value)) for value in cast(list[object], config["horizons"])
        ]
        labels = build_labels(
            window_index,
            canonical_bars,
            horizons=horizons,
            corporate_actions=corporate_actions,
        )
        stage_seconds["build_labels"] = perf_counter() - stage_started

        stage_started = perf_counter()
        walk = cast(dict[str, Any], config["walk_forward"])
        split_result = build_walk_forward_splits(
            window_index,
            labels,
            feature_windows,
            train_days=int(walk["train_days"]),
            validation_days=int(walk["validation_days"]),
            embargo_days=int(walk.get("embargo_days", 5)),
            test_days=int(walk["test_days"]),
            max_folds=walk.get("max_folds"),
        )
        stage_seconds["build_source_splits"] = perf_counter() - stage_started

        temporary.mkdir(parents=True)
        stage_started = perf_counter()
        feature_path = temporary / "feature-windows.npy"
        mapped = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype="float32",
            shape=feature_windows.shape,
        )
        for start in range(0, len(feature_windows), arguments.chunk_size):
            stop = min(len(feature_windows), start + arguments.chunk_size)
            mapped[start:stop] = np.asarray(feature_windows[start:stop], dtype="float32")
        mapped.flush()
        mmap_handle = getattr(mapped, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
        canonical_bars.to_parquet(temporary / "bars_5m.parquet", index=False)
        universe.to_parquet(temporary / "universe.parquet", index=False)
        window_index.to_parquet(temporary / "window-index.parquet", index=False)
        window_rejections.to_parquet(
            temporary / "window-rejections.parquet", index=False
        )
        labels.to_parquet(temporary / "labels.parquet", index=False)
        split_result.manifest.to_parquet(
            temporary / "split-manifest.parquet", index=False
        )
        _write_json(temporary / "data-quality.json", quality)
        (temporary / "preprocessing.json").write_text(
            _canonical_json(split_result.preprocessing) + "\n", encoding="utf-8"
        )
        _write_json(temporary / "config.resolved.json", config)
        (temporary / "input-manifest.json").write_text(
            _canonical_json(manifest) + "\n", encoding="utf-8"
        )
        stage_seconds["write_artifacts"] = perf_counter() - stage_started
        stage_seconds["total"] = perf_counter() - started
        _write_json(temporary / "timings.json", stage_seconds)

        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "pandadata-top50-training-materialization-v39/v1",
            "run_id": "pandadata-top50-training-v39",
            "source_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "data_sha256": manifest["data_sha256"],
                "instrument_state_sha256": manifest["instrument_state_sha256"],
                "corporate_action_sha256": manifest["corporate_action_sha256"],
            },
            "source_config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "storage": "npy_memmap",
                "chunk_size": arguments.chunk_size,
            },
            "quality": quality,
            "timings": stage_seconds,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "output_dir": str(output_dir),
            "sample_count": len(window_index),
            "receipt_id": receipt["receipt_id"],
            "total_seconds": stage_seconds["total"],
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
