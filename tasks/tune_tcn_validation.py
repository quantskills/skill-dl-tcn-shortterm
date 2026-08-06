"""Run immutable, ordinary-validation-only Bai TCN tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    TCNTuningTrial,
    run_tcn_validation_sweep,
    select_tcn_candidate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(marker in str(key).lower() for marker in ["password", "token", "secret", "credential"]):
                return True
            if _contains_secret_key(nested):
                return True
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _load_trials(raw_trials: object) -> list[TCNTuningTrial]:
    if not isinstance(raw_trials, list):
        raise ContractError("tuning config trials must be a list")
    trials = []
    for raw in raw_trials:
        if not isinstance(raw, dict):
            raise ContractError("each tuning trial must be an object")
        model_kind = str(raw.get("model_kind", "bai"))
        if model_kind not in {"bai", "lite"}:
            raise ContractError("TCN model kind must be bai or lite")
        dropout_kind = str(raw.get("dropout_kind", "element"))
        if dropout_kind not in {"element", "channel"}:
            raise ContractError("dropout kind must be element or channel")
        trials.append(
            TCNTuningTrial(
                trial_id=str(raw["trial_id"]),
                channels=int(raw["channels"]),
                kernel_size=int(raw["kernel_size"]),
                dilations=tuple(int(value) for value in raw["dilations"]),
                dropout=float(raw["dropout"]),
                learning_rate=float(raw["learning_rate"]),
                batch_size=int(raw["batch_size"]),
                model_kind=cast(Literal["bai", "lite"], model_kind),
                head_dropout=float(raw.get("head_dropout", 0.0)),
                dropout_kind=cast(
                    Literal["element", "channel"], dropout_kind
                ),
                weight_decay=float(raw.get("weight_decay", 0.0)),
            )
        )
    return trials


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune Bai TCN on ordinary validation")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("TCN tuning task refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ContractError("tuning config must contain an object")
        if _contains_secret_key(config):
            raise ContractError("tuning config contains a forbidden secret-like key")
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
            raise ContractError(f"tuning source artifacts missing: {', '.join(missing)}")
        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        split_manifest = pd.read_parquet(source_paths["split_manifest"])
        folds = {int(value) for value in config["folds"]}
        split_manifest = split_manifest.loc[
            split_manifest["fold"].isin(folds)
            & split_manifest["stage"].isin(["train", "validation"])
        ].copy()
        if set(split_manifest["fold"].astype(int)) != folds:
            raise ContractError("tuning config references unavailable folds")
        result = run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split_manifest,
            trials=_load_trials(config["trials"]),
            seed=int(config["seed"]),
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
            min_delta=float(config["min_delta"]),
            torch_threads=(
                int(config["torch_threads"])
                if config.get("torch_threads") is not None
                else None
            ),
        )
        ranked = (
            result.leaderboard.groupby("trial_id", observed=True)
            .agg(
                mean_rankic=("best_mean_daily_rankic", "mean"),
                parameter_count=("parameter_count", "first"),
                samples_per_second=("samples_per_second", "mean"),
            )
            .reset_index()
            .sort_values(
                ["mean_rankic", "parameter_count", "samples_per_second", "trial_id"],
                ascending=[False, True, False, True],
                kind="mergesort",
            )
        )
        ranked_ids = ranked["trial_id"].astype(str).tolist()
        if bool(config.get("apply_scale_gate", False)):
            decision = select_tcn_candidate(
                result.leaderboard,
                control_trial_id=str(config["control_trial_id"]),
                min_improvement=float(config.get("scale_gate_min_improvement", 0.01)),
            )
            selection: dict[str, Any] = {
                "status": decision.status,
                "selected_trial_id": decision.selected_trial_id,
                "mean_improvement": decision.mean_improvement,
                "non_degrading_horizon_count": decision.non_degrading_horizon_count,
                "ranked_trial_ids": ranked_ids,
            }
        else:
            selection = {
                "status": "screen_complete",
                "selected_trial_id": ranked_ids[0],
                "ranked_trial_ids": ranked_ids,
            }
        temporary.mkdir(parents=True)
        history_path = temporary / "epoch-history.parquet"
        leaderboard_path = temporary / "leaderboard.parquet"
        selection_path = temporary / "selection.json"
        config_copy = temporary / "config.resolved.json"
        result.epoch_history.to_parquet(history_path, index=False)
        result.leaderboard.to_parquet(leaderboard_path, index=False)
        selection_path.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_copy.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for checkpoint_key, state in result.best_states.items():
            torch.save(state, checkpoint_dir / f"{checkpoint_key}.pt")
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt = {
            "schema_version": 1,
            "source_artifacts": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in source_paths.items()
            },
            "source_config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "selection": selection,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        (temporary / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
        payload = {
            "status": "success",
            "output_dir": str(output_dir),
            "selection": selection,
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ContractError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
