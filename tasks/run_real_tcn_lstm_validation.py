"""Run an immutable seed-7 real TCN screen with a fair LSTM benchmark."""

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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.performance import (  # noqa: E402
    benchmark_sequence_models,
)
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    build_tcn_lstm_comparison,
    finalize_seed7_benchmark_gate,
    parse_real_tcn_trials,
    select_seed7_tcn_candidate,
)
from skill_dl_tcn_shortterm.tuning import run_tcn_validation_sweep  # noqa: E402
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(
                marker in str(key).lower()
                for marker in ["password", "token", "secret", "credential"]
            ):
                return True
            if _contains_secret_key(nested):
                return True
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded real-data TCN tuning and a parameter-matched LSTM benchmark"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()

    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("real TCN/LSTM validation refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ContractError("real validation config must contain an object")
        if _contains_secret_key(config):
            raise ContractError("real validation config contains a secret-like key")
        if config.get("precision") != "float32":
            raise ContractError("real validation precision must remain float32")
        if int(cast(Any, config["seed"])) != 7:
            raise ContractError("real validation screen must use seed 7")

        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "labels": run_dir / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError(f"real validation sources missing: {', '.join(missing)}")
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("real validation source SHA-256 identity drifted")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("real validation rejects sealed split rows")
        allowed_stages = {"train", "validation", "purged"}
        observed_stages = {
            str(value) for value in raw_split["stage"].astype(str).tolist()
        }
        if unknown := sorted(observed_stages.difference(allowed_stages)):
            raise ContractError(
                f"real validation split contains forbidden stages: {', '.join(unknown)}"
            )
        folds = {
            int(cast(Any, value))
            for value in cast(list[object], config["folds"])
        }
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(folds)
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()
        if set(split_manifest["fold"].astype(int)) != folds:
            raise ContractError("real validation config references unavailable folds")

        trials = parse_real_tcn_trials(config["trials"])
        protocol_identities = {
            "data": observed_hashes["features"],
            "fold_manifest": observed_hashes["split_manifest"],
            "evaluation": observed_hashes["labels"],
        }
        tuning = run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split_manifest,
            trials=trials,
            seed=7,
            max_epochs=int(cast(Any, config["max_epochs"])),
            patience=int(cast(Any, config["patience"])),
            min_delta=float(cast(Any, config["min_delta"])),
            torch_threads=int(cast(Any, config["torch_threads"])),
            protocol_identities=protocol_identities,
        )
        gates = cast(dict[str, object], config["gates"])
        decision = select_seed7_tcn_candidate(
            tuning.leaderboard,
            control_trial_id=str(config["control_trial_id"]),
            min_mean_rankic=float(cast(Any, gates["min_mean_rankic"])),
            min_positive_folds=int(cast(Any, gates["min_positive_folds"])),
            min_median_samples_per_second=float(
                cast(Any, gates["min_median_samples_per_second"])
            ),
        )
        configured_comparison_trial_id = config.get("comparison_trial_id")
        if configured_comparison_trial_id is not None and str(
            configured_comparison_trial_id
        ) not in {trial.trial_id for trial in trials}:
            raise ContractError("configured comparison trial is not registered")
        comparison_trial_id = (
            decision.winner_trial_id
            if decision.winner_trial_id is not None
            else (
                str(configured_comparison_trial_id)
                if configured_comparison_trial_id is not None
                else str(decision.summary.iloc[0]["trial_id"])
            )
        )

        benchmark_config = cast(dict[str, object], config["lstm_benchmark"])
        reference_trial = trials[0]
        lstm = benchmark_sequence_models(
            features,
            window_index,
            labels,
            split_manifest,
            seed=7,
            seeds=(7,),
            hidden_size=int(cast(Any, benchmark_config["hidden_size"])),
            tcn_channels=reference_trial.channels,
            tcn_kernel_size=reference_trial.kernel_size,
            tcn_dilations=reference_trial.dilations[:-1],
            epochs=int(cast(Any, benchmark_config["epochs"])),
            batch_size=int(cast(Any, benchmark_config["batch_size"])),
            device="cpu",
            num_workers=int(cast(Any, config["num_workers"])),
            torch_threads=int(cast(Any, config["torch_threads"])),
            learning_rate=float(cast(Any, benchmark_config["learning_rate"])),
            models=("lstm",),
        )
        comparison_rows = tuning.leaderboard.loc[
            tuning.leaderboard["trial_id"].eq(comparison_trial_id)
        ].copy()
        comparison: dict[str, object] = dict(
            build_tcn_lstm_comparison(comparison_rows, lstm.measurements)
        )
        comparison["tcn_trial_id"] = comparison_trial_id

        relative_speed_gates_enabled = {
            "min_model_step_speed_ratio",
            "min_end_to_end_speed_ratio",
        }.issubset(gates)
        final_decision = (
            finalize_seed7_benchmark_gate(
                decision,
                cast(dict[str, float | int], comparison),
                min_model_step_speed_ratio=float(
                    cast(Any, gates["min_model_step_speed_ratio"])
                ),
                min_end_to_end_speed_ratio=float(
                    cast(Any, gates["min_end_to_end_speed_ratio"])
                ),
            )
            if relative_speed_gates_enabled
            else None
        )

        temporary.mkdir(parents=True)
        tuning.epoch_history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        tuning.leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        decision.summary.to_parquet(temporary / "tcn-summary.parquet", index=False)
        lstm.measurements.to_parquet(temporary / "lstm-measurements.parquet", index=False)
        _write_json(temporary / "lstm-environment.json", lstm.environment)
        _write_json(temporary / "comparison.json", comparison)
        selection = {
            "status": (
                final_decision.status if final_decision is not None else decision.status
            ),
            "effect_gate_status": decision.status,
            "winner_trial_id": (
                final_decision.winner_trial_id
                if final_decision is not None
                else decision.winner_trial_id
            ),
            "comparison_trial_id": comparison_trial_id,
            "relative_speed_gate_passed": (
                final_decision.relative_speed_gate_passed
                if final_decision is not None
                else None
            ),
            "confirmation_seeds_authorized": (
                list(final_decision.confirmation_seeds_authorized)
                if final_decision is not None
                else ([17, 27] if decision.winner_trial_id is not None else [])
            ),
        }
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for checkpoint_key, state in tuning.best_states.items():
            torch.save(state, checkpoint_dir / f"{checkpoint_key}.pt")

        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": (
                "tcn-real-validation-v13/v1"
                if any(
                    trial.model_kind == "signed_temporal_context"
                    for trial in trials
                )
                else (
                    "tcn-real-validation-v12/v1"
                    if any(
                        trial.model_kind == "temporal_context"
                        or trial.strategy == "soft_rankic"
                        for trial in trials
                    )
                    else (
                        "tcn-real-validation-v11/v1"
                        if relative_speed_gates_enabled
                        else "tcn-real-validation-v10/v1"
                    )
                )
            ),
            "run_id": str(config["run_id"]),
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "source_config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
            },
            "selection": selection,
            "comparison": comparison,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "result": selection["status"],
            "output_dir": str(output_dir),
            "receipt_id": receipt["receipt_id"],
        }
    except (
        ContractError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
