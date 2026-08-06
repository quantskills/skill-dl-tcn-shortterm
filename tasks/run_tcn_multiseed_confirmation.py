"""Run pre-authorized TCN confirmation seeds without sealed data."""

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.dynamic_multiscale import (  # noqa: E402
    evaluate_dynamic_multiscale_multiseed,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.performance import benchmark_sequence_models  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    build_tcn_lstm_comparison,
    evaluate_signed_multiseed_confirmation,
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.training_data import (  # noqa: E402
    LazyWindowDataset,
    build_fold_protocols,
)
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    TCNTuningTrial,
    build_tcn_trial_model,
    run_tcn_validation_sweep,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402
from skill_dl_tcn_shortterm.v9_representation import (  # noqa: E402
    DynamicHorizonSkipTCN,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            any(
                marker in str(key).lower()
                for marker in ["password", "token", "secret", "credential"]
            )
            or _contains_secret_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else ROOT / path).resolve()


def _validate_parent(config: dict[str, object]) -> dict[str, object]:
    parent = _resolve_project_path(config["parent_artifact"])
    receipt_path = parent / "receipt.json"
    selection_path = parent / "selection.json"
    if not receipt_path.is_file() or not selection_path.is_file():
        raise ContractError("v14 parent artifact is incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or not isinstance(selection, dict):
        raise ContractError("v14 parent evidence must contain objects")
    expected_receipt_id = str(config["parent_receipt_id"])
    if receipt.get("receipt_id") != expected_receipt_id:
        raise ContractError("v14 parent receipt identity drifted")
    if receipt.get("sealed_test_accessed") is not False:
        raise ContractError("v14 parent must not have accessed sealed test")
    expected_status = str(
        config.get("parent_selection_status", "seed7_winner_admitted_v11")
    )
    if selection.get("status") != expected_status:
        raise ContractError("multi-seed parent did not authorize confirmation")
    if selection.get("winner_trial_id") != str(config["candidate_trial_id"]):
        raise ContractError("v14 parent winner identity drifted")
    expected_seeds = [
        int(cast(Any, value)) for value in cast(list[object], config["seeds"])
    ]
    if selection.get("confirmation_seeds_authorized") != expected_seeds:
        raise ContractError("v14 confirmation seeds were not authorized")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ContractError("v14 parent receipt outputs are missing")
    for relative, expected_hash in outputs.items():
        path = parent / str(relative)
        if not path.is_file() or _sha256(path) != str(expected_hash):
            raise ContractError("v14 parent output hash drifted")
    return {
        "path": str(parent),
        "receipt_id": expected_receipt_id,
        "selection_status": str(selection["status"]),
    }


def _load_fixed_lstm_evidence(
    config: dict[str, object],
    observed_hashes: dict[str, str],
    *,
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    artifact = _resolve_project_path(config["lstm_evidence_artifact"])
    receipt_path = artifact / "receipt.json"
    measurements_path = artifact / "lstm-measurements.parquet"
    environment_path = artifact / "lstm-environment.json"
    source_config_path = artifact / "config.resolved.json"
    required_paths = (
        receipt_path,
        measurements_path,
        environment_path,
        source_config_path,
    )
    if not all(path.is_file() for path in required_paths):
        raise ContractError("v21 fixed LSTM evidence is incomplete")
    receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
    environment_value = json.loads(environment_path.read_text(encoding="utf-8"))
    source_config_value = json.loads(
        source_config_path.read_text(encoding="utf-8")
    )
    if not all(
        isinstance(value, dict)
        for value in (receipt_value, environment_value, source_config_value)
    ):
        raise ContractError("v21 fixed LSTM evidence must contain objects")
    receipt = cast(dict[str, object], receipt_value)
    environment = cast(dict[str, object], environment_value)
    source_config = cast(dict[str, object], source_config_value)
    expected_receipt_id = str(config["lstm_evidence_receipt_id"])
    if receipt.get("receipt_id") != expected_receipt_id:
        raise ContractError("v21 fixed LSTM receipt identity drifted")
    if receipt.get("sealed_test_accessed") is not False:
        raise ContractError("v21 fixed LSTM evidence accessed sealed data")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ContractError("v21 fixed LSTM receipt outputs are missing")
    for relative, expected_hash in outputs.items():
        path = artifact / str(relative)
        if not path.is_file() or _sha256(path) != str(expected_hash):
            raise ContractError("v21 fixed LSTM output hash drifted")
    if _sha256(measurements_path) != str(config["lstm_measurements_sha256"]):
        raise ContractError("v21 fixed LSTM measurements identity drifted")
    if _sha256(environment_path) != str(config["lstm_environment_sha256"]):
        raise ContractError("v21 fixed LSTM environment identity drifted")
    source_artifacts = receipt.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise ContractError("v21 fixed LSTM source identities are missing")
    for name in ("features", "window_index", "labels", "split_manifest"):
        source = source_artifacts.get(name)
        if not isinstance(source, dict) or source.get("sha256") != observed_hashes[name]:
            raise ContractError("v21 fixed LSTM source SHA-256 drifted")
    expected_benchmark = cast(dict[str, object], config["lstm_benchmark"])
    if (
        source_config.get("seeds") != list(seeds)
        or source_config.get("folds") != [0, 1, 2, 3, 4]
        or source_config.get("torch_threads") != config["torch_threads"]
        or source_config.get("num_workers") != config["num_workers"]
        or source_config.get("precision") != "float32"
        or source_config.get("lstm_benchmark") != expected_benchmark
    ):
        raise ContractError("v21 fixed LSTM protocol drifted")
    if (
        environment.get("base_seeds") != list(seeds)
        or environment.get("fold_count") != 5
        or environment.get("models") != ["lstm"]
        or environment.get("device") != "cpu"
        or environment.get("torch_threads") != config["torch_threads"]
        or environment.get("data_workers") != config["num_workers"]
        or environment.get("learning_rate") != expected_benchmark["learning_rate"]
        or environment.get("torch") != torch.__version__
        or environment.get("platform") != platform.platform()
    ):
        raise ContractError("v21 fixed LSTM environment drifted")
    measurements = pd.read_parquet(measurements_path)
    required_columns = {
        "model",
        "fold",
        "base_seed",
        "parameter_count",
        "batch_size",
        "epochs",
        "precision",
        "best_validation_rankic",
        "samples_per_second",
        "model_step_samples_per_second",
    }
    if missing := sorted(required_columns.difference(measurements.columns)):
        raise ContractError(
            "v21 fixed LSTM measurements missing columns: " + ", ".join(missing)
        )
    expected_units = {(seed, fold) for seed in seeds for fold in range(5)}
    observed_units = {
        (int(cast(Any, row.base_seed)), int(cast(Any, row.fold)))
        for row in measurements.itertuples(index=False)
    }
    if (
        len(measurements) != 10
        or observed_units != expected_units
        or measurements.duplicated(["base_seed", "fold"]).any()
        or set(measurements["model"].astype(str)) != {"lstm"}
        or set(measurements["parameter_count"].astype(int)) != {6124}
        or set(measurements["batch_size"].astype(int))
        != {int(cast(Any, expected_benchmark["batch_size"]))}
        or set(measurements["epochs"].astype(int))
        != {int(cast(Any, expected_benchmark["epochs"]))}
        or set(measurements["precision"].astype(str)) != {"float32"}
    ):
        raise ContractError("v21 fixed LSTM measurement coverage drifted")
    numeric = measurements[
        [
            "best_validation_rankic",
            "samples_per_second",
            "model_step_samples_per_second",
        ]
    ].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all() or bool((numeric[:, 1:] <= 0).any()):
        raise ContractError("v21 fixed LSTM measurements are invalid")
    return (
        measurements,
        environment,
        {
            "path": str(artifact),
            "receipt_id": expected_receipt_id,
            "measurements_sha256": _sha256(measurements_path),
            "environment_sha256": _sha256(environment_path),
        },
    )


def _multiscale_diagnostics(
    features: np.ndarray,
    split_manifest: pd.DataFrame,
    candidate: TCNTuningTrial,
    best_states: dict[str, dict[str, torch.Tensor]],
    *,
    seeds: tuple[int, ...],
    batch_size: int,
) -> pd.DataFrame:
    protocols = build_fold_protocols(features, split_manifest)
    dummy_targets = np.zeros((len(features), 4), dtype="float32")
    dummy_masks = np.ones((len(features), 4), dtype="bool")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for protocol in protocols:
            model = build_tcn_trial_model(
                candidate,
                feature_count=int(features.shape[1]),
                input_steps=int(features.shape[2]),
            )
            if not isinstance(model, DynamicHorizonSkipTCN):
                raise ContractError(
                    "v21 diagnostics require the dynamic horizon-skip TCN"
                )
            checkpoint_key = (
                f"seed-{seed}-{candidate.trial_id}-fold-{protocol.fold}"
            )
            try:
                model.load_state_dict(best_states[checkpoint_key])
            except KeyError as exc:
                raise ContractError("v21 dynamic checkpoint is missing") from exc
            dataset = LazyWindowDataset(
                features,
                protocol.validation_positions,
                dummy_targets,
                dummy_masks,
                protocol.feature_mean,
                protocol.feature_std,
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            weight_batches: list[np.ndarray] = []
            model.eval()
            observed = 0
            with torch.no_grad():
                for batch_features, _, _, _ in loader:
                    weights = model.dynamic_skip_weights(
                        model.encode_blocks(batch_features)
                    )
                    weight_batches.append(weights.cpu().numpy())
                    observed += len(batch_features)
                    if observed >= 512:
                        break
            if not weight_batches:
                raise ContractError("v21 diagnostics found no validation samples")
            observed_weights = np.concatenate(weight_batches, axis=0)[:512]
            rows.append(
                {
                    "trial_id": candidate.trial_id,
                    "seed": seed,
                    "fold": protocol.fold,
                    "sample_count": int(len(observed_weights)),
                    "block_weight_variation": float(
                        np.std(observed_weights, axis=0).max()
                    ),
                    "simplex_error_max": float(
                        np.abs(observed_weights.sum(axis=2) - 1.0).max()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "fold"], ignore_index=True)


def _three_seed_summaries(
    seed7_leaderboard: pd.DataFrame,
    confirmation_leaderboard: pd.DataFrame,
    *,
    control_trial_id: str,
    candidate_trial_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [seed7_leaderboard, confirmation_leaderboard], ignore_index=True
    )
    if set(combined["seed"].astype(int)) != {7, 17, 27}:
        raise ContractError("v21 three-seed descriptive coverage drifted")
    indexed = combined.set_index(["seed", "fold", "trial_id"])[
        "best_mean_daily_rankic"
    ].unstack("trial_id")
    indexed["rankic_delta"] = (
        indexed[candidate_trial_id] - indexed[control_trial_id]
    )
    seed_summary = (
        indexed.reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            candidate_mean_rankic=(candidate_trial_id, "mean"),
            control_mean_rankic=(control_trial_id, "mean"),
            mean_rankic_delta=("rankic_delta", "mean"),
            nondegrading_folds=(
                "rankic_delta", lambda values: int((values >= 0).sum())
            ),
        )
        .sort_values("seed", ignore_index=True)
    )
    horizon_rows: list[dict[str, float | int]] = []
    for horizon in (1, 2, 3, 5):
        column = f"rankic_{horizon}d"
        means = combined.groupby("trial_id", observed=True)[column].mean()
        horizon_rows.append(
            {
                "horizon": horizon,
                "control_rankic": float(means[control_trial_id]),
                "candidate_rankic": float(means[candidate_trial_id]),
                "rankic_delta": float(
                    means[candidate_trial_id] - means[control_trial_id]
                ),
            }
        )
    return seed_summary, pd.DataFrame(horizon_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run immutable TCN confirmation seeds 17 and 27"
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
            raise ContractError("v14 confirmation refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v14 confirmation config must contain an object")
        config = cast(dict[str, object], config_value)
        protocol_version = str(config.get("protocol_version", "v14"))
        if protocol_version not in {"v14", "v21"}:
            raise ContractError("multi-seed confirmation protocol is unsupported")
        if _contains_secret_key(config):
            raise ContractError("v14 confirmation config contains a secret-like key")
        if config.get("precision") != "float32":
            raise ContractError("v14 confirmation precision must remain float32")
        seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["seeds"])
        )
        if seeds != (17, 27):
            raise ContractError("v14 confirmation seeds must be exactly 17 and 27")
        parent = _validate_parent(config)

        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "labels": run_dir / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
        }
        decision: Any
        if protocol_version == "v21":
            source_paths.update(
                {
                    "universe": run_dir / "universe.parquet",
                    "input_manifest": run_dir / "input-manifest.json",
                }
            )
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError(f"v14 confirmation sources missing: {', '.join(missing)}")
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v14 confirmation source SHA-256 identity drifted")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v14 confirmation rejects sealed split rows")
        allowed_stages = {"train", "validation", "purged"}
        observed_stages = {
            str(value) for value in raw_split["stage"].astype(str).tolist()
        }
        if unknown := sorted(observed_stages - allowed_stages):
            raise ContractError(
                "v14 confirmation split contains forbidden stages: "
                + ", ".join(unknown)
            )
        folds = {
            int(cast(Any, value))
            for value in cast(list[object], config["folds"])
        }
        if folds != set(range(5)):
            raise ContractError("v14 confirmation folds must be exactly 0 through 4")
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(folds)
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        control_trial_id = str(config["control_trial_id"])
        candidate_trial_id = str(config["candidate_trial_id"])
        if {trial.trial_id for trial in trials} != {
            control_trial_id,
            candidate_trial_id,
        } or len(trials) != 2:
            raise ContractError("v14 confirmation trial identities drifted")
        by_trial = {trial.trial_id: trial for trial in trials}
        control = by_trial[control_trial_id]
        candidate = by_trial[candidate_trial_id]
        if protocol_version == "v21" and (
            control.model_kind != "horizon_skip"
            or candidate.model_kind != "dynamic_horizon_skip"
            or candidate.dynamic_skip_hidden != 4
            or candidate.dynamic_skip_scale != 1.0
        ):
            raise ContractError("v21 dynamic multiscale mechanism drifted")
        protocol_identities = {
            "data": observed_hashes["features"],
            "fold_manifest": observed_hashes["split_manifest"],
            "evaluation": observed_hashes["labels"],
        }
        tuning_parts = []
        best_states: dict[str, dict[str, torch.Tensor]] = {}
        for seed in seeds:
            tuning = run_tcn_validation_sweep(
                features,
                window_index,
                labels,
                split_manifest,
                trials=trials,
                seed=seed,
                max_epochs=int(cast(Any, config["max_epochs"])),
                patience=int(cast(Any, config["patience"])),
                min_delta=float(cast(Any, config["min_delta"])),
                torch_threads=int(cast(Any, config["torch_threads"])),
                protocol_identities=protocol_identities,
            )
            tuning_parts.append(tuning)
            for key, state in tuning.best_states.items():
                best_states[f"seed-{seed}-{key}"] = state
        epoch_history = pd.concat(
            [part.epoch_history for part in tuning_parts], ignore_index=True
        )
        leaderboard = pd.concat(
            [part.leaderboard for part in tuning_parts], ignore_index=True
        )
        attention = (
            _multiscale_diagnostics(
                features,
                split_manifest,
                candidate,
                best_states,
                seeds=seeds,
                batch_size=candidate.batch_size,
            )
            if protocol_version == "v21"
            else pd.DataFrame()
        )

        benchmark_config = cast(dict[str, object], config["lstm_benchmark"])
        lstm_evidence: dict[str, object] | None = None
        if protocol_version == "v21":
            lstm_measurements, lstm_environment, lstm_evidence = (
                _load_fixed_lstm_evidence(
                    config,
                    observed_hashes,
                    seeds=seeds,
                )
            )
        else:
            reference_trial = trials[0]
            lstm = benchmark_sequence_models(
                features,
                window_index,
                labels,
                split_manifest,
                seed=seeds[0],
                seeds=seeds,
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
            lstm_measurements = lstm.measurements
            lstm_environment = lstm.environment
        candidate_rows = leaderboard.loc[
            leaderboard["trial_id"].eq(candidate_trial_id)
        ].copy()
        comparison: dict[str, float | int] = build_tcn_lstm_comparison(
            candidate_rows, lstm_measurements
        )
        gates = cast(dict[str, object], config["gates"])
        if protocol_version == "v21":
            decision = evaluate_dynamic_multiscale_multiseed(
                leaderboard,
                attention,
                comparison,
                control_trial_id=control_trial_id,
                candidate_trial_id=candidate_trial_id,
                expected_seeds=seeds,
                min_mean_rankic=float(cast(Any, gates["min_mean_rankic"])),
                min_positive_units=int(cast(Any, gates["min_positive_units"])),
                min_mean_rankic_delta=float(
                    cast(Any, gates["min_mean_rankic_delta"])
                ),
                min_nondegrading_folds_per_seed=int(
                    cast(Any, gates["min_nondegrading_folds_per_seed"])
                ),
                min_horizon_delta_1d=float(
                    cast(Any, gates["min_horizon_delta_1d"])
                ),
                min_horizon_delta_2d=float(
                    cast(Any, gates["min_horizon_delta_2d"])
                ),
                min_horizon_delta_3d=float(
                    cast(Any, gates["min_horizon_delta_3d"])
                ),
                min_horizon_delta_5d=float(
                    cast(Any, gates["min_horizon_delta_5d"])
                ),
                min_median_samples_per_second=float(
                    cast(Any, gates["min_median_samples_per_second"])
                ),
                min_dynamic_skip_output_weight_l2=float(
                    cast(Any, gates["min_dynamic_skip_output_weight_l2"])
                ),
                min_block_weight_variation=float(
                    cast(Any, gates["min_block_weight_variation"])
                ),
                max_simplex_error=float(cast(Any, gates["max_simplex_error"])),
                control_parameter_count=int(
                    cast(Any, gates["control_parameter_count"])
                ),
                candidate_parameter_count=int(
                    cast(Any, gates["candidate_parameter_count"])
                ),
                dynamic_parameter_count=int(
                    cast(Any, gates["dynamic_parameter_count"])
                ),
                min_model_step_speed_ratio=float(
                    cast(Any, gates["min_model_step_speed_ratio"])
                ),
                min_end_to_end_speed_ratio=float(
                    cast(Any, gates["min_end_to_end_speed_ratio"])
                ),
            )
        else:
            decision = evaluate_signed_multiseed_confirmation(
                leaderboard,
                comparison,
                control_trial_id=control_trial_id,
                candidate_trial_id=candidate_trial_id,
                expected_seeds=seeds,
                min_mean_rankic=float(cast(Any, gates["min_mean_rankic"])),
                min_positive_units=int(cast(Any, gates["min_positive_units"])),
                min_nondegrading_folds_per_seed=int(
                    cast(Any, gates["min_nondegrading_folds_per_seed"])
                ),
                min_horizon_delta_3d=float(
                    cast(Any, gates["min_horizon_delta_3d"])
                ),
                min_horizon_delta_5d=float(
                    cast(Any, gates["min_horizon_delta_5d"])
                ),
                min_median_samples_per_second=float(
                    cast(Any, gates["min_median_samples_per_second"])
                ),
                min_model_step_speed_ratio=float(
                    cast(Any, gates["min_model_step_speed_ratio"])
                ),
                min_end_to_end_speed_ratio=float(
                    cast(Any, gates["min_end_to_end_speed_ratio"])
                ),
            )

        three_seed_summary = pd.DataFrame()
        three_seed_horizons = pd.DataFrame()
        if protocol_version == "v21":
            parent_path = Path(str(parent["path"]))
            seed7_leaderboard = pd.read_parquet(
                parent_path / "tcn-leaderboard.parquet"
            )
            three_seed_summary, three_seed_horizons = _three_seed_summaries(
                seed7_leaderboard,
                leaderboard,
                control_trial_id=control_trial_id,
                candidate_trial_id=candidate_trial_id,
            )

        temporary.mkdir(parents=True)
        epoch_history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        decision.seed_summary.to_parquet(temporary / "seed-summary.parquet", index=False)
        decision.horizon_summary.to_parquet(
            temporary / "horizon-summary.parquet", index=False
        )
        if protocol_version == "v21":
            attention.to_parquet(
                temporary / "attention-diagnostics.parquet", index=False
            )
            three_seed_summary.to_parquet(
                temporary / "three-seed-summary.parquet", index=False
            )
            three_seed_horizons.to_parquet(
                temporary / "three-seed-horizon-summary.parquet", index=False
            )
        lstm_measurements.to_parquet(
            temporary / "lstm-measurements.parquet", index=False
        )
        _write_json(temporary / "lstm-environment.json", lstm_environment)
        _write_json(temporary / "comparison.json", comparison)
        selection = {
            "status": decision.status,
            "effect_passed": decision.effect_passed,
            "speed_passed": decision.speed_passed,
            "candidate_trial_id": candidate_trial_id,
            "control_trial_id": control_trial_id,
            "seeds": list(seeds),
            "aggregate": decision.aggregate,
            "sealed_test_authorized": False,
        }
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for checkpoint_key, state in best_states.items():
            torch.save(state, checkpoint_dir / f"{checkpoint_key}.pt")
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": (
                "tcn-dynamic-multiscale-multiseed-v21/v1"
                if protocol_version == "v21"
                else "tcn-signed-multiseed-confirmation-v14/v1"
            ),
            "run_id": str(config["run_id"]),
            "parent": parent,
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
            "lstm_evidence": lstm_evidence,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "result": decision.status,
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
