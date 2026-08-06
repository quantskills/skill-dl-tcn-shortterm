"""Run the frozen v42 seeds 17/27 consensus-student confirmation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.neural import HORIZONS, _label_matrices  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials  # noqa: E402
from skill_dl_tcn_shortterm.task_aligned_evaluation import (  # noqa: E402
    bootstrap_task_aligned_differences,
    compare_task_aligned_models,
    evaluate_task_aligned_predictions,
    summarize_task_aligned_metrics,
    validate_prediction_contract,
)
from skill_dl_tcn_shortterm.training_data import build_fold_protocols  # noqa: E402
from skill_dl_tcn_shortterm.tuning import run_tcn_validation_sweep  # noqa: E402
from skill_dl_tcn_shortterm.v42_validation import (  # noqa: E402
    decide_consensus_student_multiseed_gate,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_relative_feature_validation import _collect_tcn_predictions  # noqa: E402
from run_tcn_task_aligned_evaluation import _label_lookup  # noqa: E402
from run_tcn_v41_ema_seed7 import (  # noqa: E402
    _contains_secret_key,
    _geometric_mean,
    _sha256,
    _write_json,
)


EXPECTED_GATES = {
    "min_mean_rankic_delta": 0.002,
    "min_positive_seed_fold_units": 9,
    "min_per_seed_mean_rankic_delta": -0.001,
    "min_rankic_ci_low": -0.001,
    "min_broad_metric_count": 4,
    "min_top_return_delta": -0.0001,
    "min_top_precision_delta": -0.002,
    "min_ndcg_delta": -0.001,
    "min_quantile_monotonicity_delta": -0.002,
    "min_positive_horizons": 3,
    "min_worst_horizon_rankic_delta": -0.002,
    "min_model_step_retention": 0.95,
    "min_complete_cycle_retention": 0.90,
    "min_implied_tcn_lstm_model_step_ratio": 3.0,
    "inference_forward_passes": 1,
}


def _verify_teacher_manifest(receipt: dict[str, object]) -> list[dict[str, object]]:
    raw_manifest = receipt.get("teacher_checkpoints")
    if not isinstance(raw_manifest, list):
        raise ContractError("v42 phase-A teacher checkpoint manifest is missing")
    observed: set[tuple[int, int]] = set()
    manifest: list[dict[str, object]] = []
    for raw in raw_manifest:
        if not isinstance(raw, dict):
            raise ContractError("v42 phase-A teacher checkpoint entry is invalid")
        seed = int(cast(Any, raw["seed"]))
        fold = int(cast(Any, raw["fold"]))
        path = Path(str(raw["path"])).resolve()
        expected_hash = str(raw["sha256"])
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ContractError(f"v42 frozen teacher checkpoint drifted: {seed}/{fold}")
        observed.add((seed, fold))
        manifest.append(
            {"seed": seed, "fold": fold, "path": str(path), "sha256": expected_hash}
        )
    expected = {(seed, fold) for seed in (7, 17, 27) for fold in range(5)}
    if observed != expected or len(manifest) != 15:
        raise ContractError("v42 frozen teacher checkpoint coverage drifted")
    return manifest


def _load_frozen_overrides(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split: pd.DataFrame,
    target_path: Path,
    teacher_audit_path: Path,
) -> tuple[dict[int, np.ndarray], str, dict[str, object]]:
    targets = pd.read_parquet(target_path)
    required = {
        "fold",
        "sample_position",
        "signal_date",
        "horizon",
        "teacher_consensus_rank",
        "distilled_target",
        "valid",
    }
    if missing := sorted(required.difference(targets.columns)):
        raise ContractError("v42 frozen teacher targets missing: " + ", ".join(missing))
    target_identity = hashlib.sha256(
        np.asarray(pd.util.hash_pandas_object(targets, index=False).values).tobytes()
    ).hexdigest()
    teacher_audit = json.loads(teacher_audit_path.read_text(encoding="utf-8"))
    if not isinstance(teacher_audit, dict) or (
        teacher_audit.get("teacher_target_identity") != target_identity
        or teacher_audit.get("teacher_seeds") != [7, 17, 27]
        or teacher_audit.get("teacher_weight") != 0.25
        or teacher_audit.get("validation_positions_used_for_teacher_training") != 0
        or teacher_audit.get("sealed_test_accessed") is not False
    ):
        raise ContractError("v42 frozen teacher target audit drifted")
    if targets.duplicated(["fold", "sample_position", "horizon"]).any():
        raise ContractError("v42 frozen teacher targets contain duplicate keys")

    true_targets, masks = _label_matrices(window_index, labels)
    dates = window_index.set_index("sample_position")["signal_date"].astype(str)
    overrides: dict[int, np.ndarray] = {}
    for protocol in build_fold_protocols(features, split):
        fold_rows = targets.loc[targets["fold"].astype(int).eq(protocol.fold)].copy()
        expected_positions = set(int(value) for value in protocol.train_positions)
        observed_positions = set(fold_rows["sample_position"].astype(int))
        if (
            observed_positions != expected_positions
            or len(fold_rows) != len(expected_positions) * len(HORIZONS)
            or set(fold_rows["horizon"].astype(int)) != set(HORIZONS)
        ):
            raise ContractError(f"v42 frozen target coverage drifted in fold {protocol.fold}")
        positions = fold_rows["sample_position"].to_numpy(dtype="int64")
        horizons = fold_rows["horizon"].to_numpy(dtype="int64")
        horizon_offsets = np.asarray(
            [{horizon: offset for offset, horizon in enumerate(HORIZONS)}[value] for value in horizons],
            dtype="int64",
        )
        expected_dates = dates.loc[positions].to_numpy(dtype=str)
        if not np.array_equal(expected_dates, fold_rows["signal_date"].astype(str).to_numpy()):
            raise ContractError(f"v42 frozen target dates drifted in fold {protocol.fold}")
        expected_valid = masks[positions, horizon_offsets]
        if not np.array_equal(expected_valid, fold_rows["valid"].astype(bool).to_numpy()):
            raise ContractError(f"v42 frozen target masks drifted in fold {protocol.fold}")
        teacher_values = fold_rows["teacher_consensus_rank"].to_numpy(dtype="float64")
        distilled_values = fold_rows["distilled_target"].to_numpy(dtype="float64")
        if (
            not np.isfinite(teacher_values).all()
            or not np.isfinite(distilled_values).all()
            or bool((np.abs(teacher_values) > 1.0 + 1e-7).any())
            or bool((np.abs(distilled_values) > 1.0 + 1e-7).any())
        ):
            raise ContractError("v42 frozen target values escaped [-1, 1]")
        override = true_targets.copy()
        override[positions, horizon_offsets] = distilled_values.astype("float32")
        train_mask = np.zeros(len(features), dtype="bool")
        train_mask[np.asarray(protocol.train_positions, dtype="int64")] = True
        if not np.array_equal(override[~train_mask], true_targets[~train_mask]):
            raise ContractError("v42 frozen target override escaped train positions")
        overrides[protocol.fold] = override
    if set(overrides) != set(range(5)):
        raise ContractError("v42 frozen target fold coverage is incomplete")
    return overrides, target_identity, teacher_audit


def _multiseed_deltas(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = (
        metrics.groupby(
            ["model", "seed", "fold", "horizon"],
            as_index=False,
            observed=True,
        )["rankic"]
        .mean()
        .pivot(index=["seed", "fold", "horizon"], columns="model", values="rankic")
    )
    if (
        set(means.columns) != {"control_tcn", "consensus_student_tcn"}
        or means.isna().any().any()
    ):
        raise ContractError("v42 paired multi-seed metric coverage drifted")
    paired = means.reset_index()[["seed", "fold", "horizon"]].copy()
    paired["rankic_delta"] = (
        means["consensus_student_tcn"] - means["control_tcn"]
    ).to_numpy()
    seed_fold = paired.groupby(
        ["seed", "fold"], as_index=False, observed=True
    ).agg(rankic_delta=("rankic_delta", "mean"))
    per_seed = seed_fold.groupby("seed", as_index=False, observed=True).agg(
        rankic_delta=("rankic_delta", "mean")
    )
    horizon = paired.groupby("horizon", as_index=False, observed=True).agg(
        rankic_delta=("rankic_delta", "mean")
    )
    return seed_fold, per_seed, horizon


def _timing_evidence(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    v40_speed_ratio: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    columns = [
        "seed",
        "fold",
        "model_step_samples_per_second",
        "complete_cycle_seconds",
    ]
    left = control[columns].rename(
        columns={
            "model_step_samples_per_second": "control_model_step_samples_per_second",
            "complete_cycle_seconds": "control_complete_cycle_seconds",
        }
    )
    right = candidate[columns].rename(
        columns={
            "model_step_samples_per_second": "candidate_model_step_samples_per_second",
            "complete_cycle_seconds": "candidate_complete_cycle_seconds",
        }
    )
    units = left.merge(right, on=["seed", "fold"], validate="one_to_one")
    if len(units) != 15:
        raise ContractError("v42 timing coverage must be exactly 15 seed/fold units")
    units["model_step_retention"] = (
        units["candidate_model_step_samples_per_second"]
        / units["control_model_step_samples_per_second"]
    )
    units["complete_cycle_retention"] = (
        units["control_complete_cycle_seconds"]
        / units["candidate_complete_cycle_seconds"]
    )
    model_retention = _geometric_mean(
        units["model_step_retention"].to_numpy(dtype="float64")
    )
    complete_retention = _geometric_mean(
        units["complete_cycle_retention"].to_numpy(dtype="float64")
    )
    return units, {
        "model_step_retention_geomean": model_retention,
        "complete_cycle_retention_geomean": complete_retention,
        "implied_tcn_lstm_model_step_ratio": v40_speed_ratio * model_retention,
        "inference_forward_passes": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--relative-feature-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--v40-run-dir", required=True, type=Path)
    parser.add_argument("--phase-a-run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v42 multi-seed run refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("protocol_version") != "v42-phase-b":
            raise ContractError("v42 phase-B config identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v42 phase-B config contains a secret-like key")
        if config.get("gates") != EXPECTED_GATES:
            raise ContractError("v42 phase-B gates drifted")
        if (
            tuple(cast(list[object], config["student_seeds"])) != (17, 27)
            or tuple(cast(list[object], config["teacher_seeds"])) != (7, 17, 27)
            or tuple(cast(list[object], config["folds"])) != (0, 1, 2, 3, 4)
            or float(cast(Any, config["teacher_weight"])) != 0.25
            or config.get("precision") != "float32"
            or int(cast(Any, config["torch_threads"])) != 8
            or int(cast(Any, config["num_workers"])) != 0
        ):
            raise ContractError("v42 phase-B frozen execution contract drifted")

        base = arguments.base_run_dir.resolve()
        relative = arguments.relative_feature_dir.resolve()
        v40 = arguments.v40_run_dir.resolve()
        phase_a = arguments.phase_a_run_dir.resolve()
        source_paths = {
            "relative_features": relative / "feature-windows.npy",
            "relative_window_index": relative / "window-index.parquet",
            "relative_manifest": relative / "manifest.json",
            "relative_receipt": relative / "receipt.json",
            "labels": base / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "v40_predictions": v40 / "predictions.parquet",
            "v40_leaderboard": v40 / "tcn-leaderboard.parquet",
            "v40_epoch_history": v40 / "tcn-epoch-history.parquet",
            "v40_timing_summary": v40 / "timing-summary.json",
            "v40_receipt": v40 / "receipt.json",
            "phase_a_model_gate": phase_a / "model-gate.json",
            "phase_a_receipt": phase_a / "receipt.json",
            "phase_a_predictions": phase_a / "predictions.parquet",
            "phase_a_leaderboard": phase_a / "leaderboard.parquet",
            "phase_a_epoch_history": phase_a / "epoch-history.parquet",
            "phase_a_teacher_targets": phase_a / "teacher-consensus-targets.parquet",
            "phase_a_teacher_audit": phase_a / "teacher-audit.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v42 phase-B sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v42 phase-B source SHA-256 identity drifted")

        phase_a_gate = json.loads(
            source_paths["phase_a_model_gate"].read_text(encoding="utf-8")
        )
        if not isinstance(phase_a_gate, dict) or not (
            phase_a_gate.get("status") == "consensus_student_seed7_holistic_admitted_v42"
            and phase_a_gate.get("admitted") is True
            and phase_a_gate.get("phase_b_authorized") is True
            and phase_a_gate.get("sealed_test_accessed") is False
        ):
            raise ContractError("v42 phase B is not authorized by phase A")
        phase_a_receipt = json.loads(
            source_paths["phase_a_receipt"].read_text(encoding="utf-8")
        )
        v40_receipt = json.loads(source_paths["v40_receipt"].read_text(encoding="utf-8"))
        if not isinstance(phase_a_receipt, dict) or not isinstance(v40_receipt, dict):
            raise ContractError("v42 parent receipts are invalid")
        if (
            phase_a_receipt.get("sealed_test_accessed") is not False
            or v40_receipt.get("sealed_test_accessed") is not False
        ):
            raise ContractError("v42 phase B rejects sealed parent evidence")
        teacher_manifest = _verify_teacher_manifest(phase_a_receipt)

        features = np.load(
            source_paths["relative_features"], mmap_mode="r", allow_pickle=False
        )
        if features.ndim != 3 or features.shape[1:] != (10, 480):
            raise ContractError("v42 phase-B relative10 tensor shape drifted")
        window_index = pd.read_parquet(source_paths["relative_window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v42 phase B rejects sealed split rows")
        stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(stages - {"train", "validation", "purged"}):
            raise ContractError("v42 phase-B split has forbidden stages: " + ", ".join(unknown))
        split = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].astype(str).isin(["train", "validation"])
        ].copy()
        overrides, target_identity, teacher_audit = _load_frozen_overrides(
            features,
            window_index,
            labels,
            split,
            source_paths["phase_a_teacher_targets"],
            source_paths["phase_a_teacher_audit"],
        )

        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 2:
            raise ContractError("v42 phase B requires frozen control and student trials")
        by_id = {trial.trial_id: trial for trial in trials}
        control_id = str(config["control_trial_id"])
        candidate_id = str(config["candidate_trial_id"])
        if set(by_id) != {control_id, candidate_id}:
            raise ContractError("v42 phase-B trial identities drifted")
        control = by_id[control_id]
        candidate = by_id[candidate_id]
        control_contract = asdict(control)
        candidate_contract = asdict(candidate)
        control_contract.pop("trial_id")
        candidate_contract.pop("trial_id")
        if control_contract != candidate_contract or control.ema_decay is not None:
            raise ContractError("v42 phase-B trials differ by more than targets")

        temporary.mkdir(parents=True)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        contracts = cast(dict[str, str], config["contracts"])
        lookup = _label_lookup(labels)
        candidate_prediction_frames: list[pd.DataFrame] = []
        phase_a_predictions = pd.read_parquet(source_paths["phase_a_predictions"])
        seed7_candidate = phase_a_predictions.loc[
            phase_a_predictions["model"].astype(str).eq("consensus_student_tcn")
        ].copy()
        if set(seed7_candidate["seed"].astype(int)) != {7}:
            raise ContractError("v42 phase-A candidate prediction coverage drifted")
        candidate_prediction_frames.append(seed7_candidate)
        candidate_leaderboards = [
            pd.read_parquet(source_paths["phase_a_leaderboard"]).loc[
                lambda frame: frame["trial_id"].astype(str).eq(candidate_id)
            ].copy()
        ]
        candidate_histories = [
            pd.read_parquet(source_paths["phase_a_epoch_history"]).loc[
                lambda frame: frame["trial_id"].astype(str).eq(candidate_id)
            ].copy()
        ]
        phase_a_outputs = cast(dict[str, str], phase_a_receipt.get("outputs", {}))
        for fold in range(5):
            source = phase_a / "checkpoints" / f"{candidate_id}-fold-{fold}.pt"
            relative_name = str(Path("checkpoints") / f"{candidate_id}-fold-{fold}.pt")
            if not source.is_file() or _sha256(source) != phase_a_outputs.get(relative_name):
                raise ContractError(f"v42 phase-A student checkpoint drifted: fold {fold}")
            shutil.copy2(source, checkpoint_dir / f"{candidate_id}-seed-7-fold-{fold}.pt")

        for seed in (17, 27):
            tuning = run_tcn_validation_sweep(
                features,
                window_index,
                labels,
                split,
                trials=(candidate,),
                seed=seed,
                max_epochs=int(cast(Any, config["max_epochs"])),
                patience=int(cast(Any, config["patience"])),
                min_delta=float(cast(Any, config["min_delta"])),
                checkpoint_min_delta=float(cast(Any, config["checkpoint_min_delta"])),
                torch_threads=int(cast(Any, config["torch_threads"])),
                protocol_identities={
                    "data": target_identity,
                    "fold_manifest": observed_hashes["split_manifest"],
                    "evaluation": observed_hashes["labels"],
                },
                capture_epoch_states=True,
                disable_early_stopping=True,
                training_target_overrides=overrides,
            )
            if set(tuning.leaderboard["training_target_override"]) != {True}:
                raise ContractError("v42 phase-B candidate target override was not active")
            candidate_leaderboards.append(tuning.leaderboard)
            candidate_histories.append(tuning.epoch_history)
            model_contracts = dict(contracts)
            model_contracts["tcn_training_contract_id"] = (
                "top50-relative10-consensus_student_tcn-v42"
            )
            candidate_prediction_frames.append(
                _collect_tcn_predictions(
                    features,
                    labels,
                    split,
                    candidate,
                    tuning.best_states,
                    seed=seed,
                    model_name="consensus_student_tcn",
                    lookup=lookup,
                    contracts=model_contracts,
                )
            )
            for state_key, state in tuning.best_states.items():
                torch.save(
                    state,
                    checkpoint_dir / f"{candidate_id}-seed-{seed}-{state_key}.pt",
                )

        control_predictions = pd.read_parquet(source_paths["v40_predictions"]).loc[
            lambda frame: frame["model"].astype(str).eq("relative_tcn")
        ].copy()
        control_predictions["model"] = "control_tcn"
        control_predictions["prediction_contract_id"] = contracts["prediction_contract_id"]
        control_predictions["target_contract_id"] = contracts["target_contract_id"]
        control_predictions["evaluation_contract_id"] = contracts["evaluation_contract_id"]
        control_predictions["training_contract_id"] = "top50-relative10-control_tcn-v42"
        candidate_predictions = pd.concat(candidate_prediction_frames, ignore_index=True)
        candidate_predictions["prediction_contract_id"] = contracts["prediction_contract_id"]
        candidate_predictions["target_contract_id"] = contracts["target_contract_id"]
        candidate_predictions["evaluation_contract_id"] = contracts["evaluation_contract_id"]
        candidate_predictions["training_contract_id"] = (
            "top50-relative10-consensus_student_tcn-v42"
        )
        predictions = pd.concat(
            [control_predictions, candidate_predictions], ignore_index=True
        )
        validate_prediction_contract(predictions, expected_models=2)
        if (
            set(predictions["seed"].astype(int)) != {7, 17, 27}
            or set(predictions["fold"].astype(int)) != set(range(5))
            or set(predictions["horizon"].astype(int)) != set(HORIZONS)
        ):
            raise ContractError("v42 phase-B prediction coverage drifted")

        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(cast(Any, config["top_fraction"]))
        )
        summary = summarize_task_aligned_metrics(metrics)
        comparison = compare_task_aligned_models(
            metrics,
            reference_model="control_tcn",
            candidate_model="consensus_student_tcn",
        )
        bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="control_tcn",
            candidate_model="consensus_student_tcn",
            metric_columns=(
                "rankic",
                "pearson_ic",
                "top_return",
                "top_precision",
                "ndcg_at_top",
                "quantile_monotonicity",
            ),
            seed=int(cast(Any, config["bootstrap_seed"])),
            draws=int(cast(Any, config["bootstrap_draws"])),
        )
        seed_fold_deltas, per_seed_deltas, horizon_deltas = _multiseed_deltas(metrics)

        control_leaderboard = pd.read_parquet(source_paths["v40_leaderboard"]).loc[
            lambda frame: frame["variant"].astype(str).eq("relative")
        ].copy()
        candidate_leaderboard = pd.concat(candidate_leaderboards, ignore_index=True)
        if (
            set(zip(control_leaderboard["seed"].astype(int), control_leaderboard["fold"].astype(int), strict=True))
            != {(seed, fold) for seed in (7, 17, 27) for fold in range(5)}
            or set(zip(candidate_leaderboard["seed"].astype(int), candidate_leaderboard["fold"].astype(int), strict=True))
            != {(seed, fold) for seed in (7, 17, 27) for fold in range(5)}
        ):
            raise ContractError("v42 phase-B leaderboard coverage drifted")
        v40_timing = json.loads(
            source_paths["v40_timing_summary"].read_text(encoding="utf-8")
        )
        timing_units, timing = _timing_evidence(
            control_leaderboard,
            candidate_leaderboard,
            v40_speed_ratio=float(v40_timing["model_step_speed_ratio_geomean"]),
        )
        decision = decide_consensus_student_multiseed_gate(
            comparison,
            bootstrap,
            seed_fold_deltas,
            horizon_deltas,
            model_step_retention=float(timing["model_step_retention_geomean"]),
            complete_cycle_retention=float(timing["complete_cycle_retention_geomean"]),
            implied_tcn_lstm_model_step_ratio=float(
                timing["implied_tcn_lstm_model_step_ratio"]
            ),
            inference_forward_passes=int(timing["inference_forward_passes"]),
        )

        control_history = pd.read_parquet(source_paths["v40_epoch_history"]).loc[
            lambda frame: frame["variant"].astype(str).eq("relative")
        ].copy()
        control_history["training_role"] = "control"
        candidate_history = pd.concat(candidate_histories, ignore_index=True)
        candidate_history["training_role"] = "candidate"
        history = pd.concat([control_history, candidate_history], ignore_index=True)
        control_leaderboard["training_role"] = "control"
        candidate_leaderboard["training_role"] = "candidate"
        leaderboard = pd.concat(
            [control_leaderboard, candidate_leaderboard], ignore_index=True
        )

        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap.parquet", index=False)
        seed_fold_deltas.to_parquet(temporary / "seed-fold-deltas.parquet", index=False)
        per_seed_deltas.to_parquet(temporary / "per-seed-deltas.parquet", index=False)
        horizon_deltas.to_parquet(temporary / "horizon-deltas.parquet", index=False)
        timing_units.to_parquet(temporary / "timing-units.parquet", index=False)
        leaderboard.to_parquet(temporary / "leaderboard.parquet", index=False)
        history.to_parquet(temporary / "epoch-history.parquet", index=False)
        _write_json(temporary / "comparison.json", comparison)
        _write_json(temporary / "timing.json", timing)
        _write_json(
            temporary / "teacher-target-reuse-audit.json",
            {
                **teacher_audit,
                "teacher_target_identity_recomputed": target_identity,
                "phase_a_target_sha256": observed_hashes["phase_a_teacher_targets"],
                "student_seeds": [7, 17, 27],
                "validation_positions_used_for_teacher_training": 0,
                "sealed_test_accessed": False,
            },
        )
        model_gate = {
            "status": decision.status,
            "admitted": decision.admitted,
            "blockers": list(decision.blockers),
            "evidence": decision.evidence,
            "sealed_test_accessed": False,
        }
        _write_json(temporary / "model-gate.json", model_gate)
        _write_json(temporary / "config.resolved.json", config)
        report = (
            "# TCN v42 consensus-student multi-seed confirmation\n\n"
            f"- status: `{decision.status}`\n"
            f"- mean RankIC delta: `{float(comparison['mean_rankic_delta']):+.6f}`\n"
            f"- broad improved metrics: `{decision.evidence['broad_metric_count']}/6`\n"
            f"- positive seed/fold units: `{decision.evidence['positive_seed_fold_units']}/15`\n"
            f"- minimum per-seed RankIC delta: `{decision.evidence['minimum_seed_mean_rankic_delta']:+.6f}`\n"
            f"- positive horizons: `{decision.evidence['positive_horizons']}/4`\n"
            f"- model-step retention: `{float(timing['model_step_retention_geomean']):.4f}`\n"
            f"- implied TCN/LSTM model-step ratio: `{float(timing['implied_tcn_lstm_model_step_ratio']):.4f}x`\n"
            "- inference model count: `1`\n"
            "- membership turnover: diagnostic only; not a model gate.\n"
            "- sealed_test_accessed: `false`\n"
            "- evidence ceiling: ordinary-validation single-TCN stability evidence; not alpha-ready.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")

        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, object] = {
            "schema_version": "tcn-v42-consensus-student-multiseed/v1",
            "run_id": str(config["run_id"]),
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "teacher_checkpoints": teacher_manifest,
            "teacher_target_identity": target_identity,
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
            },
            "model_gate": model_gate,
            "timing": timing,
            "outputs": outputs,
            "sealed_test_accessed": False,
            "sealed_test_authorized": False,
            "alpha_ready": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "run_status": decision.status,
            "admitted": decision.admitted,
            "receipt_id": receipt["receipt_id"],
            "output_dir": str(output_dir),
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
