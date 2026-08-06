"""Run the frozen v44 validation-independent trajectory-average probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from skill_dl_tcn_shortterm.consensus_distillation import (  # noqa: E402
    blend_training_targets,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.neural import _label_matrices  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials  # noqa: E402
from skill_dl_tcn_shortterm.task_aligned_evaluation import (  # noqa: E402
    bootstrap_task_aligned_differences,
    compare_task_aligned_models,
    evaluate_task_aligned_predictions,
    summarize_task_aligned_metrics,
    validate_prediction_contract,
)
from skill_dl_tcn_shortterm.training_data import build_fold_protocols  # noqa: E402
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    build_tcn_trial_model,
    run_tcn_validation_sweep,
)
from skill_dl_tcn_shortterm.v44_validation import (  # noqa: E402
    decide_trajectory_average_seed7_gate,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_relative_feature_validation import _collect_tcn_predictions  # noqa: E402
from run_tcn_task_aligned_evaluation import _label_lookup  # noqa: E402
from run_tcn_v41_ema_seed7 import (  # noqa: E402
    _contains_secret_key,
    _geometric_mean,
    _sha256,
    _trajectory_audit,
    _unit_deltas,
    _write_json,
)
from run_tcn_v43_listwise_consensus_seed7 import (  # noqa: E402
    _load_teacher_targets,
    _teacher_ensemble,
    _teacher_fidelity,
)


EXPECTED_GATES = {
    "control_min_mean_rankic_delta": 0.002,
    "control_min_positive_folds": 3,
    "control_min_rankic_ci_low": -0.002,
    "control_min_broad_metric_count": 4,
    "control_min_top_return_delta": -0.0001,
    "control_min_top_precision_delta": -0.002,
    "control_min_ndcg_delta": -0.001,
    "control_min_quantile_monotonicity_delta": -0.002,
    "control_min_positive_horizons": 3,
    "control_min_worst_horizon_rankic_delta": -0.003,
    "pointwise_min_broad_metric_count": 3,
    "pointwise_min_rankic_delta": -0.002,
    "pointwise_min_pearson_ic_delta": -0.002,
    "pointwise_min_top_return_delta": -0.0001,
    "pointwise_min_top_precision_delta": -0.001,
    "pointwise_min_ndcg_delta": -0.001,
    "pointwise_min_quantile_monotonicity_delta": -0.002,
    "min_teacher_fidelity_delta": -0.002,
    "max_validation_volatility_ratio": 0.90,
    "max_raw_state_drift": 0.0,
    "max_arithmetic_mean_error": 1e-6,
    "min_average_parameter_distance": 0.0,
    "required_average_updates": 7,
    "min_model_step_retention": 0.95,
    "min_complete_cycle_retention": 0.90,
    "min_implied_tcn_lstm_model_step_ratio": 3.0,
    "inference_forward_passes": 1,
}


def _reliability_diagnostic(
    predictions: pd.DataFrame, targets: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    required_predictions = {
        "seed",
        "fold",
        "sample_position",
        "signal_date",
        "horizon",
        "score",
    }
    required_targets = {
        "fold",
        "sample_position",
        "signal_date",
        "horizon",
        "teacher_consensus_rank",
        "distilled_target",
        "valid",
    }
    if missing := sorted(required_predictions.difference(predictions.columns)):
        raise ContractError("v44 reliability predictions missing: " + ", ".join(missing))
    if missing := sorted(required_targets.difference(targets.columns)):
        raise ContractError("v44 reliability targets missing: " + ", ".join(missing))
    if set(predictions["seed"].astype(int)) != {7, 17, 27}:
        raise ContractError("v44 reliability teacher seeds drifted")
    ranked = predictions.copy()
    group = ["fold", "seed", "signal_date", "horizon"]
    raw_ranks = ranked.groupby(group, observed=True, sort=False)["score"].rank(
        method="average"
    )
    sizes = ranked.groupby(group, observed=True, sort=False)["score"].transform(
        "size"
    )
    ranked["teacher_rank"] = np.where(
        sizes.eq(1), 0.0, 2.0 * (raw_ranks - 1.0) / (sizes - 1.0) - 1.0
    )
    keys = ["fold", "sample_position", "signal_date", "horizon"]
    wide = ranked.pivot(index=keys, columns="seed", values="teacher_rank")
    wide["teacher_rank_dispersion"] = wide.std(axis=1, ddof=0)
    audit = wide[["teacher_rank_dispersion"]].reset_index().merge(
        targets, on=keys, validate="one_to_one"
    )
    audit = audit.loc[audit["valid"].astype(bool)].copy()
    audit["true_rank"] = (
        audit["distilled_target"] - 0.25 * audit["teacher_consensus_rank"]
    ) / 0.75
    audit["teacher_abs_error"] = (
        audit["teacher_consensus_rank"] - audit["true_rank"]
    ).abs()
    audit["teacher_consensus_abs"] = audit["teacher_consensus_rank"].abs()
    rows: list[dict[str, object]] = []
    conditional_correlations: list[float] = []
    for (fold, horizon), unit in audit.groupby(
        ["fold", "horizon"], observed=True, sort=True
    ):
        raw_correlation = unit["teacher_rank_dispersion"].corr(
            unit["teacher_abs_error"], method="spearman"
        )
        magnitude_correlation = unit["teacher_rank_dispersion"].corr(
            unit["teacher_consensus_abs"], method="spearman"
        )
        bins = pd.qcut(
            unit["teacher_consensus_abs"], 10, labels=False, duplicates="drop"
        )
        within = []
        for _, stratum in unit.groupby(bins, observed=True):
            correlation = stratum["teacher_rank_dispersion"].corr(
                stratum["teacher_abs_error"], method="spearman"
            )
            if np.isfinite(correlation):
                within.append(float(correlation))
                conditional_correlations.append(float(correlation))
        if not np.isfinite(raw_correlation) or not np.isfinite(magnitude_correlation):
            raise ContractError("v44 reliability diagnostic is non-finite")
        rows.append(
            {
                "fold": int(cast(Any, fold)),
                "horizon": int(cast(Any, horizon)),
                "valid_cells": len(unit),
                "dispersion_error_spearman": float(raw_correlation),
                "dispersion_consensus_magnitude_spearman": float(
                    magnitude_correlation
                ),
                "conditional_dispersion_error_spearman": float(np.mean(within)),
                "conditional_strata": len(within),
            }
        )
    units = pd.DataFrame(rows)
    if len(units) != 20 or len(conditional_correlations) != 200:
        raise ContractError("v44 reliability fold/horizon coverage drifted")
    summary: dict[str, object] = {
        "raw_positive_units": int(
            (units["dispersion_error_spearman"] > 0.0).sum()
        ),
        "raw_mean_dispersion_error_spearman": float(
            units["dispersion_error_spearman"].mean()
        ),
        "mean_dispersion_consensus_magnitude_spearman": float(
            units["dispersion_consensus_magnitude_spearman"].mean()
        ),
        "conditional_positive_units": int(
            (np.asarray(conditional_correlations) > 0.0).sum()
        ),
        "conditional_unit_count": len(conditional_correlations),
        "conditional_mean_dispersion_error_spearman": float(
            np.mean(conditional_correlations)
        ),
        "teacher_reliability_weighting_authorized": False,
        "sealed_test_accessed": False,
    }
    return units, summary


def _average_audit(
    tuning: Any,
    candidate_trial: Any,
    *,
    feature_count: int,
    input_steps: int,
    folds: tuple[int, ...],
    start_epoch: int,
    end_epoch: int,
) -> tuple[pd.DataFrame, float, float]:
    model = build_tcn_trial_model(
        candidate_trial, feature_count=feature_count, input_steps=input_steps
    )
    parameter_names = set(dict(model.named_parameters()))
    rows: list[dict[str, object]] = []
    for fold in folds:
        final_key = f"{candidate_trial.trial_id}-fold-{fold}"
        if final_key not in tuning.best_states:
            raise ContractError("v44 final averaged state coverage is incomplete")
        final = tuning.best_states[final_key]
        sources = [
            tuning.epoch_states[
                f"{candidate_trial.trial_id}-fold-{fold}-epoch-{epoch}"
            ]
            for epoch in range(start_epoch, end_epoch + 1)
        ]
        maximum_error = 0.0
        maximum_distance = 0.0
        for name in parameter_names:
            expected = torch.stack(
                [state[name].detach().cpu().to(torch.float64) for state in sources]
            ).mean(dim=0)
            observed = final[name].detach().cpu().to(torch.float64)
            maximum_error = max(
                maximum_error, float(torch.max(torch.abs(observed - expected)))
            )
            maximum_distance = max(
                maximum_distance,
                float(
                    torch.max(
                        torch.abs(
                            observed
                            - sources[-1][name].detach().cpu().to(torch.float64)
                        )
                    )
                ),
            )
        rows.append(
            {
                "fold": fold,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "update_count": len(sources),
                "arithmetic_mean_max_error": maximum_error,
                "raw_final_parameter_distance": maximum_distance,
            }
        )
    audit = pd.DataFrame(rows)
    return (
        audit,
        float(audit["arithmetic_mean_max_error"].max()),
        float(audit["raw_final_parameter_distance"].max()),
    )


def _replay_max_abs_error(
    replay: pd.DataFrame, frozen_pointwise: pd.DataFrame
) -> float:
    keys = ["seed", "fold", "sample_id", "instrument_id", "signal_date", "horizon"]
    left = replay[[*keys, "score"]].rename(columns={"score": "replay_score"})
    right = frozen_pointwise[[*keys, "score"]].rename(
        columns={"score": "frozen_score"}
    )
    merged = left.merge(right, on=keys, validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ContractError("v44 raw replay prediction coverage drifted")
    return float(np.max(np.abs(merged["replay_score"] - merged["frozen_score"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--relative-feature-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--v40-run-dir", required=True, type=Path)
    parser.add_argument("--v42-phase-a-run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v44 seed7 run refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("protocol_version") != "v44-phase-a":
            raise ContractError("v44 phase-A config identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v44 config contains a secret-like key")
        if config.get("gates") != EXPECTED_GATES:
            raise ContractError("v44 holistic gates drifted")
        if (
            tuple(cast(list[object], config["student_seeds"])) != (7,)
            or tuple(cast(list[object], config["teacher_seeds"])) != (7, 17, 27)
            or tuple(cast(list[object], config["folds"])) != (0, 1, 2, 3, 4)
            or float(cast(Any, config["teacher_weight"])) != 0.25
            or int(cast(Any, config["epoch_average_start"])) != 2
            or config.get("precision") != "float32"
            or int(cast(Any, config["torch_threads"])) != 8
            or int(cast(Any, config["num_workers"])) != 0
            or int(cast(Any, config["max_epochs"])) != 8
        ):
            raise ContractError("v44 frozen execution contract drifted")

        base = arguments.base_run_dir.resolve()
        relative = arguments.relative_feature_dir.resolve()
        v40 = arguments.v40_run_dir.resolve()
        v42 = arguments.v42_phase_a_run_dir.resolve()
        source_paths = {
            "relative_features": relative / "feature-windows.npy",
            "relative_window_index": relative / "window-index.parquet",
            "relative_manifest": relative / "manifest.json",
            "relative_receipt": relative / "receipt.json",
            "labels": base / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "v40_predictions": v40 / "predictions.parquet",
            "v40_timing_summary": v40 / "timing-summary.json",
            "v40_receipt": v40 / "receipt.json",
            "v42_predictions": v42 / "predictions.parquet",
            "v42_leaderboard": v42 / "leaderboard.parquet",
            "v42_teacher_targets": v42 / "teacher-consensus-targets.parquet",
            "v42_teacher_predictions": v42 / "teacher-train-predictions.parquet",
            "v42_teacher_audit": v42 / "teacher-audit.json",
            "v42_timing": v42 / "timing.json",
            "v42_model_gate": v42 / "model-gate.json",
            "v42_receipt": v42 / "receipt.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v44 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v44 source SHA-256 identity drifted")
        for receipt_name in ("v40_receipt", "v42_receipt"):
            source_receipt = json.loads(
                source_paths[receipt_name].read_text(encoding="utf-8")
            )
            if (
                not isinstance(source_receipt, dict)
                or source_receipt.get("sealed_test_accessed") is not False
            ):
                raise ContractError(f"v44 source is not ordinary validation: {receipt_name}")
        v42_gate = json.loads(source_paths["v42_model_gate"].read_text(encoding="utf-8"))
        if not isinstance(v42_gate, dict) or v42_gate.get("admitted") is not True:
            raise ContractError("v44 requires the frozen v42 seed7 pointwise student")

        features = np.load(
            source_paths["relative_features"], mmap_mode="r", allow_pickle=False
        )
        if features.ndim != 3 or features.shape[1:] != (10, 480):
            raise ContractError("v44 relative10 tensor shape drifted")
        window_index = pd.read_parquet(source_paths["relative_window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v44 rejects sealed split rows")
        stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(stages - {"train", "validation", "purged"}):
            raise ContractError("v44 split has forbidden stages: " + ", ".join(unknown))
        split = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].astype(str).isin(["train", "validation"])
        ].copy()
        teacher_targets, teacher_identity, teacher_audit = _load_teacher_targets(
            features,
            window_index,
            labels,
            split,
            source_paths["v42_teacher_targets"],
            source_paths["v42_teacher_audit"],
        )
        true_targets, masks = _label_matrices(window_index, labels)
        overrides: dict[int, np.ndarray] = {}
        for protocol in build_fold_protocols(features, split):
            overrides[protocol.fold] = blend_training_targets(
                true_targets,
                masks,
                teacher_targets[protocol.fold],
                train_positions=np.asarray(protocol.train_positions, dtype="int64"),
                teacher_weight=0.25,
            )

        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 2:
            raise ContractError("v44 fixes exactly raw replay and averaged trials")
        by_id = {trial.trial_id: trial for trial in trials}
        raw_id = str(config["raw_replay_trial_id"])
        candidate_id = str(config["candidate_trial_id"])
        if set(by_id) != {raw_id, candidate_id}:
            raise ContractError("v44 trial identities drifted")
        raw_trial = by_id[raw_id]
        candidate_trial = by_id[candidate_id]
        if (
            raw_trial.epoch_average_start is not None
            or candidate_trial.epoch_average_start != 2
            or raw_trial.ema_decay is not None
            or candidate_trial.ema_decay is not None
            or raw_trial.strategy != "smooth_l1"
            or candidate_trial.strategy != "smooth_l1"
        ):
            raise ContractError("v44 trajectory-average trial contract drifted")
        raw_contract = asdict(raw_trial)
        candidate_contract = asdict(candidate_trial)
        for contract in (raw_contract, candidate_contract):
            contract.pop("trial_id")
            contract.pop("epoch_average_start")
        if raw_contract != candidate_contract:
            raise ContractError("v44 trials differ by more than final averaging")

        reliability_units, reliability_summary = _reliability_diagnostic(
            pd.read_parquet(source_paths["v42_teacher_predictions"]),
            pd.read_parquet(source_paths["v42_teacher_targets"]),
        )
        temporary.mkdir(parents=True)
        tuning = run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split,
            trials=(raw_trial, candidate_trial),
            seed=7,
            max_epochs=8,
            patience=int(cast(Any, config["patience"])),
            min_delta=float(cast(Any, config["min_delta"])),
            checkpoint_min_delta=float(cast(Any, config["checkpoint_min_delta"])),
            torch_threads=8,
            protocol_identities={
                "data": teacher_identity,
                "fold_manifest": observed_hashes["split_manifest"],
                "evaluation": observed_hashes["labels"],
            },
            capture_epoch_states=True,
            disable_early_stopping=True,
            training_target_overrides=overrides,
        )
        if set(tuning.leaderboard["training_target_override"]) != {True}:
            raise ContractError("v44 pointwise training target override was not active")
        trajectory, raw_state_drift_max = _trajectory_audit(
            tuning.epoch_states,
            control_trial_id=raw_id,
            candidate_trial_id=candidate_id,
            folds=(0, 1, 2, 3, 4),
            max_epochs=8,
        )
        average_audit, arithmetic_error, parameter_distance = _average_audit(
            tuning,
            candidate_trial,
            feature_count=int(features.shape[1]),
            input_steps=int(features.shape[2]),
            folds=(0, 1, 2, 3, 4),
            start_epoch=2,
            end_epoch=8,
        )
        average_counts = tuning.leaderboard.loc[
            tuning.leaderboard["trial_id"].astype(str).eq(candidate_id),
            "epoch_average_update_count",
        ].to_numpy(dtype="int64")
        if len(average_counts) != 5:
            raise ContractError("v44 average update coverage drifted")

        contracts = cast(dict[str, str], config["contracts"])
        raw_contracts = dict(contracts)
        raw_contracts["tcn_training_contract_id"] = "top50-relative10-v42-raw-replay-v44"
        candidate_contracts = dict(contracts)
        candidate_contracts["tcn_training_contract_id"] = (
            "top50-relative10-trajectory-average-epochs2-8-v44"
        )
        lookup = _label_lookup(labels)
        raw_predictions = _collect_tcn_predictions(
            features,
            labels,
            split,
            raw_trial,
            tuning.best_states,
            seed=7,
            model_name="raw_replay_tcn",
            lookup=lookup,
            contracts=raw_contracts,
        )
        candidate_predictions = _collect_tcn_predictions(
            features,
            labels,
            split,
            candidate_trial,
            tuning.best_states,
            seed=7,
            model_name="trajectory_average_tcn",
            lookup=lookup,
            contracts=candidate_contracts,
        )
        v42_predictions = pd.read_parquet(source_paths["v42_predictions"])
        frozen_pointwise = v42_predictions.loc[
            v42_predictions["model"].astype(str).eq("consensus_student_tcn")
        ].copy()
        replay_error = _replay_max_abs_error(raw_predictions, frozen_pointwise)
        if replay_error > 1e-7:
            raise ContractError("v44 raw replay no longer reproduces v42 pointwise")
        v42_leaderboard = pd.read_parquet(source_paths["v42_leaderboard"])
        pointwise_id = str(config["pointwise_trial_id"])
        frozen_epochs = v42_leaderboard.loc[
            v42_leaderboard["trial_id"].astype(str).eq(pointwise_id)
        ].set_index("fold")["best_epoch"].sort_index()
        replay_epochs = tuning.leaderboard.loc[
            tuning.leaderboard["trial_id"].astype(str).eq(raw_id)
        ].set_index("fold")["best_epoch"].sort_index()
        if not np.array_equal(
            frozen_epochs.to_numpy(dtype="int64"),
            replay_epochs.to_numpy(dtype="int64"),
        ):
            raise ContractError("v44 raw replay checkpoint selection drifted")

        frozen = v42_predictions.copy()
        frozen["model"] = frozen["model"].replace(
            {"control_tcn": "control_tcn", "consensus_student_tcn": "pointwise_student_tcn"}
        )
        for column in (
            "prediction_contract_id",
            "target_contract_id",
            "evaluation_contract_id",
        ):
            frozen[column] = contracts[column]
            candidate_predictions[column] = contracts[column]
        frozen.loc[frozen["model"].eq("control_tcn"), "training_contract_id"] = (
            "top50-relative10-control-tcn-v44"
        )
        frozen.loc[
            frozen["model"].eq("pointwise_student_tcn"), "training_contract_id"
        ] = "top50-relative10-pointwise-student-v42"
        predictions = pd.concat([frozen, candidate_predictions], ignore_index=True)
        validate_prediction_contract(predictions, expected_models=3)
        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(cast(Any, config["top_fraction"]))
        )
        summary = summarize_task_aligned_metrics(metrics)
        control_comparison = compare_task_aligned_models(
            metrics,
            reference_model="control_tcn",
            candidate_model="trajectory_average_tcn",
        )
        pointwise_comparison = compare_task_aligned_models(
            metrics,
            reference_model="pointwise_student_tcn",
            candidate_model="trajectory_average_tcn",
        )
        bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="control_tcn",
            candidate_model="trajectory_average_tcn",
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
        gate_metrics = metrics.loc[
            metrics["model"].isin(["control_tcn", "trajectory_average_tcn"])
        ].replace({"trajectory_average_tcn": "ema_tcn"})
        fold_deltas, horizon_deltas = _unit_deltas(gate_metrics)
        teacher = _teacher_ensemble(pd.read_parquet(source_paths["v40_predictions"]))
        fidelity_frame, fidelity = _teacher_fidelity(
            predictions,
            teacher,
            models=("pointwise_student_tcn", "trajectory_average_tcn"),
        )

        history = tuning.epoch_history.loc[
            tuning.epoch_history["epoch"].astype(int).between(2, 8)
        ]
        volatility = (
            history.groupby(["trial_id", "fold"], observed=True)["mean_daily_rankic"]
            .std(ddof=0)
            .rename("epoch_rankic_std")
            .reset_index()
        )
        raw_volatility = float(
            volatility.loc[volatility["trial_id"].astype(str).eq(raw_id), "epoch_rankic_std"].median()
        )
        average_volatility = float(
            volatility.loc[
                volatility["trial_id"].astype(str).eq(candidate_id), "epoch_rankic_std"
            ].median()
        )
        if not np.isfinite([raw_volatility, average_volatility]).all() or raw_volatility <= 0:
            raise ContractError("v44 validation volatility audit is invalid")
        volatility_ratio = average_volatility / raw_volatility

        pointwise_timing = v42_leaderboard.loc[
            v42_leaderboard["trial_id"].astype(str).eq(pointwise_id)
        ].set_index("fold")
        candidate_timing = tuning.leaderboard.loc[
            tuning.leaderboard["trial_id"].astype(str).eq(candidate_id)
        ].set_index("fold")
        if set(pointwise_timing.index.astype(int)) != set(range(5)):
            raise ContractError("v44 pointwise timing coverage drifted")
        model_step_retention = _geometric_mean(
            (
                candidate_timing["model_step_samples_per_second"]
                / pointwise_timing["model_step_samples_per_second"]
            ).to_numpy(dtype="float64")
        )
        complete_cycle_retention = _geometric_mean(
            (
                pointwise_timing["complete_cycle_seconds"]
                / candidate_timing["complete_cycle_seconds"]
            ).to_numpy(dtype="float64")
        )
        v42_timing = json.loads(source_paths["v42_timing"].read_text(encoding="utf-8"))
        implied_ratio = (
            float(v42_timing["implied_tcn_lstm_model_step_ratio"])
            * model_step_retention
        )
        decision = decide_trajectory_average_seed7_gate(
            control_comparison,
            pointwise_comparison,
            bootstrap,
            fold_deltas,
            horizon_deltas,
            teacher_fidelity_delta=float(fidelity["teacher_fidelity_delta"]),
            validation_volatility_ratio=volatility_ratio,
            raw_state_drift_max=raw_state_drift_max,
            arithmetic_mean_max_error=arithmetic_error,
            average_parameter_distance=parameter_distance,
            average_update_count_min=int(average_counts.min()),
            average_update_count_max=int(average_counts.max()),
            model_step_retention=model_step_retention,
            complete_cycle_retention=complete_cycle_retention,
            implied_tcn_lstm_model_step_ratio=implied_ratio,
            inference_forward_passes=1,
        )
        timing = {
            "model_step_retention_geomean": model_step_retention,
            "complete_cycle_retention_geomean": complete_cycle_retention,
            "implied_tcn_lstm_model_step_ratio": implied_ratio,
            "inference_forward_passes": 1,
        }
        stability = {
            "raw_median_epoch_rankic_std": raw_volatility,
            "average_median_epoch_rankic_std": average_volatility,
            "validation_volatility_ratio": volatility_ratio,
        }

        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        raw_predictions.to_parquet(temporary / "raw-replay-predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        tuning.leaderboard.to_parquet(temporary / "leaderboard.parquet", index=False)
        tuning.epoch_history.to_parquet(temporary / "epoch-history.parquet", index=False)
        trajectory.to_parquet(temporary / "raw-trajectory-audit.parquet", index=False)
        average_audit.to_parquet(temporary / "epoch-average-audit.parquet", index=False)
        volatility.to_parquet(temporary / "validation-volatility.parquet", index=False)
        reliability_units.to_parquet(temporary / "teacher-reliability-units.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap.parquet", index=False)
        fold_deltas.to_parquet(temporary / "fold-deltas.parquet", index=False)
        horizon_deltas.to_parquet(temporary / "horizon-deltas.parquet", index=False)
        fidelity_frame.to_parquet(temporary / "teacher-fidelity.parquet", index=False)
        _write_json(temporary / "control-comparison.json", control_comparison)
        _write_json(temporary / "pointwise-comparison.json", pointwise_comparison)
        _write_json(temporary / "teacher-fidelity-summary.json", fidelity)
        _write_json(temporary / "teacher-reliability-summary.json", reliability_summary)
        _write_json(temporary / "stability.json", stability)
        _write_json(temporary / "timing.json", timing)
        _write_json(
            temporary / "trajectory-mechanism.json",
            {
                "raw_state_drift_max": raw_state_drift_max,
                "arithmetic_mean_max_error": arithmetic_error,
                "average_parameter_distance": parameter_distance,
                "average_update_count_min": int(average_counts.min()),
                "average_update_count_max": int(average_counts.max()),
                "raw_replay_prediction_max_abs_error": replay_error,
                "validation_selected_average_checkpoint": False,
                "sealed_test_accessed": False,
            },
        )
        _write_json(
            temporary / "teacher-target-audit.json",
            {
                **teacher_audit,
                "teacher_target_identity_recomputed": teacher_identity,
                "training_target_override": True,
                "teacher_weight": 0.25,
                "validation_teacher_cells_exposed": 0,
                "sealed_test_accessed": False,
            },
        )
        model_gate = {
            "status": decision.status,
            "admitted": decision.admitted,
            "blockers": list(decision.blockers),
            "evidence": decision.evidence,
            "phase_b_authorized": decision.admitted,
            "sealed_test_accessed": False,
        }
        _write_json(temporary / "model-gate.json", model_gate)
        _write_json(temporary / "config.resolved.json", config)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for key, state in tuning.best_states.items():
            torch.save(state, checkpoint_dir / f"seed-7-{key}.pt")
        epoch_state_dir = temporary / "raw-epoch-states"
        epoch_state_dir.mkdir()
        for key, state in tuning.epoch_states.items():
            torch.save(state, epoch_state_dir / f"seed-7-{key}.pt")
        report = (
            "# TCN v44 validation-independent trajectory average seed7\n\n"
            f"- status: `{decision.status}`\n"
            f"- vs control RankIC delta: `{float(control_comparison['mean_rankic_delta']):+.6f}`\n"
            f"- vs pointwise RankIC delta: `{float(pointwise_comparison['mean_rankic_delta']):+.6f}`\n"
            f"- vs pointwise improved metrics: `{decision.evidence['pointwise_broad_metric_count']}/6`\n"
            f"- validation volatility ratio: `{volatility_ratio:.6f}`\n"
            f"- teacher fidelity delta: `{float(fidelity['teacher_fidelity_delta']):+.6f}`\n"
            f"- implied TCN/LSTM model-step ratio: `{implied_ratio:.4f}x`\n"
            "- inference model count: `1`\n"
            "- sealed_test_accessed: `false`\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")

        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, object] = {
            "schema_version": "tcn-v44-trajectory-average-seed7/v1",
            "run_id": str(config["run_id"]),
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "teacher_target_identity": teacher_identity,
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "torch_threads": 8,
                "precision": "float32",
            },
            "model_gate": model_gate,
            "timing": timing,
            "stability": stability,
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
            "phase_b_authorized": decision.admitted,
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
