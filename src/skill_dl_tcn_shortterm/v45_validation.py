"""Holistic gate for linear distillation annealing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class DistillationAnnealingDecision:
    status: str
    admitted: bool
    blockers: tuple[str, ...]
    evidence: dict[str, float | int | bool]


_METRICS = (
    "mean_rankic_delta",
    "mean_pearson_ic_delta",
    "mean_top_return_delta",
    "mean_top_precision_delta",
    "mean_ndcg_at_top_delta",
    "mean_quantile_monotonicity_delta",
)


def decide_distillation_annealing_seed7_gate(
    control_comparison: Mapping[str, object],
    pointwise_comparison: Mapping[str, object],
    bootstrap: pd.DataFrame,
    fold_deltas: pd.DataFrame,
    horizon_deltas: pd.DataFrame,
    *,
    schedule_max_abs_error: float,
    terminal_teacher_weight: float,
    validation_teacher_cells_exposed: int,
    model_step_retention: float,
    complete_cycle_retention: float,
    implied_tcn_lstm_model_step_ratio: float,
    inference_forward_passes: int,
) -> DistillationAnnealingDecision:
    """Apply non-compensating global, Pareto, schedule, and speed gates."""

    try:
        control = {
            metric: float(cast(Any, control_comparison[metric]))
            for metric in _METRICS
        }
        pointwise = {
            metric: float(cast(Any, pointwise_comparison[metric]))
            for metric in _METRICS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v45 comparisons are incomplete") from exc
    numeric = np.asarray(
        [
            *control.values(),
            *pointwise.values(),
            schedule_max_abs_error,
            terminal_teacher_weight,
            model_step_retention,
            complete_cycle_retention,
            implied_tcn_lstm_model_step_ratio,
        ],
        dtype="float64",
    )
    if not np.isfinite(numeric).all():
        raise ContractError("v45 holistic evidence must be finite")
    if (
        len(fold_deltas) != 5
        or set(fold_deltas["fold"].astype(int)) != set(range(5))
        or fold_deltas["fold"].duplicated().any()
    ):
        raise ContractError("v45 seed7 fold coverage must be exactly 0..4")
    if (
        len(horizon_deltas) != 4
        or set(horizon_deltas["horizon"].astype(int)) != {1, 2, 3, 5}
        or horizon_deltas["horizon"].duplicated().any()
    ):
        raise ContractError("v45 horizon coverage must be exactly 1/2/3/5")
    fold_values = fold_deltas["rankic_delta"].to_numpy(dtype="float64")
    horizon_values = horizon_deltas["rankic_delta"].to_numpy(dtype="float64")
    rankic_bootstrap = bootstrap.loc[
        bootstrap["metric"].astype(str).eq("rankic")
    ]
    if (
        len(rankic_bootstrap) != 1
        or "bootstrap_ci_low" not in rankic_bootstrap
        or not np.isfinite(fold_values).all()
        or not np.isfinite(horizon_values).all()
    ):
        raise ContractError("v45 paired gate evidence is incomplete")
    rankic_ci_low = float(cast(Any, rankic_bootstrap["bootstrap_ci_low"].iloc[0]))
    if not np.isfinite(rankic_ci_low):
        raise ContractError("v45 RankIC bootstrap bound must be finite")

    control_broad = sum(value > 0.0 for value in control.values())
    pointwise_broad = sum(value > 0.0 for value in pointwise.values())
    positive_folds = int((fold_values > 0.0).sum())
    positive_horizons = int((horizon_values > 0.0).sum())
    blockers: list[str] = []
    if control["mean_rankic_delta"] < 0.002:
        blockers.append("control_rankic_delta_below_gate")
    if positive_folds < 3:
        blockers.append("control_positive_fold_count_below_gate")
    if rankic_ci_low < -0.002:
        blockers.append("control_rankic_ci_low_below_gate")
    if control_broad < 4:
        blockers.append("control_broad_metric_count_below_gate")
    for metric, threshold in {
        "mean_top_return_delta": -0.0001,
        "mean_top_precision_delta": -0.002,
        "mean_ndcg_at_top_delta": -0.001,
        "mean_quantile_monotonicity_delta": -0.002,
    }.items():
        if control[metric] < threshold:
            blockers.append("control_" + metric.removeprefix("mean_") + "_below_gate")
    if positive_horizons < 3:
        blockers.append("control_positive_horizon_count_below_gate")
    if float(horizon_values.min()) < -0.003:
        blockers.append("control_worst_horizon_delta_below_gate")
    if pointwise_broad < 3:
        blockers.append("pointwise_pareto_breadth_below_gate")
    for metric, threshold in {
        "mean_rankic_delta": -0.002,
        "mean_pearson_ic_delta": -0.002,
        "mean_top_return_delta": -0.0001,
        "mean_top_precision_delta": -0.001,
        "mean_ndcg_at_top_delta": -0.001,
        "mean_quantile_monotonicity_delta": -0.002,
    }.items():
        if pointwise[metric] < threshold:
            blockers.append("pointwise_" + metric.removeprefix("mean_") + "_below_gate")
    if schedule_max_abs_error > 1e-12:
        blockers.append("teacher_schedule_identity_failed")
    if terminal_teacher_weight != 0.0:
        blockers.append("terminal_true_target_gate_failed")
    if validation_teacher_cells_exposed != 0:
        blockers.append("validation_teacher_leakage_detected")
    if model_step_retention < 0.95:
        blockers.append("model_step_retention_below_gate")
    if complete_cycle_retention < 0.90:
        blockers.append("complete_cycle_retention_below_gate")
    if implied_tcn_lstm_model_step_ratio < 3.0:
        blockers.append("implied_tcn_lstm_speed_below_gate")
    if inference_forward_passes != 1:
        blockers.append("single_model_inference_gate_failed")
    blockers = list(dict.fromkeys(blockers))
    admitted = not blockers
    evidence: dict[str, float | int | bool] = {
        **{f"control_{key}": value for key, value in control.items()},
        **{f"pointwise_{key}": value for key, value in pointwise.items()},
        "control_broad_metric_count": control_broad,
        "pointwise_broad_metric_count": pointwise_broad,
        "positive_folds": positive_folds,
        "rankic_ci_low": rankic_ci_low,
        "positive_horizons": positive_horizons,
        "worst_horizon_rankic_delta": float(horizon_values.min()),
        "schedule_max_abs_error": schedule_max_abs_error,
        "terminal_teacher_weight": terminal_teacher_weight,
        "validation_teacher_cells_exposed": validation_teacher_cells_exposed,
        "model_step_retention": model_step_retention,
        "complete_cycle_retention": complete_cycle_retention,
        "implied_tcn_lstm_model_step_ratio": implied_tcn_lstm_model_step_ratio,
        "inference_forward_passes": inference_forward_passes,
        "membership_turnover_is_model_gate": False,
        "sealed_test_accessed": False,
    }
    return DistillationAnnealingDecision(
        status=(
            "linear_distillation_annealing_seed7_admitted_v45"
            if admitted
            else "stop_linear_distillation_annealing_seed7_v45"
        ),
        admitted=admitted,
        blockers=tuple(blockers),
        evidence=evidence,
    )
