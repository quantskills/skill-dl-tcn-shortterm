"""Holistic gate for gradient-normalized listwise consensus distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class ListwiseConsensusDecision:
    status: str
    admitted: bool
    blockers: tuple[str, ...]
    evidence: dict[str, float | int | bool]


_BROAD_METRICS = (
    "mean_rankic_delta",
    "mean_pearson_ic_delta",
    "mean_top_return_delta",
    "mean_top_precision_delta",
    "mean_ndcg_at_top_delta",
    "mean_quantile_monotonicity_delta",
)


def decide_listwise_consensus_seed7_gate(
    control_comparison: Mapping[str, object],
    pointwise_comparison: Mapping[str, object],
    bootstrap: pd.DataFrame,
    fold_deltas: pd.DataFrame,
    horizon_deltas: pd.DataFrame,
    *,
    teacher_fidelity_delta: float,
    median_teacher_gradient_ratio: float,
    model_step_retention: float,
    complete_cycle_retention: float,
    implied_tcn_lstm_model_step_ratio: float,
    inference_forward_passes: int,
) -> ListwiseConsensusDecision:
    """Require global gain, pointwise Pareto protection, and mechanism fidelity."""

    try:
        control = {
            metric: float(cast(Any, control_comparison[metric]))
            for metric in _BROAD_METRICS
        }
        pointwise = {
            metric: float(cast(Any, pointwise_comparison[metric]))
            for metric in _BROAD_METRICS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v43 comparisons are incomplete") from exc
    numeric = np.asarray(
        [
            *control.values(),
            *pointwise.values(),
            teacher_fidelity_delta,
            median_teacher_gradient_ratio,
            model_step_retention,
            complete_cycle_retention,
            implied_tcn_lstm_model_step_ratio,
        ],
        dtype="float64",
    )
    if not np.isfinite(numeric).all():
        raise ContractError("v43 holistic evidence must be finite")
    if (
        len(fold_deltas) != 5
        or set(fold_deltas["fold"].astype(int)) != set(range(5))
        or fold_deltas["fold"].duplicated().any()
    ):
        raise ContractError("v43 seed7 fold coverage must be exactly 0..4")
    if (
        len(horizon_deltas) != 4
        or set(horizon_deltas["horizon"].astype(int)) != {1, 2, 3, 5}
        or horizon_deltas["horizon"].duplicated().any()
    ):
        raise ContractError("v43 horizon coverage must be exactly 1/2/3/5")
    fold_values = fold_deltas["rankic_delta"].to_numpy(dtype="float64")
    horizon_values = horizon_deltas["rankic_delta"].to_numpy(dtype="float64")
    if not np.isfinite(fold_values).all() or not np.isfinite(horizon_values).all():
        raise ContractError("v43 unit deltas must be finite")
    rankic_bootstrap = bootstrap.loc[
        bootstrap["metric"].astype(str).eq("rankic")
    ]
    if len(rankic_bootstrap) != 1 or "bootstrap_ci_low" not in rankic_bootstrap:
        raise ContractError("v43 requires one RankIC bootstrap row")
    rankic_ci_low = float(cast(Any, rankic_bootstrap["bootstrap_ci_low"].iloc[0]))
    if not np.isfinite(rankic_ci_low):
        raise ContractError("v43 RankIC bootstrap bound must be finite")

    control_broad_count = sum(value > 0.0 for value in control.values())
    pointwise_broad_count = sum(value > 0.0 for value in pointwise.values())
    positive_folds = int((fold_values > 0.0).sum())
    positive_horizons = int((horizon_values > 0.0).sum())
    blockers: list[str] = []
    if control["mean_rankic_delta"] < 0.002:
        blockers.append("control_rankic_delta_below_gate")
    if positive_folds < 3:
        blockers.append("control_positive_fold_count_below_gate")
    if rankic_ci_low < -0.002:
        blockers.append("control_rankic_ci_low_below_gate")
    if control_broad_count < 4:
        blockers.append("control_broad_metric_count_below_gate")
    control_bounds = {
        "mean_top_return_delta": -0.0001,
        "mean_top_precision_delta": -0.002,
        "mean_ndcg_at_top_delta": -0.001,
        "mean_quantile_monotonicity_delta": -0.002,
    }
    for metric, threshold in control_bounds.items():
        if control[metric] < threshold:
            blockers.append("control_" + metric.removeprefix("mean_") + "_below_gate")
    if positive_horizons < 3:
        blockers.append("control_positive_horizon_count_below_gate")
    if float(horizon_values.min()) < -0.003:
        blockers.append("control_worst_horizon_delta_below_gate")
    if pointwise_broad_count < 3:
        blockers.append("pointwise_pareto_breadth_below_gate")
    pointwise_bounds = {
        "mean_rankic_delta": -0.002,
        "mean_pearson_ic_delta": -0.002,
        "mean_top_return_delta": -0.0001,
        "mean_top_precision_delta": -0.001,
        "mean_ndcg_at_top_delta": -0.001,
        "mean_quantile_monotonicity_delta": -0.002,
    }
    for metric, threshold in pointwise_bounds.items():
        if pointwise[metric] < threshold:
            blockers.append("pointwise_" + metric.removeprefix("mean_") + "_below_gate")
    if teacher_fidelity_delta < 0.002:
        blockers.append("teacher_fidelity_delta_below_gate")
    if not 0.20 <= median_teacher_gradient_ratio <= 0.30:
        blockers.append("teacher_gradient_ratio_outside_gate")
    if model_step_retention < 0.70:
        blockers.append("model_step_retention_below_gate")
    if complete_cycle_retention < 0.70:
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
        "control_broad_metric_count": control_broad_count,
        "pointwise_broad_metric_count": pointwise_broad_count,
        "positive_folds": positive_folds,
        "rankic_ci_low": rankic_ci_low,
        "positive_horizons": positive_horizons,
        "worst_horizon_rankic_delta": float(horizon_values.min()),
        "teacher_fidelity_delta": teacher_fidelity_delta,
        "median_teacher_gradient_ratio": median_teacher_gradient_ratio,
        "model_step_retention": model_step_retention,
        "complete_cycle_retention": complete_cycle_retention,
        "implied_tcn_lstm_model_step_ratio": implied_tcn_lstm_model_step_ratio,
        "inference_forward_passes": inference_forward_passes,
        "membership_turnover_is_model_gate": False,
        "sealed_test_accessed": False,
    }
    return ListwiseConsensusDecision(
        status=(
            "listwise_consensus_seed7_holistic_admitted_v43"
            if admitted
            else "stop_listwise_consensus_seed7_v43"
        ),
        admitted=admitted,
        blockers=tuple(blockers),
        evidence=evidence,
    )
