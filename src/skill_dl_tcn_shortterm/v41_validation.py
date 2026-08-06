"""Non-compensatory global gates for the v41 single-model EMA probe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class EMAHolisticDecision:
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


def decide_ema_holistic_gate(
    comparison: Mapping[str, object],
    bootstrap: pd.DataFrame,
    fold_deltas: pd.DataFrame,
    horizon_deltas: pd.DataFrame,
    *,
    raw_state_drift_max: float,
    model_step_retention: float,
    complete_cycle_retention: float,
    implied_tcn_lstm_model_step_ratio: float,
    min_model_step_retention: float = 0.90,
    min_complete_cycle_retention: float = 0.85,
    admitted_status: str = "ema_seed7_holistic_admitted_v41",
    rejected_status: str = "stop_ema_seed7_no_holistic_gain_v41",
) -> EMAHolisticDecision:
    """Require broad model improvement; no metric is allowed to hide a collapse."""

    try:
        values = {
            metric: float(cast(Any, comparison[metric]))
            for metric in _BROAD_METRICS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v41 comparison is incomplete") from exc
    numeric = np.asarray(
        [
            *values.values(),
            raw_state_drift_max,
            model_step_retention,
            complete_cycle_retention,
            implied_tcn_lstm_model_step_ratio,
        ],
        dtype="float64",
    )
    if not np.isfinite(numeric).all():
        raise ContractError("v41 holistic evidence must be finite")
    required_fold = {"fold", "rankic_delta"}
    required_horizon = {"horizon", "rankic_delta"}
    if missing := sorted(required_fold.difference(fold_deltas.columns)):
        raise ContractError("v41 fold deltas missing: " + ", ".join(missing))
    if missing := sorted(required_horizon.difference(horizon_deltas.columns)):
        raise ContractError("v41 horizon deltas missing: " + ", ".join(missing))
    if (
        len(fold_deltas) != 5
        or set(fold_deltas["fold"].astype(int)) != set(range(5))
        or fold_deltas["fold"].duplicated().any()
    ):
        raise ContractError("v41 seed7 fold coverage must be exactly 0..4")
    if (
        len(horizon_deltas) != 4
        or set(horizon_deltas["horizon"].astype(int)) != {1, 2, 3, 5}
        or horizon_deltas["horizon"].duplicated().any()
    ):
        raise ContractError("v41 horizon coverage must be exactly 1/2/3/5")
    fold_values = fold_deltas["rankic_delta"].to_numpy(dtype="float64")
    horizon_values = horizon_deltas["rankic_delta"].to_numpy(dtype="float64")
    if not np.isfinite(fold_values).all() or not np.isfinite(horizon_values).all():
        raise ContractError("v41 unit deltas must be finite")
    rankic_bootstrap = bootstrap.loc[
        bootstrap["metric"].astype(str).eq("rankic")
    ]
    if len(rankic_bootstrap) != 1 or "bootstrap_ci_low" not in rankic_bootstrap:
        raise ContractError("v41 requires one RankIC bootstrap row")
    rankic_ci_low = float(cast(Any, rankic_bootstrap["bootstrap_ci_low"].iloc[0]))
    if not np.isfinite(rankic_ci_low):
        raise ContractError("v41 RankIC bootstrap bound must be finite")

    broad_metric_count = sum(value > 0.0 for value in values.values())
    positive_folds = int((fold_values > 0.0).sum())
    positive_horizons = int((horizon_values > 0.0).sum())
    blockers: list[str] = []
    if raw_state_drift_max != 0.0:
        blockers.append("raw_training_trajectory_drifted")
    if values["mean_rankic_delta"] < 0.002:
        blockers.append("rankic_delta_below_gate")
    if positive_folds < 3:
        blockers.append("positive_fold_count_below_gate")
    if rankic_ci_low < -0.002:
        blockers.append("rankic_ci_low_below_gate")
    if broad_metric_count < 4:
        blockers.append("broad_metric_count_below_gate")
    lower_bounds = {
        "mean_top_return_delta": -0.0001,
        "mean_top_precision_delta": -0.002,
        "mean_ndcg_at_top_delta": -0.001,
        "mean_quantile_monotonicity_delta": -0.002,
    }
    for metric, threshold in lower_bounds.items():
        if values[metric] < threshold:
            blockers.append(metric.removeprefix("mean_") + "_below_gate")
    if positive_horizons < 3:
        blockers.append("positive_horizon_count_below_gate")
    if float(horizon_values.min()) < -0.003:
        blockers.append("worst_horizon_delta_below_gate")
    if model_step_retention < min_model_step_retention:
        blockers.append("model_step_retention_below_gate")
    if complete_cycle_retention < min_complete_cycle_retention:
        blockers.append("complete_cycle_retention_below_gate")
    if implied_tcn_lstm_model_step_ratio < 3.0:
        blockers.append("implied_tcn_lstm_speed_below_gate")
    blockers = list(dict.fromkeys(blockers))
    admitted = not blockers
    evidence: dict[str, float | int | bool] = {
        **values,
        "broad_metric_count": broad_metric_count,
        "positive_folds": positive_folds,
        "rankic_ci_low": rankic_ci_low,
        "positive_horizons": positive_horizons,
        "worst_horizon_rankic_delta": float(horizon_values.min()),
        "raw_state_drift_max": raw_state_drift_max,
        "model_step_retention": model_step_retention,
        "complete_cycle_retention": complete_cycle_retention,
        "implied_tcn_lstm_model_step_ratio": implied_tcn_lstm_model_step_ratio,
        "membership_turnover_is_model_gate": False,
        "sealed_test_accessed": False,
    }
    return EMAHolisticDecision(
        status=(
            admitted_status if admitted else rejected_status
        ),
        admitted=admitted,
        blockers=tuple(blockers),
        evidence=evidence,
    )
