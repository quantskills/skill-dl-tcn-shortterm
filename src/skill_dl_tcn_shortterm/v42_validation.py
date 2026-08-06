"""Frozen multi-seed gate for the v42 consensus-distilled TCN student."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class ConsensusStudentDecision:
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


def decide_consensus_student_multiseed_gate(
    comparison: Mapping[str, object],
    bootstrap: pd.DataFrame,
    seed_fold_deltas: pd.DataFrame,
    horizon_deltas: pd.DataFrame,
    *,
    model_step_retention: float,
    complete_cycle_retention: float,
    implied_tcn_lstm_model_step_ratio: float,
    inference_forward_passes: int,
) -> ConsensusStudentDecision:
    """Admit only broad, stable gains from a single distilled TCN."""

    try:
        values = {
            metric: float(cast(Any, comparison[metric]))
            for metric in _BROAD_METRICS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v42 multi-seed comparison is incomplete") from exc
    numeric = np.asarray(
        [
            *values.values(),
            model_step_retention,
            complete_cycle_retention,
            implied_tcn_lstm_model_step_ratio,
        ],
        dtype="float64",
    )
    if not np.isfinite(numeric).all():
        raise ContractError("v42 multi-seed evidence must be finite")

    required_unit = {"seed", "fold", "rankic_delta"}
    required_horizon = {"horizon", "rankic_delta"}
    if missing := sorted(required_unit.difference(seed_fold_deltas.columns)):
        raise ContractError("v42 seed/fold deltas missing: " + ", ".join(missing))
    if missing := sorted(required_horizon.difference(horizon_deltas.columns)):
        raise ContractError("v42 horizon deltas missing: " + ", ".join(missing))
    expected_units = {
        (seed, fold) for seed in (7, 17, 27) for fold in range(5)
    }
    observed_units = set(
        zip(
            seed_fold_deltas["seed"].astype(int),
            seed_fold_deltas["fold"].astype(int),
            strict=True,
        )
    )
    if (
        len(seed_fold_deltas) != 15
        or observed_units != expected_units
        or seed_fold_deltas.duplicated(["seed", "fold"]).any()
    ):
        raise ContractError("v42 multi-seed coverage must be exactly 3 seeds x 5 folds")
    if (
        len(horizon_deltas) != 4
        or set(horizon_deltas["horizon"].astype(int)) != {1, 2, 3, 5}
        or horizon_deltas["horizon"].duplicated().any()
    ):
        raise ContractError("v42 horizon coverage must be exactly 1/2/3/5")
    unit_values = seed_fold_deltas["rankic_delta"].to_numpy(dtype="float64")
    horizon_values = horizon_deltas["rankic_delta"].to_numpy(dtype="float64")
    if not np.isfinite(unit_values).all() or not np.isfinite(horizon_values).all():
        raise ContractError("v42 multi-seed unit deltas must be finite")

    rankic_bootstrap = bootstrap.loc[
        bootstrap["metric"].astype(str).eq("rankic")
    ]
    if len(rankic_bootstrap) != 1 or "bootstrap_ci_low" not in rankic_bootstrap:
        raise ContractError("v42 requires one RankIC bootstrap row")
    rankic_ci_low = float(cast(Any, rankic_bootstrap["bootstrap_ci_low"].iloc[0]))
    if not np.isfinite(rankic_ci_low):
        raise ContractError("v42 RankIC bootstrap bound must be finite")

    per_seed = seed_fold_deltas.groupby("seed", observed=True)[
        "rankic_delta"
    ].mean()
    if set(per_seed.index.astype(int)) != {7, 17, 27}:
        raise ContractError("v42 per-seed evidence coverage drifted")
    broad_metric_count = sum(value > 0.0 for value in values.values())
    positive_units = int((unit_values > 0.0).sum())
    positive_horizons = int((horizon_values > 0.0).sum())
    minimum_seed_mean = float(per_seed.min())

    blockers: list[str] = []
    if values["mean_rankic_delta"] < 0.002:
        blockers.append("rankic_delta_below_gate")
    if positive_units < 9:
        blockers.append("positive_seed_fold_count_below_gate")
    if minimum_seed_mean < -0.001:
        blockers.append("per_seed_mean_below_gate")
    if rankic_ci_low < -0.001:
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
    if float(horizon_values.min()) < -0.002:
        blockers.append("worst_horizon_delta_below_gate")
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
        **values,
        "broad_metric_count": broad_metric_count,
        "positive_seed_fold_units": positive_units,
        "minimum_seed_mean_rankic_delta": minimum_seed_mean,
        "seed7_mean_rankic_delta": float(per_seed.loc[7]),
        "seed17_mean_rankic_delta": float(per_seed.loc[17]),
        "seed27_mean_rankic_delta": float(per_seed.loc[27]),
        "rankic_ci_low": rankic_ci_low,
        "positive_horizons": positive_horizons,
        "worst_horizon_rankic_delta": float(horizon_values.min()),
        "model_step_retention": model_step_retention,
        "complete_cycle_retention": complete_cycle_retention,
        "implied_tcn_lstm_model_step_ratio": implied_tcn_lstm_model_step_ratio,
        "inference_forward_passes": inference_forward_passes,
        "membership_turnover_is_model_gate": False,
        "sealed_test_accessed": False,
    }
    return ConsensusStudentDecision(
        status=(
            "consensus_student_multiseed_admitted_v42"
            if admitted
            else "stop_consensus_student_multiseed_v42"
        ),
        admitted=admitted,
        blockers=tuple(blockers),
        evidence=evidence,
    )
