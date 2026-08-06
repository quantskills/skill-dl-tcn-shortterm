"""Prospective utility-aligned gate for the once-only V46 test."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class V46IndependentDecision:
    """Orthogonal contract, generalization, and benchmark decision."""

    status: str
    admitted: bool
    blockers: tuple[str, ...]
    evidence: dict[str, float | int | bool]


def validate_v46_window_boundaries(
    *,
    evaluation_dates: pd.Series,
    training_dates: pd.Series,
    prior_consumed_end: str,
    embargo_end: str,
    expected_start: str,
    expected_end: str,
) -> dict[str, object]:
    """Fail closed unless a prospective window is disjoint and exact."""

    if evaluation_dates.empty or training_dates.empty:
        raise ContractError("v46 window boundary evidence cannot be empty")
    evaluation = pd.to_datetime(evaluation_dates.astype(str), errors="raise")
    training = pd.to_datetime(training_dates.astype(str), errors="raise")
    prior_end = pd.Timestamp(prior_consumed_end)
    embargo = pd.Timestamp(embargo_end)
    frozen_start = pd.Timestamp(expected_start)
    frozen_end = pd.Timestamp(expected_end)
    actual_start = evaluation.min()
    actual_end = evaluation.max()
    training_end = training.max()
    if actual_start <= embargo or actual_start <= prior_end:
        raise ContractError("v46 evaluation window overlaps prior consumption or embargo")
    if actual_start != frozen_start or actual_end != frozen_end:
        raise ContractError("v46 evaluation window does not match frozen boundaries")
    if training_end >= actual_start:
        raise ContractError("v46 training dates overlap the independent window")
    return {
        "prior_consumed_end": prior_end.date().isoformat(),
        "embargo_end": embargo.date().isoformat(),
        "evaluation_start": actual_start.date().isoformat(),
        "evaluation_end": actual_end.date().isoformat(),
        "training_max_date": training_end.date().isoformat(),
        "evaluation_date_count": int(evaluation.nunique()),
        "historical_replay": False,
        "contract_valid": True,
    }


def _bootstrap_low(frame: pd.DataFrame, metric: str, *, scope: str) -> float:
    required = {"metric", "bootstrap_ci_low"}
    if missing := sorted(required.difference(frame.columns)):
        raise ContractError(f"v46 {scope} bootstrap missing: {', '.join(missing)}")
    selected = frame.loc[frame["metric"].astype(str).eq(metric)]
    if len(selected) != 1:
        raise ContractError(f"v46 {scope} bootstrap metric {metric} is not unique")
    value = float(cast(Any, selected["bootstrap_ci_low"].iloc[0]))
    if not np.isfinite(value):
        raise ContractError(f"v46 {scope} bootstrap metric {metric} is not finite")
    return value


def decide_v46_independent_gate(
    student_control_comparison: Mapping[str, object],
    student_control_bootstrap: pd.DataFrame,
    student_lstm_bootstrap: pd.DataFrame,
    seed_deltas: pd.DataFrame,
    horizon_deltas: pd.DataFrame,
    *,
    contract_valid: bool,
    historical_replay: bool,
    model_step_speed_ratio: float,
    inference_forward_passes: int,
) -> V46IndependentDecision:
    """Apply the frozen V46 gates without gating on Top10% set overlap."""

    required_comparison = (
        "mean_rankic_delta",
        "mean_top_excess_return_delta",
        "mean_ndcg_at_top_delta",
    )
    try:
        comparison = {
            key: float(cast(Any, student_control_comparison[key]))
            for key in required_comparison
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v46 student/control comparison is incomplete") from exc
    numeric = np.asarray(
        [*comparison.values(), model_step_speed_ratio], dtype="float64"
    )
    if not np.isfinite(numeric).all():
        raise ContractError("v46 gate evidence must be finite")
    if (
        set(seed_deltas.columns) < {"seed", "rankic_delta"}
        or len(seed_deltas) != 3
        or set(seed_deltas["seed"].astype(int)) != {7, 17, 27}
        or seed_deltas["seed"].duplicated().any()
    ):
        raise ContractError("v46 seed deltas must cover 7/17/27 exactly once")
    if (
        set(horizon_deltas.columns) < {"horizon", "rankic_delta"}
        or len(horizon_deltas) != 4
        or set(horizon_deltas["horizon"].astype(int)) != {1, 2, 3, 5}
        or horizon_deltas["horizon"].duplicated().any()
    ):
        raise ContractError("v46 horizon deltas must cover 1/2/3/5 exactly once")
    seed_values = seed_deltas["rankic_delta"].to_numpy(dtype="float64")
    horizon_values = horizon_deltas["rankic_delta"].to_numpy(dtype="float64")
    if not np.isfinite(seed_values).all() or not np.isfinite(horizon_values).all():
        raise ContractError("v46 breadth evidence must be finite")

    control_rankic_ci_low = _bootstrap_low(
        student_control_bootstrap, "rankic", scope="student/control"
    )
    lstm_rankic_ci_low = _bootstrap_low(
        student_lstm_bootstrap, "rankic", scope="student/LSTM"
    )
    lstm_top_excess_ci_low = _bootstrap_low(
        student_lstm_bootstrap,
        "top_excess_return",
        scope="student/LSTM",
    )
    lstm_ndcg_ci_low = _bootstrap_low(
        student_lstm_bootstrap, "ndcg_at_top", scope="student/LSTM"
    )

    contract_blockers: list[str] = []
    if not contract_valid:
        contract_blockers.append("independent_contract_invalid")
    if historical_replay:
        contract_blockers.append("historical_replay_cannot_admit")

    generalization_blockers: list[str] = []
    if comparison["mean_rankic_delta"] < 0.0:
        generalization_blockers.append("control_rankic_delta_below_gate")
    if control_rankic_ci_low < -0.002:
        generalization_blockers.append("control_rankic_ci_low_below_gate")
    positive_seeds = int((seed_values > 0.0).sum())
    if positive_seeds < 2:
        generalization_blockers.append("control_positive_seed_count_below_gate")
    positive_horizons = int((horizon_values > 0.0).sum())
    if positive_horizons < 3:
        generalization_blockers.append("control_positive_horizon_count_below_gate")
    if comparison["mean_top_excess_return_delta"] < -0.0001:
        generalization_blockers.append("control_top_excess_return_delta_below_gate")
    if comparison["mean_ndcg_at_top_delta"] < -0.001:
        generalization_blockers.append("control_ndcg_at_top_delta_below_gate")

    benchmark_blockers: list[str] = []
    if lstm_rankic_ci_low < -0.01:
        benchmark_blockers.append("lstm_rankic_noninferiority_failed")
    if lstm_top_excess_ci_low < -0.0005:
        benchmark_blockers.append("lstm_top_excess_noninferiority_failed")
    if lstm_ndcg_ci_low < -0.01:
        benchmark_blockers.append("lstm_ndcg_noninferiority_failed")
    if model_step_speed_ratio < 3.0:
        benchmark_blockers.append("tcn_lstm_speed_below_gate")
    if inference_forward_passes != 1:
        benchmark_blockers.append("single_model_inference_gate_failed")

    blockers = tuple(
        dict.fromkeys(
            [*contract_blockers, *generalization_blockers, *benchmark_blockers]
        )
    )
    if contract_blockers:
        status = "v46_contract_failed"
    elif generalization_blockers:
        status = "v46_student_not_generalized"
    elif benchmark_blockers:
        status = "v46_student_generalized_but_not_lstm_competitive"
    else:
        status = "v46_independent_research_candidate"

    membership_delta = student_control_comparison.get(
        "mean_top_membership_precision_delta",
        student_control_comparison.get("mean_top_precision_delta", float("nan")),
    )
    evidence: dict[str, float | int | bool] = {
        **comparison,
        "control_rankic_ci_low": control_rankic_ci_low,
        "positive_seed_count": positive_seeds,
        "positive_horizon_count": positive_horizons,
        "lstm_rankic_ci_low": lstm_rankic_ci_low,
        "lstm_top_excess_return_ci_low": lstm_top_excess_ci_low,
        "lstm_ndcg_at_top_ci_low": lstm_ndcg_ci_low,
        "model_step_speed_ratio": model_step_speed_ratio,
        "inference_forward_passes": inference_forward_passes,
        "contract_valid": contract_valid,
        "historical_replay": historical_replay,
        "top_membership_precision_is_gate": False,
    }
    try:
        membership_value = float(cast(Any, membership_delta))
    except (TypeError, ValueError):
        membership_value = float("nan")
    if np.isfinite(membership_value):
        evidence["mean_top_membership_precision_delta"] = membership_value
    return V46IndependentDecision(
        status=status,
        admitted=not blockers,
        blockers=blockers,
        evidence=evidence,
    )
