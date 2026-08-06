"""Pre-registered v37 decision logic for the relative-feature TCN probe."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class RelativeFeatureDecision:
    """Admission result for candidate features on ordinary validation only."""

    status: str
    admitted: bool
    blockers: tuple[str, ...]
    evidence: dict[str, float | int | bool]
    unit_deltas: pd.DataFrame


def audit_validation_effective_breadth(
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    folds: Sequence[int],
    top_fraction: float,
    min_top_count: int,
) -> dict[str, float | int | bool]:
    """Audit valid ordinary-validation cross sections before model training."""

    if not 0 < top_fraction <= 0.5 or min_top_count <= 0:
        raise ContractError("effective breadth thresholds are invalid")
    required_labels = {"sample_position", "signal_date", "horizon", "valid"}
    required_split = {"sample_position", "fold", "stage", "sealed"}
    if missing := sorted(required_labels.difference(labels.columns)):
        raise ContractError("breadth labels missing columns: " + ", ".join(missing))
    if missing := sorted(required_split.difference(split_manifest.columns)):
        raise ContractError("breadth split missing columns: " + ", ".join(missing))
    selected = split_manifest.loc[
        split_manifest["fold"].astype(int).isin(folds)
        & split_manifest["stage"].astype(str).eq("validation")
    ].copy()
    if selected.empty or selected["sealed"].astype(bool).any():
        raise ContractError("breadth audit requires non-sealed validation rows")
    if selected.duplicated(["fold", "sample_position"]).any():
        raise ContractError("breadth split contains duplicate validation positions")
    valid_labels = labels.loc[labels["valid"].astype(bool)].copy()
    joined = selected[["fold", "sample_position"]].merge(
        valid_labels[["sample_position", "signal_date", "horizon"]],
        on="sample_position",
        how="inner",
        validate="one_to_many",
    )
    if joined.empty:
        raise ContractError("breadth audit has no valid validation labels")
    counts = joined.groupby(
        ["fold", "signal_date", "horizon"], observed=True, sort=True
    ).size()
    top_counts = counts.map(lambda value: max(1, int(ceil(value * top_fraction))))
    expected_folds = {int(value) for value in folds}
    if set(joined["fold"].astype(int)) != expected_folds:
        raise ContractError("breadth audit fold coverage drifted")
    minimum_member_count = int(counts.min())
    minimum_observed_top_count = int(top_counts.min())
    return {
        "group_count": int(len(counts)),
        "minimum_member_count": minimum_member_count,
        "maximum_member_count": int(counts.max()),
        "minimum_top_count": minimum_observed_top_count,
        "required_minimum_top_count": min_top_count,
        "effective_breadth_gate_passed": minimum_observed_top_count
        >= min_top_count,
    }


def decide_relative_feature_gate(
    tcn_leaderboard: pd.DataFrame,
    task_comparison: Mapping[str, object],
    bootstrap: pd.DataFrame,
    *,
    seeds: Sequence[int],
    folds: Sequence[int],
    base_variant: str,
    candidate_variant: str,
    base_median_samples_per_second: float,
    candidate_median_samples_per_second: float,
    gates: Mapping[str, float | int],
    reference_model: str = "base_tcn",
    candidate_model: str = "relative_tcn",
    admitted_status: str = "relative_features_admitted_v37",
    rejected_status: str = "stop_relative_features_no_stable_gain_v37",
    enforce_membership_turnover_gate: bool = True,
) -> RelativeFeatureDecision:
    """Apply the immutable multiseed representation gate without model shopping."""

    required_columns = {"variant", "seed", "fold", "best_mean_daily_rankic"}
    if missing := sorted(required_columns.difference(tcn_leaderboard.columns)):
        raise ContractError("v37 TCN leaderboard missing columns: " + ", ".join(missing))
    expected_units = {
        (int(seed), int(fold)) for seed in seeds for fold in folds
    }
    selected = tcn_leaderboard.loc[
        tcn_leaderboard["variant"].astype(str).isin(
            [base_variant, candidate_variant]
        )
    ].copy()
    if selected.duplicated(["variant", "seed", "fold"]).any():
        raise ContractError("v37 TCN leaderboard contains duplicate variant units")
    pivot = selected.pivot(
        index=["seed", "fold"],
        columns="variant",
        values="best_mean_daily_rankic",
    )
    if set(pivot.index.tolist()) != expected_units or set(pivot.columns) != {
        base_variant,
        candidate_variant,
    }:
        raise ContractError("v37 TCN seed/fold coverage drifted")
    if not np.isfinite(pivot.to_numpy(dtype="float64")).all():
        raise ContractError("v37 TCN leaderboard contains non-finite RankIC")
    unit_deltas = pivot.reset_index()
    unit_deltas["rankic_delta"] = (
        unit_deltas[candidate_variant] - unit_deltas[base_variant]
    )

    required_comparison = {
        "mean_rankic_delta",
        "mean_top_precision_delta",
        "mean_ndcg_at_top_delta",
        "mean_top_return_delta",
        "mean_top_turnover_delta",
    }
    if missing := sorted(required_comparison.difference(task_comparison)):
        raise ContractError("v37 task comparison missing fields: " + ", ".join(missing))
    comparison_values = {
        key: float(cast(Any, task_comparison[key])) for key in required_comparison
    }
    if not np.isfinite(list(comparison_values.values())).all():
        raise ContractError("v37 task comparison contains non-finite values")
    rankic_rows = bootstrap.loc[
        bootstrap["metric"].astype(str).eq("rankic")
        & bootstrap["reference_model"].astype(str).eq(reference_model)
        & bootstrap["candidate_model"].astype(str).eq(candidate_model)
    ]
    if len(rankic_rows) != 1:
        raise ContractError("v37 RankIC bootstrap row is missing or ambiguous")
    rankic_ci_low = float(rankic_rows.iloc[0]["bootstrap_ci_low"])
    speed_retention = candidate_median_samples_per_second / base_median_samples_per_second
    values = np.asarray(
        [
            rankic_ci_low,
            base_median_samples_per_second,
            candidate_median_samples_per_second,
            speed_retention,
        ],
        dtype="float64",
    )
    if not np.isfinite(values).all() or bool((values[1:3] <= 0).any()):
        raise ContractError("v37 speed or bootstrap evidence is invalid")

    mean_rankic_delta = float(unit_deltas["rankic_delta"].mean())
    positive_units = int(unit_deltas["rankic_delta"].gt(0).sum())
    blockers: list[str] = []
    if mean_rankic_delta < float(gates["min_mean_rankic_delta"]):
        blockers.append("mean_rankic_delta_below_gate")
    if positive_units < int(gates["min_positive_units"]):
        blockers.append("positive_units_below_gate")
    if comparison_values["mean_top_precision_delta"] < float(
        gates["min_mean_top_precision_delta"]
    ):
        blockers.append("top_precision_delta_below_gate")
    if comparison_values["mean_ndcg_at_top_delta"] < float(
        gates["min_mean_ndcg_delta"]
    ):
        blockers.append("ndcg_delta_below_gate")
    if comparison_values["mean_top_return_delta"] < float(
        gates["min_mean_top_return_delta"]
    ):
        blockers.append("top_return_delta_below_gate")
    if enforce_membership_turnover_gate and comparison_values[
        "mean_top_turnover_delta"
    ] > float(gates["max_mean_turnover_delta"]):
        blockers.append("turnover_delta_above_gate")
    if rankic_ci_low < float(gates["min_rankic_ci_low"]):
        blockers.append("rankic_ci_low_below_gate")
    if speed_retention < float(gates["min_tcn_speed_retention"]):
        blockers.append("tcn_speed_retention_below_gate")

    admitted = not blockers
    evidence: dict[str, float | int | bool] = {
        "unit_count": len(unit_deltas),
        "mean_rankic_delta": mean_rankic_delta,
        "positive_units": positive_units,
        "mean_top_precision_delta": comparison_values[
            "mean_top_precision_delta"
        ],
        "mean_ndcg_delta": comparison_values["mean_ndcg_at_top_delta"],
        "mean_top_return_delta": comparison_values["mean_top_return_delta"],
        "mean_turnover_delta": comparison_values["mean_top_turnover_delta"],
        "rankic_ci_low": rankic_ci_low,
        "base_median_samples_per_second": base_median_samples_per_second,
        "candidate_median_samples_per_second": candidate_median_samples_per_second,
        "tcn_speed_retention": speed_retention,
        "sealed_test_accessed": False,
    }
    return RelativeFeatureDecision(
        status=admitted_status if admitted else rejected_status,
        admitted=admitted,
        blockers=tuple(blockers),
        evidence=evidence,
        unit_deltas=unit_deltas,
    )
