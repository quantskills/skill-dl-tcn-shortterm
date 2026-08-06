"""Model/portfolio boundary decisions for the ordinary-validation v40 replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .experiment import ContractError
from .relative_validation import RelativeFeatureDecision, decide_relative_feature_gate


@dataclass(frozen=True)
class StrategyGateDecision:
    """Ordinary-validation portfolio research decision, separate from model admission."""

    status: str
    admitted: bool
    blockers: tuple[str, ...]
    evidence: dict[str, float | int | bool | str]
    unit_deltas: pd.DataFrame


def frozen_training_contracts(
    predictions: pd.DataFrame, *, expected_models: Sequence[str]
) -> dict[str, str]:
    """Resolve one immutable training contract per frozen parent model."""

    required = {"model", "training_contract_id"}
    if missing := sorted(required.difference(predictions.columns)):
        raise ContractError("v40 parent predictions missing: " + ", ".join(missing))
    contracts: dict[str, str] = {}
    for model in expected_models:
        values = {
            str(value)
            for value in predictions.loc[
                predictions["model"].astype(str).eq(str(model)),
                "training_contract_id",
            ]
            if str(value)
        }
        if len(values) != 1:
            raise ContractError(f"v40 parent model has multiple training contracts: {model}")
        contracts[str(model)] = next(iter(values))
    return contracts


def validate_v40_frozen_predictions(
    predictions: pd.DataFrame,
    *,
    expected_models: Sequence[str],
    expected_seeds: Sequence[int],
    expected_folds: Sequence[int],
    expected_horizons: Sequence[int],
) -> None:
    """Fail closed unless a frozen ordinary-validation prediction panel is exact."""

    required = {
        "model",
        "seed",
        "fold",
        "sample_id",
        "instrument_id",
        "signal_date",
        "horizon",
        "score",
        "stage",
        "sealed",
    }
    if missing := sorted(required.difference(predictions.columns)):
        raise ContractError("v40 frozen predictions missing: " + ", ".join(missing))
    if predictions.empty:
        raise ContractError("v40 frozen predictions are empty")
    if not predictions["stage"].astype(str).eq("validation").all() or predictions[
        "sealed"
    ].astype(bool).any():
        raise ContractError("v40 requires non-sealed validation predictions")
    observed_models = set(predictions["model"].astype(str))
    observed_seeds = set(predictions["seed"].astype(int))
    observed_folds = set(predictions["fold"].astype(int))
    observed_horizons = set(predictions["horizon"].astype(int))
    if observed_models != set(expected_models):
        raise ContractError("v40 frozen prediction model coverage drifted")
    if observed_seeds != {int(value) for value in expected_seeds}:
        raise ContractError("v40 frozen prediction seed coverage drifted")
    if observed_folds != {int(value) for value in expected_folds}:
        raise ContractError("v40 frozen prediction fold coverage drifted")
    if observed_horizons != {int(value) for value in expected_horizons}:
        raise ContractError("v40 frozen prediction horizon coverage drifted")
    identity = ["model", "seed", "fold", "sample_id", "horizon"]
    if predictions.duplicated(identity).any():
        raise ContractError("v40 frozen predictions contain duplicate identities")
    if not np.isfinite(predictions["score"].to_numpy(dtype="float64")).all():
        raise ContractError("v40 frozen predictions contain non-finite scores")
    panel_keys = ["seed", "fold", "sample_id", "instrument_id", "signal_date", "horizon"]
    reference: set[tuple[object, ...]] | None = None
    for model in expected_models:
        keys = set(
            predictions.loc[predictions["model"].astype(str).eq(str(model)), panel_keys]
            .itertuples(index=False, name=None)
        )
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ContractError("v40 frozen prediction cross-model panel drifted")


def decide_v40_model_gate(
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
    admitted_status: str = "top50_relative_model_seed7_admitted_v40",
    rejected_status: str = "stop_top50_relative_model_seed7_v40",
) -> RelativeFeatureDecision:
    """Apply predictive gates while retaining membership churn as a diagnostic.

    Membership turnover is a property of adjacent Top-K sets. It is deliberately
    excluded from the predictive-model admission decision; executable turnover is
    evaluated later from security-level target-weight changes.
    """

    membership_delta = float(cast(Any, task_comparison["mean_top_turnover_delta"]))
    if not np.isfinite(membership_delta):
        raise ContractError("v40 membership turnover diagnostic must be finite")
    legacy_gates = dict(gates)
    legacy_gates["max_mean_turnover_delta"] = 0.0
    decision = decide_relative_feature_gate(
        tcn_leaderboard,
        task_comparison,
        bootstrap,
        seeds=seeds,
        folds=folds,
        base_variant=base_variant,
        candidate_variant=candidate_variant,
        base_median_samples_per_second=base_median_samples_per_second,
        candidate_median_samples_per_second=candidate_median_samples_per_second,
        gates=legacy_gates,
        reference_model=reference_model,
        candidate_model=candidate_model,
        admitted_status=admitted_status,
        rejected_status=rejected_status,
        enforce_membership_turnover_gate=False,
    )
    evidence = dict(decision.evidence)
    evidence["membership_turnover_diagnostic_delta"] = membership_delta
    evidence["membership_turnover_is_model_gate"] = False
    return RelativeFeatureDecision(
        status=decision.status,
        admitted=decision.admitted,
        blockers=decision.blockers,
        evidence=evidence,
        unit_deltas=decision.unit_deltas,
    )


def summarize_cost_sensitivity(
    result: BacktestResult,
    *,
    cost_bps: Sequence[float],
) -> pd.DataFrame:
    """Price scheduled buys and sells at an explicit per-side cost rate."""

    if not cost_bps:
        raise ContractError("v40 cost sensitivity requires at least one scenario")
    scenarios = tuple(float(value) for value in cost_bps)
    if not np.isfinite(scenarios).all() or any(value < 0.0 for value in scenarios):
        raise ContractError("v40 cost scenarios must be finite and non-negative")
    keys = ["model", "fold", "horizon"]
    required_ledger = {*keys, "traded_notional_turnover"}
    required_metrics = {
        *keys,
        "cumulative_gross_return_contribution",
        "cumulative_pit_equal_weight_return_contribution",
        "cumulative_excess_return_contribution",
        "max_drawdown",
        "vintage_count",
    }
    if missing := sorted(required_ledger.difference(result.portfolio_ledger.columns)):
        raise ContractError("v40 portfolio ledger missing: " + ", ".join(missing))
    if missing := sorted(required_metrics.difference(result.metrics.columns)):
        raise ContractError("v40 portfolio metrics missing: " + ", ".join(missing))
    turnover = result.portfolio_ledger.groupby(
        keys, as_index=False, observed=True
    ).agg(
        cumulative_traded_notional_turnover=(
            "traded_notional_turnover",
            "sum",
        )
    )
    metric_base = result.metrics.drop(
        columns=["cumulative_traded_notional_turnover"], errors="ignore"
    )
    base = metric_base.merge(turnover, on=keys, how="inner", validate="one_to_one")
    rows: list[pd.DataFrame] = []
    for scenario in scenarios:
        priced = base.copy()
        priced["one_way_cost_bps"] = scenario
        priced["transaction_cost"] = (
            priced["cumulative_traded_notional_turnover"] * scenario / 10_000.0
        )
        priced["cumulative_net_return_contribution"] = (
            priced["cumulative_gross_return_contribution"]
            - priced["transaction_cost"]
        )
        priced["mean_net_return_contribution"] = (
            priced["cumulative_net_return_contribution"] / priced["vintage_count"]
        )
        priced["cumulative_net_excess_vs_gross_pit_benchmark"] = (
            priced["cumulative_net_return_contribution"]
            - priced["cumulative_pit_equal_weight_return_contribution"]
        )
        rows.append(priced)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["model", "fold", "horizon", "one_way_cost_bps"], ignore_index=True
    )


def decide_v40_strategy_gate(
    cost_summary: pd.DataFrame,
    *,
    policy: str,
    reference_model: str,
    candidate_model: str,
    one_way_cost_bps: float,
    max_mean_one_way_turnover_delta: float,
    min_mean_net_return_delta: float,
) -> StrategyGateDecision:
    """Gate portfolio research on executable weights and explicit transaction cost."""

    required = {
        "policy",
        "model",
        "fold",
        "horizon",
        "one_way_cost_bps",
        "mean_one_way_turnover",
        "mean_net_return_contribution",
    }
    if missing := sorted(required.difference(cost_summary.columns)):
        raise ContractError("v40 strategy summary missing: " + ", ".join(missing))
    selected = cost_summary.loc[
        cost_summary["policy"].astype(str).eq(policy)
        & np.isclose(
            cost_summary["one_way_cost_bps"].astype(float), one_way_cost_bps
        )
        & cost_summary["model"].astype(str).isin(
            [reference_model, candidate_model]
        )
    ].copy()
    unit_columns = ["fold", "horizon"]
    if "seed" in selected.columns:
        unit_columns.insert(0, "seed")
    if selected.empty or selected.duplicated(["model", *unit_columns]).any():
        raise ContractError("v40 strategy gate has missing or duplicate units")
    turnover = selected.pivot(
        index=unit_columns, columns="model", values="mean_one_way_turnover"
    )
    net_return = selected.pivot(
        index=unit_columns, columns="model", values="mean_net_return_contribution"
    )
    expected_columns = {reference_model, candidate_model}
    if set(turnover.columns) != expected_columns or set(net_return.columns) != (
        expected_columns
    ) or not turnover.index.equals(net_return.index):
        raise ContractError("v40 strategy model coverage drifted")
    unit_deltas = turnover.reset_index()[unit_columns].copy()
    unit_deltas["one_way_turnover_delta"] = (
        turnover[candidate_model] - turnover[reference_model]
    ).to_numpy()
    unit_deltas["net_return_delta"] = (
        net_return[candidate_model] - net_return[reference_model]
    ).to_numpy()
    values = unit_deltas[["one_way_turnover_delta", "net_return_delta"]].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(values).all():
        raise ContractError("v40 strategy gate contains non-finite evidence")
    mean_turnover_delta = float(unit_deltas["one_way_turnover_delta"].mean())
    mean_net_return_delta = float(unit_deltas["net_return_delta"].mean())
    blockers: list[str] = []
    if mean_turnover_delta > max_mean_one_way_turnover_delta:
        blockers.append("executable_turnover_delta_above_gate")
    if mean_net_return_delta < min_mean_net_return_delta:
        blockers.append("net_return_delta_below_gate")
    admitted = not blockers
    return StrategyGateDecision(
        status=(
            "portfolio_research_admitted_v40"
            if admitted
            else "portfolio_research_not_admitted_v40"
        ),
        admitted=admitted,
        blockers=tuple(blockers),
        evidence={
            "policy": policy,
            "one_way_cost_bps": one_way_cost_bps,
            "unit_count": len(unit_deltas),
            "mean_one_way_turnover_delta": mean_turnover_delta,
            "mean_net_return_delta": mean_net_return_delta,
            "membership_turnover_used": False,
            "sealed_test_accessed": False,
        },
        unit_deltas=unit_deltas,
    )
