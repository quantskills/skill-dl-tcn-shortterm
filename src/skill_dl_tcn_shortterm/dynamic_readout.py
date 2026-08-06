"""Capacity- and mechanism-aware v18 dynamic-readout decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError
from .real_validation import evaluate_stabilized_residual_seed7


@dataclass(frozen=True)
class DynamicReadoutSeed7Decision:
    """Effect decision for the bounded v18 dynamic-readout screen."""

    status: str
    winner_trial_id: str | None
    summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class FinalDynamicReadoutSeed7Decision:
    """v18 effect decision combined with fixed LSTM speed gates."""

    status: str
    winner_trial_id: str | None
    relative_speed_gate_passed: bool
    confirmation_seeds_authorized: tuple[int, ...]


@dataclass(frozen=True)
class DynamicLRSeed7Decision:
    """Effect and mechanism decision for the bounded v19 optimizer screen."""

    status: str
    winner_trial_id: str | None
    summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class FinalDynamicLRSeed7Decision:
    """v19 decision combined with fixed LSTM relative-speed gates."""

    status: str
    winner_trial_id: str | None
    relative_speed_gate_passed: bool
    confirmation_seeds_authorized: tuple[int, ...]


def evaluate_dynamic_readout_seed7(
    leaderboard: pd.DataFrame,
    attention_diagnostics: pd.DataFrame,
    *,
    control_trial_id: str,
    candidate_trial_id: str,
    min_mean_rankic: float,
    min_mean_rankic_delta: float,
    min_positive_folds: int,
    min_nondegrading_folds: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    min_dynamic_attention_output_l2: float,
    min_dynamic_weight_variation: float,
    control_parameter_count: int,
    candidate_parameter_count: int,
) -> DynamicReadoutSeed7Decision:
    """Select v18 only after effect, capacity, use and sample-variation gates."""

    required = {"trial_id", "parameter_count", "dynamic_attention_output_l2"}
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            "v18 dynamic-readout leaderboard missing columns: " + ", ".join(missing)
        )
    diagnostic_required = {
        "trial_id",
        "fold",
        "day_weight_variation",
        "intraday_weight_variation",
    }
    if missing := sorted(diagnostic_required.difference(attention_diagnostics.columns)):
        raise ContractError(
            "v18 attention diagnostics missing columns: " + ", ".join(missing)
        )
    if candidate_parameter_count - control_parameter_count != 176:
        raise ContractError("v18 candidate capacity delta must be exactly 176")
    thresholds = np.asarray(
        [
            min_mean_rankic_delta,
            min_dynamic_attention_output_l2,
            min_dynamic_weight_variation,
        ],
        dtype="float64",
    )
    if not np.isfinite(thresholds).all() or bool((thresholds <= 0).any()):
        raise ContractError("v18 mechanism and delta gates must be positive")
    if set(leaderboard["trial_id"].astype(str)) != {
        control_trial_id,
        candidate_trial_id,
    }:
        raise ContractError("v18 dynamic-readout trial identities drifted")
    control_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ]
    candidate_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
    ]
    if set(control_rows["parameter_count"].astype(int)) != {control_parameter_count}:
        raise ContractError("v18 control parameter count drifted")
    if set(candidate_rows["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v18 candidate parameter count drifted")
    output_l2 = candidate_rows["dynamic_attention_output_l2"].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(output_l2).all():
        raise ContractError("v18 dynamic attention use evidence is non-finite")

    diagnostics = attention_diagnostics.loc[
        attention_diagnostics["trial_id"].astype(str).eq(candidate_trial_id)
    ].copy()
    if (
        len(diagnostics) != 5
        or set(diagnostics["fold"].astype(int)) != set(range(5))
        or diagnostics.duplicated(["trial_id", "fold"]).any()
    ):
        raise ContractError("v18 attention diagnostics must cover folds 0 through 4")
    variation_values = diagnostics[
        ["day_weight_variation", "intraday_weight_variation"]
    ].to_numpy(dtype="float64")
    if not np.isfinite(variation_values).all() or bool((variation_values < 0).any()):
        raise ContractError("v18 attention variation evidence is invalid")
    fold_variation = variation_values.max(axis=1)

    normalized = leaderboard.copy()
    normalized["parameter_count"] = candidate_parameter_count
    base = evaluate_stabilized_residual_seed7(
        normalized,
        control_trial_id=control_trial_id,
        candidate_trial_ids=(candidate_trial_id,),
        min_mean_rankic=min_mean_rankic,
        min_positive_folds=min_positive_folds,
        min_nondegrading_folds=min_nondegrading_folds,
        min_horizon_delta_1d=min_horizon_delta_1d,
        min_horizon_delta_2d=min_horizon_delta_2d,
        min_horizon_delta_3d=min_horizon_delta_3d,
        min_horizon_delta_5d=min_horizon_delta_5d,
        min_median_samples_per_second=min_median_samples_per_second,
        required_parameter_count=candidate_parameter_count,
    )
    summary = base.summary.copy()
    is_control = summary["trial_id"].astype(str).eq(control_trial_id)
    summary["parameter_count"] = np.where(
        is_control, control_parameter_count, candidate_parameter_count
    )
    summary["capacity_delta"] = summary["parameter_count"].astype(int) - int(
        control_parameter_count
    )
    summary["dynamic_attention_output_l2_min"] = np.where(
        is_control, np.nan, float(output_l2.min())
    )
    summary["dynamic_weight_variation_min"] = np.where(
        is_control, np.nan, float(fold_variation.min())
    )
    candidate_index = summary.index[
        summary["trial_id"].astype(str).eq(candidate_trial_id)
    ]
    if len(candidate_index) != 1:
        raise ContractError("v18 candidate summary is incomplete")
    row_index = int(candidate_index[0])
    blockers = [
        value for value in str(summary.at[row_index, "blockers"]).split(",") if value
    ]
    if (
        float(cast(Any, summary.at[row_index, "mean_rankic_delta"]))
        < min_mean_rankic_delta
    ):
        blockers.append("mean_rankic_delta_below_gate")
    if float(output_l2.min()) < min_dynamic_attention_output_l2:
        blockers.append("dynamic_attention_not_used")
    if float(fold_variation.min()) < min_dynamic_weight_variation:
        blockers.append("dynamic_weights_not_sample_conditioned")
    blockers = list(dict.fromkeys(blockers))
    summary.at[row_index, "blockers"] = ",".join(blockers)
    summary.at[row_index, "eligible"] = not blockers
    summary = summary.sort_values(
        ["eligible", "mean_rankic", "worst_fold_rankic", "trial_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
        ignore_index=True,
    )
    winner = (
        candidate_trial_id
        if bool(
            summary.loc[
                summary["trial_id"].astype(str).eq(candidate_trial_id), "eligible"
            ].iloc[0]
        )
        else None
    )
    return DynamicReadoutSeed7Decision(
        status=(
            "dynamic_readout_seed7_effect_admitted_v18"
            if winner is not None
            else "stop_dynamic_readout_seed7_effect_v18"
        ),
        winner_trial_id=winner,
        summary=summary,
        horizon_summary=base.horizon_summary,
    )


def finalize_dynamic_readout_seed7(
    effect_decision: DynamicReadoutSeed7Decision,
    comparison: dict[str, float | int],
    *,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FinalDynamicReadoutSeed7Decision:
    """Authorize confirmation only after the v18 effect and speed gates."""

    if effect_decision.winner_trial_id is None:
        return FinalDynamicReadoutSeed7Decision(
            status="stop_dynamic_readout_seed7_effect_v18",
            winner_trial_id=None,
            relative_speed_gate_passed=False,
            confirmation_seeds_authorized=(),
        )
    try:
        model_step_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v18 comparison is missing relative speed") from exc
    if (
        not np.isfinite([model_step_ratio, end_to_end_ratio]).all()
        or min_model_step_speed_ratio <= 0
        or min_end_to_end_speed_ratio <= 0
    ):
        raise ContractError("v18 relative speed values and gates must be positive")
    speed_passed = bool(
        model_step_ratio >= min_model_step_speed_ratio
        and end_to_end_ratio >= min_end_to_end_speed_ratio
    )
    return FinalDynamicReadoutSeed7Decision(
        status=(
            "dynamic_readout_seed7_admitted_v18"
            if speed_passed
            else "stop_dynamic_readout_seed7_speed_v18"
        ),
        winner_trial_id=(effect_decision.winner_trial_id if speed_passed else None),
        relative_speed_gate_passed=speed_passed,
        confirmation_seeds_authorized=((17, 27) if speed_passed else ()),
    )


def evaluate_dynamic_lr_seed7(
    leaderboard: pd.DataFrame,
    attention_diagnostics: pd.DataFrame,
    parent_attention_diagnostics: pd.DataFrame,
    *,
    control_trial_id: str,
    candidate_trial_id: str,
    min_mean_rankic: float,
    min_mean_rankic_delta: float,
    min_positive_folds: int,
    min_nondegrading_folds: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    min_dynamic_attention_output_weight_l2: float,
    min_dynamic_weight_variation: float,
    min_parent_variation_ratio: float,
    control_parameter_count: int,
    candidate_parameter_count: int,
    dynamic_parameter_count: int,
) -> DynamicLRSeed7Decision:
    """Select v19 only after paired parent-variation and v18 effect gates."""

    required = {
        "trial_id",
        "dynamic_attention_output_weight_l2",
        "optimizer_dynamic_attention_parameter_count",
    }
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            "v19 dynamic-LR leaderboard missing columns: " + ", ".join(missing)
        )
    if dynamic_parameter_count != 176:
        raise ContractError("v19 dynamic parameter count gate must be exactly 176")
    if not np.isfinite(min_parent_variation_ratio) or min_parent_variation_ratio <= 1:
        raise ContractError("v19 parent variation ratio gate must exceed one")
    candidate_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
    ]
    if set(
        candidate_rows["optimizer_dynamic_attention_parameter_count"].astype(int)
    ) != {dynamic_parameter_count}:
        raise ContractError("v19 optimizer dynamic parameter count drifted")
    output_weight_l2 = candidate_rows[
        "dynamic_attention_output_weight_l2"
    ].to_numpy(dtype="float64")
    if not np.isfinite(output_weight_l2).all():
        raise ContractError("v19 dynamic output weight evidence is non-finite")

    normalized = leaderboard.copy()
    normalized["dynamic_attention_output_l2"] = normalized[
        "dynamic_attention_output_weight_l2"
    ]
    base = evaluate_dynamic_readout_seed7(
        normalized,
        attention_diagnostics,
        control_trial_id=control_trial_id,
        candidate_trial_id=candidate_trial_id,
        min_mean_rankic=min_mean_rankic,
        min_mean_rankic_delta=min_mean_rankic_delta,
        min_positive_folds=min_positive_folds,
        min_nondegrading_folds=min_nondegrading_folds,
        min_horizon_delta_1d=min_horizon_delta_1d,
        min_horizon_delta_2d=min_horizon_delta_2d,
        min_horizon_delta_3d=min_horizon_delta_3d,
        min_horizon_delta_5d=min_horizon_delta_5d,
        min_median_samples_per_second=min_median_samples_per_second,
        min_dynamic_attention_output_l2=min_dynamic_attention_output_weight_l2,
        min_dynamic_weight_variation=min_dynamic_weight_variation,
        control_parameter_count=control_parameter_count,
        candidate_parameter_count=candidate_parameter_count,
    )

    diagnostic_columns = {
        "fold",
        "day_weight_variation",
        "intraday_weight_variation",
    }
    for name, diagnostics in (
        ("candidate", attention_diagnostics),
        ("parent", parent_attention_diagnostics),
    ):
        if missing := sorted(diagnostic_columns.difference(diagnostics.columns)):
            raise ContractError(
                f"v19 {name} attention diagnostics missing columns: "
                + ", ".join(missing)
            )
        if len(diagnostics) != 5 or set(diagnostics["fold"].astype(int)) != set(
            range(5)
        ):
            raise ContractError(
                f"v19 {name} attention diagnostics must cover folds 0 through 4"
            )
    current = attention_diagnostics.sort_values("fold")
    parent = parent_attention_diagnostics.sort_values("fold")
    current_variation = current[
        ["day_weight_variation", "intraday_weight_variation"]
    ].to_numpy(dtype="float64").max(axis=1)
    parent_variation = parent[
        ["day_weight_variation", "intraday_weight_variation"]
    ].to_numpy(dtype="float64").max(axis=1)
    if (
        not np.isfinite(current_variation).all()
        or not np.isfinite(parent_variation).all()
        or bool((current_variation < 0).any())
        or bool((parent_variation <= 0).any())
    ):
        raise ContractError("v19 paired attention variation evidence is invalid")
    variation_ratios = current_variation / parent_variation
    median_ratio = float(np.median(variation_ratios))

    summary = base.summary.copy()
    candidate_index = summary.index[
        summary["trial_id"].astype(str).eq(candidate_trial_id)
    ]
    if len(candidate_index) != 1:
        raise ContractError("v19 candidate summary is incomplete")
    row_index = int(candidate_index[0])
    blockers = [
        value for value in str(summary.at[row_index, "blockers"]).split(",") if value
    ]
    if median_ratio < min_parent_variation_ratio:
        blockers.append("parent_variation_ratio_below_gate")
    blockers = list(dict.fromkeys(blockers))
    summary.at[row_index, "blockers"] = ",".join(blockers)
    summary.at[row_index, "eligible"] = not blockers
    summary["parent_variation_ratio_median"] = np.where(
        summary["trial_id"].astype(str).eq(candidate_trial_id), median_ratio, np.nan
    )
    summary["dynamic_attention_output_weight_l2_min"] = np.where(
        summary["trial_id"].astype(str).eq(candidate_trial_id),
        float(output_weight_l2.min()),
        np.nan,
    )
    summary = summary.sort_values(
        ["eligible", "mean_rankic", "worst_fold_rankic", "trial_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
        ignore_index=True,
    )
    winner = (
        candidate_trial_id
        if bool(
            summary.loc[
                summary["trial_id"].astype(str).eq(candidate_trial_id), "eligible"
            ].iloc[0]
        )
        else None
    )
    return DynamicLRSeed7Decision(
        status=(
            "dynamic_lr_seed7_effect_admitted_v19"
            if winner is not None
            else "stop_dynamic_lr_seed7_effect_v19"
        ),
        winner_trial_id=winner,
        summary=summary,
        horizon_summary=base.horizon_summary,
    )


def finalize_dynamic_lr_seed7(
    effect_decision: DynamicLRSeed7Decision,
    comparison: dict[str, float | int],
    *,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FinalDynamicLRSeed7Decision:
    """Authorize v19 confirmation only after effect and relative-speed gates."""

    if effect_decision.winner_trial_id is None:
        return FinalDynamicLRSeed7Decision(
            status="stop_dynamic_lr_seed7_effect_v19",
            winner_trial_id=None,
            relative_speed_gate_passed=False,
            confirmation_seeds_authorized=(),
        )
    try:
        model_step_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v19 comparison is missing relative speed") from exc
    if (
        not np.isfinite([model_step_ratio, end_to_end_ratio]).all()
        or min_model_step_speed_ratio <= 0
        or min_end_to_end_speed_ratio <= 0
    ):
        raise ContractError("v19 relative speed values and gates must be positive")
    speed_passed = bool(
        model_step_ratio >= min_model_step_speed_ratio
        and end_to_end_ratio >= min_end_to_end_speed_ratio
    )
    return FinalDynamicLRSeed7Decision(
        status=(
            "dynamic_lr_seed7_admitted_v19"
            if speed_passed
            else "stop_dynamic_lr_seed7_speed_v19"
        ),
        winner_trial_id=(effect_decision.winner_trial_id if speed_passed else None),
        relative_speed_gate_passed=speed_passed,
        confirmation_seeds_authorized=((17, 27) if speed_passed else ()),
    )
