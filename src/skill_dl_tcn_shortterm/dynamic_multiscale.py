"""Capacity- and mechanism-aware v20 dynamic multiscale decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError
from .real_validation import evaluate_stabilized_residual_seed7


@dataclass(frozen=True)
class DynamicMultiscaleSeed7Decision:
    """Effect decision for the bounded v20 dilation-block screen."""

    status: str
    winner_trial_id: str | None
    summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class FinalDynamicMultiscaleSeed7Decision:
    """v20 effect decision combined with fixed LSTM speed gates."""

    status: str
    winner_trial_id: str | None
    relative_speed_gate_passed: bool
    confirmation_seeds_authorized: tuple[int, ...]


@dataclass(frozen=True)
class DynamicMultiscaleMultiSeedDecision:
    """Effect and speed decision for frozen v21 confirmation seeds."""

    status: str
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class DynamicSkipLRMultiSeedDecision:
    """Effect, mechanism and speed decision for the v22 optimizer probe."""

    status: str
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class DynamicSkipWarmupMultiSeedDecision:
    """Effect, bounded-mechanism and speed decision for v23 warm-up."""

    status: str
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class DynamicSkipTokenLayerNormMultiSeedDecision:
    """Effect, dispersion and speed decision for v24 token LayerNorm."""

    status: str
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class DynamicSkipShapeAmplitudeMultiSeedDecision:
    """Effect, mechanism and speed decision for v25 shape/amplitude inputs."""

    status: str
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class DynamicSkipRawShapeResidualMultiSeedDecision:
    """Effect, counterfactual and speed decision for v26 shape residual."""

    status: str
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class FrozenParentShapeResidualMultiSeedDecision:
    """Integrity, incremental effect and speed decision for v27."""

    status: str
    integrity_passed: bool
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class DecoupledCheckpointSelectionMultiSeedDecision:
    """Trajectory integrity, effect and speed decision for v28."""

    status: str
    integrity_passed: bool
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class FrozenShapeLearningRateMultiSeedDecision:
    """Frozen-training integrity, v28 paired effect and speed decision for v29."""

    status: str
    integrity_passed: bool
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class FrozenShapeSoftRankICMultiSeedDecision:
    """Objective integrity, paired rank value and speed decision for v30."""

    status: str
    integrity_passed: bool
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


def evaluate_dynamic_multiscale_seed7(
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
    min_dynamic_skip_output_weight_l2: float,
    min_block_weight_variation: float,
    max_simplex_error: float,
    control_parameter_count: int,
    candidate_parameter_count: int,
    dynamic_parameter_count: int,
) -> DynamicMultiscaleSeed7Decision:
    """Select v20 only after effect, capacity and dynamic-scale use gates."""

    required = {"trial_id", "parameter_count", "dynamic_skip_output_weight_l2"}
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            "v20 dynamic multiscale leaderboard missing columns: "
            + ", ".join(missing)
        )
    diagnostic_required = {
        "trial_id",
        "fold",
        "block_weight_variation",
        "simplex_error_max",
    }
    if missing := sorted(diagnostic_required.difference(attention_diagnostics.columns)):
        raise ContractError(
            "v20 dynamic multiscale diagnostics missing columns: "
            + ", ".join(missing)
        )
    if dynamic_parameter_count != 88:
        raise ContractError("v20 dynamic parameter count must be exactly 88")
    if candidate_parameter_count - control_parameter_count != dynamic_parameter_count:
        raise ContractError("v20 candidate capacity delta must equal dynamic capacity")
    positive_thresholds = np.asarray(
        [
            min_mean_rankic_delta,
            min_dynamic_skip_output_weight_l2,
            min_block_weight_variation,
            max_simplex_error,
        ],
        dtype="float64",
    )
    if not np.isfinite(positive_thresholds).all() or bool(
        (positive_thresholds <= 0).any()
    ):
        raise ContractError("v20 mechanism and delta gates must be positive")
    if set(leaderboard["trial_id"].astype(str)) != {
        control_trial_id,
        candidate_trial_id,
    }:
        raise ContractError("v20 dynamic multiscale trial identities drifted")

    control_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ]
    candidate_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
    ]
    if set(control_rows["parameter_count"].astype(int)) != {control_parameter_count}:
        raise ContractError("v20 control parameter count drifted")
    if set(candidate_rows["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v20 candidate parameter count drifted")
    output_l2 = candidate_rows["dynamic_skip_output_weight_l2"].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(output_l2).all():
        raise ContractError("v20 dynamic skip use evidence is non-finite")

    diagnostics = attention_diagnostics.loc[
        attention_diagnostics["trial_id"].astype(str).eq(candidate_trial_id)
    ].copy()
    if (
        len(diagnostics) != 5
        or set(diagnostics["fold"].astype(int)) != set(range(5))
        or diagnostics.duplicated(["trial_id", "fold"]).any()
    ):
        raise ContractError("v20 diagnostics must cover folds 0 through 4")
    variation = diagnostics["block_weight_variation"].to_numpy(dtype="float64")
    simplex_error = diagnostics["simplex_error_max"].to_numpy(dtype="float64")
    if (
        not np.isfinite(variation).all()
        or not np.isfinite(simplex_error).all()
        or bool((variation < 0).any())
        or bool((simplex_error < 0).any())
    ):
        raise ContractError("v20 dynamic multiscale diagnostics are invalid")

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
    summary["dynamic_skip_output_weight_l2_min"] = np.where(
        is_control, np.nan, float(output_l2.min())
    )
    summary["block_weight_variation_min"] = np.where(
        is_control, np.nan, float(variation.min())
    )
    summary["simplex_error_max"] = np.where(
        is_control, np.nan, float(simplex_error.max())
    )
    candidate_index = summary.index[
        summary["trial_id"].astype(str).eq(candidate_trial_id)
    ]
    if len(candidate_index) != 1:
        raise ContractError("v20 candidate summary is incomplete")
    row_index = int(candidate_index[0])
    blockers = [
        value for value in str(summary.at[row_index, "blockers"]).split(",") if value
    ]
    if (
        float(cast(Any, summary.at[row_index, "mean_rankic_delta"]))
        < min_mean_rankic_delta
    ):
        blockers.append("mean_rankic_delta_below_gate")
    if float(output_l2.min()) < min_dynamic_skip_output_weight_l2:
        blockers.append("dynamic_skip_not_used")
    if float(variation.min()) < min_block_weight_variation:
        blockers.append("block_weights_not_sample_conditioned")
    if float(simplex_error.max()) > max_simplex_error:
        blockers.append("dynamic_skip_simplex_invalid")
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
    return DynamicMultiscaleSeed7Decision(
        status=(
            "dynamic_multiscale_seed7_effect_admitted_v20"
            if winner is not None
            else "stop_dynamic_multiscale_seed7_effect_v20"
        ),
        winner_trial_id=winner,
        summary=summary,
        horizon_summary=base.horizon_summary,
    )


def finalize_dynamic_multiscale_seed7(
    effect_decision: DynamicMultiscaleSeed7Decision,
    comparison: dict[str, float | int],
    *,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FinalDynamicMultiscaleSeed7Decision:
    """Authorize v20 confirmation only after effect and speed gates."""

    if effect_decision.winner_trial_id is None:
        return FinalDynamicMultiscaleSeed7Decision(
            status="stop_dynamic_multiscale_seed7_effect_v20",
            winner_trial_id=None,
            relative_speed_gate_passed=False,
            confirmation_seeds_authorized=(),
        )
    try:
        model_step_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v20 comparison is missing relative speed") from exc
    if (
        not np.isfinite([model_step_ratio, end_to_end_ratio]).all()
        or min_model_step_speed_ratio <= 0
        or min_end_to_end_speed_ratio <= 0
    ):
        raise ContractError("v20 relative speed values and gates must be positive")
    speed_passed = bool(
        model_step_ratio >= min_model_step_speed_ratio
        and end_to_end_ratio >= min_end_to_end_speed_ratio
    )
    return FinalDynamicMultiscaleSeed7Decision(
        status=(
            "dynamic_multiscale_seed7_admitted_v20"
            if speed_passed
            else "stop_dynamic_multiscale_seed7_speed_v20"
        ),
        winner_trial_id=(effect_decision.winner_trial_id if speed_passed else None),
        relative_speed_gate_passed=speed_passed,
        confirmation_seeds_authorized=((17, 27) if speed_passed else ()),
    )


def evaluate_dynamic_multiscale_multiseed(
    leaderboard: pd.DataFrame,
    attention_diagnostics: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    min_dynamic_skip_output_weight_l2: float,
    min_block_weight_variation: float,
    max_simplex_error: float,
    control_parameter_count: int,
    candidate_parameter_count: int,
    dynamic_parameter_count: int,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> DynamicMultiscaleMultiSeedDecision:
    """Confirm v20 on exact new seeds using paired control comparisons."""

    required = {
        "trial_id",
        "fold",
        "seed",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "dynamic_skip_output_weight_l2",
    }
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            "v21 dynamic multiscale leaderboard missing columns: "
            + ", ".join(missing)
        )
    diagnostic_required = {
        "trial_id",
        "seed",
        "fold",
        "block_weight_variation",
        "simplex_error_max",
    }
    if missing := sorted(diagnostic_required.difference(attention_diagnostics.columns)):
        raise ContractError(
            "v21 dynamic multiscale diagnostics missing columns: "
            + ", ".join(missing)
        )
    if leaderboard.empty:
        raise ContractError("v21 dynamic multiscale leaderboard cannot be empty")
    if expected_seeds != (17, 27):
        raise ContractError("v21 expected seeds must be exactly 17 and 27")
    if set(leaderboard["seed"].astype(int)) != set(expected_seeds):
        raise ContractError("v21 leaderboard seeds drifted")
    if set(leaderboard["trial_id"].astype(str)) != {
        control_trial_id,
        candidate_trial_id,
    }:
        raise ContractError("v21 trial identities drifted")
    if leaderboard.duplicated(["trial_id", "seed", "fold"]).any():
        raise ContractError("v21 leaderboard contains duplicate units")
    expected_units = {
        (trial_id, seed, fold)
        for trial_id in (control_trial_id, candidate_trial_id)
        for seed in expected_seeds
        for fold in range(5)
    }
    observed_units = {
        (
            str(row.trial_id),
            int(cast(Any, row.seed)),
            int(cast(Any, row.fold)),
        )
        for row in leaderboard.itertuples(index=False)
    }
    if observed_units != expected_units:
        raise ContractError("v21 leaderboard fold coverage drifted")
    numeric_columns = [
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
    ]
    if not np.isfinite(leaderboard[numeric_columns].to_numpy(dtype="float64")).all():
        raise ContractError("v21 leaderboard contains non-finite values")
    control_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ]
    candidate_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
    ]
    if set(control_rows["parameter_count"].astype(int)) != {control_parameter_count}:
        raise ContractError("v21 control parameter count drifted")
    if set(candidate_rows["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v21 candidate parameter count drifted")
    if dynamic_parameter_count != 88 or (
        candidate_parameter_count - control_parameter_count
        != dynamic_parameter_count
    ):
        raise ContractError("v21 dynamic capacity must be exactly 88")
    output_l2 = candidate_rows["dynamic_skip_output_weight_l2"].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(output_l2).all() or bool((output_l2 < 0).any()):
        raise ContractError("v21 dynamic skip use evidence is invalid")

    diagnostics = attention_diagnostics.loc[
        attention_diagnostics["trial_id"].astype(str).eq(candidate_trial_id)
    ].copy()
    if diagnostics.duplicated(["trial_id", "seed", "fold"]).any():
        raise ContractError("v21 diagnostics contain duplicate units")
    diagnostic_units = {
        (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
        for row in diagnostics.itertuples(index=False)
    }
    expected_diagnostic_units = {
        (seed, fold) for seed in expected_seeds for fold in range(5)
    }
    if diagnostic_units != expected_diagnostic_units:
        raise ContractError("v21 diagnostics fold coverage drifted")
    variation = diagnostics["block_weight_variation"].to_numpy(dtype="float64")
    simplex_error = diagnostics["simplex_error_max"].to_numpy(dtype="float64")
    if (
        not np.isfinite(variation).all()
        or not np.isfinite(simplex_error).all()
        or bool((variation < 0).any())
        or bool((simplex_error < 0).any())
    ):
        raise ContractError("v21 dynamic multiscale diagnostics are invalid")

    count_thresholds = (min_positive_units, min_nondegrading_folds_per_seed)
    if min_positive_units != 10 or not 1 <= min_nondegrading_folds_per_seed <= 5:
        raise ContractError("v21 count gates are invalid")
    positive_thresholds = np.asarray(
        [
            min_mean_rankic,
            min_mean_rankic_delta,
            min_median_samples_per_second,
            min_dynamic_skip_output_weight_l2,
            min_block_weight_variation,
            max_simplex_error,
            min_model_step_speed_ratio,
            min_end_to_end_speed_ratio,
        ],
        dtype="float64",
    )
    if not np.isfinite(positive_thresholds).all() or bool(
        (positive_thresholds <= 0).any()
    ):
        raise ContractError("v21 positive gates are invalid")
    horizon_thresholds = np.asarray(
        [
            min_horizon_delta_1d,
            min_horizon_delta_2d,
            min_horizon_delta_3d,
            min_horizon_delta_5d,
        ],
        dtype="float64",
    )
    if not np.isfinite(horizon_thresholds).all():
        raise ContractError("v21 horizon gates must be finite")
    del count_thresholds

    indexed = leaderboard.set_index(["seed", "fold", "trial_id"])[
        "best_mean_daily_rankic"
    ].unstack("trial_id")
    indexed["rankic_delta"] = (
        indexed[candidate_trial_id] - indexed[control_trial_id]
    )
    seed_summary = (
        indexed.reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            candidate_mean_rankic=(candidate_trial_id, "mean"),
            control_mean_rankic=(control_trial_id, "mean"),
            mean_rankic_delta=("rankic_delta", "mean"),
            nondegrading_folds=(
                "rankic_delta",
                lambda values: int((values >= 0).sum()),
            ),
        )
        .sort_values("seed", ignore_index=True)
    )
    horizon_rows: list[dict[str, float | int]] = []
    horizon_gates = {
        1: min_horizon_delta_1d,
        2: min_horizon_delta_2d,
        3: min_horizon_delta_3d,
        5: min_horizon_delta_5d,
    }
    for horizon in (1, 2, 3, 5):
        column = f"rankic_{horizon}d"
        means = leaderboard.groupby("trial_id", observed=True)[column].mean()
        horizon_rows.append(
            {
                "horizon": horizon,
                "control_rankic": float(means[control_trial_id]),
                "candidate_rankic": float(means[candidate_trial_id]),
                "rankic_delta": float(
                    means[candidate_trial_id] - means[control_trial_id]
                ),
            }
        )
    horizon_summary = pd.DataFrame(horizon_rows)
    candidate_mean = float(candidate_rows["best_mean_daily_rankic"].mean())
    positive_units = int(candidate_rows["best_mean_daily_rankic"].gt(0).sum())
    mean_delta = float(indexed["rankic_delta"].mean())
    median_throughput = float(candidate_rows["samples_per_second"].median())
    blockers: list[str] = []
    if candidate_mean < min_mean_rankic:
        blockers.append("mean_rankic_below_gate")
    if positive_units < min_positive_units:
        blockers.append("positive_units_below_gate")
    if mean_delta < min_mean_rankic_delta:
        blockers.append("mean_rankic_delta_below_gate")
    if bool(seed_summary["mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_mean_delta_not_positive")
    if bool(
        seed_summary["nondegrading_folds"].lt(
            min_nondegrading_folds_per_seed
        ).any()
    ):
        blockers.append("per_seed_fold_stability_below_gate")
    for row in horizon_summary.itertuples(index=False):
        horizon = int(cast(Any, row.horizon))
        if float(cast(Any, row.rankic_delta)) < horizon_gates[horizon]:
            blockers.append(f"horizon_{horizon}d_degradation_below_gate")
    if median_throughput < min_median_samples_per_second:
        blockers.append("throughput_below_gate")
    if float(output_l2.min()) < min_dynamic_skip_output_weight_l2:
        blockers.append("dynamic_skip_not_used")
    if float(variation.min()) < min_block_weight_variation:
        blockers.append("block_weights_not_sample_conditioned")
    if float(simplex_error.max()) > max_simplex_error:
        blockers.append("dynamic_skip_simplex_invalid")
    effect_passed = not blockers

    try:
        model_step_speed_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_speed_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v21 LSTM comparison is incomplete") from exc
    if not np.isfinite([model_step_speed_ratio, end_to_end_speed_ratio]).all():
        raise ContractError("v21 LSTM speed comparison is non-finite")
    speed_passed = bool(
        model_step_speed_ratio >= min_model_step_speed_ratio
        and end_to_end_speed_ratio >= min_end_to_end_speed_ratio
    )
    status = (
        "stop_dynamic_multiscale_unstable_v21"
        if not effect_passed
        else (
            "dynamic_multiscale_multiseed_confirmed_v21"
            if speed_passed
            else "stop_dynamic_multiscale_speed_v21"
        )
    )
    aggregate: dict[str, float | int | bool | str] = {
        "candidate_mean_rankic": candidate_mean,
        "positive_units": positive_units,
        "mean_rankic_delta": mean_delta,
        "median_samples_per_second": median_throughput,
        "dynamic_skip_output_weight_l2_min": float(output_l2.min()),
        "block_weight_variation_min": float(variation.min()),
        "simplex_error_max": float(simplex_error.max()),
        "model_step_speed_ratio": model_step_speed_ratio,
        "end_to_end_speed_ratio": end_to_end_speed_ratio,
        "control_parameter_count": control_parameter_count,
        "candidate_parameter_count": candidate_parameter_count,
        "dynamic_parameter_count": dynamic_parameter_count,
        "blockers": ",".join(blockers),
    }
    return DynamicMultiscaleMultiSeedDecision(
        status=status,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=horizon_summary,
    )


def evaluate_dynamic_skip_lr_multiseed(
    current_leaderboard: pd.DataFrame,
    historical_leaderboard: pd.DataFrame,
    current_diagnostics: pd.DataFrame,
    parent_diagnostics: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_mean_rankic_delta: float,
    min_parent_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    min_dynamic_skip_output_weight_l2: float,
    min_block_weight_variation: float,
    min_parent_variation_ratio: float,
    max_simplex_error: float,
    control_parameter_count: int,
    candidate_parameter_count: int,
    dynamic_parameter_count: int,
    base_learning_rate: float,
    dynamic_skip_learning_rate: float,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> DynamicSkipLRMultiSeedDecision:
    """Evaluate one v22 candidate against immutable static and parent evidence."""

    current_required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "dynamic_skip_output_weight_l2",
        "dynamic_skip_learning_rate",
        "optimizer_dynamic_skip_parameter_count",
    }
    historical_required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "parameter_count",
    }
    diagnostic_required = {
        "trial_id",
        "seed",
        "fold",
        "block_weight_variation",
        "simplex_error_max",
    }
    if missing := sorted(current_required.difference(current_leaderboard.columns)):
        raise ContractError(
            "v22 current leaderboard missing columns: " + ", ".join(missing)
        )
    if missing := sorted(
        historical_required.difference(historical_leaderboard.columns)
    ):
        raise ContractError(
            "v22 historical leaderboard missing columns: " + ", ".join(missing)
        )
    for name, diagnostics in (
        ("current", current_diagnostics),
        ("parent", parent_diagnostics),
    ):
        if missing := sorted(diagnostic_required.difference(diagnostics.columns)):
            raise ContractError(
                f"v22 {name} diagnostics missing columns: " + ", ".join(missing)
            )
    if expected_seeds != (7, 17, 27):
        raise ContractError("v22 expected seeds must be exactly 7, 17 and 27")
    expected_units = {
        (seed, fold) for seed in expected_seeds for fold in range(5)
    }

    def _units(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return {
            (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
            for row in frame.itertuples(index=False)
        }

    if (
        set(current_leaderboard["trial_id"].astype(str)) != {candidate_trial_id}
        or current_leaderboard.duplicated(["trial_id", "seed", "fold"]).any()
        or _units(current_leaderboard) != expected_units
    ):
        raise ContractError("v22 current leaderboard coverage drifted")
    if set(historical_leaderboard["trial_id"].astype(str)) != {
        control_trial_id,
        parent_candidate_trial_id,
    } or historical_leaderboard.duplicated(["trial_id", "seed", "fold"]).any():
        raise ContractError("v22 historical leaderboard coverage drifted")
    for trial_id in (control_trial_id, parent_candidate_trial_id):
        rows = historical_leaderboard.loc[
            historical_leaderboard["trial_id"].astype(str).eq(trial_id)
        ]
        if _units(rows) != expected_units:
            raise ContractError("v22 historical leaderboard coverage drifted")
    for name, diagnostics, trial_id in (
        ("current", current_diagnostics, candidate_trial_id),
        ("parent", parent_diagnostics, parent_candidate_trial_id),
    ):
        if (
            set(diagnostics["trial_id"].astype(str)) != {trial_id}
            or diagnostics.duplicated(["trial_id", "seed", "fold"]).any()
            or _units(diagnostics) != expected_units
        ):
            raise ContractError(f"v22 {name} diagnostics coverage drifted")

    current_numeric = [
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "dynamic_skip_output_weight_l2",
        "dynamic_skip_learning_rate",
        "optimizer_dynamic_skip_parameter_count",
    ]
    historical_numeric = [
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "parameter_count",
    ]
    if not np.isfinite(
        current_leaderboard[current_numeric].to_numpy(dtype="float64")
    ).all() or not np.isfinite(
        historical_leaderboard[historical_numeric].to_numpy(dtype="float64")
    ).all():
        raise ContractError("v22 leaderboard contains non-finite evidence")
    if dynamic_parameter_count != 88 or (
        candidate_parameter_count - control_parameter_count != dynamic_parameter_count
    ):
        raise ContractError("v22 dynamic capacity must be exactly 88")
    control_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ]
    parent_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(
            parent_candidate_trial_id
        )
    ]
    if set(control_rows["parameter_count"].astype(int)) != {
        control_parameter_count
    }:
        raise ContractError("v22 control parameter count drifted")
    if set(parent_rows["parameter_count"].astype(int)) != {
        candidate_parameter_count
    } or set(current_leaderboard["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v22 dynamic candidate parameter count drifted")
    if set(
        current_leaderboard["optimizer_dynamic_skip_parameter_count"].astype(int)
    ) != {dynamic_parameter_count}:
        raise ContractError("v22 optimizer dynamic parameter count drifted")
    if (
        not np.isfinite([base_learning_rate, dynamic_skip_learning_rate]).all()
        or base_learning_rate <= 0
        or dynamic_skip_learning_rate <= base_learning_rate
        or dynamic_skip_learning_rate > 10 * base_learning_rate
        or not np.allclose(
            current_leaderboard["dynamic_skip_learning_rate"].to_numpy(
                dtype="float64"
            ),
            dynamic_skip_learning_rate,
            rtol=0,
            atol=1e-12,
        )
    ):
        raise ContractError("v22 dynamic skip learning rate drifted")

    positive_gates = np.asarray(
        [
            min_mean_rankic,
            min_mean_rankic_delta,
            min_parent_mean_rankic_delta,
            min_median_samples_per_second,
            min_dynamic_skip_output_weight_l2,
            min_block_weight_variation,
            min_parent_variation_ratio,
            max_simplex_error,
            min_model_step_speed_ratio,
            min_end_to_end_speed_ratio,
        ],
        dtype="float64",
    )
    horizon_gates = {
        1: min_horizon_delta_1d,
        2: min_horizon_delta_2d,
        3: min_horizon_delta_3d,
        5: min_horizon_delta_5d,
    }
    if not np.isfinite(positive_gates).all() or bool((positive_gates <= 0).any()):
        raise ContractError("v22 positive gates are invalid")
    if not np.isfinite(list(horizon_gates.values())).all():
        raise ContractError("v22 horizon gates must be finite")
    if min_positive_units != 15 or not 1 <= min_nondegrading_folds_per_seed <= 5:
        raise ContractError("v22 count gates are invalid")

    key = ["seed", "fold"]
    current_values = current_leaderboard.set_index(key)["best_mean_daily_rankic"]
    control_values = control_rows.set_index(key)["best_mean_daily_rankic"]
    parent_values = parent_rows.set_index(key)["best_mean_daily_rankic"]
    paired = pd.concat(
        {
            "candidate": current_values,
            "control": control_values,
            "parent": parent_values,
        },
        axis=1,
    )
    paired["rankic_delta"] = paired["candidate"] - paired["control"]
    paired["parent_rankic_delta"] = paired["candidate"] - paired["parent"]
    seed_summary = (
        paired.reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            candidate_mean_rankic=("candidate", "mean"),
            control_mean_rankic=("control", "mean"),
            parent_mean_rankic=("parent", "mean"),
            mean_rankic_delta=("rankic_delta", "mean"),
            parent_mean_rankic_delta=("parent_rankic_delta", "mean"),
            nondegrading_folds=(
                "rankic_delta", lambda values: int((values >= 0).sum())
            ),
        )
        .sort_values("seed", ignore_index=True)
    )

    horizon_rows: list[dict[str, float | int]] = []
    for horizon in (1, 2, 3, 5):
        column = f"rankic_{horizon}d"
        control_mean = float(control_rows[column].mean())
        candidate_mean_horizon = float(current_leaderboard[column].mean())
        horizon_rows.append(
            {
                "horizon": horizon,
                "control_rankic": control_mean,
                "candidate_rankic": candidate_mean_horizon,
                "rankic_delta": candidate_mean_horizon - control_mean,
            }
        )
    horizon_summary = pd.DataFrame(horizon_rows)

    current_variation = current_diagnostics.set_index(key)[
        "block_weight_variation"
    ].astype(float)
    parent_variation = parent_diagnostics.set_index(key)[
        "block_weight_variation"
    ].astype(float)
    current_simplex = current_diagnostics["simplex_error_max"].to_numpy(
        dtype="float64"
    )
    parent_variation_values = parent_variation.to_numpy(dtype="float64")
    current_variation_values = current_variation.to_numpy(dtype="float64")
    if (
        not np.isfinite(current_variation_values).all()
        or not np.isfinite(parent_variation_values).all()
        or not np.isfinite(current_simplex).all()
        or bool((current_variation_values < 0).any())
        or bool((parent_variation_values <= 0).any())
        or bool((current_simplex < 0).any())
    ):
        raise ContractError("v22 dynamic diagnostics are invalid")
    parent_variation_ratio = float(
        np.median(current_variation_values / parent_variation_values)
    )

    candidate_mean = float(current_values.mean())
    positive_units = int(current_values.gt(0).sum())
    mean_delta = float(paired["rankic_delta"].mean())
    parent_mean_delta = float(paired["parent_rankic_delta"].mean())
    median_throughput = float(current_leaderboard["samples_per_second"].median())
    output_l2_min = float(
        current_leaderboard["dynamic_skip_output_weight_l2"].min()
    )
    variation_min = float(current_variation.min())
    simplex_max = float(current_simplex.max())
    blockers: list[str] = []
    if candidate_mean < min_mean_rankic:
        blockers.append("mean_rankic_below_gate")
    if positive_units < min_positive_units:
        blockers.append("positive_units_below_gate")
    if mean_delta < min_mean_rankic_delta:
        blockers.append("mean_rankic_delta_below_gate")
    if parent_mean_delta < min_parent_mean_rankic_delta:
        blockers.append("parent_mean_rankic_delta_below_gate")
    if bool(seed_summary["mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_mean_delta_not_positive")
    if bool(
        seed_summary["nondegrading_folds"].lt(
            min_nondegrading_folds_per_seed
        ).any()
    ):
        blockers.append("per_seed_fold_stability_below_gate")
    for row in horizon_summary.itertuples(index=False):
        horizon = int(cast(Any, row.horizon))
        if float(cast(Any, row.rankic_delta)) < horizon_gates[horizon]:
            blockers.append(f"horizon_{horizon}d_degradation_below_gate")
    if median_throughput < min_median_samples_per_second:
        blockers.append("throughput_below_gate")
    if output_l2_min < min_dynamic_skip_output_weight_l2:
        blockers.append("dynamic_skip_not_used")
    if variation_min < min_block_weight_variation:
        blockers.append("block_weights_not_sample_conditioned")
    if parent_variation_ratio < min_parent_variation_ratio:
        blockers.append("parent_variation_ratio_below_gate")
    if simplex_max > max_simplex_error:
        blockers.append("dynamic_skip_simplex_invalid")
    effect_passed = not blockers

    try:
        model_step_speed_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_speed_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v22 LSTM comparison is incomplete") from exc
    if not np.isfinite([model_step_speed_ratio, end_to_end_speed_ratio]).all():
        raise ContractError("v22 LSTM speed comparison is non-finite")
    speed_passed = bool(
        model_step_speed_ratio >= min_model_step_speed_ratio
        and end_to_end_speed_ratio >= min_end_to_end_speed_ratio
    )
    status = (
        "stop_dynamic_skip_lr_unstable_v22"
        if not effect_passed
        else (
            "dynamic_skip_lr_multiseed_confirmed_v22"
            if speed_passed
            else "stop_dynamic_skip_lr_speed_v22"
        )
    )
    aggregate: dict[str, float | int | bool | str] = {
        "candidate_mean_rankic": candidate_mean,
        "positive_units": positive_units,
        "mean_rankic_delta": mean_delta,
        "parent_mean_rankic_delta": parent_mean_delta,
        "median_samples_per_second": median_throughput,
        "dynamic_skip_output_weight_l2_min": output_l2_min,
        "block_weight_variation_min": variation_min,
        "parent_variation_ratio": parent_variation_ratio,
        "simplex_error_max": simplex_max,
        "model_step_speed_ratio": model_step_speed_ratio,
        "end_to_end_speed_ratio": end_to_end_speed_ratio,
        "control_parameter_count": control_parameter_count,
        "candidate_parameter_count": candidate_parameter_count,
        "dynamic_parameter_count": dynamic_parameter_count,
        "base_learning_rate": base_learning_rate,
        "dynamic_skip_learning_rate": dynamic_skip_learning_rate,
        "blockers": ",".join(blockers),
    }
    return DynamicSkipLRMultiSeedDecision(
        status=status,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=horizon_summary,
    )


def evaluate_dynamic_skip_warmup_multiseed(
    current_leaderboard: pd.DataFrame,
    historical_leaderboard: pd.DataFrame,
    current_diagnostics: pd.DataFrame,
    parent_diagnostics: pd.DataFrame,
    high_lr_diagnostics: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
    high_lr_candidate_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_mean_rankic_delta: float,
    min_parent_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    min_dynamic_skip_output_weight_l2: float,
    min_block_weight_variation: float,
    min_parent_variation_ratio: float,
    max_parent_variation_ratio: float,
    max_high_lr_variation_ratio: float,
    max_simplex_error: float,
    control_parameter_count: int,
    candidate_parameter_count: int,
    dynamic_parameter_count: int,
    base_learning_rate: float,
    dynamic_skip_learning_rate: float,
    dynamic_skip_warmup_epochs: int,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> DynamicSkipWarmupMultiSeedDecision:
    """Evaluate v23 against static, common-LR and high-LR boundaries."""

    required_schedule_columns = {
        "dynamic_skip_warmup_epochs",
        "optimizer_group_identity",
    }
    if missing := sorted(
        required_schedule_columns.difference(current_leaderboard.columns)
    ):
        raise ContractError(
            "v23 current leaderboard missing schedule columns: "
            + ", ".join(missing)
        )
    if dynamic_skip_warmup_epochs <= 0:
        raise ContractError("v23 warmup epochs must be positive")
    expected_schedule_identity = (
        f"base-lr-{base_learning_rate:g}"
        f"+dynamic-skip-linear-warmup-{dynamic_skip_warmup_epochs}"
        f"-lr-{base_learning_rate:g}-to-{dynamic_skip_learning_rate:g}"
    )
    if set(current_leaderboard["dynamic_skip_warmup_epochs"].astype(int)) != {
        dynamic_skip_warmup_epochs
    } or set(current_leaderboard["optimizer_group_identity"].astype(str)) != {
        expected_schedule_identity
    }:
        raise ContractError("v23 optimizer schedule identity drifted")

    base = evaluate_dynamic_skip_lr_multiseed(
        current_leaderboard,
        historical_leaderboard,
        current_diagnostics,
        parent_diagnostics,
        comparison,
        control_trial_id=control_trial_id,
        parent_candidate_trial_id=parent_candidate_trial_id,
        candidate_trial_id=candidate_trial_id,
        expected_seeds=expected_seeds,
        min_mean_rankic=min_mean_rankic,
        min_positive_units=min_positive_units,
        min_mean_rankic_delta=min_mean_rankic_delta,
        min_parent_mean_rankic_delta=min_parent_mean_rankic_delta,
        min_nondegrading_folds_per_seed=min_nondegrading_folds_per_seed,
        min_horizon_delta_1d=min_horizon_delta_1d,
        min_horizon_delta_2d=min_horizon_delta_2d,
        min_horizon_delta_3d=min_horizon_delta_3d,
        min_horizon_delta_5d=min_horizon_delta_5d,
        min_median_samples_per_second=min_median_samples_per_second,
        min_dynamic_skip_output_weight_l2=min_dynamic_skip_output_weight_l2,
        min_block_weight_variation=min_block_weight_variation,
        min_parent_variation_ratio=min_parent_variation_ratio,
        max_simplex_error=max_simplex_error,
        control_parameter_count=control_parameter_count,
        candidate_parameter_count=candidate_parameter_count,
        dynamic_parameter_count=dynamic_parameter_count,
        base_learning_rate=base_learning_rate,
        dynamic_skip_learning_rate=dynamic_skip_learning_rate,
        min_model_step_speed_ratio=min_model_step_speed_ratio,
        min_end_to_end_speed_ratio=min_end_to_end_speed_ratio,
    )

    diagnostic_required = {
        "trial_id",
        "seed",
        "fold",
        "block_weight_variation",
        "simplex_error_max",
    }
    if missing := sorted(diagnostic_required.difference(high_lr_diagnostics.columns)):
        raise ContractError(
            "v23 high-LR diagnostics missing columns: " + ", ".join(missing)
        )
    expected_units = {
        (seed, fold) for seed in expected_seeds for fold in range(5)
    }
    observed_units = {
        (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
        for row in high_lr_diagnostics.itertuples(index=False)
    }
    if (
        set(high_lr_diagnostics["trial_id"].astype(str))
        != {high_lr_candidate_trial_id}
        or observed_units != expected_units
        or high_lr_diagnostics.duplicated(["trial_id", "seed", "fold"]).any()
    ):
        raise ContractError("v23 high-LR diagnostics coverage drifted")
    high_values = (
        high_lr_diagnostics.set_index(["seed", "fold"])[
            "block_weight_variation"
        ]
        .sort_index()
        .to_numpy(dtype="float64")
    )
    current_values = (
        current_diagnostics.set_index(["seed", "fold"])[
            "block_weight_variation"
        ]
        .sort_index()
        .to_numpy(dtype="float64")
    )
    if (
        not np.isfinite(high_values).all()
        or bool((high_values <= 0).any())
        or not np.isfinite(
            [max_parent_variation_ratio, max_high_lr_variation_ratio]
        ).all()
        or max_parent_variation_ratio <= min_parent_variation_ratio
        or not 0 < max_high_lr_variation_ratio <= 1
    ):
        raise ContractError("v23 bounded variation gates are invalid")
    high_lr_variation_ratio = float(np.median(current_values / high_values))
    parent_variation_ratio = float(base.aggregate["parent_variation_ratio"])
    blockers = [
        value for value in str(base.aggregate["blockers"]).split(",") if value
    ]
    if bool(base.seed_summary["parent_mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_parent_mean_delta_not_positive")
    if parent_variation_ratio > max_parent_variation_ratio:
        blockers.append("parent_variation_ratio_above_gate")
    if high_lr_variation_ratio > max_high_lr_variation_ratio:
        blockers.append("high_lr_variation_ratio_above_gate")
    blockers = list(dict.fromkeys(blockers))
    effect_passed = not blockers
    speed_passed = base.speed_passed
    status = (
        "stop_dynamic_skip_warmup_unstable_v23"
        if not effect_passed
        else (
            "dynamic_skip_warmup_multiseed_confirmed_v23"
            if speed_passed
            else "stop_dynamic_skip_warmup_speed_v23"
        )
    )
    aggregate = dict(base.aggregate)
    aggregate.update(
        {
            "parent_variation_ratio": parent_variation_ratio,
            "high_lr_variation_ratio": high_lr_variation_ratio,
            "dynamic_skip_warmup_epochs": dynamic_skip_warmup_epochs,
            "optimizer_schedule_identity": expected_schedule_identity,
            "blockers": ",".join(blockers),
        }
    )
    return DynamicSkipWarmupMultiSeedDecision(
        status=status,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=base.seed_summary,
        horizon_summary=base.horizon_summary,
    )


def evaluate_dynamic_skip_token_layernorm_multiseed(
    current_leaderboard: pd.DataFrame,
    historical_leaderboard: pd.DataFrame,
    current_diagnostics: pd.DataFrame,
    parent_diagnostics: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_mean_rankic_delta: float,
    min_parent_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    min_dynamic_skip_output_weight_l2: float,
    min_block_weight_variation: float,
    max_parent_variation_cv_ratio: float,
    max_simplex_error: float,
    control_parameter_count: int,
    candidate_parameter_count: int,
    dynamic_parameter_count: int,
    learning_rate: float,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> DynamicSkipTokenLayerNormMultiSeedDecision:
    """Evaluate v24 against immutable static and common-LR evidence."""

    current_required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "dynamic_skip_output_weight_l2",
        "dynamic_skip_token_normalization",
        "optimizer_group_identity",
        "optimizer_dynamic_skip_parameter_count",
    }
    historical_required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "parameter_count",
    }
    diagnostic_required = {
        "trial_id",
        "seed",
        "fold",
        "block_weight_variation",
        "simplex_error_max",
    }
    if missing := sorted(current_required.difference(current_leaderboard.columns)):
        raise ContractError(
            "v24 current leaderboard missing columns: " + ", ".join(missing)
        )
    if missing := sorted(
        historical_required.difference(historical_leaderboard.columns)
    ):
        raise ContractError(
            "v24 historical leaderboard missing columns: " + ", ".join(missing)
        )
    for name, diagnostics in (
        ("current", current_diagnostics),
        ("parent", parent_diagnostics),
    ):
        if missing := sorted(diagnostic_required.difference(diagnostics.columns)):
            raise ContractError(
                f"v24 {name} diagnostics missing columns: " + ", ".join(missing)
            )
    if expected_seeds != (7, 17, 27):
        raise ContractError("v24 expected seeds must be exactly 7, 17 and 27")
    expected_units = {
        (seed, fold) for seed in expected_seeds for fold in range(5)
    }

    def _units(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return {
            (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
            for row in frame.itertuples(index=False)
        }

    if (
        set(current_leaderboard["trial_id"].astype(str)) != {candidate_trial_id}
        or current_leaderboard.duplicated(["trial_id", "seed", "fold"]).any()
        or _units(current_leaderboard) != expected_units
    ):
        raise ContractError("v24 current leaderboard coverage drifted")
    if set(historical_leaderboard["trial_id"].astype(str)) != {
        control_trial_id,
        parent_candidate_trial_id,
    } or historical_leaderboard.duplicated(["trial_id", "seed", "fold"]).any():
        raise ContractError("v24 historical leaderboard coverage drifted")
    for trial_id in (control_trial_id, parent_candidate_trial_id):
        rows = historical_leaderboard.loc[
            historical_leaderboard["trial_id"].astype(str).eq(trial_id)
        ]
        if _units(rows) != expected_units:
            raise ContractError("v24 historical leaderboard coverage drifted")
    for name, diagnostics, trial_id in (
        ("current", current_diagnostics, candidate_trial_id),
        ("parent", parent_diagnostics, parent_candidate_trial_id),
    ):
        if (
            set(diagnostics["trial_id"].astype(str)) != {trial_id}
            or diagnostics.duplicated(["trial_id", "seed", "fold"]).any()
            or _units(diagnostics) != expected_units
        ):
            raise ContractError(f"v24 {name} diagnostics coverage drifted")

    numeric_columns = [
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "dynamic_skip_output_weight_l2",
        "optimizer_dynamic_skip_parameter_count",
    ]
    if not np.isfinite(
        current_leaderboard[numeric_columns].to_numpy(dtype="float64")
    ).all():
        raise ContractError("v24 current leaderboard contains non-finite values")
    control_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ]
    parent_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(
            parent_candidate_trial_id
        )
    ]
    if dynamic_parameter_count != 88 or (
        candidate_parameter_count - control_parameter_count != dynamic_parameter_count
    ):
        raise ContractError("v24 dynamic capacity must be exactly 88")
    if set(control_rows["parameter_count"].astype(int)) != {
        control_parameter_count
    } or set(parent_rows["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v24 historical parameter counts drifted")
    if set(current_leaderboard["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v24 LayerNorm must not add parameters")
    if (
        set(current_leaderboard["dynamic_skip_token_normalization"].astype(str))
        != {"layer_norm"}
        or set(current_leaderboard["optimizer_group_identity"].astype(str))
        != {f"all-lr-{learning_rate:g}"}
        or set(
            current_leaderboard[
                "optimizer_dynamic_skip_parameter_count"
            ].astype(int)
        )
        != {0}
    ):
        raise ContractError("v24 token normalization or optimizer identity drifted")

    key = ["seed", "fold"]
    current_values = current_leaderboard.set_index(key)["best_mean_daily_rankic"]
    control_values = control_rows.set_index(key)["best_mean_daily_rankic"]
    parent_values = parent_rows.set_index(key)["best_mean_daily_rankic"]
    paired = pd.concat(
        {
            "candidate": current_values,
            "control": control_values,
            "parent": parent_values,
        },
        axis=1,
    )
    paired["rankic_delta"] = paired["candidate"] - paired["control"]
    paired["parent_rankic_delta"] = paired["candidate"] - paired["parent"]
    seed_summary = (
        paired.reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            candidate_mean_rankic=("candidate", "mean"),
            control_mean_rankic=("control", "mean"),
            parent_mean_rankic=("parent", "mean"),
            mean_rankic_delta=("rankic_delta", "mean"),
            parent_mean_rankic_delta=("parent_rankic_delta", "mean"),
            nondegrading_folds=(
                "rankic_delta", lambda values: int((values >= 0).sum())
            ),
        )
        .sort_values("seed", ignore_index=True)
    )
    horizon_gates = {
        1: min_horizon_delta_1d,
        2: min_horizon_delta_2d,
        3: min_horizon_delta_3d,
        5: min_horizon_delta_5d,
    }
    horizon_rows: list[dict[str, float | int]] = []
    for horizon in (1, 2, 3, 5):
        column = f"rankic_{horizon}d"
        control_mean = float(control_rows[column].mean())
        candidate_mean_horizon = float(current_leaderboard[column].mean())
        horizon_rows.append(
            {
                "horizon": horizon,
                "control_rankic": control_mean,
                "candidate_rankic": candidate_mean_horizon,
                "rankic_delta": candidate_mean_horizon - control_mean,
            }
        )
    horizon_summary = pd.DataFrame(horizon_rows)

    current_variation = current_diagnostics["block_weight_variation"].to_numpy(
        dtype="float64"
    )
    parent_variation = parent_diagnostics["block_weight_variation"].to_numpy(
        dtype="float64"
    )
    simplex = current_diagnostics["simplex_error_max"].to_numpy(dtype="float64")
    if (
        not np.isfinite(current_variation).all()
        or not np.isfinite(parent_variation).all()
        or not np.isfinite(simplex).all()
        or bool((current_variation <= 0).any())
        or bool((parent_variation <= 0).any())
        or bool((simplex < 0).any())
    ):
        raise ContractError("v24 dynamic diagnostics are invalid")
    current_cv = float(np.std(current_variation) / np.mean(current_variation))
    parent_cv = float(np.std(parent_variation) / np.mean(parent_variation))
    if parent_cv <= 0 or not 0 < max_parent_variation_cv_ratio <= 1:
        raise ContractError("v24 variation CV gate is invalid")
    variation_cv_ratio = current_cv / parent_cv

    candidate_mean = float(current_values.mean())
    positive_units = int(current_values.gt(0).sum())
    mean_delta = float(paired["rankic_delta"].mean())
    parent_mean_delta = float(paired["parent_rankic_delta"].mean())
    median_throughput = float(current_leaderboard["samples_per_second"].median())
    output_l2_min = float(
        current_leaderboard["dynamic_skip_output_weight_l2"].min()
    )
    blockers: list[str] = []
    if candidate_mean < min_mean_rankic:
        blockers.append("mean_rankic_below_gate")
    if positive_units < min_positive_units:
        blockers.append("positive_units_below_gate")
    if mean_delta < min_mean_rankic_delta:
        blockers.append("mean_rankic_delta_below_gate")
    if parent_mean_delta < min_parent_mean_rankic_delta:
        blockers.append("parent_mean_rankic_delta_below_gate")
    if bool(seed_summary["mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_mean_delta_not_positive")
    if bool(seed_summary["parent_mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_parent_mean_delta_not_positive")
    if bool(
        seed_summary["nondegrading_folds"].lt(
            min_nondegrading_folds_per_seed
        ).any()
    ):
        blockers.append("per_seed_fold_stability_below_gate")
    for row in horizon_summary.itertuples(index=False):
        horizon = int(cast(Any, row.horizon))
        if float(cast(Any, row.rankic_delta)) < horizon_gates[horizon]:
            blockers.append(f"horizon_{horizon}d_degradation_below_gate")
    if median_throughput < min_median_samples_per_second:
        blockers.append("throughput_below_gate")
    if output_l2_min < min_dynamic_skip_output_weight_l2:
        blockers.append("dynamic_skip_not_used")
    if float(current_variation.min()) < min_block_weight_variation:
        blockers.append("block_weights_not_sample_conditioned")
    if variation_cv_ratio > max_parent_variation_cv_ratio:
        blockers.append("variation_cv_not_reduced")
    if float(simplex.max()) > max_simplex_error:
        blockers.append("dynamic_skip_simplex_invalid")
    effect_passed = not blockers

    try:
        model_step_speed_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_speed_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v24 LSTM comparison is incomplete") from exc
    if not np.isfinite([model_step_speed_ratio, end_to_end_speed_ratio]).all():
        raise ContractError("v24 LSTM comparison is non-finite")
    speed_passed = bool(
        model_step_speed_ratio >= min_model_step_speed_ratio
        and end_to_end_speed_ratio >= min_end_to_end_speed_ratio
    )
    status = (
        "stop_dynamic_skip_token_layernorm_unstable_v24"
        if not effect_passed
        else (
            "dynamic_skip_token_layernorm_multiseed_confirmed_v24"
            if speed_passed
            else "stop_dynamic_skip_token_layernorm_speed_v24"
        )
    )
    aggregate: dict[str, float | int | bool | str] = {
        "candidate_mean_rankic": candidate_mean,
        "positive_units": positive_units,
        "mean_rankic_delta": mean_delta,
        "parent_mean_rankic_delta": parent_mean_delta,
        "median_samples_per_second": median_throughput,
        "dynamic_skip_output_weight_l2_min": output_l2_min,
        "block_weight_variation_min": float(current_variation.min()),
        "current_variation_cv": current_cv,
        "parent_variation_cv": parent_cv,
        "parent_variation_cv_ratio": variation_cv_ratio,
        "simplex_error_max": float(simplex.max()),
        "model_step_speed_ratio": model_step_speed_ratio,
        "end_to_end_speed_ratio": end_to_end_speed_ratio,
        "control_parameter_count": control_parameter_count,
        "candidate_parameter_count": candidate_parameter_count,
        "dynamic_parameter_count": dynamic_parameter_count,
        "learning_rate": learning_rate,
        "dynamic_skip_token_normalization": "layer_norm",
        "blockers": ",".join(blockers),
    }
    return DynamicSkipTokenLayerNormMultiSeedDecision(
        status=status,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=horizon_summary,
    )


def evaluate_dynamic_skip_shape_amplitude_multiseed(
    current_leaderboard: pd.DataFrame,
    historical_leaderboard: pd.DataFrame,
    current_diagnostics: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
    ablation_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_mean_rankic_delta: float,
    min_parent_mean_rankic_delta: float,
    min_ablation_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    min_dynamic_skip_output_weight_l2: float,
    min_amplitude_projection_weight_l2: float,
    min_block_weight_variation: float,
    max_simplex_error: float,
    control_parameter_count: int,
    historical_dynamic_parameter_count: int,
    candidate_parameter_count: int,
    dynamic_parameter_count: int,
    amplitude_parameter_count: int,
    scorer_input_width: int,
    learning_rate: float,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> DynamicSkipShapeAmplitudeMultiSeedDecision:
    """Evaluate v25 against static, raw-token and LayerNorm evidence."""

    current_required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "dynamic_skip_output_weight_l2",
        "dynamic_skip_amplitude_projection_weight_l2",
        "dynamic_skip_token_normalization",
        "dynamic_skip_amplitude_feature",
        "dynamic_skip_scorer_input_width",
        "dynamic_skip_normalization_parameter_count",
        "optimizer_group_identity",
        "optimizer_dynamic_skip_parameter_count",
    }
    historical_required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "parameter_count",
    }
    diagnostic_required = {
        "trial_id",
        "seed",
        "fold",
        "block_weight_variation",
        "simplex_error_max",
    }
    if missing := sorted(current_required.difference(current_leaderboard.columns)):
        raise ContractError(
            "v25 current leaderboard missing columns: " + ", ".join(missing)
        )
    if missing := sorted(
        historical_required.difference(historical_leaderboard.columns)
    ):
        raise ContractError(
            "v25 historical leaderboard missing columns: " + ", ".join(missing)
        )
    if missing := sorted(diagnostic_required.difference(current_diagnostics.columns)):
        raise ContractError(
            "v25 current diagnostics missing columns: " + ", ".join(missing)
        )
    if expected_seeds != (7, 17, 27):
        raise ContractError("v25 expected seeds must be exactly 7, 17 and 27")
    expected_units = {
        (seed, fold) for seed in expected_seeds for fold in range(5)
    }

    def _units(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return {
            (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
            for row in frame.itertuples(index=False)
        }

    if (
        set(current_leaderboard["trial_id"].astype(str)) != {candidate_trial_id}
        or current_leaderboard.duplicated(["trial_id", "seed", "fold"]).any()
        or _units(current_leaderboard) != expected_units
    ):
        raise ContractError("v25 current leaderboard coverage drifted")
    historical_ids = {
        control_trial_id,
        parent_candidate_trial_id,
        ablation_trial_id,
    }
    if (
        set(historical_leaderboard["trial_id"].astype(str)) != historical_ids
        or historical_leaderboard.duplicated(["trial_id", "seed", "fold"]).any()
    ):
        raise ContractError("v25 historical leaderboard coverage drifted")
    for trial_id in historical_ids:
        rows = historical_leaderboard.loc[
            historical_leaderboard["trial_id"].astype(str).eq(trial_id)
        ]
        if _units(rows) != expected_units:
            raise ContractError("v25 historical leaderboard coverage drifted")
    if (
        set(current_diagnostics["trial_id"].astype(str)) != {candidate_trial_id}
        or current_diagnostics.duplicated(["trial_id", "seed", "fold"]).any()
        or _units(current_diagnostics) != expected_units
    ):
        raise ContractError("v25 current diagnostics coverage drifted")

    numeric_columns = [
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "dynamic_skip_output_weight_l2",
        "dynamic_skip_amplitude_projection_weight_l2",
        "dynamic_skip_scorer_input_width",
        "dynamic_skip_normalization_parameter_count",
        "optimizer_dynamic_skip_parameter_count",
    ]
    if not np.isfinite(
        current_leaderboard[numeric_columns].to_numpy(dtype="float64")
    ).all():
        raise ContractError("v25 current leaderboard contains non-finite values")
    control_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ]
    parent_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(
            parent_candidate_trial_id
        )
    ]
    ablation_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(ablation_trial_id)
    ]
    historical_candidate_parameters = (
        control_parameter_count + historical_dynamic_parameter_count
    )
    if (
        dynamic_parameter_count != 92
        or amplitude_parameter_count != 4
        or candidate_parameter_count - control_parameter_count
        != dynamic_parameter_count
        or dynamic_parameter_count - historical_dynamic_parameter_count
        != amplitude_parameter_count
    ):
        raise ContractError("v25 shape/amplitude capacity contract drifted")
    if set(control_rows["parameter_count"].astype(int)) != {
        control_parameter_count
    } or set(parent_rows["parameter_count"].astype(int)) != {
        historical_candidate_parameters
    } or set(ablation_rows["parameter_count"].astype(int)) != {
        historical_candidate_parameters
    }:
        raise ContractError("v25 historical parameter counts drifted")
    if set(current_leaderboard["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v25 candidate parameter count drifted")
    if (
        set(current_leaderboard["dynamic_skip_token_normalization"].astype(str))
        != {"shape_log_rms"}
        or set(current_leaderboard["dynamic_skip_amplitude_feature"].astype(str))
        != {"log1p_rms"}
        or set(current_leaderboard["dynamic_skip_scorer_input_width"].astype(int))
        != {scorer_input_width}
        or set(
            current_leaderboard[
                "dynamic_skip_normalization_parameter_count"
            ].astype(int)
        )
        != {0}
        or set(current_leaderboard["optimizer_group_identity"].astype(str))
        != {f"all-lr-{learning_rate:g}"}
        or set(
            current_leaderboard[
                "optimizer_dynamic_skip_parameter_count"
            ].astype(int)
        )
        != {0}
    ):
        raise ContractError("v25 scorer input or optimizer identity drifted")

    key = ["seed", "fold"]
    current_values = current_leaderboard.set_index(key)["best_mean_daily_rankic"]
    control_values = control_rows.set_index(key)["best_mean_daily_rankic"]
    parent_values = parent_rows.set_index(key)["best_mean_daily_rankic"]
    ablation_values = ablation_rows.set_index(key)["best_mean_daily_rankic"]
    paired = pd.concat(
        {
            "candidate": current_values,
            "control": control_values,
            "parent": parent_values,
            "ablation": ablation_values,
        },
        axis=1,
    )
    paired["rankic_delta"] = paired["candidate"] - paired["control"]
    paired["parent_rankic_delta"] = paired["candidate"] - paired["parent"]
    paired["ablation_rankic_delta"] = (
        paired["candidate"] - paired["ablation"]
    )
    seed_summary = (
        paired.reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            candidate_mean_rankic=("candidate", "mean"),
            control_mean_rankic=("control", "mean"),
            parent_mean_rankic=("parent", "mean"),
            ablation_mean_rankic=("ablation", "mean"),
            mean_rankic_delta=("rankic_delta", "mean"),
            parent_mean_rankic_delta=("parent_rankic_delta", "mean"),
            ablation_mean_rankic_delta=("ablation_rankic_delta", "mean"),
            nondegrading_folds=(
                "rankic_delta", lambda values: int((values >= 0).sum())
            ),
        )
        .sort_values("seed", ignore_index=True)
    )
    horizon_gates = {
        1: min_horizon_delta_1d,
        2: min_horizon_delta_2d,
        3: min_horizon_delta_3d,
        5: min_horizon_delta_5d,
    }
    horizon_rows: list[dict[str, float | int]] = []
    for horizon in (1, 2, 3, 5):
        column = f"rankic_{horizon}d"
        control_mean = float(control_rows[column].mean())
        candidate_mean_horizon = float(current_leaderboard[column].mean())
        horizon_rows.append(
            {
                "horizon": horizon,
                "control_rankic": control_mean,
                "candidate_rankic": candidate_mean_horizon,
                "rankic_delta": candidate_mean_horizon - control_mean,
            }
        )
    horizon_summary = pd.DataFrame(horizon_rows)

    variation = current_diagnostics["block_weight_variation"].to_numpy(
        dtype="float64"
    )
    simplex = current_diagnostics["simplex_error_max"].to_numpy(dtype="float64")
    if (
        not np.isfinite(variation).all()
        or not np.isfinite(simplex).all()
        or bool((variation <= 0).any())
        or bool((simplex < 0).any())
    ):
        raise ContractError("v25 dynamic diagnostics are invalid")

    candidate_mean = float(current_values.mean())
    positive_units = int(current_values.gt(0).sum())
    mean_delta = float(paired["rankic_delta"].mean())
    parent_mean_delta = float(paired["parent_rankic_delta"].mean())
    ablation_mean_delta = float(paired["ablation_rankic_delta"].mean())
    median_throughput = float(current_leaderboard["samples_per_second"].median())
    output_l2_min = float(
        current_leaderboard["dynamic_skip_output_weight_l2"].min()
    )
    amplitude_l2_min = float(
        current_leaderboard[
            "dynamic_skip_amplitude_projection_weight_l2"
        ].min()
    )
    blockers: list[str] = []
    if candidate_mean < min_mean_rankic:
        blockers.append("mean_rankic_below_gate")
    if positive_units < min_positive_units:
        blockers.append("positive_units_below_gate")
    if mean_delta < min_mean_rankic_delta:
        blockers.append("mean_rankic_delta_below_gate")
    if parent_mean_delta < min_parent_mean_rankic_delta:
        blockers.append("parent_mean_rankic_delta_below_gate")
    if ablation_mean_delta < min_ablation_mean_rankic_delta:
        blockers.append("ablation_mean_rankic_delta_below_gate")
    if bool(seed_summary["mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_mean_delta_not_positive")
    if bool(seed_summary["parent_mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_parent_mean_delta_not_positive")
    if bool(seed_summary["ablation_mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_ablation_mean_delta_not_positive")
    if bool(
        seed_summary["nondegrading_folds"].lt(
            min_nondegrading_folds_per_seed
        ).any()
    ):
        blockers.append("per_seed_fold_stability_below_gate")
    for row in horizon_summary.itertuples(index=False):
        horizon = int(cast(Any, row.horizon))
        if float(cast(Any, row.rankic_delta)) < horizon_gates[horizon]:
            blockers.append(f"horizon_{horizon}d_degradation_below_gate")
    if median_throughput < min_median_samples_per_second:
        blockers.append("throughput_below_gate")
    if output_l2_min < min_dynamic_skip_output_weight_l2:
        blockers.append("dynamic_skip_not_used")
    if amplitude_l2_min < min_amplitude_projection_weight_l2:
        blockers.append("amplitude_projection_not_used")
    if float(variation.min()) < min_block_weight_variation:
        blockers.append("block_weights_not_sample_conditioned")
    if float(simplex.max()) > max_simplex_error:
        blockers.append("dynamic_skip_simplex_invalid")
    effect_passed = not blockers

    try:
        model_step_speed_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_speed_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v25 LSTM comparison is incomplete") from exc
    if not np.isfinite([model_step_speed_ratio, end_to_end_speed_ratio]).all():
        raise ContractError("v25 LSTM comparison is non-finite")
    speed_passed = bool(
        model_step_speed_ratio >= min_model_step_speed_ratio
        and end_to_end_speed_ratio >= min_end_to_end_speed_ratio
    )
    status = (
        "stop_dynamic_skip_shape_amplitude_unstable_v25"
        if not effect_passed
        else (
            "dynamic_skip_shape_amplitude_multiseed_confirmed_v25"
            if speed_passed
            else "stop_dynamic_skip_shape_amplitude_speed_v25"
        )
    )
    aggregate: dict[str, float | int | bool | str] = {
        "candidate_mean_rankic": candidate_mean,
        "positive_units": positive_units,
        "mean_rankic_delta": mean_delta,
        "parent_mean_rankic_delta": parent_mean_delta,
        "ablation_mean_rankic_delta": ablation_mean_delta,
        "median_samples_per_second": median_throughput,
        "dynamic_skip_output_weight_l2_min": output_l2_min,
        "dynamic_skip_amplitude_projection_weight_l2_min": amplitude_l2_min,
        "block_weight_variation_min": float(variation.min()),
        "simplex_error_max": float(simplex.max()),
        "model_step_speed_ratio": model_step_speed_ratio,
        "end_to_end_speed_ratio": end_to_end_speed_ratio,
        "control_parameter_count": control_parameter_count,
        "candidate_parameter_count": candidate_parameter_count,
        "dynamic_parameter_count": dynamic_parameter_count,
        "amplitude_parameter_count": amplitude_parameter_count,
        "dynamic_skip_scorer_input_width": scorer_input_width,
        "learning_rate": learning_rate,
        "dynamic_skip_token_normalization": "shape_log_rms",
        "dynamic_skip_amplitude_feature": "log1p_rms",
        "blockers": ",".join(blockers),
    }
    return DynamicSkipShapeAmplitudeMultiSeedDecision(
        status=status,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=horizon_summary,
    )


def evaluate_dynamic_skip_raw_shape_residual_multiseed(
    current_leaderboard: pd.DataFrame,
    historical_leaderboard: pd.DataFrame,
    current_diagnostics: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
    ablation_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_mean_rankic_delta: float,
    min_parent_mean_rankic_delta: float,
    min_ablation_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    min_dynamic_skip_output_weight_l2: float,
    min_shape_output_weight_l2: float,
    min_shape_residual_weight_effect: float,
    min_block_weight_variation: float,
    max_simplex_error: float,
    control_parameter_count: int,
    parent_parameter_count: int,
    ablation_parameter_count: int,
    candidate_parameter_count: int,
    dynamic_parameter_count: int,
    raw_parameter_count: int,
    shape_parameter_count: int,
    shape_residual_scale: float,
    learning_rate: float,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> DynamicSkipRawShapeResidualMultiSeedDecision:
    """Evaluate v26 against static, raw-token and v25 evidence."""

    current_required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "dynamic_skip_output_weight_l2",
        "dynamic_skip_shape_output_weight_l2",
        "dynamic_skip_token_normalization",
        "dynamic_skip_shape_residual",
        "dynamic_skip_shape_residual_scale",
        "dynamic_skip_raw_parameter_count",
        "dynamic_skip_shape_residual_parameter_count",
        "dynamic_skip_shape_normalization_parameter_count",
        "optimizer_group_identity",
        "optimizer_dynamic_skip_parameter_count",
    }
    historical_required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "parameter_count",
    }
    diagnostic_required = {
        "trial_id",
        "seed",
        "fold",
        "block_weight_variation",
        "shape_residual_weight_effect_max",
        "simplex_error_max",
    }
    if missing := sorted(current_required.difference(current_leaderboard.columns)):
        raise ContractError(
            "v26 current leaderboard missing columns: " + ", ".join(missing)
        )
    if missing := sorted(
        historical_required.difference(historical_leaderboard.columns)
    ):
        raise ContractError(
            "v26 historical leaderboard missing columns: " + ", ".join(missing)
        )
    if missing := sorted(diagnostic_required.difference(current_diagnostics.columns)):
        raise ContractError(
            "v26 current diagnostics missing columns: " + ", ".join(missing)
        )
    if expected_seeds != (7, 17, 27):
        raise ContractError("v26 expected seeds must be exactly 7, 17 and 27")
    expected_units = {
        (seed, fold) for seed in expected_seeds for fold in range(5)
    }

    def _units(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return {
            (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
            for row in frame.itertuples(index=False)
        }

    if (
        set(current_leaderboard["trial_id"].astype(str)) != {candidate_trial_id}
        or current_leaderboard.duplicated(["trial_id", "seed", "fold"]).any()
        or _units(current_leaderboard) != expected_units
    ):
        raise ContractError("v26 current leaderboard coverage drifted")
    historical_ids = {
        control_trial_id,
        parent_candidate_trial_id,
        ablation_trial_id,
    }
    if (
        set(historical_leaderboard["trial_id"].astype(str)) != historical_ids
        or historical_leaderboard.duplicated(["trial_id", "seed", "fold"]).any()
    ):
        raise ContractError("v26 historical leaderboard coverage drifted")
    for trial_id in historical_ids:
        rows = historical_leaderboard.loc[
            historical_leaderboard["trial_id"].astype(str).eq(trial_id)
        ]
        if _units(rows) != expected_units:
            raise ContractError("v26 historical leaderboard coverage drifted")
    if (
        set(current_diagnostics["trial_id"].astype(str)) != {candidate_trial_id}
        or current_diagnostics.duplicated(["trial_id", "seed", "fold"]).any()
        or _units(current_diagnostics) != expected_units
    ):
        raise ContractError("v26 current diagnostics coverage drifted")

    numeric_columns = [
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "dynamic_skip_output_weight_l2",
        "dynamic_skip_shape_output_weight_l2",
        "dynamic_skip_shape_residual_scale",
        "dynamic_skip_raw_parameter_count",
        "dynamic_skip_shape_residual_parameter_count",
        "dynamic_skip_shape_normalization_parameter_count",
        "optimizer_dynamic_skip_parameter_count",
    ]
    if not np.isfinite(
        current_leaderboard[numeric_columns].to_numpy(dtype="float64")
    ).all():
        raise ContractError("v26 current leaderboard contains non-finite values")
    control_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ]
    parent_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(
            parent_candidate_trial_id
        )
    ]
    ablation_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(ablation_trial_id)
    ]
    if (
        dynamic_parameter_count != 176
        or raw_parameter_count != 88
        or shape_parameter_count != 88
        or candidate_parameter_count - control_parameter_count
        != dynamic_parameter_count
        or raw_parameter_count + shape_parameter_count != dynamic_parameter_count
        or parent_parameter_count - control_parameter_count != raw_parameter_count
    ):
        raise ContractError("v26 shape/amplitude capacity contract drifted")
    if set(control_rows["parameter_count"].astype(int)) != {
        control_parameter_count
    } or set(parent_rows["parameter_count"].astype(int)) != {
        parent_parameter_count
    } or set(ablation_rows["parameter_count"].astype(int)) != {
        ablation_parameter_count
    }:
        raise ContractError("v26 historical parameter counts drifted")
    if set(current_leaderboard["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v26 candidate parameter count drifted")
    if (
        set(current_leaderboard["dynamic_skip_token_normalization"].astype(str))
        != {"none"}
        or set(current_leaderboard["dynamic_skip_shape_residual"].astype(bool))
        != {True}
        or set(
            current_leaderboard["dynamic_skip_shape_residual_scale"].astype(float)
        )
        != {shape_residual_scale}
        or set(current_leaderboard["dynamic_skip_raw_parameter_count"].astype(int))
        != {raw_parameter_count}
        or set(
            current_leaderboard[
                "dynamic_skip_shape_residual_parameter_count"
            ].astype(int)
        )
        != {shape_parameter_count}
        or set(
            current_leaderboard[
                "dynamic_skip_shape_normalization_parameter_count"
            ].astype(int)
        )
        != {0}
        or set(current_leaderboard["optimizer_group_identity"].astype(str))
        != {f"all-lr-{learning_rate:g}"}
        or set(
            current_leaderboard[
                "optimizer_dynamic_skip_parameter_count"
            ].astype(int)
        )
        != {0}
    ):
        raise ContractError("v26 scorer input or optimizer identity drifted")

    key = ["seed", "fold"]
    current_values = current_leaderboard.set_index(key)["best_mean_daily_rankic"]
    control_values = control_rows.set_index(key)["best_mean_daily_rankic"]
    parent_values = parent_rows.set_index(key)["best_mean_daily_rankic"]
    ablation_values = ablation_rows.set_index(key)["best_mean_daily_rankic"]
    paired = pd.concat(
        {
            "candidate": current_values,
            "control": control_values,
            "parent": parent_values,
            "ablation": ablation_values,
        },
        axis=1,
    )
    paired["rankic_delta"] = paired["candidate"] - paired["control"]
    paired["parent_rankic_delta"] = paired["candidate"] - paired["parent"]
    paired["ablation_rankic_delta"] = (
        paired["candidate"] - paired["ablation"]
    )
    seed_summary = (
        paired.reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            candidate_mean_rankic=("candidate", "mean"),
            control_mean_rankic=("control", "mean"),
            parent_mean_rankic=("parent", "mean"),
            ablation_mean_rankic=("ablation", "mean"),
            mean_rankic_delta=("rankic_delta", "mean"),
            parent_mean_rankic_delta=("parent_rankic_delta", "mean"),
            ablation_mean_rankic_delta=("ablation_rankic_delta", "mean"),
            nondegrading_folds=(
                "rankic_delta", lambda values: int((values >= 0).sum())
            ),
        )
        .sort_values("seed", ignore_index=True)
    )
    horizon_gates = {
        1: min_horizon_delta_1d,
        2: min_horizon_delta_2d,
        3: min_horizon_delta_3d,
        5: min_horizon_delta_5d,
    }
    horizon_rows: list[dict[str, float | int]] = []
    for horizon in (1, 2, 3, 5):
        column = f"rankic_{horizon}d"
        control_mean = float(control_rows[column].mean())
        candidate_mean_horizon = float(current_leaderboard[column].mean())
        horizon_rows.append(
            {
                "horizon": horizon,
                "control_rankic": control_mean,
                "candidate_rankic": candidate_mean_horizon,
                "rankic_delta": candidate_mean_horizon - control_mean,
            }
        )
    horizon_summary = pd.DataFrame(horizon_rows)

    variation = current_diagnostics["block_weight_variation"].to_numpy(
        dtype="float64"
    )
    shape_effect = current_diagnostics[
        "shape_residual_weight_effect_max"
    ].to_numpy(dtype="float64")
    simplex = current_diagnostics["simplex_error_max"].to_numpy(dtype="float64")
    if (
        not np.isfinite(variation).all()
        or not np.isfinite(shape_effect).all()
        or not np.isfinite(simplex).all()
        or bool((variation <= 0).any())
        or bool((shape_effect < 0).any())
        or bool((simplex < 0).any())
    ):
        raise ContractError("v26 dynamic diagnostics are invalid")

    candidate_mean = float(current_values.mean())
    positive_units = int(current_values.gt(0).sum())
    mean_delta = float(paired["rankic_delta"].mean())
    parent_mean_delta = float(paired["parent_rankic_delta"].mean())
    ablation_mean_delta = float(paired["ablation_rankic_delta"].mean())
    median_throughput = float(current_leaderboard["samples_per_second"].median())
    output_l2_min = float(
        current_leaderboard["dynamic_skip_output_weight_l2"].min()
    )
    shape_output_l2_min = float(
        current_leaderboard["dynamic_skip_shape_output_weight_l2"].min()
    )
    blockers: list[str] = []
    if candidate_mean < min_mean_rankic:
        blockers.append("mean_rankic_below_gate")
    if positive_units < min_positive_units:
        blockers.append("positive_units_below_gate")
    if mean_delta < min_mean_rankic_delta:
        blockers.append("mean_rankic_delta_below_gate")
    if parent_mean_delta < min_parent_mean_rankic_delta:
        blockers.append("parent_mean_rankic_delta_below_gate")
    if ablation_mean_delta < min_ablation_mean_rankic_delta:
        blockers.append("ablation_mean_rankic_delta_below_gate")
    if bool(seed_summary["mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_mean_delta_not_positive")
    if bool(seed_summary["parent_mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_parent_mean_delta_not_positive")
    if bool(seed_summary["ablation_mean_rankic_delta"].le(0).any()):
        blockers.append("per_seed_ablation_mean_delta_not_positive")
    if bool(
        seed_summary["nondegrading_folds"].lt(
            min_nondegrading_folds_per_seed
        ).any()
    ):
        blockers.append("per_seed_fold_stability_below_gate")
    for row in horizon_summary.itertuples(index=False):
        horizon = int(cast(Any, row.horizon))
        if float(cast(Any, row.rankic_delta)) < horizon_gates[horizon]:
            blockers.append(f"horizon_{horizon}d_degradation_below_gate")
    if median_throughput < min_median_samples_per_second:
        blockers.append("throughput_below_gate")
    if output_l2_min < min_dynamic_skip_output_weight_l2:
        blockers.append("dynamic_skip_not_used")
    if shape_output_l2_min < min_shape_output_weight_l2:
        blockers.append("shape_residual_not_trained")
    if float(shape_effect.min()) < min_shape_residual_weight_effect:
        blockers.append("shape_residual_has_no_weight_effect")
    if float(variation.min()) < min_block_weight_variation:
        blockers.append("block_weights_not_sample_conditioned")
    if float(simplex.max()) > max_simplex_error:
        blockers.append("dynamic_skip_simplex_invalid")
    effect_passed = not blockers

    try:
        model_step_speed_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_speed_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v26 LSTM comparison is incomplete") from exc
    if not np.isfinite([model_step_speed_ratio, end_to_end_speed_ratio]).all():
        raise ContractError("v26 LSTM comparison is non-finite")
    speed_passed = bool(
        model_step_speed_ratio >= min_model_step_speed_ratio
        and end_to_end_speed_ratio >= min_end_to_end_speed_ratio
    )
    status = (
        "stop_dynamic_skip_raw_shape_residual_unstable_v26"
        if not effect_passed
        else (
            "dynamic_skip_raw_shape_residual_multiseed_confirmed_v26"
            if speed_passed
            else "stop_dynamic_skip_raw_shape_residual_speed_v26"
        )
    )
    aggregate: dict[str, float | int | bool | str] = {
        "candidate_mean_rankic": candidate_mean,
        "positive_units": positive_units,
        "mean_rankic_delta": mean_delta,
        "parent_mean_rankic_delta": parent_mean_delta,
        "ablation_mean_rankic_delta": ablation_mean_delta,
        "median_samples_per_second": median_throughput,
        "dynamic_skip_output_weight_l2_min": output_l2_min,
        "dynamic_skip_shape_output_weight_l2_min": shape_output_l2_min,
        "shape_residual_weight_effect_min": float(shape_effect.min()),
        "shape_residual_weight_effect_max": float(shape_effect.max()),
        "block_weight_variation_min": float(variation.min()),
        "simplex_error_max": float(simplex.max()),
        "model_step_speed_ratio": model_step_speed_ratio,
        "end_to_end_speed_ratio": end_to_end_speed_ratio,
        "control_parameter_count": control_parameter_count,
        "candidate_parameter_count": candidate_parameter_count,
        "dynamic_parameter_count": dynamic_parameter_count,
        "raw_parameter_count": raw_parameter_count,
        "shape_parameter_count": shape_parameter_count,
        "shape_residual_scale": shape_residual_scale,
        "learning_rate": learning_rate,
        "dynamic_skip_token_normalization": "none",
        "dynamic_skip_shape_residual": True,
        "blockers": ",".join(blockers),
    }
    return DynamicSkipRawShapeResidualMultiSeedDecision(
        status=status,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=horizon_summary,
    )


def evaluate_frozen_parent_shape_residual_multiseed(
    current_leaderboard: pd.DataFrame,
    historical_leaderboard: pd.DataFrame,
    current_diagnostics: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
    v25_trial_id: str,
    v26_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_parent_mean_rankic_delta: float,
    min_control_mean_rankic_delta: float,
    min_v26_mean_rankic_delta: float,
    min_v25_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_parent_delta_1d: float,
    min_horizon_parent_delta_2d: float,
    min_horizon_parent_delta_3d: float,
    min_horizon_parent_delta_5d: float,
    max_parent_rankic_abs_error: float,
    max_parent_prediction_abs_error: float,
    min_trained_effect_units: int,
    min_shape_output_weight_l2: float,
    min_shape_residual_weight_effect: float,
    max_simplex_error: float,
    min_median_samples_per_second: float,
    candidate_parameter_count: int,
    trainable_parameter_count: int,
    frozen_parameter_count: int,
    shape_residual_scale: float,
    learning_rate: float,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FrozenParentShapeResidualMultiSeedDecision:
    """Evaluate v27 without allowing parent drift to masquerade as shape value."""

    current_required = {
        "trial_id",
        "seed",
        "fold",
        "best_epoch",
        "baseline_epoch",
        "baseline_mean_daily_rankic",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "trainable_parameter_count",
        "frozen_parameter_count",
        "frozen_parent_state_drift_max",
        "parent_prediction_max_abs_error",
        "dynamic_skip_shape_output_weight_l2",
        "dynamic_skip_frozen_parent",
        "dynamic_skip_shape_residual",
        "dynamic_skip_shape_residual_scale",
        "dynamic_skip_raw_parameter_count",
        "dynamic_skip_shape_residual_parameter_count",
        "dynamic_skip_shape_normalization_parameter_count",
        "optimizer_group_identity",
        "parent_checkpoint_sha256",
    }
    historical_required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "parameter_count",
    }
    diagnostic_required = {
        "trial_id",
        "seed",
        "fold",
        "raw_only_mean_daily_rankic",
        "shape_residual_weight_effect_max",
        "simplex_error_max",
    }
    if missing := sorted(current_required.difference(current_leaderboard.columns)):
        raise ContractError(
            "v27 current leaderboard missing columns: " + ", ".join(missing)
        )
    if missing := sorted(
        historical_required.difference(historical_leaderboard.columns)
    ):
        raise ContractError(
            "v27 historical leaderboard missing columns: " + ", ".join(missing)
        )
    if missing := sorted(diagnostic_required.difference(current_diagnostics.columns)):
        raise ContractError(
            "v27 diagnostics missing columns: " + ", ".join(missing)
        )
    if expected_seeds != (7, 17, 27):
        raise ContractError("v27 expected seeds must be exactly 7, 17 and 27")
    expected_units = {
        (seed, fold) for seed in expected_seeds for fold in range(5)
    }

    def _units(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return {
            (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
            for row in frame.itertuples(index=False)
        }

    if (
        set(current_leaderboard["trial_id"].astype(str)) != {candidate_trial_id}
        or current_leaderboard.duplicated(["trial_id", "seed", "fold"]).any()
        or _units(current_leaderboard) != expected_units
    ):
        raise ContractError("v27 current leaderboard coverage drifted")
    if (
        set(current_diagnostics["trial_id"].astype(str)) != {candidate_trial_id}
        or current_diagnostics.duplicated(["trial_id", "seed", "fold"]).any()
        or _units(current_diagnostics) != expected_units
    ):
        raise ContractError("v27 diagnostics coverage drifted")
    historical_ids = {
        control_trial_id,
        parent_candidate_trial_id,
        v25_trial_id,
        v26_trial_id,
    }
    if set(historical_leaderboard["trial_id"].astype(str)) != historical_ids:
        raise ContractError("v27 historical identities drifted")
    for trial_id in historical_ids:
        rows = historical_leaderboard.loc[
            historical_leaderboard["trial_id"].astype(str).eq(trial_id)
        ]
        if (
            rows.duplicated(["trial_id", "seed", "fold"]).any()
            or _units(rows) != expected_units
        ):
            raise ContractError("v27 historical coverage drifted")

    numeric_columns = [
        "best_epoch",
        "baseline_epoch",
        "baseline_mean_daily_rankic",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "samples_per_second",
        "parameter_count",
        "trainable_parameter_count",
        "frozen_parameter_count",
        "frozen_parent_state_drift_max",
        "parent_prediction_max_abs_error",
        "dynamic_skip_shape_output_weight_l2",
        "dynamic_skip_shape_residual_scale",
        "dynamic_skip_raw_parameter_count",
        "dynamic_skip_shape_residual_parameter_count",
        "dynamic_skip_shape_normalization_parameter_count",
    ]
    if not np.isfinite(
        current_leaderboard[numeric_columns].to_numpy(dtype="float64")
    ).all():
        raise ContractError("v27 current leaderboard contains non-finite values")
    diagnostic_numbers = current_diagnostics[
        [
            "raw_only_mean_daily_rankic",
            "shape_residual_weight_effect_max",
            "simplex_error_max",
        ]
    ].to_numpy(dtype="float64")
    if not np.isfinite(diagnostic_numbers).all():
        raise ContractError("v27 diagnostics contain non-finite values")

    key = ["seed", "fold"]
    current_values = current_leaderboard.set_index(key)[
        "best_mean_daily_rankic"
    ]

    def _history(trial_id: str) -> pd.DataFrame:
        return historical_leaderboard.loc[
            historical_leaderboard["trial_id"].astype(str).eq(trial_id)
        ].copy()

    control_rows = _history(control_trial_id)
    parent_rows = _history(parent_candidate_trial_id)
    v25_rows = _history(v25_trial_id)
    v26_rows = _history(v26_trial_id)
    paired = pd.concat(
        {
            "candidate": current_values,
            "control": control_rows.set_index(key)["best_mean_daily_rankic"],
            "parent": parent_rows.set_index(key)["best_mean_daily_rankic"],
            "v25": v25_rows.set_index(key)["best_mean_daily_rankic"],
            "v26": v26_rows.set_index(key)["best_mean_daily_rankic"],
        },
        axis=1,
    )
    paired["parent_delta"] = paired["candidate"] - paired["parent"]
    paired["control_delta"] = paired["candidate"] - paired["control"]
    paired["v25_delta"] = paired["candidate"] - paired["v25"]
    paired["v26_delta"] = paired["candidate"] - paired["v26"]
    seed_summary = (
        paired.reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            candidate_mean_rankic=("candidate", "mean"),
            parent_mean_rankic=("parent", "mean"),
            parent_mean_rankic_delta=("parent_delta", "mean"),
            control_mean_rankic_delta=("control_delta", "mean"),
            v25_mean_rankic_delta=("v25_delta", "mean"),
            v26_mean_rankic_delta=("v26_delta", "mean"),
            nondegrading_folds=(
                "parent_delta", lambda values: int((values >= 0).sum())
            ),
        )
        .sort_values("seed", ignore_index=True)
    )
    horizon_gates = {
        1: min_horizon_parent_delta_1d,
        2: min_horizon_parent_delta_2d,
        3: min_horizon_parent_delta_3d,
        5: min_horizon_parent_delta_5d,
    }
    horizon_rows: list[dict[str, float | int]] = []
    for horizon in (1, 2, 3, 5):
        column = f"rankic_{horizon}d"
        parent_mean = float(parent_rows[column].mean())
        candidate_mean = float(current_leaderboard[column].mean())
        horizon_rows.append(
            {
                "horizon": horizon,
                "parent_rankic": parent_mean,
                "candidate_rankic": candidate_mean,
                "parent_rankic_delta": candidate_mean - parent_mean,
            }
        )
    horizon_summary = pd.DataFrame(horizon_rows)

    parent_values = parent_rows.set_index(key)["best_mean_daily_rankic"]
    baseline_values = current_leaderboard.set_index(key)[
        "baseline_mean_daily_rankic"
    ]
    raw_only_values = current_diagnostics.set_index(key)[
        "raw_only_mean_daily_rankic"
    ]
    baseline_parent_error = float(
        np.max(np.abs(baseline_values - parent_values))
    )
    raw_only_parent_error = float(
        np.max(np.abs(raw_only_values - parent_values))
    )
    prediction_error = float(
        current_leaderboard["parent_prediction_max_abs_error"].max()
    )
    state_drift = float(
        current_leaderboard["frozen_parent_state_drift_max"].max()
    )
    hashes = current_leaderboard["parent_checkpoint_sha256"].astype(str)
    integrity_blockers: list[str] = []
    if state_drift != 0.0:
        integrity_blockers.append("frozen_parent_state_drifted")
    if prediction_error > max_parent_prediction_abs_error:
        integrity_blockers.append("parent_prediction_drifted")
    if (
        baseline_parent_error > max_parent_rankic_abs_error
        or raw_only_parent_error > max_parent_rankic_abs_error
    ):
        integrity_blockers.append("parent_rankic_drifted")
    if (
        set(current_leaderboard["baseline_epoch"].astype(int)) != {0}
        or bool(cast(Any, current_leaderboard["best_epoch"]).astype(int).lt(0).any())
    ):
        integrity_blockers.append("epoch_zero_baseline_missing")
    if (
        set(current_leaderboard["parameter_count"].astype(int))
        != {candidate_parameter_count}
        or set(current_leaderboard["trainable_parameter_count"].astype(int))
        != {trainable_parameter_count}
        or set(current_leaderboard["frozen_parameter_count"].astype(int))
        != {frozen_parameter_count}
        or candidate_parameter_count
        != trainable_parameter_count + frozen_parameter_count
    ):
        integrity_blockers.append("frozen_capacity_drifted")
    if (
        set(current_leaderboard["dynamic_skip_frozen_parent"].astype(bool))
        != {True}
        or set(current_leaderboard["dynamic_skip_shape_residual"].astype(bool))
        != {True}
        or set(
            current_leaderboard[
                "dynamic_skip_shape_residual_scale"
            ].astype(float)
        )
        != {shape_residual_scale}
        or set(
            current_leaderboard["dynamic_skip_raw_parameter_count"].astype(int)
        )
        != {88}
        or set(
            current_leaderboard[
                "dynamic_skip_shape_residual_parameter_count"
            ].astype(int)
        )
        != {88}
        or set(
            current_leaderboard[
                "dynamic_skip_shape_normalization_parameter_count"
            ].astype(int)
        )
        != {0}
        or set(current_leaderboard["optimizer_group_identity"].astype(str))
        != {f"shape-residual-only-lr-{learning_rate:g}"}
    ):
        integrity_blockers.append("frozen_training_identity_drifted")
    if not bool(cast(Any, hashes).str.fullmatch(r"[0-9a-f]{64}").all()):
        integrity_blockers.append("parent_checkpoint_identity_invalid")
    integrity_passed = not integrity_blockers

    candidate_mean = float(current_values.mean())
    parent_mean_delta = float(paired["parent_delta"].mean())
    control_mean_delta = float(paired["control_delta"].mean())
    v25_mean_delta = float(paired["v25_delta"].mean())
    v26_mean_delta = float(paired["v26_delta"].mean())
    positive_units = int(current_values.gt(0).sum())
    shape_l2 = current_leaderboard[
        "dynamic_skip_shape_output_weight_l2"
    ].to_numpy(dtype="float64")
    shape_effect = current_diagnostics[
        "shape_residual_weight_effect_max"
    ].to_numpy(dtype="float64")
    trained_effect_units = int(
        np.sum(
            (shape_l2 > min_shape_output_weight_l2)
            & (shape_effect >= min_shape_residual_weight_effect)
        )
    )
    simplex_error = float(current_diagnostics["simplex_error_max"].max())
    median_throughput = float(
        current_leaderboard["samples_per_second"].median()
    )
    effect_blockers: list[str] = []
    if candidate_mean < min_mean_rankic:
        effect_blockers.append("mean_rankic_below_gate")
    if positive_units < min_positive_units:
        effect_blockers.append("positive_units_below_gate")
    if parent_mean_delta < min_parent_mean_rankic_delta:
        effect_blockers.append("parent_mean_rankic_delta_below_gate")
    if control_mean_delta < min_control_mean_rankic_delta:
        effect_blockers.append("control_mean_rankic_delta_below_gate")
    if v26_mean_delta < min_v26_mean_rankic_delta:
        effect_blockers.append("v26_mean_rankic_delta_below_gate")
    if v25_mean_delta < min_v25_mean_rankic_delta:
        effect_blockers.append("v25_mean_rankic_delta_below_gate")
    if bool(seed_summary["parent_mean_rankic_delta"].lt(0).any()):
        effect_blockers.append("per_seed_parent_delta_negative")
    if bool(
        seed_summary["nondegrading_folds"].lt(
            min_nondegrading_folds_per_seed
        ).any()
    ):
        effect_blockers.append("per_seed_parent_stability_below_gate")
    for row in horizon_summary.itertuples(index=False):
        horizon = int(cast(Any, row.horizon))
        if (
            float(cast(Any, row.parent_rankic_delta))
            < horizon_gates[horizon]
        ):
            effect_blockers.append(
                f"horizon_{horizon}d_parent_degradation_below_gate"
            )
    if trained_effect_units < min_trained_effect_units:
        effect_blockers.append("shape_effect_unit_count_below_gate")
    if simplex_error > max_simplex_error:
        effect_blockers.append("dynamic_skip_simplex_invalid")
    if median_throughput < min_median_samples_per_second:
        effect_blockers.append("throughput_below_gate")
    effect_passed = not effect_blockers

    try:
        model_step_speed_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_speed_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v27 LSTM comparison is incomplete") from exc
    if not np.isfinite([model_step_speed_ratio, end_to_end_speed_ratio]).all():
        raise ContractError("v27 LSTM comparison is non-finite")
    speed_passed = bool(
        model_step_speed_ratio >= min_model_step_speed_ratio
        and end_to_end_speed_ratio >= min_end_to_end_speed_ratio
    )
    if not integrity_passed:
        status = "stop_frozen_parent_integrity_v27"
    elif not effect_passed:
        status = "stop_frozen_parent_shape_residual_no_gain_v27"
    elif not speed_passed:
        status = "stop_frozen_parent_shape_residual_speed_v27"
    else:
        status = "frozen_parent_shape_residual_confirmed_v27"
    blockers = [*integrity_blockers, *effect_blockers]
    aggregate: dict[str, float | int | bool | str] = {
        "candidate_mean_rankic": candidate_mean,
        "positive_units": positive_units,
        "parent_mean_rankic_delta": parent_mean_delta,
        "control_mean_rankic_delta": control_mean_delta,
        "v25_mean_rankic_delta": v25_mean_delta,
        "v26_mean_rankic_delta": v26_mean_delta,
        "baseline_parent_rankic_abs_error_max": baseline_parent_error,
        "raw_only_parent_rankic_abs_error_max": raw_only_parent_error,
        "parent_prediction_abs_error_max": prediction_error,
        "frozen_parent_state_drift_max": state_drift,
        "trained_effect_units": trained_effect_units,
        "simplex_error_max": simplex_error,
        "median_samples_per_second": median_throughput,
        "model_step_speed_ratio": model_step_speed_ratio,
        "end_to_end_speed_ratio": end_to_end_speed_ratio,
        "candidate_parameter_count": candidate_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
        "shape_residual_scale": shape_residual_scale,
        "learning_rate": learning_rate,
        "blockers": ",".join(blockers),
    }
    return FrozenParentShapeResidualMultiSeedDecision(
        status=status,
        integrity_passed=integrity_passed,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=horizon_summary,
    )


def evaluate_decoupled_checkpoint_selection_multiseed(
    current_leaderboard: pd.DataFrame,
    historical_leaderboard: pd.DataFrame,
    current_diagnostics: pd.DataFrame,
    current_epoch_history: pd.DataFrame,
    v27_epoch_history: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
    v25_trial_id: str,
    v26_trial_id: str,
    v27_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_parent_mean_rankic_delta: float,
    min_control_mean_rankic_delta: float,
    min_v26_mean_rankic_delta: float,
    min_v25_mean_rankic_delta: float,
    min_v27_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_parent_delta_1d: float,
    min_horizon_parent_delta_2d: float,
    min_horizon_parent_delta_3d: float,
    min_horizon_parent_delta_5d: float,
    max_trajectory_rankic_abs_error: float,
    max_selected_best_abs_error: float,
    max_parent_rankic_abs_error: float,
    max_parent_prediction_abs_error: float,
    min_trained_effect_units: int,
    min_shape_output_weight_l2: float,
    min_shape_residual_weight_effect: float,
    max_simplex_error: float,
    min_median_samples_per_second: float,
    candidate_parameter_count: int,
    trainable_parameter_count: int,
    frozen_parameter_count: int,
    shape_residual_scale: float,
    learning_rate: float,
    checkpoint_min_delta: float,
    patience_min_delta: float,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> DecoupledCheckpointSelectionMultiSeedDecision:
    """Evaluate v28 only when training scores are identical to frozen v27."""

    required_selection = {
        "checkpoint_min_delta",
        "patience_min_delta",
        "checkpoint_selection_identity",
    }
    if missing := sorted(required_selection.difference(current_leaderboard.columns)):
        raise ContractError(
            "v28 leaderboard missing selection columns: " + ", ".join(missing)
        )
    epoch_required = {
        "trial_id",
        "seed",
        "fold",
        "epoch",
        "mean_daily_rankic",
    }
    if missing := sorted(epoch_required.difference(current_epoch_history.columns)):
        raise ContractError(
            "v28 epoch history missing columns: " + ", ".join(missing)
        )
    if missing := sorted(epoch_required.difference(v27_epoch_history.columns)):
        raise ContractError(
            "v27 epoch history missing columns: " + ", ".join(missing)
        )
    if set(current_epoch_history["trial_id"].astype(str)) != {
        candidate_trial_id
    }:
        raise ContractError("v28 epoch trial identity drifted")
    if set(v27_epoch_history["trial_id"].astype(str)) != {v27_trial_id}:
        raise ContractError("v27 epoch trial identity drifted")

    base_history = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).isin(
            [
                control_trial_id,
                parent_candidate_trial_id,
                v25_trial_id,
                v26_trial_id,
            ]
        )
    ].copy()
    base = evaluate_frozen_parent_shape_residual_multiseed(
        current_leaderboard,
        base_history,
        current_diagnostics,
        comparison,
        control_trial_id=control_trial_id,
        parent_candidate_trial_id=parent_candidate_trial_id,
        v25_trial_id=v25_trial_id,
        v26_trial_id=v26_trial_id,
        candidate_trial_id=candidate_trial_id,
        expected_seeds=expected_seeds,
        min_mean_rankic=min_mean_rankic,
        min_positive_units=min_positive_units,
        min_parent_mean_rankic_delta=min_parent_mean_rankic_delta,
        min_control_mean_rankic_delta=min_control_mean_rankic_delta,
        min_v26_mean_rankic_delta=min_v26_mean_rankic_delta,
        min_v25_mean_rankic_delta=min_v25_mean_rankic_delta,
        min_nondegrading_folds_per_seed=min_nondegrading_folds_per_seed,
        min_horizon_parent_delta_1d=min_horizon_parent_delta_1d,
        min_horizon_parent_delta_2d=min_horizon_parent_delta_2d,
        min_horizon_parent_delta_3d=min_horizon_parent_delta_3d,
        min_horizon_parent_delta_5d=min_horizon_parent_delta_5d,
        max_parent_rankic_abs_error=max_parent_rankic_abs_error,
        max_parent_prediction_abs_error=max_parent_prediction_abs_error,
        min_trained_effect_units=min_trained_effect_units,
        min_shape_output_weight_l2=min_shape_output_weight_l2,
        min_shape_residual_weight_effect=min_shape_residual_weight_effect,
        max_simplex_error=max_simplex_error,
        min_median_samples_per_second=min_median_samples_per_second,
        candidate_parameter_count=candidate_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        frozen_parameter_count=frozen_parameter_count,
        shape_residual_scale=shape_residual_scale,
        learning_rate=learning_rate,
        min_model_step_speed_ratio=min_model_step_speed_ratio,
        min_end_to_end_speed_ratio=min_end_to_end_speed_ratio,
    )

    key = ["seed", "fold", "epoch"]
    if (
        current_epoch_history.duplicated(key).any()
        or v27_epoch_history.duplicated(key).any()
    ):
        raise ContractError("v28 differential epoch coverage contains duplicates")
    current_scores = current_epoch_history.set_index(key)["mean_daily_rankic"]
    v27_scores = v27_epoch_history.set_index(key)["mean_daily_rankic"]
    trajectory_coverage_match = bool(current_scores.index.equals(v27_scores.index))
    trajectory_error = float("inf")
    if trajectory_coverage_match:
        if not np.isfinite(
            current_scores.to_numpy(dtype="float64")
        ).all() or not np.isfinite(v27_scores.to_numpy(dtype="float64")).all():
            raise ContractError("v28 differential epoch scores are non-finite")
        trajectory_error = float(np.max(np.abs(current_scores - v27_scores)))

    selected_values = current_leaderboard.set_index(["seed", "fold"])[
        "best_mean_daily_rankic"
    ]
    observed_maxima = current_epoch_history.groupby(
        ["seed", "fold"], observed=True
    )["mean_daily_rankic"].max()
    if not selected_values.index.equals(observed_maxima.index):
        raise ContractError("v28 selected checkpoint coverage drifted")
    selected_best_error = float(
        np.max(np.abs(selected_values - observed_maxima))
    )
    expected_selection_identity = (
        "best-any-strict-improvement+patience-material-"
        f"{patience_min_delta:g}"
    )
    selection_identity_passed = bool(
        set(current_leaderboard["checkpoint_min_delta"].astype(float))
        == {checkpoint_min_delta}
        and set(current_leaderboard["patience_min_delta"].astype(float))
        == {patience_min_delta}
        and set(
            current_leaderboard["checkpoint_selection_identity"].astype(str)
        )
        == {expected_selection_identity}
    )

    v27_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(v27_trial_id)
    ].copy()
    expected_units = {
        (seed, fold) for seed in expected_seeds for fold in range(5)
    }
    observed_v27_units = {
        (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
        for row in v27_rows.itertuples(index=False)
    }
    if (
        observed_v27_units != expected_units
        or v27_rows.duplicated(["trial_id", "seed", "fold"]).any()
    ):
        raise ContractError("v28 v27 leaderboard coverage drifted")
    v27_values = v27_rows.set_index(["seed", "fold"])[
        "best_mean_daily_rankic"
    ]
    v27_deltas = selected_values - v27_values
    v27_mean_delta = float(v27_deltas.mean())
    v27_seed_delta = (
        v27_deltas.rename("v27_mean_rankic_delta")
        .reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(v27_mean_rankic_delta=("v27_mean_rankic_delta", "mean"))
    )
    seed_summary = base.seed_summary.merge(
        v27_seed_delta, on="seed", how="left", validate="one_to_one"
    )

    integrity_blockers: list[str] = []
    if not base.integrity_passed:
        integrity_blockers.append("frozen_parent_integrity_failed")
    if (
        not trajectory_coverage_match
        or trajectory_error > max_trajectory_rankic_abs_error
    ):
        integrity_blockers.append("v27_training_trajectory_drifted")
    if selected_best_error > max_selected_best_abs_error:
        integrity_blockers.append("selected_checkpoint_is_not_observed_max")
    if not selection_identity_passed:
        integrity_blockers.append("checkpoint_selection_identity_drifted")
    integrity_passed = not integrity_blockers

    effect_blockers: list[str] = []
    if not base.effect_passed:
        base_blockers = str(base.aggregate.get("blockers", ""))
        if base_blockers:
            effect_blockers.extend(base_blockers.split(","))
        else:
            effect_blockers.append("frozen_shape_effect_failed")
    if v27_mean_delta < min_v27_mean_rankic_delta:
        effect_blockers.append("v27_mean_rankic_delta_below_gate")
    if bool(seed_summary["v27_mean_rankic_delta"].lt(0).any()):
        effect_blockers.append("per_seed_v27_delta_negative")
    effect_passed = not effect_blockers
    speed_passed = base.speed_passed
    if not integrity_passed:
        status = "stop_decoupled_checkpoint_integrity_v28"
    elif not effect_passed:
        status = "stop_decoupled_checkpoint_no_gain_v28"
    elif not speed_passed:
        status = "stop_decoupled_checkpoint_speed_v28"
    else:
        status = "decoupled_checkpoint_selection_confirmed_v28"
    aggregate = dict(base.aggregate)
    aggregate.update(
        {
            "v27_mean_rankic_delta": v27_mean_delta,
            "trajectory_coverage_match": trajectory_coverage_match,
            "trajectory_rankic_abs_error_max": trajectory_error,
            "selected_best_abs_error_max": selected_best_error,
            "checkpoint_min_delta": checkpoint_min_delta,
            "patience_min_delta": patience_min_delta,
            "checkpoint_selection_identity": expected_selection_identity,
            "blockers": ",".join(
                [*integrity_blockers, *dict.fromkeys(effect_blockers)]
            ),
        }
    )
    return DecoupledCheckpointSelectionMultiSeedDecision(
        status=status,
        integrity_passed=integrity_passed,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=base.horizon_summary,
    )


def evaluate_frozen_shape_learning_rate_multiseed(
    current_leaderboard: pd.DataFrame,
    historical_leaderboard: pd.DataFrame,
    current_diagnostics: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
    v25_trial_id: str,
    v26_trial_id: str,
    v28_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_parent_mean_rankic_delta: float,
    min_control_mean_rankic_delta: float,
    min_v26_mean_rankic_delta: float,
    min_v25_mean_rankic_delta: float,
    min_v28_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_parent_delta_1d: float,
    min_horizon_parent_delta_2d: float,
    min_horizon_parent_delta_3d: float,
    min_horizon_parent_delta_5d: float,
    max_parent_rankic_abs_error: float,
    max_parent_prediction_abs_error: float,
    min_trained_effect_units: int,
    min_shape_output_weight_l2: float,
    min_shape_residual_weight_effect: float,
    max_simplex_error: float,
    min_median_samples_per_second: float,
    candidate_parameter_count: int,
    trainable_parameter_count: int,
    frozen_parameter_count: int,
    shape_residual_scale: float,
    learning_rate: float,
    checkpoint_min_delta: float,
    patience_min_delta: float,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FrozenShapeLearningRateMultiSeedDecision:
    """Evaluate the v29 shape-only learning-rate probe against frozen v28."""

    required_training_identity = {
        "learning_rate",
        "checkpoint_min_delta",
        "patience_min_delta",
        "checkpoint_selection_identity",
    }
    if missing := sorted(
        required_training_identity.difference(current_leaderboard.columns)
    ):
        raise ContractError(
            "v29 leaderboard missing training identity columns: "
            + ", ".join(missing)
        )
    if expected_seeds != (7, 17, 27):
        raise ContractError("v29 expected seeds must be exactly 7, 17 and 27")

    base_ids = {
        control_trial_id,
        parent_candidate_trial_id,
        v25_trial_id,
        v26_trial_id,
    }
    base_history = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).isin(base_ids)
    ].copy()
    base = evaluate_frozen_parent_shape_residual_multiseed(
        current_leaderboard,
        base_history,
        current_diagnostics,
        comparison,
        control_trial_id=control_trial_id,
        parent_candidate_trial_id=parent_candidate_trial_id,
        v25_trial_id=v25_trial_id,
        v26_trial_id=v26_trial_id,
        candidate_trial_id=candidate_trial_id,
        expected_seeds=expected_seeds,
        min_mean_rankic=min_mean_rankic,
        min_positive_units=min_positive_units,
        min_parent_mean_rankic_delta=min_parent_mean_rankic_delta,
        min_control_mean_rankic_delta=min_control_mean_rankic_delta,
        min_v26_mean_rankic_delta=min_v26_mean_rankic_delta,
        min_v25_mean_rankic_delta=min_v25_mean_rankic_delta,
        min_nondegrading_folds_per_seed=min_nondegrading_folds_per_seed,
        min_horizon_parent_delta_1d=min_horizon_parent_delta_1d,
        min_horizon_parent_delta_2d=min_horizon_parent_delta_2d,
        min_horizon_parent_delta_3d=min_horizon_parent_delta_3d,
        min_horizon_parent_delta_5d=min_horizon_parent_delta_5d,
        max_parent_rankic_abs_error=max_parent_rankic_abs_error,
        max_parent_prediction_abs_error=max_parent_prediction_abs_error,
        min_trained_effect_units=min_trained_effect_units,
        min_shape_output_weight_l2=min_shape_output_weight_l2,
        min_shape_residual_weight_effect=min_shape_residual_weight_effect,
        max_simplex_error=max_simplex_error,
        min_median_samples_per_second=min_median_samples_per_second,
        candidate_parameter_count=candidate_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        frozen_parameter_count=frozen_parameter_count,
        shape_residual_scale=shape_residual_scale,
        learning_rate=learning_rate,
        min_model_step_speed_ratio=min_model_step_speed_ratio,
        min_end_to_end_speed_ratio=min_end_to_end_speed_ratio,
    )

    expected_units = {
        (seed, fold) for seed in expected_seeds for fold in range(5)
    }
    v28_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(v28_trial_id)
    ].copy()
    observed_v28_units = {
        (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
        for row in v28_rows.itertuples(index=False)
    }
    if (
        observed_v28_units != expected_units
        or v28_rows.duplicated(["trial_id", "seed", "fold"]).any()
    ):
        raise ContractError("v29 v28 leaderboard coverage drifted")
    if set(historical_leaderboard["trial_id"].astype(str)) != {
        *base_ids,
        v28_trial_id,
    }:
        raise ContractError("v29 historical identities drifted")
    if not np.isfinite(
        v28_rows["best_mean_daily_rankic"].to_numpy(dtype="float64")
    ).all():
        raise ContractError("v29 v28 leaderboard contains non-finite values")

    key = ["seed", "fold"]
    current_values = current_leaderboard.set_index(key)[
        "best_mean_daily_rankic"
    ]
    v28_values = v28_rows.set_index(key)["best_mean_daily_rankic"]
    v28_deltas = current_values - v28_values
    if v28_deltas.isna().any():
        raise ContractError("v29 paired v28 coverage drifted")
    v28_mean_delta = float(v28_deltas.mean())
    v28_seed = (
        pd.concat(
            {
                "v28_mean_rankic": v28_values,
                "v28_mean_rankic_delta": v28_deltas,
            },
            axis=1,
        )
        .reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            v28_mean_rankic=("v28_mean_rankic", "mean"),
            v28_mean_rankic_delta=("v28_mean_rankic_delta", "mean"),
        )
    )
    seed_summary = base.seed_summary.merge(
        v28_seed, on="seed", how="left", validate="one_to_one"
    )

    expected_selection_identity = (
        "best-any-strict-improvement+patience-material-"
        f"{patience_min_delta:g}"
    )
    training_identity_passed = bool(
        set(current_leaderboard["learning_rate"].astype(float))
        == {learning_rate}
        and set(current_leaderboard["checkpoint_min_delta"].astype(float))
        == {checkpoint_min_delta}
        and set(current_leaderboard["patience_min_delta"].astype(float))
        == {patience_min_delta}
        and set(
            current_leaderboard["checkpoint_selection_identity"].astype(str)
        )
        == {expected_selection_identity}
    )
    integrity_blockers: list[str] = []
    if not base.integrity_passed:
        integrity_blockers.append("frozen_parent_integrity_failed")
    if not training_identity_passed:
        integrity_blockers.append("frozen_shape_lr_training_identity_drifted")
    integrity_passed = not integrity_blockers

    effect_blockers: list[str] = []
    if not base.effect_passed:
        base_blockers = str(base.aggregate.get("blockers", ""))
        effect_blockers.extend(
            blocker for blocker in base_blockers.split(",") if blocker
        )
        if not base_blockers:
            effect_blockers.append("frozen_shape_effect_failed")
    if v28_mean_delta < min_v28_mean_rankic_delta:
        effect_blockers.append("v28_mean_rankic_delta_below_gate")
    if bool(seed_summary["v28_mean_rankic_delta"].lt(0).any()):
        effect_blockers.append("per_seed_v28_delta_negative")
    effect_passed = not effect_blockers
    speed_passed = base.speed_passed

    if not integrity_passed:
        status = "stop_frozen_shape_lr_integrity_v29"
    elif not effect_passed:
        status = "stop_frozen_shape_lr_no_gain_v29"
    elif not speed_passed:
        status = "stop_frozen_shape_lr_speed_v29"
    else:
        status = "frozen_shape_lr001_confirmed_v29"

    aggregate = dict(base.aggregate)
    aggregate.update(
        {
            "v28_mean_rankic_delta": v28_mean_delta,
            "checkpoint_min_delta": checkpoint_min_delta,
            "patience_min_delta": patience_min_delta,
            "checkpoint_selection_identity": expected_selection_identity,
            "blockers": ",".join(
                [*integrity_blockers, *dict.fromkeys(effect_blockers)]
            ),
        }
    )
    return FrozenShapeLearningRateMultiSeedDecision(
        status=status,
        integrity_passed=integrity_passed,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=base.horizon_summary,
    )


def evaluate_frozen_shape_soft_rankic_multiseed(
    current_leaderboard: pd.DataFrame,
    historical_leaderboard: pd.DataFrame,
    current_diagnostics: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
    v25_trial_id: str,
    v26_trial_id: str,
    v28_trial_id: str,
    grouped_control_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_parent_mean_rankic_delta: float,
    min_control_mean_rankic_delta: float,
    min_v26_mean_rankic_delta: float,
    min_v25_mean_rankic_delta: float,
    min_v28_mean_rankic_delta: float,
    min_grouped_control_mean_rankic_delta: float,
    min_nondegrading_folds_per_seed: int,
    min_horizon_parent_delta_1d: float,
    min_horizon_parent_delta_2d: float,
    min_horizon_parent_delta_3d: float,
    min_horizon_parent_delta_5d: float,
    max_parent_rankic_abs_error: float,
    max_parent_prediction_abs_error: float,
    min_trained_effect_units: int,
    min_shape_output_weight_l2: float,
    min_shape_residual_weight_effect: float,
    max_simplex_error: float,
    min_median_samples_per_second: float,
    candidate_parameter_count: int,
    trainable_parameter_count: int,
    frozen_parameter_count: int,
    shape_residual_scale: float,
    learning_rate: float,
    soft_rankic_weight: float,
    soft_rank_temperature: float,
    checkpoint_min_delta: float,
    patience_min_delta: float,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FrozenShapeSoftRankICMultiSeedDecision:
    """Evaluate v30 while separating date-grouped batching from rank loss."""

    identity_columns = {
        "strategy",
        "loss_identity",
        "batching_identity",
        "soft_rankic_weight",
        "soft_rank_temperature",
        "learning_rate",
        "checkpoint_min_delta",
        "patience_min_delta",
        "checkpoint_selection_identity",
    }
    if missing := sorted(identity_columns.difference(current_leaderboard.columns)):
        raise ContractError(
            "v30 leaderboard missing objective identity columns: "
            + ", ".join(missing)
        )
    if expected_seeds != (7, 17, 27):
        raise ContractError("v30 expected seeds must be exactly 7, 17 and 27")

    candidate_rows = current_leaderboard.loc[
        current_leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
    ].copy()
    grouped_rows = current_leaderboard.loc[
        current_leaderboard["trial_id"].astype(str).eq(grouped_control_trial_id)
    ].copy()
    candidate_diagnostics = current_diagnostics.loc[
        current_diagnostics["trial_id"].astype(str).eq(candidate_trial_id)
    ].copy()
    grouped_diagnostics = current_diagnostics.loc[
        current_diagnostics["trial_id"].astype(str).eq(grouped_control_trial_id)
    ].copy()
    if set(current_leaderboard["trial_id"].astype(str)) != {
        candidate_trial_id,
        grouped_control_trial_id,
    } or set(current_diagnostics["trial_id"].astype(str)) != {
        candidate_trial_id,
        grouped_control_trial_id,
    }:
        raise ContractError("v30 current trial identities drifted")

    candidate_base = evaluate_frozen_shape_learning_rate_multiseed(
        candidate_rows,
        historical_leaderboard,
        candidate_diagnostics,
        comparison,
        control_trial_id=control_trial_id,
        parent_candidate_trial_id=parent_candidate_trial_id,
        v25_trial_id=v25_trial_id,
        v26_trial_id=v26_trial_id,
        v28_trial_id=v28_trial_id,
        candidate_trial_id=candidate_trial_id,
        expected_seeds=expected_seeds,
        min_mean_rankic=min_mean_rankic,
        min_positive_units=min_positive_units,
        min_parent_mean_rankic_delta=min_parent_mean_rankic_delta,
        min_control_mean_rankic_delta=min_control_mean_rankic_delta,
        min_v26_mean_rankic_delta=min_v26_mean_rankic_delta,
        min_v25_mean_rankic_delta=min_v25_mean_rankic_delta,
        min_v28_mean_rankic_delta=min_v28_mean_rankic_delta,
        min_nondegrading_folds_per_seed=min_nondegrading_folds_per_seed,
        min_horizon_parent_delta_1d=min_horizon_parent_delta_1d,
        min_horizon_parent_delta_2d=min_horizon_parent_delta_2d,
        min_horizon_parent_delta_3d=min_horizon_parent_delta_3d,
        min_horizon_parent_delta_5d=min_horizon_parent_delta_5d,
        max_parent_rankic_abs_error=max_parent_rankic_abs_error,
        max_parent_prediction_abs_error=max_parent_prediction_abs_error,
        min_trained_effect_units=min_trained_effect_units,
        min_shape_output_weight_l2=min_shape_output_weight_l2,
        min_shape_residual_weight_effect=min_shape_residual_weight_effect,
        max_simplex_error=max_simplex_error,
        min_median_samples_per_second=min_median_samples_per_second,
        candidate_parameter_count=candidate_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        frozen_parameter_count=frozen_parameter_count,
        shape_residual_scale=shape_residual_scale,
        learning_rate=learning_rate,
        checkpoint_min_delta=checkpoint_min_delta,
        patience_min_delta=patience_min_delta,
        min_model_step_speed_ratio=min_model_step_speed_ratio,
        min_end_to_end_speed_ratio=min_end_to_end_speed_ratio,
    )

    base_history = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).isin(
            {
                control_trial_id,
                parent_candidate_trial_id,
                v25_trial_id,
                v26_trial_id,
            }
        )
    ].copy()
    grouped_base = evaluate_frozen_parent_shape_residual_multiseed(
        grouped_rows,
        base_history,
        grouped_diagnostics,
        comparison,
        control_trial_id=control_trial_id,
        parent_candidate_trial_id=parent_candidate_trial_id,
        v25_trial_id=v25_trial_id,
        v26_trial_id=v26_trial_id,
        candidate_trial_id=grouped_control_trial_id,
        expected_seeds=expected_seeds,
        min_mean_rankic=min_mean_rankic,
        min_positive_units=min_positive_units,
        min_parent_mean_rankic_delta=min_parent_mean_rankic_delta,
        min_control_mean_rankic_delta=min_control_mean_rankic_delta,
        min_v26_mean_rankic_delta=min_v26_mean_rankic_delta,
        min_v25_mean_rankic_delta=min_v25_mean_rankic_delta,
        min_nondegrading_folds_per_seed=min_nondegrading_folds_per_seed,
        min_horizon_parent_delta_1d=min_horizon_parent_delta_1d,
        min_horizon_parent_delta_2d=min_horizon_parent_delta_2d,
        min_horizon_parent_delta_3d=min_horizon_parent_delta_3d,
        min_horizon_parent_delta_5d=min_horizon_parent_delta_5d,
        max_parent_rankic_abs_error=max_parent_rankic_abs_error,
        max_parent_prediction_abs_error=max_parent_prediction_abs_error,
        min_trained_effect_units=min_trained_effect_units,
        min_shape_output_weight_l2=min_shape_output_weight_l2,
        min_shape_residual_weight_effect=min_shape_residual_weight_effect,
        max_simplex_error=max_simplex_error,
        min_median_samples_per_second=0.0,
        candidate_parameter_count=candidate_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        frozen_parameter_count=frozen_parameter_count,
        shape_residual_scale=shape_residual_scale,
        learning_rate=learning_rate,
        min_model_step_speed_ratio=0.0,
        min_end_to_end_speed_ratio=0.0,
    )

    expected_selection_identity = (
        "best-any-strict-improvement+patience-material-"
        f"{patience_min_delta:g}"
    )
    candidate_identity_passed = bool(
        set(candidate_rows["strategy"].astype(str)) == {"soft_rankic"}
        and set(candidate_rows["loss_identity"].astype(str))
        == {
            f"smooth-l1+{soft_rankic_weight:g}-soft-rankic"
            f"-tau-{soft_rank_temperature:g}"
        }
        and set(candidate_rows["batching_identity"].astype(str))
        == {"date-grouped"}
        and set(candidate_rows["soft_rankic_weight"].astype(float))
        == {soft_rankic_weight}
        and set(candidate_rows["soft_rank_temperature"].astype(float))
        == {soft_rank_temperature}
    )
    grouped_identity_passed = bool(
        set(grouped_rows["strategy"].astype(str)) == {"grouped_smooth_l1"}
        and set(grouped_rows["loss_identity"].astype(str))
        == {"date-grouped-smooth-l1"}
        and set(grouped_rows["batching_identity"].astype(str))
        == {"date-grouped"}
        and bool(grouped_rows["soft_rankic_weight"].isna().all())
        and bool(grouped_rows["soft_rank_temperature"].isna().all())
        and set(grouped_rows["learning_rate"].astype(float)) == {learning_rate}
        and set(grouped_rows["checkpoint_min_delta"].astype(float))
        == {checkpoint_min_delta}
        and set(grouped_rows["patience_min_delta"].astype(float))
        == {patience_min_delta}
        and set(grouped_rows["checkpoint_selection_identity"].astype(str))
        == {expected_selection_identity}
    )

    key = ["seed", "fold"]
    candidate_values = candidate_rows.set_index(key)["best_mean_daily_rankic"]
    grouped_values = grouped_rows.set_index(key)["best_mean_daily_rankic"]
    if not candidate_values.index.equals(grouped_values.index):
        raise ContractError("v30 candidate/grouped control coverage drifted")
    grouped_deltas = candidate_values - grouped_values
    if not np.isfinite(grouped_deltas.to_numpy(dtype="float64")).all():
        raise ContractError("v30 paired grouped-control delta is non-finite")
    grouped_control_mean_delta = float(grouped_deltas.mean())
    grouped_seed = (
        pd.concat(
            {
                "grouped_control_mean_rankic": grouped_values,
                "grouped_control_mean_rankic_delta": grouped_deltas,
            },
            axis=1,
        )
        .reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            grouped_control_mean_rankic=("grouped_control_mean_rankic", "mean"),
            grouped_control_mean_rankic_delta=(
                "grouped_control_mean_rankic_delta",
                "mean",
            ),
        )
    )
    seed_summary = candidate_base.seed_summary.merge(
        grouped_seed, on="seed", how="left", validate="one_to_one"
    )

    candidate_hashes = candidate_rows.set_index(key)[
        "parent_checkpoint_sha256"
    ].astype(str)
    grouped_hashes = grouped_rows.set_index(key)[
        "parent_checkpoint_sha256"
    ].astype(str)
    parent_checkpoint_match = bool(candidate_hashes.equals(grouped_hashes))
    integrity_blockers: list[str] = []
    if not candidate_base.integrity_passed:
        integrity_blockers.append("candidate_frozen_integrity_failed")
    if not grouped_base.integrity_passed:
        integrity_blockers.append("grouped_control_frozen_integrity_failed")
    if not candidate_identity_passed:
        integrity_blockers.append("candidate_rank_objective_identity_drifted")
    if not grouped_identity_passed:
        integrity_blockers.append("grouped_control_identity_drifted")
    if not parent_checkpoint_match:
        integrity_blockers.append("candidate_control_parent_checkpoint_mismatch")
    integrity_passed = not integrity_blockers

    effect_blockers: list[str] = []
    if not candidate_base.effect_passed:
        effect_blockers.extend(
            blocker
            for blocker in str(candidate_base.aggregate.get("blockers", "")).split(",")
            if blocker
        )
    if grouped_control_mean_delta < min_grouped_control_mean_rankic_delta:
        effect_blockers.append("grouped_control_mean_rankic_delta_below_gate")
    if bool(seed_summary["grouped_control_mean_rankic_delta"].lt(0).any()):
        effect_blockers.append("per_seed_grouped_control_delta_negative")
    effect_passed = not effect_blockers
    speed_passed = candidate_base.speed_passed

    if not integrity_passed:
        status = "stop_shape_rank_integrity_v30"
    elif not effect_passed:
        status = "stop_shape_rank_no_gain_v30"
    elif not speed_passed:
        status = "stop_shape_rank_speed_v30"
    else:
        status = "shape_rank_objective_confirmed_v30"

    v28_rows = historical_leaderboard.loc[
        historical_leaderboard["trial_id"].astype(str).eq(v28_trial_id)
    ]
    grouped_v28_delta = float(
        grouped_values.mean() - v28_rows["best_mean_daily_rankic"].mean()
    )
    aggregate = dict(candidate_base.aggregate)
    aggregate.update(
        {
            "grouped_control_mean_rankic": float(grouped_values.mean()),
            "grouped_control_mean_rankic_delta": grouped_control_mean_delta,
            "grouped_control_v28_mean_rankic_delta": grouped_v28_delta,
            "soft_rankic_weight": soft_rankic_weight,
            "soft_rank_temperature": soft_rank_temperature,
            "candidate_grouped_parent_checkpoint_match": parent_checkpoint_match,
            "blockers": ",".join(
                [*integrity_blockers, *dict.fromkeys(effect_blockers)]
            ),
        }
    )
    return FrozenShapeSoftRankICMultiSeedDecision(
        status=status,
        integrity_passed=integrity_passed,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=candidate_base.horizon_summary,
    )
