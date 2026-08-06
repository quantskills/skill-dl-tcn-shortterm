"""Governed selection and fair TCN/LSTM comparison for real validation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from .experiment import ContractError
from .tuning import TCNTuningTrial


@dataclass(frozen=True)
class Seed7TCNDecision:
    """Outcome of the bounded seed-7 TCN-only Pareto screen."""

    status: str
    winner_trial_id: str | None
    summary: pd.DataFrame


@dataclass(frozen=True)
class FinalSeed7BenchmarkDecision:
    """Final v11 decision after effect and relative-speed gates."""

    status: str
    winner_trial_id: str | None
    relative_speed_gate_passed: bool
    confirmation_seeds_authorized: tuple[int, ...]


@dataclass(frozen=True)
class SignedMultiSeedConfirmationDecision:
    """Effect and speed decision for the pre-authorized signed TCN seeds."""

    status: str
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | bool | str]
    seed_summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class StabilizedResidualSeed7Decision:
    """Effect decision for the bounded v15 residual-parameterization screen."""

    status: str
    winner_trial_id: str | None
    summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class FinalStabilizedResidualSeed7Decision:
    """v15 effect decision combined with the fixed LSTM speed gates."""

    status: str
    winner_trial_id: str | None
    relative_speed_gate_passed: bool
    confirmation_seeds_authorized: tuple[int, ...]


@dataclass(frozen=True)
class DecoupledResidualSeed7Decision:
    """Capacity-aware effect decision for the v16 decoupled residual screen."""

    status: str
    winner_trial_id: str | None
    summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class FinalDecoupledResidualSeed7Decision:
    """v16 decoupled effect decision combined with fixed LSTM speed gates."""

    status: str
    winner_trial_id: str | None
    relative_speed_gate_passed: bool
    confirmation_seeds_authorized: tuple[int, ...]


@dataclass(frozen=True)
class PITMarketConditioningSeed7Decision:
    """Effect decision for the bounded v17 PIT market-conditioning screen."""

    status: str
    winner_trial_id: str | None
    summary: pd.DataFrame
    horizon_summary: pd.DataFrame


@dataclass(frozen=True)
class FinalPITMarketConditioningSeed7Decision:
    """v17 effect decision combined with the fixed LSTM speed gates."""

    status: str
    winner_trial_id: str | None
    relative_speed_gate_passed: bool
    confirmation_seeds_authorized: tuple[int, ...]


def parse_real_tcn_trials(raw_trials: object) -> tuple[TCNTuningTrial, ...]:
    """Parse every behavior-affecting TCN field from a public real-run config."""

    if not isinstance(raw_trials, list) or not raw_trials:
        raise ContractError("real validation trials must be a non-empty list")
    trials: list[TCNTuningTrial] = []
    for raw_value in raw_trials:
        if not isinstance(raw_value, dict):
            raise ContractError("each real validation trial must be an object")
        raw = cast(dict[str, object], raw_value)
        model_kind = str(raw.get("model_kind", "lite"))
        strategy = str(raw.get("strategy", "smooth_l1"))
        padding_mode = str(raw.get("padding_mode", "explicit"))
        dropout_kind = str(raw.get("dropout_kind", "element"))
        date_batch_order = str(raw.get("date_batch_order", "fixed_once"))
        grouped_smooth_l1_reduction = str(
            raw.get("grouped_smooth_l1_reduction", "label_mean")
        )
        if model_kind not in {
            "bai",
            "lite",
            "horizon_skip",
            "temporal_context",
            "signed_temporal_context",
            "stabilized_temporal_context",
            "decoupled_temporal_context",
            "market_conditioned_temporal_context",
            "dynamic_temporal_context",
            "dynamic_horizon_skip",
        }:
            raise ContractError("real validation TCN model kind is unsupported")
        if strategy not in {
            "smooth_l1",
            "grouped_smooth_l1",
            "rank_objective",
            "soft_rankic",
            "top_tail",
            "teacher_listwise",
            "pcgrad",
        }:
            raise ContractError("real validation TCN strategy is unsupported")
        if padding_mode not in {"explicit", "chomp"}:
            raise ContractError("real validation padding mode is unsupported")
        if dropout_kind not in {"element", "channel"}:
            raise ContractError("real validation dropout kind is unsupported")
        if date_batch_order not in {"fixed_once", "epoch_seeded"}:
            raise ContractError("real validation date batch order is unsupported")
        if grouped_smooth_l1_reduction not in {
            "label_mean",
            "date_horizon_mean",
        }:
            raise ContractError(
                "real validation grouped SmoothL1 reduction is unsupported"
            )
        if (
            model_kind
            in {
                "temporal_context",
                "signed_temporal_context",
                "stabilized_temporal_context",
                "decoupled_temporal_context",
                "market_conditioned_temporal_context",
                "dynamic_temporal_context",
            }
            and "bars_per_day" not in raw
        ):
            raise ContractError(
                "temporal context real validation requires bars_per_day"
            )
        if model_kind == "stabilized_temporal_context" and not {
            "residual_scale",
            "adapter_learning_rate",
        } <= set(raw):
            raise ContractError(
                "stabilized temporal context requires explicit residual scale and "
                "adapter learning rate"
            )
        if model_kind == "decoupled_temporal_context" and not {
            "residual_scale",
            "residual_learning_rate",
        } <= set(raw):
            raise ContractError(
                "decoupled temporal context requires explicit residual scale and "
                "residual learning rate"
            )
        if model_kind == "market_conditioned_temporal_context" and not {
            "market_context_dim",
            "market_context_hidden",
            "market_gate_scale",
        } <= set(raw):
            raise ContractError(
                "market-conditioned temporal context requires explicit context and gate parameters"
            )
        if model_kind == "dynamic_temporal_context" and not {
            "dynamic_attention_hidden",
            "dynamic_attention_scale",
        } <= set(raw):
            raise ContractError(
                "dynamic attention temporal context requires explicit hidden size and scale"
            )
        if model_kind == "dynamic_horizon_skip" and not {
            "dynamic_skip_hidden",
            "dynamic_skip_scale",
        } <= set(raw):
            raise ContractError(
                "dynamic horizon skip requires explicit hidden size and scale"
            )
        shape_residual_value = raw.get("dynamic_skip_shape_residual", False)
        if not isinstance(shape_residual_value, bool):
            raise ContractError("dynamic skip shape residual must be boolean")
        frozen_parent_value = raw.get("dynamic_skip_frozen_parent", False)
        if not isinstance(frozen_parent_value, bool):
            raise ContractError("dynamic skip frozen parent must be boolean")
        if strategy == "soft_rankic" and not {
            "soft_rankic_weight",
            "soft_rank_temperature",
        } <= set(raw):
            raise ContractError(
                "soft RankIC real validation requires explicit public parameters"
            )
        if strategy == "top_tail" and not {
            "top_tail_weight",
            "top_tail_fraction",
            "top_tail_temperature",
        } <= set(raw):
            raise ContractError(
                "top-tail real validation requires explicit public parameters"
            )
        if strategy == "teacher_listwise" and not {
            "teacher_listwise_gradient_ratio",
            "teacher_listwise_temperature",
        } <= set(raw):
            raise ContractError(
                "teacher listwise validation requires explicit public parameters"
            )
        trials.append(
            TCNTuningTrial(
                trial_id=str(raw["trial_id"]),
                channels=int(cast(Any, raw["channels"])),
                kernel_size=int(cast(Any, raw["kernel_size"])),
                dilations=tuple(
                    int(cast(Any, value))
                    for value in cast(list[object], raw["dilations"])
                ),
                dropout=float(cast(Any, raw["dropout"])),
                learning_rate=float(cast(Any, raw["learning_rate"])),
                batch_size=int(cast(Any, raw["batch_size"])),
                model_kind=cast(
                    Literal[
                        "bai",
                        "lite",
                        "horizon_skip",
                        "temporal_context",
                        "signed_temporal_context",
                        "stabilized_temporal_context",
                        "decoupled_temporal_context",
                        "market_conditioned_temporal_context",
                        "dynamic_temporal_context",
                        "dynamic_horizon_skip",
                    ],
                    model_kind,
                ),
                head_dropout=float(cast(Any, raw.get("head_dropout", 0.0))),
                dropout_kind=cast(Literal["element", "channel"], dropout_kind),
                weight_decay=float(cast(Any, raw.get("weight_decay", 0.0))),
                strategy=cast(
                    Literal[
                        "smooth_l1",
                        "grouped_smooth_l1",
                        "rank_objective",
                        "soft_rankic",
                        "top_tail",
                        "teacher_listwise",
                        "pcgrad",
                    ],
                    strategy,
                ),
                padding_mode=cast(Literal["explicit", "chomp"], padding_mode),
                pcgrad_blocks=(
                    tuple(
                        int(cast(Any, value))
                        for value in cast(list[object], raw["pcgrad_blocks"])
                    )
                    if "pcgrad_blocks" in raw
                    else None
                ),
                pcgrad_horizons=(
                    tuple(
                        int(cast(Any, value))
                        for value in cast(list[object], raw["pcgrad_horizons"])
                    )
                    if "pcgrad_horizons" in raw
                    else None
                ),
                bars_per_day=int(cast(Any, raw.get("bars_per_day", 48))),
                soft_rankic_weight=float(cast(Any, raw.get("soft_rankic_weight", 0.2))),
                soft_rank_temperature=float(
                    cast(Any, raw.get("soft_rank_temperature", 0.1))
                ),
                top_tail_weight=float(
                    cast(Any, raw.get("top_tail_weight", 0.05))
                ),
                top_tail_fraction=float(
                    cast(Any, raw.get("top_tail_fraction", 0.1))
                ),
                top_tail_temperature=float(
                    cast(Any, raw.get("top_tail_temperature", 0.1))
                ),
                teacher_listwise_gradient_ratio=float(
                    cast(Any, raw.get("teacher_listwise_gradient_ratio", 0.25))
                ),
                teacher_listwise_temperature=float(
                    cast(Any, raw.get("teacher_listwise_temperature", 0.1))
                ),
                residual_scale=float(cast(Any, raw.get("residual_scale", 0.05))),
                adapter_learning_rate=(
                    float(cast(Any, raw["adapter_learning_rate"]))
                    if "adapter_learning_rate" in raw
                    else None
                ),
                residual_learning_rate=(
                    float(cast(Any, raw["residual_learning_rate"]))
                    if "residual_learning_rate" in raw
                    else None
                ),
                market_context_dim=int(cast(Any, raw.get("market_context_dim", 24))),
                market_context_hidden=int(
                    cast(Any, raw.get("market_context_hidden", 4))
                ),
                market_gate_scale=float(cast(Any, raw.get("market_gate_scale", 0.25))),
                dynamic_attention_hidden=int(
                    cast(Any, raw.get("dynamic_attention_hidden", 4))
                ),
                dynamic_attention_scale=float(
                    cast(Any, raw.get("dynamic_attention_scale", 1.0))
                ),
                dynamic_attention_learning_rate=(
                    float(cast(Any, raw["dynamic_attention_learning_rate"]))
                    if "dynamic_attention_learning_rate" in raw
                    else None
                ),
                dynamic_skip_hidden=int(cast(Any, raw.get("dynamic_skip_hidden", 4))),
                dynamic_skip_scale=float(cast(Any, raw.get("dynamic_skip_scale", 1.0))),
                dynamic_skip_learning_rate=(
                    float(cast(Any, raw["dynamic_skip_learning_rate"]))
                    if "dynamic_skip_learning_rate" in raw
                    else None
                ),
                dynamic_skip_warmup_epochs=int(
                    cast(Any, raw.get("dynamic_skip_warmup_epochs", 0))
                ),
                dynamic_skip_token_normalization=cast(
                    Literal["none", "layer_norm", "shape_log_rms"],
                    str(raw.get("dynamic_skip_input_normalization", "none")),
                ),
                dynamic_skip_shape_residual=shape_residual_value,
                dynamic_skip_shape_residual_scale=float(
                    cast(Any, raw.get("dynamic_skip_shape_residual_scale", 0.25))
                ),
                dynamic_skip_frozen_parent=frozen_parent_value,
                date_batch_order=cast(
                    Literal["fixed_once", "epoch_seeded"], date_batch_order
                ),
                grouped_smooth_l1_reduction=cast(
                    Literal["label_mean", "date_horizon_mean"],
                    grouped_smooth_l1_reduction,
                ),
                ema_decay=(
                    float(cast(Any, raw["ema_decay"]))
                    if "ema_decay" in raw
                    else None
                ),
                epoch_average_start=(
                    int(cast(Any, raw["epoch_average_start"]))
                    if "epoch_average_start" in raw
                    else None
                ),
                teacher_blend_start_weight=(
                    float(cast(Any, raw["teacher_blend_start_weight"]))
                    if "teacher_blend_start_weight" in raw
                    else None
                ),
                teacher_blend_end_weight=float(
                    cast(Any, raw.get("teacher_blend_end_weight", 0.0))
                ),
            )
        )
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise ContractError("real validation trial IDs must be unique")
    return tuple(trials)


def select_seed7_tcn_candidate(
    leaderboard: pd.DataFrame,
    *,
    control_trial_id: str,
    min_mean_rankic: float,
    min_positive_folds: int,
    min_median_samples_per_second: float,
) -> Seed7TCNDecision:
    """Select one non-control TCN only when every pre-registered gate passes."""

    required = {
        "trial_id",
        "fold",
        "seed",
        "best_mean_daily_rankic",
        "samples_per_second",
        "parameter_count",
    }
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            f"real TCN leaderboard missing columns: {', '.join(missing)}"
        )
    if leaderboard.empty:
        raise ContractError("real TCN leaderboard cannot be empty")
    if set(leaderboard["seed"].astype(int)) != {7}:
        raise ContractError("seed-7 TCN selection accepts only base seed 7")
    if leaderboard.duplicated(["trial_id", "fold", "seed"]).any():
        raise ContractError("real TCN leaderboard contains duplicate units")
    numeric = leaderboard[
        [
            "best_mean_daily_rankic",
            "samples_per_second",
            "parameter_count",
        ]
    ].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all():
        raise ContractError("real TCN leaderboard contains non-finite evidence")
    if min_positive_folds <= 0:
        raise ContractError("minimum positive folds must be positive")

    fold_sets = {
        str(trial_id): tuple(sorted(rows["fold"].astype(int).tolist()))
        for trial_id, rows in leaderboard.groupby("trial_id", observed=True)
    }
    if len(set(fold_sets.values())) != 1:
        raise ContractError("real TCN trials must cover identical folds")
    observed_folds = next(iter(fold_sets.values()))
    if len(observed_folds) < min_positive_folds:
        raise ContractError("real TCN leaderboard has too few folds for the gate")
    if control_trial_id not in fold_sets:
        raise ContractError("real TCN leaderboard is missing its control")

    control_mean = float(
        leaderboard.loc[
            leaderboard["trial_id"].eq(control_trial_id),
            "best_mean_daily_rankic",
        ].mean()
    )
    summaries: list[dict[str, object]] = []
    for trial_id_value, rows in leaderboard.groupby("trial_id", observed=True):
        trial_id = str(trial_id_value)
        rankics = rows["best_mean_daily_rankic"].astype(float)
        mean_rankic = float(rankics.mean())
        worst_fold_rankic = float(rankics.min())
        positive_folds = int(rankics.gt(0).sum())
        throughput = float(rows["samples_per_second"].median())
        blockers: list[str] = []
        if trial_id == control_trial_id:
            blockers.append("control_reference_only")
        else:
            if mean_rankic < min_mean_rankic:
                blockers.append("mean_rankic_below_gate")
            if positive_folds < min_positive_folds:
                blockers.append("positive_folds_below_gate")
            if throughput < min_median_samples_per_second:
                blockers.append("throughput_below_gate")
            if mean_rankic < control_mean:
                blockers.append("control_mean_rankic_degradation")
        summaries.append(
            {
                "trial_id": trial_id,
                "mean_rankic": mean_rankic,
                "worst_fold_rankic": worst_fold_rankic,
                "positive_folds": positive_folds,
                "median_samples_per_second": throughput,
                "parameter_count": int(rows["parameter_count"].iloc[0]),
                "eligible": not blockers,
                "blockers": ",".join(blockers),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        [
            "eligible",
            "mean_rankic",
            "worst_fold_rankic",
            "median_samples_per_second",
            "parameter_count",
            "trial_id",
        ],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
        ignore_index=True,
    )
    eligible = summary.loc[summary["eligible"].astype(bool)]
    if eligible.empty:
        return Seed7TCNDecision("stop_no_seed7_pareto_v10", None, summary)
    return Seed7TCNDecision(
        "seed7_winner_admitted_v10",
        str(eligible.iloc[0]["trial_id"]),
        summary,
    )


def evaluate_stabilized_residual_seed7(
    leaderboard: pd.DataFrame,
    *,
    control_trial_id: str,
    candidate_trial_ids: tuple[str, ...],
    min_mean_rankic: float,
    min_positive_folds: int,
    min_nondegrading_folds: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    required_parameter_count: int,
) -> StabilizedResidualSeed7Decision:
    """Select one v15 arm only when every effect and stability gate passes."""

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
    }
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            "v15 stabilized leaderboard missing columns: " + ", ".join(missing)
        )
    if leaderboard.empty:
        raise ContractError("v15 stabilized leaderboard cannot be empty")
    if set(leaderboard["seed"].astype(int)) != {7}:
        raise ContractError("v15 stabilized screen accepts only seed 7")
    if not candidate_trial_ids or len(set(candidate_trial_ids)) != len(
        candidate_trial_ids
    ):
        raise ContractError("v15 stabilized candidate trial IDs must be unique")
    expected_trials = {control_trial_id, *candidate_trial_ids}
    if set(leaderboard["trial_id"].astype(str)) != expected_trials:
        raise ContractError("v15 stabilized trial identities drifted")
    if leaderboard.duplicated(["trial_id", "fold", "seed"]).any():
        raise ContractError("v15 stabilized leaderboard contains duplicate units")
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
        raise ContractError("v15 stabilized leaderboard contains non-finite evidence")
    if set(leaderboard["parameter_count"].astype(int)) != {required_parameter_count}:
        raise ContractError("v15 stabilized parameter count drifted")
    fold_sets = {
        str(trial_id): tuple(sorted(rows["fold"].astype(int).tolist()))
        for trial_id, rows in leaderboard.groupby("trial_id", observed=True)
    }
    if any(folds != tuple(range(5)) for folds in fold_sets.values()):
        raise ContractError("v15 stabilized trials must cover folds 0 through 4")
    if not 0 < min_positive_folds <= 5 or not 0 < min_nondegrading_folds <= 5:
        raise ContractError("v15 stabilized fold gates must be in [1, 5]")
    thresholds = np.asarray(
        [
            min_mean_rankic,
            min_horizon_delta_1d,
            min_horizon_delta_2d,
            min_horizon_delta_3d,
            min_horizon_delta_5d,
            min_median_samples_per_second,
        ],
        dtype="float64",
    )
    if not np.isfinite(thresholds).all() or min_median_samples_per_second <= 0:
        raise ContractError(
            "v15 stabilized gates must be finite and positive where required"
        )

    indexed = leaderboard.set_index(["trial_id", "fold"]).sort_index()
    control = indexed.loc[control_trial_id]
    horizon_columns = {
        1: "rankic_1d",
        2: "rankic_2d",
        3: "rankic_3d",
        5: "rankic_5d",
    }
    horizon_gates = {
        1: min_horizon_delta_1d,
        2: min_horizon_delta_2d,
        3: min_horizon_delta_3d,
        5: min_horizon_delta_5d,
    }
    summaries: list[dict[str, object]] = [
        {
            "trial_id": control_trial_id,
            "mean_rankic": float(control["best_mean_daily_rankic"].mean()),
            "worst_fold_rankic": float(control["best_mean_daily_rankic"].min()),
            "positive_folds": int(control["best_mean_daily_rankic"].gt(0).sum()),
            "mean_rankic_delta": 0.0,
            "nondegrading_folds": 5,
            "horizon_delta_1d": 0.0,
            "horizon_delta_2d": 0.0,
            "horizon_delta_3d": 0.0,
            "horizon_delta_5d": 0.0,
            "median_samples_per_second": float(control["samples_per_second"].median()),
            "parameter_count": required_parameter_count,
            "eligible": False,
            "blockers": "control_reference_only",
        }
    ]
    horizon_rows: list[dict[str, object]] = []
    for candidate_trial_id in candidate_trial_ids:
        candidate = indexed.loc[candidate_trial_id]
        rankics = cast(Any, candidate["best_mean_daily_rankic"].astype(float))
        rankic_delta = rankics - cast(
            Any, control["best_mean_daily_rankic"].astype(float)
        )
        horizon_deltas: dict[int, float] = {}
        for horizon, column in horizon_columns.items():
            control_rankic = float(control[column].mean())
            candidate_rankic = float(candidate[column].mean())
            delta = candidate_rankic - control_rankic
            horizon_deltas[horizon] = delta
            horizon_rows.append(
                {
                    "trial_id": candidate_trial_id,
                    "horizon": horizon,
                    "control_rankic": control_rankic,
                    "candidate_rankic": candidate_rankic,
                    "rankic_delta": delta,
                }
            )
        mean_rankic = float(rankics.mean())
        positive_folds = int(rankics.gt(0).sum())
        nondegrading_folds = int(rankic_delta.ge(0).sum())
        throughput = float(candidate["samples_per_second"].median())
        blockers: list[str] = []
        if mean_rankic < min_mean_rankic:
            blockers.append("mean_rankic_below_gate")
        if positive_folds < min_positive_folds:
            blockers.append("positive_folds_below_gate")
        if float(rankic_delta.mean()) <= 0:
            blockers.append("control_mean_rankic_degradation")
        if nondegrading_folds < min_nondegrading_folds:
            blockers.append("fold_stability_below_gate")
        for horizon, gate in horizon_gates.items():
            if horizon_deltas[horizon] < gate:
                blockers.append(f"horizon_{horizon}d_degradation_below_gate")
        if throughput < min_median_samples_per_second:
            blockers.append("throughput_below_gate")
        summaries.append(
            {
                "trial_id": candidate_trial_id,
                "mean_rankic": mean_rankic,
                "worst_fold_rankic": float(rankics.min()),
                "positive_folds": positive_folds,
                "mean_rankic_delta": float(rankic_delta.mean()),
                "nondegrading_folds": nondegrading_folds,
                "horizon_delta_1d": horizon_deltas[1],
                "horizon_delta_2d": horizon_deltas[2],
                "horizon_delta_3d": horizon_deltas[3],
                "horizon_delta_5d": horizon_deltas[5],
                "median_samples_per_second": throughput,
                "parameter_count": required_parameter_count,
                "eligible": not blockers,
                "blockers": ",".join(blockers),
            }
        )

    summary = pd.DataFrame(summaries).sort_values(
        [
            "eligible",
            "mean_rankic",
            "worst_fold_rankic",
            "horizon_delta_3d",
            "median_samples_per_second",
            "trial_id",
        ],
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
        ignore_index=True,
    )
    eligible = summary.loc[summary["eligible"].astype(bool)]
    winner = None if eligible.empty else str(eligible.iloc[0]["trial_id"])
    return StabilizedResidualSeed7Decision(
        status=(
            "stop_stabilized_residual_seed7_effect_v15"
            if winner is None
            else "stabilized_residual_seed7_effect_admitted_v15"
        ),
        winner_trial_id=winner,
        summary=summary,
        horizon_summary=pd.DataFrame(horizon_rows).sort_values(
            ["trial_id", "horizon"], ignore_index=True
        ),
    )


def finalize_stabilized_residual_seed7(
    effect_decision: StabilizedResidualSeed7Decision,
    comparison: dict[str, float | int],
    *,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FinalStabilizedResidualSeed7Decision:
    """Authorize v15 confirmation seeds only after effect and 3x speed gates."""

    if effect_decision.winner_trial_id is None:
        return FinalStabilizedResidualSeed7Decision(
            status="stop_stabilized_residual_seed7_effect_v15",
            winner_trial_id=None,
            relative_speed_gate_passed=False,
            confirmation_seeds_authorized=(),
        )
    try:
        model_step_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v15 comparison is missing relative speed") from exc
    if (
        not np.isfinite([model_step_ratio, end_to_end_ratio]).all()
        or min_model_step_speed_ratio <= 0
        or min_end_to_end_speed_ratio <= 0
    ):
        raise ContractError("v15 relative speed values and gates must be positive")
    speed_passed = bool(
        model_step_ratio >= min_model_step_speed_ratio
        and end_to_end_ratio >= min_end_to_end_speed_ratio
    )
    return FinalStabilizedResidualSeed7Decision(
        status=(
            "stabilized_residual_seed7_admitted_v15"
            if speed_passed
            else "stop_stabilized_residual_seed7_speed_v15"
        ),
        winner_trial_id=(effect_decision.winner_trial_id if speed_passed else None),
        relative_speed_gate_passed=speed_passed,
        confirmation_seeds_authorized=((17, 27) if speed_passed else ()),
    )


def evaluate_decoupled_residual_seed7(
    leaderboard: pd.DataFrame,
    *,
    control_trial_id: str,
    candidate_trial_ids: tuple[str, ...],
    min_mean_rankic: float,
    min_positive_folds: int,
    min_nondegrading_folds: int,
    min_horizon_delta_1d: float,
    min_horizon_delta_2d: float,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    control_parameter_count: int,
    candidate_parameter_count: int,
) -> DecoupledResidualSeed7Decision:
    """Apply the v15 effect gates while preserving v16's explicit capacity delta."""

    if "trial_id" not in leaderboard or "parameter_count" not in leaderboard:
        raise ContractError("v16 decoupled leaderboard is missing parameter evidence")
    if control_parameter_count <= 0 or candidate_parameter_count <= 0:
        raise ContractError("v16 parameter gates must be positive")
    if candidate_parameter_count <= control_parameter_count:
        raise ContractError("v16 candidate parameter count must exceed the control")
    control_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ]
    candidate_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).isin(candidate_trial_ids)
    ]
    if control_rows.empty or candidate_rows.empty:
        raise ContractError("v16 decoupled parameter evidence is incomplete")
    if set(control_rows["parameter_count"].astype(int)) != {control_parameter_count}:
        raise ContractError("v16 control parameter count drifted")
    if set(candidate_rows["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v16 candidate parameter count drifted")

    normalized = leaderboard.copy()
    normalized["parameter_count"] = candidate_parameter_count
    base = evaluate_stabilized_residual_seed7(
        normalized,
        control_trial_id=control_trial_id,
        candidate_trial_ids=candidate_trial_ids,
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
    winner = base.winner_trial_id
    return DecoupledResidualSeed7Decision(
        status=(
            "stop_decoupled_residual_seed7_effect_v16"
            if winner is None
            else "decoupled_residual_seed7_effect_admitted_v16"
        ),
        winner_trial_id=winner,
        summary=summary,
        horizon_summary=base.horizon_summary,
    )


def finalize_decoupled_residual_seed7(
    effect_decision: DecoupledResidualSeed7Decision,
    comparison: dict[str, float | int],
    *,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FinalDecoupledResidualSeed7Decision:
    """Authorize v16 confirmation only after effect and relative-speed gates."""

    if effect_decision.winner_trial_id is None:
        return FinalDecoupledResidualSeed7Decision(
            status="stop_decoupled_residual_seed7_effect_v16",
            winner_trial_id=None,
            relative_speed_gate_passed=False,
            confirmation_seeds_authorized=(),
        )
    try:
        model_step_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v16 comparison is missing relative speed") from exc
    if (
        not np.isfinite([model_step_ratio, end_to_end_ratio]).all()
        or min_model_step_speed_ratio <= 0
        or min_end_to_end_speed_ratio <= 0
    ):
        raise ContractError("v16 relative speed values and gates must be positive")
    speed_passed = bool(
        model_step_ratio >= min_model_step_speed_ratio
        and end_to_end_ratio >= min_end_to_end_speed_ratio
    )
    return FinalDecoupledResidualSeed7Decision(
        status=(
            "decoupled_residual_seed7_admitted_v16"
            if speed_passed
            else "stop_decoupled_residual_seed7_speed_v16"
        ),
        winner_trial_id=(effect_decision.winner_trial_id if speed_passed else None),
        relative_speed_gate_passed=speed_passed,
        confirmation_seeds_authorized=((17, 27) if speed_passed else ()),
    )


def evaluate_pit_market_conditioning_seed7(
    leaderboard: pd.DataFrame,
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
    min_market_gate_output_l2: float,
    control_parameter_count: int,
    candidate_parameter_count: int,
) -> PITMarketConditioningSeed7Decision:
    """Select the sole v17 candidate only after effect, use, and capacity gates."""

    required = {"trial_id", "parameter_count", "market_gate_output_l2"}
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            "v17 market-conditioning leaderboard missing columns: " + ", ".join(missing)
        )
    if control_parameter_count <= 0 or candidate_parameter_count <= 0:
        raise ContractError("v17 parameter gates must be positive")
    if candidate_parameter_count - control_parameter_count != 260:
        raise ContractError("v17 candidate capacity delta must be exactly 260")
    if not np.isfinite(min_mean_rankic_delta) or min_mean_rankic_delta <= 0:
        raise ContractError("v17 mean RankIC delta gate must be positive")
    if not np.isfinite(min_market_gate_output_l2) or min_market_gate_output_l2 <= 0:
        raise ContractError("v17 market gate use threshold must be positive")
    if set(leaderboard["trial_id"].astype(str)) != {
        control_trial_id,
        candidate_trial_id,
    }:
        raise ContractError("v17 market-conditioning trial identities drifted")
    control_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ]
    candidate_rows = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
    ]
    if set(control_rows["parameter_count"].astype(int)) != {control_parameter_count}:
        raise ContractError("v17 control parameter count drifted")
    if set(candidate_rows["parameter_count"].astype(int)) != {
        candidate_parameter_count
    }:
        raise ContractError("v17 candidate parameter count drifted")
    gate_l2 = candidate_rows["market_gate_output_l2"].to_numpy(dtype="float64")
    if not np.isfinite(gate_l2).all():
        raise ContractError("v17 market gate use evidence is non-finite")

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
    summary["market_gate_output_l2_min"] = np.where(
        is_control, np.nan, float(gate_l2.min())
    )
    candidate_mask = summary["trial_id"].astype(str).eq(candidate_trial_id)
    candidate_index = summary.index[candidate_mask]
    if len(candidate_index) != 1:
        raise ContractError("v17 candidate summary is incomplete")
    row_index = int(candidate_index[0])
    blockers = [
        value for value in str(summary.at[row_index, "blockers"]).split(",") if value
    ]
    if (
        float(cast(Any, summary.at[row_index, "mean_rankic_delta"]))
        < min_mean_rankic_delta
    ):
        blockers.append("mean_rankic_delta_below_gate")
    if float(gate_l2.min()) < min_market_gate_output_l2:
        blockers.append("market_gate_not_used")
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
    return PITMarketConditioningSeed7Decision(
        status=(
            "pit_market_conditioning_seed7_effect_admitted_v17"
            if winner is not None
            else "stop_pit_market_conditioning_seed7_effect_v17"
        ),
        winner_trial_id=winner,
        summary=summary,
        horizon_summary=base.horizon_summary,
    )


def finalize_pit_market_conditioning_seed7(
    effect_decision: PITMarketConditioningSeed7Decision,
    comparison: dict[str, float | int],
    *,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FinalPITMarketConditioningSeed7Decision:
    """Authorize v17 confirmation only after the effect and relative-speed gates."""

    if effect_decision.winner_trial_id is None:
        return FinalPITMarketConditioningSeed7Decision(
            status="stop_pit_market_conditioning_seed7_effect_v17",
            winner_trial_id=None,
            relative_speed_gate_passed=False,
            confirmation_seeds_authorized=(),
        )
    try:
        model_step_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("v17 comparison is missing relative speed") from exc
    if (
        not np.isfinite([model_step_ratio, end_to_end_ratio]).all()
        or min_model_step_speed_ratio <= 0
        or min_end_to_end_speed_ratio <= 0
    ):
        raise ContractError("v17 relative speed values and gates must be positive")
    speed_passed = bool(
        model_step_ratio >= min_model_step_speed_ratio
        and end_to_end_ratio >= min_end_to_end_speed_ratio
    )
    return FinalPITMarketConditioningSeed7Decision(
        status=(
            "pit_market_conditioning_seed7_admitted_v17"
            if speed_passed
            else "stop_pit_market_conditioning_seed7_speed_v17"
        ),
        winner_trial_id=(effect_decision.winner_trial_id if speed_passed else None),
        relative_speed_gate_passed=speed_passed,
        confirmation_seeds_authorized=((17, 27) if speed_passed else ()),
    )


def build_tcn_lstm_comparison(
    tcn_measurements: pd.DataFrame,
    lstm_measurements: pd.DataFrame,
) -> dict[str, float | int]:
    """Compare effect and speed only across identical fold/base-seed units."""

    tcn_required = {
        "fold",
        "seed",
        "best_mean_daily_rankic",
        "samples_per_second",
        "model_step_samples_per_second",
        "parameter_count",
    }
    lstm_required = {
        "model",
        "fold",
        "base_seed",
        "best_validation_rankic",
        "samples_per_second",
        "model_step_samples_per_second",
        "parameter_count",
    }
    if missing := sorted(tcn_required.difference(tcn_measurements.columns)):
        raise ContractError(
            f"TCN comparison rows missing columns: {', '.join(missing)}"
        )
    if missing := sorted(lstm_required.difference(lstm_measurements.columns)):
        raise ContractError(
            f"LSTM comparison rows missing columns: {', '.join(missing)}"
        )
    if tcn_measurements.empty or lstm_measurements.empty:
        raise ContractError("TCN/LSTM comparison requires non-empty measurements")
    if set(lstm_measurements["model"].astype(str)) != {"lstm"}:
        raise ContractError("TCN/LSTM comparison accepts only LSTM reference rows")
    if (
        tcn_measurements.duplicated(["fold", "seed"]).any()
        or lstm_measurements.duplicated(["fold", "base_seed"]).any()
    ):
        raise ContractError("TCN/LSTM comparison contains duplicate units")

    tcn = tcn_measurements.set_index(["fold", "seed"]).sort_index()
    lstm = lstm_measurements.set_index(["fold", "base_seed"]).sort_index()
    if not tcn.index.equals(lstm.index):
        raise ContractError("TCN/LSTM comparison units do not match")
    numeric = np.column_stack(
        [
            tcn["best_mean_daily_rankic"].to_numpy(dtype="float64"),
            lstm["best_validation_rankic"].to_numpy(dtype="float64"),
            tcn["samples_per_second"].to_numpy(dtype="float64"),
            lstm["samples_per_second"].to_numpy(dtype="float64"),
            tcn["model_step_samples_per_second"].to_numpy(dtype="float64"),
            lstm["model_step_samples_per_second"].to_numpy(dtype="float64"),
        ]
    )
    if not np.isfinite(numeric).all() or (numeric[:, 2:] <= 0).any():
        raise ContractError("TCN/LSTM comparison contains invalid numeric evidence")

    rankic_delta = tcn["best_mean_daily_rankic"].to_numpy(dtype="float64") - lstm[
        "best_validation_rankic"
    ].to_numpy(dtype="float64")
    model_step_ratios = tcn["model_step_samples_per_second"].to_numpy(
        dtype="float64"
    ) / lstm["model_step_samples_per_second"].to_numpy(dtype="float64")
    end_to_end_ratios = tcn["samples_per_second"].to_numpy(dtype="float64") / lstm[
        "samples_per_second"
    ].to_numpy(dtype="float64")
    return {
        "paired_unit_count": int(len(tcn)),
        "tcn_mean_rankic": float(tcn["best_mean_daily_rankic"].mean()),
        "lstm_mean_rankic": float(lstm["best_validation_rankic"].mean()),
        "paired_mean_rankic_difference": float(rankic_delta.mean()),
        "tcn_parameter_count": int(tcn["parameter_count"].iloc[0]),
        "lstm_parameter_count": int(lstm["parameter_count"].iloc[0]),
        "model_step_speed_ratio": float(np.exp(np.median(np.log(model_step_ratios)))),
        "end_to_end_speed_ratio": float(np.exp(np.median(np.log(end_to_end_ratios)))),
    }


def finalize_seed7_benchmark_gate(
    effect_decision: Seed7TCNDecision,
    comparison: dict[str, float | int],
    *,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> FinalSeed7BenchmarkDecision:
    """Require the TCN effect gate and both pre-registered relative-speed gates."""

    if min_model_step_speed_ratio <= 0 or min_end_to_end_speed_ratio <= 0:
        raise ContractError("relative speed gates must be positive")
    try:
        model_step_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("TCN/LSTM comparison is missing relative speed") from exc
    if not np.isfinite([model_step_ratio, end_to_end_ratio]).all():
        raise ContractError("TCN/LSTM relative speed must be finite")
    speed_passed = bool(
        model_step_ratio >= min_model_step_speed_ratio
        and end_to_end_ratio >= min_end_to_end_speed_ratio
    )
    if effect_decision.winner_trial_id is None:
        return FinalSeed7BenchmarkDecision(
            "stop_no_seed7_effect_pareto_v11",
            None,
            speed_passed,
            (),
        )
    if not speed_passed:
        return FinalSeed7BenchmarkDecision(
            "stop_no_seed7_speed_pareto_v11",
            None,
            False,
            (),
        )
    return FinalSeed7BenchmarkDecision(
        "seed7_winner_admitted_v11",
        effect_decision.winner_trial_id,
        True,
        (17, 27),
    )


def evaluate_signed_multiseed_confirmation(
    leaderboard: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...],
    min_mean_rankic: float,
    min_positive_units: int,
    min_nondegrading_folds_per_seed: int,
    min_horizon_delta_3d: float,
    min_horizon_delta_5d: float,
    min_median_samples_per_second: float,
    min_model_step_speed_ratio: float,
    min_end_to_end_speed_ratio: float,
) -> SignedMultiSeedConfirmationDecision:
    """Confirm one frozen signed candidate on exact pre-authorized seeds."""

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
    }
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            "signed multi-seed leaderboard missing columns: " + ", ".join(missing)
        )
    if leaderboard.empty:
        raise ContractError("signed multi-seed leaderboard cannot be empty")
    if not expected_seeds or tuple(sorted(set(expected_seeds))) != expected_seeds:
        raise ContractError("signed multi-seed expected seeds are invalid")
    if set(leaderboard["seed"].astype(int)) != set(expected_seeds):
        raise ContractError("signed multi-seed leaderboard seeds drifted")
    if set(leaderboard["trial_id"].astype(str)) != {
        control_trial_id,
        candidate_trial_id,
    }:
        raise ContractError("signed multi-seed trial identities drifted")
    if leaderboard.duplicated(["trial_id", "seed", "fold"]).any():
        raise ContractError("signed multi-seed leaderboard contains duplicate units")
    expected_units = {
        (trial, seed, fold)
        for trial in (control_trial_id, candidate_trial_id)
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
        raise ContractError("signed multi-seed fold coverage drifted")
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
        raise ContractError("signed multi-seed leaderboard contains non-finite values")
    parameter_counts = leaderboard.groupby("trial_id", observed=True)[
        "parameter_count"
    ].nunique()
    if bool(parameter_counts.ne(1).any()):
        raise ContractError("signed multi-seed parameter count drifted within a trial")
    first_parameters = leaderboard.groupby("trial_id", observed=True)[
        "parameter_count"
    ].first()
    if int(first_parameters[control_trial_id]) != int(
        first_parameters[candidate_trial_id]
    ):
        raise ContractError(
            "signed multi-seed candidate and control parameter mismatch"
        )
    if min_positive_units <= 0 or min_nondegrading_folds_per_seed <= 0:
        raise ContractError("signed multi-seed count gates must be positive")
    if min_nondegrading_folds_per_seed > 5:
        raise ContractError("signed multi-seed fold gate exceeds five folds")

    indexed = leaderboard.set_index(["seed", "fold", "trial_id"])[
        "best_mean_daily_rankic"
    ].unstack("trial_id")
    indexed["rankic_delta"] = indexed[candidate_trial_id] - indexed[control_trial_id]
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
    )
    horizon_rows: list[dict[str, float | int]] = []
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
    candidate_rows = leaderboard.loc[leaderboard["trial_id"].eq(candidate_trial_id)]
    candidate_mean = float(candidate_rows["best_mean_daily_rankic"].mean())
    positive_units = int(candidate_rows["best_mean_daily_rankic"].gt(0).sum())
    median_throughput = float(candidate_rows["samples_per_second"].median())
    mean_delta = float(indexed["rankic_delta"].mean())
    horizon_delta_3d = float(
        horizon_summary.loc[horizon_summary["horizon"].eq(3), "rankic_delta"].iloc[0]
    )
    horizon_delta_5d = float(
        horizon_summary.loc[horizon_summary["horizon"].eq(5), "rankic_delta"].iloc[0]
    )
    blockers: list[str] = []
    if candidate_mean < min_mean_rankic:
        blockers.append("mean_rankic_below_gate")
    if positive_units < min_positive_units:
        blockers.append("positive_units_below_gate")
    if mean_delta <= 0:
        blockers.append("control_mean_rankic_degradation")
    if bool(
        seed_summary["nondegrading_folds"].lt(min_nondegrading_folds_per_seed).any()
    ):
        blockers.append("per_seed_fold_stability_below_gate")
    if horizon_delta_3d < min_horizon_delta_3d:
        blockers.append("horizon_3d_degradation_below_gate")
    if horizon_delta_5d < min_horizon_delta_5d:
        blockers.append("horizon_5d_degradation_below_gate")
    if median_throughput < min_median_samples_per_second:
        blockers.append("throughput_below_gate")
    effect_passed = not blockers

    try:
        model_step_speed_ratio = float(comparison["model_step_speed_ratio"])
        end_to_end_speed_ratio = float(comparison["end_to_end_speed_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("signed multi-seed comparison is incomplete") from exc
    if not np.isfinite([model_step_speed_ratio, end_to_end_speed_ratio]).all():
        raise ContractError("signed multi-seed speed comparison is non-finite")
    speed_passed = bool(
        model_step_speed_ratio >= min_model_step_speed_ratio
        and end_to_end_speed_ratio >= min_end_to_end_speed_ratio
    )
    status = (
        "stop_signed_candidate_unstable_v14"
        if not effect_passed
        else (
            "signed_candidate_multiseed_confirmed_v14"
            if speed_passed
            else "stop_signed_candidate_speed_v14"
        )
    )
    aggregate: dict[str, float | int | bool | str] = {
        "candidate_mean_rankic": candidate_mean,
        "positive_units": positive_units,
        "mean_rankic_delta": mean_delta,
        "median_samples_per_second": median_throughput,
        "horizon_delta_3d": horizon_delta_3d,
        "horizon_delta_5d": horizon_delta_5d,
        "model_step_speed_ratio": model_step_speed_ratio,
        "end_to_end_speed_ratio": end_to_end_speed_ratio,
        "parameter_count": int(first_parameters[candidate_trial_id]),
        "blockers": ",".join(blockers),
    }
    return SignedMultiSeedConfirmationDecision(
        status=status,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        horizon_summary=horizon_summary,
    )
