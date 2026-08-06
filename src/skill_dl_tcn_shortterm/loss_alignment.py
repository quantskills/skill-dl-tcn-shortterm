"""Decision gates for equal-date/equal-horizon grouped SmoothL1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .batch_stability import bootstrap_paired_daily_rankic
from .experiment import ContractError


@dataclass(frozen=True)
class DateHorizonEqualDecision:
    """One fail-closed v32 objective-alignment decision."""

    status: str
    integrity_passed: bool
    mechanism_passed: bool
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | str | bool]
    seed_summary: pd.DataFrame
    bootstrap_summary: pd.DataFrame


def evaluate_date_horizon_equal_smooth_l1(
    leaderboard: pd.DataFrame,
    epoch_history: pd.DataFrame,
    paired_daily: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...] = (7, 17, 27),
    min_mean_rankic_delta: float = 0.00015,
    min_nondegrading_units: int = 12,
    max_unit_degradation: float = 0.00010,
    min_candidate_mean_rankic: float = 0.0999409483,
    min_throughput_ratio: float = 0.90,
    min_model_step_speed_ratio: float = 3.0,
    min_end_to_end_speed_ratio: float = 3.0,
    bootstrap_seed: int = 32,
    bootstrap_draws: int = 2_000,
) -> DateHorizonEqualDecision:
    """Apply v32 integrity, mechanism, effect and speed gates."""

    required = {
        "trial_id",
        "seed",
        "fold",
        "best_epoch",
        "best_mean_daily_rankic",
        "samples_per_second",
        "strategy",
        "loss_identity",
        "batching_identity",
        "date_batch_order",
        "grouped_smooth_l1_reduction",
        "date_order_fingerprint_count",
        "median_epoch_gradient_norm_cv",
        "median_labels_per_loss_group",
        "frozen_parent_state_drift_max",
        "parent_prediction_max_abs_error",
        "parent_checkpoint_sha256",
    }
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            "v32 leaderboard missing columns: " + ", ".join(missing)
        )
    if expected_seeds != (7, 17, 27):
        raise ContractError("v32 seeds must remain exactly 7, 17 and 27")
    control = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ].copy()
    candidate = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
    ].copy()
    expected_units = {(seed, fold) for seed in expected_seeds for fold in range(5)}
    integrity_blockers: list[str] = []
    for label, rows in (("control", control), ("candidate", candidate)):
        observed = {
            (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
            for row in rows.itertuples(index=False)
        }
        if observed != expected_units or rows.duplicated(["seed", "fold"]).any():
            integrity_blockers.append(f"{label}_coverage_drifted")
    if not bool(
        set(leaderboard["strategy"].astype(str)) == {"grouped_smooth_l1"}
        and set(leaderboard["batching_identity"].astype(str)) == {"date-grouped"}
        and set(leaderboard["date_batch_order"].astype(str)) == {"fixed_once"}
        and set(control["grouped_smooth_l1_reduction"].astype(str))
        == {"label_mean"}
        and set(candidate["grouped_smooth_l1_reduction"].astype(str))
        == {"date_horizon_mean"}
        and set(control["loss_identity"].astype(str))
        == {"date-grouped-smooth-l1"}
        and set(candidate["loss_identity"].astype(str))
        == {"date-horizon-equal-smooth-l1"}
    ):
        integrity_blockers.append("loss_reduction_identity_drifted")
    key = ["seed", "fold"]
    control_indexed = control.set_index(key).sort_index()
    candidate_indexed = candidate.set_index(key).sort_index()
    if not control_indexed.index.equals(candidate_indexed.index):
        raise ContractError("v32 paired unit coverage drifted")
    if not control_indexed["parent_checkpoint_sha256"].astype(str).equals(
        candidate_indexed["parent_checkpoint_sha256"].astype(str)
    ):
        integrity_blockers.append("parent_checkpoint_mismatch")
    if not bool(
        leaderboard["frozen_parent_state_drift_max"].astype(float).eq(0).all()
        and leaderboard["parent_prediction_max_abs_error"].astype(float).eq(0).all()
    ):
        integrity_blockers.append("frozen_parent_integrity_failed")
    numeric_diagnostics = leaderboard[
        ["median_epoch_gradient_norm_cv", "median_labels_per_loss_group"]
    ].to_numpy(dtype="float64")
    if not np.isfinite(numeric_diagnostics).all():
        integrity_blockers.append("loss_diagnostics_nonfinite")
    if not bool(leaderboard["date_order_fingerprint_count"].astype(int).eq(1).all()):
        integrity_blockers.append("fixed_date_order_drifted")
    integrity_passed = not integrity_blockers

    mechanism_blockers: list[str] = []
    validation_history = epoch_history.loc[
        epoch_history["stage"].astype(str).eq("validation")
    ]
    mechanism_columns = {
        "loss_group_count_mean",
        "valid_label_count_mean",
        "labels_per_loss_group_mean",
    }
    if missing := sorted(mechanism_columns.difference(validation_history.columns)):
        mechanism_blockers.append("loss_group_receipts_missing")
    else:
        candidate_history = validation_history.loc[
            validation_history["trial_id"].astype(str).eq(candidate_trial_id)
        ]
        values = candidate_history[list(mechanism_columns)].to_numpy(dtype="float64")
        if (
            values.size == 0
            or not np.isfinite(values).all()
            or bool((values <= 0).any())
        ):
            mechanism_blockers.append("loss_group_receipts_invalid")
    if not bool(candidate["median_labels_per_loss_group"].astype(float).gt(1).all()):
        mechanism_blockers.append("real_batches_not_group_weighted")
    mechanism_passed = not mechanism_blockers

    control_values = pd.Series(
        control_indexed["best_mean_daily_rankic"].to_numpy(dtype="float64"),
        index=control_indexed.index,
        dtype="float64",
    )
    candidate_values = pd.Series(
        candidate_indexed["best_mean_daily_rankic"].to_numpy(dtype="float64"),
        index=candidate_indexed.index,
        dtype="float64",
    )
    deltas = pd.Series(
        candidate_values.to_numpy(dtype="float64")
        - control_values.to_numpy(dtype="float64"),
        index=control_values.index,
        dtype="float64",
    )
    seed_summary = (
        pd.DataFrame(
            {
                "control_mean_rankic": control_values,
                "candidate_mean_rankic": candidate_values,
                "rankic_delta": deltas,
                "control_gradient_norm_cv": control_indexed[
                    "median_epoch_gradient_norm_cv"
                ].to_numpy(dtype="float64"),
                "candidate_gradient_norm_cv": candidate_indexed[
                    "median_epoch_gradient_norm_cv"
                ].to_numpy(dtype="float64"),
            },
            index=control_values.index,
        )
        .reset_index()
        .groupby("seed", as_index=False, observed=True)
        .agg(
            control_mean_rankic=("control_mean_rankic", "mean"),
            candidate_mean_rankic=("candidate_mean_rankic", "mean"),
            rankic_delta=("rankic_delta", "mean"),
            nondegrading_units=(
                "rankic_delta",
                lambda values: int((values >= -max_unit_degradation).sum()),
            ),
            control_gradient_norm_cv=("control_gradient_norm_cv", "median"),
            candidate_gradient_norm_cv=("candidate_gradient_norm_cv", "median"),
        )
    )
    bootstrap = bootstrap_paired_daily_rankic(
        paired_daily, seed=bootstrap_seed, draws=bootstrap_draws
    )
    bootstrap_by_scope = bootstrap.set_index("scope")
    mean_delta = float(deltas.mean())
    candidate_mean = float(candidate_values.mean())
    nondegrading_units = int((deltas >= -max_unit_degradation).sum())
    control_trained_units = int(control["best_epoch"].astype(int).gt(0).sum())
    candidate_trained_units = int(candidate["best_epoch"].astype(int).gt(0).sum())
    effect_blockers: list[str] = []
    if mean_delta < min_mean_rankic_delta:
        effect_blockers.append("mean_rankic_delta_below_gate")
    if bool(seed_summary["rankic_delta"].lt(0).any()):
        effect_blockers.append("per_seed_rankic_delta_negative")
    if nondegrading_units < min_nondegrading_units:
        effect_blockers.append("nondegrading_units_below_gate")
    if candidate_mean < min_candidate_mean_rankic:
        effect_blockers.append("candidate_mean_rankic_below_gate")
    if float(cast(Any, bootstrap_by_scope.loc["all", "bootstrap_ci_low"])) <= 0:
        effect_blockers.append("global_bootstrap_ci_crosses_zero")
    if float(
        cast(Any, bootstrap_by_scope.loc["seed_27", "bootstrap_ci_low"])
    ) < 0:
        effect_blockers.append("seed27_bootstrap_ci_crosses_zero")
    if candidate_trained_units < control_trained_units:
        effect_blockers.append("trained_effect_units_decreased")
    effect_passed = not effect_blockers

    model_step_ratio = float(comparison["model_step_speed_ratio"])
    end_to_end_ratio = float(comparison["end_to_end_speed_ratio"])
    throughput_ratio = float(
        candidate["samples_per_second"].median()
        / control["samples_per_second"].median()
    )
    speed_passed = bool(
        model_step_ratio >= min_model_step_speed_ratio
        and end_to_end_ratio >= min_end_to_end_speed_ratio
        and throughput_ratio >= min_throughput_ratio
    )
    if not integrity_passed:
        status = "stop_date_horizon_equal_integrity_v32"
    elif not mechanism_passed:
        status = "stop_date_horizon_equal_mechanism_v32"
    elif not effect_passed:
        status = "stop_date_horizon_equal_no_gain_v32"
    elif not speed_passed:
        status = "stop_date_horizon_equal_speed_v32"
    else:
        status = "date_horizon_equal_smooth_l1_confirmed_v32"
    blockers = [*integrity_blockers, *mechanism_blockers, *effect_blockers]
    aggregate: dict[str, float | int | str | bool] = {
        "candidate_mean_rankic": candidate_mean,
        "control_mean_rankic": float(control_values.mean()),
        "mean_rankic_delta": mean_delta,
        "nondegrading_units": nondegrading_units,
        "control_trained_effect_units": control_trained_units,
        "candidate_trained_effect_units": candidate_trained_units,
        "candidate_control_throughput_ratio": throughput_ratio,
        "candidate_median_gradient_norm_cv": float(
            candidate["median_epoch_gradient_norm_cv"].median()
        ),
        "control_median_gradient_norm_cv": float(
            control["median_epoch_gradient_norm_cv"].median()
        ),
        "candidate_median_labels_per_loss_group": float(
            candidate["median_labels_per_loss_group"].median()
        ),
        "model_step_speed_ratio": model_step_ratio,
        "end_to_end_speed_ratio": end_to_end_ratio,
        "global_bootstrap_ci_low": float(
            cast(Any, bootstrap_by_scope.loc["all", "bootstrap_ci_low"])
        ),
        "global_bootstrap_ci_high": float(
            cast(Any, bootstrap_by_scope.loc["all", "bootstrap_ci_high"])
        ),
        "seed27_bootstrap_ci_low": float(
            cast(Any, bootstrap_by_scope.loc["seed_27", "bootstrap_ci_low"])
        ),
        "seed27_bootstrap_ci_high": float(
            cast(Any, bootstrap_by_scope.loc["seed_27", "bootstrap_ci_high"])
        ),
        "blockers": ",".join(dict.fromkeys(blockers)),
    }
    return DateHorizonEqualDecision(
        status=status,
        integrity_passed=integrity_passed,
        mechanism_passed=mechanism_passed,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        bootstrap_summary=bootstrap,
    )
