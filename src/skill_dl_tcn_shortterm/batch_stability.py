"""Evidence and decision gates for deterministic date-batch order stability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
import torch

from .experiment import ContractError
from .training_data import LazyWindowDataset, build_fold_protocols
from .tuning import (
    TCNTuningTrial,
    _predict_tcn_trial,
    build_tcn_trial_model,
    build_validation_rankic_plan,
)
from .v9_statistics import _paired_block_bootstrap


@dataclass(frozen=True)
class GroupedBatchOrderDecision:
    """One fail-closed v31 stability decision."""

    status: str
    integrity_passed: bool
    mechanism_passed: bool
    effect_passed: bool
    speed_passed: bool
    aggregate: dict[str, float | int | str | bool]
    seed_summary: pd.DataFrame
    bootstrap_summary: pd.DataFrame


def collect_frozen_shape_daily_rankic(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    trials: Sequence[TCNTuningTrial],
    best_states: dict[str, dict[str, torch.Tensor]],
    *,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    """Re-evaluate best checkpoints into receipt-ready daily RankIC rows."""

    protocols = build_fold_protocols(features, split_manifest)
    dummy_targets = np.zeros((len(features), 4), dtype="float32")
    dummy_masks = np.ones((len(features), 4), dtype="bool")
    rows: list[dict[str, object]] = []
    for trial in trials:
        for seed in seeds:
            for protocol in protocols:
                model = build_tcn_trial_model(
                    trial,
                    feature_count=int(features.shape[1]),
                    input_steps=int(features.shape[2]),
                )
                checkpoint_key = f"seed-{seed}-{trial.trial_id}-fold-{protocol.fold}"
                try:
                    model.load_state_dict(best_states[checkpoint_key])
                except KeyError as exc:
                    raise ContractError(
                        "v31 daily RankIC checkpoint is missing"
                    ) from exc
                dataset = LazyWindowDataset(
                    features,
                    protocol.validation_positions,
                    dummy_targets,
                    dummy_masks,
                    protocol.feature_mean,
                    protocol.feature_std,
                )
                scores, positions = _predict_tcn_trial(
                    model, dataset, batch_size=trial.batch_size
                )
                plan = build_validation_rankic_plan(
                    protocol.validation_positions, window_index, labels
                )
                daily = plan.evaluate_daily(scores, positions)
                for value in daily.itertuples(index=False):
                    rows.append(
                        {
                            "trial_id": trial.trial_id,
                            "seed": seed,
                            "fold": protocol.fold,
                            "horizon": int(cast(Any, value.horizon)),
                            "signal_date": str(value.signal_date),
                            "rankic": float(cast(Any, value.rankic)),
                            "valid_member_count": int(
                                cast(Any, value.valid_member_count)
                            ),
                            "stage": "validation",
                            "sealed": False,
                        }
                    )
    daily_rankic = pd.DataFrame(rows)
    key = ["trial_id", "seed", "fold", "horizon", "signal_date"]
    if daily_rankic.empty or daily_rankic.duplicated(key).any():
        raise ContractError("v31 daily RankIC evidence is empty or duplicated")
    if daily_rankic["sealed"].astype(bool).any():
        raise ContractError("v31 daily RankIC evidence accessed sealed data")
    return daily_rankic.sort_values(key, kind="mergesort", ignore_index=True)


def pair_daily_rankic(
    daily_rankic: pd.DataFrame,
    *,
    control_trial_id: str,
    candidate_trial_id: str,
) -> pd.DataFrame:
    """Bind candidate/control RankIC by the exact ordinary-validation date."""

    required = {
        "trial_id",
        "seed",
        "fold",
        "horizon",
        "signal_date",
        "rankic",
        "valid_member_count",
        "stage",
        "sealed",
    }
    if missing := sorted(required.difference(daily_rankic.columns)):
        raise ContractError(
            "v31 daily RankIC evidence missing columns: " + ", ".join(missing)
        )
    if set(daily_rankic["trial_id"].astype(str)) != {
        control_trial_id,
        candidate_trial_id,
    }:
        raise ContractError("v31 daily RankIC trial identities drifted")
    if daily_rankic["sealed"].astype(bool).any() or set(
        daily_rankic["stage"].astype(str)
    ) != {"validation"}:
        raise ContractError("v31 paired RankIC accepts ordinary validation only")
    key = ["seed", "fold", "horizon", "signal_date"]

    def _side(trial_id: str, prefix: str) -> pd.DataFrame:
        return daily_rankic.loc[
            daily_rankic["trial_id"].astype(str).eq(trial_id),
            key + ["rankic", "valid_member_count"],
        ].rename(
            columns={
                "rankic": f"{prefix}_rankic",
                "valid_member_count": f"{prefix}_valid_member_count",
            }
        )

    paired = _side(control_trial_id, "control").merge(
        _side(candidate_trial_id, "candidate"),
        on=key,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if set(paired["_merge"].astype(str)) != {"both"}:
        raise ContractError("v31 candidate/control daily RankIC coverage drifted")
    paired = paired.drop(columns="_merge")
    if not np.array_equal(
        paired["control_valid_member_count"].to_numpy(dtype="int64"),
        paired["candidate_valid_member_count"].to_numpy(dtype="int64"),
    ):
        raise ContractError("v31 paired daily member counts drifted")
    values = paired[["control_rankic", "candidate_rankic"]].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(values).all():
        raise ContractError("v31 paired daily RankIC contains non-finite values")
    paired["rankic_delta"] = (
        paired["candidate_rankic"] - paired["control_rankic"]
    )
    paired["stage"] = "validation"
    paired["sealed"] = False
    return paired.sort_values(key, kind="mergesort", ignore_index=True)


def bootstrap_paired_daily_rankic(
    paired_daily: pd.DataFrame,
    *,
    seed: int = 31,
    draws: int = 2_000,
) -> pd.DataFrame:
    """Bootstrap ordered dates within every seed/fold/horizon unit."""

    required = {"seed", "fold", "horizon", "signal_date", "rankic_delta"}
    if missing := sorted(required.difference(paired_daily.columns)):
        raise ContractError(
            "v31 paired bootstrap missing columns: " + ", ".join(missing)
        )
    if draws <= 0 or paired_daily.empty:
        raise ContractError("v31 paired bootstrap requires evidence and draws")
    rng = np.random.default_rng(seed)
    unit_draws: dict[tuple[int, int, int], np.ndarray] = {}
    for key_values, group in paired_daily.groupby(
        ["seed", "fold", "horizon"], observed=True
    ):
        seed_value, fold_value, horizon_value = key_values
        ordered = group.sort_values("signal_date", kind="mergesort")
        deltas = ordered["rankic_delta"].to_numpy(dtype="float64")
        if len(deltas) < 2 or not np.isfinite(deltas).all():
            raise ContractError("v31 paired bootstrap unit is not resolvable")
        sampled, _ = _paired_block_bootstrap(deltas, rng, draws=draws)
        unit_draws[
            (
                int(cast(Any, seed_value)),
                int(cast(Any, fold_value)),
                int(cast(Any, horizon_value)),
            )
        ] = sampled

    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, int | None]] = [("all", None)] + [
        (f"seed_{value}", int(value))
        for value in sorted(paired_daily["seed"].astype(int).unique())
    ]
    for scope, scope_seed in scopes:
        keys = [
            key for key in unit_draws if scope_seed is None or key[0] == scope_seed
        ]
        if not keys:
            raise ContractError("v31 paired bootstrap scope is empty")
        scope_draws = np.stack([unit_draws[key] for key in keys]).mean(axis=0)
        scope_rows = paired_daily.loc[
            paired_daily["seed"].astype(int).eq(scope_seed)
            if scope_seed is not None
            else np.ones(len(paired_daily), dtype=bool)
        ]
        low, high = np.quantile(scope_draws, [0.025, 0.975]).tolist()
        rows.append(
            {
                "scope": scope,
                "seed": scope_seed,
                "paired_date_horizon_count": len(scope_rows),
                "unit_count": len(keys),
                "paired_mean_delta": float(scope_rows["rankic_delta"].mean()),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(rows)


def evaluate_grouped_batch_order_stability(
    leaderboard: pd.DataFrame,
    epoch_history: pd.DataFrame,
    paired_daily: pd.DataFrame,
    comparison: dict[str, float | int],
    *,
    control_trial_id: str,
    candidate_trial_id: str,
    expected_seeds: tuple[int, ...] = (7, 17, 27),
    min_mean_rankic_delta: float = 0.00015,
    min_seed27_rankic_delta: float = 0.00015,
    min_nondegrading_units: int = 12,
    max_unit_degradation: float = 0.00010,
    min_candidate_mean_rankic: float = 0.099791,
    min_throughput_ratio: float = 0.90,
    min_model_step_speed_ratio: float = 3.0,
    min_end_to_end_speed_ratio: float = 3.0,
    bootstrap_seed: int = 31,
    bootstrap_draws: int = 2_000,
) -> GroupedBatchOrderDecision:
    """Apply v31 integrity, mechanism, effect and speed gates."""

    required = {
        "trial_id",
        "seed",
        "fold",
        "best_mean_daily_rankic",
        "samples_per_second",
        "strategy",
        "loss_identity",
        "batching_identity",
        "date_batch_order",
        "date_order_fingerprint_count",
        "median_epoch_gradient_norm_cv",
        "completed_epochs",
        "frozen_parent_state_drift_max",
        "parent_prediction_max_abs_error",
        "parent_checkpoint_sha256",
    }
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(
            "v31 leaderboard missing columns: " + ", ".join(missing)
        )
    if expected_seeds != (7, 17, 27):
        raise ContractError("v31 seeds must remain exactly 7, 17 and 27")
    control = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(control_trial_id)
    ].copy()
    candidate = leaderboard.loc[
        leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
    ].copy()
    expected_units = {(seed, fold) for seed in expected_seeds for fold in range(5)}
    blockers: list[str] = []
    for label, rows in (("control", control), ("candidate", candidate)):
        observed = {
            (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
            for row in rows.itertuples(index=False)
        }
        if observed != expected_units or rows.duplicated(["seed", "fold"]).any():
            blockers.append(f"{label}_coverage_drifted")
    identity_ok = bool(
        set(leaderboard["strategy"].astype(str)) == {"grouped_smooth_l1"}
        and set(leaderboard["loss_identity"].astype(str))
        == {"date-grouped-smooth-l1"}
        and set(leaderboard["batching_identity"].astype(str)) == {"date-grouped"}
        and set(control["date_batch_order"].astype(str)) == {"fixed_once"}
        and set(candidate["date_batch_order"].astype(str)) == {"epoch_seeded"}
    )
    if not identity_ok:
        blockers.append("batch_order_identity_drifted")
    key = ["seed", "fold"]
    control_indexed = control.set_index(key).sort_index()
    candidate_indexed = candidate.set_index(key).sort_index()
    if not control_indexed.index.equals(candidate_indexed.index):
        raise ContractError("v31 paired unit coverage drifted")
    if not control_indexed["parent_checkpoint_sha256"].astype(str).equals(
        candidate_indexed["parent_checkpoint_sha256"].astype(str)
    ):
        blockers.append("parent_checkpoint_mismatch")
    if not bool(
        leaderboard["frozen_parent_state_drift_max"].astype(float).eq(0).all()
        and leaderboard["parent_prediction_max_abs_error"].astype(float).eq(0).all()
    ):
        blockers.append("frozen_parent_integrity_failed")
    gradient_values = leaderboard["median_epoch_gradient_norm_cv"].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(gradient_values).all():
        blockers.append("gradient_diagnostics_nonfinite")
    integrity_passed = not blockers

    mechanism_blockers: list[str] = []
    if not bool(control["date_order_fingerprint_count"].astype(int).eq(1).all()):
        mechanism_blockers.append("control_order_not_fixed")
    candidate_counts = candidate["date_order_fingerprint_count"].astype(int)
    candidate_epochs = candidate["completed_epochs"].astype(int)
    if not bool(candidate_counts.eq(candidate_epochs).all()):
        mechanism_blockers.append("candidate_epoch_orders_not_unique")
    history = epoch_history.loc[epoch_history["stage"].astype(str).eq("validation")]
    if "date_order_fingerprint" not in history or history[
        "date_order_fingerprint"
    ].isna().any():
        mechanism_blockers.append("epoch_order_receipts_missing")
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
                ].astype(float),
                "candidate_gradient_norm_cv": candidate_indexed[
                    "median_epoch_gradient_norm_cv"
                ].astype(float),
            }
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
    effect_blockers: list[str] = []
    mean_delta = float(deltas.mean())
    candidate_mean = float(candidate_values.mean())
    nondegrading_units = int((deltas >= -max_unit_degradation).sum())
    if mean_delta < min_mean_rankic_delta:
        effect_blockers.append("mean_rankic_delta_below_gate")
    if bool(seed_summary["rankic_delta"].lt(0).any()):
        effect_blockers.append("per_seed_rankic_delta_negative")
    seed27_delta = float(
        cast(Any, seed_summary.set_index("seed").loc[27, "rankic_delta"])
    )
    if seed27_delta < min_seed27_rankic_delta:
        effect_blockers.append("seed27_rankic_delta_below_gate")
    if nondegrading_units < min_nondegrading_units:
        effect_blockers.append("nondegrading_units_below_gate")
    if candidate_mean < min_candidate_mean_rankic:
        effect_blockers.append("candidate_mean_rankic_below_v30")
    if float(cast(Any, bootstrap_by_scope.loc["all", "bootstrap_ci_low"])) <= 0:
        effect_blockers.append("global_bootstrap_ci_crosses_zero")
    if float(
        cast(Any, bootstrap_by_scope.loc["seed_27", "bootstrap_ci_low"])
    ) < 0:
        effect_blockers.append("seed27_bootstrap_ci_crosses_zero")
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
        status = "stop_epoch_seeded_integrity_v31"
    elif not mechanism_passed:
        status = "stop_epoch_seeded_mechanism_not_confirmed_v31"
    elif not effect_passed:
        status = "stop_epoch_seeded_no_gain_v31"
    elif not speed_passed:
        status = "stop_epoch_seeded_speed_v31"
    else:
        status = "epoch_seeded_grouped_batch_confirmed_v31"
    all_blockers = [*blockers, *mechanism_blockers, *effect_blockers]
    aggregate: dict[str, float | int | str | bool] = {
        "candidate_mean_rankic": candidate_mean,
        "control_mean_rankic": float(control_values.mean()),
        "mean_rankic_delta": mean_delta,
        "seed27_rankic_delta": seed27_delta,
        "nondegrading_units": nondegrading_units,
        "candidate_control_throughput_ratio": throughput_ratio,
        "candidate_median_gradient_norm_cv": float(
            candidate["median_epoch_gradient_norm_cv"].median()
        ),
        "control_median_gradient_norm_cv": float(
            control["median_epoch_gradient_norm_cv"].median()
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
        "blockers": ",".join(dict.fromkeys(all_blockers)),
    }
    return GroupedBatchOrderDecision(
        status=status,
        integrity_passed=integrity_passed,
        mechanism_passed=mechanism_passed,
        effect_passed=effect_passed,
        speed_passed=speed_passed,
        aggregate=aggregate,
        seed_summary=seed_summary,
        bootstrap_summary=bootstrap,
    )
