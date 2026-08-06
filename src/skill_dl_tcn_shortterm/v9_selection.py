"""Evidence-built seed-7 trial registry and bounded Pareto selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class Seed7Trial:
    kind: str
    trial_id: str
    seed: int = 7
    fold_ids: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_epochs: int = 8
    patience: int = 2
    min_delta: float = 0.002
    infra_enabled: bool = False


@dataclass(frozen=True)
class Seed7Decision:
    status: str
    winner_trial_id: str | None
    confirmation_seeds: tuple[int, ...]
    blockers: tuple[str, ...]
    summary: pd.DataFrame


def build_seed7_trials(
    upstream_receipts: Mapping[str, Mapping[str, object]],
) -> tuple[Seed7Trial, ...]:
    """Construct, never replace, the three frozen conditional candidate slots."""

    required = {"horizon_skip", "rank_objective", "pcgrad", "infra"}
    if set(upstream_receipts) != required:
        raise ContractError("seed-7 planning requires exactly four upstream receipts")
    for receipt in upstream_receipts.values():
        if receipt.get("sealed_test_accessed") is not False:
            raise ContractError("seed-7 planning rejects sealed or unaudited receipts")
        if not isinstance(receipt.get("status"), str):
            raise ContractError("seed-7 upstream receipt is missing a status")
    status_sets = {
        "horizon_skip": {"horizon_skip_applicable", "horizon_skip_not_applicable"},
        "rank_objective": {
            "rank_objective_allowed",
            "rank_objective_not_resolvable",
            "rank_objective_not_applicable",
        },
        "pcgrad": {"pcgrad_applicable", "pcgrad_not_applicable"},
        "infra": {
            "causal_infra_acceleration_accepted",
            "infra_optimization_not_applicable",
        },
    }
    for name, allowed in status_sets.items():
        if str(upstream_receipts[name]["status"]) not in allowed:
            raise ContractError(f"seed-7 upstream {name} status is unsupported")
    infra_enabled = (
        upstream_receipts["infra"]["status"]
        == "causal_infra_acceleration_accepted"
    )
    definitions = [
        (
            "horizon_skip",
            "v9b-horizon-skip",
            upstream_receipts["horizon_skip"]["status"]
            == "horizon_skip_applicable",
        ),
        (
            "rank_objective",
            "v9c-rank-objective",
            upstream_receipts["rank_objective"]["status"]
            == "rank_objective_allowed",
        ),
        (
            "pcgrad",
            "v9d-pcgrad",
            upstream_receipts["pcgrad"]["status"] == "pcgrad_applicable",
        ),
    ]
    return tuple(
        Seed7Trial(kind=kind, trial_id=trial_id, infra_enabled=infra_enabled)
        for kind, trial_id, applicable in definitions
        if applicable
    )


def _empty_stop(blocker: str) -> Seed7Decision:
    return Seed7Decision(
        status="stop_no_pareto_gain_v9",
        winner_trial_id=None,
        confirmation_seeds=(),
        blockers=(blocker,),
        summary=pd.DataFrame(),
    )


def select_seed7_candidate(
    leaderboard: pd.DataFrame,
    *,
    registered_trials: tuple[Seed7Trial, ...],
    control_trial_id: str,
) -> Seed7Decision:
    """Apply the fixed five-fold unified effect/throughput admission gate."""

    if not registered_trials:
        if leaderboard.empty:
            return _empty_stop("no_triggered_trials")
    required = {
        "trial_id",
        "fold",
        "seed",
        "best_mean_daily_rankic",
        "samples_per_second",
        "parameter_count",
        "model_step_seconds",
        "data_wait_seconds",
        "validation_seconds",
        "complete_cycle_seconds",
        "time_to_best_seconds",
        "precision",
        "torch_threads",
        "batch_size",
        "data_identity",
        "fold_identity",
        "evaluation_identity",
        "sealed_test_accessed",
    }
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(f"seed-7 leaderboard missing columns: {', '.join(missing)}")
    if leaderboard.empty:
        raise ContractError("registered seed-7 trials require a leaderboard")
    if leaderboard["sealed_test_accessed"].astype(bool).any():
        raise ContractError("seed-7 selection rejects sealed evidence")
    if set(leaderboard["seed"].astype(int)) != {7}:
        raise ContractError("seed-7 selection accepts only seed 7")
    registered_ids = {trial.trial_id for trial in registered_trials}
    observed_ids = set(leaderboard["trial_id"].astype(str))
    allowed_ids = registered_ids | {control_trial_id}
    if observed_ids - allowed_ids:
        raise ContractError("seed-7 leaderboard contains an unregistered trial")
    if observed_ids != allowed_ids:
        raise ContractError("seed-7 leaderboard is missing a registered trial or control")
    if leaderboard.duplicated(["trial_id", "fold", "seed"]).any():
        raise ContractError("seed-7 leaderboard contains duplicate fold units")
    for column in [
        "precision",
        "torch_threads",
        "batch_size",
        "data_identity",
        "fold_identity",
        "evaluation_identity",
    ]:
        if leaderboard[column].nunique(dropna=False) != 1:
            raise ContractError(f"seed-7 protocol drift detected in {column}")
    numeric = [
        "best_mean_daily_rankic",
        "samples_per_second",
        "parameter_count",
        "model_step_seconds",
        "data_wait_seconds",
        "validation_seconds",
        "complete_cycle_seconds",
        "time_to_best_seconds",
    ]
    if not np.isfinite(leaderboard[numeric].to_numpy(dtype="float64")).all():
        raise ContractError("seed-7 leaderboard contains non-finite evidence")
    if leaderboard[["samples_per_second", "parameter_count", "model_step_seconds", "complete_cycle_seconds"]].le(0).any().any():
        raise ContractError("seed-7 throughput, parameters, and elapsed times must be positive")
    summary_rows = []
    for trial_value, rows in leaderboard.groupby("trial_id", observed=True):
        trial_id = str(trial_value)
        if set(rows["fold"].astype(int)) != {0, 1, 2, 3, 4} or len(rows) != 5:
            raise ContractError("every seed-7 trial must cover exactly five folds")
        if rows["parameter_count"].nunique() != 1:
            raise ContractError("seed-7 parameter count must be stable across folds")
        summary_rows.append(
            {
                "trial_id": trial_id,
                "mean_rankic": float(rows["best_mean_daily_rankic"].mean()),
                "worst_fold_rankic": float(rows["best_mean_daily_rankic"].min()),
                "positive_fold_count": int(rows["best_mean_daily_rankic"].gt(0).sum()),
                "samples_per_second": float(rows["samples_per_second"].median()),
                "parameter_count": int(cast(int, rows["parameter_count"].iloc[0])),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not registered_trials:
        return _empty_stop("no_speed_qualified_parent")
    candidates = summary.loc[summary["trial_id"].isin(registered_ids)].sort_values(
        [
            "mean_rankic",
            "worst_fold_rankic",
            "samples_per_second",
            "parameter_count",
            "trial_id",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    selected = candidates.iloc[0]
    control_mean = float(
        summary.loc[summary["trial_id"].eq(control_trial_id), "mean_rankic"].iloc[0]
    )
    blockers = []
    if float(selected["mean_rankic"]) < 0.09:
        blockers.append("mean_rankic_below_0.09")
    if int(selected["positive_fold_count"]) != 5:
        blockers.append("not_all_folds_positive")
    if float(selected["samples_per_second"]) < 5000:
        blockers.append("throughput_below_5000")
    if float(selected["mean_rankic"]) < control_mean:
        blockers.append("degrades_lite_c16_no_dropout")
    if blockers:
        return Seed7Decision(
            status="stop_no_pareto_gain_v9",
            winner_trial_id=None,
            confirmation_seeds=(),
            blockers=tuple(blockers),
            summary=summary,
        )
    return Seed7Decision(
        status="seed7_winner_admitted",
        winner_trial_id=str(selected["trial_id"]),
        confirmation_seeds=(17, 27),
        blockers=(),
        summary=summary,
    )
