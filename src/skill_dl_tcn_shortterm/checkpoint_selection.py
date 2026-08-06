"""Deterministic checkpoint selection over one shared TCN epoch trajectory."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


SELECTION_METRICS = (
    "rankic",
    "top_precision",
    "ndcg_at_top",
    "top_return",
    "top_turnover",
)


def select_constrained_tail_checkpoints(
    epoch_metrics: pd.DataFrame,
    *,
    expected_epochs: tuple[int, ...],
    rankic_tolerance: float,
) -> pd.DataFrame:
    """Compare RankIC and constrained tail selection on a shared trajectory.

    The candidate maximizes an equal-weight average of top precision and NDCG
    only among checkpoints whose RankIC is within ``rankic_tolerance`` of the
    unit's best RankIC.  Tie-breaks are public and deterministic.
    """

    required = {"seed", "fold", "epoch", *SELECTION_METRICS}
    if missing := sorted(required.difference(epoch_metrics.columns)):
        raise ContractError(
            "checkpoint selection metrics missing columns: " + ", ".join(missing)
        )
    if epoch_metrics.empty:
        raise ContractError("checkpoint selection metrics cannot be empty")
    if epoch_metrics.duplicated(["seed", "fold", "epoch"]).any():
        raise ContractError("checkpoint selection contains duplicate unit epochs")
    if (
        not expected_epochs
        or tuple(sorted(set(expected_epochs))) != expected_epochs
        or expected_epochs[0] != 0
    ):
        raise ContractError("checkpoint selection expected epochs are invalid")
    if not np.isfinite(rankic_tolerance) or rankic_tolerance <= 0:
        raise ContractError("checkpoint selection RankIC tolerance must be positive")
    numeric = epoch_metrics[["seed", "fold", "epoch", *SELECTION_METRICS]].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(numeric).all():
        raise ContractError("checkpoint selection metrics must be finite")

    expected_set = set(expected_epochs)
    rows: list[dict[str, object]] = []
    for (seed_value, fold_value), raw_group in epoch_metrics.groupby(
        ["seed", "fold"], observed=True, sort=True
    ):
        group = raw_group.copy()
        observed_epochs = set(group["epoch"].astype(int))
        if observed_epochs != expected_set or len(group) != len(expected_epochs):
            raise ContractError("checkpoint selection epoch coverage drifted")
        group["tail_selection_score"] = 0.5 * (
            group["top_precision"].astype(float)
            + group["ndcg_at_top"].astype(float)
        )
        control = group.sort_values(
            ["rankic", "epoch"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        unit_max_rankic = float(group["rankic"].max())
        rankic_floor = unit_max_rankic - rankic_tolerance
        feasible = group.loc[
            group["rankic"].astype(float).ge(rankic_floor - 1e-12)
        ].copy()
        if feasible.empty:
            raise ContractError("checkpoint selection has no RankIC-feasible epoch")
        candidate = feasible.sort_values(
            ["tail_selection_score", "rankic", "top_turnover", "epoch"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).iloc[0]
        row: dict[str, object] = {
            "seed": int(cast(Any, seed_value)),
            "fold": int(cast(Any, fold_value)),
            "control_epoch": int(cast(Any, control["epoch"])),
            "candidate_epoch": int(cast(Any, candidate["epoch"])),
            "unit_max_rankic": unit_max_rankic,
            "rankic_tolerance": rankic_tolerance,
            "candidate_rankic_floor": rankic_floor,
            "candidate_rankic_feasible": bool(
                float(candidate["rankic"]) >= rankic_floor - 1e-12
            ),
            "selection_changed": bool(control["epoch"] != candidate["epoch"]),
            "control_tail_selection_score": float(
                control["tail_selection_score"]
            ),
            "candidate_tail_selection_score": float(
                candidate["tail_selection_score"]
            ),
        }
        for metric in SELECTION_METRICS:
            control_value = float(control[metric])
            candidate_value = float(candidate[metric])
            row[f"control_{metric}"] = control_value
            row[f"candidate_{metric}"] = candidate_value
            row[f"{metric}_delta"] = candidate_value - control_value
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["seed", "fold"], ignore_index=True)
    if result.empty or result.duplicated(["seed", "fold"]).any():
        raise ContractError("checkpoint selection unit coverage is invalid")
    return result
