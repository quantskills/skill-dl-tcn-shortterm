"""Paired ordinary-validation RankIC resolution audit for TCN-v9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from .experiment import ContractError
from .neural import HORIZONS


@dataclass(frozen=True)
class RankICResolutionAudit:
    status: str
    rank_objective_allowed: bool
    blockers: tuple[str, ...]
    summary: pd.DataFrame


def _paired_block_bootstrap(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    draws: int,
) -> tuple[np.ndarray, float]:
    ordered = np.asarray(values, dtype="float64")
    if ordered.ndim != 1 or len(ordered) < 2 or draws <= 0:
        raise ContractError("RankIC bootstrap requires at least two dates and positive draws")
    block_length = max(1, min(len(ordered), int(np.ceil(np.sqrt(len(ordered))))))
    block_count = int(np.ceil(len(ordered) / block_length))
    offsets = np.arange(block_length)
    means = np.empty(draws, dtype="float64")
    degenerate = 0
    for draw in range(draws):
        starts = rng.integers(0, len(ordered), size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % len(ordered)).reshape(-1)
        sample = ordered[indices[: len(ordered)]]
        means[draw] = float(sample.mean())
        if np.unique(sample).size < 2:
            degenerate += 1
    return means, degenerate / draws


def audit_rankic_resolution(
    daily_evidence: pd.DataFrame,
    *,
    seed: int,
    bootstrap_draws: int = 1_000,
    expected_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> RankICResolutionAudit:
    """Estimate whether a paired 0.005 RankIC effect is statistically resolvable."""

    required = {
        "fold",
        "horizon",
        "signal_date",
        "control_rankic",
        "candidate_rankic",
        "valid_member_count",
        "label_overlap_days",
        "valid",
        "stage",
        "sealed",
    }
    if missing := sorted(required.difference(daily_evidence.columns)):
        raise ContractError(f"RankIC resolution evidence missing columns: {', '.join(missing)}")
    if daily_evidence.empty:
        raise ContractError("RankIC resolution evidence cannot be empty")
    if daily_evidence["sealed"].astype(bool).any():
        raise ContractError("RankIC resolution audit rejects sealed rows")
    if set(daily_evidence["stage"].astype(str)) != {"validation"}:
        raise ContractError("RankIC resolution audit accepts only ordinary validation")
    if daily_evidence.duplicated(["fold", "horizon", "signal_date"]).any():
        raise ContractError("RankIC resolution evidence contains duplicate paired dates")
    valid = daily_evidence.loc[
        daily_evidence["valid"].astype(bool)
        & daily_evidence["valid_member_count"].ge(2)
    ].copy()
    numeric = [
        "control_rankic",
        "candidate_rankic",
        "valid_member_count",
        "label_overlap_days",
    ]
    if valid.empty or not np.isfinite(valid[numeric].to_numpy(dtype="float64")).all():
        raise ContractError("RankIC resolution valid evidence is empty or non-finite")
    if valid["label_overlap_days"].lt(0).any():
        raise ContractError("label overlap days cannot be negative")

    rng = np.random.default_rng(seed)
    rows = []
    observed_units: set[tuple[int, int]] = set()
    for unit_values, group in valid.groupby(["fold", "horizon"], observed=True):
        fold_value, horizon_value = unit_values
        fold = int(cast(int, fold_value))
        horizon = int(cast(int, horizon_value))
        observed_units.add((fold, horizon))
        ordered = group.sort_values("signal_date", kind="mergesort")
        deltas = (
            ordered["candidate_rankic"] - ordered["control_rankic"]
        ).to_numpy(dtype="float64")
        draws, degenerate_rate = _paired_block_bootstrap(
            deltas,
            rng,
            draws=bootstrap_draws,
        )
        standard_error = float(np.std(draws, ddof=1))
        minimum_detectable_effect = float((1.9599639845 + 0.8416212336) * standard_error)
        autocorrelation = float(ordered["control_rankic"].autocorr(lag=1))
        low, high = np.quantile(draws, [0.025, 0.975]).tolist()
        rows.append(
            {
                "fold": fold,
                "horizon": horizon,
                "paired_date_count": int(ordered["signal_date"].nunique()),
                "median_valid_member_count": float(ordered["valid_member_count"].median()),
                "label_overlap_days": float(ordered["label_overlap_days"].mean()),
                "rankic_autocorrelation": autocorrelation,
                "paired_mean_delta": float(deltas.mean()),
                "bootstrap_standard_error": standard_error,
                "delta_ci_low": float(low),
                "delta_ci_high": float(high),
                "degenerate_bootstrap_rate": float(degenerate_rate),
                "minimum_detectable_effect": minimum_detectable_effect,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["fold", "horizon"], kind="mergesort").reset_index(drop=True)
    if not expected_folds or len(set(expected_folds)) != len(expected_folds):
        raise ContractError("RankIC resolution expected folds are invalid")
    expected_units = {
        (int(fold), int(horizon))
        for fold in expected_folds
        for horizon in HORIZONS
    }
    blockers = []
    if observed_units != expected_units:
        blockers.append("missing_fold_horizon_units")
    if summary["paired_date_count"].lt(40).any():
        blockers.append("insufficient_paired_dates")
    if summary["degenerate_bootstrap_rate"].gt(0.05).any():
        blockers.append("degenerate_bootstrap")
    if (
        ~np.isfinite(summary["minimum_detectable_effect"].to_numpy(dtype="float64"))
    ).any() or summary["minimum_detectable_effect"].gt(0.005).any():
        blockers.append("minimum_detectable_effect_above_0.005")
    allowed = not blockers
    return RankICResolutionAudit(
        status=("rank_objective_allowed" if allowed else "rank_objective_not_resolvable"),
        rank_objective_allowed=allowed,
        blockers=tuple(blockers),
        summary=summary,
    )
