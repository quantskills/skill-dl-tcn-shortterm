"""Fold-scoped cross-seed TCN consensus targets for single-model distillation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError
from .neural import HORIZONS


def _validated_positions(
    positions: np.ndarray, *, sample_count: int
) -> np.ndarray:
    values = np.asarray(positions, dtype="int64")
    if (
        values.ndim != 1
        or len(values) == 0
        or len(np.unique(values)) != len(values)
        or bool((values < 0).any())
        or bool((values >= sample_count).any())
    ):
        raise ContractError("consensus train positions are invalid")
    return np.sort(values)


def build_fold_consensus_rank_targets(
    teacher_predictions: pd.DataFrame,
    *,
    sample_count: int,
    train_positions: np.ndarray,
    expected_seeds: Sequence[int],
) -> np.ndarray:
    """Rank a frozen seed ensemble within each train-date/horizon cross-section."""

    required = {"seed", "sample_position", "signal_date", "horizon", "score"}
    if missing := sorted(required.difference(teacher_predictions.columns)):
        raise ContractError("consensus teacher predictions missing: " + ", ".join(missing))
    if sample_count <= 0:
        raise ContractError("consensus sample count must be positive")
    positions = _validated_positions(train_positions, sample_count=sample_count)
    seeds = tuple(int(value) for value in expected_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ContractError("consensus expected seeds must be unique and non-empty")
    predictions = teacher_predictions.copy()
    predictions["seed"] = predictions["seed"].astype(int)
    predictions["sample_position"] = predictions["sample_position"].astype(int)
    predictions["horizon"] = predictions["horizon"].astype(int)
    if set(predictions["seed"]) != set(seeds):
        raise ContractError("consensus teacher seed coverage drifted")
    if set(predictions["horizon"]) != set(HORIZONS):
        raise ContractError("consensus teacher horizon coverage drifted")
    if predictions.duplicated(["seed", "sample_position", "horizon"]).any():
        raise ContractError("consensus teacher predictions contain duplicates")
    expected_units = {
        (seed, int(position), int(horizon))
        for seed in seeds
        for position in positions
        for horizon in HORIZONS
    }
    observed_units = {
        (
            int(cast(Any, row.seed)),
            int(cast(Any, row.sample_position)),
            int(cast(Any, row.horizon)),
        )
        for row in predictions.itertuples(index=False)
    }
    if observed_units != expected_units:
        raise ContractError("consensus teacher train-position coverage drifted")
    scores = predictions["score"].to_numpy(dtype="float64")
    if not np.isfinite(scores).all():
        raise ContractError("consensus teacher scores must be finite")
    date_counts = predictions.groupby("sample_position", observed=True)[
        "signal_date"
    ].nunique()
    if not date_counts.eq(1).all():
        raise ContractError("consensus teacher sample dates drifted across seeds")

    ensemble = predictions.groupby(
        ["sample_position", "signal_date", "horizon"],
        as_index=False,
        observed=True,
    ).agg(score=("score", "mean"))

    rank_groups = ensemble.groupby(["signal_date", "horizon"], observed=True)
    ranks = rank_groups["score"].rank(method="average")
    sizes = rank_groups["score"].transform("size").astype("float64")
    ensemble["teacher_rank"] = np.where(
        sizes.eq(1.0), 0.0, 2.0 * ((ranks - 1.0) / (sizes - 1.0)) - 1.0
    )
    teacher_ranks = ensemble["teacher_rank"].to_numpy(dtype="float64")
    if (
        not np.isfinite(teacher_ranks).all()
        or bool((teacher_ranks < -1.0).any())
        or bool((teacher_ranks > 1.0).any())
    ):
        raise ContractError("consensus teacher ranks are invalid")
    result = np.full((sample_count, len(HORIZONS)), np.nan, dtype="float32")
    horizon_offsets = {horizon: offset for offset, horizon in enumerate(HORIZONS)}
    for row in ensemble.itertuples(index=False):
        result[
            int(cast(Any, row.sample_position)),
            horizon_offsets[int(cast(Any, row.horizon))],
        ] = float(cast(Any, row.teacher_rank))
    if not np.isfinite(result[positions]).all():
        raise ContractError("consensus target matrix is incomplete on train positions")
    return result


def blend_training_targets(
    true_targets: np.ndarray,
    masks: np.ndarray,
    teacher_targets: np.ndarray,
    *,
    train_positions: np.ndarray,
    teacher_weight: float,
) -> np.ndarray:
    """Blend only valid train cells and preserve every non-train target exactly."""

    true_values = np.asarray(true_targets)
    valid_masks = np.asarray(masks, dtype="bool")
    teacher_values = np.asarray(teacher_targets)
    if (
        true_values.ndim != 2
        or true_values.shape != valid_masks.shape
        or true_values.shape != teacher_values.shape
    ):
        raise ContractError("distillation target matrices must have one shared shape")
    if not np.isfinite(teacher_weight) or not 0.0 < teacher_weight < 1.0:
        raise ContractError("teacher weight must be finite and in (0, 1)")
    positions = _validated_positions(
        train_positions, sample_count=int(true_values.shape[0])
    )
    train_mask = valid_masks[positions]
    if not np.isfinite(true_values[positions][train_mask]).all():
        raise ContractError("true train targets contain non-finite valid values")
    if not np.isfinite(teacher_values[positions][train_mask]).all():
        raise ContractError("teacher targets are incomplete on valid train cells")
    bounded = teacher_values[positions][train_mask]
    if bool((bounded < -1.0).any()) or bool((bounded > 1.0).any()):
        raise ContractError("teacher targets must stay in [-1, 1]")
    blended = true_values.copy()
    train_values = blended[positions].copy()
    source_values = true_values[positions]
    teacher_train = teacher_values[positions]
    train_values[train_mask] = (
        (1.0 - teacher_weight) * source_values[train_mask]
        + teacher_weight * teacher_train[train_mask]
    )
    blended[positions] = train_values
    return blended.astype(true_values.dtype, copy=False)
