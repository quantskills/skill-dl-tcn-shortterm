"""Point-in-time market context derived from frozen causal stock windows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .experiment import ContractError
from .training_data import LazyWindowDataset


@dataclass(frozen=True)
class PITMarketContext:
    """One shared, label-free market-state vector per signal date."""

    values: np.ndarray
    available_positions: np.ndarray
    field_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    bars_per_day: int
    identity: str
    date_sample_counts: dict[str, int]


@dataclass(frozen=True)
class MarketContextStandardizer:
    """Fold-local context transform fitted on unique training dates."""

    mean: np.ndarray
    std: np.ndarray
    fit_date_count: int
    identity: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        observed = np.asarray(values, dtype="float32")
        if observed.ndim != 2 or observed.shape[1] != len(self.mean):
            raise ContractError("market context transform shape is invalid")
        if not np.isfinite(observed).all():
            raise ContractError("market context transform contains non-finite values")
        return np.asarray((observed - self.mean) / self.std, dtype="float32")


def _context_identity(
    values: np.ndarray,
    available_positions: np.ndarray,
    *,
    field_names: tuple[str, ...],
    feature_indices: tuple[int, ...],
    bars_per_day: int,
    date_sample_counts: dict[str, int],
) -> str:
    metadata = {
        "schema_version": "pit-market-context-v17/v1",
        "shape": list(values.shape),
        "available_positions": available_positions.tolist(),
        "field_names": list(field_names),
        "feature_indices": list(feature_indices),
        "bars_per_day": bars_per_day,
        "date_sample_counts": date_sample_counts,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(np.asarray(values[available_positions], dtype="float32").tobytes())
    return digest.hexdigest()


def build_pit_market_context(
    features: Any,
    window_index: pd.DataFrame,
    *,
    allowed_positions: Sequence[int] | np.ndarray,
    feature_indices: Sequence[int],
    bars_per_day: int,
) -> PITMarketContext:
    """Build median-center and MAD summaries without labels or future dates."""

    if getattr(features, "ndim", None) != 3:
        raise ContractError(
            "market context features must be a three-dimensional tensor"
        )
    required = {
        "sample_position",
        "instrument_id",
        "signal_date",
        "window_end_at",
    }
    if missing := sorted(required.difference(window_index.columns)):
        raise ContractError(
            "market context window index missing columns: " + ", ".join(missing)
        )
    if window_index["sample_position"].duplicated().any():
        raise ContractError(
            "market context window index has duplicate sample positions"
        )
    positions = np.asarray(allowed_positions, dtype="int64")
    if positions.ndim != 1 or len(positions) == 0:
        raise ContractError("market context allowed positions must be non-empty")
    if len(np.unique(positions)) != len(positions):
        raise ContractError("market context allowed positions must be unique")
    if positions.min() < 0 or positions.max() >= int(features.shape[0]):
        raise ContractError("market context allowed positions are out of bounds")
    resolved_features = tuple(int(value) for value in feature_indices)
    if (
        not resolved_features
        or len(set(resolved_features)) != len(resolved_features)
        or min(resolved_features) < 0
        or max(resolved_features) >= int(features.shape[1])
    ):
        raise ContractError("market context feature indices are invalid")
    if bars_per_day <= 0 or bars_per_day > int(features.shape[2]):
        raise ContractError("market context bars per day is invalid")

    indexed = window_index.set_index("sample_position", drop=False)
    missing_positions = sorted(set(map(int, positions)).difference(indexed.index))
    if missing_positions:
        raise ContractError(
            "market context positions are missing from the window index"
        )
    rows = indexed.loc[positions].copy()
    rows["signal_date"] = rows["signal_date"].astype(str)
    window_dates = pd.to_datetime(rows["window_end_at"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    if window_dates.isna().any() or not np.array_equal(
        window_dates.to_numpy(), rows["signal_date"].to_numpy()
    ):
        raise ContractError("every market context window must end on its signal date")

    field_names = tuple(
        f"{prefix}:{feature_index}"
        for prefix in (
            "center_last_day",
            "center_full_window",
            "dispersion_last_day",
            "dispersion_full_window",
        )
        for feature_index in resolved_features
    )
    context_values = np.full(
        (int(features.shape[0]), len(field_names)), np.nan, dtype="float32"
    )
    date_sample_counts: dict[str, int] = {}
    for signal_date, date_rows in rows.groupby("signal_date", sort=True):
        date_positions = date_rows["sample_position"].to_numpy(dtype="int64")
        instrument_count = int(date_rows["instrument_id"].astype(str).nunique())
        if instrument_count < 2:
            raise ContractError(
                f"market context date {signal_date} requires at least two instruments"
            )
        windows = np.asarray(
            features[date_positions][:, resolved_features, :], dtype="float32"
        )
        if not np.isfinite(windows).all():
            raise ContractError(
                "market context source windows contain non-finite values"
            )
        center = np.median(windows, axis=0)
        dispersion = np.median(np.abs(windows - center[None, :, :]), axis=0)
        summary = np.concatenate(
            [
                center[:, -bars_per_day:].mean(axis=1),
                center.mean(axis=1),
                dispersion[:, -bars_per_day:].mean(axis=1),
                dispersion.mean(axis=1),
            ]
        ).astype("float32")
        if not np.isfinite(summary).all():
            raise ContractError("market context summary contains non-finite values")
        context_values[date_positions] = summary
        date_sample_counts[str(signal_date)] = instrument_count

    available = np.sort(positions)
    identity = _context_identity(
        context_values,
        available,
        field_names=field_names,
        feature_indices=resolved_features,
        bars_per_day=bars_per_day,
        date_sample_counts=date_sample_counts,
    )
    return PITMarketContext(
        values=context_values,
        available_positions=available,
        field_names=field_names,
        feature_indices=resolved_features,
        bars_per_day=int(bars_per_day),
        identity=identity,
        date_sample_counts=date_sample_counts,
    )


def fit_market_context_standardizer(
    context: PITMarketContext,
    window_index: pd.DataFrame,
    *,
    train_positions: Sequence[int] | np.ndarray,
) -> MarketContextStandardizer:
    """Fit each fold on one equally weighted context vector per training date."""

    if not {"sample_position", "signal_date"} <= set(window_index.columns):
        raise ContractError("market context scaler requires positions and signal dates")
    positions = np.asarray(train_positions, dtype="int64")
    if (
        positions.ndim != 1
        or len(positions) == 0
        or len(np.unique(positions)) != len(positions)
    ):
        raise ContractError("market context scaler train positions are invalid")
    available = set(map(int, context.available_positions))
    if not set(map(int, positions)) <= available:
        raise ContractError("market context scaler requested unavailable positions")
    indexed = window_index.set_index("sample_position", drop=False)
    if not set(map(int, positions)) <= set(map(int, indexed.index)):
        raise ContractError(
            "market context scaler positions are missing from the index"
        )
    rows = indexed.loc[positions, ["sample_position", "signal_date"]].copy()
    rows["signal_date"] = rows["signal_date"].astype(str)
    representatives: list[np.ndarray] = []
    for signal_date, date_rows in rows.groupby("signal_date", sort=True):
        date_positions = date_rows["sample_position"].to_numpy(dtype="int64")
        date_values = np.asarray(context.values[date_positions], dtype="float32")
        if not np.isfinite(date_values).all():
            raise ContractError(
                "market context scaler source contains non-finite values"
            )
        if not np.array_equal(
            date_values, np.repeat(date_values[:1], len(date_values), axis=0)
        ):
            raise ContractError(
                f"market context is not shared within signal date {signal_date}"
            )
        representatives.append(date_values[0])
    fit_values = np.stack(representatives).astype("float32")
    mean = fit_values.mean(axis=0, dtype="float64").astype("float32")
    std = fit_values.std(axis=0, dtype="float64").astype("float32")
    std[std == 0] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ContractError("market context scaler statistics are non-finite")
    payload = {
        "context_identity": context.identity,
        "fit_dates": sorted(rows["signal_date"].unique().tolist()),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MarketContextStandardizer(mean, std, len(representatives), identity)


class ContextualLazyWindowDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]]
):
    """Lazy stock windows paired with a fold-normalized PIT context vector."""

    def __init__(
        self,
        features: Any,
        positions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        context: PITMarketContext,
        context_standardizer: MarketContextStandardizer,
    ) -> None:
        self._base = LazyWindowDataset(features, positions, targets, masks, mean, std)
        self.positions = np.asarray(positions, dtype="int64")
        available = set(map(int, context.available_positions))
        if not set(map(int, self.positions)) <= available:
            raise ContractError(
                "dataset requested unavailable market context positions"
            )
        selected = np.asarray(context.values[self.positions], dtype="float32")
        self._contexts = context_standardizer.transform(selected)
        self.context_identity = context.identity
        self.context_scaler_identity = context_standardizer.identity

    def __len__(self) -> int:
        return len(self._base)

    @property
    def storage(self) -> str:
        return self._base.storage

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        features, targets, masks, position = self._base[index]
        return (
            features,
            targets,
            masks,
            position,
            torch.from_numpy(self._contexts[index]),
        )
