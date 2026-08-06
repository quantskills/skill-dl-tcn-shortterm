"""Shared lazy, fold-aware training protocol for sequence models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .experiment import ContractError


class LazyWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    """Normalize one ndarray/memmap window at access time, never materializing all folds."""

    def __init__(
        self,
        features: Any,
        positions: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self._feature_path = (
            Path(str(features.filename)).resolve()
            if isinstance(features, np.memmap)
            else None
        )
        self._features = None if self._feature_path is not None else features
        self.positions = np.asarray(positions, dtype="int64")
        self.targets = targets
        self.masks = masks
        self.mean = mean.astype("float32")[:, None]
        self.std = std.astype("float32")[:, None]

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def storage(self) -> str:
        return "read_only_memmap" if self._feature_path is not None else "ndarray"

    def _feature_array(self) -> Any:
        if self._features is None:
            assert self._feature_path is not None
            self._features = np.load(
                self._feature_path, mmap_mode="r", allow_pickle=False
            )
        return self._features

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if self._feature_path is not None:
            state["_features"] = None
        return state

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        position = int(self.positions[index])
        window = np.asarray(self._feature_array()[position], dtype="float32")
        normalized = np.asarray((window - self.mean) / self.std, dtype="float32")
        return (
            torch.from_numpy(normalized),
            torch.from_numpy(self.targets[position]),
            torch.from_numpy(self.masks[position]),
            position,
        )


@dataclass(frozen=True)
class FoldProtocol:
    fold: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray


@dataclass(frozen=True)
class TrainingReceipt:
    final_loss: float
    training_seconds: float
    samples_per_second: float


def build_fold_protocols(
    features: Any, split_manifest: pd.DataFrame
) -> list[FoldProtocol]:
    """Compute train-only statistics independently for every walk-forward fold."""

    protocols = []
    for fold in sorted(split_manifest["fold"].unique()):
        rows = split_manifest.loc[split_manifest["fold"] == fold]
        train_positions = rows.loc[
            rows["stage"] == "train", "sample_position"
        ].to_numpy(dtype="int64")
        validation_positions = rows.loc[
            rows["stage"] == "validation", "sample_position"
        ].to_numpy(dtype="int64")
        if len(train_positions) == 0 or len(validation_positions) == 0:
            raise ContractError(f"fold {fold} requires train and validation samples")
        feature_sum = np.zeros(features.shape[1], dtype="float64")
        feature_sum_square = np.zeros(features.shape[1], dtype="float64")
        value_count = 0
        for position in train_positions:
            window = np.asarray(features[int(position)], dtype="float64")
            feature_sum += window.sum(axis=1)
            feature_sum_square += np.square(window).sum(axis=1)
            value_count += window.shape[1]
        mean = feature_sum / value_count
        variance = np.maximum(feature_sum_square / value_count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[std == 0] = 1.0
        protocols.append(
            FoldProtocol(
                fold=int(fold),
                train_positions=train_positions,
                validation_positions=validation_positions,
                feature_mean=mean,
                feature_std=std,
            )
        )
    return protocols


def masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    losses = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    selected = losses[mask]
    if selected.numel() == 0:
        raise ContractError("a neural batch has no valid labels")
    return selected.mean()


def fit_model(
    model: nn.Module,
    dataset: LazyWindowDataset,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    num_workers: int = 0,
) -> TrainingReceipt:
    """Fit one fixed-budget model from a lazy window loader."""

    if num_workers < 0:
        raise ContractError("num_workers cannot be negative")
    if num_workers > 0 and dataset.storage != "read_only_memmap":
        raise ContractError("multi-worker training requires read-only memmap storage")
    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "generator": torch.Generator().manual_seed(seed),
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = 2
        loader_options["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_options)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    final_loss = float("nan")
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        losses = []
        for batch_features, batch_targets, batch_masks, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            loss = masked_smooth_l1(prediction, batch_targets, batch_masks)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        final_loss = float(np.mean(losses))
    elapsed = time.perf_counter() - start
    return TrainingReceipt(
        final_loss=final_loss,
        training_seconds=elapsed,
        samples_per_second=len(dataset) * epochs / elapsed,
    )


def predict_model(
    model: nn.Module,
    dataset: LazyWindowDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validation scores and sample positions in deterministic index order."""

    if num_workers < 0:
        raise ContractError("num_workers cannot be negative")
    if num_workers > 0 and dataset.storage != "read_only_memmap":
        raise ContractError("multi-worker prediction requires read-only memmap storage")
    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = 2
        loader_options["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_options)
    score_batches = []
    position_batches = []
    model.eval()
    with torch.no_grad():
        for batch_features, _, _, batch_positions in loader:
            score_batches.append(model(batch_features).cpu().numpy())
            position_batches.append(batch_positions.numpy())
    return np.concatenate(score_batches), np.concatenate(position_batches)
