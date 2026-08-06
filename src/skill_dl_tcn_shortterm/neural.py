"""Comparable recurrent sequence baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baselines import _summarize_predictions
from .experiment import ContractError
from .training_data import (
    LazyWindowDataset,
    build_fold_protocols,
    fit_model,
    predict_model,
)


HORIZONS = [1, 2, 3, 5]


class RecurrentRegressor(nn.Module):
    """LSTM or GRU implementing the shared four-horizon protocol."""

    def __init__(self, kind: str, feature_count: int, hidden_size: int) -> None:
        super().__init__()
        if kind == "lstm":
            self.recurrent: nn.Module = nn.LSTM(
                feature_count, hidden_size, batch_first=True
            )
        elif kind == "gru":
            self.recurrent = nn.GRU(feature_count, hidden_size, batch_first=True)
        else:
            raise ValueError("kind must be lstm or gru")
        self.head = nn.Linear(hidden_size, len(HORIZONS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.transpose(1, 2)
        outputs, _ = self.recurrent(sequence)
        return self.head(outputs[:, -1, :])


@dataclass(frozen=True)
class NeuralResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    training_metadata: pd.DataFrame


def _label_matrices(
    window_index: pd.DataFrame, labels: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((len(window_index), len(HORIZONS)), dtype="float32")
    mask = np.zeros_like(targets, dtype="bool")
    position_by_id = dict(
        zip(window_index["sample_id"], window_index["sample_position"], strict=True)
    )
    horizon_position = {horizon: position for position, horizon in enumerate(HORIZONS)}
    for label in labels.itertuples(index=False):
        label = cast(Any, label)
        if (
            label.sample_id not in position_by_id
            or label.horizon not in horizon_position
            or not bool(label.valid)
        ):
            continue
        position = int(position_by_id[label.sample_id])
        column = horizon_position[int(label.horizon)]
        targets[position, column] = float(label.rank_target)
        mask[position, column] = True
    return targets, mask


def run_sequence_baselines(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    seed: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    num_workers: int = 0,
) -> NeuralResult:
    """Train LSTM and GRU using identical samples, labels, and budgets."""

    if len(features) != len(window_index):
        raise ContractError("features and window index sample counts must match")
    targets, masks = _label_matrices(window_index, labels)
    protocols = build_fold_protocols(features, split_manifest)
    sample_by_position = window_index.set_index("sample_position")[
        "sample_id"
    ].to_dict()
    labels_by_key = labels.set_index(["sample_id", "horizon"])
    prediction_rows = []
    metadata_rows = []

    for protocol in protocols:
        train_dataset = LazyWindowDataset(
            features,
            protocol.train_positions,
            targets,
            masks,
            protocol.feature_mean,
            protocol.feature_std,
        )
        validation_dataset = LazyWindowDataset(
            features,
            protocol.validation_positions,
            targets,
            masks,
            protocol.feature_mean,
            protocol.feature_std,
        )
        for model_offset, kind in enumerate(["lstm", "gru"]):
            model_seed = seed + protocol.fold * 100 + model_offset
            torch.manual_seed(model_seed)
            model = RecurrentRegressor(kind, features.shape[1], hidden_size)
            receipt = fit_model(
                model,
                train_dataset,
                seed=model_seed,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            validation_scores, validation_positions = predict_model(
                model,
                validation_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            for row_position, sample_position in enumerate(validation_positions):
                sample_id = sample_by_position[int(sample_position)]
                for column, horizon in enumerate(HORIZONS):
                    key = (sample_id, horizon)
                    if key not in labels_by_key.index or not bool(
                        labels_by_key.loc[key, "valid"]
                    ):
                        continue
                    label = labels_by_key.loc[key]
                    prediction_rows.append(
                        {
                            "model": kind,
                            "fold": protocol.fold,
                            "stage": "validation",
                            "sample_id": sample_id,
                            "instrument_id": label["instrument_id"],
                            "signal_date": label["signal_date"],
                            "horizon": horizon,
                            "score": float(validation_scores[row_position, column]),
                            "target": float(label["rank_target"]),
                        }
                    )
            metadata_rows.append(
                {
                    "model": kind,
                    "fold": protocol.fold,
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "layer_count": 1,
                    "hidden_size": hidden_size,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "precision": "float32",
                    "optimizer": "Adam(lr=0.01)",
                    "stopping_rule": "fixed_epochs_validation_only",
                    "seed": model_seed,
                    "final_train_loss": receipt.final_loss,
                    "training_seconds": receipt.training_seconds,
                    "samples_per_second": receipt.samples_per_second,
                    "loader": "worker_safe_lazy_window_dataset",
                    "storage": train_dataset.storage,
                    "num_workers": num_workers,
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    return NeuralResult(
        predictions=predictions,
        metrics=_summarize_predictions(predictions, seed),
        training_metadata=pd.DataFrame(metadata_rows),
    )
