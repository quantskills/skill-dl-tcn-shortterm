"""Bai-style causal dilated residual TCN."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as functional
from torch.nn.utils.parametrizations import weight_norm

from .baselines import _summarize_predictions
from .experiment import ContractError
from .neural import HORIZONS, NeuralResult, _label_matrices
from .training_data import (
    LazyWindowDataset,
    build_fold_protocols,
    fit_model,
    predict_model,
)


def receptive_field(
    *, kernel_size: int, dilations: Sequence[int], convolutions_per_block: int = 2
) -> int:
    return 1 + convolutions_per_block * (kernel_size - 1) * sum(
        int(value) for value in dilations
    )


def validate_receptive_field(
    *, input_steps: int, kernel_size: int, dilations: Sequence[int]
) -> None:
    observed = receptive_field(kernel_size=kernel_size, dilations=dilations)
    if observed < input_steps:
        raise ContractError(
            f"receptive field {observed} is smaller than input window {input_steps}"
        )


class CausalResidualBlock(nn.Module):
    """Two left-padded WeightNorm convolutions and one residual path."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                dilation=dilation,
                padding=0,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                dilation=dilation,
                padding=0,
            )
        )
        self.dropout = nn.Dropout(dropout)
        self.projection: nn.Module = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        outputs = functional.pad(inputs, (self.left_padding, 0))
        outputs = self.dropout(functional.relu(self.conv1(outputs)))
        outputs = functional.pad(outputs, (self.left_padding, 0))
        outputs = self.dropout(functional.relu(self.conv2(outputs)))
        return functional.relu(outputs + residual)


class BaiTCN(nn.Module):
    """Shared causal TCN trunk with four independent horizon scores."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = []
        input_channels = feature_count
        for dilation in dilations:
            blocks.append(
                CausalResidualBlock(
                    input_channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
            )
            input_channels = channels
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 4)
        self.kernel_size = kernel_size
        self.dilations = tuple(int(value) for value in dilations)

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.trunk(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.encode_sequence(inputs)
        return self.head(sequence[:, :, -1])


def run_bai_tcn(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    seed: int,
    channels: int,
    kernel_size: int,
    dilations: Sequence[int],
    dropout: float,
    epochs: int,
    batch_size: int,
    num_workers: int = 0,
    model_kind: str = "bai",
    active_horizons: Sequence[int] | None = None,
    model_name: str | None = None,
) -> NeuralResult:
    """Train one Bai TCN under the shared sequence-model contract."""

    if len(features) != len(window_index):
        raise ContractError("features and window index sample counts must match")
    if model_kind == "bai":
        observed_receptive_field = receptive_field(
            kernel_size=kernel_size, dilations=dilations
        )
        validate_receptive_field(
            input_steps=features.shape[2], kernel_size=kernel_size, dilations=dilations
        )
        resolved_model_name = model_name or "bai-tcn"

        def model_factory() -> nn.Module:
            return BaiTCN(
                feature_count=features.shape[1],
                channels=channels,
                kernel_size=kernel_size,
                dilations=dilations,
                dropout=dropout,
            )

    elif model_kind == "lite":
        from .tcn_lite import TCNLite, lite_receptive_field

        observed_receptive_field = lite_receptive_field(
            kernel_size=kernel_size, dilations=dilations
        )
        if observed_receptive_field < features.shape[2]:
            raise ContractError(
                f"receptive field {observed_receptive_field} is smaller than input window {features.shape[2]}"
            )
        resolved_model_name = model_name or "tcn-lite"

        def model_factory() -> nn.Module:
            return TCNLite(
                feature_count=features.shape[1],
                channels=channels,
                kernel_size=kernel_size,
                dilations=dilations,
                dropout=dropout,
            )

    else:
        raise ContractError("model_kind must be bai or lite")
    targets, masks = _label_matrices(window_index, labels)
    active = set(HORIZONS if active_horizons is None else active_horizons)
    for column, horizon in enumerate(HORIZONS):
        if horizon not in active:
            masks[:, column] = False
    protocols = build_fold_protocols(features, split_manifest)
    sample_by_position = window_index.set_index("sample_position")[
        "sample_id"
    ].to_dict()
    labels_by_key = labels.set_index(["sample_id", "horizon"])
    rows = []
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
        model_seed = seed + protocol.fold * 100
        torch.manual_seed(model_seed)
        model = model_factory()
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
                if horizon not in active:
                    continue
                key = (sample_id, horizon)
                if key not in labels_by_key.index or not bool(
                    labels_by_key.loc[key, "valid"]
                ):
                    continue
                label = labels_by_key.loc[key]
                rows.append(
                    {
                        "model": resolved_model_name,
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
                "model": resolved_model_name,
                "fold": protocol.fold,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "channels": channels,
                "kernel_size": kernel_size,
                "dilations": ",".join(str(value) for value in dilations),
                "receptive_field": observed_receptive_field,
                "active_horizons": ",".join(str(value) for value in sorted(active)),
                "dropout": dropout,
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
    predictions = pd.DataFrame(rows)
    metadata = pd.DataFrame(metadata_rows)
    return NeuralResult(
        predictions=predictions,
        metrics=_summarize_predictions(predictions, seed),
        training_metadata=metadata,
    )
