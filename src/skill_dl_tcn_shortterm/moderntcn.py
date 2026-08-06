"""Causal ModernTCN adaptation for the post-MVP validation experiment.

The project keeps the paper's central patch embedding, large-kernel depthwise
temporal convolution, per-variable pointwise FFN, and cross-variable pointwise
FFN. Padding is adapted to be left-only because this repository's prediction
contract forbids future context.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional

from .baselines import _summarize_predictions
from .experiment import ContractError
from .neural import HORIZONS, NeuralResult, _label_matrices
from .training_data import (
    LazyWindowDataset,
    build_fold_protocols,
    fit_model,
    predict_model,
)


def modern_receptive_field(
    *, patch_size: int, patch_stride: int, large_kernel_size: int, block_count: int
) -> int:
    """Return the effective raw-step receptive field of the causal backbone."""

    return patch_size + patch_stride * block_count * (large_kernel_size - 1)


class ModernTCNBlock(nn.Module):
    """Large-kernel temporal and decoupled variable-mixing residual block."""

    def __init__(
        self,
        *,
        variable_count: int,
        d_model: int,
        ffn_ratio: int,
        large_kernel_size: int,
        small_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = variable_count * d_model
        self.large_padding = large_kernel_size - 1
        self.small_padding = small_kernel_size - 1
        self.large_depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=large_kernel_size,
            groups=channels,
            padding=0,
        )
        self.small_depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=small_kernel_size,
            groups=channels,
            padding=0,
        )
        self.normalization = nn.LayerNorm(d_model)
        per_variable_hidden = d_model * ffn_ratio
        self.variable_ffn = nn.Sequential(
            nn.Conv1d(
                channels, variable_count * per_variable_hidden, 1, groups=variable_count
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                variable_count * per_variable_hidden, channels, 1, groups=variable_count
            ),
            nn.Dropout(dropout),
        )
        cross_hidden = variable_count * ffn_ratio
        self.cross_variable_up = nn.Conv1d(
            d_model * variable_count,
            d_model * cross_hidden,
            1,
            groups=d_model,
        )
        self.cross_variable_down = nn.Conv1d(
            d_model * cross_hidden,
            d_model * variable_count,
            1,
            groups=d_model,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.variable_count = variable_count
        self.d_model = d_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, variables, dimensions, steps = inputs.shape
        outputs = inputs.reshape(batch, variables * dimensions, steps)
        temporal = self.large_depthwise(
            functional.pad(outputs, (self.large_padding, 0))
        )
        temporal = temporal + self.small_depthwise(
            functional.pad(outputs, (self.small_padding, 0))
        )
        temporal = temporal.reshape(batch, variables, dimensions, steps)
        temporal = self.normalization(temporal.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        temporal = self.variable_ffn(
            temporal.reshape(batch, variables * dimensions, steps)
        )
        temporal = temporal.reshape(batch, variables, dimensions, steps)
        cross = temporal.permute(0, 2, 1, 3).reshape(
            batch, dimensions * variables, steps
        )
        cross = self.dropout(self.activation(self.cross_variable_up(cross)))
        cross = self.dropout(self.cross_variable_down(cross))
        cross = cross.reshape(batch, dimensions, variables, steps).permute(0, 2, 1, 3)
        return inputs + cross


class ModernTCN(nn.Module):
    """Four-horizon causal ModernTCN backbone for one cross-sectional sample."""

    def __init__(
        self,
        *,
        feature_count: int,
        d_model: int,
        ffn_ratio: int,
        patch_size: int,
        patch_stride: int,
        large_kernel_size: int,
        small_kernel_size: int,
        block_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if (
            min(
                feature_count, d_model, ffn_ratio, patch_size, patch_stride, block_count
            )
            <= 0
        ):
            raise ContractError("ModernTCN dimensions and block count must be positive")
        if not 0 < small_kernel_size <= large_kernel_size:
            raise ContractError(
                "ModernTCN small kernel must be positive and no larger than large kernel"
            )
        self.patch_left_padding = patch_size - 1
        self.patch_embedding = nn.Conv1d(
            1, d_model, kernel_size=patch_size, stride=patch_stride, padding=0
        )
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(
                    variable_count=feature_count,
                    d_model=d_model,
                    ffn_ratio=ffn_ratio,
                    large_kernel_size=large_kernel_size,
                    small_kernel_size=small_kernel_size,
                    dropout=dropout,
                )
                for _ in range(block_count)
            ]
        )
        self.head = nn.Linear(feature_count * d_model, len(HORIZONS))

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, variables, steps = inputs.shape
        patches = self.patch_embedding(
            functional.pad(
                inputs.reshape(batch * variables, 1, steps),
                (self.patch_left_padding, 0),
            )
        )
        outputs = patches.reshape(batch, variables, patches.shape[1], patches.shape[2])
        for block in self.blocks:
            outputs = block(outputs)
        return outputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.encode_sequence(inputs)
        return self.head(sequence[..., -1].reshape(inputs.shape[0], -1))


def run_moderntcn_experiment(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    seed: int,
    d_model: int,
    ffn_ratio: int,
    patch_size: int,
    patch_stride: int,
    large_kernel_size: int,
    small_kernel_size: int,
    block_count: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    num_workers: int = 0,
) -> NeuralResult:
    """Run a validation-only post-MVP experiment without sealed-test access."""

    if len(features) != len(window_index):
        raise ContractError("features and window index sample counts must match")
    receptive_field = modern_receptive_field(
        patch_size=patch_size,
        patch_stride=patch_stride,
        large_kernel_size=large_kernel_size,
        block_count=block_count,
    )
    if receptive_field < features.shape[2]:
        raise ContractError(
            f"ModernTCN receptive field {receptive_field} is smaller than input window {features.shape[2]}"
        )
    targets, masks = _label_matrices(window_index, labels)
    protocols = build_fold_protocols(features, split_manifest)
    sample_by_position = window_index.set_index("sample_position")[
        "sample_id"
    ].to_dict()
    labels_by_key = labels.set_index(["sample_id", "horizon"])
    prediction_rows = []
    architecture = {
        "feature_count": features.shape[1],
        "d_model": d_model,
        "ffn_ratio": ffn_ratio,
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "large_kernel_size": large_kernel_size,
        "small_kernel_size": small_kernel_size,
        "block_count": block_count,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
    }
    architecture_json = json.dumps(architecture, sort_keys=True, separators=(",", ":"))
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
        model = ModernTCN(
            feature_count=features.shape[1],
            d_model=d_model,
            ffn_ratio=ffn_ratio,
            patch_size=patch_size,
            patch_stride=patch_stride,
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            block_count=block_count,
            dropout=dropout,
        )
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
            for horizon_position, horizon in enumerate(HORIZONS):
                key = (sample_id, horizon)
                if key not in labels_by_key.index or not bool(
                    labels_by_key.loc[key, "valid"]
                ):
                    continue
                label = labels_by_key.loc[key]
                prediction_rows.append(
                    {
                        "model": "moderntcn-post-mvp",
                        "fold": protocol.fold,
                        "stage": "validation",
                        "sample_id": sample_id,
                        "instrument_id": label["instrument_id"],
                        "signal_date": label["signal_date"],
                        "horizon": horizon,
                        "score": float(
                            validation_scores[row_position, horizon_position]
                        ),
                        "target": float(label["rank_target"]),
                    }
                )
        metadata_rows.append(
            {
                "model": "moderntcn-post-mvp",
                "fold": protocol.fold,
                "experiment_class": "post_mvp_non_blocking",
                "architecture_sha256": hashlib.sha256(
                    architecture_json.encode()
                ).hexdigest(),
                "architecture_json": architecture_json,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "receptive_field": receptive_field,
                "epochs": epochs,
                "batch_size": batch_size,
                "precision": "float32",
                "optimizer": "Adam(lr=0.01)",
                "seed": model_seed,
                "stopping_rule": "fixed_epochs_validation_only",
                "sealed_test_access": False,
                "final_train_loss": receipt.final_loss,
                "training_seconds": receipt.training_seconds,
                "samples_per_second": receipt.samples_per_second,
                "loader": "worker_safe_lazy_window_dataset",
                "storage": train_dataset.storage,
                "num_workers": num_workers,
                "conclusion_policy": "negative_result_does_not_change_mvp_status",
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    metadata = pd.DataFrame(metadata_rows)
    return NeuralResult(
        predictions=predictions,
        metrics=_summarize_predictions(predictions, seed),
        training_metadata=metadata,
    )
