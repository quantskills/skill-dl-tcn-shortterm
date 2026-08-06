"""Lightweight causal TCN and horizon-sharing diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as functional
from torch.nn.utils.parametrizations import weight_norm

from .baselines import _block_bootstrap_means, _rankic_by_date
from .neural import NeuralResult
from .tcn import run_bai_tcn


def lite_receptive_field(*, kernel_size: int, dilations: Sequence[int]) -> int:
    return 1 + (kernel_size - 1) * sum(int(value) for value in dilations)


class CausalLiteBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        dropout_kind: Literal["element", "channel"],
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = weight_norm(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                dilation=dilation,
                padding=0,
            )
        )
        self.dropout: nn.Module = (
            nn.Dropout(dropout)
            if dropout_kind == "element"
            else nn.Dropout1d(dropout)
        )
        self.projection: nn.Module = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        outputs = functional.pad(inputs, (self.left_padding, 0))
        outputs = self.dropout(functional.relu(self.convolution(outputs)))
        return functional.relu(outputs + residual)


class TCNLite(nn.Module):
    """One-convolution residual blocks with a four-horizon head."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        dropout_kind: Literal["element", "channel"] = "element",
    ) -> None:
        super().__init__()
        if not 0 <= head_dropout < 1:
            raise ValueError("head dropout must be in [0, 1)")
        if dropout_kind not in {"element", "channel"}:
            raise ValueError("dropout kind must be element or channel")
        blocks = []
        input_channels = feature_count
        for dilation in dilations:
            blocks.append(
                CausalLiteBlock(
                    input_channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                    dropout_kind=dropout_kind,
                )
            )
            input_channels = channels
        self.trunk = nn.Sequential(*blocks)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(channels, 4)

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.trunk(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        final_representation = self.encode_sequence(inputs)[:, :, -1]
        return self.head(self.head_dropout(final_representation))


def compare_shared_and_single_horizon(
    shared_metrics: pd.DataFrame,
    single_metrics: pd.DataFrame,
    *,
    shared_predictions: pd.DataFrame | None = None,
    single_predictions: pd.DataFrame | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Classify per-horizon negative transfer using paired validation dates."""

    shared = shared_metrics[["horizon", "rankic"]].rename(
        columns={"rankic": "shared_rankic"}
    )
    candidate = single_metrics[["horizon", "rankic"]].rename(
        columns={"rankic": "candidate_rankic"}
    )
    comparison = shared.merge(candidate, on="horizon", validate="one_to_one")
    comparison["candidate_minus_shared"] = (
        comparison["candidate_rankic"] - comparison["shared_rankic"]
    )
    comparison["paired_date_count"] = 0
    comparison["delta_ci_low"] = float("nan")
    comparison["delta_ci_high"] = float("nan")

    if shared_predictions is not None and single_predictions is not None:
        shared_daily = _daily_rankic_frame(shared_predictions, "shared_rankic")
        single_daily = _daily_rankic_frame(single_predictions, "candidate_rankic")
        paired = shared_daily.merge(
            single_daily,
            on=["fold", "horizon", "signal_date"],
            validate="one_to_one",
        )
        paired["delta"] = paired["candidate_rankic"] - paired["shared_rankic"]
        rng = np.random.default_rng(seed)
        for horizon, group in paired.groupby("horizon", observed=True):
            horizon = cast(Any, horizon)
            deltas = group["delta"].dropna()
            horizon_mask = comparison["horizon"] == int(horizon)
            comparison.loc[horizon_mask, "paired_date_count"] = len(deltas)
            if not deltas.empty:
                comparison.loc[horizon_mask, "candidate_minus_shared"] = float(
                    deltas.mean()
                )
                draws = _block_bootstrap_means(deltas.to_numpy(), rng)
                low, high = np.quantile(draws, [0.025, 0.975]).tolist()
                comparison.loc[horizon_mask, "delta_ci_low"] = low
                comparison.loc[horizon_mask, "delta_ci_high"] = high

    conclusions = []
    for row in comparison.itertuples(index=False):
        row = cast(Any, row)
        delta = float(row.candidate_minus_shared)
        if not np.isfinite(delta):
            conclusions.append("insufficient-evidence")
        elif int(row.paired_date_count) > 0 and float(row.delta_ci_low) > 0:
            conclusions.append("negative-transfer")
        elif int(row.paired_date_count) > 0 and float(row.delta_ci_high) < 0:
            conclusions.append("no-negative-transfer")
        elif int(row.paired_date_count) > 0:
            conclusions.append("uncertain")
        elif delta > 1e-12:
            conclusions.append("negative-transfer")
        elif delta < -1e-12:
            conclusions.append("no-negative-transfer")
        else:
            conclusions.append("no-difference")
    comparison["conclusion"] = conclusions
    return comparison


def _daily_rankic_frame(predictions: pd.DataFrame, value_name: str) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["fold", "horizon"], observed=True):
        fold, horizon = cast(tuple[Any, Any], keys)
        for signal_date, value in _rankic_by_date(group).items():
            rows.append(
                {
                    "fold": int(fold),
                    "horizon": int(horizon),
                    "signal_date": str(signal_date),
                    value_name: float(value),
                }
            )
    return pd.DataFrame(rows, columns=["fold", "horizon", "signal_date", value_name])


def run_tcn_ablations(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    shared_metrics: pd.DataFrame,
    *,
    shared_predictions: pd.DataFrame | None = None,
    seed: int,
    channels: int,
    kernel_size: int,
    lite_dilations: Sequence[int],
    bai_dilations: Sequence[int],
    dropout: float,
    epochs: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[NeuralResult, pd.DataFrame]:
    """Run TCN-lite plus four independently trained single-horizon Bai models."""

    lite = run_bai_tcn(
        features,
        window_index,
        labels,
        split_manifest,
        seed=seed,
        channels=channels,
        kernel_size=kernel_size,
        dilations=lite_dilations,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        model_kind="lite",
        model_name="tcn-lite",
    )
    single_results = []
    for offset, horizon in enumerate([1, 2, 3, 5], start=1):
        single_results.append(
            run_bai_tcn(
                features,
                window_index,
                labels,
                split_manifest,
                seed=seed + offset,
                channels=channels,
                kernel_size=kernel_size,
                dilations=bai_dilations,
                dropout=dropout,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                active_horizons=[horizon],
                model_name=f"bai-tcn-{horizon}d",
            )
        )
    single_predictions = pd.concat(
        [result.predictions for result in single_results], ignore_index=True
    )
    single_metrics = pd.concat(
        [result.metrics for result in single_results], ignore_index=True
    )
    single_metadata = pd.concat(
        [result.training_metadata for result in single_results], ignore_index=True
    )
    all_predictions = pd.concat(
        [lite.predictions, single_predictions], ignore_index=True
    )
    all_metrics = pd.concat([lite.metrics, single_metrics], ignore_index=True)
    all_metadata = pd.concat(
        [lite.training_metadata, single_metadata], ignore_index=True
    )
    shared_per_horizon = shared_metrics.groupby("horizon", as_index=False)[
        ["rankic"]
    ].mean()
    single_per_horizon = single_metrics.groupby("horizon", as_index=False)[
        ["rankic"]
    ].mean()
    single_comparison = compare_shared_and_single_horizon(
        shared_per_horizon,
        single_per_horizon,
        shared_predictions=shared_predictions,
        single_predictions=single_predictions,
        seed=seed,
    )
    single_comparison.insert(1, "comparison_type", "single-vs-shared")
    single_comparison.insert(
        2,
        "candidate_model",
        single_comparison["horizon"].map(lambda value: f"bai-tcn-{int(value)}d"),
    )
    lite_per_horizon = lite.metrics.groupby("horizon", as_index=False)[
        ["rankic"]
    ].mean()
    lite_comparison = compare_shared_and_single_horizon(
        shared_per_horizon,
        lite_per_horizon,
        shared_predictions=shared_predictions,
        single_predictions=lite.predictions,
        seed=seed + 10_000,
    )
    lite_comparison.insert(1, "comparison_type", "lite-vs-shared")
    lite_comparison.insert(2, "candidate_model", "tcn-lite")
    comparison = pd.concat(
        [single_comparison, lite_comparison], ignore_index=True
    ).sort_values(["horizon", "comparison_type"], kind="mergesort")
    return NeuralResult(all_predictions, all_metrics, all_metadata), comparison
