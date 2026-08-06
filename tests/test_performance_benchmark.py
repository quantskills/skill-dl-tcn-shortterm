from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from skill_dl_tcn_shortterm.neural import _label_matrices
from skill_dl_tcn_shortterm.performance import (
    _validation_rankic,
    benchmark_sequence_models,
)
from skill_dl_tcn_shortterm.training_data import LazyWindowDataset


def test_benchmark_records_comparable_protocol_and_performance_receipt(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(17)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, rng.normal(size=(24, 3, 16)).astype("float32"))
    features = np.load(feature_path, mmap_mode="r")
    index = pd.DataFrame(
        {
            "sample_position": range(24),
            "sample_id": [f"s{i}" for i in range(24)],
            "instrument_id": [f"I{i % 6}" for i in range(24)],
            "signal_date": [f"2024-01-{2 + i // 6:02d}" for i in range(24)],
        }
    )
    labels = []
    for row in index.itertuples(index=False):
        for horizon in [1, 2, 3, 5]:
            labels.append(
                {
                    "sample_id": row.sample_id,
                    "signal_date": row.signal_date,
                    "horizon": horizon,
                    "rank_target": float(rng.uniform(-1, 1)),
                    "valid": True,
                }
            )
    fold_zero = index.copy()
    fold_zero["fold"] = 0
    fold_zero["stage"] = ["train"] * 18 + ["validation"] * 6
    fold_one = index.iloc[:18].copy()
    fold_one["fold"] = 1
    fold_one["stage"] = ["train"] * 12 + ["validation"] * 6
    split = pd.concat([fold_zero, fold_one], ignore_index=True)

    result = benchmark_sequence_models(
        features,
        index,
        pd.DataFrame(labels),
        split,
        seed=5,
        hidden_size=4,
        tcn_channels=4,
        tcn_kernel_size=2,
        tcn_dilations=(1, 2, 4, 8),
        epochs=1,
        batch_size=6,
        device="cpu",
    )

    assert set(result.measurements["model"]) == {"lstm", "gru", "bai-tcn"}
    assert set(result.measurements["fold"]) == {0, 1}
    assert result.measurements.groupby("fold").size().eq(3).all()
    required = {
        "data_fingerprint",
        "fold",
        "batch_size",
        "precision",
        "train_sample_count",
        "validation_sample_count",
        "parameter_count",
        "samples_per_second",
        "mean_epoch_seconds",
        "time_to_best_seconds",
        "best_validation_rankic",
        "peak_ram_bytes",
        "peak_vram_bytes",
        "data_wait_seconds",
        "model_step_seconds",
        "validation_seconds",
        "end_to_end_seconds",
        "model_step_samples_per_second",
        "train_pipeline_samples_per_second",
    }
    assert required <= set(result.measurements.columns)
    assert result.measurements.groupby("fold")["data_fingerprint"].nunique().eq(1).all()
    assert result.measurements.loc[
        result.measurements["fold"] == 0, "train_sample_count"
    ].eq(18).all()
    assert result.measurements.loc[
        result.measurements["fold"] == 1, "train_sample_count"
    ].eq(12).all()
    assert result.measurements["samples_per_second"].gt(0).all()
    assert result.measurements["mean_epoch_seconds"].gt(0).all()
    assert result.measurements["data_wait_seconds"].ge(0).all()
    assert set(result.measurements["loader"]) == {"lazy_window_dataset"}
    assert set(result.measurements["storage"]) == {"read_only_memmap"}
    assert result.environment["device"] == "cpu"
    assert result.environment["window_storage"] == "read_only_memmap"
    assert result.environment["data_workers"] == 0
    assert result.environment["amp"] is False
    ratios = result.environment["observed_tcn_speed_ratios"]
    assert isinstance(ratios, dict)
    assert {
        "tcn_over_lstm_geomean",
        "tcn_over_gru_geomean",
        "tcn_over_lstm_model_step_geomean",
        "tcn_over_gru_model_step_geomean",
    } <= set(ratios)


class _ChannelScoreModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :, 0]


def test_performance_validation_uses_date_horizon_cross_sectional_rankic() -> None:
    scores = np.zeros((6, 4, 1), dtype="float32")
    scores[:, 0, 0] = [-1.0, 0.0, 1.0, 1.0, 0.0, -1.0]
    scores[:, 1, 0] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    index = pd.DataFrame(
        {
            "sample_position": range(6),
            "sample_id": [f"s{value}" for value in range(6)],
            "signal_date": ["2025-01-02"] * 3 + ["2025-01-03"] * 3,
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "signal_date": index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": [-1.0, 0.0, 1.0][sample % 3],
                "valid": True,
            }
            for sample in range(6)
            for horizon in [1, 2]
        ]
    )
    targets, masks = _label_matrices(index, labels)
    dataset = LazyWindowDataset(
        scores,
        np.arange(6),
        targets,
        masks,
        np.zeros(4, dtype="float32"),
        np.ones(4, dtype="float32"),
    )

    observed = _validation_rankic(
        _ChannelScoreModel(),
        dataset,
        batch_size=3,
        device=torch.device("cpu"),
        num_workers=0,
        window_index=index,
        labels=labels,
        validation_loader=DataLoader(dataset, batch_size=3, shuffle=False),
    )

    assert observed == pytest.approx(0.5)


def test_benchmark_can_scope_threads_repeat_seeds_and_include_tcn_lite(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(31)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, rng.normal(size=(12, 2, 8)).astype("float32"))
    features = np.load(feature_path, mmap_mode="r")
    index = pd.DataFrame(
        {
            "sample_position": range(12),
            "sample_id": [f"s{value}" for value in range(12)],
            "signal_date": ["2025-01-02"] * 4
            + ["2025-01-03"] * 4
            + ["2025-01-04"] * 4,
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": row.sample_id,
                "signal_date": row.signal_date,
                "horizon": horizon,
                "rank_target": float((sample_position % 4) / 3 * 2 - 1),
                "valid": True,
            }
            for sample_position, row in enumerate(index.itertuples(index=False))
            for horizon in [1, 2, 3, 5]
        ]
    )
    split = index[["sample_position"]].copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 8 + ["validation"] * 4
    original_threads = torch.get_num_threads()

    result = benchmark_sequence_models(
        features,
        index,
        labels,
        split,
        seed=5,
        seeds=[5, 11],
        hidden_size=2,
        tcn_channels=2,
        tcn_kernel_size=2,
        tcn_dilations=(1, 2, 4),
        epochs=1,
        batch_size=4,
        device="cpu",
        torch_threads=1,
        include_tcn_lite=True,
        tcn_lite_channels=2,
        tcn_lite_dilations=(1, 2, 4, 8),
        models=("lstm", "tcn-lite"),
    )

    assert set(result.measurements["model"]) == {"lstm", "tcn-lite"}
    assert set(result.measurements["base_seed"]) == {5, 11}
    assert result.environment["torch_threads"] == 1
    assert result.environment["models"] == ["lstm", "tcn-lite"]
    observed_ratios = result.environment["observed_tcn_speed_ratios"]
    assert isinstance(observed_ratios, dict)
    assert set(observed_ratios) == {
        "tcn_lite_over_lstm_geomean",
        "tcn_lite_over_lstm_model_step_geomean",
    }
    assert torch.get_num_threads() == original_threads
