from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.optimized_tcn import (
    V40_PORTABLE_MODEL,
    build_optimized_tcn_model,
    optimized_tcn_trial,
    resolve_optimized_tcn_profile,
    run_optimized_tcn,
)
from skill_dl_tcn_shortterm.performance import benchmark_sequence_models


def _protocol(tmp_path: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(47)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, rng.normal(size=(16, 3, 16)).astype("float32"))
    features = np.load(feature_path, mmap_mode="r")
    index = pd.DataFrame(
        {
            "sample_position": range(16),
            "sample_id": [f"s{value}" for value in range(16)],
            "instrument_id": [f"I{value % 4}" for value in range(16)],
            "signal_date": ["2025-01-02"] * 4
            + ["2025-01-03"] * 4
            + ["2025-01-06"] * 4
            + ["2025-01-07"] * 4,
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": row.sample_id,
                "instrument_id": row.instrument_id,
                "signal_date": row.signal_date,
                "horizon": horizon,
                "rank_target": float((position % 4) / 3 * 2 - 1),
                "valid": True,
            }
            for position, row in enumerate(index.itertuples(index=False))
            for horizon in [1, 2, 3, 5]
        ]
    )
    split = index[["sample_position"]].copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 8 + ["validation"] * 4 + ["test"] * 4
    return features, index, labels, split


def test_v40_portable_profile_is_frozen_and_fail_closed() -> None:
    profile = resolve_optimized_tcn_profile()
    trial = optimized_tcn_trial(profile)

    assert profile.model_name == V40_PORTABLE_MODEL
    assert trial.model_kind == "dynamic_horizon_skip"
    assert trial.channels == 16
    assert trial.dilations == (1, 2, 4, 8, 16, 32, 64, 128)
    assert trial.padding_mode == "chomp"
    assert trial.strategy == "smooth_l1"
    assert trial.dynamic_skip_hidden == 4
    assert trial.dynamic_skip_scale == 1.0
    with pytest.raises(ContractError, match="v40-portable"):
        resolve_optimized_tcn_profile("unregistered")


def test_portable_training_excludes_test_and_emits_profile_metadata(
    tmp_path: Path,
) -> None:
    features, index, labels, split = _protocol(tmp_path)
    profile = resolve_optimized_tcn_profile(
        epochs=2, batch_size=4, torch_threads=1
    )

    result = run_optimized_tcn(
        features,
        index,
        labels,
        split,
        seed=7,
        profile=profile,
    )

    assert set(result.predictions["model"]) == {V40_PORTABLE_MODEL}
    assert set(result.predictions["stage"]) == {"validation"}
    assert set(result.predictions["sample_id"]) == {"s8", "s9", "s10", "s11"}
    assert result.training_metadata["profile"].eq("v40-portable").all()
    assert result.training_metadata["torch_threads"].eq(1).all()
    assert result.training_metadata["sealed_test_accessed"].eq(False).all()


def test_performance_uses_portable_factory_and_reports_lstm_ratios(
    tmp_path: Path,
) -> None:
    features, index, labels, split = _protocol(tmp_path)
    profile = resolve_optimized_tcn_profile(
        epochs=1, batch_size=4, torch_threads=1
    )
    model = build_optimized_tcn_model(
        feature_count=3, input_steps=16, profile=profile
    )
    assert model(torch.zeros(2, 3, 16)).shape == (2, 4)

    result = benchmark_sequence_models(
        features,
        index,
        labels,
        split,
        seed=7,
        hidden_size=4,
        tcn_channels=4,
        tcn_kernel_size=2,
        tcn_dilations=(1, 2, 4, 8),
        epochs=1,
        batch_size=4,
        device="cpu",
        torch_threads=1,
        learning_rate=0.003,
        models=("lstm", V40_PORTABLE_MODEL),
    )

    assert set(result.measurements["model"]) == {"lstm", V40_PORTABLE_MODEL}
    ratios = result.environment["observed_tcn_speed_ratios"]
    assert isinstance(ratios, dict)
    assert set(ratios) == {
        "optimized_tcn_over_lstm_geomean",
        "optimized_tcn_over_lstm_model_step_geomean",
    }
    assert result.environment["optimized_tcn_profile"] == "v40-portable"
