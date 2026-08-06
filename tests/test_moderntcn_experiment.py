from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from skill_dl_tcn_shortterm.moderntcn import (
    ModernTCN,
    modern_receptive_field,
    run_moderntcn_experiment,
)


def test_moderntcn_style_block_is_causal_and_has_four_horizon_output() -> None:
    assert (
        modern_receptive_field(
            patch_size=4, patch_stride=2, large_kernel_size=7, block_count=2
        )
        == 28
    )
    torch.manual_seed(11)
    model = ModernTCN(
        feature_count=3,
        d_model=4,
        ffn_ratio=2,
        patch_size=4,
        patch_stride=2,
        large_kernel_size=7,
        small_kernel_size=3,
        block_count=2,
        dropout=0.0,
    ).eval()
    first = torch.randn(2, 3, 16)
    second = first.clone()
    second[:, :, 12:] = torch.randn_like(second[:, :, 12:])

    with torch.no_grad():
        first_sequence = model.encode_sequence(first)
        second_sequence = model.encode_sequence(second)
        scores = model(first)

    torch.testing.assert_close(first_sequence[..., :5], second_sequence[..., :5])
    assert scores.shape == (2, 4)


def test_moderntcn_uses_only_train_validation_and_records_post_mvp_protocol() -> None:
    rng = np.random.default_rng(19)
    index = pd.DataFrame(
        {
            "sample_position": range(15),
            "sample_id": [f"s{i}" for i in range(15)],
            "instrument_id": [f"I{i % 3}" for i in range(15)],
            "signal_date": [f"2024-01-{2 + i // 3:02d}" for i in range(15)],
        }
    )
    features = rng.normal(size=(15, 3, 16)).astype("float32")
    labels = pd.DataFrame(
        [
            {
                "sample_id": sample.sample_id,
                "instrument_id": sample.instrument_id,
                "signal_date": sample.signal_date,
                "horizon": horizon,
                "rank_target": float(rng.uniform(-1, 1)),
                "valid": True,
            }
            for sample in index.itertuples(index=False)
            for horizon in [1, 2, 3, 5]
        ]
    )
    split = index.copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 9 + ["validation"] * 3 + ["test"] * 3
    split["sealed"] = [False] * 12 + [True] * 3

    result = run_moderntcn_experiment(
        features,
        index,
        labels,
        split,
        seed=13,
        d_model=4,
        ffn_ratio=2,
        patch_size=4,
        patch_stride=2,
        large_kernel_size=7,
        small_kernel_size=3,
        block_count=2,
        dropout=0.0,
        epochs=1,
        batch_size=3,
    )

    assert result.predictions["model"].unique().tolist() == ["moderntcn-post-mvp"]
    assert set(result.predictions["horizon"]) == {1, 2, 3, 5}
    assert set(result.predictions["stage"]) == {"validation"}
    assert not set(result.predictions["sample_id"]) & {"s12", "s13", "s14"}
    metadata = result.training_metadata.iloc[0]
    assert metadata["experiment_class"] == "post_mvp_non_blocking"
    assert not bool(metadata["sealed_test_access"])
    assert metadata["stopping_rule"] == "fixed_epochs_validation_only"
    assert metadata["receptive_field"] == 28
    assert metadata["parameter_count"] > 0
