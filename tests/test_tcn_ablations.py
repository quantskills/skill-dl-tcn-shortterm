from __future__ import annotations

from typing import Any, cast

import pandas as pd
import numpy as np
import pytest
import torch

from skill_dl_tcn_shortterm.tcn_lite import (
    CausalLiteBlock,
    TCNLite,
    compare_shared_and_single_horizon,
    lite_receptive_field,
    run_tcn_ablations,
)


def test_tcn_lite_is_causal_and_negative_transfer_is_reported() -> None:
    assert (
        lite_receptive_field(kernel_size=3, dilations=[1, 2, 4, 8, 16, 32, 64, 128])
        == 511
    )
    torch.manual_seed(7)
    model = TCNLite(
        feature_count=3, channels=4, kernel_size=3, dilations=[1, 2, 4], dropout=0.0
    ).eval()
    first = torch.randn(2, 3, 15)
    second = first.clone()
    second[:, :, 8:] = torch.randn_like(second[:, :, 8:])
    with torch.no_grad():
        first_sequence = model.encode_sequence(first)
        second_sequence = model.encode_sequence(second)
        scores = model(first)
    torch.testing.assert_close(first_sequence[:, :, :8], second_sequence[:, :, :8])
    assert scores.shape == (2, 4)

    shared = pd.DataFrame({"horizon": [1, 2, 3, 5], "rankic": [0.10, 0.20, 0.30, 0.40]})
    single = pd.DataFrame({"horizon": [1, 2, 3, 5], "rankic": [0.20, 0.15, 0.30, 0.45]})
    comparison = compare_shared_and_single_horizon(shared, single).set_index("horizon")
    assert comparison.loc[1, "conclusion"] == "negative-transfer"
    assert comparison.loc[2, "conclusion"] == "no-negative-transfer"
    assert comparison.loc[3, "conclusion"] == "no-difference"


def test_tcn_lite_head_dropout_is_cheap_compatible_and_validated() -> None:
    torch.manual_seed(13)
    control = TCNLite(
        feature_count=3,
        channels=4,
        kernel_size=3,
        dilations=[1, 2, 4],
        dropout=0.0,
    )
    candidate = TCNLite(
        feature_count=3,
        channels=4,
        kernel_size=3,
        dilations=[1, 2, 4],
        dropout=0.0,
        head_dropout=0.1,
    )
    candidate.load_state_dict(control.state_dict(), strict=True)
    inputs = torch.randn(5, 3, 15)

    control.eval()
    candidate.eval()
    with torch.no_grad():
        torch.testing.assert_close(control(inputs), candidate(inputs))

    observed_shapes: list[tuple[int, ...]] = []
    hook = candidate.head_dropout.register_forward_pre_hook(
        lambda _module, values: observed_shapes.append(tuple(values[0].shape))
    )
    candidate.train()
    candidate(inputs)
    hook.remove()
    assert observed_shapes == [(5, 4)]

    with pytest.raises(ValueError, match="head dropout"):
        TCNLite(
            feature_count=3,
            channels=4,
            kernel_size=3,
            dilations=[1, 2, 4],
            dropout=0.0,
            head_dropout=1.0,
        )


def test_tcn_lite_channel_dropout_shares_one_mask_across_time() -> None:
    model = TCNLite(
        feature_count=3,
        channels=4,
        kernel_size=3,
        dilations=[1, 2, 4],
        dropout=0.5,
        dropout_kind="channel",
    )
    block = next(
        module for module in model.modules() if isinstance(module, CausalLiteBlock)
    )
    block.dropout.train()
    torch.manual_seed(17)
    dropped = block.dropout(torch.ones(8, 3, 15))
    assert dropped.eq(dropped[:, :, :1]).all()

    with pytest.raises(ValueError, match="dropout kind"):
        TCNLite(
            feature_count=3,
            channels=4,
            kernel_size=3,
            dilations=[1, 2, 4],
            dropout=0.1,
            dropout_kind=cast(Any, "unknown"),
        )


def test_tcn_lite_and_single_horizon_models_use_the_same_validation_samples() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    index_rows = []
    label_rows = []
    feature_rows = []
    for position, (date, target) in enumerate(
        (date, target) for date in dates for target in (-1.0, 0.0, 1.0)
    ):
        instrument = ["A", "B", "C"][position % 3]
        sample_id = f"{date}-{instrument}"
        index_rows.append(
            {
                "sample_position": position,
                "sample_id": sample_id,
                "instrument_id": instrument,
                "signal_date": date,
            }
        )
        feature_rows.append(np.full((2, 12), target, dtype="float32"))
        for horizon in [1, 2, 3, 5]:
            label_rows.append(
                {
                    "sample_id": sample_id,
                    "instrument_id": instrument,
                    "signal_date": date,
                    "horizon": horizon,
                    "rank_target": target,
                    "valid": True,
                }
            )
    window_index = pd.DataFrame(index_rows)
    labels = pd.DataFrame(label_rows)
    split = window_index.copy()
    split["fold"] = 0
    split["stage"] = [
        "train" if date <= "2024-01-03" else "validation"
        for date in split["signal_date"]
    ]
    split["sealed"] = False
    validation_ids = set(
        split.loc[split["stage"] == "validation", "sample_id"].astype(str)
    )
    shared_predictions = labels.loc[labels["sample_id"].isin(validation_ids)].rename(
        columns={"rank_target": "target"}
    )
    shared_predictions = shared_predictions.assign(
        model="bai-tcn",
        fold=0,
        score=shared_predictions["target"].astype(float),
    )

    result, comparison = run_tcn_ablations(
        np.stack(feature_rows),
        window_index,
        labels,
        split,
        pd.DataFrame({"horizon": [1, 2, 3, 5], "rankic": [0.0, 0.0, 0.0, 0.0]}),
        shared_predictions=shared_predictions,
        seed=7,
        channels=2,
        kernel_size=3,
        lite_dilations=[1, 2, 4],
        bai_dilations=[1, 2],
        dropout=0.0,
        epochs=1,
        batch_size=6,
    )

    assert set(result.predictions["model"]) == {
        "tcn-lite",
        "bai-tcn-1d",
        "bai-tcn-2d",
        "bai-tcn-3d",
        "bai-tcn-5d",
    }
    assert set(comparison["horizon"]) == {1, 2, 3, 5}
    assert set(comparison["comparison_type"]) == {
        "single-vs-shared",
        "lite-vs-shared",
    }
    assert len(comparison) == 8
    assert comparison["paired_date_count"].eq(2).all()
    assert comparison["delta_ci_low"].notna().all()
    assert comparison["delta_ci_high"].notna().all()
