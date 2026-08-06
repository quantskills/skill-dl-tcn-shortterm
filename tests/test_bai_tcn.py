from __future__ import annotations

from typing import Any, cast

import pytest
import numpy as np
import pandas as pd
import torch
from torch import nn

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.tcn import (
    BaiTCN,
    receptive_field,
    run_bai_tcn,
    validate_receptive_field,
)


def test_bai_tcn_is_causal_and_covers_the_default_window() -> None:
    assert receptive_field(kernel_size=3, dilations=[1, 2, 4, 8, 16, 32, 64]) == 509
    validate_receptive_field(
        input_steps=480, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32, 64]
    )
    with pytest.raises(
        ContractError, match="receptive field 509 is smaller than input window 510"
    ):
        validate_receptive_field(
            input_steps=510, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32, 64]
        )

    torch.manual_seed(7)
    model = BaiTCN(
        feature_count=3,
        channels=4,
        kernel_size=3,
        dilations=[1, 2],
        dropout=0.0,
    ).eval()
    first = torch.randn(2, 3, 12)
    second = first.clone()
    second[:, :, 6:] = torch.randn_like(second[:, :, 6:])

    with torch.no_grad():
        first_sequence = model.encode_sequence(first)
        second_sequence = model.encode_sequence(second)
        scores = model(first)

    torch.testing.assert_close(first_sequence[:, :, :6], second_sequence[:, :, :6])
    assert scores.shape == (2, 4)
    assert not any(isinstance(module, nn.BatchNorm1d) for module in model.modules())
    convolutions = [
        module for module in model.modules() if isinstance(module, nn.Conv1d)
    ]
    assert any(hasattr(convolution, "parametrizations") for convolution in convolutions)


def test_bai_tcn_enters_the_shared_training_and_validation_flow() -> None:
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

    result = run_bai_tcn(
        np.stack(feature_rows),
        window_index,
        labels,
        split,
        seed=7,
        channels=4,
        kernel_size=3,
        dilations=[1, 2],
        dropout=0.0,
        epochs=1,
        batch_size=4,
    )

    assert result.predictions["model"].unique().tolist() == ["bai-tcn"]
    assert set(result.predictions["horizon"]) == {1, 2, 3, 5}
    assert np.isfinite(result.predictions["score"]).all()
    assert result.training_metadata.loc[0, "receptive_field"] == 13
    assert int(cast(Any, result.training_metadata.loc[0, "parameter_count"])) > 0
