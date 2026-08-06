from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.checkpoint_selection import (
    select_constrained_tail_checkpoints,
)
from skill_dl_tcn_shortterm.tuning import TCNTuningTrial, run_tcn_validation_sweep
from skill_dl_tcn_shortterm.v9_representation import DynamicHorizonSkipTCN


def _epoch_metrics() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    values = {
        (7, 0): [
            (0, 0.1000, 0.110, 0.560, 0.60),
            (1, 0.0990, 0.125, 0.570, 0.61),
            (2, 0.0970, 0.200, 0.650, 0.62),
        ],
        (17, 0): [
            (0, 0.1000, 0.110, 0.560, 0.60),
            (1, 0.1010, 0.115, 0.565, 0.61),
            (2, 0.1005, 0.125, 0.575, 0.59),
        ],
    }
    for (seed, fold), epochs in values.items():
        for epoch, rankic, precision, ndcg, turnover in epochs:
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "epoch": epoch,
                    "rankic": rankic,
                    "top_precision": precision,
                    "ndcg_at_top": ndcg,
                    "top_return": 0.003,
                    "top_turnover": turnover,
                }
            )
    return pd.DataFrame(rows)


def test_v35_selection_uses_tail_score_only_inside_rankic_feasible_set() -> None:
    selected = select_constrained_tail_checkpoints(
        _epoch_metrics(),
        expected_epochs=(0, 1, 2),
        rankic_tolerance=0.002,
    ).set_index(["seed", "fold"])

    assert int(cast(Any, selected.loc[(7, 0), "control_epoch"])) == 0
    assert int(cast(Any, selected.loc[(7, 0), "candidate_epoch"])) == 1
    assert int(cast(Any, selected.loc[(17, 0), "control_epoch"])) == 1
    assert int(cast(Any, selected.loc[(17, 0), "candidate_epoch"])) == 2
    assert selected["candidate_rankic_feasible"].all()
    assert selected["selection_changed"].all()
    assert float(
        cast(Any, selected.loc[(7, 0), "candidate_tail_selection_score"])
    ) == pytest.approx(0.5 * (0.125 + 0.570))


def test_v35_selection_rejects_missing_epoch_and_duplicate_unit_epoch() -> None:
    missing = _epoch_metrics().loc[lambda frame: ~(
        frame["seed"].eq(7) & frame["epoch"].eq(2)
    )]
    with pytest.raises(ContractError, match="epoch coverage"):
        select_constrained_tail_checkpoints(
            missing, expected_epochs=(0, 1, 2), rankic_tolerance=0.002
        )

    duplicate = pd.concat([_epoch_metrics(), _epoch_metrics().iloc[[0]]])
    with pytest.raises(ContractError, match="duplicate"):
        select_constrained_tail_checkpoints(
            duplicate, expected_epochs=(0, 1, 2), rankic_tolerance=0.002
        )


def test_v35_selection_rejects_nonfinite_metrics_and_unregistered_tolerance() -> None:
    nonfinite = _epoch_metrics()
    nonfinite.loc[0, "ndcg_at_top"] = np.nan
    with pytest.raises(ContractError, match="finite"):
        select_constrained_tail_checkpoints(
            nonfinite, expected_epochs=(0, 1, 2), rankic_tolerance=0.002
        )
    with pytest.raises(ContractError, match="tolerance"):
        select_constrained_tail_checkpoints(
            _epoch_metrics(), expected_epochs=(0, 1, 2), rankic_tolerance=-0.1
        )


def test_v35_tuning_captures_epoch_zero_through_fixed_budget_without_stopping() -> None:
    rng = np.random.default_rng(35)
    features = rng.normal(size=(18, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(18),
            "sample_id": [f"trajectory-{value}" for value in range(18)],
            "signal_date": [
                f"2025-09-{1 + value // 6:02d}" for value in range(18)
            ],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"trajectory-{sample}",
                "signal_date": index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": float((sample % 6) / 5 * 2 - 1),
                "valid": True,
            }
            for sample in range(18)
            for horizon in (1, 2, 3, 5)
        ]
    )
    split = index[["sample_position"]].copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 12 + ["validation"] * 6
    trial = TCNTuningTrial(
        trial_id="trajectory",
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=8,
        model_kind="dynamic_horizon_skip",
        strategy="top_tail",
        padding_mode="chomp",
        dynamic_skip_hidden=2,
        dynamic_skip_scale=1.0,
        dynamic_skip_shape_residual=True,
        dynamic_skip_shape_residual_scale=0.25,
        dynamic_skip_frozen_parent=True,
        top_tail_weight=0.05,
        top_tail_fraction=0.1,
        top_tail_temperature=0.1,
    )
    parent = DynamicHorizonSkipTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        dropout=0.0,
        padding_mode="chomp",
        dynamic_skip_hidden=2,
        dynamic_skip_scale=1.0,
    )
    parent_state = {
        name: tensor.detach().clone() for name, tensor in parent.state_dict().items()
    }

    result = run_tcn_validation_sweep(
        features,
        index,
        labels,
        split,
        trials=(trial,),
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0005,
        checkpoint_min_delta=0.0,
        torch_threads=1,
        frozen_parent_states={"trajectory-fold-0": parent_state},
        capture_epoch_states=True,
        disable_early_stopping=True,
    )

    assert set(result.epoch_states) == {
        "trajectory-fold-0-epoch-0",
        "trajectory-fold-0-epoch-1",
        "trajectory-fold-0-epoch-2",
    }
    assert int(result.leaderboard.iloc[0]["completed_epochs"]) == 2
    assert result.leaderboard.iloc[0]["stopping_reason"] == "max_epochs"
    for state in result.epoch_states.values():
        assert state
        assert all(isinstance(value, torch.Tensor) for value in state.values())
