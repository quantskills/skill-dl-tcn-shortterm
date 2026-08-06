from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from skill_dl_tcn_shortterm.stability_ema import (
    ParameterEMA,
    state_dict_max_abs_error,
)
from skill_dl_tcn_shortterm.v41_validation import decide_ema_holistic_gate
from skill_dl_tcn_shortterm.tuning import TCNTuningTrial, run_tcn_validation_sweep


class _BufferedLinear(nn.Module):
    counter: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_parameter_ema_first_step_copy_update_swap_and_restore() -> None:
    model = _BufferedLinear()
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
    ema = ParameterEMA(decay=0.99)

    ema.update(model)
    with torch.no_grad():
        model.linear.weight.fill_(3.0)
    ema.update(model)

    assert ema.update_count == 2
    assert model.linear.weight.item() == pytest.approx(3.0)
    with ema.average_parameters(model):
        assert model.linear.weight.item() == pytest.approx(1.02)
        assert model.counter.item() == 7
    assert model.linear.weight.item() == pytest.approx(3.0)
    averaged = ema.averaged_state_dict(model)
    assert averaged["linear.weight"].item() == pytest.approx(1.02)
    assert averaged["counter"].item() == 7


def test_parameter_ema_does_not_change_raw_training_trajectory() -> None:
    torch.manual_seed(41)
    control = nn.Linear(3, 1)
    candidate = copy.deepcopy(control)
    control_optimizer = torch.optim.SGD(control.parameters(), lr=0.1)
    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.1)
    ema = ParameterEMA(decay=0.99)
    inputs = torch.tensor([[1.0, -2.0, 0.5], [0.2, 0.3, -0.7]])
    targets = torch.tensor([[0.4], [-0.1]])

    for _ in range(4):
        for model, optimizer in (
            (control, control_optimizer),
            (candidate, candidate_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(inputs), targets).backward()
            optimizer.step()
        ema.update(candidate)

    assert state_dict_max_abs_error(control.state_dict(), candidate.state_dict()) == 0


def test_tuning_ema_checkpoint_keeps_raw_epoch_trajectory_exact() -> None:
    rng = np.random.default_rng(41)
    features = rng.normal(size=(24, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(24),
            "sample_id": [f"s{value}" for value in range(24)],
            "signal_date": [f"2025-01-{2 + value // 6:02d}" for value in range(24)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "signal_date": index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": float((sample % 6) / 5 * 2 - 1),
                "valid": True,
            }
            for sample in range(24)
            for horizon in (1, 2, 3, 5)
        ]
    )
    split = index[["sample_position"]].copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 12 + ["validation"] * 12
    shared = {
        "channels": 3,
        "kernel_size": 2,
        "dilations": (1, 2, 4, 8),
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 6,
    }
    result = run_tcn_validation_sweep(
        features,
        index,
        labels,
        split,
        trials=[
            TCNTuningTrial(trial_id="raw", **shared),  # type: ignore[arg-type]
            TCNTuningTrial(
                trial_id="ema", ema_decay=0.99, **shared  # type: ignore[arg-type]
            ),
        ],
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0,
        torch_threads=1,
        capture_epoch_states=True,
        disable_early_stopping=True,
    )

    for epoch in range(3):
        raw = result.epoch_states[f"raw-fold-0-epoch-{epoch}"]
        ema_raw = result.epoch_states[f"ema-fold-0-epoch-{epoch}"]
        assert state_dict_max_abs_error(raw, ema_raw) == 0.0
    rows = result.leaderboard.set_index("trial_id")
    assert rows.loc["raw", "checkpoint_parameter_source"] == "raw"
    assert rows.loc["ema", "checkpoint_parameter_source"] == "ema"
    assert rows.loc["raw", "ema_update_count"] == 0
    assert int(cast(Any, rows.loc["ema", "ema_update_count"])) > 0


def test_holistic_gate_rejects_local_win_and_accepts_broad_win() -> None:
    fold_deltas = pd.DataFrame(
        {"fold": range(5), "rankic_delta": [0.004, 0.003, 0.002, -0.001, 0.005]}
    )
    horizon_deltas = pd.DataFrame(
        {
            "horizon": [1, 2, 3, 5],
            "rankic_delta": [0.001, 0.003, 0.004, 0.002],
        }
    )
    broad = {
        "mean_rankic_delta": 0.003,
        "mean_pearson_ic_delta": 0.002,
        "mean_top_return_delta": 0.0002,
        "mean_top_precision_delta": 0.001,
        "mean_ndcg_at_top_delta": 0.002,
        "mean_quantile_monotonicity_delta": -0.001,
    }
    bootstrap = pd.DataFrame(
        [{"metric": "rankic", "bootstrap_ci_low": -0.0005}]
    )

    admitted = decide_ema_holistic_gate(
        broad,
        bootstrap,
        fold_deltas,
        horizon_deltas,
        raw_state_drift_max=0.0,
        model_step_retention=0.96,
        complete_cycle_retention=0.90,
        implied_tcn_lstm_model_step_ratio=4.4,
    )
    local_only = dict(broad)
    local_only.update(
        {
            "mean_top_return_delta": -0.001,
            "mean_top_precision_delta": -0.01,
            "mean_ndcg_at_top_delta": -0.02,
            "mean_quantile_monotonicity_delta": -0.03,
        }
    )
    rejected = decide_ema_holistic_gate(
        local_only,
        bootstrap,
        fold_deltas,
        horizon_deltas,
        raw_state_drift_max=0.0,
        model_step_retention=0.96,
        complete_cycle_retention=0.90,
        implied_tcn_lstm_model_step_ratio=4.4,
    )

    assert admitted.admitted is True
    assert admitted.status == "ema_seed7_holistic_admitted_v41"
    assert rejected.admitted is False
    assert "broad_metric_count_below_gate" in rejected.blockers
    assert "top_return_delta_below_gate" in rejected.blockers
