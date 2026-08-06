from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.stability_ema import (
    EpochParameterAverage,
    state_dict_max_abs_error,
)
from skill_dl_tcn_shortterm.tuning import TCNTuningTrial, run_tcn_validation_sweep
from skill_dl_tcn_shortterm.v44_validation import (
    decide_trajectory_average_seed7_gate,
)


def test_epoch_parameter_average_is_exact_online_mean_and_restores_raw_model() -> None:
    model = nn.Linear(2, 1, bias=True)
    average = EpochParameterAverage()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 3.0]]))
        model.bias.fill_(2.0)
    average.update(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[5.0, 7.0]]))
        model.bias.fill_(6.0)
    average.update(model)

    assert average.update_count == 2
    with average.average_parameters(model):
        torch.testing.assert_close(model.weight, torch.tensor([[3.0, 5.0]]))
        torch.testing.assert_close(model.bias, torch.tensor([4.0]))
    torch.testing.assert_close(model.weight, torch.tensor([[5.0, 7.0]]))
    torch.testing.assert_close(model.bias, torch.tensor([6.0]))


def _tiny_validation_inputs() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(44)
    features = rng.normal(size=(24, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(24),
            "sample_id": [f"s{value}" for value in range(24)],
            "signal_date": [f"2025-02-{2 + value // 6:02d}" for value in range(24)],
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
    return features, index, labels, split


def test_tuning_final_epoch_average_preserves_raw_trajectory_and_ignores_peak_selection() -> None:
    features, index, labels, split = _tiny_validation_inputs()
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
            TCNTuningTrial(
                trial_id="raw", **shared  # type: ignore[arg-type]
            ),
            TCNTuningTrial(
                trial_id="average",
                epoch_average_start=1,
                **shared,  # type: ignore[arg-type]
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
        averaged_raw = result.epoch_states[f"average-fold-0-epoch-{epoch}"]
        assert state_dict_max_abs_error(raw, averaged_raw) == 0.0
    final = result.best_states["average-fold-0"]
    epoch_1 = result.epoch_states["average-fold-0-epoch-1"]
    epoch_2 = result.epoch_states["average-fold-0-epoch-2"]
    for name, tensor in final.items():
        if tensor.is_floating_point():
            expected = (epoch_1[name] + epoch_2[name]) / 2.0
            torch.testing.assert_close(tensor, expected, atol=1e-6, rtol=1e-6)
    row = result.leaderboard.set_index("trial_id").loc["average"]
    assert row["best_epoch"] == 2
    assert row["epoch_average_start"] == 1
    assert row["epoch_average_update_count"] == 2
    assert row["checkpoint_parameter_source"] == "epoch_uniform_average_final"


def test_epoch_average_contract_is_parsed_and_rejects_ema_or_invalid_start() -> None:
    trial = parse_real_tcn_trials(
        [
            {
                "trial_id": "average",
                "channels": 3,
                "kernel_size": 2,
                "dilations": [1, 2, 4, 8],
                "dropout": 0.0,
                "learning_rate": 0.003,
                "batch_size": 6,
                "epoch_average_start": 2,
            }
        ]
    )[0]
    assert trial.epoch_average_start == 2

    features, index, labels, split = _tiny_validation_inputs()
    with pytest.raises(ContractError, match="mutually exclusive"):
        run_tcn_validation_sweep(
            features,
            index,
            labels,
            split,
            trials=[
                TCNTuningTrial(
                    trial_id="invalid",
                    channels=3,
                    kernel_size=2,
                    dilations=(1, 2, 4, 8),
                    dropout=0.0,
                    learning_rate=0.003,
                    batch_size=6,
                    ema_decay=0.99,
                    epoch_average_start=1,
                )
            ],
            seed=7,
            max_epochs=2,
            patience=1,
            min_delta=0.0,
            torch_threads=1,
        )
    with pytest.raises(ContractError, match="average start"):
        run_tcn_validation_sweep(
            features,
            index,
            labels,
            split,
            trials=[
                TCNTuningTrial(
                    trial_id="late",
                    channels=3,
                    kernel_size=2,
                    dilations=(1, 2, 4, 8),
                    dropout=0.0,
                    learning_rate=0.003,
                    batch_size=6,
                    epoch_average_start=3,
                )
            ],
            seed=7,
            max_epochs=2,
            patience=1,
            min_delta=0.0,
            torch_threads=1,
        )


def test_v44_gate_rejects_local_or_unstable_result_and_accepts_global_result() -> None:
    control = {
        "mean_rankic_delta": 0.004,
        "mean_pearson_ic_delta": 0.004,
        "mean_top_return_delta": 0.0002,
        "mean_top_precision_delta": 0.001,
        "mean_ndcg_at_top_delta": 0.002,
        "mean_quantile_monotonicity_delta": 0.002,
    }
    pointwise = {
        "mean_rankic_delta": 0.001,
        "mean_pearson_ic_delta": 0.001,
        "mean_top_return_delta": 0.0001,
        "mean_top_precision_delta": 0.001,
        "mean_ndcg_at_top_delta": -0.0005,
        "mean_quantile_monotonicity_delta": -0.0005,
    }
    bootstrap = pd.DataFrame(
        [{"metric": "rankic", "bootstrap_ci_low": 0.001}]
    )
    folds = pd.DataFrame(
        {"fold": range(5), "rankic_delta": [0.004, 0.003, 0.005, 0.002, 0.004]}
    )
    horizons = pd.DataFrame(
        {"horizon": [1, 2, 3, 5], "rankic_delta": [0.004, 0.003, 0.005, 0.002]}
    )
    common: dict[str, Any] = {
        "teacher_fidelity_delta": 0.0,
        "validation_volatility_ratio": 0.8,
        "raw_state_drift_max": 0.0,
        "arithmetic_mean_max_error": 1e-7,
        "average_parameter_distance": 0.1,
        "average_update_count_min": 7,
        "average_update_count_max": 7,
        "model_step_retention": 0.98,
        "complete_cycle_retention": 0.95,
        "implied_tcn_lstm_model_step_ratio": 4.5,
        "inference_forward_passes": 1,
    }
    admitted = decide_trajectory_average_seed7_gate(
        control, pointwise, bootstrap, folds, horizons, **common
    )
    local = dict(pointwise)
    local.update(
        {
            "mean_rankic_delta": -0.01,
            "mean_pearson_ic_delta": -0.01,
            "mean_ndcg_at_top_delta": -0.01,
            "mean_quantile_monotonicity_delta": -0.01,
        }
    )
    rejected = decide_trajectory_average_seed7_gate(
        control,
        local,
        bootstrap,
        folds,
        horizons,
        **{**common, "validation_volatility_ratio": 1.1},
    )

    assert admitted.admitted is True
    assert admitted.status == "trajectory_average_seed7_holistic_admitted_v44"
    assert rejected.admitted is False
    assert "pointwise_pareto_breadth_below_gate" in rejected.blockers
    assert "validation_volatility_ratio_above_gate" in rejected.blockers
