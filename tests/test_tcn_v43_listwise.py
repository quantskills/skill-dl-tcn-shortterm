from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.tuning import TCNTuningTrial, run_tcn_validation_sweep
from skill_dl_tcn_shortterm.v9_objective import teacher_listwise_components
from skill_dl_tcn_shortterm.v43_validation import (
    decide_listwise_consensus_seed7_gate,
)


def test_teacher_listwise_component_prefers_teacher_ordering() -> None:
    teacher = torch.tensor(
        [
            [-1.0, -1.0, -1.0, -1.0],
            [-0.3, -0.3, -0.3, -0.3],
            [0.3, 0.3, 0.3, 0.3],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    true_target = teacher.clone()
    mask = torch.ones_like(teacher, dtype=torch.bool)
    aligned = teacher.clone().requires_grad_(True)
    reversed_score = torch.flip(teacher, dims=(0,)).requires_grad_(True)

    aligned_loss = teacher_listwise_components(
        aligned,
        true_target,
        teacher,
        mask,
        ["2025-01-02"] * 4,
        temperature=0.1,
    )
    reversed_loss = teacher_listwise_components(
        reversed_score,
        true_target,
        teacher,
        mask,
        ["2025-01-02"] * 4,
        temperature=0.1,
    )

    assert aligned_loss.group_count == 4
    assert aligned_loss.valid_label_count == 16
    assert aligned_loss.teacher_listwise < reversed_loss.teacher_listwise


def _tiny_inputs() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(43)
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
    return features, index, labels, split


def _trial() -> TCNTuningTrial:
    return TCNTuningTrial(
        trial_id="teacher-listwise",
        model_kind="dynamic_horizon_skip",
        channels=3,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=6,
        strategy="teacher_listwise",
        padding_mode="chomp",
        dynamic_skip_hidden=2,
        dynamic_skip_scale=1.0,
        teacher_listwise_gradient_ratio=0.25,
        teacher_listwise_temperature=0.1,
    )


def test_teacher_listwise_sweep_enforces_train_scope_and_gradient_ratio() -> None:
    features, index, labels, split = _tiny_inputs()
    teacher = np.full((24, 4), np.nan, dtype="float32")
    teacher[:12] = np.tile(np.linspace(-1.0, 1.0, 6), 2)[:, None]

    result = run_tcn_validation_sweep(
        features,
        index,
        labels,
        split,
        trials=[_trial()],
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0,
        torch_threads=1,
        training_teacher_targets={0: teacher},
    )

    row = result.leaderboard.iloc[0]
    assert bool(row["training_teacher_target"]) is True
    assert row["batching_identity"] == "date-grouped"
    assert row["median_teacher_gradient_ratio"] == pytest.approx(0.25)
    assert "gradient-normalized-teacher-listwise" in row["loss_identity"]

    leaked = teacher.copy()
    leaked[12] = 0.0
    with pytest.raises(ContractError, match="non-train position"):
        run_tcn_validation_sweep(
            features,
            index,
            labels,
            split,
            trials=[_trial()],
            seed=7,
            max_epochs=2,
            patience=1,
            min_delta=0.0,
            torch_threads=1,
            training_teacher_targets={0: leaked},
        )


def test_v43_gate_rejects_local_gain_and_accepts_holistic_gain() -> None:
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
    admitted = decide_listwise_consensus_seed7_gate(
        control,
        pointwise,
        bootstrap,
        folds,
        horizons,
        teacher_fidelity_delta=0.003,
        median_teacher_gradient_ratio=0.25,
        model_step_retention=0.8,
        complete_cycle_retention=0.8,
        implied_tcn_lstm_model_step_ratio=3.8,
        inference_forward_passes=1,
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
    rejected = decide_listwise_consensus_seed7_gate(
        control,
        local,
        bootstrap,
        folds,
        horizons,
        teacher_fidelity_delta=0.003,
        median_teacher_gradient_ratio=0.25,
        model_step_retention=0.8,
        complete_cycle_retention=0.8,
        implied_tcn_lstm_model_step_ratio=3.8,
        inference_forward_passes=1,
    )

    assert admitted.admitted is True
    assert rejected.admitted is False
    assert "pointwise_pareto_breadth_below_gate" in rejected.blockers
