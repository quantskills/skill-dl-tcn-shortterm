from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import TCNTuningTrial, run_tcn_validation_sweep
from skill_dl_tcn_shortterm.v45_validation import (
    decide_distillation_annealing_seed7_gate,
)


def _inputs() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(45)
    features = rng.normal(size=(24, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(24),
            "sample_id": [f"s{value}" for value in range(24)],
            "signal_date": [f"2025-03-{2 + value // 6:02d}" for value in range(24)],
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


def _trial(**changes: object) -> TCNTuningTrial:
    values: dict[str, object] = {
        "trial_id": "annealed",
        "channels": 3,
        "kernel_size": 2,
        "dilations": (1, 2, 4, 8),
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 6,
        "teacher_blend_start_weight": 0.25,
        "teacher_blend_end_weight": 0.0,
    }
    values.update(changes)
    return TCNTuningTrial(**values)  # type: ignore[arg-type]


def test_linear_teacher_blend_schedule_is_exact_and_reaches_true_target() -> None:
    features, index, labels, split = _inputs()
    teacher = np.full((24, 4), np.nan, dtype="float32")
    teacher[:12] = -0.5
    result = run_tcn_validation_sweep(
        features,
        index,
        labels,
        split,
        trials=[_trial()],
        seed=7,
        max_epochs=3,
        patience=2,
        min_delta=0.0,
        torch_threads=1,
        training_teacher_targets={0: teacher},
        disable_early_stopping=True,
        capture_epoch_states=True,
    )

    history = result.epoch_history.sort_values("epoch")
    np.testing.assert_allclose(
        history["teacher_blend_weight"].to_numpy(dtype="float64"),
        [0.25, 0.125, 0.0],
    )
    row = result.leaderboard.iloc[0]
    assert row["teacher_blend_start_weight"] == pytest.approx(0.25)
    assert row["teacher_blend_end_weight"] == pytest.approx(0.0)
    assert bool(row["training_target_schedule"]) is True
    assert "linear-teacher-blend" in str(row["loss_identity"])


def test_teacher_blend_schedule_parser_and_fail_closed_combinations() -> None:
    parsed = parse_real_tcn_trials(
        [
            {
                "trial_id": "annealed",
                "channels": 3,
                "kernel_size": 2,
                "dilations": [1, 2, 4, 8],
                "dropout": 0.0,
                "learning_rate": 0.003,
                "batch_size": 6,
                "teacher_blend_start_weight": 0.25,
                "teacher_blend_end_weight": 0.0,
            }
        ]
    )[0]
    assert parsed.teacher_blend_start_weight == pytest.approx(0.25)
    assert parsed.teacher_blend_end_weight == pytest.approx(0.0)

    features, index, labels, split = _inputs()
    teacher = np.full((24, 4), np.nan, dtype="float32")
    teacher[:12] = 0.0
    with pytest.raises(ContractError, match="static target overrides"):
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
            training_teacher_targets={0: teacher},
            training_target_overrides={0: np.zeros((24, 4), dtype="float32")},
        )
    with pytest.raises(ContractError, match="blend weights"):
        run_tcn_validation_sweep(
            features,
            index,
            labels,
            split,
            trials=[_trial(teacher_blend_end_weight=0.25)],
            seed=7,
            max_epochs=2,
            patience=1,
            min_delta=0.0,
            torch_threads=1,
            training_teacher_targets={0: teacher},
        )


def test_v45_gate_rejects_local_gain_and_accepts_global_gain() -> None:
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
        "schedule_max_abs_error": 0.0,
        "terminal_teacher_weight": 0.0,
        "validation_teacher_cells_exposed": 0,
        "model_step_retention": 0.98,
        "complete_cycle_retention": 0.95,
        "implied_tcn_lstm_model_step_ratio": 4.5,
        "inference_forward_passes": 1,
    }
    admitted = decide_distillation_annealing_seed7_gate(
        control, pointwise, bootstrap, folds, horizons, **common
    )
    rejected = decide_distillation_annealing_seed7_gate(
        control,
        {**pointwise, "mean_rankic_delta": -0.01, "mean_top_precision_delta": -0.01},
        bootstrap,
        folds,
        horizons,
        **common,
    )

    assert admitted.admitted is True
    assert admitted.status == "linear_distillation_annealing_seed7_admitted_v45"
    assert rejected.admitted is False
    assert "pointwise_rankic_delta_below_gate" in rejected.blockers
