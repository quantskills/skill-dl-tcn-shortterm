from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_decoupled_checkpoint_selection_multiseed,
)
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    ValidationSelectionState,
    advance_validation_selection,
    run_tcn_validation_sweep,
)
from skill_dl_tcn_shortterm.v9_representation import DynamicHorizonSkipTCN


def test_checkpoint_selection_can_improve_without_resetting_patience() -> None:
    state = ValidationSelectionState(
        best_score=0.1000,
        patience_anchor_score=0.1000,
        epochs_without_material_improvement=0,
        has_checkpoint=True,
    )
    small = advance_validation_selection(
        state,
        score=0.1003,
        checkpoint_min_delta=0.0,
        patience_min_delta=0.0005,
    )
    assert small.checkpoint_improved is True
    assert small.patience_improved is False
    assert small.state.best_score == 0.1003
    assert small.state.patience_anchor_score == 0.1000
    assert small.state.epochs_without_material_improvement == 1

    material = advance_validation_selection(
        small.state,
        score=0.1006,
        checkpoint_min_delta=0.0,
        patience_min_delta=0.0005,
    )
    assert material.checkpoint_improved is True
    assert material.patience_improved is True
    assert material.state.best_score == 0.1006
    assert material.state.patience_anchor_score == 0.1006
    assert material.state.epochs_without_material_improvement == 0

    lower = advance_validation_selection(
        material.state,
        score=0.1004,
        checkpoint_min_delta=0.0,
        patience_min_delta=0.0005,
    )
    assert lower.checkpoint_improved is False
    assert lower.patience_improved is False
    assert lower.state.best_score == 0.1006
    assert lower.state.epochs_without_material_improvement == 1


def test_equal_thresholds_preserve_historical_selection_behavior() -> None:
    state = ValidationSelectionState(
        best_score=0.1000,
        patience_anchor_score=0.1000,
        epochs_without_material_improvement=0,
        has_checkpoint=True,
    )
    result = advance_validation_selection(
        state,
        score=0.1003,
        checkpoint_min_delta=0.0005,
        patience_min_delta=0.0005,
    )
    assert result.checkpoint_improved is False
    assert result.patience_improved is False
    assert result.state.best_score == 0.1000
    assert result.state.patience_anchor_score == 0.1000
    assert result.state.epochs_without_material_improvement == 1


def test_selection_thresholds_fail_closed() -> None:
    state = ValidationSelectionState(
        best_score=0.1,
        patience_anchor_score=0.1,
        epochs_without_material_improvement=0,
        has_checkpoint=True,
    )
    with pytest.raises(ContractError, match="checkpoint selection deltas"):
        advance_validation_selection(
            state,
            score=0.2,
            checkpoint_min_delta=0.001,
            patience_min_delta=0.0005,
        )
    nonfinite = advance_validation_selection(
        state,
        score=float("nan"),
        checkpoint_min_delta=0.0,
        patience_min_delta=0.0005,
    )
    assert nonfinite.checkpoint_improved is False
    assert nonfinite.patience_improved is False
    assert nonfinite.state.epochs_without_material_improvement == 1


def test_frozen_sweep_publishes_decoupled_selection_and_saves_observed_max() -> None:
    rng = np.random.default_rng(281)
    features = rng.normal(size=(18, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(18),
            "sample_id": [f"select-{value}" for value in range(18)],
            "signal_date": [f"2025-06-{2 + value // 6:02d}" for value in range(18)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"select-{sample}",
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
        trial_id="decoupled-selection",
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=6,
        model_kind="dynamic_horizon_skip",
        padding_mode="chomp",
        dynamic_skip_hidden=2,
        dynamic_skip_scale=1.0,
        dynamic_skip_shape_residual=True,
        dynamic_skip_shape_residual_scale=0.25,
        dynamic_skip_frozen_parent=True,
    )
    torch.manual_seed(282)
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
        max_epochs=3,
        patience=2,
        min_delta=0.0005,
        checkpoint_min_delta=0.0,
        torch_threads=1,
        frozen_parent_states={"decoupled-selection-fold-0": parent_state},
    )
    row = result.leaderboard.iloc[0]
    assert row["checkpoint_min_delta"] == 0.0
    assert row["patience_min_delta"] == 0.0005
    assert row["checkpoint_selection_identity"] == (
        "best-any-strict-improvement+patience-material-0.0005"
    )
    assert {
        "checkpoint_improved",
        "patience_improved",
        "epochs_without_material_improvement",
    } <= set(result.epoch_history.columns)
    assert row["best_mean_daily_rankic"] == pytest.approx(
        result.epoch_history["mean_daily_rankic"].max(), abs=1e-12
    )


def _decision_frames(
    *, trajectory_error: float = 0.0, v27_delta: float = 0.0003
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current: list[dict[str, object]] = []
    historical: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    current_epochs: list[dict[str, object]] = []
    v27_epochs: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        for fold in range(5):
            parent = 0.099
            candidate = 0.100
            v27 = candidate - v27_delta
            for trial_id, score, parameters in (
                ("control", 0.096, 6260),
                ("parent", parent, 6348),
                ("v25", 0.095, 6352),
                ("v26", 0.097, 6436),
                ("v27", v27, 6436),
            ):
                historical.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "fold": fold,
                        "best_mean_daily_rankic": score,
                        "rankic_1d": score,
                        "rankic_2d": score,
                        "rankic_3d": score,
                        "rankic_5d": score,
                        "parameter_count": parameters,
                    }
                )
            current.append(
                {
                    "trial_id": "candidate",
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": 1,
                    "baseline_epoch": 0,
                    "baseline_mean_daily_rankic": parent,
                    "best_mean_daily_rankic": candidate,
                    "rankic_1d": candidate,
                    "rankic_2d": candidate,
                    "rankic_3d": candidate,
                    "rankic_5d": candidate,
                    "samples_per_second": 5000.0,
                    "parameter_count": 6436,
                    "trainable_parameter_count": 88,
                    "frozen_parameter_count": 6348,
                    "frozen_parent_state_drift_max": 0.0,
                    "parent_prediction_max_abs_error": 0.0,
                    "dynamic_skip_shape_output_weight_l2": 0.1,
                    "dynamic_skip_frozen_parent": True,
                    "dynamic_skip_shape_residual": True,
                    "dynamic_skip_shape_residual_scale": 0.25,
                    "dynamic_skip_raw_parameter_count": 88,
                    "dynamic_skip_shape_residual_parameter_count": 88,
                    "dynamic_skip_shape_normalization_parameter_count": 0,
                    "optimizer_group_identity": "shape-residual-only-lr-0.003",
                    "parent_checkpoint_sha256": f"{seed + fold:064x}",
                    "checkpoint_min_delta": 0.0,
                    "patience_min_delta": 0.0005,
                    "checkpoint_selection_identity": (
                        "best-any-strict-improvement+patience-material-0.0005"
                    ),
                }
            )
            diagnostics.append(
                {
                    "trial_id": "candidate",
                    "seed": seed,
                    "fold": fold,
                    "raw_only_mean_daily_rankic": parent,
                    "shape_residual_weight_effect_max": 1e-4,
                    "simplex_error_max": 1e-7,
                }
            )
            for epoch, score in ((0, parent), (1, candidate)):
                v27_epochs.append(
                    {
                        "trial_id": "v27",
                        "seed": seed,
                        "fold": fold,
                        "epoch": epoch,
                        "mean_daily_rankic": score,
                    }
                )
                current_epochs.append(
                    {
                        "trial_id": "candidate",
                        "seed": seed,
                        "fold": fold,
                        "epoch": epoch,
                        "mean_daily_rankic": (
                            score + trajectory_error if epoch == 1 else score
                        ),
                    }
                )
    return (
        pd.DataFrame(current),
        pd.DataFrame(historical),
        pd.DataFrame(diagnostics),
        pd.DataFrame(current_epochs),
        pd.DataFrame(v27_epochs),
    )


def _decision(*, trajectory_error: float = 0.0, v27_delta: float = 0.0003):
    current, historical, diagnostics, current_epochs, v27_epochs = _decision_frames(
        trajectory_error=trajectory_error, v27_delta=v27_delta
    )
    return evaluate_decoupled_checkpoint_selection_multiseed(
        current,
        historical,
        diagnostics,
        current_epochs,
        v27_epochs,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        parent_candidate_trial_id="parent",
        v25_trial_id="v25",
        v26_trial_id="v26",
        v27_trial_id="v27",
        candidate_trial_id="candidate",
        expected_seeds=(7, 17, 27),
        min_mean_rankic=0.0995,
        min_positive_units=15,
        min_parent_mean_rankic_delta=0.0007,
        min_control_mean_rankic_delta=0.002,
        min_v26_mean_rankic_delta=0.0015,
        min_v25_mean_rankic_delta=0.003,
        min_v27_mean_rankic_delta=0.00015,
        min_nondegrading_folds_per_seed=5,
        min_horizon_parent_delta_1d=-0.001,
        min_horizon_parent_delta_2d=-0.001,
        min_horizon_parent_delta_3d=-0.001,
        min_horizon_parent_delta_5d=-0.001,
        max_trajectory_rankic_abs_error=1e-12,
        max_selected_best_abs_error=1e-12,
        max_parent_rankic_abs_error=1e-7,
        max_parent_prediction_abs_error=1e-7,
        min_trained_effect_units=8,
        min_shape_output_weight_l2=1e-12,
        min_shape_residual_weight_effect=1e-6,
        max_simplex_error=1e-6,
        min_median_samples_per_second=4500.0,
        candidate_parameter_count=6436,
        trainable_parameter_count=88,
        frozen_parameter_count=6348,
        shape_residual_scale=0.25,
        learning_rate=0.003,
        checkpoint_min_delta=0.0,
        patience_min_delta=0.0005,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v28_decision_requires_identical_trajectory_and_incremental_gain() -> None:
    decision = _decision()
    assert decision.status == "decoupled_checkpoint_selection_confirmed_v28"
    assert decision.integrity_passed is True
    assert decision.effect_passed is True

    drifted = _decision(trajectory_error=1e-6)
    assert drifted.status == "stop_decoupled_checkpoint_integrity_v28"
    assert drifted.integrity_passed is False

    no_gain = _decision(v27_delta=0.0)
    assert no_gain.status == "stop_decoupled_checkpoint_no_gain_v28"
    assert no_gain.integrity_passed is True
    assert no_gain.effect_passed is False
