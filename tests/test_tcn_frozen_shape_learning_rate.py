from __future__ import annotations

import pandas as pd
import pytest

from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_frozen_shape_learning_rate_multiseed,
)
from skill_dl_tcn_shortterm.tuning import TCNTuningTrial, build_tcn_optimizer
from skill_dl_tcn_shortterm.v9_representation import (
    DynamicHorizonSkipTCN,
    ShapeResidualDynamicHorizonSkipTCN,
)


def _parent() -> DynamicHorizonSkipTCN:
    return DynamicHorizonSkipTCN(
        feature_count=8,
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        input_steps=480,
        dropout=0.0,
        padding_mode="chomp",
        dynamic_skip_hidden=4,
        dynamic_skip_scale=1.0,
    )


def _candidate() -> ShapeResidualDynamicHorizonSkipTCN:
    return ShapeResidualDynamicHorizonSkipTCN(
        feature_count=8,
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        input_steps=480,
        dropout=0.0,
        padding_mode="chomp",
        dynamic_skip_hidden=4,
        dynamic_skip_scale=1.0,
        dynamic_skip_shape_residual_scale=0.25,
    )


def test_v29_optimizer_only_updates_88_shape_parameters_at_lr001() -> None:
    parent = _parent()
    candidate = _candidate()
    candidate.load_frozen_raw_parent(parent.state_dict())
    trial = TCNTuningTrial(
        trial_id="candidate",
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.001,
        batch_size=128,
        model_kind="dynamic_horizon_skip",
        padding_mode="chomp",
        dynamic_skip_hidden=4,
        dynamic_skip_scale=1.0,
        dynamic_skip_shape_residual=True,
        dynamic_skip_shape_residual_scale=0.25,
        dynamic_skip_frozen_parent=True,
    )

    bundle = build_tcn_optimizer(candidate, trial)
    optimized = {
        id(parameter)
        for group in bundle.optimizer.param_groups
        for parameter in group["params"]
    }
    trainable = {
        id(parameter) for parameter in candidate.parameters() if parameter.requires_grad
    }
    assert optimized == trainable
    assert sum(
        parameter.numel()
        for parameter in candidate.parameters()
        if parameter.requires_grad
    ) == 88
    assert {float(group["lr"]) for group in bundle.optimizer.param_groups} == {0.001}
    assert bundle.parameter_group_identity == "shape-residual-only-lr-0.001"


def _evidence(
    *, optimizer_identity: str = "shape-residual-only-lr-0.001", v28_delta: float = 0.0002
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current: list[dict[str, object]] = []
    historical: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        for fold in range(5):
            parent = 0.099
            candidate = 0.100
            for trial_id, score, parameters in (
                ("control", 0.096, 6260),
                ("parent", parent, 6348),
                ("v25", 0.095, 6352),
                ("v26", 0.097, 6436),
                ("v28", candidate - v28_delta, 6436),
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
                    "optimizer_group_identity": optimizer_identity,
                    "learning_rate": 0.001,
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
    return pd.DataFrame(current), pd.DataFrame(historical), pd.DataFrame(diagnostics)


def _decision(
    *, optimizer_identity: str = "shape-residual-only-lr-0.001", v28_delta: float = 0.0002
):
    current, historical, diagnostics = _evidence(
        optimizer_identity=optimizer_identity, v28_delta=v28_delta
    )
    return evaluate_frozen_shape_learning_rate_multiseed(
        current,
        historical,
        diagnostics,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        parent_candidate_trial_id="parent",
        v25_trial_id="v25",
        v26_trial_id="v26",
        v28_trial_id="v28",
        candidate_trial_id="candidate",
        expected_seeds=(7, 17, 27),
        min_mean_rankic=0.0996,
        min_positive_units=15,
        min_parent_mean_rankic_delta=0.00075,
        min_control_mean_rankic_delta=0.002,
        min_v26_mean_rankic_delta=0.0015,
        min_v25_mean_rankic_delta=0.003,
        min_v28_mean_rankic_delta=0.00015,
        min_nondegrading_folds_per_seed=5,
        min_horizon_parent_delta_1d=-0.001,
        min_horizon_parent_delta_2d=-0.001,
        min_horizon_parent_delta_3d=-0.001,
        min_horizon_parent_delta_5d=-0.001,
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
        learning_rate=0.001,
        checkpoint_min_delta=0.0,
        patience_min_delta=0.0005,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v29_decision_separates_integrity_and_v28_incremental_gain() -> None:
    decision = _decision()
    assert decision.status == "frozen_shape_lr001_confirmed_v29"
    assert decision.integrity_passed is True
    assert decision.effect_passed is True
    assert decision.aggregate["v28_mean_rankic_delta"] == pytest.approx(0.0002)

    drifted = _decision(optimizer_identity="shape-residual-only-lr-0.003")
    assert drifted.status == "stop_frozen_shape_lr_integrity_v29"
    assert drifted.integrity_passed is False

    no_gain = _decision(v28_delta=0.0)
    assert no_gain.status == "stop_frozen_shape_lr_no_gain_v29"
    assert no_gain.integrity_passed is True
    assert no_gain.effect_passed is False
