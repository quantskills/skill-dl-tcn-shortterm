from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_frozen_parent_shape_residual_multiseed,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    run_tcn_validation_sweep,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import (
    DynamicHorizonSkipTCN,
    ShapeResidualDynamicHorizonSkipTCN,
)


def _parent() -> DynamicHorizonSkipTCN:
    return DynamicHorizonSkipTCN(
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


def _candidate() -> ShapeResidualDynamicHorizonSkipTCN:
    return ShapeResidualDynamicHorizonSkipTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        dropout=0.0,
        padding_mode="chomp",
        dynamic_skip_hidden=2,
        dynamic_skip_scale=1.0,
        dynamic_skip_shape_residual_scale=0.25,
    )


def test_frozen_parent_loader_preserves_parent_and_only_trains_shape() -> None:
    torch.manual_seed(271)
    parent = _parent()
    parent_state = {
        name: tensor.detach().clone() for name, tensor in parent.state_dict().items()
    }
    torch.manual_seed(999)
    candidate = _candidate()

    candidate.load_frozen_raw_parent(parent_state)
    inputs = torch.randn(7, 3, 16)
    assert torch.equal(parent(inputs), candidate(inputs))
    assert torch.equal(parent(inputs), candidate.forward_without_shape_residual(inputs))
    trainable = {
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        "dynamic_skip_shape_hidden.weight",
        "dynamic_skip_shape_hidden.bias",
        "dynamic_skip_shape_output.weight",
        "dynamic_skip_shape_output.bias",
    }
    assert sum(
        parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad
    ) == 22
    metadata = candidate.receipt_metadata()
    assert metadata["frozen_parent"] is True
    assert metadata["trainable_parameter_count"] == 22
    assert metadata["frozen_parameter_count"] == 226

    incomplete = dict(parent_state)
    incomplete.pop("dynamic_skip_output.bias")
    with pytest.raises(ContractError, match="parent state keys"):
        _candidate().load_frozen_raw_parent(incomplete)


def _tiny_data() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(272)
    features = rng.normal(size=(18, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(18),
            "sample_id": [f"frozen-{value}" for value in range(18)],
            "signal_date": [f"2025-05-{2 + value // 6:02d}" for value in range(18)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"frozen-{sample}",
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
    return features, index, labels, split


def _trial() -> TCNTuningTrial:
    return TCNTuningTrial(
        trial_id="frozen-shape",
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


def test_frozen_parent_sweep_keeps_epoch_zero_and_audits_no_drift() -> None:
    features, index, labels, split = _tiny_data()
    trial = _trial()
    torch.manual_seed(273)
    parent_state = {
        name: tensor.detach().clone() for name, tensor in _parent().state_dict().items()
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
        min_delta=10.0,
        torch_threads=1,
        frozen_parent_states={"frozen-shape-fold-0": parent_state},
    )

    assert result.epoch_history["epoch"].tolist()[0] == 0
    assert result.epoch_history["stage"].tolist()[0] == "frozen_parent_baseline"
    row = result.leaderboard.iloc[0]
    assert row["best_epoch"] == 0
    assert row["baseline_epoch"] == 0
    assert row["trainable_parameter_count"] == 22
    assert row["frozen_parameter_count"] == 226
    assert row["frozen_parent_state_drift_max"] == 0.0
    assert row["optimizer_group_identity"] == "shape-residual-only-lr-0.003"
    assert row["parent_prediction_max_abs_error"] == 0.0


def test_frozen_parent_config_is_strict_and_requires_parent_states() -> None:
    raw = {
        "trial_id": "frozen-shape",
        "model_kind": "dynamic_horizon_skip",
        "channels": 4,
        "kernel_size": 2,
        "dilations": [1, 2, 4, 8],
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 6,
        "strategy": "smooth_l1",
        "padding_mode": "chomp",
        "dynamic_skip_hidden": 2,
        "dynamic_skip_scale": 1.0,
        "dynamic_skip_shape_residual": True,
        "dynamic_skip_shape_residual_scale": 0.25,
        "dynamic_skip_frozen_parent": True,
    }
    trial = parse_real_tcn_trials([raw])[0]
    assert trial.dynamic_skip_frozen_parent is True
    assert validate_tcn_tuning_plan(
        (trial,), input_steps=16, max_epochs=2, patience=1, min_delta=0.0005
    ) == (trial,)
    invalid = TCNTuningTrial(
        **{
            **trial.__dict__,
            "dynamic_skip_shape_residual": False,
        }
    )
    with pytest.raises(ContractError, match="requires shape residual"):
        validate_tcn_tuning_plan(
            (invalid,), input_steps=16, max_epochs=2, patience=1, min_delta=0.0005
        )

    features, index, labels, split = _tiny_data()
    with pytest.raises(ContractError, match="parent state"):
        run_tcn_validation_sweep(
            features,
            index,
            labels,
            split,
            trials=(trial,),
            seed=7,
            max_epochs=2,
            patience=1,
            min_delta=0.0005,
            torch_threads=1,
        )


def _decision_evidence(
    *, drift: float = 0.0, parent_delta: float = 0.001
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current: list[dict[str, object]] = []
    historical: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        for fold in range(5):
            control = 0.096
            parent = 0.099
            v25 = 0.095
            v26 = 0.097
            candidate = parent + parent_delta
            for trial_id, score, parameters in (
                ("control", control, 6260),
                ("parent", parent, 6348),
                ("v25", v25, 6352),
                ("v26", v26, 6436),
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
                    "frozen_parent_state_drift_max": drift,
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


def _decision(*, drift: float = 0.0, parent_delta: float = 0.001):
    current, historical, diagnostics = _decision_evidence(
        drift=drift, parent_delta=parent_delta
    )
    return evaluate_frozen_parent_shape_residual_multiseed(
        current,
        historical,
        diagnostics,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        parent_candidate_trial_id="parent",
        v25_trial_id="v25",
        v26_trial_id="v26",
        candidate_trial_id="candidate",
        expected_seeds=(7, 17, 27),
        min_mean_rankic=0.0995,
        min_positive_units=15,
        min_parent_mean_rankic_delta=0.0005,
        min_control_mean_rankic_delta=0.002,
        min_v26_mean_rankic_delta=0.0015,
        min_v25_mean_rankic_delta=0.003,
        min_nondegrading_folds_per_seed=3,
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
        learning_rate=0.003,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v27_decision_separates_integrity_effect_and_speed() -> None:
    decision = _decision()
    assert decision.status == "frozen_parent_shape_residual_confirmed_v27"
    assert decision.integrity_passed is True
    assert decision.effect_passed is True
    assert decision.speed_passed is True

    corrupted = _decision(drift=1e-6)
    assert corrupted.status == "stop_frozen_parent_integrity_v27"
    assert corrupted.integrity_passed is False

    no_gain = _decision(parent_delta=0.0)
    assert no_gain.status == "stop_frozen_parent_shape_residual_no_gain_v27"
    assert no_gain.integrity_passed is True
    assert no_gain.effect_passed is False
