from __future__ import annotations

import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_dynamic_skip_lr_multiseed,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    build_tcn_optimizer,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import DynamicHorizonSkipTCN


def _production_model() -> DynamicHorizonSkipTCN:
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


def _trial(*, dynamic_skip_learning_rate: float | None) -> TCNTuningTrial:
    return TCNTuningTrial(
        trial_id="candidate",
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=128,
        model_kind="dynamic_horizon_skip",
        padding_mode="chomp",
        dynamic_skip_hidden=4,
        dynamic_skip_scale=1.0,
        dynamic_skip_learning_rate=dynamic_skip_learning_rate,
    )


def test_dynamic_skip_optimizer_groups_are_complete_disjoint_and_auditable() -> None:
    model = _production_model()
    dynamic_parameters = model.dynamic_skip_parameters()
    assert sum(parameter.numel() for parameter in model.parameters()) == 6348
    assert sum(parameter.numel() for parameter in dynamic_parameters) == 88

    bundle = build_tcn_optimizer(
        model, _trial(dynamic_skip_learning_rate=0.01)
    )
    groups = {
        str(group["group_name"]): (
            float(group["lr"]),
            sum(parameter.numel() for parameter in group["params"]),
        )
        for group in bundle.optimizer.param_groups
    }
    assert groups == {
        "base": (pytest.approx(0.003), 6260),
        "dynamic_skip": (pytest.approx(0.01), 88),
    }
    assert bundle.dynamic_skip_parameter_count == 88
    assert bundle.parameter_group_identity == (
        "base-lr-0.003+dynamic-skip-lr-0.01"
    )


def test_dynamic_skip_learning_rate_is_explicit_and_fail_closed() -> None:
    raw = {
        "trial_id": "candidate",
        "model_kind": "dynamic_horizon_skip",
        "channels": 16,
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 128,
        "strategy": "smooth_l1",
        "padding_mode": "chomp",
        "dynamic_skip_hidden": 4,
        "dynamic_skip_scale": 1.0,
        "dynamic_skip_learning_rate": 0.01,
    }
    parsed = parse_real_tcn_trials([raw])[0]
    assert parsed.dynamic_skip_learning_rate == pytest.approx(0.01)
    assert validate_tcn_tuning_plan(
        (parsed,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
    ) == (parsed,)

    invalid_model = TCNTuningTrial(
        **{**parsed.__dict__, "model_kind": "horizon_skip"}
    )
    with pytest.raises(ContractError, match="only valid for dynamic horizon skip"):
        validate_tcn_tuning_plan(
            (invalid_model,),
            input_steps=480,
            max_epochs=8,
            patience=2,
            min_delta=0.002,
        )

    excessive = TCNTuningTrial(
        **{**parsed.__dict__, "dynamic_skip_learning_rate": 0.031}
    )
    with pytest.raises(ContractError, match="no greater than ten times"):
        validate_tcn_tuning_plan(
            (excessive,),
            input_steps=480,
            max_epochs=8,
            patience=2,
            min_delta=0.002,
        )


def test_independent_learning_rate_increases_dynamic_parameter_movement() -> None:
    torch.manual_seed(91)
    baseline = DynamicHorizonSkipTCN(
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
    torch.manual_seed(91)
    candidate = DynamicHorizonSkipTCN(
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
    inputs = torch.randn(6, 3, 16)
    targets = torch.randn(6, 4)
    initial = baseline.dynamic_skip_weights(baseline.encode_blocks(inputs))
    baseline_initial_parameters = tuple(
        parameter.detach().clone() for parameter in baseline.dynamic_skip_parameters()
    )
    candidate_initial_parameters = tuple(
        parameter.detach().clone() for parameter in candidate.dynamic_skip_parameters()
    )
    for model, trial in (
        (baseline, _trial(dynamic_skip_learning_rate=None)),
        (candidate, _trial(dynamic_skip_learning_rate=0.01)),
    ):
        optimizer = build_tcn_optimizer(model, trial).optimizer
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.smooth_l1_loss(model(inputs), targets).backward()
        optimizer.step()
    baseline_weights = baseline.dynamic_skip_weights(baseline.encode_blocks(inputs))
    candidate_weights = candidate.dynamic_skip_weights(candidate.encode_blocks(inputs))
    baseline_movement = torch.linalg.vector_norm(baseline_weights - initial)
    candidate_movement = torch.linalg.vector_norm(candidate_weights - initial)
    assert float(candidate_movement.detach()) > float(baseline_movement.detach())
    baseline_parameter_movement = sum(
        float(torch.linalg.vector_norm(parameter.detach() - initial_parameter))
        for parameter, initial_parameter in zip(
            baseline.dynamic_skip_parameters(), baseline_initial_parameters, strict=True
        )
    )
    candidate_parameter_movement = sum(
        float(torch.linalg.vector_norm(parameter.detach() - initial_parameter))
        for parameter, initial_parameter in zip(
            candidate.dynamic_skip_parameters(), candidate_initial_parameters, strict=True
        )
    )
    assert candidate_parameter_movement > 3.0 * baseline_parameter_movement


def _evidence(
    *,
    candidate_static_delta: float = 0.006,
    candidate_parent_delta: float = 0.002,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current: list[dict[str, object]] = []
    historical: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        for fold in range(5):
            control = 0.095 + 0.001 * fold
            parent = control + candidate_static_delta - candidate_parent_delta
            candidate = control + candidate_static_delta
            shared = {
                "seed": seed,
                "fold": fold,
                "rankic_1d": control - 0.01,
                "rankic_2d": control,
                "rankic_3d": control + 0.005,
                "rankic_5d": control + 0.01,
                "samples_per_second": 5400.0,
            }
            historical.extend(
                [
                    {
                        **shared,
                        "trial_id": "control",
                        "best_mean_daily_rankic": control,
                        "parameter_count": 6260,
                    },
                    {
                        **shared,
                        "trial_id": "parent",
                        "best_mean_daily_rankic": parent,
                        "parameter_count": 6348,
                    },
                ]
            )
            current.append(
                {
                    **shared,
                    "trial_id": "candidate",
                    "best_mean_daily_rankic": candidate,
                    "rankic_1d": candidate - 0.01,
                    "rankic_2d": candidate,
                    "rankic_3d": candidate + 0.005,
                    "rankic_5d": candidate + 0.01,
                    "parameter_count": 6348,
                    "dynamic_skip_output_weight_l2": 0.2,
                    "dynamic_skip_learning_rate": 0.01,
                    "optimizer_dynamic_skip_parameter_count": 88,
                }
            )
    return pd.DataFrame(current), pd.DataFrame(historical)


def _diagnostics(trial_id: str, variation: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_id": trial_id,
                "seed": seed,
                "fold": fold,
                "block_weight_variation": variation,
                "simplex_error_max": 1e-7,
            }
            for seed in (7, 17, 27)
            for fold in range(5)
        ]
    )


def _decision(
    *,
    parent_delta: float = 0.002,
    variation: float = 0.004,
    speed: float = 3.4,
):
    current, historical = _evidence(candidate_parent_delta=parent_delta)
    return evaluate_dynamic_skip_lr_multiseed(
        current,
        historical,
        _diagnostics("candidate", variation),
        _diagnostics("parent", 0.002),
        {
            "model_step_speed_ratio": speed,
            "end_to_end_speed_ratio": 3.2,
        },
        control_trial_id="control",
        parent_candidate_trial_id="parent",
        candidate_trial_id="candidate",
        expected_seeds=(7, 17, 27),
        min_mean_rankic=0.1,
        min_positive_units=15,
        min_mean_rankic_delta=0.003,
        min_parent_mean_rankic_delta=0.001,
        min_nondegrading_folds_per_seed=3,
        min_horizon_delta_1d=0.0,
        min_horizon_delta_2d=-0.003,
        min_horizon_delta_3d=-0.005,
        min_horizon_delta_5d=-0.005,
        min_median_samples_per_second=5000.0,
        min_dynamic_skip_output_weight_l2=1e-12,
        min_block_weight_variation=1e-6,
        min_parent_variation_ratio=1.5,
        max_simplex_error=1e-6,
        control_parameter_count=6260,
        candidate_parameter_count=6348,
        dynamic_parameter_count=88,
        base_learning_rate=0.003,
        dynamic_skip_learning_rate=0.01,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v22_requires_static_gain_parent_gain_mechanism_and_speed() -> None:
    decision = _decision()
    assert decision.status == "dynamic_skip_lr_multiseed_confirmed_v22"
    assert decision.effect_passed is True
    assert decision.speed_passed is True
    assert decision.aggregate["mean_rankic_delta"] == pytest.approx(0.006)
    assert decision.aggregate["parent_mean_rankic_delta"] == pytest.approx(0.002)
    assert decision.aggregate["parent_variation_ratio"] == pytest.approx(2.0)
    assert decision.seed_summary["nondegrading_folds"].tolist() == [5, 5, 5]

    weak_parent = _decision(parent_delta=0.0005)
    assert weak_parent.status == "stop_dynamic_skip_lr_unstable_v22"
    assert "parent_mean_rankic_delta_below_gate" in str(
        weak_parent.aggregate["blockers"]
    )

    inactive = _decision(variation=0.0025)
    assert inactive.status == "stop_dynamic_skip_lr_unstable_v22"
    assert "parent_variation_ratio_below_gate" in str(
        inactive.aggregate["blockers"]
    )

    slow = _decision(speed=2.9)
    assert slow.status == "stop_dynamic_skip_lr_speed_v22"
    assert slow.effect_passed is True
    assert slow.speed_passed is False


def test_v22_rejects_incomplete_current_or_historical_evidence() -> None:
    current, historical = _evidence()
    with pytest.raises(ContractError, match="coverage"):
        evaluate_dynamic_skip_lr_multiseed(
            current.iloc[:-1],
            historical,
            _diagnostics("candidate", 0.004),
            _diagnostics("parent", 0.002),
            {"model_step_speed_ratio": 3.4, "end_to_end_speed_ratio": 3.2},
            control_trial_id="control",
            parent_candidate_trial_id="parent",
            candidate_trial_id="candidate",
            expected_seeds=(7, 17, 27),
            min_mean_rankic=0.1,
            min_positive_units=15,
            min_mean_rankic_delta=0.003,
            min_parent_mean_rankic_delta=0.001,
            min_nondegrading_folds_per_seed=3,
            min_horizon_delta_1d=0.0,
            min_horizon_delta_2d=-0.003,
            min_horizon_delta_3d=-0.005,
            min_horizon_delta_5d=-0.005,
            min_median_samples_per_second=5000.0,
            min_dynamic_skip_output_weight_l2=1e-12,
            min_block_weight_variation=1e-6,
            min_parent_variation_ratio=1.5,
            max_simplex_error=1e-6,
            control_parameter_count=6260,
            candidate_parameter_count=6348,
            dynamic_parameter_count=88,
            base_learning_rate=0.003,
            dynamic_skip_learning_rate=0.01,
            min_model_step_speed_ratio=3.0,
            min_end_to_end_speed_ratio=3.0,
        )
