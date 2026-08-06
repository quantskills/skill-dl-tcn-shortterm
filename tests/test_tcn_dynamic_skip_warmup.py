from __future__ import annotations

import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_dynamic_skip_warmup_multiseed,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    apply_tcn_epoch_learning_rates,
    build_tcn_optimizer,
    dynamic_skip_learning_rate_for_epoch,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import DynamicHorizonSkipTCN


SCHEDULE_IDENTITY = (
    "base-lr-0.003+dynamic-skip-linear-warmup-2-lr-0.003-to-0.005"
)


def _trial(**overrides: object) -> TCNTuningTrial:
    values: dict[str, object] = {
        "trial_id": "warmup",
        "channels": 16,
        "kernel_size": 3,
        "dilations": (1, 2, 4, 8, 16, 32, 64, 128),
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 128,
        "model_kind": "dynamic_horizon_skip",
        "padding_mode": "chomp",
        "dynamic_skip_hidden": 4,
        "dynamic_skip_scale": 1.0,
        "dynamic_skip_learning_rate": 0.005,
        "dynamic_skip_warmup_epochs": 2,
    }
    values.update(overrides)
    return TCNTuningTrial(**values)  # type: ignore[arg-type]


def _model() -> DynamicHorizonSkipTCN:
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


def test_dynamic_skip_warmup_has_exact_public_epoch_schedule() -> None:
    trial = _trial()
    assert [
        dynamic_skip_learning_rate_for_epoch(trial, epoch)
        for epoch in range(1, 6)
    ] == pytest.approx([0.003, 0.004, 0.005, 0.005, 0.005])
    with pytest.raises(ContractError, match="epoch must start at one"):
        dynamic_skip_learning_rate_for_epoch(trial, 0)

    bundle = build_tcn_optimizer(_model(), trial)
    assert bundle.parameter_group_identity == SCHEDULE_IDENTITY
    observed = []
    for epoch in range(1, 4):
        applied = apply_tcn_epoch_learning_rates(bundle, trial, epoch)
        groups = {
            str(group["group_name"]): float(group["lr"])
            for group in bundle.optimizer.param_groups
        }
        observed.append(applied)
        assert groups["base"] == pytest.approx(0.003)
        assert groups["dynamic_skip"] == pytest.approx(applied)
    assert observed == pytest.approx([0.003, 0.004, 0.005])


def test_dynamic_skip_warmup_config_is_explicit_and_fail_closed() -> None:
    raw = {
        "trial_id": "warmup",
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
        "dynamic_skip_learning_rate": 0.005,
        "dynamic_skip_warmup_epochs": 2,
    }
    parsed = parse_real_tcn_trials([raw])[0]
    assert parsed.dynamic_skip_warmup_epochs == 2
    assert validate_tcn_tuning_plan(
        (parsed,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
    ) == (parsed,)

    without_target = _trial(dynamic_skip_learning_rate=None)
    with pytest.raises(ContractError, match="requires a target learning rate"):
        validate_tcn_tuning_plan(
            (without_target,),
            input_steps=480,
            max_epochs=8,
            patience=2,
            min_delta=0.002,
        )
    static = _trial(model_kind="horizon_skip")
    with pytest.raises(ContractError, match="only valid for dynamic horizon skip"):
        validate_tcn_tuning_plan(
            (static,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
        )
    too_long = _trial(dynamic_skip_warmup_epochs=8)
    with pytest.raises(ContractError, match="smaller than max_epochs"):
        validate_tcn_tuning_plan(
            (too_long,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
        )


def _evidence(
    *, seed27_parent_delta: float = 0.001
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current: list[dict[str, object]] = []
    historical: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        for fold in range(5):
            control = 0.096 + 0.001 * fold
            parent = control + 0.004
            parent_delta = seed27_parent_delta if seed == 27 else 0.001
            candidate = parent + parent_delta
            shared = {
                "seed": seed,
                "fold": fold,
                "samples_per_second": 5350.0,
            }
            historical.extend(
                [
                    {
                        **shared,
                        "trial_id": "control",
                        "best_mean_daily_rankic": control,
                        "rankic_1d": control - 0.01,
                        "rankic_2d": control,
                        "rankic_3d": control + 0.005,
                        "rankic_5d": control + 0.01,
                        "parameter_count": 6260,
                    },
                    {
                        **shared,
                        "trial_id": "parent",
                        "best_mean_daily_rankic": parent,
                        "rankic_1d": parent - 0.01,
                        "rankic_2d": parent,
                        "rankic_3d": parent + 0.005,
                        "rankic_5d": parent + 0.01,
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
                    "dynamic_skip_output_weight_l2": 0.35,
                    "dynamic_skip_learning_rate": 0.005,
                    "dynamic_skip_warmup_epochs": 2,
                    "optimizer_group_identity": SCHEDULE_IDENTITY,
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
    seed27_parent_delta: float = 0.001,
    current_variation: float = 0.004,
    speed: float = 3.5,
):
    current, historical = _evidence(
        seed27_parent_delta=seed27_parent_delta
    )
    return evaluate_dynamic_skip_warmup_multiseed(
        current,
        historical,
        _diagnostics("candidate", current_variation),
        _diagnostics("parent", 0.002),
        _diagnostics("high", 0.008),
        {"model_step_speed_ratio": speed, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        parent_candidate_trial_id="parent",
        high_lr_candidate_trial_id="high",
        candidate_trial_id="candidate",
        expected_seeds=(7, 17, 27),
        min_mean_rankic=0.099,
        min_positive_units=15,
        min_mean_rankic_delta=0.003,
        min_parent_mean_rankic_delta=0.0005,
        min_nondegrading_folds_per_seed=3,
        min_horizon_delta_1d=0.0,
        min_horizon_delta_2d=-0.003,
        min_horizon_delta_3d=-0.005,
        min_horizon_delta_5d=-0.005,
        min_median_samples_per_second=5000.0,
        min_dynamic_skip_output_weight_l2=1e-12,
        min_block_weight_variation=1e-6,
        min_parent_variation_ratio=1.2,
        max_parent_variation_ratio=3.0,
        max_high_lr_variation_ratio=0.75,
        max_simplex_error=1e-6,
        control_parameter_count=6260,
        candidate_parameter_count=6348,
        dynamic_parameter_count=88,
        base_learning_rate=0.003,
        dynamic_skip_learning_rate=0.005,
        dynamic_skip_warmup_epochs=2,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v23_requires_parent_gain_bounded_variation_and_speed() -> None:
    decision = _decision()
    assert decision.status == "dynamic_skip_warmup_multiseed_confirmed_v23"
    assert decision.effect_passed is True
    assert decision.speed_passed is True
    assert decision.aggregate["parent_variation_ratio"] == pytest.approx(2.0)
    assert decision.aggregate["high_lr_variation_ratio"] == pytest.approx(0.5)

    parent_regression = _decision(seed27_parent_delta=-0.001)
    assert parent_regression.status == "stop_dynamic_skip_warmup_unstable_v23"
    assert "per_seed_parent_mean_delta_not_positive" in str(
        parent_regression.aggregate["blockers"]
    )
    overactive = _decision(current_variation=0.007)
    assert overactive.status == "stop_dynamic_skip_warmup_unstable_v23"
    assert "parent_variation_ratio_above_gate" in str(
        overactive.aggregate["blockers"]
    )
    slow = _decision(speed=2.9)
    assert slow.status == "stop_dynamic_skip_warmup_speed_v23"
    assert slow.effect_passed is True
    assert slow.speed_passed is False
