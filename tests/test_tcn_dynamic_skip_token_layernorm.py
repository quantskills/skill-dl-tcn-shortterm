from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_dynamic_skip_token_layernorm_multiseed,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    build_tcn_trial_model,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import (
    DynamicHorizonSkipTCN,
    HorizonSkipTCN,
)


def _dynamic(
    normalization: Literal["none", "layer_norm"]
) -> DynamicHorizonSkipTCN:
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
        dynamic_skip_token_normalization=normalization,
    )


def test_token_layernorm_is_parameter_free_scale_invariant_and_static_at_init() -> None:
    torch.manual_seed(51)
    control = HorizonSkipTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        dropout=0.0,
        padding_mode="chomp",
    )
    torch.manual_seed(51)
    candidate = _dynamic("layer_norm")
    inputs = torch.randn(6, 3, 16)
    assert torch.equal(control(inputs), candidate(inputs))
    assert sum(parameter.numel() for parameter in candidate.parameters()) == (
        sum(parameter.numel() for parameter in control.parameters()) + 22
    )
    assert sum(
        parameter.numel() for parameter in candidate.dynamic_skip_normalizer.parameters()
    ) == 0

    torch.nn.init.normal_(candidate.dynamic_skip_output.weight)
    sequences = candidate.encode_blocks(inputs)
    original = candidate.dynamic_skip_weights(sequences)
    transformed = []
    for offset, sequence in enumerate(sequences):
        scale = torch.linspace(0.5, 2.0, len(sequence)).reshape(-1, 1, 1)
        shift = torch.linspace(-3.0, 3.0, len(sequence)).reshape(-1, 1, 1)
        transformed.append(sequence * scale + shift + offset)
    normalized = candidate.dynamic_skip_weights(transformed)
    assert torch.allclose(original, normalized, atol=2e-5, rtol=2e-5)

    raw = _dynamic("none")
    raw.load_state_dict(candidate.state_dict(), strict=False)
    raw_original = raw.dynamic_skip_weights(sequences)
    raw_transformed = raw.dynamic_skip_weights(transformed)
    assert not torch.allclose(raw_original, raw_transformed, atol=1e-5, rtol=1e-5)


def test_token_layernorm_config_factory_and_validation_fail_closed() -> None:
    raw = {
        "trial_id": "token-ln",
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
        "dynamic_skip_input_normalization": "layer_norm",
    }
    trial = parse_real_tcn_trials([raw])[0]
    assert trial.dynamic_skip_token_normalization == "layer_norm"
    assert validate_tcn_tuning_plan(
        (trial,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
    ) == (trial,)
    model = build_tcn_trial_model(trial, feature_count=8, input_steps=480)
    assert isinstance(model, DynamicHorizonSkipTCN)
    assert model.receipt_metadata()["dynamic_skip_token_normalization"] == "layer_norm"
    assert sum(parameter.numel() for parameter in model.parameters()) == 6348

    invalid = TCNTuningTrial(
        **{
            **trial.__dict__,
            "model_kind": "horizon_skip",
        }
    )
    with pytest.raises(ContractError, match="only valid for dynamic horizon skip"):
        validate_tcn_tuning_plan(
            (invalid,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
        )


def _evidence(
    *, seed27_parent_delta: float = 0.0015
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current: list[dict[str, object]] = []
    historical: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        for fold in range(5):
            control = 0.096 + 0.001 * fold
            parent = control + 0.004
            parent_delta = seed27_parent_delta if seed == 27 else 0.0015
            candidate = parent + parent_delta
            historical.extend(
                [
                    {
                        "trial_id": "control",
                        "seed": seed,
                        "fold": fold,
                        "best_mean_daily_rankic": control,
                        "rankic_1d": control - 0.01,
                        "rankic_2d": control,
                        "rankic_3d": control + 0.005,
                        "rankic_5d": control + 0.01,
                        "parameter_count": 6260,
                    },
                    {
                        "trial_id": "parent",
                        "seed": seed,
                        "fold": fold,
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
                    "trial_id": "candidate",
                    "seed": seed,
                    "fold": fold,
                    "best_mean_daily_rankic": candidate,
                    "rankic_1d": candidate - 0.01,
                    "rankic_2d": candidate,
                    "rankic_3d": candidate + 0.005,
                    "rankic_5d": candidate + 0.01,
                    "samples_per_second": 5300.0,
                    "model_step_samples_per_second": 5700.0,
                    "parameter_count": 6348,
                    "dynamic_skip_output_weight_l2": 0.2,
                    "dynamic_skip_token_normalization": "layer_norm",
                    "optimizer_group_identity": "all-lr-0.003",
                    "optimizer_dynamic_skip_parameter_count": 0,
                }
            )
    return pd.DataFrame(current), pd.DataFrame(historical)


def _diagnostics(trial_id: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_id": trial_id,
                "seed": seed,
                "fold": fold,
                "block_weight_variation": values[fold] * 1e-3,
                "simplex_error_max": 1e-7,
            }
            for seed in (7, 17, 27)
            for fold in range(5)
        ]
    )


def _decision(
    *,
    seed27_parent_delta: float = 0.0015,
    current_variation: list[float] | None = None,
    speed: float = 3.5,
):
    current, historical = _evidence(seed27_parent_delta=seed27_parent_delta)
    return evaluate_dynamic_skip_token_layernorm_multiseed(
        current,
        historical,
        _diagnostics("candidate", current_variation or [1.5, 2.0, 2.5, 1.5, 2.0]),
        _diagnostics("parent", [1.0, 2.0, 3.0, 1.0, 2.0]),
        {"model_step_speed_ratio": speed, "end_to_end_speed_ratio": 3.2},
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
        max_parent_variation_cv_ratio=0.9,
        max_simplex_error=1e-6,
        control_parameter_count=6260,
        candidate_parameter_count=6348,
        dynamic_parameter_count=88,
        learning_rate=0.003,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v24_requires_parent_gain_lower_variation_cv_and_speed() -> None:
    decision = _decision()
    assert decision.status == "dynamic_skip_token_layernorm_multiseed_confirmed_v24"
    assert decision.effect_passed is True
    assert decision.speed_passed is True
    assert float(decision.aggregate["parent_variation_cv_ratio"]) < 0.9

    parent_regression = _decision(seed27_parent_delta=-0.001)
    assert parent_regression.status == "stop_dynamic_skip_token_layernorm_unstable_v24"
    assert "per_seed_parent_mean_delta_not_positive" in str(
        parent_regression.aggregate["blockers"]
    )
    unstable = _decision(current_variation=[1.0, 2.0, 3.0, 1.0, 2.0])
    assert unstable.status == "stop_dynamic_skip_token_layernorm_unstable_v24"
    assert "variation_cv_not_reduced" in str(unstable.aggregate["blockers"])
    slow = _decision(speed=2.9)
    assert slow.status == "stop_dynamic_skip_token_layernorm_speed_v24"
    assert slow.effect_passed is True
    assert slow.speed_passed is False
