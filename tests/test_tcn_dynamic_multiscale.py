from __future__ import annotations

import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    DynamicMultiscaleSeed7Decision,
    evaluate_dynamic_multiscale_seed7,
    finalize_dynamic_multiscale_seed7,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    build_tcn_trial_model,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import (
    DynamicHorizonSkipTCN,
    HorizonSkipTCN,
)


def _model(model_type: type[HorizonSkipTCN]) -> HorizonSkipTCN:
    if model_type is DynamicHorizonSkipTCN:
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
    return HorizonSkipTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        dropout=0.0,
        padding_mode="chomp",
    )


def _raw_trial(model_kind: str) -> dict[str, object]:
    raw: dict[str, object] = {
        "trial_id": model_kind,
        "model_kind": model_kind,
        "channels": 16,
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 128,
        "strategy": "smooth_l1",
        "padding_mode": "chomp",
    }
    if model_kind == "dynamic_horizon_skip":
        raw.update({"dynamic_skip_hidden": 4, "dynamic_skip_scale": 1.0})
    return raw


def test_dynamic_multiscale_initializes_as_exact_static_control_with_88_parameters() -> None:
    torch.manual_seed(41)
    control = HorizonSkipTCN(
        feature_count=8,
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        input_steps=480,
        dropout=0.0,
        padding_mode="chomp",
    )
    torch.manual_seed(41)
    candidate = DynamicHorizonSkipTCN(
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
    inputs = torch.randn(2, 8, 480)
    assert torch.equal(control(inputs), candidate(inputs))
    assert sum(parameter.numel() for parameter in control.parameters()) == 6260
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 6348
    dynamic = candidate.dynamic_skip_parameters()
    assert len({id(parameter) for parameter in dynamic}) == len(dynamic)
    assert sum(parameter.numel() for parameter in dynamic) == 88


def test_dynamic_multiscale_weights_are_sample_conditioned_simplexes_and_causal() -> None:
    torch.manual_seed(9)
    candidate = _model(DynamicHorizonSkipTCN)
    assert isinstance(candidate, DynamicHorizonSkipTCN)
    inputs = torch.randn(6, 3, 16)
    blocks = candidate.encode_blocks(inputs)
    initial = candidate.dynamic_skip_weights(blocks)
    assert initial.shape == (6, 4, 4)
    assert torch.allclose(initial.sum(dim=2), torch.ones(6, 4), atol=1e-7)

    optimizer = torch.optim.Adam(candidate.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.smooth_l1_loss(
            candidate(inputs), torch.randn(6, 4)
        ).backward()
        optimizer.step()
    updated = candidate.dynamic_skip_weights(candidate.encode_blocks(inputs))
    assert float(updated.std(dim=0).max().detach()) > 1e-6
    metadata = candidate.receipt_metadata()
    assert metadata["dynamic_skip_parameter_count"] == 22
    output_l2 = metadata["dynamic_skip_output_weight_l2"]
    assert isinstance(output_l2, float) and output_l2 > 0

    altered = inputs.clone()
    altered[:, :, 10:] += 100.0
    for original, changed in zip(
        candidate.encode_blocks(inputs), candidate.encode_blocks(altered), strict=True
    ):
        assert torch.allclose(original[:, :, :10], changed[:, :, :10], atol=1e-6)


def test_dynamic_multiscale_parser_factory_and_validation_are_fail_closed() -> None:
    control, candidate = parse_real_tcn_trials(
        [_raw_trial("horizon_skip"), _raw_trial("dynamic_horizon_skip")]
    )
    assert candidate.dynamic_skip_hidden == 4
    assert candidate.dynamic_skip_scale == pytest.approx(1.0)
    assert validate_tcn_tuning_plan(
        (control, candidate),
        input_steps=480,
        max_epochs=8,
        patience=2,
        min_delta=0.002,
    ) == (control, candidate)
    model = build_tcn_trial_model(candidate, feature_count=8, input_steps=480)
    assert isinstance(model, DynamicHorizonSkipTCN)

    invalid = _raw_trial("dynamic_horizon_skip")
    invalid["dynamic_skip_scale"] = 0.0
    parsed = parse_real_tcn_trials([invalid])[0]
    with pytest.raises(ContractError, match="dynamic skip scale"):
        validate_tcn_tuning_plan(
            (parsed,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
        )


def _leaderboard() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        rows.extend(
            [
                {
                    "trial_id": "control",
                    "fold": fold,
                    "seed": 7,
                    "best_mean_daily_rankic": 0.087,
                    "rankic_1d": 0.08,
                    "rankic_2d": 0.085,
                    "rankic_3d": 0.09,
                    "rankic_5d": 0.093,
                    "samples_per_second": 5700.0,
                    "parameter_count": 6260,
                    "dynamic_skip_output_weight_l2": None,
                },
                {
                    "trial_id": "candidate",
                    "fold": fold,
                    "seed": 7,
                    "best_mean_daily_rankic": 0.094,
                    "rankic_1d": 0.084,
                    "rankic_2d": 0.089,
                    "rankic_3d": 0.094,
                    "rankic_5d": 0.097,
                    "samples_per_second": 5300.0,
                    "parameter_count": 6348,
                    "dynamic_skip_output_weight_l2": 0.2,
                },
            ]
        )
    return pd.DataFrame(rows)


def _effect(variation: float = 0.002) -> DynamicMultiscaleSeed7Decision:
    diagnostics = pd.DataFrame(
        {
            "trial_id": ["candidate"] * 5,
            "fold": list(range(5)),
            "block_weight_variation": [variation] * 5,
            "simplex_error_max": [1e-7] * 5,
        }
    )
    return evaluate_dynamic_multiscale_seed7(
        _leaderboard(),
        diagnostics,
        control_trial_id="control",
        candidate_trial_id="candidate",
        min_mean_rankic=0.09,
        min_mean_rankic_delta=0.003,
        min_positive_folds=5,
        min_nondegrading_folds=3,
        min_horizon_delta_1d=0.0,
        min_horizon_delta_2d=-0.003,
        min_horizon_delta_3d=-0.005,
        min_horizon_delta_5d=-0.005,
        min_median_samples_per_second=5000.0,
        min_dynamic_skip_output_weight_l2=1e-12,
        min_block_weight_variation=1e-6,
        max_simplex_error=1e-6,
        control_parameter_count=6260,
        candidate_parameter_count=6348,
        dynamic_parameter_count=88,
    )


def test_v20_decision_requires_effect_mechanism_capacity_and_speed() -> None:
    effect = _effect()
    assert effect.status == "dynamic_multiscale_seed7_effect_admitted_v20"
    final = finalize_dynamic_multiscale_seed7(
        effect,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert final.status == "dynamic_multiscale_seed7_admitted_v20"
    assert final.confirmation_seeds_authorized == (17, 27)

    rejected = _effect(variation=0.0)
    assert rejected.status == "stop_dynamic_multiscale_seed7_effect_v20"
    row = rejected.summary.loc[
        rejected.summary["trial_id"].astype(str).eq("candidate")
    ].iloc[0]
    assert "block_weights_not_sample_conditioned" in str(row["blockers"])
