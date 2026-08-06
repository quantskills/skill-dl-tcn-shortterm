from __future__ import annotations

import pytest
import torch
import pandas as pd

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_readout import (
    DynamicLRSeed7Decision,
    evaluate_dynamic_lr_seed7,
    finalize_dynamic_lr_seed7,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    build_tcn_optimizer,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import DynamicTemporalContextTCN


def _production_model() -> DynamicTemporalContextTCN:
    return DynamicTemporalContextTCN(
        feature_count=8,
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        input_steps=480,
        bars_per_day=48,
        dropout=0.0,
        padding_mode="chomp",
        dynamic_attention_hidden=4,
        dynamic_attention_scale=1.0,
    )


def _trial(*, dynamic_learning_rate: float | None) -> TCNTuningTrial:
    return TCNTuningTrial(
        trial_id="dynamic",
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=128,
        model_kind="dynamic_temporal_context",
        padding_mode="chomp",
        bars_per_day=48,
        dynamic_attention_hidden=4,
        dynamic_attention_scale=1.0,
        dynamic_attention_learning_rate=dynamic_learning_rate,
    )


def _small_model() -> DynamicTemporalContextTCN:
    return DynamicTemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
        padding_mode="chomp",
        dynamic_attention_hidden=2,
        dynamic_attention_scale=1.0,
    )


def test_dynamic_attention_optimizer_groups_are_complete_disjoint_and_auditable() -> None:
    model = _production_model()
    dynamic_parameters = model.dynamic_attention_parameters()
    assert len({id(parameter) for parameter in dynamic_parameters}) == len(
        dynamic_parameters
    )
    assert sum(parameter.numel() for parameter in dynamic_parameters) == 176

    bundle = build_tcn_optimizer(model, _trial(dynamic_learning_rate=0.01))
    groups = {
        str(group["group_name"]): (
            float(group["lr"]),
            sum(parameter.numel() for parameter in group["params"]),
        )
        for group in bundle.optimizer.param_groups
    }
    assert groups == {
        "base": (pytest.approx(0.003), 6524),
        "dynamic_attention": (pytest.approx(0.01), 176),
    }
    assert bundle.dynamic_attention_parameter_count == 176
    assert (
        bundle.parameter_group_identity
        == "base-lr-0.003+dynamic-attention-lr-0.01"
    )


def test_dynamic_learning_rate_config_is_explicit_and_fail_closed() -> None:
    raw = {
        "trial_id": "dynamic",
        "model_kind": "dynamic_temporal_context",
        "channels": 16,
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 128,
        "strategy": "smooth_l1",
        "padding_mode": "chomp",
        "bars_per_day": 48,
        "dynamic_attention_hidden": 4,
        "dynamic_attention_scale": 1.0,
        "dynamic_attention_learning_rate": 0.01,
    }
    parsed = parse_real_tcn_trials([raw])[0]
    assert parsed.dynamic_attention_learning_rate == pytest.approx(0.01)
    assert validate_tcn_tuning_plan(
        (parsed,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
    ) == (parsed,)

    invalid_model = TCNTuningTrial(
        **{
            **parsed.__dict__,
            "trial_id": "static",
            "model_kind": "temporal_context",
        }
    )
    with pytest.raises(ContractError, match="only valid for dynamic temporal context"):
        validate_tcn_tuning_plan(
            (invalid_model,),
            input_steps=480,
            max_epochs=8,
            patience=2,
            min_delta=0.002,
        )

    excessive = TCNTuningTrial(
        **{**parsed.__dict__, "dynamic_attention_learning_rate": 0.031}
    )
    with pytest.raises(ContractError, match="no greater than ten times"):
        validate_tcn_tuning_plan(
            (excessive,),
            input_steps=480,
            max_epochs=8,
            patience=2,
            min_delta=0.002,
        )


def test_independent_learning_rate_increases_public_dynamic_weight_movement() -> None:
    torch.manual_seed(91)
    baseline = _small_model()
    torch.manual_seed(91)
    candidate = _small_model()
    inputs = torch.randn(6, 3, 16)
    targets = torch.randn(6, 4)
    initial_sequence = baseline.encode_sequence(inputs)
    initial_day, _ = baseline.dynamic_weights(initial_sequence)

    baseline_trial = _trial(dynamic_learning_rate=None)
    candidate_trial = _trial(dynamic_learning_rate=0.01)
    for model, trial in (
        (baseline, baseline_trial),
        (candidate, candidate_trial),
    ):
        optimizer = build_tcn_optimizer(model, trial).optimizer
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()

    baseline_day, _ = baseline.dynamic_weights(baseline.encode_sequence(inputs))
    candidate_day, _ = candidate.dynamic_weights(candidate.encode_sequence(inputs))
    baseline_movement = torch.linalg.vector_norm(baseline_day - initial_day)
    candidate_movement = torch.linalg.vector_norm(candidate_day - initial_day)
    assert float(candidate_movement.detach()) > 1.5 * float(
        baseline_movement.detach()
    )

    metadata = candidate.receipt_metadata()
    output_weight_l2 = metadata["dynamic_attention_output_weight_l2"]
    output_bias_l2 = metadata["dynamic_attention_output_bias_l2"]
    assert isinstance(output_weight_l2, float) and output_weight_l2 > 0
    assert isinstance(output_bias_l2, float) and output_bias_l2 > 0


def _leaderboard() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        rows.append(
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
                "parameter_count": 6524,
                "dynamic_attention_output_weight_l2": None,
                "optimizer_dynamic_attention_parameter_count": 0,
            }
        )
        rows.append(
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
                "parameter_count": 6700,
                "dynamic_attention_output_weight_l2": 0.2,
                "optimizer_dynamic_attention_parameter_count": 176,
            }
        )
    return pd.DataFrame(rows)


def _attention(variation: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": ["candidate"] * 5,
            "fold": list(range(5)),
            "day_weight_variation": [variation] * 5,
            "intraday_weight_variation": [variation / 2] * 5,
        }
    )


def _dynamic_lr_effect(variation: float) -> DynamicLRSeed7Decision:
    return evaluate_dynamic_lr_seed7(
        _leaderboard(),
        _attention(variation),
        _attention(0.0015),
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
        min_dynamic_attention_output_weight_l2=1e-12,
        min_dynamic_weight_variation=0.002,
        min_parent_variation_ratio=2.0,
        control_parameter_count=6524,
        candidate_parameter_count=6700,
        dynamic_parameter_count=176,
    )


def test_v19_decision_requires_effect_dynamic_capacity_parent_gain_and_speed() -> None:
    effect = _dynamic_lr_effect(0.004)
    assert effect.status == "dynamic_lr_seed7_effect_admitted_v19"
    assert effect.winner_trial_id == "candidate"
    final = finalize_dynamic_lr_seed7(
        effect,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert final.status == "dynamic_lr_seed7_admitted_v19"
    assert final.confirmation_seeds_authorized == (17, 27)

    rejected = _dynamic_lr_effect(0.002)
    assert rejected.status == "stop_dynamic_lr_seed7_effect_v19"
    candidate = rejected.summary.loc[
        rejected.summary["trial_id"].astype(str).eq("candidate")
    ].iloc[0]
    assert "parent_variation_ratio_below_gate" in str(candidate["blockers"])
