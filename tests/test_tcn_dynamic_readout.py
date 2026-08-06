from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_readout import (
    evaluate_dynamic_readout_seed7,
    finalize_dynamic_readout_seed7,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    build_tcn_trial_model,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import (
    DynamicTemporalContextTCN,
    TemporalContextTCN,
)


def _control_model() -> TemporalContextTCN:
    return TemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
        padding_mode="chomp",
    )


def _dynamic_model() -> DynamicTemporalContextTCN:
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


def _raw_trial(*, model_kind: str = "dynamic_temporal_context") -> dict[str, object]:
    value: dict[str, object] = {
        "trial_id": "candidate",
        "model_kind": model_kind,
        "channels": 4,
        "kernel_size": 2,
        "dilations": [1, 2, 4, 8],
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 8,
        "strategy": "smooth_l1",
        "padding_mode": "chomp",
        "bars_per_day": 4,
    }
    if model_kind == "dynamic_temporal_context":
        value.update(
            {"dynamic_attention_hidden": 2, "dynamic_attention_scale": 1.0}
        )
    return value


def test_dynamic_readout_is_initially_control_equivalent_and_parameter_bounded() -> None:
    torch.manual_seed(71)
    control = _control_model().eval()
    torch.manual_seed(71)
    candidate = _dynamic_model().eval()
    inputs = torch.randn(5, 3, 16)

    torch.testing.assert_close(candidate(inputs), control(inputs), rtol=0, atol=0)
    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    candidate_parameters = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    assert candidate_parameters - control_parameters == 44

    production_control = TemporalContextTCN(
        feature_count=8,
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        input_steps=480,
        bars_per_day=48,
        dropout=0.0,
        padding_mode="chomp",
    )
    production_candidate = DynamicTemporalContextTCN(
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
    assert sum(p.numel() for p in production_control.parameters()) == 6524
    assert sum(p.numel() for p in production_candidate.parameters()) == 6700


def test_dynamic_weights_are_sample_conditioned_simplexes_and_causal() -> None:
    torch.manual_seed(72)
    model = _dynamic_model()
    inputs = torch.randn(2, 3, 16)
    optimizer = torch.optim.SGD(model.parameters(), lr=2.0)
    for _ in range(2):
        sequence = model.encode_sequence(inputs)
        day, intraday = model.dynamic_weights(sequence)
        loss = -day[0, 0, 0] - day[1, 0, 1] - intraday[0, 1, 0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    sequence = model.encode_sequence(inputs)
    day, intraday = model.dynamic_weights(sequence)
    assert day.shape == (2, 4, 4)
    assert intraday.shape == (2, 4, 4)
    assert bool(torch.isfinite(day).all() and torch.isfinite(intraday).all())
    torch.testing.assert_close(day.sum(dim=2), torch.ones(2, 4))
    torch.testing.assert_close(intraday.sum(dim=2), torch.ones(2, 4))
    assert not torch.allclose(day[0], day[1])

    changed = inputs.clone()
    changed[:, :, 12:] += 100.0
    original_sequence = model.encode_sequence(inputs)
    changed_sequence = model.encode_sequence(changed)
    torch.testing.assert_close(original_sequence[:, :, :12], changed_sequence[:, :, :12])

    metadata = model.receipt_metadata()
    assert metadata["readout"] == "horizon_dual_scale_stock_conditioned_attention"
    assert metadata["dynamic_attention_hidden"] == 2
    assert metadata["dynamic_attention_scale"] == pytest.approx(1.0)
    assert metadata["dynamic_attention_parameter_count"] == 44
    output_l2 = metadata["dynamic_attention_output_l2"]
    assert isinstance(output_l2, float)
    assert output_l2 > 0


def test_dynamic_trial_parser_factory_and_validation_fail_closed() -> None:
    missing = _raw_trial()
    missing.pop("dynamic_attention_hidden")
    with pytest.raises(ContractError, match="dynamic attention"):
        parse_real_tcn_trials([missing])

    trial = parse_real_tcn_trials([_raw_trial()])[0]
    assert trial.dynamic_attention_hidden == 2
    assert trial.dynamic_attention_scale == pytest.approx(1.0)
    validated = validate_tcn_tuning_plan(
        (trial,), input_steps=16, max_epochs=3, patience=1, min_delta=0.0
    )
    model = build_tcn_trial_model(
        validated[0], feature_count=3, input_steps=16
    )
    assert isinstance(model, DynamicTemporalContextTCN)

    invalid = parse_real_tcn_trials(
        [{**_raw_trial(), "dynamic_attention_scale": 0.0}]
    )[0]
    with pytest.raises(ContractError, match="dynamic attention scale"):
        validate_tcn_tuning_plan(
            (invalid,), input_steps=16, max_epochs=3, patience=1, min_delta=0.0
        )


def _leaderboard() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        rows.append(
            {
                "trial_id": "control",
                "fold": fold,
                "seed": 7,
                "best_mean_daily_rankic": 0.087 + fold * 0.0001,
                "rankic_1d": 0.08,
                "rankic_2d": 0.085,
                "rankic_3d": 0.09,
                "rankic_5d": 0.093,
                "samples_per_second": 5700.0,
                "parameter_count": 6524,
                "dynamic_attention_output_l2": np.nan,
            }
        )
        rows.append(
            {
                "trial_id": "candidate",
                "fold": fold,
                "seed": 7,
                "best_mean_daily_rankic": 0.094 + fold * 0.0001,
                "rankic_1d": 0.084,
                "rankic_2d": 0.089,
                "rankic_3d": 0.094,
                "rankic_5d": 0.097,
                "samples_per_second": 5300.0,
                "parameter_count": 6700,
                "dynamic_attention_output_l2": 0.02,
            }
        )
    return pd.DataFrame(rows)


def _diagnostics(variation: float = 0.01) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": ["candidate"] * 5,
            "fold": list(range(5)),
            "day_weight_variation": [variation] * 5,
            "intraday_weight_variation": [variation / 2] * 5,
        }
    )


def _effect(variation: float = 0.01):
    return evaluate_dynamic_readout_seed7(
        _leaderboard(),
        _diagnostics(variation),
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
        min_dynamic_attention_output_l2=1e-12,
        min_dynamic_weight_variation=1e-6,
        control_parameter_count=6524,
        candidate_parameter_count=6700,
    )


def test_dynamic_effect_and_speed_gates_require_mechanism_use() -> None:
    admitted = _effect()
    assert admitted.winner_trial_id == "candidate"
    assert admitted.status == "dynamic_readout_seed7_effect_admitted_v18"
    final = finalize_dynamic_readout_seed7(
        admitted,
        {"model_step_speed_ratio": 3.4, "end_to_end_speed_ratio": 3.2},
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert final.status == "dynamic_readout_seed7_admitted_v18"
    assert final.confirmation_seeds_authorized == (17, 27)

    rejected = _effect(variation=0.0)
    assert rejected.winner_trial_id is None
    assert rejected.status == "stop_dynamic_readout_seed7_effect_v18"
    candidate = rejected.summary.loc[
        rejected.summary["trial_id"].astype(str).eq("candidate")
    ].iloc[0]
    assert "dynamic_weights_not_sample_conditioned" in str(candidate["blockers"])
