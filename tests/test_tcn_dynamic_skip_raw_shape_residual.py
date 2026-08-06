from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_dynamic_skip_raw_shape_residual_multiseed,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    build_tcn_trial_model,
    run_tcn_validation_sweep,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import (
    DynamicHorizonSkipTCN,
    HorizonSkipTCN,
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


def test_zero_shape_residual_strictly_preserves_raw_parent_behavior() -> None:
    torch.manual_seed(81)
    control = HorizonSkipTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        dropout=0.0,
        padding_mode="chomp",
    )
    torch.manual_seed(81)
    parent = _parent()
    torch.manual_seed(81)
    candidate = _candidate()
    inputs = torch.randn(6, 3, 16)
    sequences = parent.encode_blocks(inputs)
    tokens = torch.stack([sequence[:, :, -1] for sequence in sequences], dim=1)

    assert torch.equal(control(inputs), parent(inputs))
    assert torch.equal(parent(inputs), candidate(inputs))
    assert torch.equal(
        parent.raw_dynamic_skip_logits(tokens),
        candidate.raw_dynamic_skip_logits(tokens),
    )
    assert torch.count_nonzero(candidate.shape_residual_logits(tokens)) == 0
    candidate_sequences = candidate.encode_blocks(inputs)
    assert torch.equal(
        candidate.dynamic_skip_weights_without_shape_residual(candidate_sequences),
        candidate.dynamic_skip_weights(candidate_sequences),
    )
    assert sum(parameter.numel() for parameter in candidate.parameters()) == (
        sum(parameter.numel() for parameter in parent.parameters()) + 22
    )
    assert sum(
        parameter.numel() for parameter in candidate.shape_residual_parameters()
    ) == 22


def test_shape_residual_is_affine_invariant_and_can_change_dynamic_weights() -> None:
    torch.manual_seed(82)
    candidate = _candidate()
    tokens = torch.tensor(
        [[[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0]]]
    )
    transformed = tokens * 2.0 + 3.0
    assert torch.allclose(
        candidate.shape_residual_inputs(tokens),
        candidate.shape_residual_inputs(transformed),
        atol=2e-5,
        rtol=2e-5,
    )

    inputs = torch.randn(6, 3, 16)
    sequences = candidate.encode_blocks(inputs)
    raw_only = candidate.dynamic_skip_weights_without_shape_residual(sequences)
    torch.nn.init.normal_(candidate.dynamic_skip_shape_output.weight)
    with_shape = candidate.dynamic_skip_weights(sequences)
    assert not torch.allclose(raw_only, with_shape, atol=1e-6, rtol=1e-6)


def test_raw_shape_residual_config_factory_and_validation_fail_closed() -> None:
    raw = {
        "trial_id": "raw-shape-residual",
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
        "dynamic_skip_shape_residual": True,
        "dynamic_skip_shape_residual_scale": 0.25,
    }
    trial = parse_real_tcn_trials([raw])[0]
    assert trial.dynamic_skip_shape_residual is True
    assert trial.dynamic_skip_shape_residual_scale == 0.25
    assert validate_tcn_tuning_plan(
        (trial,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
    ) == (trial,)
    model = build_tcn_trial_model(trial, feature_count=8, input_steps=480)
    assert isinstance(model, ShapeResidualDynamicHorizonSkipTCN)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6436
    metadata = model.receipt_metadata()
    assert metadata["dynamic_skip_parameter_count"] == 176
    assert metadata["dynamic_skip_shape_residual_parameter_count"] == 88

    invalid = TCNTuningTrial(
        **{
            **trial.__dict__,
            "dynamic_skip_token_normalization": "layer_norm",
        }
    )
    with pytest.raises(ContractError, match="requires raw scorer inputs"):
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
            ablation = control - 0.0005
            parent_delta = seed27_parent_delta if seed == 27 else 0.0015
            candidate = parent + parent_delta
            for trial_id, value, parameters in (
                ("control", control, 6260),
                ("parent", parent, 6348),
                ("ablation", ablation, 6352),
            ):
                historical.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "fold": fold,
                        "best_mean_daily_rankic": value,
                        "rankic_1d": value - 0.01,
                        "rankic_2d": value,
                        "rankic_3d": value + 0.005,
                        "rankic_5d": value + 0.01,
                        "parameter_count": parameters,
                    }
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
                    "samples_per_second": 4800.0,
                    "parameter_count": 6436,
                    "dynamic_skip_output_weight_l2": 0.2,
                    "dynamic_skip_shape_output_weight_l2": 0.1,
                    "dynamic_skip_token_normalization": "none",
                    "dynamic_skip_shape_residual": True,
                    "dynamic_skip_shape_residual_scale": 0.25,
                    "dynamic_skip_raw_parameter_count": 88,
                    "dynamic_skip_shape_residual_parameter_count": 88,
                    "dynamic_skip_shape_normalization_parameter_count": 0,
                    "optimizer_group_identity": "all-lr-0.003",
                    "optimizer_dynamic_skip_parameter_count": 0,
                }
            )
    return pd.DataFrame(current), pd.DataFrame(historical)


def _diagnostics(effect: float = 1e-4) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_id": "candidate",
                "seed": seed,
                "fold": fold,
                "block_weight_variation": (fold + 1) * 1e-3,
                "shape_residual_weight_effect_max": effect,
                "simplex_error_max": 1e-7,
            }
            for seed in (7, 17, 27)
            for fold in range(5)
        ]
    )


def _decision(*, seed27_parent_delta: float = 0.0015, speed: float = 3.5):
    current, historical = _evidence(seed27_parent_delta=seed27_parent_delta)
    return evaluate_dynamic_skip_raw_shape_residual_multiseed(
        current,
        historical,
        _diagnostics(),
        {"model_step_speed_ratio": speed, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        parent_candidate_trial_id="parent",
        ablation_trial_id="ablation",
        candidate_trial_id="candidate",
        expected_seeds=(7, 17, 27),
        min_mean_rankic=0.1,
        min_positive_units=15,
        min_mean_rankic_delta=0.003,
        min_parent_mean_rankic_delta=0.001,
        min_ablation_mean_rankic_delta=0.003,
        min_nondegrading_folds_per_seed=3,
        min_horizon_delta_1d=0.0,
        min_horizon_delta_2d=-0.003,
        min_horizon_delta_3d=-0.005,
        min_horizon_delta_5d=-0.005,
        min_median_samples_per_second=4500.0,
        min_dynamic_skip_output_weight_l2=1e-12,
        min_shape_output_weight_l2=1e-12,
        min_shape_residual_weight_effect=1e-6,
        min_block_weight_variation=1e-6,
        max_simplex_error=1e-6,
        control_parameter_count=6260,
        parent_parameter_count=6348,
        ablation_parameter_count=6352,
        candidate_parameter_count=6436,
        dynamic_parameter_count=176,
        raw_parameter_count=88,
        shape_parameter_count=88,
        shape_residual_scale=0.25,
        learning_rate=0.003,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v26_requires_parent_ablation_counterfactual_capacity_and_speed() -> None:
    decision = _decision()
    assert decision.status == (
        "dynamic_skip_raw_shape_residual_multiseed_confirmed_v26"
    )
    assert decision.effect_passed is True
    assert decision.speed_passed is True
    assert float(decision.aggregate["parent_mean_rankic_delta"]) >= 0.001

    parent_regression = _decision(seed27_parent_delta=-0.001)
    assert parent_regression.status == (
        "stop_dynamic_skip_raw_shape_residual_unstable_v26"
    )
    assert "per_seed_parent_mean_delta_not_positive" in str(
        parent_regression.aggregate["blockers"]
    )

    slow = _decision(speed=2.9)
    assert slow.status == "stop_dynamic_skip_raw_shape_residual_speed_v26"
    assert slow.effect_passed is True
    assert slow.speed_passed is False


def test_raw_shape_residual_sweep_publishes_complete_audit_columns() -> None:
    rng = np.random.default_rng(83)
    features = rng.normal(size=(18, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(18),
            "sample_id": [f"residual-{value}" for value in range(18)],
            "signal_date": [f"2025-04-{2 + value // 6:02d}" for value in range(18)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"residual-{sample}",
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
        trial_id="shape-residual-audit",
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
    )
    result = run_tcn_validation_sweep(
        features,
        index,
        labels,
        split,
        trials=(trial,),
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0,
        torch_threads=1,
    )
    row = result.leaderboard.iloc[0]
    assert bool(row["dynamic_skip_shape_residual"]) is True
    assert row["dynamic_skip_shape_residual_scale"] == 0.25
    assert row["dynamic_skip_raw_parameter_count"] == 22
    assert row["dynamic_skip_shape_residual_parameter_count"] == 22
    assert row["dynamic_skip_shape_normalization_parameter_count"] == 0
    assert row["parameter_count"] == 248
