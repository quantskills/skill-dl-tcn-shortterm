from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_dynamic_skip_shape_amplitude_multiseed,
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
)


def _candidate() -> DynamicHorizonSkipTCN:
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
        dynamic_skip_token_normalization="shape_log_rms",
    )


def test_shape_amplitude_inputs_preserve_shape_and_recover_scale_signal() -> None:
    torch.manual_seed(61)
    control = HorizonSkipTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        dropout=0.0,
        padding_mode="chomp",
    )
    torch.manual_seed(61)
    candidate = _candidate()
    inputs = torch.randn(6, 3, 16)

    assert torch.equal(control(inputs), candidate(inputs))
    assert sum(parameter.numel() for parameter in candidate.parameters()) == (
        sum(parameter.numel() for parameter in control.parameters()) + 24
    )

    tokens = torch.tensor(
        [[[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0]]]
    )
    scaled = tokens * 2.0
    scorer = candidate.dynamic_skip_scorer_inputs(tokens)
    scaled_scorer = candidate.dynamic_skip_scorer_inputs(scaled)
    assert scorer.shape == (1, 2, 5)
    assert torch.allclose(scorer[..., :4], scaled_scorer[..., :4], atol=1e-6)
    assert torch.all(scaled_scorer[..., 4] > scorer[..., 4])
    assert candidate.receipt_metadata()["dynamic_skip_amplitude_feature"] == (
        "log1p_rms"
    )
    assert candidate.receipt_metadata()["dynamic_skip_scorer_input_width"] == 5


def test_shape_amplitude_config_factory_and_validation_fail_closed() -> None:
    raw = {
        "trial_id": "shape-amplitude",
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
        "dynamic_skip_input_normalization": "shape_log_rms",
    }
    trial = parse_real_tcn_trials([raw])[0]
    assert trial.dynamic_skip_token_normalization == "shape_log_rms"
    assert validate_tcn_tuning_plan(
        (trial,), input_steps=480, max_epochs=8, patience=2, min_delta=0.002
    ) == (trial,)
    model = build_tcn_trial_model(trial, feature_count=8, input_steps=480)
    assert isinstance(model, DynamicHorizonSkipTCN)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6352
    assert model.receipt_metadata()["dynamic_skip_parameter_count"] == 92

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
            ablation = control + 0.0005
            parent_delta = seed27_parent_delta if seed == 27 else 0.0015
            candidate = parent + parent_delta
            for trial_id, value, parameters in (
                ("control", control, 6260),
                ("parent", parent, 6348),
                ("ablation", ablation, 6348),
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
                    "samples_per_second": 5300.0,
                    "parameter_count": 6352,
                    "dynamic_skip_output_weight_l2": 0.2,
                    "dynamic_skip_amplitude_projection_weight_l2": 0.3,
                    "dynamic_skip_token_normalization": "shape_log_rms",
                    "dynamic_skip_amplitude_feature": "log1p_rms",
                    "dynamic_skip_scorer_input_width": 17,
                    "dynamic_skip_normalization_parameter_count": 0,
                    "optimizer_group_identity": "all-lr-0.003",
                    "optimizer_dynamic_skip_parameter_count": 0,
                }
            )
    return pd.DataFrame(current), pd.DataFrame(historical)


def _diagnostics(trial_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_id": trial_id,
                "seed": seed,
                "fold": fold,
                "block_weight_variation": (fold + 1) * 1e-3,
                "simplex_error_max": 1e-7,
            }
            for seed in (7, 17, 27)
            for fold in range(5)
        ]
    )


def _decision(*, seed27_parent_delta: float = 0.0015, speed: float = 3.5):
    current, historical = _evidence(seed27_parent_delta=seed27_parent_delta)
    return evaluate_dynamic_skip_shape_amplitude_multiseed(
        current,
        historical,
        _diagnostics("candidate"),
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
        min_median_samples_per_second=5000.0,
        min_dynamic_skip_output_weight_l2=1e-12,
        min_amplitude_projection_weight_l2=1e-12,
        min_block_weight_variation=1e-6,
        max_simplex_error=1e-6,
        control_parameter_count=6260,
        historical_dynamic_parameter_count=88,
        candidate_parameter_count=6352,
        dynamic_parameter_count=92,
        amplitude_parameter_count=4,
        scorer_input_width=17,
        learning_rate=0.003,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v25_requires_raw_parent_ablation_capacity_and_speed_gates() -> None:
    decision = _decision()
    assert decision.status == "dynamic_skip_shape_amplitude_multiseed_confirmed_v25"
    assert decision.effect_passed is True
    assert decision.speed_passed is True
    assert float(decision.aggregate["parent_mean_rankic_delta"]) >= 0.001
    assert float(decision.aggregate["ablation_mean_rankic_delta"]) >= 0.003

    parent_regression = _decision(seed27_parent_delta=-0.001)
    assert parent_regression.status == "stop_dynamic_skip_shape_amplitude_unstable_v25"
    assert "per_seed_parent_mean_delta_not_positive" in str(
        parent_regression.aggregate["blockers"]
    )

    slow = _decision(speed=2.9)
    assert slow.status == "stop_dynamic_skip_shape_amplitude_speed_v25"
    assert slow.effect_passed is True
    assert slow.speed_passed is False


def test_shape_amplitude_sweep_publishes_complete_audit_columns() -> None:
    rng = np.random.default_rng(71)
    features = rng.normal(size=(18, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(18),
            "sample_id": [f"shape-{value}" for value in range(18)],
            "signal_date": [f"2025-03-{2 + value // 6:02d}" for value in range(18)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"shape-{sample}",
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
        trial_id="shape-audit",
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
        dynamic_skip_token_normalization="shape_log_rms",
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
    assert row["dynamic_skip_normalization_parameter_count"] == 0
    assert row["dynamic_skip_amplitude_feature"] == "log1p_rms"
    assert row["dynamic_skip_scorer_input_width"] == 5
    assert row["parameter_count"] == 228
