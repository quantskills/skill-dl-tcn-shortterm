from __future__ import annotations

import pandas as pd
import pytest

from skill_dl_tcn_shortterm.real_validation import (
    build_tcn_lstm_comparison,
    evaluate_decoupled_residual_seed7,
    evaluate_stabilized_residual_seed7,
    evaluate_signed_multiseed_confirmation,
    finalize_stabilized_residual_seed7,
    finalize_seed7_benchmark_gate,
    finalize_decoupled_residual_seed7,
    parse_real_tcn_trials,
    select_seed7_tcn_candidate,
)


def _tcn_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    settings = {
        "control": ([0.08, 0.09, 0.08, 0.10, 0.09], 6_000.0, 6_228),
        "eligible": ([0.09, 0.10, 0.09, 0.10, 0.09], 5_200.0, 6_260),
        "too-slow": ([0.10, 0.10, 0.10, 0.10, 0.10], 4_900.0, 6_260),
    }
    for trial_id, (rankics, throughput, parameters) in settings.items():
        for fold, rankic in enumerate(rankics):
            rows.append(
                {
                    "trial_id": trial_id,
                    "fold": fold,
                    "seed": 7,
                    "best_mean_daily_rankic": rankic,
                    "samples_per_second": throughput,
                    "model_step_samples_per_second": throughput * 1.2,
                    "parameter_count": parameters,
                }
            )
    return pd.DataFrame(rows)


def test_seed7_selection_requires_one_effect_and_speed_qualified_tcn() -> None:
    decision = select_seed7_tcn_candidate(
        _tcn_rows(),
        control_trial_id="control",
        min_mean_rankic=0.09,
        min_positive_folds=5,
        min_median_samples_per_second=5_000.0,
    )

    assert decision.status == "seed7_winner_admitted_v10"
    assert decision.winner_trial_id == "eligible"
    summary = decision.summary.set_index("trial_id")
    assert bool(summary.loc["eligible", "eligible"]) is True
    assert summary.loc["too-slow", "blockers"] == "throughput_below_gate"


def test_seed7_selection_stops_instead_of_promoting_the_control() -> None:
    rows = _tcn_rows()
    rows.loc[rows["trial_id"].eq("eligible"), "best_mean_daily_rankic"] = 0.089

    decision = select_seed7_tcn_candidate(
        rows,
        control_trial_id="control",
        min_mean_rankic=0.09,
        min_positive_folds=5,
        min_median_samples_per_second=5_000.0,
    )

    assert decision.status == "stop_no_seed7_pareto_v10"
    assert decision.winner_trial_id is None


def test_tcn_lstm_comparison_uses_paired_fold_seed_units() -> None:
    tcn = _tcn_rows().loc[lambda frame: frame["trial_id"].eq("eligible")].copy()
    lstm = pd.DataFrame(
        {
            "model": ["lstm"] * 5,
            "fold": range(5),
            "base_seed": [7] * 5,
            "best_validation_rankic": [0.08] * 5,
            "samples_per_second": [2_600.0] * 5,
            "model_step_samples_per_second": [3_120.0] * 5,
            "parameter_count": [6_124] * 5,
        }
    )

    comparison = build_tcn_lstm_comparison(tcn, lstm)

    assert comparison["paired_unit_count"] == 5
    assert comparison["tcn_mean_rankic"] == pytest.approx(0.094)
    assert comparison["lstm_mean_rankic"] == pytest.approx(0.08)
    assert comparison["paired_mean_rankic_difference"] == pytest.approx(0.014)
    assert comparison["model_step_speed_ratio"] == pytest.approx(2.0)
    assert comparison["end_to_end_speed_ratio"] == pytest.approx(2.0)


def test_real_runner_loads_v9_tcn_fields_without_hidden_defaults() -> None:
    trials = parse_real_tcn_trials(
        [
            {
                "trial_id": "skip-rank",
                "model_kind": "horizon_skip",
                "channels": 16,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "dropout": 0.0,
                "learning_rate": 0.003,
                "batch_size": 128,
                "strategy": "rank_objective",
                "padding_mode": "chomp",
            }
        ]
    )

    assert trials[0].model_kind == "horizon_skip"
    assert trials[0].strategy == "rank_objective"
    assert trials[0].padding_mode == "chomp"


def test_real_runner_loads_local_pcgrad_scope_without_hidden_defaults() -> None:
    trials = parse_real_tcn_trials(
        [
            {
                "trial_id": "skip-local-pcgrad",
                "model_kind": "horizon_skip",
                "channels": 16,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "dropout": 0.0,
                "learning_rate": 0.003,
                "batch_size": 128,
                "strategy": "pcgrad",
                "pcgrad_blocks": [4, 6],
                "pcgrad_horizons": [1, 5],
                "padding_mode": "chomp",
            }
        ]
    )

    assert trials[0].pcgrad_blocks == (4, 6)
    assert trials[0].pcgrad_horizons == (1, 5)


def test_real_runner_loads_temporal_context_soft_rankic_without_hidden_defaults() -> (
    None
):
    trials = parse_real_tcn_trials(
        [
            {
                "trial_id": "context-soft-rankic",
                "model_kind": "temporal_context",
                "channels": 16,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "dropout": 0.0,
                "learning_rate": 0.003,
                "batch_size": 128,
                "strategy": "soft_rankic",
                "padding_mode": "chomp",
                "bars_per_day": 48,
                "soft_rankic_weight": 0.2,
                "soft_rank_temperature": 0.1,
            }
        ]
    )

    assert trials[0].model_kind == "temporal_context"
    assert trials[0].strategy == "soft_rankic"
    assert trials[0].bars_per_day == 48
    assert trials[0].soft_rankic_weight == pytest.approx(0.2)
    assert trials[0].soft_rank_temperature == pytest.approx(0.1)


def test_real_runner_loads_signed_temporal_context_explicitly() -> None:
    trials = parse_real_tcn_trials(
        [
            {
                "trial_id": "signed-context",
                "model_kind": "signed_temporal_context",
                "channels": 16,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "dropout": 0.0,
                "learning_rate": 0.003,
                "batch_size": 128,
                "strategy": "smooth_l1",
                "padding_mode": "chomp",
                "bars_per_day": 48,
            }
        ]
    )

    assert trials[0].model_kind == "signed_temporal_context"
    assert trials[0].bars_per_day == 48


def test_real_runner_loads_stabilized_residual_fields_explicitly() -> None:
    trials = parse_real_tcn_trials(
        [
            {
                "trial_id": "stable-residual",
                "model_kind": "stabilized_temporal_context",
                "channels": 16,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "dropout": 0.0,
                "learning_rate": 0.003,
                "adapter_learning_rate": 0.0003,
                "residual_scale": 0.05,
                "batch_size": 128,
                "strategy": "smooth_l1",
                "padding_mode": "chomp",
                "bars_per_day": 48,
            }
        ]
    )

    assert trials[0].model_kind == "stabilized_temporal_context"
    assert trials[0].adapter_learning_rate == pytest.approx(0.0003)
    assert trials[0].residual_scale == pytest.approx(0.05)


def test_real_runner_loads_decoupled_residual_fields_explicitly() -> None:
    trials = parse_real_tcn_trials(
        [
            {
                "trial_id": "decoupled-residual",
                "model_kind": "decoupled_temporal_context",
                "channels": 16,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
                "dropout": 0.0,
                "learning_rate": 0.003,
                "residual_learning_rate": 0.001,
                "residual_scale": 0.05,
                "batch_size": 128,
                "strategy": "smooth_l1",
                "padding_mode": "chomp",
                "bars_per_day": 48,
            }
        ]
    )

    assert trials[0].model_kind == "decoupled_temporal_context"
    assert trials[0].residual_learning_rate == pytest.approx(0.001)
    assert trials[0].adapter_learning_rate is None
    assert trials[0].residual_scale == pytest.approx(0.05)


def test_final_gate_requires_effect_and_three_x_relative_speed() -> None:
    admitted = select_seed7_tcn_candidate(
        _tcn_rows(),
        control_trial_id="control",
        min_mean_rankic=0.09,
        min_positive_folds=5,
        min_median_samples_per_second=5_000.0,
    )
    passed = finalize_seed7_benchmark_gate(
        admitted,
        {"model_step_speed_ratio": 3.2, "end_to_end_speed_ratio": 3.1},
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert passed.status == "seed7_winner_admitted_v11"
    assert passed.winner_trial_id == "eligible"
    assert passed.confirmation_seeds_authorized == (17, 27)

    too_slow = finalize_seed7_benchmark_gate(
        admitted,
        {"model_step_speed_ratio": 2.9, "end_to_end_speed_ratio": 3.1},
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert too_slow.status == "stop_no_seed7_speed_pareto_v11"
    assert too_slow.winner_trial_id is None
    assert too_slow.confirmation_seeds_authorized == ()


def _multiseed_confirmation_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in [17, 27]:
        for fold in range(5):
            for trial_id, rankic, delta in [
                ("control", 0.091 + fold * 0.001, 0.0),
                ("candidate", 0.096 + fold * 0.001, 0.005),
            ]:
                rows.append(
                    {
                        "trial_id": trial_id,
                        "fold": fold,
                        "seed": seed,
                        "best_mean_daily_rankic": rankic,
                        "rankic_1d": rankic + 0.01,
                        "rankic_2d": rankic + 0.005,
                        "rankic_3d": 0.10 + delta,
                        "rankic_5d": 0.11 + delta,
                        "samples_per_second": 5_300.0,
                        "parameter_count": 6_524,
                    }
                )
    return pd.DataFrame(rows)


def test_signed_multiseed_confirmation_requires_effect_horizons_and_speed() -> None:
    decision = evaluate_signed_multiseed_confirmation(
        _multiseed_confirmation_rows(),
        {
            "model_step_speed_ratio": 3.4,
            "end_to_end_speed_ratio": 3.2,
        },
        control_trial_id="control",
        candidate_trial_id="candidate",
        expected_seeds=(17, 27),
        min_mean_rankic=0.09,
        min_positive_units=10,
        min_nondegrading_folds_per_seed=3,
        min_horizon_delta_3d=-0.005,
        min_horizon_delta_5d=-0.005,
        min_median_samples_per_second=5_000.0,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )

    assert decision.status == "signed_candidate_multiseed_confirmed_v14"
    assert decision.effect_passed is True
    assert decision.speed_passed is True
    assert decision.aggregate["mean_rankic_delta"] == pytest.approx(0.005)
    assert set(decision.seed_summary["seed"]) == {17, 27}


def test_signed_multiseed_confirmation_stops_on_horizon_or_speed_failure() -> None:
    horizon_failure = _multiseed_confirmation_rows()
    candidate = horizon_failure["trial_id"].eq("candidate")
    horizon_failure.loc[candidate, "rankic_5d"] = 0.09
    effect_stopped = evaluate_signed_multiseed_confirmation(
        horizon_failure,
        {"model_step_speed_ratio": 3.4, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        candidate_trial_id="candidate",
        expected_seeds=(17, 27),
        min_mean_rankic=0.09,
        min_positive_units=10,
        min_nondegrading_folds_per_seed=3,
        min_horizon_delta_3d=-0.005,
        min_horizon_delta_5d=-0.005,
        min_median_samples_per_second=5_000.0,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert effect_stopped.status == "stop_signed_candidate_unstable_v14"
    assert effect_stopped.effect_passed is False

    speed_stopped = evaluate_signed_multiseed_confirmation(
        _multiseed_confirmation_rows(),
        {"model_step_speed_ratio": 2.9, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        candidate_trial_id="candidate",
        expected_seeds=(17, 27),
        min_mean_rankic=0.09,
        min_positive_units=10,
        min_nondegrading_folds_per_seed=3,
        min_horizon_delta_3d=-0.005,
        min_horizon_delta_5d=-0.005,
        min_median_samples_per_second=5_000.0,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert speed_stopped.status == "stop_signed_candidate_speed_v14"
    assert speed_stopped.effect_passed is True
    assert speed_stopped.speed_passed is False


def test_signed_multiseed_confirmation_rejects_seed_or_parameter_drift() -> None:
    wrong_seed = _multiseed_confirmation_rows()
    wrong_seed.loc[wrong_seed["seed"].eq(27), "seed"] = 7
    with pytest.raises(Exception, match="seeds"):
        evaluate_signed_multiseed_confirmation(
            wrong_seed,
            {"model_step_speed_ratio": 3.4, "end_to_end_speed_ratio": 3.2},
            control_trial_id="control",
            candidate_trial_id="candidate",
            expected_seeds=(17, 27),
            min_mean_rankic=0.09,
            min_positive_units=10,
            min_nondegrading_folds_per_seed=3,
            min_horizon_delta_3d=-0.005,
            min_horizon_delta_5d=-0.005,
            min_median_samples_per_second=5_000.0,
            min_model_step_speed_ratio=3.0,
            min_end_to_end_speed_ratio=3.0,
        )

    parameter_drift = _multiseed_confirmation_rows()
    parameter_drift.loc[
        parameter_drift["trial_id"].eq("candidate"), "parameter_count"
    ] = 6_525
    with pytest.raises(Exception, match="parameter"):
        evaluate_signed_multiseed_confirmation(
            parameter_drift,
            {"model_step_speed_ratio": 3.4, "end_to_end_speed_ratio": 3.2},
            control_trial_id="control",
            candidate_trial_id="candidate",
            expected_seeds=(17, 27),
            min_mean_rankic=0.09,
            min_positive_units=10,
            min_nondegrading_folds_per_seed=3,
            min_horizon_delta_3d=-0.005,
            min_horizon_delta_5d=-0.005,
            min_median_samples_per_second=5_000.0,
            min_model_step_speed_ratio=3.0,
            min_end_to_end_speed_ratio=3.0,
        )


def _stabilized_seed7_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        settings = [
            ("control", 0.085 + fold * 0.002, 0.080, 0.090, 0.100, 0.110),
            ("stable-same", 0.091 + fold * 0.002, 0.086, 0.080, 0.101, 0.111),
            ("stable-low", 0.094 + fold * 0.002, 0.087, 0.091, 0.101, 0.111),
        ]
        for trial_id, rankic, rankic_1d, rankic_2d, rankic_3d, rankic_5d in settings:
            rows.append(
                {
                    "trial_id": trial_id,
                    "fold": fold,
                    "seed": 7,
                    "best_mean_daily_rankic": rankic,
                    "rankic_1d": rankic_1d,
                    "rankic_2d": rankic_2d,
                    "rankic_3d": rankic_3d,
                    "rankic_5d": rankic_5d,
                    "samples_per_second": 5_300.0,
                    "parameter_count": 6_524,
                }
            )
    return pd.DataFrame(rows)


def test_stabilized_seed7_selects_only_horizon_safe_parameter_matched_arm() -> None:
    decision = evaluate_stabilized_residual_seed7(
        _stabilized_seed7_rows(),
        control_trial_id="control",
        candidate_trial_ids=("stable-same", "stable-low"),
        min_mean_rankic=0.09,
        min_positive_folds=5,
        min_nondegrading_folds=3,
        min_horizon_delta_1d=0.0,
        min_horizon_delta_2d=-0.003,
        min_horizon_delta_3d=-0.005,
        min_horizon_delta_5d=-0.005,
        min_median_samples_per_second=5_000.0,
        required_parameter_count=6_524,
    )

    assert decision.status == "stabilized_residual_seed7_effect_admitted_v15"
    assert decision.winner_trial_id == "stable-low"
    summary = decision.summary.set_index("trial_id")
    assert bool(summary.loc["stable-low", "eligible"]) is True
    assert "horizon_2d_degradation_below_gate" in str(
        summary.loc["stable-same", "blockers"]
    )

    final = finalize_stabilized_residual_seed7(
        decision,
        {"model_step_speed_ratio": 3.4, "end_to_end_speed_ratio": 3.2},
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert final.status == "stabilized_residual_seed7_admitted_v15"
    assert final.confirmation_seeds_authorized == (17, 27)

    speed_failure = finalize_stabilized_residual_seed7(
        decision,
        {"model_step_speed_ratio": 2.9, "end_to_end_speed_ratio": 3.2},
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert speed_failure.status == "stop_stabilized_residual_seed7_speed_v15"
    assert speed_failure.confirmation_seeds_authorized == ()


def test_stabilized_seed7_rejects_seed_and_parameter_drift() -> None:
    wrong_seed = _stabilized_seed7_rows()
    wrong_seed["seed"] = 17
    with pytest.raises(Exception, match="seed 7"):
        evaluate_stabilized_residual_seed7(
            wrong_seed,
            control_trial_id="control",
            candidate_trial_ids=("stable-same", "stable-low"),
            min_mean_rankic=0.09,
            min_positive_folds=5,
            min_nondegrading_folds=3,
            min_horizon_delta_1d=0.0,
            min_horizon_delta_2d=-0.003,
            min_horizon_delta_3d=-0.005,
            min_horizon_delta_5d=-0.005,
            min_median_samples_per_second=5_000.0,
            required_parameter_count=6_524,
        )

    parameter_drift = _stabilized_seed7_rows()
    parameter_drift.loc[
        parameter_drift["trial_id"].eq("stable-low"), "parameter_count"
    ] = 6_525
    with pytest.raises(Exception, match="parameter"):
        evaluate_stabilized_residual_seed7(
            parameter_drift,
            control_trial_id="control",
            candidate_trial_ids=("stable-same", "stable-low"),
            min_mean_rankic=0.09,
            min_positive_folds=5,
            min_nondegrading_folds=3,
            min_horizon_delta_1d=0.0,
            min_horizon_delta_2d=-0.003,
            min_horizon_delta_3d=-0.005,
            min_horizon_delta_5d=-0.005,
            min_median_samples_per_second=5_000.0,
            required_parameter_count=6_524,
        )


def _decoupled_seed7_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        settings = [
            ("control", 6_524, 0.086 + fold * 0.002, 0.080, 0.090, 0.100, 0.110),
            ("decoupled-low", 6_756, 0.091 + fold * 0.002, 0.083, 0.091, 0.101, 0.111),
            ("decoupled-mid", 6_756, 0.094 + fold * 0.002, 0.087, 0.092, 0.102, 0.112),
        ]
        for (
            trial_id,
            parameter_count,
            rankic,
            rankic_1d,
            rankic_2d,
            rankic_3d,
            rankic_5d,
        ) in settings:
            rows.append(
                {
                    "trial_id": trial_id,
                    "fold": fold,
                    "seed": 7,
                    "best_mean_daily_rankic": rankic,
                    "rankic_1d": rankic_1d,
                    "rankic_2d": rankic_2d,
                    "rankic_3d": rankic_3d,
                    "rankic_5d": rankic_5d,
                    "samples_per_second": 5_300.0,
                    "parameter_count": parameter_count,
                }
            )
    return pd.DataFrame(rows)


def test_decoupled_seed7_requires_exact_capacity_effect_horizons_and_speed() -> None:
    decision = evaluate_decoupled_residual_seed7(
        _decoupled_seed7_rows(),
        control_trial_id="control",
        candidate_trial_ids=("decoupled-low", "decoupled-mid"),
        min_mean_rankic=0.09,
        min_positive_folds=5,
        min_nondegrading_folds=3,
        min_horizon_delta_1d=0.0,
        min_horizon_delta_2d=-0.003,
        min_horizon_delta_3d=-0.005,
        min_horizon_delta_5d=-0.005,
        min_median_samples_per_second=5_000.0,
        control_parameter_count=6_524,
        candidate_parameter_count=6_756,
    )

    assert decision.status == "decoupled_residual_seed7_effect_admitted_v16"
    assert decision.winner_trial_id == "decoupled-mid"
    assert set(decision.summary["capacity_delta"]) == {0, 232}
    final = finalize_decoupled_residual_seed7(
        decision,
        {"model_step_speed_ratio": 3.3, "end_to_end_speed_ratio": 3.1},
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert final.status == "decoupled_residual_seed7_admitted_v16"
    assert final.confirmation_seeds_authorized == (17, 27)


def test_decoupled_seed7_rejects_candidate_capacity_drift() -> None:
    rows = _decoupled_seed7_rows()
    rows.loc[rows["trial_id"].eq("decoupled-low"), "parameter_count"] = 6_755
    with pytest.raises(Exception, match="parameter"):
        evaluate_decoupled_residual_seed7(
            rows,
            control_trial_id="control",
            candidate_trial_ids=("decoupled-low", "decoupled-mid"),
            min_mean_rankic=0.09,
            min_positive_folds=5,
            min_nondegrading_folds=3,
            min_horizon_delta_1d=0.0,
            min_horizon_delta_2d=-0.003,
            min_horizon_delta_3d=-0.005,
            min_horizon_delta_5d=-0.005,
            min_median_samples_per_second=5_000.0,
            control_parameter_count=6_524,
            candidate_parameter_count=6_756,
        )
