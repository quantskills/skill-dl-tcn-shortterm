from __future__ import annotations

import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_dynamic_multiscale_multiseed,
)


def _leaderboard(*, seed27_delta: float = 0.006) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in (17, 27):
        delta = 0.006 if seed == 17 else seed27_delta
        for fold in range(5):
            control = 0.09 + 0.002 * fold
            for trial_id, rankic, parameters in (
                ("control", control, 6260),
                ("candidate", control + delta, 6348),
            ):
                rows.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "fold": fold,
                        "best_mean_daily_rankic": rankic,
                        "rankic_1d": rankic - 0.01,
                        "rankic_2d": rankic,
                        "rankic_3d": rankic + 0.005,
                        "rankic_5d": rankic + 0.01,
                        "samples_per_second": 5300.0,
                        "parameter_count": parameters,
                        "dynamic_skip_output_weight_l2": (
                            0.2 if trial_id == "candidate" else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _diagnostics(*, variation: float = 0.002) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_id": "candidate",
                "seed": seed,
                "fold": fold,
                "block_weight_variation": variation,
                "simplex_error_max": 1e-7,
            }
            for seed in (17, 27)
            for fold in range(5)
        ]
    )


def _evaluate(
    *,
    seed27_delta: float = 0.006,
    variation: float = 0.002,
    model_step_speed: float = 3.5,
):
    return evaluate_dynamic_multiscale_multiseed(
        _leaderboard(seed27_delta=seed27_delta),
        _diagnostics(variation=variation),
        {
            "model_step_speed_ratio": model_step_speed,
            "end_to_end_speed_ratio": 3.2,
        },
        control_trial_id="control",
        candidate_trial_id="candidate",
        expected_seeds=(17, 27),
        min_mean_rankic=0.09,
        min_positive_units=10,
        min_mean_rankic_delta=0.003,
        min_nondegrading_folds_per_seed=3,
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
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v21_confirms_only_stable_paired_dynamic_multiscale_gain() -> None:
    decision = _evaluate()
    assert decision.status == "dynamic_multiscale_multiseed_confirmed_v21"
    assert decision.effect_passed is True
    assert decision.speed_passed is True
    assert decision.aggregate["mean_rankic_delta"] == pytest.approx(0.006)
    assert decision.seed_summary["nondegrading_folds"].tolist() == [5, 5]


def test_v21_stops_on_seed_instability_mechanism_or_speed_without_sealed_access() -> None:
    unstable = _evaluate(seed27_delta=-0.002)
    assert unstable.status == "stop_dynamic_multiscale_unstable_v21"
    assert "per_seed_mean_delta_not_positive" in str(
        unstable.aggregate["blockers"]
    )

    inactive = _evaluate(variation=0.0)
    assert inactive.status == "stop_dynamic_multiscale_unstable_v21"
    assert "block_weights_not_sample_conditioned" in str(
        inactive.aggregate["blockers"]
    )

    slow = _evaluate(model_step_speed=2.9)
    assert slow.status == "stop_dynamic_multiscale_speed_v21"
    assert slow.effect_passed is True
    assert slow.speed_passed is False


def test_v21_rejects_incomplete_seed_fold_evidence() -> None:
    incomplete = _leaderboard().iloc[:-1].copy()
    with pytest.raises(ContractError, match="fold coverage"):
        evaluate_dynamic_multiscale_multiseed(
            incomplete,
            _diagnostics(),
            {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
            control_trial_id="control",
            candidate_trial_id="candidate",
            expected_seeds=(17, 27),
            min_mean_rankic=0.09,
            min_positive_units=10,
            min_mean_rankic_delta=0.003,
            min_nondegrading_folds_per_seed=3,
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
            min_model_step_speed_ratio=3.0,
            min_end_to_end_speed_ratio=3.0,
        )
