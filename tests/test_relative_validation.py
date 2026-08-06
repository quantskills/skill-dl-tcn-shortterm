from __future__ import annotations

import pandas as pd

from skill_dl_tcn_shortterm.relative_validation import (
    audit_validation_effective_breadth,
    decide_relative_feature_gate,
)


GATES = {
    "min_mean_rankic_delta": 0.002,
    "min_positive_units": 9,
    "min_mean_top_precision_delta": 0.0,
    "min_mean_ndcg_delta": 0.0,
    "min_mean_top_return_delta": -0.0001,
    "max_mean_turnover_delta": 0.02,
    "min_rankic_ci_low": -0.002,
    "min_tcn_speed_retention": 0.9,
}


def _leaderboard(candidate_delta: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": variant,
                "seed": seed,
                "fold": fold,
                "best_mean_daily_rankic": 0.05
                + (candidate_delta if variant == "relative" else 0.0),
            }
            for variant in ["base", "relative"]
            for seed in [7, 17, 27]
            for fold in range(5)
        ]
    )


def _comparison() -> dict[str, float]:
    return {
        "mean_rankic_delta": 0.003,
        "mean_top_precision_delta": 0.01,
        "mean_ndcg_at_top_delta": 0.01,
        "mean_top_return_delta": 0.0002,
        "mean_top_turnover_delta": 0.01,
    }


def _bootstrap(ci_low: float = -0.001) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "rankic",
                "reference_model": "base_tcn",
                "candidate_model": "relative_tcn",
                "bootstrap_ci_low": ci_low,
            }
        ]
    )


def test_relative_feature_gate_admits_complete_stable_gain() -> None:
    decision = decide_relative_feature_gate(
        _leaderboard(0.003),
        _comparison(),
        _bootstrap(),
        seeds=[7, 17, 27],
        folds=range(5),
        base_variant="base",
        candidate_variant="relative",
        base_median_samples_per_second=8_000.0,
        candidate_median_samples_per_second=7_500.0,
        gates=GATES,
    )

    assert decision.status == "relative_features_admitted_v37"
    assert decision.admitted is True
    assert decision.evidence["positive_units"] == 15


def test_relative_feature_gate_reports_effect_and_speed_blockers() -> None:
    comparison = _comparison()
    comparison["mean_top_precision_delta"] = -0.01
    decision = decide_relative_feature_gate(
        _leaderboard(-0.001),
        comparison,
        _bootstrap(-0.003),
        seeds=[7, 17, 27],
        folds=range(5),
        base_variant="base",
        candidate_variant="relative",
        base_median_samples_per_second=8_000.0,
        candidate_median_samples_per_second=6_000.0,
        gates=GATES,
    )

    assert decision.status == "stop_relative_features_no_stable_gain_v37"
    assert "mean_rankic_delta_below_gate" in decision.blockers
    assert "top_precision_delta_below_gate" in decision.blockers
    assert "rankic_ci_low_below_gate" in decision.blockers
    assert "tcn_speed_retention_below_gate" in decision.blockers


def test_relative_feature_gate_supports_preregistered_stage_statuses() -> None:
    decision = decide_relative_feature_gate(
        _leaderboard(0.003),
        _comparison(),
        _bootstrap(),
        seeds=[7, 17, 27],
        folds=range(5),
        base_variant="base",
        candidate_variant="relative",
        base_median_samples_per_second=8_000.0,
        candidate_median_samples_per_second=7_500.0,
        gates=GATES,
        admitted_status="append_relative_sequence_seed7_admitted_v38",
        rejected_status="stop_append_relative_sequence_seed7_v38",
    )

    assert decision.status == "append_relative_sequence_seed7_admitted_v38"


def test_effective_breadth_audit_uses_only_valid_validation_labels() -> None:
    split = pd.DataFrame(
        {
            "sample_position": list(range(35)),
            "fold": [0] * 35,
            "stage": ["validation"] * 35,
            "sealed": [False] * 35,
        }
    )
    labels = pd.DataFrame(
        {
            "sample_position": list(range(35)) * 2,
            "signal_date": ["2025-01-02"] * 70,
            "horizon": [1] * 35 + [5] * 35,
            "valid": [True] * 70,
        }
    )
    evidence = audit_validation_effective_breadth(
        labels,
        split,
        folds=(0,),
        top_fraction=0.1,
        min_top_count=4,
    )
    assert evidence["minimum_member_count"] == 35
    assert evidence["minimum_top_count"] == 4
    assert evidence["effective_breadth_gate_passed"] is True
