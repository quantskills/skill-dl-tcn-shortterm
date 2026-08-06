from __future__ import annotations

import pandas as pd

from skill_dl_tcn_shortterm.v42_validation import (
    decide_consensus_student_multiseed_gate,
)


def _unit_deltas() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"seed": seed, "fold": fold, "rankic_delta": 0.003}
            for seed in (7, 17, 27)
            for fold in range(5)
        ]
    )


def _comparison() -> dict[str, float]:
    return {
        "mean_rankic_delta": 0.003,
        "mean_pearson_ic_delta": 0.003,
        "mean_top_return_delta": 0.0002,
        "mean_top_precision_delta": 0.001,
        "mean_ndcg_at_top_delta": 0.002,
        "mean_quantile_monotonicity_delta": 0.001,
    }


def test_multiseed_gate_admits_broad_single_model_gain() -> None:
    decision = decide_consensus_student_multiseed_gate(
        _comparison(),
        pd.DataFrame([{"metric": "rankic", "bootstrap_ci_low": 0.001}]),
        _unit_deltas(),
        pd.DataFrame(
            {
                "horizon": [1, 2, 3, 5],
                "rankic_delta": [0.003, 0.004, 0.002, 0.001],
            }
        ),
        model_step_retention=0.98,
        complete_cycle_retention=0.96,
        implied_tcn_lstm_model_step_ratio=4.5,
        inference_forward_passes=1,
    )

    assert decision.admitted is True
    assert decision.status == "consensus_student_multiseed_admitted_v42"
    assert decision.evidence["positive_seed_fold_units"] == 15


def test_multiseed_gate_rejects_local_or_seed_specific_win() -> None:
    units = _unit_deltas()
    units.loc[units["seed"].eq(27), "rankic_delta"] = -0.002
    comparison = _comparison()
    comparison.update(
        {
            "mean_top_return_delta": -0.001,
            "mean_top_precision_delta": -0.01,
            "mean_ndcg_at_top_delta": -0.02,
            "mean_quantile_monotonicity_delta": -0.03,
        }
    )
    decision = decide_consensus_student_multiseed_gate(
        comparison,
        pd.DataFrame([{"metric": "rankic", "bootstrap_ci_low": 0.001}]),
        units,
        pd.DataFrame(
            {
                "horizon": [1, 2, 3, 5],
                "rankic_delta": [0.003, 0.004, 0.002, 0.001],
            }
        ),
        model_step_retention=0.98,
        complete_cycle_retention=0.96,
        implied_tcn_lstm_model_step_ratio=4.5,
        inference_forward_passes=1,
    )

    assert decision.admitted is False
    assert "per_seed_mean_below_gate" in decision.blockers
    assert "broad_metric_count_below_gate" in decision.blockers
