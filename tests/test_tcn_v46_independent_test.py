from __future__ import annotations

import pandas as pd
import pytest

from skill_dl_tcn_shortterm.experiment import ContractError
from skill_dl_tcn_shortterm.v46_validation import (
    decide_v46_independent_gate,
    validate_v46_window_boundaries,
)


def _bootstrap(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": metric, "bootstrap_ci_low": ci_low}
            for metric, ci_low in rows
        ]
    )


def test_v46_membership_precision_is_diagnostic_not_a_blocker() -> None:
    decision = decide_v46_independent_gate(
        {
            "mean_rankic_delta": 0.01,
            "mean_top_excess_return_delta": 0.0002,
            "mean_ndcg_at_top_delta": 0.002,
            "mean_top_membership_precision_delta": -1.0,
        },
        _bootstrap([("rankic", 0.001)]),
        _bootstrap(
            [
                ("rankic", -0.005),
                ("top_excess_return", -0.0002),
                ("ndcg_at_top", -0.005),
            ]
        ),
        pd.DataFrame(
            {"seed": [7, 17, 27], "rankic_delta": [0.01, 0.01, 0.01]}
        ),
        pd.DataFrame(
            {
                "horizon": [1, 2, 3, 5],
                "rankic_delta": [0.01, 0.01, 0.01, 0.01],
            }
        ),
        contract_valid=True,
        historical_replay=False,
        model_step_speed_ratio=4.8789,
        inference_forward_passes=1,
    )

    assert decision.admitted is True
    assert decision.status == "v46_independent_research_candidate"
    assert not any("membership" in blocker for blocker in decision.blockers)
    assert decision.evidence["top_membership_precision_is_gate"] is False


def test_v46_rejects_reuse_of_a_prior_consumed_window() -> None:
    with pytest.raises(ContractError, match="embargo"):
        validate_v46_window_boundaries(
            evaluation_dates=pd.Series(["2025-03-27", "2025-04-07"]),
            training_dates=pd.Series(["2024-06-11"]),
            prior_consumed_end="2025-03-27",
            embargo_end="2025-04-03",
            expected_start="2025-04-07",
            expected_end="2025-04-07",
        )


def test_v46_membership_precision_cannot_compensate_for_primary_failure() -> None:
    decision = decide_v46_independent_gate(
        {
            "mean_rankic_delta": -0.01,
            "mean_top_excess_return_delta": 0.0002,
            "mean_ndcg_at_top_delta": 0.002,
            "mean_top_membership_precision_delta": 1.0,
        },
        _bootstrap([("rankic", 0.001)]),
        _bootstrap(
            [
                ("rankic", -0.005),
                ("top_excess_return", -0.0002),
                ("ndcg_at_top", -0.005),
            ]
        ),
        pd.DataFrame(
            {"seed": [7, 17, 27], "rankic_delta": [0.01, 0.01, 0.01]}
        ),
        pd.DataFrame(
            {
                "horizon": [1, 2, 3, 5],
                "rankic_delta": [0.01, 0.01, 0.01, 0.01],
            }
        ),
        contract_valid=True,
        historical_replay=False,
        model_step_speed_ratio=4.8789,
        inference_forward_passes=1,
    )

    assert decision.admitted is False
    assert decision.status == "v46_student_not_generalized"
    assert "control_rankic_delta_below_gate" in decision.blockers
    assert not any("membership" in blocker for blocker in decision.blockers)
