from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.v9_statistics import audit_rankic_resolution


def _daily_evidence(days: int = 50) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for horizon in [1, 2, 3, 5]:
            for day in range(days):
                control = 0.05 + 0.01 * np.sin(day / 5 + fold)
                delta = 0.003 + 0.001 * np.cos(day / 4 + horizon)
                rows.append(
                    {
                        "fold": fold,
                        "horizon": horizon,
                        "signal_date": f"2025-{fold + 1:02d}-{day + 1:02d}",
                        "control_rankic": control,
                        "candidate_rankic": control + delta,
                        "valid_member_count": 20,
                        "label_overlap_days": horizon - 1,
                        "valid": True,
                        "stage": "validation",
                        "sealed": False,
                    }
                )
    return pd.DataFrame(rows)


def test_rankic_resolution_audit_allows_only_resolvable_ordinary_validation() -> None:
    evidence = _daily_evidence()
    result = audit_rankic_resolution(evidence, seed=7, bootstrap_draws=400)
    replay = audit_rankic_resolution(evidence, seed=7, bootstrap_draws=400)

    assert result.status == "rank_objective_allowed"
    assert result.rank_objective_allowed is True
    assert result.blockers == ()
    assert len(result.summary) == 20
    assert result.summary["paired_date_count"].min() == 50
    assert result.summary["minimum_detectable_effect"].max() <= 0.005
    assert result.summary["degenerate_bootstrap_rate"].max() <= 0.05
    assert set(result.summary["label_overlap_days"]) == {0.0, 1.0, 2.0, 4.0}
    pd.testing.assert_frame_equal(result.summary, replay.summary)


def test_rankic_resolution_audit_blocks_few_dates_and_degenerate_differences() -> None:
    few = audit_rankic_resolution(_daily_evidence(39), seed=7, bootstrap_draws=200)
    assert few.status == "rank_objective_not_resolvable"
    assert "insufficient_paired_dates" in few.blockers

    constant = _daily_evidence()
    constant["candidate_rankic"] = constant["control_rankic"] + 0.003
    degenerate = audit_rankic_resolution(constant, seed=7, bootstrap_draws=200)
    assert degenerate.rank_objective_allowed is False
    assert "degenerate_bootstrap" in degenerate.blockers

    missing_fold = audit_rankic_resolution(
        _daily_evidence().loc[lambda frame: frame["fold"].ne(4)],
        seed=7,
        bootstrap_draws=200,
    )
    assert missing_fold.rank_objective_allowed is False
    assert "missing_fold_horizon_units" in missing_fold.blockers


def test_rankic_resolution_audit_fails_closed_on_test_or_sealed_rows() -> None:
    test_rows = _daily_evidence()
    test_rows.loc[0, "stage"] = "test"
    with pytest.raises(ContractError, match="ordinary validation"):
        audit_rankic_resolution(test_rows, seed=7)

    sealed = _daily_evidence()
    sealed.loc[0, "sealed"] = True
    with pytest.raises(ContractError, match="sealed"):
        audit_rankic_resolution(sealed, seed=7)
