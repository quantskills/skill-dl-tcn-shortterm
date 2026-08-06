from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skill_dl_tcn_shortterm.experiment import ContractError
from skill_dl_tcn_shortterm.sealed_evaluation import (
    bootstrap_paired_daily,
    claim_sealed_consumption,
    complete_sealed_consumption,
    decide_sealed_candidate,
    paired_daily_unit_mean,
    summarize_paired_daily,
)


def _metrics() -> pd.DataFrame:
    rows = []
    for sealed_fold, training_folds in ((0, range(3)), (1, range(5))):
        for seed in (7, 17, 27):
            for training_fold in training_folds:
                evaluation_fold = sealed_fold * 10 + training_fold
                for date_index, date in enumerate(
                    pd.bdate_range("2024-01-02", periods=30)
                ):
                    for model, delta in (("control", 0.0), ("candidate", 0.01)):
                        rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "fold": evaluation_fold,
                                "signal_date": date.strftime("%Y-%m-%d"),
                                "horizon": 1,
                                "rankic": 0.1 + delta,
                                "pearson_ic": 0.1 + delta,
                                "top_return": 0.01 + delta,
                                "top_excess_return": 0.01 + delta,
                                "long_short_spread": 0.01 + delta,
                                "top_precision": 0.1 + delta,
                                "ndcg_at_top": 0.5 + delta,
                                "quantile_monotonicity": 0.1 + delta,
                                "top_turnover": (
                                    np.nan if date_index == 0 else 0.5 - delta
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def _gates() -> dict[str, float]:
    return {
        "min_mean_top_precision_delta": 0.0,
        "min_mean_ndcg_at_top_delta": 0.0,
        "min_primary_tail_ci_low": 0.0,
        "min_secondary_tail_ci_low": -0.002,
        "min_mean_rankic_delta": -0.002,
        "min_top_return_ci_low": -0.0005,
        "min_net_return_after_cost_ci_low": -0.0005,
        "max_mean_top_turnover_delta": 0.02,
        "min_model_step_speed_ratio": 3.0,
        "min_end_to_end_speed_ratio": 3.0,
    }


def test_daily_aggregation_averages_repeated_model_units_before_bootstrap() -> None:
    paired = paired_daily_unit_mean(
        _metrics(),
        reference_model="control",
        candidate_model="candidate",
        one_way_cost_bps=10.0,
    )
    assert len(paired) == 60
    assert np.allclose(paired["rankic"].to_numpy(), 0.01)
    summary = summarize_paired_daily(paired)
    assert summary["mean_top_precision_delta"] == pytest.approx(0.01)
    bootstrap = bootstrap_paired_daily(paired, seed=36, draws=1000)
    assert set(bootstrap["unit_count"]) == {2}
    decision = decide_sealed_candidate(
        summary,
        bootstrap,
        speed={"model_step_speed_ratio": 6.0, "end_to_end_speed_ratio": 5.5},
        gates=_gates(),
    )
    assert decision["status"] == "sealed_confirmed_tcn_candidate_v36"


def test_once_only_claim_cannot_be_retried_even_before_completion(
    tmp_path: Path,
) -> None:
    marker = claim_sealed_consumption(
        tmp_path, freeze_id="a" * 64, sealed_data_sha256="b" * 64
    )
    with pytest.raises(ContractError, match="already been consumed or claimed"):
        claim_sealed_consumption(
            tmp_path, freeze_id="a" * 64, sealed_data_sha256="b" * 64
        )
    complete_sealed_consumption(marker, result_receipt="receipt.json")
    value = json.loads(marker.read_text(encoding="utf-8"))
    assert value["status"] == "completed"
    assert value["result_receipt"] == "receipt.json"


def test_decision_rejects_when_task_aligned_effect_gate_fails() -> None:
    paired = paired_daily_unit_mean(
        _metrics(),
        reference_model="candidate",
        candidate_model="control",
        one_way_cost_bps=10.0,
    )
    summary = summarize_paired_daily(paired)
    bootstrap = bootstrap_paired_daily(paired, seed=36, draws=1000)
    decision = decide_sealed_candidate(
        summary,
        bootstrap,
        speed={"model_step_speed_ratio": 6.0, "end_to_end_speed_ratio": 5.5},
        gates=_gates(),
    )
    assert decision["status"] == "sealed_rejected_tcn_candidate_v36"
    assert decision["candidate_model"] is False
