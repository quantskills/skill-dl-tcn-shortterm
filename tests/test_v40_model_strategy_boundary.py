from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from skill_dl_tcn_shortterm.backtest import build_executable_long_only
from skill_dl_tcn_shortterm.v40_validation import (
    decide_v40_strategy_gate,
    decide_v40_model_gate,
    frozen_training_contracts,
    summarize_cost_sensitivity,
    validate_v40_frozen_predictions,
)
from skill_dl_tcn_shortterm.experiment import ContractError


def _leaderboard() -> pd.DataFrame:
    rows = []
    for fold, base, candidate in [(0, 0.05, 0.06), (1, 0.04, 0.045)]:
        rows.extend(
            [
                {
                    "variant": "base",
                    "seed": 7,
                    "fold": fold,
                    "best_mean_daily_rankic": base,
                },
                {
                    "variant": "relative",
                    "seed": 7,
                    "fold": fold,
                    "best_mean_daily_rankic": candidate,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_v40_model_gate_keeps_membership_turnover_as_diagnostic_only() -> None:
    comparison = {
        "mean_rankic_delta": 0.0075,
        "mean_top_precision_delta": 0.008,
        "mean_ndcg_at_top_delta": 0.006,
        "mean_top_return_delta": 0.0002,
        "mean_top_turnover_delta": 0.25,
    }
    bootstrap = pd.DataFrame(
        [
            {
                "metric": "rankic",
                "reference_model": "base_tcn",
                "candidate_model": "relative_tcn",
                "bootstrap_ci_low": 0.001,
            }
        ]
    )

    decision = decide_v40_model_gate(
        _leaderboard(),
        comparison,
        bootstrap,
        seeds=(7,),
        folds=(0, 1),
        base_variant="base",
        candidate_variant="relative",
        base_median_samples_per_second=6000.0,
        candidate_median_samples_per_second=5900.0,
        gates={
            "min_mean_rankic_delta": 0.002,
            "min_positive_units": 2,
            "min_mean_top_precision_delta": 0.0,
            "min_mean_ndcg_delta": 0.0,
            "min_mean_top_return_delta": -0.0001,
            "min_rankic_ci_low": -0.002,
            "min_tcn_speed_retention": 0.9,
        },
    )

    assert decision.admitted is True
    assert decision.status == "top50_relative_model_seed7_admitted_v40"
    assert decision.blockers == ()
    assert decision.evidence["membership_turnover_diagnostic_delta"] == 0.25
    assert decision.evidence["membership_turnover_is_model_gate"] is False


def test_cost_sensitivity_uses_buys_plus_sells_not_membership_turnover() -> None:
    prediction_rows = []
    label_rows = []
    for date_number, signal_date in enumerate(["2024-01-02", "2024-01-03"]):
        for instrument_number in range(10):
            sample_id = f"{signal_date}-I{instrument_number}"
            score = float(instrument_number)
            if date_number == 1:
                score = float(9 - instrument_number)
            prediction_rows.append(
                {
                    "model": "tcn",
                    "fold": 0,
                    "sample_id": sample_id,
                    "instrument_id": f"I{instrument_number}",
                    "signal_date": signal_date,
                    "horizon": 1,
                    "score": score,
                }
            )
            entry = pd.Timestamp("2024-01-03", tz="Asia/Shanghai") + pd.Timedelta(
                days=date_number
            )
            label_rows.append(
                {
                    "sample_id": sample_id,
                    "signal_date": signal_date,
                    "horizon": 1,
                    "entry_at": entry + pd.Timedelta(hours=9, minutes=30),
                    "label_end_at": entry + pd.Timedelta(days=1, hours=9, minutes=30),
                    "raw_return": 0.01,
                    "valid": True,
                }
            )
    result = build_executable_long_only(
        pd.DataFrame(prediction_rows), pd.DataFrame(label_rows), top_fraction=0.10
    )

    sensitivity = summarize_cost_sensitivity(result, cost_bps=(10.0,))

    row = sensitivity.iloc[0]
    assert row["cumulative_traded_notional_turnover"] == pytest.approx(4.0)
    assert row["transaction_cost"] == pytest.approx(0.004)
    assert row["cumulative_net_return_contribution"] == pytest.approx(
        row["cumulative_gross_return_contribution"] - 0.004
    )
    incomplete = replace(
        result, metrics=result.metrics.drop(columns=["vintage_count"])
    )
    with pytest.raises(ContractError, match="vintage_count"):
        summarize_cost_sensitivity(incomplete, cost_bps=(10.0,))


def test_strategy_gate_is_separate_and_uses_executable_turnover() -> None:
    summary = pd.DataFrame(
        [
            {
                "policy": "raw_topk",
                "model": "base_tcn",
                "fold": 0,
                "horizon": 1,
                "one_way_cost_bps": 10.0,
                "mean_one_way_turnover": 0.40,
                "mean_net_return_contribution": 0.0010,
            },
            {
                "policy": "raw_topk",
                "model": "relative_tcn",
                "fold": 0,
                "horizon": 1,
                "one_way_cost_bps": 10.0,
                "mean_one_way_turnover": 0.41,
                "mean_net_return_contribution": 0.0012,
            },
        ]
    )

    decision = decide_v40_strategy_gate(
        summary,
        policy="raw_topk",
        reference_model="base_tcn",
        candidate_model="relative_tcn",
        one_way_cost_bps=10.0,
        max_mean_one_way_turnover_delta=0.02,
        min_mean_net_return_delta=-0.0001,
    )

    assert decision.admitted is True
    assert decision.status == "portfolio_research_admitted_v40"
    assert decision.evidence["mean_one_way_turnover_delta"] == pytest.approx(0.01)
    assert decision.evidence["mean_net_return_delta"] == pytest.approx(0.0002)


def test_strategy_gate_pairs_multiseed_units_without_false_duplicates() -> None:
    rows = []
    for seed in (7, 17):
        for model, turnover, net in (
            ("base_tcn", 0.40, 0.0010),
            ("relative_tcn", 0.41, 0.0012),
        ):
            rows.append(
                {
                    "policy": "raw_topk",
                    "model": model,
                    "seed": seed,
                    "fold": 0,
                    "horizon": 1,
                    "one_way_cost_bps": 10.0,
                    "mean_one_way_turnover": turnover,
                    "mean_net_return_contribution": net,
                }
            )

    decision = decide_v40_strategy_gate(
        pd.DataFrame(rows),
        policy="raw_topk",
        reference_model="base_tcn",
        candidate_model="relative_tcn",
        one_way_cost_bps=10.0,
        max_mean_one_way_turnover_delta=0.02,
        min_mean_net_return_delta=-0.0001,
    )

    assert decision.admitted is True
    assert decision.evidence["unit_count"] == 2
    assert set(decision.unit_deltas["seed"]) == {7, 17}


def test_v40_frozen_prediction_contract_rejects_sealed_rows() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "base_tcn",
                "seed": 7,
                "fold": 0,
                "sample_id": "s1",
                "instrument_id": "A",
                "signal_date": "2024-01-01",
                "horizon": 1,
                "score": 0.1,
                "stage": "validation",
                "sealed": True,
            },
            {
                "model": "relative_tcn",
                "seed": 7,
                "fold": 0,
                "sample_id": "s1",
                "instrument_id": "A",
                "signal_date": "2024-01-01",
                "horizon": 1,
                "score": 0.2,
                "stage": "validation",
                "sealed": True,
            },
        ]
    )

    with pytest.raises(ContractError, match="non-sealed validation"):
        validate_v40_frozen_predictions(
            rows,
            expected_models=("base_tcn", "relative_tcn"),
            expected_seeds=(7,),
            expected_folds=(0,),
            expected_horizons=(1,),
        )


def test_parent_training_contract_must_be_unique_before_phase_b_training() -> None:
    parent = pd.DataFrame(
        {
            "model": ["base_tcn", "base_tcn", "relative_tcn"],
            "training_contract_id": ["base-v39", "base-v40", "relative-v39"],
        }
    )

    with pytest.raises(ContractError, match="multiple training contracts"):
        frozen_training_contracts(
            parent, expected_models=("base_tcn", "relative_tcn")
        )
