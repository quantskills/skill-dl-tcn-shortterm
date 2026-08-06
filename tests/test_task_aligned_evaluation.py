from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing import cast

from skill_dl_tcn_shortterm.experiment import ContractError
from skill_dl_tcn_shortterm.task_aligned_evaluation import (
    compare_task_aligned_models,
    evaluate_task_aligned_predictions,
    validate_prediction_contract,
)


def _prediction_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_date in ("2025-01-02", "2025-01-03", "2025-01-06"):
        raw_returns = np.arange(10, dtype="float64") / 100.0
        # A gets almost the full order right but deliberately misses the best stock.
        full_rank_scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], dtype="float64")
        # B identifies the best stock but scrambles the rest of the cross-section.
        top_tail_scores = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4, 9], dtype="float64")
        rank_target = pd.Series(raw_returns).rank(pct=True).to_numpy() * 2.0 - 1.0
        for model, scores in (
            ("full-rank-winner", full_rank_scores),
            ("top-tail-winner", top_tail_scores),
        ):
            for position in range(10):
                rows.append(
                    {
                        "model": model,
                        "seed": 7,
                        "fold": 0,
                        "sample_id": f"{signal_date}-{position}",
                        "instrument_id": f"stock-{position}",
                        "signal_date": signal_date,
                        "horizon": 1,
                        "score": float(scores[position]),
                        "rank_target": float(rank_target[position]),
                        "raw_return": float(raw_returns[position]),
                        "stage": "validation",
                        "sealed": False,
                        "prediction_contract_id": "cross-sectional-score-v1",
                        "target_contract_id": "next-open-rank-v2",
                        "evaluation_contract_id": "ordinary-validation-multimetric-v33",
                        "training_contract_id": f"{model}-training-v1",
                    }
                )
    return pd.DataFrame(rows)


def test_v33_rejects_cross_model_target_contract_drift() -> None:
    predictions = _prediction_rows()
    predictions.loc[
        predictions["model"].eq("top-tail-winner"), "target_contract_id"
    ] = "different-target-v9"

    with pytest.raises(ContractError, match="target contract"):
        validate_prediction_contract(predictions, expected_models=2)


def test_v33_rejects_missing_or_duplicate_prediction_keys() -> None:
    predictions = _prediction_rows()
    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate"):
        validate_prediction_contract(duplicate, expected_models=2)

    missing = predictions.loc[
        ~(
            predictions["model"].eq("top-tail-winner")
            & predictions["sample_id"].eq("2025-01-02-0")
        )
    ]
    with pytest.raises(ContractError, match="sample coverage"):
        validate_prediction_contract(missing, expected_models=2)


def test_v33_exposes_rankic_and_top_tail_winner_reversal() -> None:
    metrics = evaluate_task_aligned_predictions(_prediction_rows(), top_fraction=0.1)
    comparison = compare_task_aligned_models(
        metrics,
        reference_model="full-rank-winner",
        candidate_model="top-tail-winner",
    )

    assert cast(float, comparison["mean_rankic_delta"]) < 0
    assert cast(float, comparison["mean_top_return_delta"]) > 0
    assert cast(float, comparison["mean_top_precision_delta"]) > 0
    assert comparison["mean_top_membership_precision_delta"] == comparison[
        "mean_top_precision_delta"
    ]
    assert "mean_top_positive_return_rate_delta" in comparison
    assert "mean_top_above_cross_section_mean_rate_delta" in comparison
    assert comparison["winner_consensus"] == "mixed"


def test_v33_rankic_ignores_monotonic_score_scale_but_pearson_does_not() -> None:
    predictions = _prediction_rows()
    base = predictions.loc[predictions["model"].eq("full-rank-winner")].copy()
    transformed = base.copy()
    transformed["model"] = "monotonic-transform"
    transformed["training_contract_id"] = "monotonic-transform-training-v1"
    transformed["score"] = np.exp(transformed["score"].astype(float))
    paired = pd.concat([base, transformed], ignore_index=True)

    metrics = evaluate_task_aligned_predictions(paired, top_fraction=0.1)
    summary = metrics.groupby("model", observed=True).mean(numeric_only=True)
    assert summary.loc["full-rank-winner", "rankic"] == pytest.approx(
        summary.loc["monotonic-transform", "rankic"]
    )
    assert summary.loc["full-rank-winner", "pearson_ic"] != pytest.approx(
        summary.loc["monotonic-transform", "pearson_ic"]
    )


def test_explicit_sealed_mode_requires_test_stage_and_all_sealed() -> None:
    sealed = _prediction_rows().copy()
    sealed["stage"] = "test"
    sealed["sealed"] = True
    validate_prediction_contract(sealed, expected_stage="test", allow_sealed=True)
    metrics = evaluate_task_aligned_predictions(
        sealed, expected_stage="test", allow_sealed=True
    )
    assert set(metrics["stage"]) == {"test"}
    assert metrics["sealed"].all()

    altered = sealed.copy()
    altered.loc[altered.index[0], "sealed"] = False
    with pytest.raises(ContractError, match="requires stage=test"):
        validate_prediction_contract(
            altered, expected_stage="test", allow_sealed=True
        )


def test_top_membership_and_return_hit_diagnostics_have_distinct_semantics() -> None:
    predictions = _prediction_rows().loc[
        lambda rows: rows["model"].eq("top-tail-winner")
        & rows["signal_date"].eq("2025-01-02")
    ].copy()
    predictions["score"] = np.array(
        [9, 10, 8, 7, 6, 5, 4, 3, 2, 1], dtype="float64"
    )

    metric = evaluate_task_aligned_predictions(predictions, top_fraction=0.1).iloc[0]

    assert metric["top_membership_precision"] == 0.0
    assert metric["top_precision"] == metric["top_membership_precision"]
    assert metric["top_positive_return_rate"] == 1.0
    assert metric["top_above_cross_section_mean_rate"] == 0.0
