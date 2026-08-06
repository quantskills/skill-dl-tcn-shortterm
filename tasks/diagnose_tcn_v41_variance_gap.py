"""Reproduce the broad v40 single-TCN stability gap without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.task_aligned_evaluation import (  # noqa: E402
    evaluate_task_aligned_predictions,
    summarize_task_aligned_metrics,
)


METRICS = (
    "mean_rankic",
    "mean_top_return",
    "mean_top_precision",
    "mean_ndcg_at_top",
    "mean_quantile_monotonicity",
)


def _ensemble_predictions(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    keys = ["fold", "sample_id", "instrument_id", "signal_date", "horizon"]
    fixed = [
        "rank_target",
        "raw_return",
        "stage",
        "sealed",
        "prediction_contract_id",
        "target_contract_id",
        "evaluation_contract_id",
    ]
    selected = predictions.loc[predictions["model"].astype(str).eq(model)]
    if set(selected["seed"].astype(int)) != {7, 17, 27}:
        raise ContractError(f"v41 diagnostic {model} seed coverage drifted")
    aggregations = {column: (column, "first") for column in fixed}
    ensemble = selected.groupby(keys, as_index=False, observed=True).agg(
        score=("score", "mean"), **aggregations
    )
    ensemble["model"] = f"{model}_ensemble"
    ensemble["seed"] = 0
    ensemble["training_contract_id"] = f"{model}-prediction-ensemble-v41"
    return ensemble


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        run_dir = arguments.run_dir.resolve()
        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        single_summary = pd.read_parquet(run_dir / "task-aligned-summary.parquet")
        ensembles = pd.concat(
            [
                _ensemble_predictions(predictions, "relative_tcn"),
                _ensemble_predictions(predictions, "relative_lstm"),
            ],
            ignore_index=True,
        )
        ensemble_summary = summarize_task_aligned_metrics(
            evaluate_task_aligned_predictions(ensembles)
        )
        tcn_single = single_summary.loc[
            single_summary["model"].astype(str).eq("relative_tcn")
        ].iloc[0]
        tcn_ensemble = ensemble_summary.loc[
            ensemble_summary["model"].astype(str).eq("relative_tcn_ensemble")
        ].iloc[0]
        lstm_ensemble = ensemble_summary.loc[
            ensemble_summary["model"].astype(str).eq("relative_lstm_ensemble")
        ].iloc[0]
        deltas = {
            metric: float(tcn_ensemble[metric] - tcn_single[metric])
            for metric in METRICS
        }
        improved_metric_count = sum(value > 0.0 for value in deltas.values())
        broad_variance_gap = bool(
            deltas["mean_rankic"] >= 0.005
            and improved_metric_count >= 4
        )
        payload = {
            "status": (
                "red_seed_variance_gap_v41"
                if broad_variance_gap
                else "no_broad_seed_variance_gap_v41"
            ),
            "red_capable": True,
            "single_tcn": {metric: float(tcn_single[metric]) for metric in METRICS},
            "ensemble_tcn": {
                metric: float(tcn_ensemble[metric]) for metric in METRICS
            },
            "ensemble_lstm": {
                metric: float(lstm_ensemble[metric]) for metric in METRICS
            },
            "ensemble_minus_single_tcn": deltas,
            "improved_metric_count": improved_metric_count,
            "sealed_test_accessed": False,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 1 if broad_variance_gap else 0
    except (ContractError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
