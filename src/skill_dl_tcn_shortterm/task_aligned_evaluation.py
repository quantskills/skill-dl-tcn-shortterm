"""Fail-closed, task-aligned evaluation for cross-sectional model scores.

RankIC remains the primary statistical metric for the repository's full-pool
ranking task.  This module adds top-tail and raw-return diagnostics so a model
cannot be declared economically superior from RankIC alone.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from .experiment import ContractError
from .v9_statistics import _paired_block_bootstrap


PREDICTION_KEY = ["model", "seed", "fold", "sample_id", "horizon"]
PAIRING_KEY = ["seed", "fold", "sample_id", "horizon"]
GROUP_KEY = ["model", "seed", "fold", "signal_date", "horizon"]
METRIC_PAIRING_KEY = ["seed", "fold", "signal_date", "horizon"]
SHARED_CONTRACT_COLUMNS = [
    "prediction_contract_id",
    "target_contract_id",
    "evaluation_contract_id",
]
REQUIRED_PREDICTION_COLUMNS = {
    *PREDICTION_KEY,
    "instrument_id",
    "signal_date",
    "score",
    "rank_target",
    "raw_return",
    "stage",
    "sealed",
    *SHARED_CONTRACT_COLUMNS,
    "training_contract_id",
}


def validate_prediction_contract(
    predictions: pd.DataFrame,
    *,
    expected_models: int | None = None,
    expected_stage: str = "validation",
    allow_sealed: bool = False,
) -> None:
    """Validate that model scores are semantically and sample-wise comparable."""

    if missing := sorted(REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)):
        raise ContractError(
            "task-aligned predictions missing columns: " + ", ".join(missing)
        )
    if predictions.empty:
        raise ContractError("task-aligned predictions cannot be empty")
    models = tuple(sorted(predictions["model"].astype(str).unique()))
    if expected_models is not None and len(models) != expected_models:
        raise ContractError(
            f"expected {expected_models} models but observed {len(models)}"
        )
    if predictions.duplicated(PREDICTION_KEY).any():
        raise ContractError("task-aligned predictions contain duplicate keys")
    if set(predictions["stage"].astype(str)) != {expected_stage}:
        raise ContractError(f"task-aligned evaluation stage must equal {expected_stage}")
    sealed = predictions["sealed"].astype(bool)
    if allow_sealed:
        if not sealed.all() or expected_stage != "test":
            raise ContractError("sealed evaluation requires stage=test and sealed=true")
    elif sealed.any():
        raise ContractError("task-aligned evaluation cannot access sealed rows")

    friendly_names = {
        "prediction_contract_id": "prediction contract",
        "target_contract_id": "target contract",
        "evaluation_contract_id": "evaluation contract",
    }
    for column in SHARED_CONTRACT_COLUMNS:
        values = predictions[column].astype(str)
        if values.isna().any() or values.nunique(dropna=False) != 1:
            raise ContractError(f"cross-model {friendly_names[column]} drifted")
    training_counts = predictions.groupby("model", observed=True)[
        "training_contract_id"
    ].nunique(dropna=False)
    if not training_counts.eq(1).all():
        raise ContractError("a model has multiple training contracts")

    numeric = predictions[["score", "rank_target", "raw_return"]].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(numeric).all():
        raise ContractError("task-aligned predictions contain non-finite values")

    reference_model = models[0]
    reference = predictions.loc[
        predictions["model"].astype(str).eq(reference_model)
    ].sort_values(PAIRING_KEY, kind="mergesort")
    reference_index = pd.MultiIndex.from_frame(reference[PAIRING_KEY])
    label_columns = [
        "instrument_id",
        "signal_date",
        "rank_target",
        "raw_return",
    ]
    reference_labels = reference.set_index(PAIRING_KEY)[label_columns].sort_index()
    for model in models[1:]:
        candidate = predictions.loc[
            predictions["model"].astype(str).eq(model)
        ].sort_values(PAIRING_KEY, kind="mergesort")
        candidate_index = pd.MultiIndex.from_frame(candidate[PAIRING_KEY])
        if not candidate_index.equals(reference_index):
            raise ContractError("cross-model sample coverage drifted")
        candidate_labels = candidate.set_index(PAIRING_KEY)[label_columns].sort_index()
        if not candidate_labels[["instrument_id", "signal_date"]].equals(
            reference_labels[["instrument_id", "signal_date"]]
        ):
            raise ContractError("cross-model sample identity drifted")
        for column in ("rank_target", "raw_return"):
            if not np.allclose(
                candidate_labels[column].to_numpy(dtype="float64"),
                reference_labels[column].to_numpy(dtype="float64"),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ContractError(f"cross-model {column} values drifted")


def _correlation(
    left: pd.Series,
    right: pd.Series,
    *,
    method: Literal["pearson", "spearman"],
) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right, method=method))


def _ndcg_at_k(group: pd.DataFrame, top_count: int) -> float:
    relevance = np.clip(
        (group["rank_target"].to_numpy(dtype="float64") + 1.0) / 2.0,
        0.0,
        1.0,
    )
    score_order = np.argsort(
        -group["score"].to_numpy(dtype="float64"), kind="mergesort"
    )[:top_count]
    ideal_order = np.argsort(-relevance, kind="mergesort")[:top_count]
    discounts = 1.0 / np.log2(np.arange(2, top_count + 2, dtype="float64"))
    observed = float(np.sum(relevance[score_order] * discounts))
    ideal = float(np.sum(relevance[ideal_order] * discounts))
    return observed / ideal if ideal > 0 else float("nan")


def _quantile_monotonicity(group: pd.DataFrame) -> float:
    bucket_count = min(10, len(group))
    if bucket_count < 3 or group["score"].nunique() < 2:
        return float("nan")
    score_rank = group["score"].rank(method="first", pct=True)
    buckets = np.minimum(
        (score_rank.to_numpy(dtype="float64") * bucket_count).astype(int),
        bucket_count - 1,
    )
    means = (
        pd.DataFrame(
            {
                "bucket": buckets,
                "raw_return": group["raw_return"].to_numpy(dtype="float64"),
            }
        )
        .groupby("bucket", observed=True)["raw_return"]
        .mean()
    )
    return _correlation(
        pd.Series(means.index.to_numpy(dtype="float64")),
        means.reset_index(drop=True),
        method="spearman",
    )


def evaluate_task_aligned_predictions(
    predictions: pd.DataFrame,
    *,
    top_fraction: float = 0.1,
    expected_stage: str = "validation",
    allow_sealed: bool = False,
) -> pd.DataFrame:
    """Evaluate full-order ranking and long-only top-tail diagnostics together."""

    if not 0 < top_fraction <= 0.5:
        raise ContractError("top_fraction must be in (0, 0.5]")
    validate_prediction_contract(
        predictions,
        expected_stage=expected_stage,
        allow_sealed=allow_sealed,
    )
    rows: list[dict[str, object]] = []
    top_sets: dict[tuple[str, int, int, str, int], set[str]] = {}
    for key_values, group in predictions.groupby(GROUP_KEY, observed=True, sort=True):
        model, seed, fold, signal_date, horizon = key_values
        group = group.sort_values(
            ["score", "instrument_id"], ascending=[False, True], kind="mergesort"
        )
        member_count = len(group)
        top_count = max(1, int(ceil(member_count * top_fraction)))
        predicted_top = group.head(top_count)
        predicted_bottom = group.tail(top_count)
        realized_top = group.sort_values(
            ["raw_return", "instrument_id"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top_count)
        predicted_ids = {
            str(value) for value in predicted_top["instrument_id"].tolist()
        }
        realized_ids = {
            str(value) for value in realized_top["instrument_id"].tolist()
        }
        top_sets[
            (
                str(model),
                int(cast(Any, seed)),
                int(cast(Any, fold)),
                str(signal_date),
                int(cast(Any, horizon)),
            )
        ] = predicted_ids
        cross_section_mean = float(group["raw_return"].mean())
        top_return = float(predicted_top["raw_return"].mean())
        bottom_return = float(predicted_bottom["raw_return"].mean())
        top_membership_precision = (
            len(predicted_ids.intersection(realized_ids)) / top_count
        )
        rows.append(
            {
                "model": str(model),
                "seed": int(cast(Any, seed)),
                "fold": int(cast(Any, fold)),
                "signal_date": str(signal_date),
                "horizon": int(cast(Any, horizon)),
                "member_count": member_count,
                "top_count": top_count,
                "rankic": _correlation(
                    group["score"], group["rank_target"], method="spearman"
                ),
                "pearson_ic": _correlation(
                    group["score"], group["raw_return"], method="pearson"
                ),
                "top_return": top_return,
                "cross_section_mean_return": cross_section_mean,
                "top_excess_return": top_return - cross_section_mean,
                "bottom_return": bottom_return,
                "long_short_spread": top_return - bottom_return,
                # Historical compatibility alias.  This is set overlap, not a
                # positive-return hit rate.
                "top_precision": top_membership_precision,
                "top_membership_precision": top_membership_precision,
                "top_positive_return_rate": float(
                    (predicted_top["raw_return"] > 0.0).mean()
                ),
                "top_above_cross_section_mean_rate": float(
                    (predicted_top["raw_return"] > cross_section_mean).mean()
                ),
                "ndcg_at_top": _ndcg_at_k(group, top_count),
                "quantile_monotonicity": _quantile_monotonicity(group),
                "top_turnover": float("nan"),
                "stage": expected_stage,
                "sealed": allow_sealed,
                "prediction_contract_id": str(
                    group["prediction_contract_id"].iloc[0]
                ),
                "target_contract_id": str(group["target_contract_id"].iloc[0]),
                "evaluation_contract_id": str(
                    group["evaluation_contract_id"].iloc[0]
                ),
                "training_contract_id": str(
                    group["training_contract_id"].iloc[0]
                ),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(GROUP_KEY, kind="mergesort")
    if metrics.empty or metrics.duplicated(GROUP_KEY).any():
        raise ContractError("task-aligned metric groups are empty or duplicated")

    for _, group_indices in metrics.groupby(
        ["model", "seed", "fold", "horizon"], observed=True, sort=True
    ).groups.items():
        ordered_indices = metrics.loc[list(group_indices)].sort_values(
            "signal_date", kind="mergesort"
        ).index
        previous: set[str] | None = None
        for index in ordered_indices:
            row = metrics.loc[index]
            key = (
                str(row["model"]),
                int(row["seed"]),
                int(row["fold"]),
                str(row["signal_date"]),
                int(row["horizon"]),
            )
            current = top_sets[key]
            if previous is not None:
                denominator = max(1, min(len(previous), len(current)))
                metrics.loc[index, "top_turnover"] = 1.0 - (
                    len(previous.intersection(current)) / denominator
                )
            previous = current
    return metrics.reset_index(drop=True)


def compare_task_aligned_models(
    metrics: pd.DataFrame,
    *,
    reference_model: str,
    candidate_model: str,
) -> dict[str, float | int | str]:
    """Return paired deltas without collapsing conflicting task preferences."""

    required = {
        *GROUP_KEY,
        "rankic",
        "pearson_ic",
        "top_return",
        "top_excess_return",
        "long_short_spread",
        "top_precision",
        "top_membership_precision",
        "top_positive_return_rate",
        "top_above_cross_section_mean_rate",
        "ndcg_at_top",
        "quantile_monotonicity",
        "top_turnover",
    }
    if missing := sorted(required.difference(metrics.columns)):
        raise ContractError(
            "task-aligned metrics missing columns: " + ", ".join(missing)
        )
    selected = metrics.loc[
        metrics["model"].astype(str).isin([reference_model, candidate_model])
    ].copy()
    if set(selected["model"].astype(str)) != {reference_model, candidate_model}:
        raise ContractError("task-aligned comparison model coverage is incomplete")
    if selected.duplicated(GROUP_KEY).any():
        raise ContractError("task-aligned comparison contains duplicate groups")
    value_columns = [
        "rankic",
        "pearson_ic",
        "top_return",
        "top_excess_return",
        "long_short_spread",
        "top_precision",
        "top_membership_precision",
        "top_positive_return_rate",
        "top_above_cross_section_mean_rate",
        "ndcg_at_top",
        "quantile_monotonicity",
        "top_turnover",
    ]
    reference = selected.loc[
        selected["model"].astype(str).eq(reference_model)
    ].set_index(METRIC_PAIRING_KEY)[value_columns].sort_index()
    candidate = selected.loc[
        selected["model"].astype(str).eq(candidate_model)
    ].set_index(METRIC_PAIRING_KEY)[value_columns].sort_index()
    if not reference.index.equals(candidate.index):
        raise ContractError("task-aligned metric sample coverage drifted")
    deltas = candidate - reference
    primary = [
        float(deltas["rankic"].mean()),
        float(deltas["top_return"].mean()),
        float(deltas["top_precision"].mean()),
    ]
    if all(value > 0 for value in primary):
        consensus = "candidate"
    elif all(value < 0 for value in primary):
        consensus = "reference"
    else:
        consensus = "mixed"
    result: dict[str, float | int | str] = {
        "reference_model": reference_model,
        "candidate_model": candidate_model,
        "paired_group_count": len(deltas),
        "winner_consensus": consensus,
    }
    for column in value_columns:
        result[f"mean_{column}_delta"] = float(deltas[column].mean())
    return result


def summarize_task_aligned_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize each model while retaining the seed/fold protocol identity."""

    value_columns = [
        "rankic",
        "pearson_ic",
        "top_return",
        "top_excess_return",
        "long_short_spread",
        "top_precision",
        "top_membership_precision",
        "top_positive_return_rate",
        "top_above_cross_section_mean_rate",
        "ndcg_at_top",
        "quantile_monotonicity",
        "top_turnover",
    ]
    if missing := sorted({"model", *value_columns}.difference(metrics.columns)):
        raise ContractError(
            "task-aligned summary missing columns: " + ", ".join(missing)
        )
    rows: list[dict[str, object]] = []
    for model, group in metrics.groupby("model", observed=True, sort=True):
        row: dict[str, object] = {
            "model": str(model),
            "group_count": len(group),
            "seed_count": group["seed"].nunique(),
            "fold_count": group["fold"].nunique(),
            "date_count": group["signal_date"].nunique(),
        }
        for column in value_columns:
            values = group[column].to_numpy(dtype="float64")
            finite = values[np.isfinite(values)]
            row[f"mean_{column}"] = (
                float(np.mean(finite)) if len(finite) else float("nan")
            )
            row[f"std_{column}"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
            )
        rankic_mean = cast(float, row["mean_rankic"])
        rankic_std = cast(float, row["std_rankic"])
        row["rankic_ir"] = (
            rankic_mean / rankic_std
            if np.isfinite(rankic_std) and rankic_std > 0
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_task_aligned_differences(
    metrics: pd.DataFrame,
    *,
    reference_model: str,
    candidate_model: str,
    metric_columns: tuple[str, ...] = (
        "rankic",
        "top_return",
        "top_excess_return",
        "top_precision",
        "ndcg_at_top",
    ),
    seed: int = 33,
    draws: int = 2_000,
) -> pd.DataFrame:
    """Block-bootstrap paired date deltas within seed/fold/horizon units."""

    if draws <= 0:
        raise ContractError("task-aligned bootstrap draws must be positive")
    required = {*GROUP_KEY, *metric_columns}
    if missing := sorted(required.difference(metrics.columns)):
        raise ContractError(
            "task-aligned bootstrap missing columns: " + ", ".join(missing)
        )
    selected = metrics.loc[
        metrics["model"].astype(str).isin([reference_model, candidate_model])
    ].copy()
    reference = selected.loc[
        selected["model"].astype(str).eq(reference_model)
    ].set_index(METRIC_PAIRING_KEY)[list(metric_columns)].sort_index()
    candidate = selected.loc[
        selected["model"].astype(str).eq(candidate_model)
    ].set_index(METRIC_PAIRING_KEY)[list(metric_columns)].sort_index()
    if reference.empty or not reference.index.equals(candidate.index):
        raise ContractError("task-aligned bootstrap coverage drifted")
    paired = (candidate - reference).reset_index()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for metric in metric_columns:
        unit_draws: list[np.ndarray] = []
        observed_unit_means: list[float] = []
        for _, group in paired.groupby(
            ["seed", "fold", "horizon"], observed=True, sort=True
        ):
            values = group.sort_values("signal_date", kind="mergesort")[
                metric
            ].to_numpy(dtype="float64")
            values = values[np.isfinite(values)]
            if len(values) < 2:
                raise ContractError(
                    f"task-aligned bootstrap metric {metric} has an unresolved unit"
                )
            sampled, _ = _paired_block_bootstrap(values, rng, draws=draws)
            unit_draws.append(sampled)
            observed_unit_means.append(float(values.mean()))
        aggregate_draws = np.stack(unit_draws).mean(axis=0)
        low, high = np.quantile(aggregate_draws, [0.025, 0.975]).tolist()
        rows.append(
            {
                "metric": metric,
                "reference_model": reference_model,
                "candidate_model": candidate_model,
                "unit_count": len(unit_draws),
                "paired_group_count": len(paired),
                "paired_mean_delta": float(np.mean(observed_unit_means)),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(rows)
