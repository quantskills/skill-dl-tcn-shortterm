"""Comparable statistical baselines and cross-sectional evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .experiment import ContractError


@dataclass(frozen=True)
class BaselineResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def _block_bootstrap_means(
    values: np.ndarray, rng: np.random.Generator, *, draws: int = 500
) -> np.ndarray:
    """Bootstrap a time-ordered daily series with circular contiguous blocks."""

    ordered = np.asarray(values, dtype="float64")
    if ordered.ndim != 1 or len(ordered) == 0:
        raise ValueError("block bootstrap requires a non-empty one-dimensional series")
    block_length = max(1, min(len(ordered), int(np.ceil(np.sqrt(len(ordered))))))
    block_count = int(np.ceil(len(ordered) / block_length))
    offsets = np.arange(block_length)
    means = []
    for _ in range(draws):
        starts = rng.integers(0, len(ordered), size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % len(ordered)).reshape(-1)
        means.append(float(ordered[indices[: len(ordered)]].mean()))
    return np.asarray(means, dtype="float64")


def _rankic_by_date(predictions: pd.DataFrame) -> pd.Series:
    values: dict[str, float] = {}
    for signal_date, group in predictions.groupby("signal_date", observed=True):
        if group["score"].nunique() < 2 or group["target"].nunique() < 2:
            values[str(signal_date)] = float("nan")
        else:
            values[str(signal_date)] = float(
                spearmanr(group["score"], group["target"]).statistic
            )
    return pd.Series(values, dtype="float64")


def build_risk_exposure_report(
    predictions: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    """Report score/target behavior by PIT industry, size, liquidity, and state."""

    required = {
        "instrument_id",
        "signal_date",
        "industry",
        "market_cap",
        "adv20",
        "market_state",
    }
    missing = required - set(universe.columns)
    if missing:
        raise ContractError(f"universe risk fields missing: {sorted(missing)}")
    risk = universe[sorted(required)].copy()
    merged = predictions.merge(
        risk,
        on=["instrument_id", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    merged["industry"] = merged["industry"].fillna("unavailable").astype(str)
    merged["market_state"] = merged["market_state"].fillna("unavailable").astype(str)

    def quantile_bucket(values: pd.Series, labels: list[str]) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.notna()
        result = pd.Series("unavailable", index=values.index, dtype="object")
        unique = int(numeric.loc[valid].nunique())
        bucket_count = min(len(labels), unique)
        if bucket_count >= 2:
            result.loc[valid] = cast(
                Any,
                pd.qcut(
                    numeric.loc[valid].rank(method="first"),
                    q=bucket_count,
                    labels=labels[:bucket_count],
                ).astype(str),
            )
        elif valid.any():
            result.loc[valid] = labels[0]
        return result

    merged["size_bucket"] = merged.groupby("signal_date", group_keys=False)[
        "market_cap"
    ].apply(lambda values: quantile_bucket(values, ["small", "mid", "large"]))
    merged["liquidity_bucket"] = merged.groupby("signal_date", group_keys=False)[
        "adv20"
    ].apply(lambda values: quantile_bucket(values, ["low", "mid", "high"]))
    dimensions = {
        "industry": "industry",
        "size": "size_bucket",
        "liquidity": "liquidity_bucket",
        "market_state": "market_state",
    }
    rows = []
    for dimension, column in dimensions.items():
        for keys, group in merged.groupby(
            ["model", "fold", "horizon", column], observed=True, dropna=False
        ):
            model, fold, horizon, bucket = cast(tuple[Any, Any, Any, Any], keys)
            daily = _rankic_by_date(group).dropna()
            rows.append(
                {
                    "model": model,
                    "fold": int(fold),
                    "horizon": int(horizon),
                    "dimension": dimension,
                    "bucket": str(bucket),
                    "sample_count": int(len(group)),
                    "coverage": float(group["target"].notna().mean()),
                    "mean_score": float(group["score"].mean()),
                    "mean_target": float(group["target"].mean()),
                    "mean_daily_rankic": (
                        float(daily.mean()) if not daily.empty else float("nan")
                    ),
                    "scores_unchanged": True,
                }
            )
    return pd.DataFrame(rows)


def _summarize_predictions(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    daily_by_key: dict[tuple[str, int, int], pd.Series] = {}
    for keys, group in predictions.groupby(["model", "fold", "horizon"], observed=True):
        model, fold, horizon = cast(tuple[Any, Any, Any], keys)
        daily = _rankic_by_date(group).dropna()
        daily_by_key[(str(model), int(fold), int(horizon))] = daily
        if daily.empty:
            mean = std = icir = ci_low = ci_high = float("nan")
        else:
            mean = float(daily.mean())
            std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
            icir = float(mean / std) if std > 0 else float("nan")
            bootstrap = _block_bootstrap_means(daily.to_numpy(), rng)
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        monotonicity_by_date = []
        for _, date_group in group.groupby("signal_date", observed=True):
            quantile_count = min(5, int(date_group["score"].nunique()), len(date_group))
            if quantile_count < 2:
                continue
            quantiles = pd.qcut(
                date_group["score"].rank(method="first"),
                q=quantile_count,
                labels=False,
                duplicates="drop",
            )
            means = (
                date_group.assign(_quantile=quantiles)
                .groupby("_quantile")["target"]
                .mean()
            )
            if len(means) > 1 and means.nunique() > 1:
                monotonicity_by_date.append(
                    float(spearmanr(means.index.to_numpy(), means.to_numpy()).statistic)
                )
        rows.append(
            {
                "model": model,
                "fold": int(fold),
                "horizon": int(horizon),
                "rankic": mean,
                "rankic_std": std,
                "icir": icir,
                "rankic_ci_low": ci_low,
                "rankic_ci_high": ci_high,
                "validation_sample_count": int(len(group)),
                "validation_date_count": int(group["signal_date"].nunique()),
                "coverage": float(group["target"].notna().mean()),
                "quantile_monotonicity": (
                    float(np.mean(monotonicity_by_date))
                    if monotonicity_by_date
                    else float("nan")
                ),
                "direction_accuracy": float(
                    (np.sign(group["score"]) == np.sign(group["target"])).mean()
                ),
                "direction_accuracy_role": "auxiliary",
            }
        )
    summary = pd.DataFrame(rows)
    one_day = summary.loc[summary["horizon"] == 1, ["model", "fold", "rankic"]].rename(
        columns={"rankic": "rankic_1d"}
    )
    summary = summary.merge(one_day, on=["model", "fold"], how="left")
    summary["rankic_decay_from_1d"] = summary["rankic"] - summary["rankic_1d"]
    comparison_rows = []
    baseline_names = {
        "constant-zero",
        "historical-mean",
        "ridge",
        "lightgbm",
        "lstm",
        "gru",
    }
    for keys, group in summary.groupby(["fold", "horizon"], observed=True):
        fold, horizon = cast(tuple[Any, Any], keys)
        eligible = group.loc[group["rankic"].notna()]
        baselines = eligible.loc[eligible["model"].isin(baseline_names)]
        candidates = baselines if not baselines.empty else eligible
        comparison_model = (
            str(
                candidates.sort_values(
                    ["rankic", "model"], ascending=[False, True]
                ).iloc[0]["model"]
            )
            if not candidates.empty
            else ""
        )
        comparison_daily = daily_by_key.get(
            (comparison_model, int(fold), int(horizon)), pd.Series(dtype="float64")
        )
        for row in group.itertuples(index=False):
            row = cast(Any, row)
            model_daily = daily_by_key.get(
                (str(row.model), int(fold), int(horizon)), pd.Series(dtype="float64")
            )
            common_dates = model_daily.index.intersection(comparison_daily.index)
            paired = model_daily.loc[common_dates] - comparison_daily.loc[common_dates]
            if paired.empty:
                delta = delta_low = delta_high = float("nan")
            else:
                delta = float(paired.mean())
                draws = _block_bootstrap_means(paired.to_numpy(), rng)
                delta_low, delta_high = np.quantile(draws, [0.025, 0.975]).tolist()
            comparison_rows.append(
                {
                    "model": row.model,
                    "fold": int(fold),
                    "horizon": int(horizon),
                    "comparison_baseline": comparison_model,
                    "paired_delta_rankic": delta,
                    "paired_delta_ci_low": delta_low,
                    "paired_delta_ci_high": delta_high,
                }
            )
    return summary.merge(
        pd.DataFrame(comparison_rows),
        on=["model", "fold", "horizon"],
        validate="one_to_one",
    )


def run_statistical_baselines(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    seed: int,
) -> BaselineResult:
    """Fit constant, Ridge, and LightGBM on identical walk-forward folds."""

    if len(features) != len(window_index):
        raise ContractError("features and window index sample counts must match")
    feature_by_sample = {
        cast(Any, row).sample_id: features[int(cast(Any, row).sample_position)].reshape(
            -1
        )
        for row in window_index.itertuples(index=False)
    }
    label_lookup = labels.loc[labels["valid"]].set_index(["sample_id", "horizon"])
    prediction_rows = []

    for fold in sorted(split_manifest["fold"].unique()):
        fold_split = split_manifest.loc[split_manifest["fold"] == fold]
        train_ids = fold_split.loc[fold_split["stage"] == "train", "sample_id"].tolist()
        validation_ids = fold_split.loc[
            fold_split["stage"] == "validation", "sample_id"
        ].tolist()
        for horizon in [1, 2, 3, 5]:
            usable_train = [
                sample_id
                for sample_id in train_ids
                if (sample_id, horizon) in label_lookup.index
            ]
            usable_validation = [
                sample_id
                for sample_id in validation_ids
                if (sample_id, horizon) in label_lookup.index
            ]
            if not usable_train or not usable_validation:
                continue
            x_train = np.stack(
                [feature_by_sample[sample_id] for sample_id in usable_train]
            )
            y_train = np.asarray(
                [
                    label_lookup.loc[(sample_id, horizon), "rank_target"]
                    for sample_id in usable_train
                ],
                dtype="float64",
            )
            x_validation = np.stack(
                [feature_by_sample[sample_id] for sample_id in usable_validation]
            )
            models = {
                "constant-zero": None,
                "ridge": Ridge(alpha=1.0),
                "lightgbm": LGBMRegressor(
                    n_estimators=50,
                    learning_rate=0.05,
                    max_depth=3,
                    num_leaves=7,
                    min_child_samples=1,
                    verbosity=-1,
                    random_state=seed,
                    n_jobs=1,
                ),
            }
            scores: dict[str, np.ndarray] = {}
            for model_name, model in models.items():
                if model is None:
                    scores[model_name] = np.zeros(
                        len(usable_validation), dtype="float64"
                    )
                else:
                    model.fit(x_train, y_train)
                    scores[model_name] = np.asarray(
                        model.predict(x_validation), dtype="float64"
                    )
            for model_name, model_scores in scores.items():
                for sample_id, score in zip(
                    usable_validation, model_scores, strict=True
                ):
                    label = label_lookup.loc[(sample_id, horizon)]
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "fold": int(fold),
                            "stage": "validation",
                            "sample_id": sample_id,
                            "instrument_id": label["instrument_id"],
                            "signal_date": label["signal_date"],
                            "horizon": horizon,
                            "score": float(score),
                            "target": float(label["rank_target"]),
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ContractError(
            "statistical baselines have no valid train/validation samples"
        )
    metrics = _summarize_predictions(predictions, seed)
    return BaselineResult(predictions=predictions, metrics=metrics)
