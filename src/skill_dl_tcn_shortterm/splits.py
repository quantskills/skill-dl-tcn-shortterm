"""Chronological walk-forward split contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class SplitResult:
    manifest: pd.DataFrame
    preprocessing: dict[str, Any]


def build_walk_forward_splits(
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    features: np.ndarray,
    *,
    train_days: int,
    validation_days: int,
    embargo_days: int,
    test_days: int,
    max_folds: int | None = None,
) -> SplitResult:
    """Build expanding chronological folds with dynamic purge and sealed tests."""

    if len(window_index) != len(features):
        raise ContractError("window index and feature tensor sample counts must match")
    for value, name in [
        (train_days, "train_days"),
        (validation_days, "validation_days"),
        (embargo_days, "embargo_days"),
        (test_days, "test_days"),
    ]:
        if int(value) <= 0:
            raise ContractError(f"{name} must be positive")
    dates = sorted(window_index["signal_date"].astype(str).unique().tolist())
    block_size = validation_days + embargo_days + test_days + embargo_days
    fold_rows: list[dict[str, Any]] = []
    max_label_end = (
        labels.loc[labels["valid"]]
        .groupby("sample_id", observed=True)["label_end_at"]
        .max()
        .to_dict()
    )
    source_fingerprints = (
        sorted(window_index["source_fingerprint"].dropna().astype(str).unique())
        if "source_fingerprint" in window_index
        else []
    )
    data_fingerprint = (
        source_fingerprints[0]
        if len(source_fingerprints) == 1
        else hashlib.sha256(
            "|".join(window_index["sample_id"].map(str).tolist()).encode("utf-8")
        ).hexdigest()
    )
    previously_sealed_dates: set[str] = set()
    fold = 0
    while True:
        validation_start_position = train_days + fold * block_size
        validation_end_position = validation_start_position + validation_days
        pre_test_embargo_end = validation_end_position + embargo_days
        test_end_position = pre_test_embargo_end + test_days
        post_test_embargo_end = test_end_position + embargo_days
        if post_test_embargo_end > len(dates):
            break
        if max_folds is not None and fold >= max_folds:
            break

        train_dates = set(dates[:validation_start_position]) - previously_sealed_dates
        validation_dates = set(dates[validation_start_position:validation_end_position])
        post_validation_embargo = set(
            dates[validation_end_position:pre_test_embargo_end]
        )
        post_test_embargo = set(dates[test_end_position:post_test_embargo_end])
        embargo_dates = post_validation_embargo | post_test_embargo
        test_dates = set(dates[pre_test_embargo_end:test_end_position])
        validation_start = pd.Timestamp(
            dates[validation_start_position], tz="Asia/Shanghai"
        )

        for sample in window_index.itertuples(index=False):
            sample = cast(Any, sample)
            signal_date = str(sample.signal_date)
            stage: str | None = None
            stage_reason = ""
            sealed = False
            if signal_date in previously_sealed_dates:
                stage = "sealed_holdout"
                stage_reason = "previous_fold_sealed_test"
                sealed = True
            elif signal_date in train_dates:
                end_at = max_label_end.get(sample.sample_id)
                if (
                    end_at is None
                    or pd.isna(end_at)
                    or pd.Timestamp(end_at) >= validation_start
                ):
                    stage = "purged"
                    stage_reason = "label_end_crosses_validation"
                else:
                    stage = "train"
                    stage_reason = "expanding_train"
            elif signal_date in validation_dates:
                stage = "validation"
                stage_reason = "out_of_sample_validation"
            elif signal_date in embargo_dates:
                stage = "embargo"
                stage_reason = (
                    "post_validation_embargo"
                    if signal_date in post_validation_embargo
                    else "post_test_embargo"
                )
            elif signal_date in test_dates:
                stage = "test"
                stage_reason = "sealed_test"
                sealed = True
            if stage is not None:
                fold_rows.append(
                    {
                        "fold": fold,
                        "sample_id": sample.sample_id,
                        "sample_position": int(sample.sample_position),
                        "instrument_id": sample.instrument_id,
                        "signal_date": signal_date,
                        "stage": stage,
                        "stage_reason": stage_reason,
                        "sealed": sealed,
                        "data_fingerprint": data_fingerprint,
                        "split_version": "expanding-purge-embargo-v2",
                        "validation_start_date": dates[validation_start_position],
                        "validation_end_date": dates[validation_end_position - 1],
                        "test_start_date": dates[pre_test_embargo_end],
                        "test_end_date": dates[test_end_position - 1],
                    }
                )
        previously_sealed_dates.update(test_dates)
        fold += 1

    manifest = pd.DataFrame(fold_rows)
    if manifest.empty:
        raise ContractError(
            "not enough signal dates for the configured walk-forward split"
        )
    manifest["stage_sample_count"] = manifest.groupby(["fold", "stage"], observed=True)[
        "sample_id"
    ].transform("size")
    fold_preprocessing: dict[str, dict[str, Any]] = {}
    for fold_number in sorted(manifest["fold"].unique()):
        fold_train = manifest.loc[
            (manifest["fold"] == fold_number) & (manifest["stage"] == "train")
        ]
        positions = fold_train["sample_position"].to_numpy(dtype="int64")
        if len(positions) == 0:
            raise ContractError(
                f"dynamic purge removed every training sample in fold {fold_number}"
            )
        train_features = features[positions]
        feature_mean = train_features.mean(axis=(0, 2)).astype("float64")
        feature_std = train_features.std(axis=(0, 2)).astype("float64")
        feature_std[feature_std == 0] = 1.0
        fold_value = {
            "fit_fold": int(fold_number),
            "fit_sample_count": int(len(positions)),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "fit_sample_ids": fold_train["sample_id"].tolist(),
        }
        fold_value["fingerprint"] = hashlib.sha256(
            json.dumps(fold_value, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        fold_preprocessing[str(fold_number)] = fold_value
    first_fold = fold_preprocessing["0"]
    preprocessing = {**first_fold, "folds": fold_preprocessing}
    return SplitResult(manifest=manifest, preprocessing=preprocessing)
