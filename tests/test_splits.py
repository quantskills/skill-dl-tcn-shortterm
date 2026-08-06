from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skill_dl_tcn_shortterm.splits import build_walk_forward_splits


def test_walk_forward_purges_overlap_embargoes_and_fits_train_only() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    window_index = pd.DataFrame(
        {
            "sample_position": range(30),
            "sample_id": [f"sample-{i:02d}" for i in range(30)],
            "instrument_id": "600000.XSHG",
            "signal_date": dates.strftime("%Y-%m-%d"),
        }
    )
    label_rows = []
    for sample_id, signal_date in zip(window_index["sample_id"], dates, strict=True):
        for horizon in [1, 2, 3, 5]:
            label_rows.append(
                {
                    "sample_id": sample_id,
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "horizon": horizon,
                    "label_end_at": signal_date.tz_localize("Asia/Shanghai")
                    + pd.offsets.BDay(horizon + 1)
                    + pd.Timedelta(hours=9, minutes=30),
                    "valid": True,
                }
            )
    labels = pd.DataFrame(label_rows)
    features = np.repeat(
        np.arange(30, dtype="float32")[:, None, None], repeats=6, axis=2
    )

    result = build_walk_forward_splits(
        window_index,
        labels,
        features,
        train_days=10,
        validation_days=3,
        embargo_days=5,
        test_days=3,
        max_folds=1,
    )

    manifest = result.manifest
    validation_start = pd.Timestamp(dates[10]).tz_localize("Asia/Shanghai")
    train_ids = manifest.loc[manifest["stage"] == "train", "sample_id"]
    train_label_ends = labels.loc[labels["sample_id"].isin(train_ids), "label_end_at"]
    assert (train_label_ends < validation_start).all()
    assert manifest.loc[manifest["stage"] == "purged", "sample_id"].tolist() == [
        "sample-04",
        "sample-05",
        "sample-06",
        "sample-07",
        "sample-08",
        "sample-09",
    ]
    assert manifest.loc[manifest["stage"] == "embargo", "signal_date"].nunique() == 10
    assert manifest.loc[manifest["stage"] == "test", "sealed"].all()
    assert result.preprocessing["feature_mean"] == pytest.approx([1.5])
    assert result.preprocessing["fit_sample_count"] == 4


def test_previous_sealed_test_never_reenters_later_fold_training() -> None:
    dates = pd.bdate_range("2024-01-02", periods=60)
    index = pd.DataFrame(
        {
            "sample_position": range(60),
            "sample_id": [f"s{i}" for i in range(60)],
            "instrument_id": "A",
            "signal_date": dates.strftime("%Y-%m-%d"),
            "source_fingerprint": "fixture-data",
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": sample.sample_id,
                "horizon": 1,
                "label_end_at": pd.Timestamp(
                    str(sample.signal_date), tz="Asia/Shanghai"
                )
                + pd.offsets.BDay(1),
                "valid": True,
            }
            for sample in index.itertuples(index=False)
        ]
    )

    result = build_walk_forward_splits(
        index,
        labels,
        np.arange(60, dtype="float32")[:, None, None],
        train_days=10,
        validation_days=3,
        embargo_days=5,
        test_days=3,
        max_folds=2,
    )

    fold_zero_test = set(
        result.manifest.loc[
            (result.manifest["fold"] == 0) & (result.manifest["stage"] == "test"),
            "sample_id",
        ]
    )
    fold_one_train = set(
        result.manifest.loc[
            (result.manifest["fold"] == 1) & (result.manifest["stage"] == "train"),
            "sample_id",
        ]
    )
    assert fold_zero_test
    assert not fold_zero_test & fold_one_train
    assert set(result.preprocessing["folds"]) == {"0", "1"}
    assert result.manifest["data_fingerprint"].eq("fixture-data").all()
    assert result.manifest["stage_reason"].notna().all()
