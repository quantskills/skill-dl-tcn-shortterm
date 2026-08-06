from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from skill_dl_tcn_shortterm.neural import run_sequence_baselines


def test_lstm_and_gru_share_the_four_horizon_training_contract(tmp_path: Path) -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    instruments = ["A", "B", "C"]
    targets = [-1.0, 0.0, 1.0]
    index_rows = []
    label_rows = []
    feature_rows = []
    position = 0
    for date in dates:
        for instrument, target in zip(instruments, targets, strict=True):
            sample_id = f"{date}-{instrument}"
            index_rows.append(
                {
                    "sample_position": position,
                    "sample_id": sample_id,
                    "instrument_id": instrument,
                    "signal_date": date,
                }
            )
            feature_rows.append(np.full((2, 6), target, dtype="float32"))
            for horizon in [1, 2, 3, 5]:
                label_rows.append(
                    {
                        "sample_id": sample_id,
                        "instrument_id": instrument,
                        "signal_date": date,
                        "horizon": horizon,
                        "rank_target": target,
                        "valid": not (instrument == "B" and horizon == 5),
                    }
                )
            position += 1
    window_index = pd.DataFrame(index_rows)
    labels = pd.DataFrame(label_rows)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.stack(feature_rows), allow_pickle=False)
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    fold_zero = window_index.copy()
    fold_zero["fold"] = 0
    fold_zero["stage"] = [
        "train" if date <= "2024-01-03" else "validation"
        for date in fold_zero["signal_date"]
    ]
    fold_zero["sealed"] = False
    fold_one = window_index.copy()
    fold_one["fold"] = 1
    fold_one["stage"] = [
        "train" if date <= "2024-01-04" else "validation"
        for date in fold_one["signal_date"]
    ]
    fold_one["sealed"] = False
    split = pd.concat([fold_zero, fold_one], ignore_index=True)

    result = run_sequence_baselines(
        features,
        window_index,
        labels,
        split,
        seed=7,
        hidden_size=4,
        epochs=2,
        batch_size=4,
    )

    assert set(result.predictions["model"]) == {"lstm", "gru"}
    assert set(result.predictions["horizon"]) == {1, 2, 3, 5}
    assert set(result.predictions["stage"]) == {"validation"}
    assert set(result.predictions["fold"]) == {0, 1}
    assert np.isfinite(result.predictions["score"]).all()
    assert (
        result.predictions.loc[result.predictions["horizon"] == 5]
        .groupby("model")
        .size()
        .eq(6)
        .all()
    )

    metadata = result.training_metadata
    assert metadata.groupby("model")["fold"].nunique().eq(2).all()
    assert metadata["parameter_count"].gt(0).all()
    assert metadata["epochs"].eq(2).all()
    assert np.isfinite(metadata["final_train_loss"]).all()
    assert set(metadata["loader"]) == {"worker_safe_lazy_window_dataset"}
    assert set(metadata["storage"]) == {"read_only_memmap"}
