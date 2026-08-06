"""Reproduce the v42 train-loss / validation-trajectory divergence without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    run_dir = arguments.run_dir.resolve()
    history_path = run_dir / "epoch-history.parquet"
    leaderboard_path = run_dir / "leaderboard.parquet"
    if not history_path.is_file() or not leaderboard_path.is_file():
        print(json.dumps({"status": "error", "error": "v42 trajectory inputs missing"}))
        return 2
    history = pd.read_parquet(history_path)
    leaderboard = pd.read_parquet(leaderboard_path)
    required_history = {
        "trial_id",
        "seed",
        "fold",
        "epoch",
        "train_loss",
        "mean_daily_rankic",
    }
    required_leaderboard = {"trial_id", "seed", "fold", "best_epoch"}
    if missing := sorted(required_history.difference(history.columns)):
        print(json.dumps({"status": "error", "missing_history": missing}))
        return 2
    if missing := sorted(required_leaderboard.difference(leaderboard.columns)):
        print(json.dumps({"status": "error", "missing_leaderboard": missing}))
        return 2
    selected_history = history.loc[
        history["trial_id"].map(str).str.contains("consensus-student025-v42")
    ].copy()
    selected_leaderboard = leaderboard.loc[
        leaderboard["trial_id"].map(str).str.contains("consensus-student025-v42")
    ].copy()
    if (
        selected_history.empty
        or len(selected_leaderboard) != 15
        or set(selected_history["epoch"].astype(int)) != set(range(1, 9))
        or selected_history.groupby(["seed", "fold"], observed=True).ngroups != 15
    ):
        print(json.dumps({"status": "error", "error": "v42 trajectory coverage drifted"}))
        return 2
    by_epoch = selected_history.groupby("epoch", observed=True).agg(
        mean_train_loss=("train_loss", "mean"),
        mean_validation_rankic=("mean_daily_rankic", "mean"),
    )
    numeric = by_epoch.to_numpy(dtype="float64")
    if not np.isfinite(numeric).all():
        print(json.dumps({"status": "error", "error": "trajectory is non-finite"}))
        return 2
    first = by_epoch.loc[1]
    last = by_epoch.loc[8]
    best_epochs = sorted(set(selected_leaderboard["best_epoch"].astype(int)))
    diverged = bool(
        float(cast(Any, last["mean_train_loss"]))
        < float(cast(Any, first["mean_train_loss"]))
        and float(cast(Any, last["mean_validation_rankic"]))
        < float(cast(Any, first["mean_validation_rankic"]))
        and len(best_epochs) >= 4
        and min(best_epochs) == 1
        and max(best_epochs) >= 7
    )
    payload = {
        "status": "red_training_trajectory_gap_v44" if diverged else "green",
        "unit_count": 15,
        "epoch_1_mean_train_loss": float(cast(Any, first["mean_train_loss"])),
        "epoch_8_mean_train_loss": float(cast(Any, last["mean_train_loss"])),
        "epoch_1_mean_validation_rankic": float(
            cast(Any, first["mean_validation_rankic"])
        ),
        "epoch_8_mean_validation_rankic": float(
            cast(Any, last["mean_validation_rankic"])
        ),
        "validation_rankic_delta_epoch8_minus_epoch1": float(
            cast(Any, last["mean_validation_rankic"])
            - cast(Any, first["mean_validation_rankic"])
        ),
        "selected_best_epochs": best_epochs,
        "sealed_test_accessed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 1 if diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
