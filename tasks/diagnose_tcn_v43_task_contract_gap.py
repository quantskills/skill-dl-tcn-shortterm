"""Reproduce the v42 broad-ranking gain / Top-precision contract gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "rankic",
    "pearson_ic",
    "top_return",
    "top_precision",
    "ndcg_at_top",
    "quantile_monotonicity",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    metrics_path = arguments.run_dir.resolve() / "task-aligned-metrics.parquet"
    if not metrics_path.is_file():
        print(json.dumps({"status": "error", "error": "metrics are missing"}))
        return 2
    metrics = pd.read_parquet(metrics_path)
    keys = ["seed", "fold", "signal_date", "horizon"]
    required = {"model", *keys, *METRICS}
    if missing := sorted(required.difference(metrics.columns)):
        print(json.dumps({"status": "error", "missing": missing}))
        return 2
    control = metrics.loc[metrics["model"].astype(str).eq("control_tcn")].set_index(keys)
    candidate = metrics.loc[
        metrics["model"].astype(str).eq("consensus_student_tcn")
    ].set_index(keys)
    if control.empty or not control.index.equals(candidate.index):
        print(json.dumps({"status": "error", "error": "paired coverage drifted"}))
        return 2
    deltas = candidate[list(METRICS)] - control[list(METRICS)]
    mean_deltas = {metric: float(deltas[metric].mean()) for metric in METRICS}
    broad_count = sum(value > 0.0 for value in mean_deltas.values())
    exact_gap = (
        mean_deltas["rankic"] >= 0.002
        and broad_count >= 4
        and mean_deltas["top_precision"] < -0.002
    )
    horizon_precision = deltas.groupby("horizon", observed=True)["top_precision"].mean()
    fold_precision = deltas.groupby("fold", observed=True)["top_precision"].mean()
    numeric = np.asarray(
        [*mean_deltas.values(), *horizon_precision, *fold_precision], dtype="float64"
    )
    if not np.isfinite(numeric).all():
        print(json.dumps({"status": "error", "error": "non-finite evidence"}))
        return 2
    payload = {
        "status": "red_task_contract_gap_v43" if exact_gap else "green",
        "mean_deltas": mean_deltas,
        "broad_improved_metrics": broad_count,
        "negative_precision_horizons": int((horizon_precision < 0.0).sum()),
        "negative_precision_folds": int((fold_precision < 0.0).sum()),
        "paired_groups": len(deltas),
        "sealed_test_accessed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 1 if exact_gap else 0


if __name__ == "__main__":
    raise SystemExit(main())
