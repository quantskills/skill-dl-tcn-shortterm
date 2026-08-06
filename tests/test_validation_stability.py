from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.stability import build_validation_stability_manifest
from skill_dl_tcn_shortterm.stability import evaluate_tcn_stability_gate


def _stability_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=12).strftime("%Y-%m-%d").tolist()
    rows = []
    labels = []
    for position, (date, instrument) in enumerate(
        (date, instrument) for date in dates for instrument in ["A", "B"]
    ):
        sample_id = f"{date}-{instrument}"
        rows.append(
            {
                "sample_position": position,
                "sample_id": sample_id,
                "signal_date": date,
                "instrument_id": instrument,
            }
        )
        date_position = dates.index(date)
        label_end = dates[min(date_position + 1, len(dates) - 1)]
        labels.append(
            {
                "sample_id": sample_id,
                "horizon": 1,
                "valid": True,
                "label_end_at": f"{label_end} 09:30:00+08:00",
            }
        )
    index = pd.DataFrame(rows)
    source = index[["sample_position", "sample_id", "signal_date"]].copy()
    source["fold"] = 1
    source["stage"] = np.where(source["signal_date"].isin(dates[-2:]), "test", "train")
    source["sealed"] = source["stage"].eq("test")
    return index, pd.DataFrame(labels), source


@pytest.mark.parametrize("window_kind", ["expanding", "sliding"])
def test_stability_manifest_excludes_test_and_purges_overlapping_labels(
    window_kind: str,
) -> None:
    index, labels, source = _stability_fixture()

    result = build_validation_stability_manifest(
        index,
        labels,
        source,
        source_fold=1,
        train_days=4,
        validation_days=2,
        fold_count=3,
        window_kind=cast(Literal["expanding", "sliding"], window_kind),
    )

    assert set(result.manifest["stage"]) <= {"train", "validation", "purged"}
    assert set(result.manifest["fold"]) == {0, 1, 2}
    forbidden = set(source.loc[source["stage"] == "test", "sample_position"])
    assert not forbidden & set(result.manifest["sample_position"])
    label_end = pd.to_datetime(labels.set_index("sample_id")["label_end_at"])
    for fold, rows in result.manifest.groupby("fold"):
        validation_start = pd.Timestamp(
            rows.loc[rows["stage"] == "validation", "signal_date"].min()
        )
        train_ids = rows.loc[rows["stage"] == "train", "sample_id"]
        assert (label_end.loc[train_ids].dt.tz_localize(None) < validation_start).all()
        if window_kind == "sliding":
            assert rows.loc[rows["stage"] == "train", "signal_date"].nunique() <= 4
    assert result.summary["validation_days"].eq(2).all()
    assert len(result.fingerprint) == 64


def test_stability_manifest_fails_closed_when_ordinary_history_is_insufficient() -> None:
    index, labels, source = _stability_fixture()

    with pytest.raises(ContractError, match="insufficient ordinary-validation dates"):
        build_validation_stability_manifest(
            index,
            labels,
            source,
            source_fold=1,
            train_days=8,
            validation_days=2,
            fold_count=2,
            window_kind="expanding",
        )


def test_stability_manifest_task_writes_an_immutable_receipt(tmp_path: Path) -> None:
    index, labels, source = _stability_fixture()
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    np.save(
        run_dir / "feature-windows.npy",
        np.zeros((len(index), 1, 1), dtype="float32"),
    )
    index.to_parquet(run_dir / "window-index.parquet", index=False)
    labels.to_parquet(run_dir / "labels.parquet", index=False)
    source.to_parquet(run_dir / "split-manifest.parquet", index=False)
    output_dir = tmp_path / "stability"
    command = [
        sys.executable,
        "tasks/build_validation_stability_manifest.py",
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--source-fold",
        "1",
        "--train-days",
        "4",
        "--validation-days",
        "2",
        "--fold-count",
        "3",
        "--window-kind",
        "expanding",
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["sealed_test_accessed"] is False
    assert receipt["protocol"]["fold_count"] == 3
    assert (output_dir / "validation-stability-manifest.parquet").is_file()
    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode == 2
    assert "refuses to overwrite" in json.loads(refused.stdout)["error"]


def test_tcn_stability_gate_separates_speed_from_prediction_effect() -> None:
    rows = []
    values = {
        "lstm": [0.08, 0.10, 0.12, 0.09],
        "bai-tcn": [0.02, 0.03, 0.04, 0.01],
        "tcn-lite": [0.04, 0.05, 0.06, 0.03],
    }
    for model, rankics in values.items():
        for unit, rankic in enumerate(rankics):
            rows.append(
                {
                    "model": model,
                    "fold": unit // 2,
                    "base_seed": 7 + unit % 2,
                    "best_validation_rankic": rankic,
                    "model_step_samples_per_second": (
                        100.0 if model == "lstm" else 200.0
                    ),
                    "samples_per_second": 100.0 if model == "lstm" else 130.0,
                }
            )

    decision = evaluate_tcn_stability_gate(
        pd.DataFrame(rows),
        candidate_model="tcn-lite",
        recurrent_baseline="lstm",
        tcn_control="bai-tcn",
        model_step_speedup_min=1.5,
        end_to_end_speedup_min=1.2,
        positive_rate_min=0.6,
        control_median_improvement_min=0.005,
        worst_fold_min=-0.01,
    )

    assert decision.speed_status == "cpu_end_to_end_speedup_confirmed"
    assert decision.model_step_speedup == pytest.approx(2.0)
    assert decision.end_to_end_speedup == pytest.approx(1.3)
    assert decision.effect_status == "stop_unstable_validation"
    assert decision.candidate_median_rankic < decision.baseline_median_rankic
