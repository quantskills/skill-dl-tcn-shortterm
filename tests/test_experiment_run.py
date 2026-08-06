from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError, run_experiment
from skill_dl_tcn_shortterm.evidence import verify_evidence_bundle
from skill_dl_tcn_shortterm.experiment import (
    REQUIRED_ENGINEERING_MODELS,
    _assert_engineering_complete,
    _resolve_cost_schedule,
)


def _write_minimal_dataset(root: Path) -> Path:
    data_path = root / "samples.parquet"
    pd.DataFrame(
        {
            "instrument_id": ["600000.XSHG", "000001.XSHE"],
            "signal_date": ["2024-01-02", "2024-01-02"],
            "target_1d": [-1.0, 1.0],
            "target_2d": [-1.0, 1.0],
            "target_3d": [-1.0, 1.0],
            "target_5d": [-1.0, 1.0],
        }
    ).to_parquet(data_path, index=False)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_kind": "prebuilt_samples",
                "data_path": data_path.name,
                "data_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_engineering_dataset(root: Path) -> Path:
    days = pd.bdate_range("2024-01-02", periods=30)
    instruments = ["600000.XSHG", "000001.XSHE", "300001.XSHE"]
    price_curves = {
        "600000.XSHG": [10.0 + position * 0.20 for position in range(len(days))],
        "000001.XSHE": [20.0 - position * 0.10 for position in range(len(days))],
        "300001.XSHE": [15.0 + position * 0.05 for position in range(len(days))],
    }
    bar_frames = []
    execution_rows = []
    for instrument in instruments:
        for day, daily_open in zip(days, price_curves[instrument], strict=True):
            text = day.strftime("%Y-%m-%d")
            ends = pd.date_range(
                f"{text} 09:31", f"{text} 11:30", freq="1min", tz="Asia/Shanghai"
            ).append(
                pd.date_range(
                    f"{text} 13:01", f"{text} 15:00", freq="1min", tz="Asia/Shanghai"
                )
            )
            drift = np.arange(len(ends), dtype="float64") * 0.0001
            price = daily_open + drift
            bar_frames.append(
                pd.DataFrame(
                    {
                        "instrument_id": instrument,
                        "bar_end_at": ends,
                        "open": price,
                        "high": price + 0.01,
                        "low": price - 0.01,
                        "close": price + 0.001,
                        "volume": 10_000.0,
                        "amount": 10_000.0 * price,
                        "quality_flag": "ok",
                    }
                )
            )
            execution_rows.append(
                {
                    "instrument_id": instrument,
                    "trade_date": text,
                    "open_price": daily_open,
                    "adv20": 100_000_000.0,
                    "buyable": True,
                    "sellable": True,
                    "price_source": "synthetic-open",
                }
            )
    files = {
        "data": root / "bars.parquet",
        "instrument_state": root / "states.parquet",
        "corporate_action": root / "actions.parquet",
        "execution_state": root / "execution.parquet",
    }
    pd.concat(bar_frames, ignore_index=True).to_parquet(files["data"], index=False)
    pd.DataFrame(
        [
            {
                "instrument_id": instrument,
                "effective_at": "2000-01-01T00:00:00+08:00",
                "exchange": "XSHG" if instrument.endswith("XSHG") else "XSHE",
                "security_type": "A_SHARE",
                "listed_date": "1990-01-01",
                "delisted_date": None,
                "is_st": False,
                "is_delisting": False,
                "is_suspended": False,
                "industry": "bank" if instrument != "300001.XSHE" else "tech",
                "market_cap": 1_000_000_000.0 + offset * 100_000_000.0,
                "adv20": 100_000_000.0,
                "market_state": "normal",
            }
            for offset, instrument in enumerate(instruments)
        ]
    ).to_parquet(files["instrument_state"], index=False)
    pd.DataFrame(
        [
            {
                "instrument_id": instruments[0],
                "effective_date": "1999-01-01",
                "pit_reliable": True,
            }
        ]
    ).to_parquet(files["corporate_action"], index=False)
    pd.DataFrame(execution_rows).to_parquet(files["execution_state"], index=False)
    manifest = {
        "schema_version": 1,
        "dataset_kind": "raw_1m",
        "timezone": "Asia/Shanghai",
        "price_unit": "CNY",
        "volume_unit": "share",
        "amount_unit": "CNY",
        "source_version": "engineering-fixture-v1",
    }
    for key, path in files.items():
        path_key = "data_path" if key == "data" else f"{key}_path"
        hash_key = "data_sha256" if key == "data" else f"{key}_sha256"
        manifest[path_key] = path.name
        manifest[hash_key] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = root / "engineering-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_researcher_can_run_minimal_offline_experiment(tmp_path: Path) -> None:
    manifest_path = _write_minimal_dataset(tmp_path)

    result = run_experiment(
        config={"run_name": "smoke", "seed": 7, "horizons": [1, 2, 3, 5]},
        manifest_path=manifest_path,
        output_root=tmp_path / "runs",
    )

    assert result.run_id
    assert result.run_dir.is_dir()
    assert result.manifest_path.is_file()
    assert result.predictions_path.is_file()
    assert result.metrics_path.is_file()
    assert result.report_path.is_file()
    assert result.evidence_index_path is not None
    assert result.environment_path is not None
    assert result.evidence_index_path is not None
    verify_evidence_bundle(result.evidence_index_path)

    run_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["status"] == "success"
    assert run_manifest["run_id"] == result.run_id
    assert run_manifest["model"] == "constant-zero"
    assert run_manifest["models"] == ["constant-zero"]
    assert len(run_manifest["code"]["source_sha256"]) == 64
    assert "revision" in run_manifest["code"]
    assert "dirty" in run_manifest["code"]
    assert run_manifest["artifacts"]["evidence_index"] == "evidence-index.json"

    predictions = pd.read_parquet(result.predictions_path)
    assert list(predictions["horizon"].unique()) == [1, 2, 3, 5]
    assert predictions["run_id"].unique().tolist() == [result.run_id]

    report = result.report_path.read_text(encoding="utf-8")
    assert "工程状态：完成最小离线运行" in report
    assert "Alpha 证据：未评估" in report
    assert "速度假设：未评估" in report


def test_invalid_parquet_fails_before_artifacts_are_created(tmp_path: Path) -> None:
    data_path = tmp_path / "not-parquet.parquet"
    data_path.write_bytes(b"not a parquet file")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_kind": "prebuilt_samples",
                "data_path": data_path.name,
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "runs"

    with pytest.raises(ContractError, match="cannot read prebuilt_samples data"):
        run_experiment(
            config={"run_name": "invalid", "seed": 7, "horizons": [1, 2, 3, 5]},
            manifest_path=manifest_path,
            output_root=output_root,
        )

    assert not output_root.exists()


def test_engineering_complete_requires_every_model_in_predictions_and_metrics() -> None:
    shared = REQUIRED_ENGINEERING_MODELS - {
        "bai-tcn-1d",
        "bai-tcn-2d",
        "bai-tcn-3d",
        "bai-tcn-5d",
    }
    matrix = [
        {"model": model, "fold": 0, "horizon": horizon, "sample_id": "s1"}
        for horizon in [1, 2, 3, 5]
        for model in sorted(shared | {f"bai-tcn-{horizon}d"})
    ]
    predictions = pd.DataFrame(matrix)
    metrics = pd.DataFrame(matrix).drop(columns="sample_id")
    comparison = pd.DataFrame(
        [
            {"comparison_type": kind, "horizon": horizon, "paired_date_count": 1}
            for kind in ["single-vs-shared", "lite-vs-shared"]
            for horizon in [1, 2, 3, 5]
        ]
    )
    split = pd.DataFrame({"fold": [0]})

    with pytest.raises(ContractError, match="metric matrix is incomplete"):
        _assert_engineering_complete(
            {"component": object()},
            predictions,
            metrics.iloc[:-1],
            comparison,
            split,
        )

    with pytest.raises(ContractError, match="missing TCN comparison rows"):
        _assert_engineering_complete(
            {"component": object()},
            predictions,
            metrics,
            comparison.loc[comparison["comparison_type"] == "single-vs-shared"],
            split,
        )

    mismatched = predictions.copy()
    mismatch_mask = (
        (mismatched["model"] == "bai-tcn")
        & (mismatched["fold"] == 0)
        & (mismatched["horizon"] == 1)
    )
    mismatched.loc[mismatch_mask, "sample_id"] = "different-sample"
    with pytest.raises(ContractError, match="do not share validation samples"):
        _assert_engineering_complete(
            {"component": object()}, mismatched, metrics, comparison, split
        )

    unpaired = comparison.copy()
    unpaired.loc[0, "paired_date_count"] = 0
    with pytest.raises(ContractError, match="require paired validation dates"):
        _assert_engineering_complete(
            {"component": object()}, predictions, metrics, unpaired, split
        )

    _assert_engineering_complete(
        {"component": object()},
        predictions,
        metrics,
        comparison,
        split,
    )


def test_engineering_complete_main_seam_runs_every_required_component(
    tmp_path: Path,
) -> None:
    manifest_path = _write_engineering_dataset(tmp_path)

    result = run_experiment(
        config={
            "run_name": "engineering-acceptance",
            "seed": 7,
            "horizons": [1, 2, 3, 5],
            "lookback_days": 5,
            "engineering_complete": True,
            "walk_forward": {
                "train_days": 12,
                "validation_days": 2,
                "embargo_days": 1,
                "test_days": 2,
                "max_folds": 1,
            },
            "sequence_models": {
                "enabled": True,
                "hidden_size": 2,
                "epochs": 1,
                "batch_size": 6,
            },
            "tcn": {
                "enabled": True,
                "channels": 2,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32],
                "dropout": 0.0,
                "epochs": 1,
                "batch_size": 6,
            },
            "tcn_ablations": {
                "enabled": True,
                "channels": 2,
                "kernel_size": 3,
                "lite_dilations": [1, 2, 4, 8, 16, 32, 64],
                "bai_dilations": [1, 2, 4, 8, 16, 32],
                "dropout": 0.0,
                "epochs": 1,
                "batch_size": 6,
            },
            "portfolio": {"top_fraction": 0.5},
            "execution": {
                "capital": 100_000.0,
                "capacity_fraction": 0.05,
                "rules": {
                    "version": "cn-equity-test-v1",
                    "t_plus_one": [{"effective_from": "2020-01-01", "enabled": True}],
                },
                "cost_schedule": [
                    {
                        "version": "costs-v1",
                        "effective_from": "2024-01-01",
                        "commission_bps": 2.0,
                        "sell_tax_bps": 5.0,
                        "slippage_bps": 5.0,
                    },
                    {
                        "version": "costs-v2",
                        "effective_from": "2024-01-19",
                        "commission_bps": 1.5,
                        "sell_tax_bps": 5.0,
                        "slippage_bps": 6.0,
                    },
                ],
            },
            "performance": {
                "enabled": True,
                "hidden_size": 2,
                "tcn_channels": 2,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32],
                "epochs": 1,
                "batch_size": 6,
                "device": "cpu",
            },
        },
        manifest_path=manifest_path,
        output_root=tmp_path / "engineering-runs",
    )

    run_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["status"] == "engineering_complete"
    assert run_manifest["model"] == "multi-model-comparison"
    assert "bai-tcn" in run_manifest["models"]
    assert "lstm" in run_manifest["models"]
    assert result.execution_metrics_path is not None
    execution_metrics = pd.read_parquet(result.execution_metrics_path)
    assert {
        "pit_equal_weight_benchmark",
        "executable_long_only",
    }.issubset(set(execution_metrics["portfolio_type"]))
    assert (
        execution_metrics.loc[
            execution_metrics["portfolio_type"] == "executable_long_only",
            "benchmark_net_return",
        ]
        .notna()
        .all()
    )
    evidence_index_path = result.evidence_index_path
    assert evidence_index_path is not None
    verify_evidence_bundle(evidence_index_path)


def test_raw_experiment_routes_portable_optimized_tcn_and_matched_benchmark(
    tmp_path: Path,
) -> None:
    manifest_path = _write_engineering_dataset(tmp_path)

    result = run_experiment(
        config={
            "run_name": "optimized-tcn-routing",
            "seed": 7,
            "horizons": [1, 2, 3, 5],
            "lookback_days": 5,
            "walk_forward": {
                "train_days": 12,
                "validation_days": 2,
                "embargo_days": 1,
                "test_days": 2,
                "max_folds": 1,
            },
            "optimized_tcn": {
                "enabled": True,
                "profile": "v40-portable",
                "relative_features": True,
                "min_cross_section": 2,
                "torch_threads": 1,
                "epochs": 2,
                "batch_size": 6,
                "learning_rate": 0.003,
            },
            "execution": {
                "rules": {
                    "version": "cn-equity-test-v1",
                    "t_plus_one": [
                        {"effective_from": "2020-01-01", "enabled": True}
                    ],
                },
                "cost_schedule": [
                    {
                        "version": "costs-v1",
                        "effective_from": "2024-01-01",
                        "commission_bps": 2.0,
                        "sell_tax_bps": 5.0,
                        "slippage_bps": 5.0,
                    }
                ],
            },
            "performance": {
                "enabled": True,
                "models": ["lstm", "optimized-tcn-v40-portable"],
                "hidden_size": 4,
                "tcn_channels": 4,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64],
                "optimized_tcn_profile": "v40-portable",
                "torch_threads": 1,
                "epochs": 1,
                "batch_size": 6,
                "learning_rate": 0.003,
                "device": "cpu",
            },
        },
        manifest_path=manifest_path,
        output_root=tmp_path / "optimized-runs",
    )

    predictions = pd.read_parquet(result.predictions_path)
    assert "optimized-tcn-v40-portable" in set(predictions["model"])
    assert result.training_metadata_path is not None
    training = pd.read_parquet(result.training_metadata_path)
    assert training.loc[
        training["model"].eq("optimized-tcn-v40-portable"), "profile"
    ].eq("v40-portable").all()
    assert result.performance_metrics_path is not None
    performance = pd.read_parquet(result.performance_metrics_path)
    assert set(performance["model"]) == {
        "lstm",
        "optimized-tcn-v40-portable",
    }
    run_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert "performance_environment" in run_manifest["artifacts"]


def test_cost_schedule_requires_effective_version_and_supports_version_spans() -> None:
    config = {
        "cost_schedule": [
            {
                "version": "costs-2024-h1",
                "effective_from": "2024-01-01T00:00:00+08:00",
                "commission_bps": 2.0,
                "sell_tax_bps": 5.0,
                "slippage_bps": 7.0,
            },
            {
                "version": "costs-2024-h2",
                "effective_from": "2024-06-01",
                "commission_bps": 1.5,
                "sell_tax_bps": 5.0,
                "slippage_bps": 6.0,
            },
        ]
    }
    january_orders = pd.DataFrame(
        {"entry_at": [pd.Timestamp("2024-01-03 09:30", tz="Asia/Shanghai")]}
    )

    resolved = _resolve_cost_schedule(config, january_orders)

    assert [item["version"] for item in resolved] == [
        "costs-2024-h1",
        "costs-2024-h2",
    ]
    spanning = _resolve_cost_schedule(
        config,
        pd.DataFrame(
            {
                "entry_at": [
                    pd.Timestamp("2024-01-03 09:30", tz="Asia/Shanghai"),
                    pd.Timestamp("2024-06-03 09:30", tz="Asia/Shanghai"),
                ]
            }
        ),
    )
    assert len(spanning) == 2
    with pytest.raises(ContractError, match="must be a non-empty list"):
        _resolve_cost_schedule({}, january_orders)
