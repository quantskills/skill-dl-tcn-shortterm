from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from skill_dl_tcn_shortterm.evidence import (
    render_research_report,
    verify_evidence_bundle,
    write_evidence_bundle,
)
from skill_dl_tcn_shortterm.experiment import ContractError


def _parquet(path: Path, run_id: str, **columns: list[object]) -> Path:
    pd.DataFrame({"run_id": [run_id], **columns}).to_parquet(path, index=False)
    return path


def test_evidence_bundle_requires_complete_structured_evidence_and_detects_drift(
    tmp_path: Path,
) -> None:
    run_id = "run-fixture"
    metrics = _parquet(
        tmp_path / "metrics.parquet",
        run_id,
        model=["ridge"],
        fold=[0],
        horizon=[1],
        rankic=[0.1],
    )

    with pytest.raises(ContractError, match="engineering-complete evidence is missing"):
        write_evidence_bundle(
            tmp_path,
            run_id=run_id,
            config={"run_name": "fixture", "seed": 7},
            seed=7,
            artifacts={"metrics": metrics},
            required_artifacts={"metrics", "orders"},
            engineering_complete=True,
        )

    orders = _parquet(
        tmp_path / "orders.parquet",
        run_id,
        model=["ridge"],
        fold=[0],
        horizon=[1],
    )
    bundle = write_evidence_bundle(
        tmp_path,
        run_id=run_id,
        config={"run_name": "fixture", "seed": 7},
        seed=7,
        artifacts={"metrics": metrics, "orders": orders},
        required_artifacts={"metrics", "orders"},
        engineering_complete=True,
    )

    assert bundle.environment_path.is_file()
    assert bundle.index_path.is_file()
    environment = json.loads(bundle.environment_path.read_text(encoding="utf-8"))
    assert len(environment["code"]["source_sha256"]) == 64
    index = json.loads(bundle.index_path.read_text(encoding="utf-8"))
    assert index["run_id"] == run_id
    assert index["engineering_complete"] is True
    assert index["seed"] == 7
    verify_evidence_bundle(bundle.index_path)

    orders.write_bytes(orders.read_bytes() + b"drift")
    with pytest.raises(ContractError, match="artifact fingerprint mismatch"):
        verify_evidence_bundle(bundle.index_path)


def test_report_recomputes_structured_numbers_and_keeps_conclusions_separate() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["bai-tcn"],
            "fold": [0],
            "horizon": [1],
            "rankic": [0.123456],
            "icir": [0.4],
            "rankic_ci_low": [0.02],
            "rankic_ci_high": [0.25],
            "comparison_baseline": ["ridge"],
            "paired_delta_rankic": [-0.01],
            "paired_delta_ci_low": [-0.03],
            "paired_delta_ci_high": [0.01],
        }
    )
    portfolio = pd.DataFrame(
        {
            "model": ["ridge"],
            "fold": [0],
            "horizon": [1],
            "cumulative_gross_return_contribution": [0.03125],
            "max_drawdown": [-0.01],
            "mean_one_way_turnover": [1.0],
            "mean_gross_exposure": [1.0],
        }
    )
    execution = pd.DataFrame(
        {
            "portfolio_type": ["executable_long_only"],
            "slippage_scenario": ["scheduled"],
            "slippage_bps": [5.0],
            "net_return": [0.025],
            "benchmark_net_return": [0.02],
            "excess_net_return": [0.005],
            "unused_cash": [10.0],
        }
    )
    performance = pd.DataFrame(
        {
            "model": ["lstm", "bai-tcn"],
            "samples_per_second": [100.0, 180.0],
            "time_to_best_seconds": [10.0, 7.0],
        }
    )

    report = render_research_report(
        engineering_complete=True,
        metrics=metrics,
        portfolio_metrics=portfolio,
        execution_metrics=execution,
        performance_metrics=performance,
        execution_ledger=pd.DataFrame(
            {
                "unfilled_reason": ["buy_unavailable", ""],
                "sell_delay_sessions": [0, 2],
                "capacity_clipped_amount": [10.0, 0.0],
            }
        ),
        data_quality={
            "raw_bar_count": 240,
            "canonical_bar_count": 48,
            "complete_session_count": 1,
        },
        canonical_bars=pd.DataFrame({"quality_flag": ["ok", "vendor_missing"]}),
        universe=pd.DataFrame(
            {
                "eligible": [True, False],
                "universe_version": ["pit-v1", "pit-v1"],
                "admission_config": ["{}", "{}"],
                "exclusion_reasons": ["", "suspended"],
            }
        ),
        window_index=pd.DataFrame(
            {"window_version": ["5m-5d-v1"], "time_steps": [240]}
        ),
        window_rejections=pd.DataFrame(
            {
                "rejection_reason": ["insufficient_history"],
                "observed_steps": [96],
            }
        ),
    )

    assert "0.123456" in report
    assert "0.03125" in report
    assert "0.005" in report
    assert "eligibility_coverage" in report
    assert "insufficient_history" in report
    assert "actual_time_steps" in report
    assert "工程完成：是" in report
    assert "Alpha 证据：未发现相对最强非 TCN 基准的增量证据" in report
    assert "3–5× 速度假设：未获支持" in report
    assert "诊断性多空不属于可实现多头组合" in report
