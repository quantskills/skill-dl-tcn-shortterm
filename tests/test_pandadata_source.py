from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from skill_dl_tcn_shortterm.pandadata_source import (
    build_monthly_fetch_plan,
    audit_pandadata_coverage,
    canonicalize_pandadata_minutes,
    download_pandadata_pilot,
    fetch_pandadata_pit_universe,
    materialize_pandadata_runtime_slice,
    select_daily_top_weight_universe,
    write_pandadata_coverage_receipt,
)


def test_researcher_can_canonicalize_reverse_ordered_pandadata_minutes() -> None:
    vendor = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "date": "20250102",
                "datetime": "2025-01-02 09:32:00",
                "open": 10.1,
                "high": 10.3,
                "low": 10.0,
                "close": 10.2,
                "volume": 200.0,
                "amount": 2040.0,
            },
            {
                "symbol": "000001.SZ",
                "date": "20250102",
                "datetime": "2025-01-02 09:31:00",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1010.0,
            },
        ]
    )

    bars = canonicalize_pandadata_minutes(
        vendor, source_version="pandadata-0.0.12"
    )

    assert bars["instrument_id"].tolist() == ["000001.XSHE", "000001.XSHE"]
    assert bars["bar_end_at"].dt.strftime("%H:%M").tolist() == ["09:31", "09:32"]
    assert str(bars["bar_end_at"].dt.tz) == "Asia/Shanghai"
    assert bars["quality_flag"].tolist() == ["ok", "ok"]
    assert list(bars.columns) == [
        "instrument_id",
        "bar_end_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "quality_flag",
    ]


def test_researcher_gets_each_dates_stable_pit_top_weight_universe() -> None:
    weights = pd.DataFrame(
        [
            ("000300.SH", "000003.SZ", "20250102", 0.02),
            ("000300.SH", "600001.SH", "20250102", 0.04),
            ("000300.SH", "000001.SZ", "20250102", 0.04),
            ("000300.SH", "600001.SH", "20250103", 0.01),
            ("000300.SH", "600002.SH", "20250103", 0.05),
            ("000300.SH", "000001.SZ", "20250103", 0.03),
        ],
        columns=["index_symbol", "stock_symbol", "date", "weight"],
    )

    universe = select_daily_top_weight_universe(weights, top_n=2)

    assert universe[["trade_date", "vendor_symbol"]].to_dict("records") == [
        {"trade_date": date(2025, 1, 2), "vendor_symbol": "000001.SZ"},
        {"trade_date": date(2025, 1, 2), "vendor_symbol": "600001.SH"},
        {"trade_date": date(2025, 1, 3), "vendor_symbol": "600002.SH"},
        {"trade_date": date(2025, 1, 3), "vendor_symbol": "000001.SZ"},
    ]
    assert universe["instrument_id"].tolist() == [
        "000001.XSHE",
        "600001.XSHG",
        "600002.XSHG",
        "000001.XSHE",
    ]


def test_researcher_gets_bounded_monthly_symbol_batches_from_daily_membership() -> None:
    universe = pd.DataFrame(
        {
            "trade_date": [
                date(2025, 1, 2),
                date(2025, 1, 2),
                date(2025, 1, 3),
                date(2025, 2, 3),
            ],
            "vendor_symbol": [
                "000001.SZ",
                "600001.SH",
                "000002.SZ",
                "600002.SH",
            ],
        }
    )

    plan = build_monthly_fetch_plan(
        universe,
        start_date="2025-01-01",
        end_date="2025-02-28",
        symbol_batch_size=2,
    )

    assert [item.to_dict() for item in plan] == [
        {
            "start_date": "20250101",
            "end_date": "20250131",
            "symbols": ["000001.SZ", "000002.SZ"],
            "chunk_id": "2025-01-000",
        },
        {
            "start_date": "20250101",
            "end_date": "20250131",
            "symbols": ["600001.SH"],
            "chunk_id": "2025-01-001",
        },
        {
            "start_date": "20250201",
            "end_date": "20250228",
            "symbols": ["600002.SH"],
            "chunk_id": "2025-02-000",
        },
    ]


class _MinuteAPI:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def get_stock_min(self, **kwargs: object) -> pd.DataFrame:
        self.calls += 1
        return self.frame.copy()


class _WeightAPI:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def get_index_weights(self, **kwargs: object) -> pd.DataFrame:
        self.requests.append(kwargs)
        request_date = str(kwargs["start_date"])
        return pd.DataFrame(
            [
                ("000300.SH", "000001.SZ", request_date, 0.03),
                ("000300.SH", "600001.SH", request_date, 0.02),
            ],
            columns=["index_symbol", "stock_symbol", "date", "weight"],
        )


def test_researcher_fetches_pit_weights_in_bounded_calendar_months() -> None:
    api = _WeightAPI()

    weights, universe = fetch_pandadata_pit_universe(
        api,
        index_symbol="000300.SH",
        start_date="2024-06-01",
        end_date="2025-03-31",
        top_n=1,
    )

    requests = [(item["start_date"], item["end_date"]) for item in api.requests]
    assert len(requests) == 10
    assert requests[0] == ("20240601", "20240630")
    assert requests[-1] == ("20250301", "20250331")
    assert len(weights) == 20
    assert universe["vendor_symbol"].tolist() == ["000001.SZ"] * 10


def test_researcher_can_resume_hash_verified_pit_filtered_chunks(
    tmp_path: Path,
) -> None:
    universe = pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 2), date(2025, 1, 3)],
            "vendor_symbol": ["000001.SZ", "600001.SH"],
            "instrument_id": ["000001.XSHE", "600001.XSHG"],
            "index_symbol": ["000300.SH", "000300.SH"],
            "weight": [0.05, 0.04],
        }
    )
    plan = build_monthly_fetch_plan(
        universe,
        start_date="2025-01-01",
        end_date="2025-01-31",
        symbol_batch_size=25,
    )
    vendor = pd.DataFrame(
        [
            ("000001.SZ", "20250102", "2025-01-02 09:31", 10, 10, 10, 10, 1, 10),
            ("600001.SH", "20250102", "2025-01-02 09:31", 20, 20, 20, 20, 1, 20),
            ("600001.SH", "20250103", "2025-01-03 09:31", 21, 21, 21, 21, 1, 21),
        ],
        columns=[
            "symbol",
            "date",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ],
    )
    api = _MinuteAPI(vendor)

    manifest_path = download_pandadata_pilot(
        api,
        universe=universe,
        plan=plan,
        output_dir=tmp_path / "pilot",
        source_version="pandadata-0.0.12",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = manifest["partitions"][0]
    part_path = manifest_path.parent / partition["path"]
    bars = pd.read_parquet(part_path)
    assert api.calls == 1
    assert bars[["instrument_id", "bar_end_at"]].astype(str).to_dict("records") == [
        {"instrument_id": "000001.XSHE", "bar_end_at": "2025-01-02 09:31:00+08:00"},
        {"instrument_id": "600001.XSHG", "bar_end_at": "2025-01-03 09:31:00+08:00"},
    ]
    assert partition["sha256"] == hashlib.sha256(part_path.read_bytes()).hexdigest()
    assert manifest["parameters"]["membership_filter"] == "trade_date+instrument_id"
    assert "password" not in json.dumps(manifest).lower()

    resumed = download_pandadata_pilot(
        _MinuteAPI(pd.DataFrame()),
        universe=universe,
        plan=plan,
        output_dir=tmp_path / "pilot",
        source_version="pandadata-0.0.12",
    )
    assert resumed == manifest_path


def test_fetch_cli_requires_environment_credentials_without_echoing_values(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PANDADATA_USERNAME", None)
    environment.pop("PANDADATA_PASSWORD", None)

    result = subprocess.run(
        [
            sys.executable,
            "tasks/fetch_pandadata_pilot.py",
            "--output-dir",
            str(tmp_path / "pilot"),
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-31",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["status"] == "error"
    assert payload["error"] == (
        "PANDADATA_USERNAME and PANDADATA_PASSWORD environment variables are required"
    )


def test_researcher_can_audit_partitioned_pandadata_training_coverage(
    tmp_path: Path,
) -> None:
    dates = [date(2025, 1, 2), date(2025, 1, 3)]
    universe = pd.DataFrame(
        {
            "trade_date": dates,
            "vendor_symbol": ["000001.SZ", "000001.SZ"],
            "instrument_id": ["000001.XSHE", "000001.XSHE"],
            "index_symbol": ["000300.SH", "000300.SH"],
            "weight": [0.05, 0.05],
        }
    )
    rows = []
    for trade_date in dates:
        ends = pd.date_range(
            f"{trade_date} 09:31", f"{trade_date} 11:30", freq="1min"
        ).append(
            pd.date_range(
                f"{trade_date} 13:01", f"{trade_date} 15:00", freq="1min"
            )
        )
        for end_at in ends:
            rows.append(
                ("000001.SZ", trade_date.strftime("%Y%m%d"), end_at, 10, 10, 10, 10, 1, 10)
            )
    vendor = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "date",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ],
    )
    plan = build_monthly_fetch_plan(
        universe,
        start_date="2025-01-01",
        end_date="2025-01-31",
        symbol_batch_size=25,
    )
    manifest_path = download_pandadata_pilot(
        _MinuteAPI(vendor),
        universe=universe,
        plan=plan,
        output_dir=tmp_path / "pilot",
        source_version="pandadata-fixture",
    )

    audit = audit_pandadata_coverage(
        manifest_path,
        min_complete_instruments=1,
        lookback_days=1,
        max_horizon_days=1,
        required_signal_days=1,
    )

    assert audit.status == "ready"
    assert audit.blockers == ()
    assert audit.daily_coverage["eligible"].tolist() == [True, True]
    assert audit.daily_coverage["complete_instruments"].tolist() == [1, 1]
    assert len(audit.candidate_signal_days) == 1

    receipt_path = write_pandadata_coverage_receipt(
        audit, output_dir=tmp_path / "coverage"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "ready"
    assert receipt["summary"]["candidate_signal_day_count"] == 1
    assert len(receipt["audit_id"]) == 16

    cli = subprocess.run(
        [
            sys.executable,
            "tasks/audit_pandadata_coverage.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "cli-coverage"),
            "--min-complete-instruments",
            "1",
            "--lookback-days",
            "1",
            "--max-horizon-days",
            "1",
            "--required-signal-days",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 0
    assert json.loads(cli.stdout)["status"] == "ready"

    runtime_manifest_path = materialize_pandadata_runtime_slice(
        manifest_path,
        output_dir=tmp_path / "runtime",
        top_n=1,
    )
    runtime_manifest = json.loads(
        runtime_manifest_path.read_text(encoding="utf-8")
    )
    runtime_bars = pd.read_parquet(
        runtime_manifest_path.parent / runtime_manifest["data_path"]
    )
    runtime_states = pd.read_parquet(
        runtime_manifest_path.parent / runtime_manifest["instrument_state_path"]
    )
    assert runtime_manifest["dataset_kind"] == "raw_1m"
    assert len(runtime_bars) == 480
    assert runtime_states["instrument_id"].tolist() == ["000001.XSHE"]
