from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.duckdb_source import (
    audit_duckdb_trade_dates,
    export_duckdb_minute_slice,
)
from skill_dl_tcn_shortterm.market_data import aggregate_five_minute_bars


def _create_minute_database(path: Path) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE stocks_1m (
                exchange VARCHAR,
                symbol VARCHAR,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                amount DOUBLE,
                volume DOUBLE,
                bob TIMESTAMPTZ,
                eob TIMESTAMPTZ,
                type INTEGER,
                sequence INTEGER,
                trade_date DATE,
                _source_file VARCHAR
            )
            """
        )
        rows = []
        for symbol, exchange in (("600000", "SHSE"), ("000001", "SZSE")):
            for minute in range(1, 6):
                rows.append(
                    (
                        exchange,
                        symbol,
                        10.0,
                        10.0,
                        10.1,
                        9.9,
                        1000.0,
                        100.0,
                        f"2024-01-02 09:{29 + minute}:00+08:00",
                        f"2024-01-02 09:{30 + minute}:00+08:00",
                        11,
                        minute,
                        "2024-01-02",
                        "20240102.zip",
                    )
                )
        rows.extend(
            [
                (
                    "SZSE",
                    "000001",
                    10.0,
                    10.0,
                    10.1,
                    9.9,
                    1000.0,
                    100.0,
                    "2024-01-03 09:30:00+08:00",
                    "2024-01-03 09:31:00+08:00",
                    11,
                    1,
                    "2024-01-03",
                    "20240103.zip",
                ),
                (
                    "SHSE",
                    "900948",
                    1.0,
                    1.0,
                    1.1,
                    0.9,
                    100.0,
                    10.0,
                    "2024-01-03 09:30:00+08:00",
                    "2024-01-03 09:31:00+08:00",
                    11,
                    1,
                    "2024-01-03",
                    "20240103.zip",
                ),
            ]
        )
        connection.executemany(
            "INSERT INTO stocks_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        connection.close()


def _create_vendor_session_database(
    path: Path, *, missing_times: set[str]
) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE stocks_1m (
                exchange VARCHAR,
                symbol VARCHAR,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                amount DOUBLE,
                volume DOUBLE,
                bob TIMESTAMPTZ,
                eob TIMESTAMPTZ,
                type INTEGER,
                sequence INTEGER,
                trade_date DATE,
                _source_file VARCHAR
            )
            """
        )
        expected = pd.date_range(
            "2024-01-02 09:31",
            "2024-01-02 11:30",
            freq="1min",
            tz="Asia/Shanghai",
        ).append(
            pd.date_range(
                "2024-01-02 13:01",
                "2024-01-02 15:00",
                freq="1min",
                tz="Asia/Shanghai",
            )
        )
        observed = [
            value for value in expected if value.strftime("%H:%M") not in missing_times
        ]
        observed.extend(
            [
                pd.Timestamp("2024-01-02 09:30", tz="Asia/Shanghai"),
                pd.Timestamp("2024-01-02 11:31", tz="Asia/Shanghai"),
            ]
        )
        rows = []
        for sequence, end_at in enumerate(sorted(observed), start=1):
            rows.append(
                (
                    "SHSE",
                    "600000",
                    10.0,
                    10.0,
                    10.1,
                    9.9,
                    1000.0,
                    100.0,
                    end_at - pd.Timedelta(minutes=1),
                    end_at,
                    11,
                    sequence,
                    "2024-01-02",
                    "20240102.zip",
                )
            )
        connection.executemany(
            "INSERT INTO stocks_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        connection.close()


def test_researcher_can_audit_complete_a_share_trade_dates(tmp_path: Path) -> None:
    database = tmp_path / "minute.duckdb"
    _create_minute_database(database)

    audit = audit_duckdb_trade_dates(
        database,
        start_date="2024-01-02",
        end_date="2024-01-03",
        min_instruments=2,
        min_average_bars=5.0,
    )

    assert audit["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert audit["instrument_count"].tolist() == [2, 1]
    assert audit["row_count"].tolist() == [10, 1]
    assert audit["eligible"].tolist() == [True, False]
    assert audit.loc[1, "rejection_reasons"] == (
        "too_few_instruments|too_few_average_bars"
    )


def test_researcher_can_export_a_bounded_canonical_minute_slice(
    tmp_path: Path,
) -> None:
    database = tmp_path / "minute.duckdb"
    _create_minute_database(database)

    manifest_path = export_duckdb_minute_slice(
        database,
        output_dir=tmp_path / "export",
        trade_dates=["2024-01-02"],
        instrument_ids=["600000.XSHG", "000001.XSHE"],
        max_rows=20,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = manifest_path.parent / manifest["data_path"]
    bars = pd.read_parquet(data_path)

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
    assert bars["instrument_id"].drop_duplicates().tolist() == [
        "000001.XSHE",
        "600000.XSHG",
    ]
    assert bars["bar_end_at"].dt.tz is not None
    assert str(bars["bar_end_at"].dt.tz) == "Asia/Shanghai"
    assert bars["quality_flag"].unique().tolist() == ["ok"]
    assert manifest["dataset_kind"] == "raw_1m"
    assert manifest["source_table"] == "stocks_1m"
    assert manifest["data_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()


def test_export_normalizes_vendor_session_edges_and_auction_gaps(
    tmp_path: Path,
) -> None:
    database = tmp_path / "minute.duckdb"
    _create_vendor_session_database(
        database, missing_times={"14:58", "14:59"}
    )

    manifest_path = export_duckdb_minute_slice(
        database,
        output_dir=tmp_path / "normalized",
        trade_dates=["2024-01-02"],
        instrument_ids=["600000.XSHG"],
        max_rows=300,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bars = pd.read_parquet(manifest_path.parent / manifest["data_path"])
    normalization = manifest["normalization"]

    assert len(bars) == 240
    assert bars["bar_end_at"].dt.strftime("%H:%M").iloc[[0, -1]].tolist() == [
        "09:31",
        "15:00",
    ]
    assert bars.loc[
        bars["bar_end_at"].dt.strftime("%H:%M").isin(["14:58", "14:59"]),
        ["open", "high", "low", "close", "volume", "amount", "quality_flag"],
    ].to_dict("records") == [
        {
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 0.0,
            "amount": 0.0,
            "quality_flag": "auction_no_trade_fill",
        },
        {
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 0.0,
            "amount": 0.0,
            "quality_flag": "auction_no_trade_fill",
        },
    ]
    assert normalization == {
        "queried_row_count": 240,
        "dropped_out_of_session_rows": 2,
        "auction_no_trade_fill_rows": 2,
        "unresolved_missing_rows": 0,
        "exported_row_count": 240,
    }


def test_export_keeps_unknown_continuous_session_gaps_fail_closed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "minute.duckdb"
    _create_vendor_session_database(
        database, missing_times={"10:00", "14:58", "14:59"}
    )

    manifest_path = export_duckdb_minute_slice(
        database,
        output_dir=tmp_path / "unresolved",
        trade_dates=["2024-01-02"],
        instrument_ids=["600000.XSHG"],
        max_rows=300,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bars = pd.read_parquet(manifest_path.parent / manifest["data_path"])
    canonical, quality = aggregate_five_minute_bars(bars, manifest)

    assert "10:00" not in bars["bar_end_at"].dt.strftime("%H:%M").tolist()
    assert manifest["normalization"]["auction_no_trade_fill_rows"] == 2
    assert manifest["normalization"]["unresolved_missing_rows"] == 1
    assert len(canonical) == 48
    assert quality["incomplete_bar_count"] == 1
    assert quality["complete_session_count"] == 0


def test_export_refuses_a_slice_larger_than_the_declared_bound(tmp_path: Path) -> None:
    database = tmp_path / "minute.duckdb"
    _create_minute_database(database)

    with pytest.raises(ContractError, match="exceeds max_rows=9"):
        export_duckdb_minute_slice(
            database,
            output_dir=tmp_path / "export",
            trade_dates=["2024-01-02"],
            instrument_ids=["600000.XSHG", "000001.XSHE"],
            max_rows=9,
        )

    assert not (tmp_path / "export" / "bars-1m.parquet").exists()


def test_cli_exports_only_dates_that_pass_the_quality_gate(tmp_path: Path) -> None:
    database = tmp_path / "minute.duckdb"
    _create_minute_database(database)

    result = subprocess.run(
        [
            sys.executable,
            "tasks/export_duckdb_pilot.py",
            "--database",
            str(database),
            "--output-dir",
            str(tmp_path / "pilot"),
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-03",
            "--instrument",
            "600000.XSHG",
            "--instrument",
            "000001.XSHE",
            "--min-instruments",
            "2",
            "--min-average-bars",
            "5",
            "--max-rows",
            "20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["accepted_dates"] == ["2024-01-02"]
    assert payload["rejected_dates"] == ["2024-01-03"]
    assert Path(payload["manifest_path"]).is_file()
