from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.duckdb_source import (
    audit_duckdb_training_coverage,
    write_training_coverage_receipt,
)


def _create_coverage_database(path: Path) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE stocks_1m (
                exchange VARCHAR,
                symbol VARCHAR,
                eob TIMESTAMPTZ,
                trade_date DATE,
                _source_file VARCHAR
            )
            """
        )
        rows: list[tuple[str, str, str, str, str | None]] = []
        for day in range(2, 10):
            trade_date = f"2024-01-{day:02d}"
            for symbol, exchange in (("600000", "SHSE"), ("000001", "SZSE")):
                source_file = "supplement.zip" if day == 5 and symbol == "000001" else None
                for minute in range(3):
                    rows.append(
                        (
                            exchange,
                            symbol,
                            f"{trade_date} 09:{31 + minute}:00+08:00",
                            trade_date,
                            source_file,
                        )
                    )
        connection.executemany("INSERT INTO stocks_1m VALUES (?, ?, ?, ?, ?)", rows)
    finally:
        connection.close()


def test_researcher_can_find_only_contiguous_candidate_signal_days(
    tmp_path: Path,
) -> None:
    database = tmp_path / "coverage.duckdb"
    _create_coverage_database(database)

    audit = audit_duckdb_training_coverage(
        database,
        start_date="2024-01-02",
        end_date="2024-01-09",
        min_instruments=2,
        min_average_bars=3.0,
        min_primary_source_ratio=0.75,
        lookback_days=2,
        max_horizon_days=1,
        train_signal_days=2,
        validation_signal_days=1,
        ordinary_test_signal_days=1,
    )

    rejected = audit.daily_coverage.loc[
        ~audit.daily_coverage["eligible"],
        ["trade_date", "primary_source_ratio", "rejection_reasons"],
    ]
    assert rejected["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-05"
    ]
    assert rejected["primary_source_ratio"].tolist() == [0.5]
    assert rejected["rejection_reasons"].tolist() == [
        "too_little_primary_source"
    ]
    assert audit.eligible_runs["day_count"].tolist() == [3, 4]
    assert audit.candidate_signal_days["signal_date"].dt.strftime(
        "%Y-%m-%d"
    ).tolist() == ["2024-01-03", "2024-01-07", "2024-01-08"]
    assert audit.status == "blocked"
    assert audit.blockers == ("insufficient_candidate_signal_days",)


def test_researcher_can_write_an_immutable_coverage_receipt(tmp_path: Path) -> None:
    database = tmp_path / "coverage.duckdb"
    _create_coverage_database(database)
    audit = audit_duckdb_training_coverage(
        database,
        start_date="2024-01-02",
        end_date="2024-01-09",
        min_instruments=2,
        min_average_bars=3.0,
        min_primary_source_ratio=0.75,
        lookback_days=2,
        max_horizon_days=1,
        train_signal_days=2,
        validation_signal_days=1,
        ordinary_test_signal_days=1,
    )

    receipt_path = write_training_coverage_receipt(
        audit,
        output_dir=tmp_path / "receipt",
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_path.name == "coverage-receipt.json"
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == ["insufficient_candidate_signal_days"]
    assert receipt["summary"] == {
        "candidate_signal_day_count": 3,
        "eligible_date_count": 7,
        "eligible_run_count": 2,
        "longest_eligible_run_days": 4,
        "required_signal_day_count": 4,
        "total_date_count": 8,
        "total_row_count": 48,
        "unique_instrument_count_peak": 2,
    }
    assert len(receipt["audit_id"]) == 16
    for name, digest in receipt["artifacts"].items():
        artifact = receipt_path.parent / name
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    assert str(database.resolve()) not in receipt_path.read_text(encoding="utf-8")

    with pytest.raises(ContractError, match="refuses to overwrite"):
        write_training_coverage_receipt(
            audit,
            output_dir=tmp_path / "receipt",
        )


def test_cli_reports_a_blocked_audit_without_treating_it_as_an_error(
    tmp_path: Path,
) -> None:
    database = tmp_path / "coverage.duckdb"
    _create_coverage_database(database)
    output_dir = tmp_path / "cli-receipt"
    command = [
        sys.executable,
        "tasks/audit_duckdb_training_coverage.py",
        "--database",
        str(database),
        "--output-dir",
        str(output_dir),
        "--start-date",
        "2024-01-02",
        "--end-date",
        "2024-01-09",
        "--min-instruments",
        "2",
        "--min-average-bars",
        "3",
        "--min-primary-source-ratio",
        "0.75",
        "--lookback-days",
        "2",
        "--max-horizon-days",
        "1",
        "--train-signal-days",
        "2",
        "--validation-signal-days",
        "1",
        "--ordinary-test-signal-days",
        "1",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["status"] == "blocked"
    assert payload["candidate_signal_day_count"] == 3
    assert Path(payload["receipt_path"]).is_file()

    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    error = json.loads(repeated.stdout)
    assert repeated.returncode == 2
    assert error["status"] == "error"
    assert "refuses to overwrite" in error["error"]
