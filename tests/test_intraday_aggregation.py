from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from skill_dl_tcn_shortterm import run_experiment
from skill_dl_tcn_shortterm.market_data import aggregate_five_minute_bars


def _minute_ends(day: str) -> pd.DatetimeIndex:
    morning = pd.date_range(
        f"{day} 09:31", f"{day} 11:30", freq="1min", tz="Asia/Shanghai"
    )
    afternoon = pd.date_range(
        f"{day} 13:01", f"{day} 15:00", freq="1min", tz="Asia/Shanghai"
    )
    return pd.DatetimeIndex(morning.append(afternoon))


def test_researcher_can_aggregate_a_complete_a_share_session(tmp_path: Path) -> None:
    timestamps = _minute_ends("2024-01-02")
    prices = pd.Series(range(len(timestamps)), dtype="float64") / 100 + 10.0
    raw = pd.DataFrame(
        {
            "instrument_id": "600000.XSHG",
            "bar_end_at": timestamps,
            "open": prices,
            "high": prices + 0.10,
            "low": prices - 0.10,
            "close": prices + 0.02,
            "volume": 10.0,
            "amount": 100.0,
            "quality_flag": "ok",
        }
    )
    data_path = tmp_path / "bars_1m.parquet"
    raw.to_parquet(data_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_kind": "raw_1m",
                "data_path": data_path.name,
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "timezone": "Asia/Shanghai",
                "price_unit": "CNY",
                "volume_unit": "share",
                "amount_unit": "CNY",
                "source_version": "synthetic-v1",
            }
        ),
        encoding="utf-8",
    )

    result = run_experiment(
        config={"run_name": "aggregate", "seed": 7, "horizons": [1, 2, 3, 5]},
        manifest_path=manifest_path,
        output_root=tmp_path / "runs",
    )

    assert result.canonical_bars_path is not None
    bars_path = result.canonical_bars_path
    assert bars_path is not None
    bars = pd.read_parquet(bars_path)
    assert len(bars) == 48
    assert bars["bar_end_at"].dt.strftime("%H:%M").tolist()[23:25] == ["11:30", "13:05"]

    first = bars.iloc[0]
    assert first["bar_end_at"].strftime("%H:%M") == "09:35"
    assert first["open"] == 10.0
    assert first["high"] == pytest.approx(10.14)
    assert first["low"] == pytest.approx(9.9)
    assert first["close"] == pytest.approx(10.06)
    assert first["volume"] == 50.0
    assert first["amount"] == 500.0
    assert first["quality_flag"] == "ok"

    assert result.quality_path is not None
    quality = json.loads(result.quality_path.read_text(encoding="utf-8"))
    assert quality["canonical_bar_count"] == 48
    assert quality["incomplete_bar_count"] == 0
    assert quality["complete_session_count"] == 1


def test_source_quality_reasons_are_preserved_in_canonical_bars(
    tmp_path: Path,
) -> None:
    timestamps = _minute_ends("2024-01-02")[:5]
    raw = pd.DataFrame(
        {
            "instrument_id": "600000.XSHG",
            "bar_end_at": timestamps,
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "volume": 10.0,
            "amount": 100.0,
            "quality_flag": [
                "ok",
                "suspended",
                "vendor_missing",
                "ok",
                "suspended",
            ],
        }
    )
    data_path = tmp_path / "quality-bars.parquet"
    raw.to_parquet(data_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_kind": "raw_1m",
                "data_path": data_path.name,
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "timezone": "Asia/Shanghai",
                "price_unit": "CNY",
                "volume_unit": "share",
                "amount_unit": "CNY",
                "source_version": "synthetic-v1",
            }
        ),
        encoding="utf-8",
    )

    result = run_experiment(
        config={"run_name": "quality", "seed": 7, "horizons": [1, 2, 3, 5]},
        manifest_path=manifest_path,
        output_root=tmp_path / "runs",
    )

    bars_path = result.canonical_bars_path
    assert bars_path is not None
    bars = pd.read_parquet(bars_path)
    assert bars.loc[0, "source_quality_reasons"] == "suspended|vendor_missing"
    assert bars.loc[0, "quality_flag"] == "source_suspended|vendor_missing"


def test_auction_no_trade_fill_is_an_accepted_source_normalization() -> None:
    timestamps = _minute_ends("2024-01-02")
    raw = pd.DataFrame(
        {
            "instrument_id": "600000.XSHG",
            "bar_end_at": timestamps,
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "volume": 10.0,
            "amount": 100.0,
            "quality_flag": "ok",
        }
    )
    auction_fill = raw["bar_end_at"].dt.strftime("%H:%M").isin(
        ["14:58", "14:59"]
    )
    raw.loc[auction_fill, ["open", "high", "low", "close"]] = 10.0
    raw.loc[auction_fill, ["volume", "amount"]] = 0.0
    raw.loc[auction_fill, "quality_flag"] = "auction_no_trade_fill"

    canonical, quality = aggregate_five_minute_bars(
        raw,
        {
            "timezone": "Asia/Shanghai",
            "price_unit": "CNY",
            "volume_unit": "share",
            "amount_unit": "CNY",
            "source_version": "vendor-normalized-v1",
        },
    )

    assert len(canonical) == 48
    assert canonical["quality_flag"].unique().tolist() == ["ok"]
    assert canonical.iloc[-1]["source_quality_reasons"] == ""
    assert quality["incomplete_bar_count"] == 0
    assert quality["complete_session_count"] == 1
