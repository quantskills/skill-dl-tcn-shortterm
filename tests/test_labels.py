from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skill_dl_tcn_shortterm import run_experiment
from skill_dl_tcn_shortterm.labels import build_labels


def _bars_for_prices(
    instrument_id: str, days: pd.DatetimeIndex, daily_opens: list[float]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day, daily_open in zip(days, daily_opens, strict=True):
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
        frames.append(
            pd.DataFrame(
                {
                    "instrument_id": instrument_id,
                    "bar_end_at": ends,
                    "open": price,
                    "high": price + 0.01,
                    "low": price - 0.01,
                    "close": price + 0.001,
                    "volume": 100.0,
                    "amount": 100.0 * price,
                    "quality_flag": "ok",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_next_open_labels_and_cross_sectional_ranks_are_correct(tmp_path: Path) -> None:
    days = pd.bdate_range("2024-01-02", periods=16)
    raw = pd.concat(
        [
            _bars_for_prices("600000.XSHG", days, [10.0 + i for i in range(16)]),
            _bars_for_prices("000001.XSHE", days, [30.0 - i for i in range(16)]),
        ],
        ignore_index=True,
    ).sort_values(["instrument_id", "bar_end_at"])
    bars_path = tmp_path / "bars.parquet"
    raw.to_parquet(bars_path, index=False)
    state_rows = []
    for instrument_id, exchange in [("600000.XSHG", "XSHG"), ("000001.XSHE", "XSHE")]:
        state_rows.append(
            {
                "instrument_id": instrument_id,
                "effective_at": "2000-01-01T00:00:00+08:00",
                "exchange": exchange,
                "security_type": "A_SHARE",
                "listed_date": "1990-01-01",
                "delisted_date": None,
                "is_st": False,
                "is_delisting": False,
                "is_suspended": False,
            }
        )
    states_path = tmp_path / "states.parquet"
    pd.DataFrame(state_rows).to_parquet(states_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_kind": "raw_1m",
                "data_path": bars_path.name,
                "data_sha256": hashlib.sha256(bars_path.read_bytes()).hexdigest(),
                "instrument_state_path": states_path.name,
                "instrument_state_sha256": hashlib.sha256(
                    states_path.read_bytes()
                ).hexdigest(),
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
        config={
            "run_name": "labels",
            "seed": 7,
            "horizons": [1, 2, 3, 5],
            "lookback_days": 10,
        },
        manifest_path=manifest_path,
        output_root=tmp_path / "runs",
    )

    assert result.labels_path is not None
    labels = pd.read_parquet(result.labels_path)
    signal_date = days[9].strftime("%Y-%m-%d")
    first = labels.loc[
        (labels["signal_date"] == signal_date) & (labels["horizon"] == 1)
    ].set_index("instrument_id")
    assert bool(first.loc["600000.XSHG", "valid"])
    assert bool(first.loc["000001.XSHE", "valid"])
    assert first.loc["600000.XSHG", "raw_return"] == pytest.approx(21.0 / 20.0 - 1.0)
    assert first.loc["000001.XSHE", "raw_return"] == pytest.approx(19.0 / 20.0 - 1.0)
    assert first.loc["600000.XSHG", "rank_target"] == 1.0
    assert first.loc["000001.XSHE", "rank_target"] == -1.0
    assert pd.Timestamp(str(first.loc["600000.XSHG", "entry_at"])).strftime(
        "%Y-%m-%d %H:%M"
    ) == (days[10].strftime("%Y-%m-%d") + " 09:30")

    five_day = labels.loc[
        (labels["signal_date"] == signal_date)
        & (labels["instrument_id"] == "600000.XSHG")
        & (labels["horizon"] == 5)
    ].iloc[0]
    assert (
        five_day["label_end_at"].strftime("%Y-%m-%d %H:%M")
        == days[15].strftime("%Y-%m-%d") + " 09:30"
    )


def test_unreliable_crossing_corporate_action_invalidates_label() -> None:
    rows = []
    days = pd.bdate_range("2024-01-02", periods=3)
    for instrument in ["A", "B"]:
        for day_number, day in enumerate(days):
            for bar_number in range(48):
                rows.append(
                    {
                        "instrument_id": instrument,
                        "trade_date": day.date(),
                        "bar_end_at": pd.Timestamp(day, tz="Asia/Shanghai")
                        + pd.Timedelta(hours=9, minutes=35 + 5 * bar_number),
                        "open": 10.0 + day_number,
                        "quality_flag": "ok",
                    }
                )
    index = pd.DataFrame(
        {
            "sample_id": ["A-sample", "B-sample"],
            "sample_position": [0, 1],
            "instrument_id": ["A", "B"],
            "signal_date": [days[0].date(), days[0].date()],
        }
    )

    labels = build_labels(
        index,
        pd.DataFrame(rows),
        horizons=[1],
        corporate_actions=pd.DataFrame(
            {
                "instrument_id": ["A"],
                "effective_date": [days[1].date()],
                "pit_reliable": [False],
            }
        ),
    ).set_index("instrument_id")

    assert not bool(labels.loc["A", "valid"])
    assert (
        labels.loc["A", "missing_reason"] == "corporate_action_without_pit_adjustment"
    )
    assert bool(labels.loc["B", "valid"])
