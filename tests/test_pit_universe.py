from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from skill_dl_tcn_shortterm import run_experiment
from skill_dl_tcn_shortterm.universe import build_pit_universe


def _complete_day(instrument_id: str, day: str, base: float) -> pd.DataFrame:
    ends = pd.date_range(
        f"{day} 09:31", f"{day} 11:30", freq="1min", tz="Asia/Shanghai"
    ).append(
        pd.date_range(f"{day} 13:01", f"{day} 15:00", freq="1min", tz="Asia/Shanghai")
    )
    return pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "bar_end_at": ends,
            "open": base,
            "high": base + 0.1,
            "low": base - 0.1,
            "close": base + 0.02,
            "volume": 100.0,
            "amount": 1000.0,
            "quality_flag": "ok",
        }
    )


def test_future_security_state_does_not_change_historical_pit_universe(
    tmp_path: Path,
) -> None:
    raw = pd.concat(
        [
            _complete_day("600000.XSHG", "2024-01-02", 10.0),
            _complete_day("000001.XSHE", "2024-01-02", 12.0),
        ],
        ignore_index=True,
    ).sort_values(["instrument_id", "bar_end_at"])
    bars_path = tmp_path / "bars_1m.parquet"
    raw.to_parquet(bars_path, index=False)

    states = pd.DataFrame(
        [
            {
                "instrument_id": "600000.XSHG",
                "effective_at": "2023-01-01T00:00:00+08:00",
                "exchange": "XSHG",
                "security_type": "A_SHARE",
                "listed_date": "2000-01-01",
                "delisted_date": None,
                "is_st": False,
                "is_delisting": False,
                "is_suspended": False,
            },
            {
                "instrument_id": "600000.XSHG",
                "effective_at": "2024-01-03T00:00:00+08:00",
                "exchange": "XSHG",
                "security_type": "A_SHARE",
                "listed_date": "2000-01-01",
                "delisted_date": None,
                "is_st": True,
                "is_delisting": False,
                "is_suspended": False,
            },
            {
                "instrument_id": "000001.XSHE",
                "effective_at": "2023-01-01T00:00:00+08:00",
                "exchange": "XSHE",
                "security_type": "A_SHARE",
                "listed_date": "1991-01-01",
                "delisted_date": None,
                "is_st": True,
                "is_delisting": False,
                "is_suspended": False,
            },
        ]
    )
    states_path = tmp_path / "instrument_states.parquet"
    states.to_parquet(states_path, index=False)

    manifest = {
        "schema_version": 1,
        "dataset_kind": "raw_1m",
        "data_path": bars_path.name,
        "data_sha256": hashlib.sha256(bars_path.read_bytes()).hexdigest(),
        "instrument_state_path": states_path.name,
        "instrument_state_sha256": hashlib.sha256(states_path.read_bytes()).hexdigest(),
        "timezone": "Asia/Shanghai",
        "price_unit": "CNY",
        "volume_unit": "share",
        "amount_unit": "CNY",
        "source_version": "synthetic-v1",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_experiment(
        config={"run_name": "pit", "seed": 7, "horizons": [1, 2, 3, 5]},
        manifest_path=manifest_path,
        output_root=tmp_path / "runs",
    )

    assert result.universe_path is not None
    universe = pd.read_parquet(result.universe_path).set_index("instrument_id")
    assert bool(universe.loc["600000.XSHG", "eligible"])
    assert universe.loc["600000.XSHG", "exclusion_reasons"] == ""
    assert not bool(universe.loc["000001.XSHE", "eligible"])
    assert universe.loc["000001.XSHE", "exclusion_reasons"] == "st"

    predictions = pd.read_parquet(result.predictions_path)
    assert predictions["instrument_id"].unique().tolist() == ["600000.XSHG"]


def test_configured_pit_liquidity_threshold_uses_asof_adv20() -> None:
    day = "2024-01-02"
    canonical = pd.concat(
        [
            pd.DataFrame(
                {
                    "instrument_id": instrument,
                    "trade_date": [pd.Timestamp(day).date()] * 48,
                    "bar_end_at": pd.date_range(
                        f"{day} 09:35", periods=48, freq="5min", tz="Asia/Shanghai"
                    ),
                    "quality_flag": ["ok"] * 48,
                }
            )
            for instrument in ["A", "B"]
        ],
        ignore_index=True,
    )
    states = pd.DataFrame(
        [
            {
                "instrument_id": instrument,
                "effective_at": "2024-01-01T00:00:00+08:00",
                "exchange": "XSHG",
                "security_type": "A_SHARE",
                "listed_date": "2000-01-01",
                "delisted_date": None,
                "is_st": False,
                "is_delisting": False,
                "is_suspended": False,
                "adv20": adv20,
            }
            for instrument, adv20 in [("A", 100.0), ("B", 1_000.0)]
        ]
    )

    universe = build_pit_universe(canonical, states, {"min_adv20": 500.0}).set_index(
        "instrument_id"
    )

    assert not bool(universe.loc["A", "eligible"])
    assert universe.loc["A", "exclusion_reasons"] == "liquidity"
    assert bool(universe.loc["B", "eligible"])
