from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skill_dl_tcn_shortterm import run_experiment


def _ten_complete_days() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day_number, day in enumerate(pd.bdate_range("2024-01-02", periods=10)):
        day_text = day.strftime("%Y-%m-%d")
        ends = pd.date_range(
            f"{day_text} 09:31", f"{day_text} 11:30", freq="1min", tz="Asia/Shanghai"
        ).append(
            pd.date_range(
                f"{day_text} 13:01",
                f"{day_text} 15:00",
                freq="1min",
                tz="Asia/Shanghai",
            )
        )
        minute = np.arange(len(ends), dtype="float64")
        price = 10.0 + day_number * 0.2 + minute * 0.001
        frames.append(
            pd.DataFrame(
                {
                    "instrument_id": "600000.XSHG",
                    "bar_end_at": ends,
                    "open": price,
                    "high": price + 0.05,
                    "low": price - 0.05,
                    "close": price + 0.01,
                    "volume": 100.0 + minute,
                    "amount": (100.0 + minute) * price,
                    "quality_flag": "ok",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_researcher_gets_a_causal_ten_day_feature_window(tmp_path: Path) -> None:
    raw = _ten_complete_days()
    bars_path = tmp_path / "bars.parquet"
    raw.to_parquet(bars_path, index=False)
    states_path = tmp_path / "states.parquet"
    pd.DataFrame(
        [
            {
                "instrument_id": "600000.XSHG",
                "effective_at": "2000-01-01T00:00:00+08:00",
                "exchange": "XSHG",
                "security_type": "A_SHARE",
                "listed_date": "2000-01-01",
                "delisted_date": None,
                "is_st": False,
                "is_delisting": False,
                "is_suspended": False,
            }
        ]
    ).to_parquet(states_path, index=False)
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
            "run_name": "features",
            "seed": 7,
            "horizons": [1, 2, 3, 5],
            "lookback_days": 10,
        },
        manifest_path=manifest_path,
        output_root=tmp_path / "runs",
    )

    assert result.window_index_path is not None
    assert result.windows_path is not None
    assert result.window_rejections_path is not None
    index = pd.read_parquet(result.window_index_path)
    assert len(index) == 1
    assert index.loc[0, "instrument_id"] == "600000.XSHG"
    assert index.loc[0, "signal_date"] == pd.bdate_range("2024-01-02", periods=10)[
        -1
    ].strftime("%Y-%m-%d")
    assert index.loc[0, "time_steps"] == 480
    assert index.loc[0, "window_version"] == "5m-10d-v1"

    with np.load(result.windows_path, allow_pickle=False) as windows:
        features = windows["features"]
        names = windows["feature_names"].tolist()
    assert features.shape == (1, 8, 480)
    assert names == [
        "close_return",
        "open_close_return",
        "intrabar_range",
        "log_volume",
        "log_amount",
        "vwap_deviation",
        "time_sin",
        "time_cos",
    ]
    assert np.isfinite(features).all()
    rejections = pd.read_parquet(result.window_rejections_path)
    assert len(rejections) == 9
    assert set(rejections["rejection_reason"]) == {"insufficient_history"}
    assert result.quality_path is not None
    quality = json.loads(result.quality_path.read_text(encoding="utf-8"))
    assert quality["valid_window_count"] == 1
    assert quality["rejected_window_count"] == 9
    assert quality["training_window_storage"] == "read_only_memmap"
