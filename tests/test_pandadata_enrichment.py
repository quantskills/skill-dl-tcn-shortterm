from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.pandadata_source import (
    build_pandadata_causal_states,
    canonicalize_pandadata_adjustments,
    canonicalize_pandadata_daily,
    canonicalize_pandadata_share_float,
    materialize_pandadata_enriched_runtime,
    merge_pit_enrichment_frames,
)


def test_supplier_enrichment_frames_have_stable_pit_semantics() -> None:
    daily = canonicalize_pandadata_daily(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "date": ["20250102"],
                "name": ["*ST测试"],
                "close": [10.0],
                "amount": [1_000.0],
                "trade_status": [1],
            }
        )
    )
    assert daily.to_dict("records") == [
        {
            "instrument_id": "000001.XSHE",
            "trade_date": date(2025, 1, 2),
            "name": "*ST测试",
            "close": 10.0,
            "amount": 1000.0,
            "is_st": True,
            "is_suspended": True,
        }
    ]

    shares = canonicalize_pandadata_share_float(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "date": ["20250102"],
                "total": ["100"],
                "circulation_a": ["80"],
                "free_circulation": ["60"],
            }
        )
    )
    assert shares.loc[0, "known_date"] == date(2025, 1, 2)
    assert shares.loc[0, "total_shares"] == 100.0

    actions = canonicalize_pandadata_adjustments(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "ex_date": ["20250106"],
                "announcement_date": ["20250103"],
                "ex_cum_factor": [12.0],
                "ex_factor": [1.03],
                "ex_end_date": [None],
            }
        )
    )
    assert actions.loc[0, "effective_date"] == date(2025, 1, 6)
    assert actions.loc[0, "known_date"] == date(2025, 1, 3)
    assert not bool(actions.loc[0, "pit_reliable"])


def test_causal_states_do_not_backfill_future_amount_or_share_announcements() -> None:
    start = date(2025, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(22)]
    daily = pd.DataFrame(
        {
            "instrument_id": ["000001.XSHE"] * 22,
            "trade_date": dates,
            "name": ["测试"] * 20 + ["*ST测试", "*ST测试"],
            "close": [10.0] * 22,
            "amount": [float(value) for value in range(1, 22)] + [10_000.0],
            "is_st": [False] * 20 + [True, True],
            "is_suspended": [False] * 21 + [True],
        }
    )
    membership = daily[["trade_date", "instrument_id"]].copy()
    shares = pd.DataFrame(
        {
            "instrument_id": ["000001.XSHE", "000001.XSHE"],
            "known_date": [dates[9], dates[21]],
            "total_shares": [100.0, 999.0],
            "circulation_a": [80.0, 900.0],
            "free_circulation": [60.0, 800.0],
        }
    )

    states = build_pandadata_causal_states(
        daily, membership=membership, share_float=shares
    ).set_index(states_key := "effective_at")
    day_9 = pd.Timestamp(f"{dates[8]} 15:00", tz="Asia/Shanghai")
    day_20 = pd.Timestamp(f"{dates[19]} 15:00", tz="Asia/Shanghai")
    day_21 = pd.Timestamp(f"{dates[20]} 15:00", tz="Asia/Shanghai")
    assert pd.isna(states.loc[day_9, "market_cap"])
    assert states.loc[day_20, "market_cap"] == 1_000.0
    assert states.loc[day_20, "adv20"] == pytest.approx(10.5)
    assert states.loc[day_21, "adv20"] == pytest.approx(11.5)
    assert bool(states.loc[day_21, "is_st"])
    assert states_key == "effective_at"


def test_enriched_runtime_reuses_verified_bars_and_writes_action_contract(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    bars_path = base / "bars.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(bars_path, index=False)
    old_states = base / "old-states.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(old_states, index=False)
    base_manifest = base / "manifest.json"
    base_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_kind": "raw_1m",
                "data_path": bars_path.name,
                "data_sha256": hashlib.sha256(bars_path.read_bytes()).hexdigest(),
                "instrument_state_path": old_states.name,
                "instrument_state_sha256": hashlib.sha256(
                    old_states.read_bytes()
                ).hexdigest(),
                "timezone": "Asia/Shanghai",
                "price_unit": "CNY",
                "volume_unit": "share",
                "amount_unit": "CNY",
                "source_version": "fixture",
            }
        ),
        encoding="utf-8",
    )
    states = pd.DataFrame(
        {
            "instrument_id": ["000001.XSHE"],
            "effective_at": [pd.Timestamp("2025-01-02 15:00", tz="Asia/Shanghai")],
        }
    )
    actions = pd.DataFrame(
        {
            "instrument_id": ["000001.XSHE"],
            "effective_date": [date(2025, 1, 6)],
            "pit_reliable": [False],
        }
    )

    enriched = materialize_pandadata_enriched_runtime(
        base_manifest,
        states=states,
        corporate_actions=actions,
        output_dir=tmp_path / "enriched",
        enrichment_identity={"daily": "abc", "adjustments": "def"},
    )
    manifest = json.loads(enriched.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["enrichment"]["schema_version"] == 2
    reused_bars = (enriched.parent / manifest["data_path"]).resolve()
    assert reused_bars == bars_path.resolve()
    assert manifest["data_sha256"] == hashlib.sha256(bars_path.read_bytes()).hexdigest()
    assert manifest["corporate_action_sha256"]
    assert manifest["enrichment"]["industry_history"] == "unavailable"

    with pytest.raises(ContractError, match="refuses to overwrite"):
        materialize_pandadata_enriched_runtime(
            base_manifest,
            states=states,
            corporate_actions=actions,
            output_dir=tmp_path / "enriched",
            enrichment_identity={},
        )


def test_incremental_enrichment_merge_preserves_exact_rows_and_rejects_conflicts() -> None:
    reused = pd.DataFrame(
        {
            "instrument_id": ["000001.XSHE"],
            "trade_date": [date(2025, 1, 2)],
            "close": [10.0],
        }
    )
    fetched = pd.DataFrame(
        {
            "instrument_id": ["000002.XSHE"],
            "trade_date": [date(2025, 1, 2)],
            "close": [20.0],
        }
    )
    merged = merge_pit_enrichment_frames(
        reused,
        fetched,
        keys=["instrument_id", "trade_date"],
        name="daily",
    )
    assert merged[["instrument_id", "close"]].to_dict("records") == [
        {"instrument_id": "000001.XSHE", "close": 10.0},
        {"instrument_id": "000002.XSHE", "close": 20.0},
    ]

    identical = merge_pit_enrichment_frames(
        reused,
        reused.copy(),
        keys=["instrument_id", "trade_date"],
        name="daily",
    )
    assert identical.to_dict("records") == reused.to_dict("records")

    conflicting = reused.assign(close=11.0)
    with pytest.raises(ContractError, match="conflicting reused and fetched PIT keys"):
        merge_pit_enrichment_frames(
            reused,
            conflicting,
            keys=["instrument_id", "trade_date"],
            name="daily",
        )
