from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd

from skill_dl_tcn_shortterm.backtest import build_executable_long_only


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    label_rows = []
    for date_number, signal_date in enumerate(["2024-01-02", "2024-01-03"]):
        for instrument_number in range(10):
            sample_id = f"{signal_date}-I{instrument_number}"
            for horizon in [1, 2]:
                prediction_rows.append(
                    {
                        "model": "ridge",
                        "fold": 0,
                        "stage": "validation",
                        "sample_id": sample_id,
                        "instrument_id": f"I{instrument_number}",
                        "signal_date": signal_date,
                        "horizon": horizon,
                        "score": float(instrument_number),
                        "target": -1.0,
                    }
                )
                entry = pd.Timestamp("2024-01-03", tz="Asia/Shanghai") + pd.Timedelta(
                    days=date_number
                )
                label_rows.append(
                    {
                        "sample_id": sample_id,
                        "instrument_id": f"I{instrument_number}",
                        "signal_date": signal_date,
                        "horizon": horizon,
                        "entry_at": entry + pd.Timedelta(hours=9, minutes=30),
                        "label_end_at": entry
                        + pd.Timedelta(days=horizon, hours=9, minutes=30),
                        "raw_return": instrument_number / 100.0,
                        "rank_target": -1.0,
                        "valid": True,
                    }
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(label_rows)


def test_long_only_vintages_are_horizon_specific_and_use_raw_returns() -> None:
    predictions, labels = _fixture()

    result = build_executable_long_only(predictions, labels, top_fraction=0.10)

    long_orders = result.orders.loc[
        result.orders["portfolio_type"] == "executable_long_only"
    ]
    assert set(long_orders["instrument_id"]) == {"I9"}
    assert set(long_orders["horizon"]) == {1, 2}
    assert long_orders.groupby(["signal_date", "horizon"]).size().eq(1).all()
    assert long_orders.loc[long_orders["horizon"] == 1, "planned_weight"].eq(1.0).all()
    assert long_orders.loc[long_orders["horizon"] == 2, "planned_weight"].eq(0.5).all()
    assert (
        result.vintages.loc[result.vintages["horizon"] == 1, "capital_fraction"]
        .eq(1.0)
        .all()
    )
    assert (
        result.vintages.loc[result.vintages["horizon"] == 2, "capital_fraction"]
        .eq(0.5)
        .all()
    )
    assert math.isclose(long_orders.iloc[0]["raw_return"], 0.09)
    assert long_orders["raw_return"].ne(long_orders["target"]).all()
    horizon_two_ledger = result.portfolio_ledger.loc[
        result.portfolio_ledger["horizon"] == 2
    ]
    assert horizon_two_ledger["active_vintage_count"].max() == 2
    assert horizon_two_ledger["active_capital_fraction"].max() == 1.0
    assert horizon_two_ledger["cash_fraction"].min() == 0.0
    assert (
        result.portfolio_ledger.loc[
            result.portfolio_ledger["horizon"] == 1, "active_vintage_count"
        ].max()
        == 1
    )
    assert (
        horizon_two_ledger.loc[
            horizon_two_ledger["realized_vintage_count"] == 0,
            "realized_pnl_contribution",
        ]
        .eq(0.0)
        .all()
    )
    assert np.allclose(result.vintages["pit_equal_weight_return"], 0.045)
    metrics = result.metrics.set_index("horizon")
    assert math.isclose(
        float(
            cast(
                Any,
                metrics.loc[1, "cumulative_pit_equal_weight_return_contribution"],
            )
        ),
        0.09,
    )
    assert math.isclose(
        float(cast(Any, metrics.loc[2, "cumulative_excess_return_contribution"])),
        0.045,
    )


def test_long_short_diagnostic_is_kept_separate_from_executable_orders() -> None:
    predictions, labels = _fixture()

    result = build_executable_long_only(predictions, labels, top_fraction=0.10)

    assert set(result.orders["portfolio_type"]) == {
        "executable_long_only",
        "pit_equal_weight_benchmark",
    }
    assert set(result.diagnostic["portfolio_type"]) == {"diagnostic_long_short"}
    assert set(result.diagnostic["leg"]) == {"long", "short"}
    assert set(
        result.diagnostic.loc[result.diagnostic["leg"] == "long", "instrument_id"]
    ) == {"I9"}
    assert set(
        result.diagnostic.loc[result.diagnostic["leg"] == "short", "instrument_id"]
    ) == {"I0"}


def test_security_level_turnover_nets_matching_exit_and_entry_weights() -> None:
    predictions, labels = _fixture()

    result = build_executable_long_only(predictions, labels, top_fraction=0.10)

    horizon_one = result.portfolio_ledger.loc[
        result.portfolio_ledger["horizon"].eq(1)
    ].sort_values("event_at")
    rebalance = horizon_one.loc[
        horizon_one["entry_capital_fraction"].gt(0)
        & horizon_one["exit_capital_fraction"].gt(0)
    ].iloc[0]
    assert rebalance["legacy_vintage_flow_turnover"] == 1.0
    assert rebalance["buy_turnover"] == 0.0
    assert rebalance["sell_turnover"] == 0.0
    assert rebalance["one_way_turnover"] == 0.0
    assert rebalance["traded_notional_turnover"] == 0.0
    continued = result.portfolio_holdings.loc[
        result.portfolio_holdings["event_at"].eq(rebalance["event_at"])
        & result.portfolio_holdings["horizon"].eq(1)
        & result.portfolio_holdings["instrument_id"].eq("I9")
    ].iloc[0]
    assert continued["previous_target_weight"] == 1.0
    assert continued["target_weight"] == 1.0
    assert continued["weight_delta"] == 0.0


def test_incumbent_buffer_is_a_causal_portfolio_policy() -> None:
    predictions, labels = _fixture()
    second_date = predictions["signal_date"].eq("2024-01-03")
    predictions.loc[second_date & predictions["instrument_id"].eq("I8"), "score"] = 10.0

    raw = build_executable_long_only(predictions, labels, top_fraction=0.10)
    buffered = build_executable_long_only(
        predictions,
        labels,
        top_fraction=0.10,
        incumbent_buffer_fraction=0.20,
    )

    raw_second = raw.orders.loc[
        raw.orders["portfolio_type"].eq("executable_long_only")
        & raw.orders["signal_date"].eq("2024-01-03")
    ]
    buffered_second = buffered.orders.loc[
        buffered.orders["portfolio_type"].eq("executable_long_only")
        & buffered.orders["signal_date"].eq("2024-01-03")
    ]
    assert set(raw_second["instrument_id"]) == {"I8"}
    assert set(buffered_second["instrument_id"]) == {"I9"}
