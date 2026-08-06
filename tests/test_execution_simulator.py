from __future__ import annotations

import math
from typing import Any, cast

import pandas as pd
import pytest

from skill_dl_tcn_shortterm.execution import simulate_a_share_execution


def _orders() -> pd.DataFrame:
    rows = []
    for instrument in ["A", "B", "C"]:
        rows.append(
            {
                "vintage_id": "v1",
                "portfolio_type": "executable_long_only",
                "model": "ridge",
                "fold": 0,
                "sample_id": f"sample-{instrument}",
                "instrument_id": instrument,
                "signal_date": "2024-01-01",
                "horizon": 1,
                "entry_at": pd.Timestamp("2024-01-02 09:30", tz="Asia/Shanghai"),
                "label_end_at": pd.Timestamp("2024-01-02 09:30", tz="Asia/Shanghai"),
                "planned_weight": 0.1,
                "score": 1.0,
                "raw_return": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _execution_state() -> pd.DataFrame:
    rows = []
    prices = {
        "A": [10.0, 10.5, 11.0],
        "B": [10.0, 10.0, 10.0],
        "C": [20.0, 22.0, 22.0],
    }
    for instrument in ["A", "B", "C"]:
        for position, date in enumerate(["2024-01-02", "2024-01-03", "2024-01-04"]):
            rows.append(
                {
                    "instrument_id": instrument,
                    "trade_date": date,
                    "open_price": prices[instrument][position],
                    "adv20": 1000.0 if instrument == "C" else 10000.0,
                    "buyable": not (instrument == "B" and position == 0),
                    "sellable": not (instrument == "A" and position == 1),
                    "price_source": "fixture-open",
                }
            )
    return pd.DataFrame(rows)


def test_execution_enforces_buy_availability_t1_delayed_exit_and_capacity() -> None:
    result = simulate_a_share_execution(
        _orders(),
        _execution_state(),
        capital=1000.0,
        commission_bps=2.0,
        sell_tax_bps=5.0,
        base_slippage_bps=5.0,
        capacity_fraction=0.05,
        rules={
            "version": "cn-equity-fixture-v1",
            "t_plus_one": [{"effective_from": "2020-01-01", "enabled": True}],
        },
    )

    base = result.ledger.loc[result.ledger["slippage_bps"] == 5.0].set_index(
        "instrument_id"
    )
    assert base.loc["B", "unfilled_reason"] == "buy_unavailable"
    assert base.loc["B", "filled_buy_amount"] == 0.0
    assert (
        pd.Timestamp(str(base.loc["A", "actual_sell_at"])).date().isoformat()
        == "2024-01-04"
    )
    assert base.loc["A", "sell_delay_sessions"] == 2
    assert base.loc["C", "planned_buy_amount"] == 100.0
    assert base.loc["C", "filled_buy_amount"] == 50.0
    assert base.loc["C", "capacity_clipped_amount"] == 50.0
    assert math.isclose(
        float(cast(Any, base.loc["C", "unused_cash"])),
        float(cast(Any, base.loc["C", "planned_buy_amount"]))
        - float(cast(Any, base.loc["C", "filled_buy_amount"])),
    )


def test_execution_reports_cost_breakdown_and_forced_stress_scenarios() -> None:
    result = simulate_a_share_execution(
        _orders(),
        _execution_state(),
        capital=1000.0,
        commission_bps=2.0,
        sell_tax_bps=5.0,
        base_slippage_bps=7.0,
        capacity_fraction=0.05,
        rules={
            "version": "cn-equity-fixture-v1",
            "t_plus_one": [{"effective_from": "2020-01-01", "enabled": True}],
        },
    )

    assert set(result.scenario_metrics["slippage_bps"]) == {5.0, 7.0, 10.0, 20.0}
    assert result.ledger["commission_cost"].ge(0.0).all()
    assert result.ledger["sell_tax_cost"].ge(0.0).all()
    assert result.ledger["slippage_cost"].ge(0.0).all()
    metrics = result.scenario_metrics.set_index("slippage_bps")
    stressed = metrics.loc[metrics.index == 20.0].iloc[0]
    base_case = metrics.loc[metrics.index == 5.0].iloc[0]
    assert float(stressed["net_pnl"]) < float(base_case["net_pnl"])
    assert float(base_case["ending_value"]) == 1000.0 + float(base_case["net_pnl"])
    assert set(result.ledger["rule_version"]) == {"cn-equity-fixture-v1"}


def test_execution_metrics_remain_separate_by_model_fold_and_horizon() -> None:
    orders = _orders()
    second = orders.copy()
    second["model"] = "other-model"
    second["fold"] = 1
    second["horizon"] = 2

    result = simulate_a_share_execution(
        pd.concat([orders, second], ignore_index=True),
        _execution_state(),
        capital=1000.0,
        commission_bps=2.0,
        sell_tax_bps=5.0,
        base_slippage_bps=5.0,
        capacity_fraction=0.05,
        rules={
            "version": "cn-equity-fixture-v1",
            "t_plus_one": [{"effective_from": "2020-01-01", "enabled": True}],
        },
        stress_slippage_bps=(5.0,),
    )

    keys = result.scenario_metrics[["model", "fold", "horizon"]]
    assert len(keys) == 2
    assert not keys.duplicated().any()


def test_execution_applies_cost_schedule_at_buy_and_each_sell_event() -> None:
    order = _orders().loc[lambda frame: frame["instrument_id"] == "A"].copy()

    result = simulate_a_share_execution(
        order,
        _execution_state(),
        capital=1000.0,
        capacity_fraction=1.0,
        rules={
            "version": "cn-equity-fixture-v1",
            "t_plus_one": [{"effective_from": "2020-01-01", "enabled": True}],
        },
        cost_schedule=[
            {
                "version": "costs-v1",
                "effective_from": "2024-01-01",
                "commission_bps": 0.0,
                "sell_tax_bps": 0.0,
                "slippage_bps": 0.0,
            },
            {
                "version": "costs-v2",
                "effective_from": "2024-01-03",
                "commission_bps": 100.0,
                "sell_tax_bps": 0.0,
                "slippage_bps": 0.0,
            },
        ],
        stress_slippage_bps=(0.0,),
    )

    scheduled = result.ledger.loc[
        result.ledger["slippage_scenario"] == "scheduled"
    ].iloc[0]
    assert scheduled["buy_cost_version"] == "costs-v1"
    assert scheduled["sell_cost_versions"] == "costs-v2"
    assert scheduled["cost_version"] == "costs-v1->costs-v2"
    assert float(scheduled["sell_commission_cost"]) > 0.0


def test_executable_equal_weight_benchmark_produces_net_excess_metrics() -> None:
    long_orders = _orders()
    benchmark_orders = long_orders.copy()
    benchmark_orders["portfolio_type"] = "pit_equal_weight_benchmark"
    benchmark_orders["vintage_id"] = "benchmark-v1"
    benchmark_orders["planned_weight"] = 1.0 / len(benchmark_orders)

    result = simulate_a_share_execution(
        pd.concat([long_orders, benchmark_orders], ignore_index=True),
        _execution_state(),
        capital=1000.0,
        commission_bps=2.0,
        sell_tax_bps=5.0,
        base_slippage_bps=5.0,
        capacity_fraction=0.05,
        rules={
            "version": "cn-equity-fixture-v1",
            "t_plus_one": [{"effective_from": "2020-01-01", "enabled": True}],
        },
        stress_slippage_bps=(5.0,),
    )

    long_metric = result.scenario_metrics.loc[
        result.scenario_metrics["portfolio_type"] == "executable_long_only"
    ].iloc[0]
    assert pd.notna(long_metric["benchmark_net_return"])
    assert float(long_metric["excess_net_return"]) == pytest.approx(
        float(long_metric["net_return"]) - float(long_metric["benchmark_net_return"])
    )


def test_delayed_exit_keeps_capital_unavailable_to_later_vintages() -> None:
    orders = pd.DataFrame(
        [
            {
                **_orders().iloc[0].to_dict(),
                "vintage_id": "v1",
                "instrument_id": "A",
                "sample_id": "first",
                "entry_at": pd.Timestamp("2024-01-02 09:30", tz="Asia/Shanghai"),
                "label_end_at": pd.Timestamp("2024-01-03 09:30", tz="Asia/Shanghai"),
                "planned_weight": 1.0,
            },
            {
                **_orders().iloc[0].to_dict(),
                "vintage_id": "v2",
                "instrument_id": "B",
                "sample_id": "second",
                "entry_at": pd.Timestamp("2024-01-03 09:30", tz="Asia/Shanghai"),
                "label_end_at": pd.Timestamp("2024-01-04 09:30", tz="Asia/Shanghai"),
                "planned_weight": 1.0,
            },
        ]
    )
    state = _execution_state()
    state.loc[
        (state["instrument_id"] == "A") & (state["trade_date"] == "2024-01-03"),
        "sellable",
    ] = False
    state.loc[
        (state["instrument_id"] == "B") & (state["trade_date"] == "2024-01-03"),
        "buyable",
    ] = True
    state["adv20"] = 1_000_000.0

    result = simulate_a_share_execution(
        orders,
        state,
        capital=1000.0,
        commission_bps=0.0,
        sell_tax_bps=0.0,
        base_slippage_bps=0.0,
        capacity_fraction=1.0,
        rules={
            "version": "cn-equity-fixture-v1",
            "t_plus_one": [{"effective_from": "2020-01-01", "enabled": True}],
        },
        stress_slippage_bps=(0.0,),
    )

    ledger = result.ledger.set_index("vintage_id")
    assert (
        pd.Timestamp(str(ledger.loc["v1", "actual_sell_at"])).date().isoformat()
        == "2024-01-04"
    )
    assert ledger.loc["v2", "filled_buy_amount"] == 0.0
    assert ledger.loc["v2", "cash_constrained_amount"] == 1000.0
    assert ledger.loc["v2", "portfolio_cash_after_settlement"] == 0.0
