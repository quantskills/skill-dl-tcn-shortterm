"""Auditable A-share execution, transaction-cost, and capacity simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class ExecutionResult:
    ledger: pd.DataFrame
    scenario_metrics: pd.DataFrame


def _t_plus_one_enabled(entry_date: pd.Timestamp, rules: Mapping[str, Any]) -> bool:
    applicable = []
    for item in rules.get("t_plus_one", []):
        effective = pd.Timestamp(item["effective_from"])
        if effective.date() <= entry_date.date():
            applicable.append((effective, bool(item["enabled"])))
    return max(applicable, key=lambda item: item[0])[1] if applicable else False


def _validate(
    orders: pd.DataFrame,
    state: pd.DataFrame,
    capital: float,
    capacity_fraction: float,
    rules: Mapping[str, Any],
) -> None:
    order_columns = {
        "vintage_id",
        "model",
        "fold",
        "sample_id",
        "instrument_id",
        "horizon",
        "entry_at",
        "label_end_at",
        "planned_weight",
    }
    state_columns = {
        "instrument_id",
        "trade_date",
        "open_price",
        "adv20",
        "buyable",
        "sellable",
        "price_source",
    }
    if missing := order_columns - set(orders.columns):
        raise ContractError(f"orders missing columns: {sorted(missing)}")
    if missing := state_columns - set(state.columns):
        raise ContractError(f"execution state missing columns: {sorted(missing)}")
    if capital <= 0:
        raise ContractError("capital must be positive")
    if not 0 < capacity_fraction <= 1:
        raise ContractError("capacity_fraction must be in (0, 1]")
    if not isinstance(rules.get("version"), str) or not rules["version"]:
        raise ContractError("rules.version must be a non-empty string")


def _normalize_cost_schedule(
    schedule: Sequence[Mapping[str, Any]] | None,
    *,
    commission_bps: float | None,
    sell_tax_bps: float | None,
    base_slippage_bps: float | None,
    cost_version: str,
) -> list[dict[str, Any]]:
    if schedule is None:
        if commission_bps is None or sell_tax_bps is None or base_slippage_bps is None:
            raise ContractError(
                "commission, sell tax, and base slippage are required without a cost schedule"
            )
        schedule = [
            {
                "version": cost_version,
                "effective_from": "1900-01-01",
                "commission_bps": commission_bps,
                "sell_tax_bps": sell_tax_bps,
                "slippage_bps": base_slippage_bps,
            }
        ]
    normalized: list[dict[str, Any]] = []
    required = {
        "version",
        "effective_from",
        "commission_bps",
        "sell_tax_bps",
        "slippage_bps",
    }
    for raw in schedule:
        if missing := required - set(raw):
            raise ContractError(
                f"cost schedule entry missing fields: {sorted(missing)}"
            )
        version = raw["version"]
        if not isinstance(version, str) or not version:
            raise ContractError("cost schedule version must be a non-empty string")
        try:
            effective_at = pd.Timestamp(raw["effective_from"])
            if pd.isna(effective_at):
                raise ValueError("effective_from")
            effective_at = (
                effective_at.tz_localize("Asia/Shanghai")
                if effective_at.tzinfo is None
                else effective_at.tz_convert("Asia/Shanghai")
            )
            commission = float(raw["commission_bps"])
            sell_tax = float(raw["sell_tax_bps"])
            slippage = float(raw["slippage_bps"])
            if any(
                not np.isfinite(value) or value < 0
                for value in [commission, sell_tax, slippage]
            ):
                raise ValueError("negative or non-finite rate")
            entry: dict[str, Any] = {
                "version": version,
                "effective_at": effective_at,
                "commission_bps": commission,
                "sell_tax_bps": sell_tax,
                "slippage_bps": slippage,
            }
        except (TypeError, ValueError) as exc:
            raise ContractError("cost schedule values are invalid") from exc
        normalized.append(entry)
    normalized.sort(key=lambda item: item["effective_at"])
    if len({item["effective_at"] for item in normalized}) != len(normalized):
        raise ContractError("cost schedule effective dates must be unique")
    return normalized


def _cost_at(
    at: pd.Timestamp, schedule: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    timestamp = pd.Timestamp(at)
    timestamp = (
        timestamp.tz_localize("Asia/Shanghai")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("Asia/Shanghai")
    )
    candidates = [item for item in schedule if item["effective_at"] <= timestamp]
    if not candidates:
        raise ContractError(f"no cost schedule is effective at {timestamp.isoformat()}")
    return candidates[-1]


def _simulate_order(
    order: Any,
    states: pd.DataFrame,
    *,
    capital: float,
    capacity_fraction: float,
    cost_schedule: Sequence[Mapping[str, Any]],
    slippage_override_bps: float | None,
    rules: Mapping[str, Any],
    buy_budget: float | None = None,
) -> dict[str, Any]:
    entry_at = pd.Timestamp(order.entry_at)
    planned_sell_at = pd.Timestamp(order.label_end_at)
    buy_cost = _cost_at(entry_at, cost_schedule)
    buy_commission_rate = float(buy_cost["commission_bps"]) / 10_000.0
    buy_slippage_bps = (
        float(buy_cost["slippage_bps"])
        if slippage_override_bps is None
        else float(slippage_override_bps)
    )
    buy_slippage_rate = buy_slippage_bps / 10_000.0
    entry_date = entry_at.date()
    planned_sell_date = planned_sell_at.date()
    instrument_states = states.loc[
        states["instrument_id"] == order.instrument_id
    ].copy()
    buy_rows = instrument_states.loc[
        instrument_states["trade_date"].dt.date == entry_date
    ]
    planned_buy = capital * float(order.planned_weight)
    budgeted_buy = min(
        planned_buy,
        max(0.0, planned_buy if buy_budget is None else float(buy_budget)),
    )
    base = {
        **order._asdict(),
        "planned_buy_at": entry_at,
        "planned_sell_at": planned_sell_at,
        "planned_buy_amount": planned_buy,
        "budgeted_buy_amount": budgeted_buy,
        "cash_constrained_amount": planned_buy - budgeted_buy,
        "rule_version": rules["version"],
        "cost_version": str(buy_cost["version"]),
        "buy_cost_version": str(buy_cost["version"]),
        "sell_cost_versions": "",
        "slippage_scenario": (
            "scheduled"
            if slippage_override_bps is None
            else f"stress-{float(slippage_override_bps):g}bps"
        ),
        "slippage_bps": buy_slippage_bps,
        "buy_slippage_bps": buy_slippage_bps,
        "buy_commission_bps": float(buy_cost["commission_bps"]),
    }
    if buy_rows.empty:
        return {
            **base,
            "actual_buy_at": pd.NaT,
            "actual_sell_at": pd.NaT,
            "filled_buy_amount": 0.0,
            "capacity_clipped_amount": 0.0,
            "unused_cash": planned_buy,
            "buy_price": np.nan,
            "sell_price": np.nan,
            "shares": 0.0,
            "sell_delay_sessions": 0,
            "commission_cost": 0.0,
            "buy_commission_cost": 0.0,
            "sell_commission_cost": 0.0,
            "sell_tax_cost": 0.0,
            "slippage_cost": 0.0,
            "buy_slippage_cost": 0.0,
            "sell_slippage_cost": 0.0,
            "sell_proceeds": 0.0,
            "unresolved_value": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "unfilled_reason": "missing_buy_state",
            "price_source": "",
        }
    buy = buy_rows.iloc[0]
    buy_reason = buy.get("buy_block_reason", "buy_unavailable")
    if not bool(buy["buyable"]) or float(buy["open_price"]) <= 0:
        return {
            **base,
            "actual_buy_at": pd.NaT,
            "actual_sell_at": pd.NaT,
            "filled_buy_amount": 0.0,
            "capacity_clipped_amount": 0.0,
            "unused_cash": planned_buy,
            "buy_price": float(buy["open_price"]),
            "sell_price": np.nan,
            "shares": 0.0,
            "sell_delay_sessions": 0,
            "commission_cost": 0.0,
            "buy_commission_cost": 0.0,
            "sell_commission_cost": 0.0,
            "sell_tax_cost": 0.0,
            "slippage_cost": 0.0,
            "buy_slippage_cost": 0.0,
            "sell_slippage_cost": 0.0,
            "sell_proceeds": 0.0,
            "unresolved_value": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "unfilled_reason": str(buy_reason),
            "price_source": str(buy["price_source"]),
        }
    buy_capacity = max(0.0, float(buy["adv20"]) * capacity_fraction)
    filled_buy = min(budgeted_buy, buy_capacity)
    clipped = budgeted_buy - filled_buy
    shares = filled_buy / float(buy["open_price"])

    t_plus_one = _t_plus_one_enabled(entry_at, rules)
    candidates = instrument_states.loc[
        instrument_states["trade_date"].dt.date >= planned_sell_date
    ].sort_values("trade_date")
    if t_plus_one:
        candidates = candidates.loc[candidates["trade_date"].dt.date > entry_date]
    remaining_shares = shares
    sell_proceeds = 0.0
    sold_shares = 0.0
    sell_commission = 0.0
    sell_tax = 0.0
    sell_slippage = 0.0
    sell_cost_versions: list[str] = []
    sell_slippage_rates: list[float] = []
    actual_sell_row: pd.Series | None = None
    price_sources = [str(buy["price_source"])]
    for _, candidate in candidates.iterrows():
        if not bool(candidate["sellable"]) or float(candidate["open_price"]) <= 0:
            continue
        sell_capacity = max(0.0, float(candidate["adv20"]) * capacity_fraction)
        shares_today = min(
            remaining_shares, sell_capacity / float(candidate["open_price"])
        )
        if shares_today <= 0:
            continue
        candidate_proceeds = shares_today * float(candidate["open_price"])
        candidate_at = pd.Timestamp(
            f"{pd.Timestamp(candidate['trade_date']).date()} 09:30",
            tz="Asia/Shanghai",
        )
        sell_cost = _cost_at(candidate_at, cost_schedule)
        sell_slippage_bps = (
            float(sell_cost["slippage_bps"])
            if slippage_override_bps is None
            else float(slippage_override_bps)
        )
        sell_commission += (
            float(sell_cost["commission_bps"]) / 10_000.0 * candidate_proceeds
        )
        sell_tax += float(sell_cost["sell_tax_bps"]) / 10_000.0 * candidate_proceeds
        sell_slippage += sell_slippage_bps / 10_000.0 * candidate_proceeds
        sell_cost_versions.append(str(sell_cost["version"]))
        sell_slippage_rates.append(sell_slippage_bps)
        sell_proceeds += candidate_proceeds
        sold_shares += shares_today
        remaining_shares -= shares_today
        actual_sell_row = candidate
        price_sources.append(str(candidate["price_source"]))
        if remaining_shares <= 1e-12:
            break
    unresolved_value = remaining_shares * (
        float(actual_sell_row["open_price"])
        if actual_sell_row is not None
        else float(buy["open_price"])
    )
    buy_commission = buy_commission_rate * filled_buy
    commission = buy_commission + sell_commission
    buy_slippage = buy_slippage_rate * filled_buy
    slippage = buy_slippage + sell_slippage
    gross_pnl = sell_proceeds + unresolved_value - filled_buy
    net_pnl = gross_pnl - commission - sell_tax - slippage
    if actual_sell_row is None:
        actual_sell_at: Any = pd.NaT
        sell_price = np.nan
        delay = 0
        reason = "no_sellable_session"
    else:
        sell_date = pd.Timestamp(actual_sell_row["trade_date"])
        actual_sell_at = pd.Timestamp(f"{sell_date.date()} 09:30", tz=entry_at.tz)
        sell_price = sell_proceeds / sold_shares
        session_dates = instrument_states.loc[
            (instrument_states["trade_date"].dt.date >= planned_sell_date)
            & (instrument_states["trade_date"].dt.date <= sell_date.date()),
            "trade_date",
        ].dt.date.unique()
        delay = max(0, len(session_dates) - 1)
        reason = "partial_sell_unresolved" if remaining_shares > 1e-12 else ""
    unique_sell_versions = list(dict.fromkeys(sell_cost_versions))
    all_cost_versions = list(
        dict.fromkeys([str(buy_cost["version"]), *unique_sell_versions])
    )
    return {
        **base,
        "cost_version": "->".join(all_cost_versions),
        "sell_cost_versions": ";".join(unique_sell_versions),
        "sell_slippage_bps": (
            ";".join(f"{value:g}" for value in dict.fromkeys(sell_slippage_rates))
            if sell_slippage_rates
            else ""
        ),
        "actual_buy_at": entry_at,
        "actual_sell_at": actual_sell_at,
        "filled_buy_amount": filled_buy,
        "capacity_clipped_amount": clipped,
        "unused_cash": planned_buy - filled_buy,
        "buy_price": float(buy["open_price"]),
        "sell_price": sell_price,
        "shares": shares,
        "sold_shares": sold_shares,
        "unresolved_shares": remaining_shares,
        "sell_delay_sessions": delay,
        "commission_cost": commission,
        "buy_commission_cost": buy_commission,
        "sell_commission_cost": sell_commission,
        "sell_tax_cost": sell_tax,
        "slippage_cost": slippage,
        "buy_slippage_cost": buy_slippage,
        "sell_slippage_cost": sell_slippage,
        "sell_proceeds": sell_proceeds,
        "unresolved_value": unresolved_value,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "unfilled_reason": reason,
        "price_source": "+".join(dict.fromkeys(price_sources)),
    }


def simulate_a_share_execution(
    orders: pd.DataFrame,
    execution_state: pd.DataFrame,
    *,
    capital: float,
    capacity_fraction: float,
    rules: Mapping[str, Any],
    commission_bps: float | None = None,
    sell_tax_bps: float | None = None,
    base_slippage_bps: float | None = None,
    stress_slippage_bps: Sequence[float] = (5.0, 10.0, 20.0),
    cost_version: str = "inline-costs",
    cost_schedule: Sequence[Mapping[str, Any]] | None = None,
) -> ExecutionResult:
    """Simulate next-open A-share fills with mandatory slippage stresses."""

    _validate(orders, execution_state, capital, capacity_fraction, rules)
    if not isinstance(cost_version, str) or not cost_version:
        raise ContractError("cost_version must be a non-empty string")
    normalized_costs = _normalize_cost_schedule(
        cost_schedule,
        commission_bps=commission_bps,
        sell_tax_bps=sell_tax_bps,
        base_slippage_bps=base_slippage_bps,
        cost_version=cost_version,
    )
    state = execution_state.copy()
    state["trade_date"] = pd.to_datetime(state["trade_date"])
    if state.duplicated(["instrument_id", "trade_date"]).any():
        raise ContractError(
            "execution state must be unique by instrument and trade_date"
        )
    if (state[["open_price", "adv20"]].astype(float) < 0).any(axis=None):
        raise ContractError("execution prices and ADV20 cannot be negative")
    scenarios: list[float | None]
    if cost_schedule is None:
        assert base_slippage_bps is not None
        scenarios = []
        scenarios.extend(
            sorted(
                {
                    float(base_slippage_bps),
                    *(float(value) for value in stress_slippage_bps),
                }
            )
        )
    else:
        scenarios = [None, *sorted({float(value) for value in stress_slippage_bps})]
    ledger_rows = []
    working_orders = orders.copy()
    if "portfolio_type" not in working_orders:
        working_orders["portfolio_type"] = "executable_long_only"
    for slippage_bps in scenarios:
        for _, portfolio_orders in working_orders.groupby(
            ["portfolio_type", "model", "fold", "horizon"],
            observed=True,
            sort=True,
        ):
            available_cash = float(capital)
            pending_releases: list[tuple[pd.Timestamp, float]] = []
            ordered = portfolio_orders.sort_values(
                ["entry_at", "vintage_id", "instrument_id"], kind="mergesort"
            )
            for _, vintage_orders in ordered.groupby(
                ["entry_at", "vintage_id"], observed=True, sort=False
            ):
                entry_at = pd.Timestamp(vintage_orders["entry_at"].iloc[0])
                entry_cost = _cost_at(entry_at, normalized_costs)
                commission_rate = float(entry_cost["commission_bps"]) / 10_000.0
                entry_slippage_bps = (
                    float(entry_cost["slippage_bps"])
                    if slippage_bps is None
                    else float(slippage_bps)
                )
                slippage_rate = entry_slippage_bps / 10_000.0
                due = [item for item in pending_releases if item[0] <= entry_at]
                pending_releases = [
                    item for item in pending_releases if item[0] > entry_at
                ]
                available_cash += sum(amount for _, amount in due)
                cash_after_settlement = available_cash
                planned = capital * vintage_orders["planned_weight"].astype(float)
                planned_total = float(planned.sum())
                maximum_notional = available_cash / (
                    1.0 + commission_rate + slippage_rate
                )
                budget_total = min(planned_total, maximum_notional)
                if planned_total > 0:
                    budgets = planned / planned_total * budget_total
                else:
                    budgets = planned * 0.0
                vintage_results = []
                for order, buy_budget in zip(
                    vintage_orders.itertuples(index=False), budgets, strict=True
                ):
                    vintage_results.append(
                        _simulate_order(
                            order,
                            state,
                            capital=capital,
                            capacity_fraction=capacity_fraction,
                            cost_schedule=normalized_costs,
                            slippage_override_bps=slippage_bps,
                            rules=rules,
                            buy_budget=float(buy_budget),
                        )
                    )
                buy_cash_used = sum(
                    float(row["filled_buy_amount"])
                    + float(row["buy_commission_cost"])
                    + float(row["buy_slippage_cost"])
                    for row in vintage_results
                )
                available_cash = max(0.0, available_cash - buy_cash_used)
                for row in vintage_results:
                    row["portfolio_cash_after_settlement"] = cash_after_settlement
                    row["portfolio_cash_after_buy"] = available_cash
                    sell_at = row["actual_sell_at"]
                    if pd.notna(sell_at) and float(row["sell_proceeds"]) > 0:
                        release = (
                            float(row["sell_proceeds"])
                            - float(row["sell_commission_cost"])
                            - float(row["sell_tax_cost"])
                            - float(row["sell_slippage_cost"])
                        )
                        pending_releases.append((pd.Timestamp(sell_at), release))
                ledger_rows.extend(vintage_results)
    ledger = pd.DataFrame(ledger_rows)
    metrics = []
    metric_keys = [
        "portfolio_type",
        "model",
        "fold",
        "horizon",
        "slippage_scenario",
    ]
    for key, group in ledger.groupby(metric_keys, observed=True):
        portfolio_type, model, fold, horizon, slippage_scenario = cast(
            tuple[Any, Any, Any, Any, Any], key
        )
        net_pnl = float(group["net_pnl"].sum())
        metrics.append(
            {
                "portfolio_type": portfolio_type,
                "model": model,
                "fold": fold,
                "horizon": horizon,
                "slippage_scenario": slippage_scenario,
                "slippage_bps": float(group["slippage_bps"].mean()),
                "planned_buy_amount": float(group["planned_buy_amount"].sum()),
                "filled_buy_amount": float(group["filled_buy_amount"].sum()),
                "unused_cash": float(group["unused_cash"].sum()),
                "cash_constrained_amount": float(
                    group["cash_constrained_amount"].sum()
                ),
                "minimum_portfolio_cash": float(
                    group["portfolio_cash_after_buy"].min()
                ),
                "commission_cost": float(group["commission_cost"].sum()),
                "sell_tax_cost": float(group["sell_tax_cost"].sum()),
                "slippage_cost": float(group["slippage_cost"].sum()),
                "gross_pnl": float(group["gross_pnl"].sum()),
                "net_pnl": net_pnl,
                "net_return": net_pnl / capital,
                "ending_value": capital + net_pnl,
            }
        )
    scenario_metrics = pd.DataFrame(metrics)
    join_keys = ["model", "fold", "horizon", "slippage_scenario"]
    benchmark = scenario_metrics.loc[
        scenario_metrics["portfolio_type"] == "pit_equal_weight_benchmark",
        [*join_keys, "net_return", "net_pnl"],
    ].rename(
        columns={
            "net_return": "benchmark_net_return",
            "net_pnl": "benchmark_net_pnl",
        }
    )
    scenario_metrics = scenario_metrics.merge(
        benchmark,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    scenario_metrics["excess_net_return"] = (
        scenario_metrics["net_return"] - scenario_metrics["benchmark_net_return"]
    )
    scenario_metrics["excess_net_pnl"] = (
        scenario_metrics["net_pnl"] - scenario_metrics["benchmark_net_pnl"]
    )
    return ExecutionResult(ledger=ledger, scenario_metrics=scenario_metrics)
