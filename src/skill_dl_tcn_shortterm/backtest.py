"""Portfolio construction for horizon-specific long-only vintages."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, cast

import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class BacktestResult:
    orders: pd.DataFrame
    vintages: pd.DataFrame
    portfolio_ledger: pd.DataFrame
    portfolio_holdings: pd.DataFrame
    metrics: pd.DataFrame
    diagnostic: pd.DataFrame


def _validate_inputs(predictions: pd.DataFrame, labels: pd.DataFrame) -> None:
    prediction_columns = {
        "model",
        "fold",
        "sample_id",
        "instrument_id",
        "signal_date",
        "horizon",
        "score",
    }
    label_columns = {
        "sample_id",
        "horizon",
        "entry_at",
        "label_end_at",
        "raw_return",
        "valid",
    }
    missing_predictions = prediction_columns - set(predictions.columns)
    missing_labels = label_columns - set(labels.columns)
    if missing_predictions:
        raise ContractError(
            f"predictions missing columns: {sorted(missing_predictions)}"
        )
    if missing_labels:
        raise ContractError(f"labels missing columns: {sorted(missing_labels)}")


def build_executable_long_only(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    top_fraction: float = 0.10,
    incumbent_buffer_fraction: float = 0.0,
) -> BacktestResult:
    """Build independent, capital-sliced vintages from next-open predictions.

    A horizon ``h`` starts one new vintage each signal day. Each vintage receives
    ``1 / h`` of portfolio capital and equally weights its selected securities.
    The raw next-open holding return, never the rank target, drives P&L.
    """

    _validate_inputs(predictions, labels)
    if not 0.0 < top_fraction <= 0.5:
        raise ContractError("top_fraction must be in (0, 0.5]")
    if not 0.0 <= incumbent_buffer_fraction <= 1.0:
        raise ContractError("incumbent_buffer_fraction must be in [0, 1]")
    valid_labels = labels.loc[labels["valid"]].copy()
    label_columns = [
        "sample_id",
        "horizon",
        "entry_at",
        "label_end_at",
        "raw_return",
    ]
    merged = predictions.merge(
        valid_labels[label_columns],
        on=["sample_id", "horizon"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged.loc[merged["score"].notna() & merged["raw_return"].notna()].copy()
    if merged.empty:
        raise ContractError("no predictions have valid executable labels")

    group_columns = ["model", "fold", "signal_date", "horizon"]
    order_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    vintage_rows: list[dict[str, object]] = []
    previous_selections: dict[tuple[str, int, int], set[str]] = {}
    for keys, group in merged.groupby(group_columns, observed=True, sort=True):
        model, fold, signal_date, horizon = cast(tuple[Any, Any, Any, Any], keys)
        ordered = group.sort_values(
            ["score", "instrument_id"], ascending=[False, True], kind="mergesort"
        )
        selection_count = max(1, ceil(len(ordered) * top_fraction))
        selection_key = (str(model), int(fold), int(horizon))
        previous = previous_selections.get(selection_key, set())
        if incumbent_buffer_fraction > 0.0 and previous:
            buffer_count = max(1, ceil(selection_count * incumbent_buffer_fraction))
            eligible = ordered.head(selection_count + buffer_count)
            incumbent_mask = eligible["instrument_id"].astype(str).isin(previous)
            retained = eligible.loc[incumbent_mask].head(selection_count)
            fill_count = selection_count - len(retained)
            fill = ordered.loc[
                ~ordered["instrument_id"].astype(str).isin(
                    set(retained["instrument_id"].astype(str))
                )
            ].head(fill_count)
            long_leg = pd.concat([retained, fill], ignore_index=False)
        else:
            long_leg = ordered.head(selection_count)
        previous_selections[selection_key] = {
            str(value) for value in long_leg["instrument_id"].tolist()
        }
        capital_fraction = 1.0 / int(horizon)
        planned_weight = capital_fraction / selection_count
        benchmark_returns = valid_labels.loc[
            (valid_labels["signal_date"] == signal_date)
            & (valid_labels["horizon"] == horizon),
            "raw_return",
        ].dropna()
        if benchmark_returns.empty:
            raise ContractError("PIT equal-weight benchmark has no valid returns")
        pit_equal_weight_return = float(benchmark_returns.mean())
        vintage_id = f"{model}|{fold}|{signal_date}|h{horizon}"
        short_leg = ordered.tail(selection_count)
        for row in long_leg.itertuples(index=False):
            row = cast(Any, row)
            record = row._asdict()
            record.update(
                {
                    "vintage_id": vintage_id,
                    "portfolio_type": "executable_long_only",
                    "planned_weight": planned_weight,
                    "gross_pnl": planned_weight * float(row.raw_return),
                    "selection_policy": (
                        "raw_topk"
                        if incumbent_buffer_fraction == 0.0
                        else "incumbent_buffer"
                    ),
                    "incumbent_buffer_fraction": incumbent_buffer_fraction,
                }
            )
            order_rows.append(record)
        benchmark_weight = capital_fraction / len(ordered)
        benchmark_vintage_id = f"{vintage_id}|pit-equal-weight"
        for row in ordered.itertuples(index=False):
            row = cast(Any, row)
            record = row._asdict()
            record.update(
                {
                    "vintage_id": benchmark_vintage_id,
                    "portfolio_type": "pit_equal_weight_benchmark",
                    "planned_weight": benchmark_weight,
                    "gross_pnl": benchmark_weight * float(row.raw_return),
                }
            )
            order_rows.append(record)
        for leg, selected, sign in [
            ("long", long_leg, 1.0),
            ("short", short_leg, -1.0),
        ]:
            for row in selected.itertuples(index=False):
                row = cast(Any, row)
                diagnostic_rows.append(
                    {
                        **row._asdict(),
                        "portfolio_type": "diagnostic_long_short",
                        "leg": leg,
                        "diagnostic_weight": sign / selection_count,
                        "diagnostic_pnl": sign
                        * float(row.raw_return)
                        / selection_count,
                    }
                )
        vintage_rows.append(
            {
                "vintage_id": vintage_id,
                "model": model,
                "fold": int(fold),
                "signal_date": signal_date,
                "horizon": int(horizon),
                "entry_at": long_leg["entry_at"].min(),
                "planned_exit_at": long_leg["label_end_at"].max(),
                "capital_fraction": capital_fraction,
                "selected_count": selection_count,
                "gross_return_contribution": sum(
                    planned_weight * float(value) for value in long_leg["raw_return"]
                ),
                "pit_equal_weight_return": pit_equal_weight_return,
                "pit_equal_weight_return_contribution": (
                    capital_fraction * pit_equal_weight_return
                ),
            }
        )

    orders = pd.DataFrame(order_rows)
    vintages = pd.DataFrame(vintage_rows)
    diagnostic = pd.DataFrame(diagnostic_rows)
    executable_orders = orders.loc[
        orders["portfolio_type"].eq("executable_long_only")
    ].copy()
    positions_by_vintage: dict[str, dict[str, float]] = {}
    for raw_vintage_id, group in executable_orders.groupby(
        "vintage_id", observed=True
    ):
        positions: dict[str, float] = {}
        for raw_row in group.itertuples(index=False):
            row = cast(Any, raw_row)
            positions[str(row.instrument_id)] = float(row.planned_weight)
        positions_by_vintage[str(raw_vintage_id)] = positions
    ledger_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    for keys, group in vintages.groupby(["model", "fold", "horizon"], observed=True):
        model, fold, horizon = cast(tuple[Any, Any, Any], keys)
        group = group.copy()
        group["entry_at"] = pd.to_datetime(group["entry_at"], utc=True).dt.tz_convert(
            "Asia/Shanghai"
        )
        group["planned_exit_at"] = pd.to_datetime(
            group["planned_exit_at"], utc=True
        ).dt.tz_convert("Asia/Shanghai")
        event_times = sorted(
            set(group["entry_at"].tolist()) | set(group["planned_exit_at"].tolist())
        )
        active: dict[str, dict[str, float]] = {}
        previous_holdings: dict[str, float] = {}
        cumulative_pnl = 0.0
        for event_at in event_times:
            exiting = group.loc[group["planned_exit_at"] == event_at]
            entering = group.loc[group["entry_at"] == event_at]
            realized_pnl = float(exiting["gross_return_contribution"].sum())
            for vintage_id in exiting["vintage_id"]:
                active.pop(str(vintage_id), None)
            for vintage in entering.itertuples(index=False):
                vintage = cast(Any, vintage)
                active[str(vintage.vintage_id)] = positions_by_vintage[
                    str(vintage.vintage_id)
                ]
            target_holdings: dict[str, float] = {}
            for vintage_positions in active.values():
                for instrument_id, weight in vintage_positions.items():
                    target_holdings[instrument_id] = (
                        target_holdings.get(instrument_id, 0.0) + weight
                    )
            buy_turnover = 0.0
            sell_turnover = 0.0
            for instrument_id in sorted(set(previous_holdings) | set(target_holdings)):
                previous_weight = previous_holdings.get(instrument_id, 0.0)
                target_weight = target_holdings.get(instrument_id, 0.0)
                delta = target_weight - previous_weight
                buy_weight = max(delta, 0.0)
                sell_weight = max(-delta, 0.0)
                buy_turnover += buy_weight
                sell_turnover += sell_weight
                holdings_rows.append(
                    {
                        "model": model,
                        "fold": int(fold),
                        "horizon": int(horizon),
                        "event_at": event_at,
                        "instrument_id": instrument_id,
                        "previous_target_weight": previous_weight,
                        "target_weight": target_weight,
                        "weight_delta": delta,
                        "buy_turnover": buy_weight,
                        "sell_turnover": sell_weight,
                    }
                )
            entry_capital = float(entering["capital_fraction"].sum())
            exit_capital = float(exiting["capital_fraction"].sum())
            exposure = float(sum(target_holdings.values()))
            cumulative_pnl += realized_pnl
            ledger_rows.append(
                {
                    "model": model,
                    "fold": int(fold),
                    "horizon": int(horizon),
                    "event_at": event_at,
                    "active_vintage_count": len(active),
                    "active_capital_fraction": exposure,
                    "cash_fraction": max(0.0, 1.0 - exposure),
                    "entry_capital_fraction": entry_capital,
                    "exit_capital_fraction": exit_capital,
                    "legacy_vintage_flow_turnover": max(
                        entry_capital, exit_capital
                    ),
                    "buy_turnover": buy_turnover,
                    "sell_turnover": sell_turnover,
                    "one_way_turnover": max(buy_turnover, sell_turnover),
                    "traded_notional_turnover": buy_turnover + sell_turnover,
                    "realized_vintage_count": len(exiting),
                    "realized_pnl_contribution": realized_pnl,
                    "cumulative_pnl_contribution": cumulative_pnl,
                    "equity": 1.0 + cumulative_pnl,
                }
            )
            previous_holdings = target_holdings
    portfolio_ledger = pd.DataFrame(ledger_rows)
    portfolio_holdings = pd.DataFrame(holdings_rows)
    metric_rows = []
    for keys, group in vintages.groupby(["model", "fold", "horizon"], observed=True):
        model, fold, horizon = cast(tuple[Any, Any, Any], keys)
        contributions = group["gross_return_contribution"]
        benchmark_contributions = group["pit_equal_weight_return_contribution"]
        group_ledger = portfolio_ledger.loc[
            (portfolio_ledger["model"] == model)
            & (portfolio_ledger["fold"] == fold)
            & (portfolio_ledger["horizon"] == horizon)
        ].sort_values("event_at")
        equity = group_ledger["equity"]
        drawdown = equity / equity.cummax() - 1.0
        metric_rows.append(
            {
                "model": model,
                "fold": int(fold),
                "horizon": int(horizon),
                "portfolio_type": "executable_long_only",
                "vintage_count": len(group),
                "mean_gross_return_contribution": float(contributions.mean()),
                "cumulative_gross_return_contribution": float(contributions.sum()),
                "cumulative_pit_equal_weight_return_contribution": float(
                    benchmark_contributions.sum()
                ),
                "cumulative_excess_return_contribution": float(
                    contributions.sum() - benchmark_contributions.sum()
                ),
                "final_equity": float(equity.iloc[-1]),
                "max_drawdown": float(drawdown.min()),
                "mean_one_way_turnover": float(group_ledger["one_way_turnover"].mean()),
                "cumulative_traded_notional_turnover": float(
                    group_ledger["traded_notional_turnover"].sum()
                ),
                "mean_gross_exposure": float(
                    group_ledger["active_capital_fraction"].mean()
                ),
                "maximum_concurrent_vintages": int(
                    group_ledger["active_vintage_count"].max()
                ),
                "minimum_cash_fraction": float(group_ledger["cash_fraction"].min()),
            }
        )
    return BacktestResult(
        orders=orders,
        vintages=vintages,
        portfolio_ledger=portfolio_ledger,
        portfolio_holdings=portfolio_holdings,
        metrics=pd.DataFrame(metric_rows),
        diagnostic=diagnostic,
    )
