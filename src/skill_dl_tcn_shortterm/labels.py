"""Next-open multi-horizon labels and cross-sectional rank targets."""

from __future__ import annotations

from typing import Any, Iterable, cast

import numpy as np
import pandas as pd

from .experiment import ContractError


def build_labels(
    window_index: pd.DataFrame,
    canonical_bars: pd.DataFrame,
    *,
    horizons: Iterable[int],
    corporate_actions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create raw holding returns and independent date/horizon rank targets."""

    daily = (
        canonical_bars.sort_values(["instrument_id", "bar_end_at"])
        .groupby(["instrument_id", "trade_date"], observed=True)
        .agg(
            open_price=("open", "first"),
            bar_count=("bar_end_at", "size"),
            all_ok=("quality_flag", lambda values: bool((values == "ok").all())),
        )
        .reset_index()
    )
    by_instrument = {
        instrument_id: group.sort_values("trade_date").reset_index(drop=True)
        for instrument_id, group in daily.groupby("instrument_id", observed=True)
    }
    action_by_instrument: dict[str, pd.DataFrame] = {}
    if corporate_actions is not None:
        required_actions = {"instrument_id", "effective_date"}
        if missing := required_actions - set(corporate_actions.columns):
            raise ContractError(f"corporate actions missing columns: {sorted(missing)}")
        actions = corporate_actions.copy()
        actions["effective_date"] = pd.to_datetime(
            actions["effective_date"], errors="raise"
        ).dt.date
        if "pit_reliable" not in actions:
            actions["pit_reliable"] = False
        action_by_instrument = {
            str(instrument_id): group
            for instrument_id, group in actions.groupby("instrument_id", observed=True)
        }
    rows: list[dict[str, Any]] = []
    for sample in window_index.itertuples(index=False):
        sample = cast(Any, sample)
        prices = by_instrument[sample.instrument_id]
        date_positions = {
            date: position for position, date in enumerate(prices["trade_date"])
        }
        signal_position = date_positions[sample.signal_date]
        for horizon in horizons:
            entry_position = signal_position + 1
            exit_position = entry_position + int(horizon)
            valid = exit_position < len(prices)
            raw_return = float("nan")
            entry_at: Any = pd.NaT
            label_end_at: Any = pd.NaT
            reason = ""
            if valid:
                entry = prices.iloc[entry_position]
                exit_row = prices.iloc[exit_position]
                crossing_actions = action_by_instrument.get(str(sample.instrument_id))
                has_unreliable_action = False
                if crossing_actions is not None:
                    entry_date = pd.Timestamp(entry["trade_date"]).date()
                    exit_date = pd.Timestamp(exit_row["trade_date"]).date()
                    has_unreliable_action = bool(
                        crossing_actions.loc[
                            (crossing_actions["effective_date"] >= entry_date)
                            & (crossing_actions["effective_date"] <= exit_date),
                            "pit_reliable",
                        ]
                        .eq(False)
                        .any()
                    )
                if has_unreliable_action:
                    valid = False
                    reason = "corporate_action_without_pit_adjustment"
                elif (
                    int(entry["bar_count"]) != 48
                    or int(exit_row["bar_count"]) != 48
                    or not bool(entry["all_ok"])
                    or not bool(exit_row["all_ok"])
                    or float(entry["open_price"]) <= 0
                    or float(exit_row["open_price"]) <= 0
                ):
                    valid = False
                    reason = "invalid_execution_price"
                else:
                    raw_return = float(
                        exit_row["open_price"] / entry["open_price"] - 1.0
                    )
                    entry_at = pd.Timestamp(
                        f"{entry['trade_date']} 09:30", tz="Asia/Shanghai"
                    )
                    label_end_at = pd.Timestamp(
                        f"{exit_row['trade_date']} 09:30", tz="Asia/Shanghai"
                    )
            else:
                reason = "insufficient_future"
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_position": int(sample.sample_position),
                    "instrument_id": sample.instrument_id,
                    "signal_date": sample.signal_date,
                    "horizon": int(horizon),
                    "entry_at": entry_at,
                    "label_end_at": label_end_at,
                    "raw_return": raw_return,
                    "rank_target": float("nan"),
                    "valid": bool(valid),
                    "missing_reason": reason,
                    "label_version": "next-open-rank-v2",
                }
            )
    labels = pd.DataFrame(rows)
    if labels.empty:
        return labels
    valid_labels = labels.loc[labels["valid"]]
    for _, group in valid_labels.groupby(["signal_date", "horizon"], observed=True):
        if len(group) == 1:
            targets = pd.Series(0.0, index=group.index)
        else:
            ranks = group["raw_return"].rank(method="average")
            percentile = (ranks - 1.0) / (len(group) - 1.0)
            targets = 2.0 * percentile - 1.0
        labels.loc[group.index, "rank_target"] = targets.astype("float64")
    labels["valid"] = labels["valid"] & np.isfinite(labels["rank_target"])
    return labels
