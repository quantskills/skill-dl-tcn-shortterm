"""Causal feature windows for point-in-time samples."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd


FEATURE_NAMES = np.asarray(
    [
        "close_return",
        "open_close_return",
        "intrabar_range",
        "log_volume",
        "log_amount",
        "vwap_deviation",
        "time_sin",
        "time_cos",
    ]
)


@dataclass(frozen=True)
class _InstrumentHistory:
    """Columnar, date-addressable bars for one instrument."""

    dates: np.ndarray
    day_offsets: np.ndarray
    bar_end_at: pd.Series
    quality_flag: np.ndarray
    close: np.ndarray
    open_price: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    amount: np.ndarray


def _index_instrument_histories(
    canonical_bars: pd.DataFrame,
) -> dict[str, _InstrumentHistory]:
    histories: dict[str, _InstrumentHistory] = {}
    ordered = canonical_bars.sort_values(["instrument_id", "bar_end_at"])
    for instrument_id, raw_group in ordered.groupby(
        "instrument_id", observed=True, sort=False
    ):
        group = raw_group.reset_index(drop=True)
        raw_dates = pd.to_datetime(group["trade_date"]).to_numpy(
            dtype="datetime64[D]"
        )
        if len(raw_dates) == 0:
            continue
        day_starts = np.flatnonzero(
            np.concatenate(([True], raw_dates[1:] != raw_dates[:-1]))
        )
        unique_dates = raw_dates[day_starts]
        day_offsets = np.concatenate((day_starts, np.asarray([len(group)])))
        histories[str(instrument_id)] = _InstrumentHistory(
            dates=unique_dates,
            day_offsets=day_offsets,
            bar_end_at=group["bar_end_at"],
            quality_flag=group["quality_flag"].astype(str).to_numpy(),
            close=group["close"].to_numpy(dtype="float64"),
            open_price=group["open"].to_numpy(dtype="float64"),
            high=group["high"].to_numpy(dtype="float64"),
            low=group["low"].to_numpy(dtype="float64"),
            volume=group["volume"].to_numpy(dtype="float64"),
            amount=group["amount"].to_numpy(dtype="float64"),
        )
    return histories


def _build_feature_windows(
    canonical_bars: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    lookback_days: int,
    source_fingerprint: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build fixed causal windows for eligible universe snapshots."""

    expected_steps = lookback_days * 48
    window_version = f"5m-{lookback_days}d-v1"
    histories = _index_instrument_histories(canonical_bars)
    index_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    eligible = universe.loc[universe["eligible"]].sort_values(
        ["signal_date", "instrument_id"]
    )
    tensor = np.empty(
        (len(eligible), len(FEATURE_NAMES), expected_steps), dtype="float32"
    )
    step = np.arange(expected_steps, dtype="float64") % 48
    time_sin = np.sin(2.0 * np.pi * step / 48.0)
    time_cos = np.cos(2.0 * np.pi * step / 48.0)
    accepted = 0
    for sample in eligible.itertuples(index=False):
        history = histories.get(str(sample.instrument_id))
        if history is None:
            observed_days = 0
            observed_steps = 0
        else:
            signal_date = np.datetime64(str(sample.signal_date), "D")
            observed_days = int(
                np.searchsorted(history.dates, signal_date, side="right")
            )
            observed_steps = int(history.day_offsets[observed_days])
        if history is None or observed_days < lookback_days:
            rejection_rows.append(
                {
                    "instrument_id": sample.instrument_id,
                    "signal_date": sample.signal_date,
                    "rejection_reason": "insufficient_history",
                    "observed_days": observed_days,
                    "observed_steps": observed_steps,
                }
            )
            continue
        start = int(history.day_offsets[observed_days - lookback_days])
        stop = int(history.day_offsets[observed_days])
        observed_window_steps = stop - start
        quality_ok = bool((history.quality_flag[start:stop] == "ok").all())
        if observed_window_steps != expected_steps or not quality_ok:
            rejection_rows.append(
                {
                    "instrument_id": sample.instrument_id,
                    "signal_date": sample.signal_date,
                    "rejection_reason": (
                        "invalid_window_quality" if not quality_ok else "incomplete_window"
                    ),
                    "observed_days": lookback_days,
                    "observed_steps": observed_window_steps,
                }
            )
            continue

        close = history.close[start:stop]
        open_price = history.open_price[start:stop]
        high = history.high[start:stop]
        low = history.low[start:stop]
        volume = history.volume[start:stop]
        amount = history.amount[start:stop]
        close_return = np.zeros_like(close)
        close_return[1:] = close[1:] / close[:-1] - 1.0
        open_close_return = close / open_price - 1.0
        intrabar_range = (high - low) / close
        log_volume = np.log1p(volume)
        log_amount = np.log1p(amount)
        vwap = np.divide(amount, volume, out=open_price.copy(), where=volume > 0)
        vwap_deviation = close / vwap - 1.0
        features = np.vstack(
            [
                close_return,
                open_close_return,
                intrabar_range,
                log_volume,
                log_amount,
                vwap_deviation,
                time_sin,
                time_cos,
            ]
        ).astype("float32")
        sample_key = f"{sample.instrument_id}|{sample.signal_date}|{window_version}|{source_fingerprint}"
        sample_id = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:20]
        index_rows.append(
            {
                "sample_position": accepted,
                "sample_id": sample_id,
                "instrument_id": sample.instrument_id,
                "signal_date": sample.signal_date,
                "window_start_at": history.bar_end_at.iloc[start],
                "window_end_at": history.bar_end_at.iloc[stop - 1],
                "time_steps": expected_steps,
                "feature_count": len(FEATURE_NAMES),
                "window_version": window_version,
                "feature_version": "causal-basic-v1",
                "source_fingerprint": source_fingerprint,
            }
        )
        tensor[accepted] = features
        accepted += 1

    return tensor[:accepted], pd.DataFrame(index_rows), pd.DataFrame(rejection_rows)


def build_feature_windows(
    canonical_bars: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    lookback_days: int,
    source_fingerprint: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build valid windows while retaining the original two-value API."""

    features, index, _ = _build_feature_windows(
        canonical_bars,
        universe,
        lookback_days=lookback_days,
        source_fingerprint=source_fingerprint,
    )
    return features, index


def build_feature_windows_with_quality(
    canonical_bars: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    lookback_days: int,
    source_fingerprint: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build valid windows plus structured rejected-sample evidence."""

    return _build_feature_windows(
        canonical_bars,
        universe,
        lookback_days=lookback_days,
        source_fingerprint=source_fingerprint,
    )
