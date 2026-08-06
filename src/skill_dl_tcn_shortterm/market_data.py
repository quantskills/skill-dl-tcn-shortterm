"""Canonical intraday market-data behavior."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .experiment import ContractError


REQUIRED_BAR_COLUMNS = {
    "instrument_id",
    "bar_end_at",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "quality_flag",
}

ACCEPTED_SOURCE_QUALITY_FLAGS = {"ok", "auction_no_trade_fill"}


def aggregate_five_minute_bars(
    raw: pd.DataFrame, manifest: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate canonical one-minute bars and aggregate closed five-minute bars."""

    missing = sorted(REQUIRED_BAR_COLUMNS.difference(raw.columns))
    if missing:
        raise ContractError(
            f"raw one-minute bars missing columns: {', '.join(missing)}"
        )
    for field in (
        "timezone",
        "price_unit",
        "volume_unit",
        "amount_unit",
        "source_version",
    ):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ContractError(
                f"manifest.{field} must be a non-empty string for raw_1m data"
            )
    if manifest["timezone"] != "Asia/Shanghai":
        raise ContractError("raw_1m manifest timezone must be Asia/Shanghai")
    if raw.empty:
        raise ContractError("raw one-minute bars must contain at least one row")

    bars = raw.copy()
    try:
        bars["bar_end_at"] = pd.to_datetime(bars["bar_end_at"], errors="raise")
    except Exception as exc:
        raise ContractError(f"invalid bar_end_at values: {exc}") from exc
    if bars["bar_end_at"].dt.tz is None:
        raise ContractError("bar_end_at must be timezone-aware")
    bars["bar_end_at"] = bars["bar_end_at"].dt.tz_convert("Asia/Shanghai")

    if bars.duplicated(["instrument_id", "bar_end_at"]).any():
        raise ContractError("raw one-minute bar primary keys must be unique")
    for _, group in bars.groupby("instrument_id", sort=False):
        if not group["bar_end_at"].is_monotonic_increasing:
            raise ContractError("raw one-minute bars must be ordered per instrument")

    price_columns = ["open", "high", "low", "close"]
    numeric_columns = [*price_columns, "volume", "amount"]
    for column in numeric_columns:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    if bars[numeric_columns].isna().any(axis=None):
        raise ContractError(
            "raw one-minute numeric fields cannot contain null or non-numeric values"
        )
    if (bars[price_columns] <= 0).any(axis=None):
        raise ContractError("raw one-minute prices must be positive")
    if (bars[["volume", "amount"]] < 0).any(axis=None):
        raise ContractError("raw one-minute volume and amount cannot be negative")

    local = bars["bar_end_at"]
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    in_morning = (minute_of_day > 9 * 60 + 30) & (minute_of_day <= 11 * 60 + 30)
    in_afternoon = (minute_of_day > 13 * 60) & (minute_of_day <= 15 * 60)
    if not (in_morning | in_afternoon).all():
        raise ContractError(
            "raw one-minute bars must end inside an A-share continuous session"
        )

    bars["trade_date"] = local.dt.strftime("%Y-%m-%d")
    bars["five_minute_end"] = local.dt.ceil("5min")
    keys = ["instrument_id", "trade_date", "five_minute_end"]

    canonical = (
        bars.groupby(keys, sort=True, observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            minute_count=("bar_end_at", "size"),
            source_quality_reasons=(
                "quality_flag",
                lambda values: "|".join(
                    sorted(
                        {
                            str(value)
                            for value in values
                            if str(value) not in ACCEPTED_SOURCE_QUALITY_FLAGS
                        }
                    )
                ),
            ),
        )
        .reset_index()
        .rename(columns={"five_minute_end": "bar_end_at"})
    )
    canonical["quality_flag"] = "ok"
    canonical.loc[canonical["minute_count"] != 5, "quality_flag"] = "incomplete"
    has_source_issue = canonical["source_quality_reasons"] != ""
    canonical.loc[has_source_issue, "quality_flag"] = canonical.loc[
        has_source_issue, "source_quality_reasons"
    ].map(lambda value: f"source_{value}")
    canonical["source_version"] = manifest["source_version"]

    session_summary = canonical.groupby(
        ["instrument_id", "trade_date"], observed=True
    ).agg(
        bar_count=("bar_end_at", "size"),
        all_ok=("quality_flag", lambda values: bool((values == "ok").all())),
    )
    complete_sessions = (
        (session_summary["bar_count"] == 48) & session_summary["all_ok"]
    ).sum()
    quality = {
        "raw_bar_count": int(len(bars)),
        "canonical_bar_count": int(len(canonical)),
        "incomplete_bar_count": int((canonical["quality_flag"] != "ok").sum()),
        "session_count": int(len(session_summary)),
        "complete_session_count": int(complete_sessions),
        "timezone": manifest["timezone"],
        "source_version": manifest["source_version"],
    }
    return canonical, quality
