"""Point-in-time A-share universe behavior."""

from __future__ import annotations

from typing import Any, Mapping, cast

import pandas as pd

from .experiment import ContractError


REQUIRED_STATE_COLUMNS = {
    "instrument_id",
    "effective_at",
    "exchange",
    "security_type",
    "listed_date",
    "delisted_date",
    "is_st",
    "is_delisting",
    "is_suspended",
}


def build_pit_universe(
    canonical_bars: pd.DataFrame,
    states: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build an auditable universe snapshot using only state known by each signal date."""

    missing = sorted(REQUIRED_STATE_COLUMNS.difference(states.columns))
    if missing:
        raise ContractError(f"instrument states missing columns: {', '.join(missing)}")
    state = states.copy()
    try:
        state["effective_at"] = pd.to_datetime(
            state["effective_at"], utc=True, errors="raise"
        ).dt.tz_convert("Asia/Shanghai")
        state["listed_date"] = pd.to_datetime(
            state["listed_date"], errors="raise"
        ).dt.date
        state["delisted_date"] = pd.to_datetime(
            state["delisted_date"], errors="coerce"
        ).dt.date
    except Exception as exc:
        raise ContractError(f"invalid instrument state dates: {exc}") from exc
    if state.duplicated(["instrument_id", "effective_at"]).any():
        raise ContractError("instrument state effective keys must be unique")
    state = state.sort_values(["instrument_id", "effective_at"])

    universe_config = dict(config or {})
    allowed_exchanges = set(universe_config.get("allowed_exchanges", ["XSHG", "XSHE"]))
    security_type = universe_config.get("security_type", "A_SHARE")
    min_listing_days = int(universe_config.get("min_listing_days", 0))
    min_adv20 = universe_config.get("min_adv20")
    if min_adv20 is not None and "adv20" not in state:
        raise ContractError(
            "instrument states require adv20 when universe.min_adv20 is configured"
        )
    if min_adv20 is not None:
        min_adv20 = float(min_adv20)
        if min_adv20 < 0:
            raise ContractError("universe.min_adv20 cannot be negative")

    sessions = (
        canonical_bars.groupby(["instrument_id", "trade_date"], observed=True)
        .agg(
            bar_count=("bar_end_at", "size"),
            all_bars_ok=("quality_flag", lambda values: bool((values == "ok").all())),
        )
        .reset_index()
        .sort_values(["trade_date", "instrument_id"])
    )
    current_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    for instrument_id, instrument_sessions in sessions.groupby(
        "instrument_id", sort=False, observed=True
    ):
        instrument_state = state.loc[state["instrument_id"] == instrument_id]
        if instrument_state.empty:
            continue
        cutoffs = instrument_sessions[["trade_date"]].copy()
        cutoffs["cutoff"] = cutoffs["trade_date"].map(
            lambda value: pd.Timestamp(
                f"{value} 23:59:59", tz="Asia/Shanghai"
            )
        )
        known = pd.merge_asof(
            cutoffs.sort_values("cutoff"),
            instrument_state.sort_values("effective_at"),
            left_on="cutoff",
            right_on="effective_at",
            direction="backward",
        )
        known_records = cast(list[dict[str, Any]], known.to_dict("records"))
        for record in known_records:
            if pd.notna(record.get("effective_at")):
                current_by_key[(str(instrument_id), record["trade_date"])] = record
    rows: list[dict[str, Any]] = []
    for session in sessions.itertuples(index=False):
        session = cast(Any, session)
        signal_date = pd.Timestamp(session.trade_date).date()
        current = current_by_key.get((str(session.instrument_id), session.trade_date))
        reasons: list[str] = []
        risk_values: dict[str, Any] = {
            "industry": "unavailable",
            "market_cap": float("nan"),
            "adv20": float("nan"),
            "market_state": "unavailable",
        }
        if current is None:
            reasons.append("missing_state")
        else:
            risk_values = {
                "industry": str(current.get("industry", "unavailable")),
                "market_cap": pd.to_numeric(
                    current.get("market_cap", float("nan")), errors="coerce"
                ),
                "adv20": pd.to_numeric(
                    current.get("adv20", float("nan")), errors="coerce"
                ),
                "market_state": str(
                    current.get(
                        "market_state",
                        "suspended" if bool(current["is_suspended"]) else "normal",
                    )
                ),
            }
            if current["exchange"] not in allowed_exchanges:
                reasons.append("exchange")
            if current["security_type"] != security_type:
                reasons.append("security_type")
            if current["listed_date"] > signal_date:
                reasons.append("not_listed")
            if current["delisted_date"] is not pd.NaT and pd.notna(
                current["delisted_date"]
            ):
                if current["delisted_date"] <= signal_date:
                    reasons.append("delisted")
            if (signal_date - current["listed_date"]).days < min_listing_days:
                reasons.append("listing_history")
            if bool(current["is_st"]):
                reasons.append("st")
            if bool(current["is_delisting"]):
                reasons.append("delisting")
            if bool(current["is_suspended"]):
                reasons.append("suspended")
            if min_adv20 is not None and (
                pd.isna(current["adv20"]) or float(current["adv20"]) < min_adv20
            ):
                reasons.append("liquidity")
        if int(session.bar_count) != 48 or not bool(session.all_bars_ok):
            reasons.append("incomplete_session")
        rows.append(
            {
                "signal_date": session.trade_date,
                "instrument_id": session.instrument_id,
                "eligible": not reasons,
                "exclusion_reasons": ";".join(reasons),
                **risk_values,
            }
        )
    return pd.DataFrame(rows)
