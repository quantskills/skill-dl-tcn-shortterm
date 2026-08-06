"""Causal, scale-relative feature windows for cross-sectional TCN research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .experiment import ContractError


BASE_FEATURE_NAMES = (
    "close_return",
    "open_close_return",
    "intrabar_range",
    "log_volume",
    "log_amount",
    "vwap_deviation",
    "time_sin",
    "time_cos",
)

RELATIVE_FEATURE_NAMES = (
    "close_return",
    "open_close_return",
    "intrabar_range",
    "log_amount_to_adv20",
    "log_amount_to_market_cap",
    "vwap_deviation",
    "time_sin",
    "time_cos",
    "cross_section_return_1d_rank",
    "cross_section_realized_vol_10d_rank",
    "cross_section_amount_to_adv20_rank",
    "cross_section_amount_to_market_cap_rank",
    "cross_section_market_cap_rank",
)

FEATURE_VERSION = "causal-relative-cross-sectional-v37"
APPENDED_SEQUENCE_FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    "log_amount_to_adv20",
    "log_amount_to_market_cap",
)
APPENDED_SEQUENCE_FEATURE_VERSION = "causal-base-plus-relative-sequence-v38"
TOP50_APPENDED_SEQUENCE_FEATURE_VERSION = (
    "causal-base-plus-relative-sequence-v39-top50"
)


@dataclass(frozen=True)
class RelativeFeatureResult:
    """Materialized candidate features plus row-level and aggregate evidence."""

    features: np.ndarray
    window_index: pd.DataFrame
    audit: pd.DataFrame
    quality: dict[str, Any]


@dataclass(frozen=True)
class TopNReadiness:
    """Fail-closed PIT-state coverage decision for a wider universe."""

    status: str
    ready: bool
    evidence: dict[str, Any]


def materialize_appended_relative_sequence_features(
    base_features: np.ndarray,
    relative_features: np.ndarray,
    window_index: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 512,
) -> RelativeFeatureResult:
    """Append only the two audited relative sequences to the unchanged base8 tensor."""

    if (
        base_features.ndim != 3
        or base_features.shape[1] != len(BASE_FEATURE_NAMES)
        or relative_features.ndim != 3
        or relative_features.shape[1] != len(RELATIVE_FEATURE_NAMES)
    ):
        raise ContractError("v38 requires base8 and audited v37 relative13 tensors")
    if base_features.shape[0] != relative_features.shape[0] or base_features.shape[
        2
    ] != relative_features.shape[2]:
        raise ContractError("v38 base and relative tensor coverage drifted")
    if len(window_index) != len(base_features):
        raise ContractError("v38 tensor and window index sample counts differ")
    if chunk_size <= 0:
        raise ContractError("v38 feature chunk size must be positive")
    index = window_index.sort_values("sample_position", kind="mergesort").reset_index(
        drop=True
    )
    positions = index["sample_position"].to_numpy(dtype="int64")
    if not np.array_equal(positions, np.arange(len(index), dtype="int64")):
        raise ContractError("v38 window positions must be contiguous and array-aligned")

    shape = (
        len(index),
        len(APPENDED_SEQUENCE_FEATURE_NAMES),
        int(base_features.shape[2]),
    )
    if output_path is None:
        candidate: np.ndarray = np.empty(shape, dtype="float32")
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=shape
        )
    for start in range(0, len(index), chunk_size):
        stop = min(len(index), start + chunk_size)
        base = np.asarray(base_features[start:stop], dtype="float32")
        relative = np.asarray(relative_features[start:stop], dtype="float32")
        if not np.isfinite(base).all() or not np.isfinite(relative[:, 3:5]).all():
            raise ContractError("v38 source feature channels contain non-finite values")
        candidate[start:stop, : len(BASE_FEATURE_NAMES), :] = base
        candidate[start:stop, len(BASE_FEATURE_NAMES) :, :] = relative[:, 3:5, :]
    if isinstance(candidate, np.memmap):
        candidate.flush()
    candidate_index = index.copy()
    candidate_index["feature_count"] = len(APPENDED_SEQUENCE_FEATURE_NAMES)
    candidate_index["feature_version"] = APPENDED_SEQUENCE_FEATURE_VERSION
    quality: dict[str, Any] = {
        "feature_version": APPENDED_SEQUENCE_FEATURE_VERSION,
        "feature_names": list(APPENDED_SEQUENCE_FEATURE_NAMES),
        "sample_count": len(index),
        "signal_date_count": int(index["signal_date"].astype(str).nunique()),
        "base_shape": list(base_features.shape),
        "relative_source_shape": list(relative_features.shape),
        "candidate_shape": list(shape),
        "base_channels_preserved": True,
        "static_rank_channels_included": False,
        "sealed_test_accessed": False,
    }
    audit = candidate_index[
        ["sample_position", "sample_id", "instrument_id", "signal_date"]
    ].copy()
    audit["base_channels_preserved"] = True
    audit["relative_sequence_channel_count"] = 2
    return RelativeFeatureResult(candidate, candidate_index, audit, quality)


def _scaled_cross_section_rank(values: pd.Series) -> pd.Series:
    count = len(values)
    if count < 2:
        raise ContractError("cross-sectional rank requires at least two members")
    ranks = values.rank(method="average")
    return 2.0 * (ranks - 1.0) / float(count - 1) - 1.0


def _validate_and_align(
    features: np.ndarray,
    window_index: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    bars_per_day: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features.ndim != 3 or features.shape[1] != len(BASE_FEATURE_NAMES):
        raise ContractError("relative features require an [N, 8, T] base tensor")
    if bars_per_day <= 0 or features.shape[2] % bars_per_day:
        raise ContractError("time steps must contain complete trading days")
    required_index = {
        "sample_position",
        "sample_id",
        "instrument_id",
        "signal_date",
        "window_end_at",
    }
    if missing := sorted(required_index.difference(window_index.columns)):
        raise ContractError("window index missing columns: " + ", ".join(missing))
    required_state = {
        "signal_date",
        "instrument_id",
        "eligible",
        "market_cap",
        "adv20",
    }
    if missing := sorted(required_state.difference(universe.columns)):
        raise ContractError("PIT universe missing columns: " + ", ".join(missing))
    if len(features) != len(window_index):
        raise ContractError("feature tensor and window index counts differ")

    index = window_index.copy()
    index["sample_position"] = index["sample_position"].astype(int)
    index = index.sort_values("sample_position", kind="mergesort").reset_index(drop=True)
    expected_positions = np.arange(len(index), dtype="int64")
    if not np.array_equal(index["sample_position"].to_numpy(), expected_positions):
        raise ContractError("window sample positions must be contiguous and array-aligned")
    if index.duplicated(["signal_date", "instrument_id"]).any():
        raise ContractError("window index contains duplicate signal-date instruments")
    index["signal_date"] = index["signal_date"].astype(str)
    end_dates = pd.to_datetime(index["window_end_at"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    if end_dates.isna().any() or not end_dates.equals(index["signal_date"]):
        raise ContractError("each causal window must end on its signal date")

    state = universe[list(required_state)].copy()
    state["signal_date"] = state["signal_date"].astype(str)
    state["instrument_id"] = state["instrument_id"].astype(str)
    if state.duplicated(["signal_date", "instrument_id"]).any():
        raise ContractError("PIT universe contains duplicate signal-date instruments")
    aligned = index[["sample_position", "signal_date", "instrument_id"]].merge(
        state,
        on=["signal_date", "instrument_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned["eligible"].isna().any() or not aligned["eligible"].astype(bool).all():
        raise ContractError("every feature sample requires eligible signal-date PIT state")
    market_cap = pd.to_numeric(aligned["market_cap"], errors="coerce")
    if not np.isfinite(market_cap).all() or bool((market_cap <= 0).any()):
        raise ContractError("market cap must be finite and positive for every sample")
    aligned["market_cap"] = market_cap.astype("float64")
    aligned["adv20"] = pd.to_numeric(aligned["adv20"], errors="coerce")
    return index, aligned


def _build_audit(
    features: np.ndarray,
    index: pd.DataFrame,
    aligned: pd.DataFrame,
    *,
    bars_per_day: int,
    chunk_size: int,
    min_cross_section: int,
) -> pd.DataFrame:
    if chunk_size <= 0:
        raise ContractError("relative feature chunk size must be positive")
    if min_cross_section < 2:
        raise ContractError("minimum cross section must be at least two")
    sample_count = len(index)
    return_1d = np.empty(sample_count, dtype="float64")
    realized_vol = np.empty(sample_count, dtype="float64")
    amount_1d = np.empty(sample_count, dtype="float64")
    fallback_adv = np.empty(sample_count, dtype="float64")

    for start in range(0, sample_count, chunk_size):
        stop = min(sample_count, start + chunk_size)
        block = np.asarray(features[start:stop], dtype="float64")
        close_return = block[:, 0, :]
        if bool((close_return <= -1.0).any()):
            raise ContractError("close returns must be greater than -1")
        amount = np.expm1(block[:, 4, :])
        if not np.isfinite(amount).all() or bool((amount < 0).any()):
            raise ContractError("base log amount cannot be converted to finite amount")
        return_1d[start:stop] = np.expm1(
            np.log1p(close_return[:, -bars_per_day:]).sum(axis=1)
        )
        realized_vol[start:stop] = close_return.std(axis=1, ddof=0)
        amount_1d[start:stop] = amount[:, -bars_per_day:].sum(axis=1)
        prior = amount[:, :-bars_per_day]
        if prior.shape[1] < bars_per_day:
            raise ContractError("ADV fallback requires at least one prior complete day")
        prior_daily = prior.reshape(len(block), -1, bars_per_day).sum(axis=2)
        fallback_adv[start:stop] = prior_daily.mean(axis=1)

    state_adv = aligned["adv20"].to_numpy(dtype="float64")
    uses_fallback = ~np.isfinite(state_adv) | (state_adv <= 0)
    effective_adv = state_adv.copy()
    effective_adv[uses_fallback] = fallback_adv[uses_fallback]
    if not np.isfinite(effective_adv).all() or bool((effective_adv <= 0).any()):
        raise ContractError("ADV20 state and causal fallback are both unavailable")
    market_cap = aligned["market_cap"].to_numpy(dtype="float64")
    audit = index[["sample_position", "sample_id", "signal_date", "instrument_id"]].copy()
    audit["market_cap"] = market_cap
    audit["state_adv20"] = state_adv
    audit["fallback_adv"] = fallback_adv
    audit["effective_adv20"] = effective_adv
    audit["adv_source"] = np.where(uses_fallback, "prior_window_days", "pit_adv20")
    audit["return_1d"] = return_1d
    audit["realized_vol_10d"] = realized_vol
    audit["amount_to_adv20"] = amount_1d / effective_adv
    audit["amount_to_market_cap"] = amount_1d / market_cap
    audit["cross_section_count"] = audit.groupby(
        "signal_date", observed=True
    )["sample_position"].transform("size")
    if bool((audit["cross_section_count"] < min_cross_section).any()):
        raise ContractError("a signal date is below the minimum cross-sectional width")

    raw_rank_columns = {
        "return_1d": "cross_section_return_1d_rank",
        "realized_vol_10d": "cross_section_realized_vol_10d_rank",
        "amount_to_adv20": "cross_section_amount_to_adv20_rank",
        "amount_to_market_cap": "cross_section_amount_to_market_cap_rank",
        "market_cap": "cross_section_market_cap_rank",
    }
    for source, target in raw_rank_columns.items():
        audit[target] = audit.groupby("signal_date", observed=True)[source].transform(
            _scaled_cross_section_rank
        )
    numeric = audit[
        ["effective_adv20", *raw_rank_columns.values()]
    ].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all():
        raise ContractError("relative feature audit contains non-finite values")
    return audit


def materialize_causal_relative_features(
    features: np.ndarray,
    window_index: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    bars_per_day: int = 48,
    chunk_size: int = 512,
    min_cross_section: int = 10,
) -> RelativeFeatureResult:
    """Build the fixed v37 representation without loading the full tensor into RAM."""

    index, aligned = _validate_and_align(
        features, window_index, universe, bars_per_day=bars_per_day
    )
    audit = _build_audit(
        features,
        index,
        aligned,
        bars_per_day=bars_per_day,
        chunk_size=chunk_size,
        min_cross_section=min_cross_section,
    )
    shape = (len(index), len(RELATIVE_FEATURE_NAMES), int(features.shape[2]))
    if output_path is None:
        candidate: np.ndarray = np.empty(shape, dtype="float32")
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=shape
        )

    rank_columns = list(RELATIVE_FEATURE_NAMES[8:])
    ranks = audit[rank_columns].to_numpy(dtype="float32")
    adv = audit["effective_adv20"].to_numpy(dtype="float64")
    market_cap = audit["market_cap"].to_numpy(dtype="float64")
    for start in range(0, len(index), chunk_size):
        stop = min(len(index), start + chunk_size)
        block = np.asarray(features[start:stop], dtype="float32")
        amount = np.expm1(block[:, 4, :].astype("float64"))
        candidate[start:stop, 0:3, :] = block[:, 0:3, :]
        candidate[start:stop, 3, :] = np.log1p(
            amount / adv[start:stop, np.newaxis]
        ).astype("float32")
        candidate[start:stop, 4, :] = np.log1p(
            amount / market_cap[start:stop, np.newaxis]
        ).astype("float32")
        candidate[start:stop, 5:8, :] = block[:, 5:8, :]
        candidate[start:stop, 8:, :] = ranks[start:stop, :, np.newaxis]
    if not np.isfinite(candidate).all():
        raise ContractError("materialized relative features contain non-finite values")
    if isinstance(candidate, np.memmap):
        candidate.flush()

    candidate_index = index.copy()
    candidate_index["feature_count"] = len(RELATIVE_FEATURE_NAMES)
    candidate_index["feature_version"] = FEATURE_VERSION
    quality: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "feature_names": list(RELATIVE_FEATURE_NAMES),
        "sample_count": len(index),
        "signal_date_count": int(index["signal_date"].nunique()),
        "base_shape": list(features.shape),
        "candidate_shape": list(shape),
        "bars_per_day": bars_per_day,
        "adv_fallback_count": int(audit["adv_source"].eq("prior_window_days").sum()),
        "min_cross_section_count": int(audit["cross_section_count"].min()),
        "max_cross_section_count": int(audit["cross_section_count"].max()),
        "sealed_test_accessed": False,
    }
    return RelativeFeatureResult(candidate, candidate_index, audit, quality)


def materialize_top50_appended_relative_sequence_features(
    features: np.ndarray,
    window_index: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    bars_per_day: int = 48,
    chunk_size: int = 512,
    min_cross_section: int = 31,
) -> RelativeFeatureResult:
    """Build v39 base8+relative2 directly from audited top50 PIT state."""

    index, aligned = _validate_and_align(
        features, window_index, universe, bars_per_day=bars_per_day
    )
    audit = _build_audit(
        features,
        index,
        aligned,
        bars_per_day=bars_per_day,
        chunk_size=chunk_size,
        min_cross_section=min_cross_section,
    )
    shape = (
        len(index),
        len(APPENDED_SEQUENCE_FEATURE_NAMES),
        int(features.shape[2]),
    )
    if output_path is None:
        candidate: np.ndarray = np.empty(shape, dtype="float32")
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=shape
        )
    adv = audit["effective_adv20"].to_numpy(dtype="float64")
    market_cap = audit["market_cap"].to_numpy(dtype="float64")
    for start in range(0, len(index), chunk_size):
        stop = min(len(index), start + chunk_size)
        block = np.asarray(features[start:stop], dtype="float32")
        if not np.isfinite(block).all():
            raise ContractError("v39 base feature channels contain non-finite values")
        amount = np.expm1(block[:, 4, :].astype("float64"))
        candidate[start:stop, : len(BASE_FEATURE_NAMES), :] = block
        candidate[start:stop, len(BASE_FEATURE_NAMES), :] = np.log1p(
            amount / adv[start:stop, np.newaxis]
        ).astype("float32")
        candidate[start:stop, len(BASE_FEATURE_NAMES) + 1, :] = np.log1p(
            amount / market_cap[start:stop, np.newaxis]
        ).astype("float32")
    if not np.isfinite(candidate).all():
        raise ContractError("v39 appended relative features contain non-finite values")
    if isinstance(candidate, np.memmap):
        candidate.flush()

    candidate_index = index.copy()
    candidate_index["feature_count"] = len(APPENDED_SEQUENCE_FEATURE_NAMES)
    candidate_index["feature_version"] = TOP50_APPENDED_SEQUENCE_FEATURE_VERSION
    quality: dict[str, Any] = {
        "feature_version": TOP50_APPENDED_SEQUENCE_FEATURE_VERSION,
        "feature_names": list(APPENDED_SEQUENCE_FEATURE_NAMES),
        "sample_count": len(index),
        "signal_date_count": int(index["signal_date"].nunique()),
        "base_shape": list(features.shape),
        "candidate_shape": list(shape),
        "bars_per_day": bars_per_day,
        "adv_fallback_count": int(audit["adv_source"].eq("prior_window_days").sum()),
        "min_cross_section_count": int(audit["cross_section_count"].min()),
        "max_cross_section_count": int(audit["cross_section_count"].max()),
        "base_channels_preserved": True,
        "static_rank_channels_included": False,
        "sealed_test_accessed": False,
    }
    return RelativeFeatureResult(candidate, candidate_index, audit, quality)


def audit_top_n_state_readiness(
    pit_universe: pd.DataFrame,
    states: pd.DataFrame,
    *,
    top_n: int = 50,
) -> TopNReadiness:
    """Require complete signal-date state before widening a PIT weight universe."""

    if top_n <= 0:
        raise ContractError("top-n readiness requires a positive width")
    required_universe = {"trade_date", "instrument_id", "weight"}
    if missing := sorted(required_universe.difference(pit_universe.columns)):
        raise ContractError("PIT weight universe missing columns: " + ", ".join(missing))
    required_state = {"signal_date", "instrument_id", "market_cap"}
    if missing := sorted(required_state.difference(states.columns)):
        raise ContractError("PIT state table missing columns: " + ", ".join(missing))

    universe = pit_universe[list(required_universe)].copy()
    universe["trade_date"] = universe["trade_date"].astype(str)
    universe["instrument_id"] = universe["instrument_id"].astype(str)
    universe["weight"] = pd.to_numeric(universe["weight"], errors="coerce")
    if universe.duplicated(["trade_date", "instrument_id"]).any():
        raise ContractError("PIT weight universe contains duplicate keys")
    if not np.isfinite(universe["weight"]).all():
        raise ContractError("PIT weights must be finite")
    counts = universe.groupby("trade_date", observed=True).size()
    insufficient_dates = counts.loc[counts < top_n]
    top = (
        universe.sort_values(
            ["trade_date", "weight", "instrument_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("trade_date", observed=True, sort=True)
        .head(top_n)
        .reset_index(drop=True)
    )
    state = states[list(required_state)].copy()
    state["signal_date"] = state["signal_date"].astype(str)
    state["instrument_id"] = state["instrument_id"].astype(str)
    if state.duplicated(["signal_date", "instrument_id"]).any():
        raise ContractError("PIT state table contains duplicate keys")
    joined = top.merge(
        state,
        left_on=["trade_date", "instrument_id"],
        right_on=["signal_date", "instrument_id"],
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    missing_state = joined["_merge"].ne("both")
    market_cap = pd.to_numeric(joined["market_cap"], errors="coerce")
    missing_market_cap = ~np.isfinite(market_cap) | (market_cap <= 0)
    complete_width = bool(counts.ge(top_n).all())
    ready = bool(
        complete_width and not missing_state.any() and not missing_market_cap.any()
    )
    evidence: dict[str, Any] = {
        "top_n": top_n,
        "date_count": int(counts.size),
        "date_from": str(counts.index.min()) if len(counts) else None,
        "date_to": str(counts.index.max()) if len(counts) else None,
        "required_key_count": int(len(top)),
        "insufficient_universe_date_count": int(len(insufficient_dates)),
        "missing_state_key_count": int(missing_state.sum()),
        "missing_market_cap_key_count": int(missing_market_cap.sum()),
        "sealed_test_accessed": False,
    }
    status = "ready_top50" if ready and top_n == 50 else (
        "ready_top_n" if ready else "blocked_missing_pit_state"
    )
    return TopNReadiness(status=status, ready=ready, evidence=evidence)
