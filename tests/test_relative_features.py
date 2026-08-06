from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.relative_features import (
    APPENDED_SEQUENCE_FEATURE_NAMES,
    APPENDED_SEQUENCE_FEATURE_VERSION,
    FEATURE_VERSION,
    RELATIVE_FEATURE_NAMES,
    audit_top_n_state_readiness,
    materialize_appended_relative_sequence_features,
    materialize_causal_relative_features,
    materialize_top50_appended_relative_sequence_features,
)


def _fixture() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    dates = ["2025-01-02", "2025-01-03"]
    instruments = ["A", "B", "C"]
    rows = []
    features = np.zeros((6, 8, 8), dtype="float32")
    for position, (date, instrument) in enumerate(
        (date, instrument) for date in dates for instrument in instruments
    ):
        features[position, 0] = 0.001 * (position + 1)
        features[position, 1] = 0.002 * (position + 1)
        features[position, 2] = 0.003 * (position + 1)
        features[position, 3] = np.log1p(10.0 * (position + 1))
        features[position, 4] = np.log1p(100.0 * (position + 1))
        features[position, 5] = 0.0001 * position
        features[position, 6] = np.sin(np.arange(8) * np.pi / 2)
        features[position, 7] = np.cos(np.arange(8) * np.pi / 2)
        rows.append(
            {
                "sample_position": position,
                "sample_id": f"sample-{position}",
                "instrument_id": instrument,
                "signal_date": date,
                "window_end_at": f"{date} 15:00:00",
                "feature_count": 8,
                "feature_version": "causal-basic-v1",
            }
        )
    states = pd.DataFrame(
        [
            {
                "signal_date": date,
                "instrument_id": instrument,
                "eligible": True,
                "market_cap": 1_000_000.0 * (index + 2) ** 2,
                "adv20": 10_000.0 * (index + 2) ** 2,
            }
            for date in dates
            for index, instrument in enumerate(instruments)
        ]
    )
    return features, pd.DataFrame(rows), states


def test_relative_features_are_finite_ranked_and_scale_invariant() -> None:
    features, index, states = _fixture()
    result = materialize_causal_relative_features(
        features,
        index,
        states,
        bars_per_day=4,
        min_cross_section=3,
        chunk_size=2,
    )
    scaled = features.copy()
    scaled[:, 3] = np.log1p(np.expm1(scaled[:, 3]) * 10.0)
    scaled[:, 4] = np.log1p(np.expm1(scaled[:, 4]) * 10.0)
    scaled_states = states.copy()
    scaled_states["adv20"] *= 10.0
    scaled_states["market_cap"] *= 10.0
    scaled_result = materialize_causal_relative_features(
        scaled,
        index,
        scaled_states,
        bars_per_day=4,
        min_cross_section=3,
        chunk_size=3,
    )

    assert result.features.shape == (6, len(RELATIVE_FEATURE_NAMES), 8)
    assert np.isfinite(result.features).all()
    assert result.window_index["feature_version"].eq(FEATURE_VERSION).all()
    np.testing.assert_allclose(result.features, scaled_result.features, atol=1e-6)
    for date_group in result.audit.groupby("signal_date"):
        _, group = date_group
        assert set(group["cross_section_market_cap_rank"]) == {-1.0, 0.0, 1.0}


def test_later_date_mutation_cannot_change_earlier_relative_features() -> None:
    features, index, states = _fixture()
    baseline = materialize_causal_relative_features(
        features, index, states, bars_per_day=4, min_cross_section=3
    )
    mutated = features.copy()
    mutated[3:, 4] = np.log1p(np.expm1(mutated[3:, 4]) * 100.0)
    mutated_states = states.copy()
    mutated_states.loc[mutated_states["signal_date"].eq("2025-01-03"), "adv20"] *= 50
    candidate = materialize_causal_relative_features(
        mutated, index, mutated_states, bars_per_day=4, min_cross_section=3
    )

    np.testing.assert_array_equal(baseline.features[:3], candidate.features[:3])


def test_missing_adv_uses_only_prior_window_days() -> None:
    features, index, states = _fixture()
    states["adv20"] = np.nan
    result = materialize_causal_relative_features(
        features, index, states, bars_per_day=4, min_cross_section=3
    )

    assert result.quality["adv_fallback_count"] == 6
    assert result.audit["adv_source"].eq("prior_window_days").all()
    expected = np.expm1(features[0, 4, :4]).sum()
    assert result.audit.iloc[0]["effective_adv20"] == pytest.approx(expected)


def test_relative_features_fail_closed_on_missing_market_cap() -> None:
    features, index, states = _fixture()
    states.loc[0, "market_cap"] = np.nan
    with pytest.raises(ContractError, match="market cap"):
        materialize_causal_relative_features(
            features, index, states, bars_per_day=4, min_cross_section=3
        )


def test_top50_readiness_requires_complete_signal_date_state() -> None:
    universe = pd.DataFrame(
        [
            {
                "trade_date": date,
                "instrument_id": f"S{instrument:03d}",
                "weight": float(100 - instrument),
            }
            for date in ["2025-01-02", "2025-01-03"]
            for instrument in range(100)
        ]
    )
    top20_states = pd.DataFrame(
        [
            {
                "signal_date": date,
                "instrument_id": f"S{instrument:03d}",
                "market_cap": 1_000_000.0,
            }
            for date in ["2025-01-02", "2025-01-03"]
            for instrument in range(20)
        ]
    )
    blocked = audit_top_n_state_readiness(universe, top20_states, top_n=50)
    assert blocked.status == "blocked_missing_pit_state"
    assert blocked.ready is False
    assert blocked.evidence["missing_state_key_count"] == 60

    top50_states = pd.DataFrame(
        [
            {
                "signal_date": date,
                "instrument_id": f"S{instrument:03d}",
                "market_cap": 1_000_000.0,
            }
            for date in ["2025-01-02", "2025-01-03"]
            for instrument in range(50)
        ]
    )
    ready = audit_top_n_state_readiness(universe, top50_states, top_n=50)
    assert ready.status == "ready_top50"
    assert ready.ready is True


def test_v38_appends_only_relative_sequences_and_preserves_base_exactly() -> None:
    features, index, states = _fixture()
    relative = materialize_causal_relative_features(
        features, index, states, bars_per_day=4, min_cross_section=3
    )
    appended = materialize_appended_relative_sequence_features(
        features, relative.features, index, chunk_size=2
    )

    assert appended.features.shape == (6, len(APPENDED_SEQUENCE_FEATURE_NAMES), 8)
    np.testing.assert_array_equal(appended.features[:, :8], features)
    np.testing.assert_array_equal(appended.features[:, 8:], relative.features[:, 3:5])
    assert appended.window_index["feature_version"].eq(
        APPENDED_SEQUENCE_FEATURE_VERSION
    ).all()
    assert appended.quality["static_rank_channels_included"] is False


def test_v39_builds_same_appended_sequences_directly_from_pit_state() -> None:
    features, index, states = _fixture()
    relative = materialize_causal_relative_features(
        features, index, states, bars_per_day=4, min_cross_section=3
    )
    expected = materialize_appended_relative_sequence_features(
        features, relative.features, index, chunk_size=2
    )
    observed = materialize_top50_appended_relative_sequence_features(
        features,
        index,
        states,
        bars_per_day=4,
        min_cross_section=3,
        chunk_size=2,
    )
    np.testing.assert_array_equal(observed.features, expected.features)
    assert observed.quality["base_channels_preserved"] is True
    assert observed.quality["static_rank_channels_included"] is False
