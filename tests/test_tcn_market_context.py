from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm.experiment import ContractError
from skill_dl_tcn_shortterm.market_context import (
    ContextualLazyWindowDataset,
    PITMarketContext,
    build_pit_market_context,
    fit_market_context_standardizer,
)
from skill_dl_tcn_shortterm.real_validation import (
    evaluate_pit_market_conditioning_seed7,
    finalize_pit_market_conditioning_seed7,
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    run_tcn_validation_sweep,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import (
    MarketConditionedTemporalContextTCN,
    TemporalContextTCN,
)


def _fixture() -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(7, 8, 8)).astype("float32")
    dates = ["2025-01-02"] * 3 + ["2025-01-03"] * 3 + ["2025-01-06"]
    index = pd.DataFrame(
        {
            "sample_position": np.arange(7, dtype="int64"),
            "sample_id": [f"sample-{value}" for value in range(7)],
            "instrument_id": [f"stock-{value}" for value in range(7)],
            "signal_date": dates,
            "window_end_at": pd.to_datetime(
                [f"{value} 15:00:00+08:00" for value in dates]
            ),
        }
    )
    return features, index


def test_pit_market_context_is_shared_by_date_and_future_isolated() -> None:
    features, index = _fixture()
    allowed = np.arange(6, dtype="int64")
    context = build_pit_market_context(
        features,
        index,
        allowed_positions=allowed,
        feature_indices=(0, 1),
        bars_per_day=4,
    )

    assert context.values.shape == (7, 8)
    np.testing.assert_array_equal(context.available_positions, allowed)
    np.testing.assert_allclose(context.values[0], context.values[1], rtol=0, atol=0)
    np.testing.assert_allclose(context.values[1], context.values[2], rtol=0, atol=0)
    assert not np.array_equal(context.values[0], context.values[3])
    assert np.isnan(context.values[6]).all()
    assert context.field_names == (
        "center_last_day:0",
        "center_last_day:1",
        "center_full_window:0",
        "center_full_window:1",
        "dispersion_last_day:0",
        "dispersion_last_day:1",
        "dispersion_full_window:0",
        "dispersion_full_window:1",
    )

    changed = features.copy()
    changed[3:6] += 10_000.0
    rebuilt = build_pit_market_context(
        changed,
        index,
        allowed_positions=allowed,
        feature_indices=(0, 1),
        bars_per_day=4,
    )
    np.testing.assert_allclose(rebuilt.values[:3], context.values[:3], rtol=0, atol=0)
    assert rebuilt.identity != context.identity


def test_pit_market_context_fails_closed_on_non_pit_or_undercovered_dates() -> None:
    features, index = _fixture()
    duplicated = pd.concat([index, index.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate sample positions"):
        build_pit_market_context(
            features,
            duplicated,
            allowed_positions=np.arange(6),
            feature_indices=(0, 1),
            bars_per_day=4,
        )

    future_end = index.copy()
    future_end.loc[0, "window_end_at"] = pd.Timestamp("2025-01-03 09:35:00+08:00")
    with pytest.raises(ContractError, match="must end on its signal date"):
        build_pit_market_context(
            features,
            future_end,
            allowed_positions=np.arange(6),
            feature_indices=(0, 1),
            bars_per_day=4,
        )

    with pytest.raises(ContractError, match="at least two instruments"):
        build_pit_market_context(
            features,
            index,
            allowed_positions=np.arange(7),
            feature_indices=(0, 1),
            bars_per_day=4,
        )


def test_market_context_scaler_fits_unique_training_dates_only() -> None:
    values = np.asarray(
        [[0.0], [0.0], [2.0], [2.0], [2.0], [2.0], [100.0]],
        dtype="float32",
    )
    context = PITMarketContext(
        values=values,
        available_positions=np.arange(7, dtype="int64"),
        field_names=("state",),
        feature_indices=(0,),
        bars_per_day=4,
        identity="fixture",
        date_sample_counts={"2025-01-02": 2, "2025-01-03": 4, "2025-01-06": 1},
    )
    index = pd.DataFrame(
        {
            "sample_position": np.arange(7),
            "signal_date": ["2025-01-02"] * 2 + ["2025-01-03"] * 4 + ["2025-01-06"],
        }
    )
    scaler = fit_market_context_standardizer(
        context,
        index,
        train_positions=np.arange(6),
    )

    np.testing.assert_allclose(scaler.mean, [1.0])
    np.testing.assert_allclose(scaler.std, [1.0])
    assert scaler.fit_date_count == 2
    assert scaler.transform(values[[6]])[0, 0] == pytest.approx(99.0)


def test_market_conditioned_tcn_is_zero_init_equivalent_and_bounded() -> None:
    common: dict[str, Any] = {
        "feature_count": 8,
        "channels": 16,
        "kernel_size": 3,
        "dilations": (1, 2, 4, 8),
        "input_steps": 24,
        "bars_per_day": 8,
        "dropout": 0.0,
        "padding_mode": "chomp",
    }
    torch.manual_seed(19)
    control = TemporalContextTCN(**common)
    torch.manual_seed(19)
    candidate = MarketConditionedTemporalContextTCN(
        **common,
        market_context_dim=24,
        market_context_hidden=4,
        market_gate_scale=0.25,
    )
    inputs = torch.randn(5, 8, 24)
    context = torch.randn(5, 24)

    torch.testing.assert_close(
        candidate(inputs, context), control(inputs), rtol=0, atol=0
    )
    assert (
        sum(value.numel() for value in candidate.parameters())
        - sum(value.numel() for value in control.parameters())
        == 260
    )
    with torch.no_grad():
        candidate.market_gate_output.weight.fill_(0.5)
        candidate.market_gate_output.bias.fill_(0.5)
    gate = candidate.market_gate(context)
    assert float(gate.detach().min()) >= 0.75
    assert float(gate.detach().max()) <= 1.25
    assert not torch.equal(candidate(inputs, context), control(inputs))
    with pytest.raises(ContractError, match="market context shape"):
        candidate(inputs, torch.randn(5, 23))


def test_contextual_dataset_rejects_unavailable_context_positions() -> None:
    features, index = _fixture()
    context = build_pit_market_context(
        features,
        index,
        allowed_positions=np.arange(6),
        feature_indices=(0, 1),
        bars_per_day=4,
    )
    targets = np.zeros((7, 4), dtype="float32")
    masks = np.ones((7, 4), dtype=bool)
    scaler = fit_market_context_standardizer(
        context,
        index,
        train_positions=np.arange(6),
    )

    with pytest.raises(ContractError, match="unavailable market context"):
        ContextualLazyWindowDataset(
            features,
            np.asarray([6]),
            targets,
            masks,
            np.zeros(8),
            np.ones(8),
            context,
            scaler,
        )


def test_v17_trial_parser_and_decision_enforce_effect_capacity_and_gate_use() -> None:
    raw = [
        {
            "trial_id": "candidate",
            "model_kind": "market_conditioned_temporal_context",
            "channels": 16,
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
            "dropout": 0.0,
            "learning_rate": 0.003,
            "batch_size": 128,
            "strategy": "smooth_l1",
            "padding_mode": "chomp",
            "bars_per_day": 48,
            "market_context_dim": 24,
            "market_context_hidden": 4,
            "market_gate_scale": 0.25,
        }
    ]
    trials = parse_real_tcn_trials(raw)
    validate_tcn_tuning_plan(
        trials, input_steps=480, max_epochs=8, patience=2, min_delta=0.002
    )
    assert trials[0].model_kind == "market_conditioned_temporal_context"

    rows: list[dict[str, float | int | str]] = []
    for fold in range(5):
        for trial_id, rankic, parameter_count, gate_l2 in (
            ("control", 0.087, 6524, float("nan")),
            ("candidate", 0.091, 6784, 0.01),
        ):
            rows.append(
                {
                    "trial_id": trial_id,
                    "fold": fold,
                    "seed": 7,
                    "best_mean_daily_rankic": rankic,
                    "rankic_1d": rankic,
                    "rankic_2d": rankic,
                    "rankic_3d": rankic,
                    "rankic_5d": rankic,
                    "samples_per_second": 5500.0,
                    "parameter_count": parameter_count,
                    "market_gate_output_l2": gate_l2,
                }
            )
    leaderboard = pd.DataFrame(rows)
    kwargs: dict[str, Any] = {
        "control_trial_id": "control",
        "candidate_trial_id": "candidate",
        "min_mean_rankic": 0.09,
        "min_mean_rankic_delta": 0.003,
        "min_positive_folds": 5,
        "min_nondegrading_folds": 3,
        "min_horizon_delta_1d": 0.0,
        "min_horizon_delta_2d": -0.003,
        "min_horizon_delta_3d": -0.005,
        "min_horizon_delta_5d": -0.005,
        "min_median_samples_per_second": 5000.0,
        "min_market_gate_output_l2": 1e-12,
        "control_parameter_count": 6524,
        "candidate_parameter_count": 6784,
    }
    effect = evaluate_pit_market_conditioning_seed7(leaderboard, **kwargs)
    assert effect.winner_trial_id == "candidate"
    final = finalize_pit_market_conditioning_seed7(
        effect,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )
    assert final.status == "pit_market_conditioning_seed7_admitted_v17"
    assert final.confirmation_seeds_authorized == (17, 27)

    weak = leaderboard.copy()
    weak.loc[weak["trial_id"].eq("candidate"), "best_mean_daily_rankic"] = 0.089
    stopped = evaluate_pit_market_conditioning_seed7(weak, **kwargs)
    candidate = stopped.summary.loc[stopped.summary["trial_id"].eq("candidate")].iloc[0]
    assert stopped.winner_trial_id is None
    assert "mean_rankic_delta_below_gate" in candidate["blockers"]


def test_market_context_runs_through_the_real_tuning_batch_seam() -> None:
    rng = np.random.default_rng(71)
    dates = [f"2025-01-{value:02d}" for value in range(2, 10)]
    sample_count = len(dates) * 4
    features = rng.normal(size=(sample_count, 3, 24)).astype("float32")
    index_rows = []
    label_rows = []
    for position in range(sample_count):
        signal_date = dates[position // 4]
        sample_id = f"sample-{position}"
        index_rows.append(
            {
                "sample_position": position,
                "sample_id": sample_id,
                "instrument_id": f"stock-{position % 4}",
                "signal_date": signal_date,
                "window_end_at": pd.Timestamp(f"{signal_date} 15:00:00+08:00"),
            }
        )
        for horizon in (1, 2, 3, 5):
            label_rows.append(
                {
                    "sample_position": position,
                    "sample_id": sample_id,
                    "signal_date": signal_date,
                    "horizon": horizon,
                    "rank_target": float((position % 4) / 1.5 - 1.0),
                    "valid": True,
                }
            )
    window_index = pd.DataFrame(index_rows)
    labels = pd.DataFrame(label_rows)
    split_manifest = pd.DataFrame(
        {
            "fold": 0,
            "sample_position": np.arange(sample_count),
            "stage": [
                "train" if position < 20 else "validation"
                for position in range(sample_count)
            ],
        }
    )
    context = build_pit_market_context(
        features,
        window_index,
        allowed_positions=np.arange(sample_count),
        feature_indices=(0, 1),
        bars_per_day=8,
    )
    trial = TCNTuningTrial(
        trial_id="contextual",
        channels=4,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=8,
        model_kind="market_conditioned_temporal_context",
        padding_mode="chomp",
        bars_per_day=8,
        market_context_dim=8,
        market_context_hidden=2,
        market_gate_scale=0.25,
    )

    result = run_tcn_validation_sweep(
        features,
        window_index,
        labels,
        split_manifest,
        trials=(trial,),
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0,
        torch_threads=1,
        market_context=context,
    )
    assert result.leaderboard["model_kind"].tolist() == [
        "market_conditioned_temporal_context"
    ]
    assert result.leaderboard["market_context_identity"].tolist() == [context.identity]
    assert float(result.leaderboard.iloc[0]["market_gate_output_l2"]) > 0

    with pytest.raises(ContractError, match="requires PIT market context"):
        run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split_manifest,
            trials=(trial,),
            seed=7,
            max_epochs=2,
            patience=1,
            min_delta=0.0,
            torch_threads=1,
        )
