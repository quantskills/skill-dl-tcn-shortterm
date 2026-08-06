from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.v9_selection import Seed7Trial
from skill_dl_tcn_shortterm.v9_training import (
    V9TrainingRequest,
    run_v9_candidate_sweep,
)
from skill_dl_tcn_shortterm.tuning import TCNTuningResult


def _request() -> V9TrainingRequest:
    rng = np.random.default_rng(71)
    sample_count = 60
    features = rng.normal(size=(sample_count, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(sample_count),
            "sample_id": [f"s{value}" for value in range(sample_count)],
            "signal_date": [f"2025-01-{value // 5 + 1:02d}" for value in range(sample_count)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "signal_date": index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": float((sample % 5) / 2 - 1),
                "valid": True,
            }
            for sample in range(sample_count)
            for horizon in [1, 2, 3, 5]
        ]
    )
    split_parts = []
    for fold in range(5):
        part = pd.DataFrame(
            {
                "sample_position": range(sample_count),
                "fold": fold,
                "stage": ["train"] * 40 + ["validation"] * 20,
                "sealed": False,
            }
        )
        split_parts.append(part)
    return V9TrainingRequest(
        features=features,
        window_index=index,
        labels=labels,
        split_manifest=pd.concat(split_parts, ignore_index=True),
        channels=3,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=10,
        max_epochs=2,
        patience=1,
        min_delta=0.0,
        torch_threads=1,
        protocol_identities={
            "data": "a" * 64,
            "fold_manifest": "b" * 64,
            "evaluation": "c" * 64,
        },
    )


def test_actual_v9_sweep_trains_every_triggered_candidate_and_accounts_cycle_time() -> None:
    trials = (
        Seed7Trial("horizon_skip", "v9b-horizon-skip", max_epochs=2, patience=1, min_delta=0.0, infra_enabled=True),
        Seed7Trial("rank_objective", "v9c-rank-objective", max_epochs=2, patience=1, min_delta=0.0, infra_enabled=True),
        Seed7Trial("pcgrad", "v9d-pcgrad", max_epochs=2, patience=1, min_delta=0.0, infra_enabled=True),
    )
    result = run_v9_candidate_sweep(
        _request(),
        registered_trials=trials,
        seed=7,
        # This test exercises the full two-phase training/accounting path. Parent
        # speed qualification is covered independently with deterministic
        # leaderboard fixtures below; using wall-clock throughput here made the
        # expected phase-two candidates depend on concurrent test-runner load.
        frozen_parent_kind="horizon_skip",
    )

    assert set(result.leaderboard["trial_id"]) == {
        "lite-c16-no-dropout",
        "v9b-horizon-skip",
        "v9c-rank-objective",
        "v9d-pcgrad",
    }
    assert len(result.leaderboard) == 20
    assert set(result.leaderboard["fold"]) == {0, 1, 2, 3, 4}
    assert set(result.leaderboard["seed"]) == {7}
    assert not result.epoch_history.empty
    assert result.leaderboard[
        [
            "model_step_seconds",
            "data_wait_seconds",
            "validation_seconds",
            "complete_cycle_seconds",
            "time_to_best_seconds",
            "samples_per_second",
            "model_step_samples_per_second",
        ]
    ].gt(0).all().all()
    assert set(result.leaderboard["infra_identity"]) == {"padding-chomp"}
    assert set(result.leaderboard["sealed_test_accessed"]) == {False}
    assert any("pairwise" in value for value in result.leaderboard["loss_identity"])
    assert any("pcgrad" in value for value in result.leaderboard["loss_identity"])
    batching = result.leaderboard.groupby("trial_id")["batching_identity"].first()
    assert batching["v9c-rank-objective"] == "date-grouped"
    assert batching["v9d-pcgrad"] == "seeded-random"
    assert len(result.best_states) == 20
    horizon_weights = result.leaderboard.loc[
        result.leaderboard["model_kind"].eq("horizon_skip"),
        "simplex_weights",
    ]
    assert not horizon_weights.empty
    for serialized in horizon_weights:
        weights = np.asarray(json.loads(serialized), dtype="float64")
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(4), atol=1e-6)


def test_actual_v9_sweep_rejects_sealed_rows_before_training() -> None:
    request = _request()
    sealed = request.split_manifest.copy()
    sealed.loc[0, "sealed"] = True
    request = V9TrainingRequest(**{**request.__dict__, "split_manifest": sealed})
    with pytest.raises(ContractError, match="sealed"):
        run_v9_candidate_sweep(
            request,
            registered_trials=(Seed7Trial("horizon_skip", "v9b-horizon-skip", max_epochs=2, patience=1, min_delta=0.0),),
            seed=7,
        )


def test_rank_and_pcgrad_trials_inherit_the_selected_horizon_skip_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_sweep(*args: object, trials: tuple[object, ...], **kwargs: object) -> TCNTuningResult:
        del args, kwargs
        calls.append(trials)
        rows = []
        states: dict[str, dict[str, torch.Tensor]] = {}
        for trial in trials:
            trial_id = str(getattr(trial, "trial_id"))
            mean_rankic = 0.12 if trial_id == "v9b-horizon-skip" else 0.10
            for fold in range(5):
                rows.append(
                    {
                        "trial_id": trial_id,
                        "fold": fold,
                        "best_mean_daily_rankic": mean_rankic,
                        "samples_per_second": 6_000.0,
                    }
                )
                states[f"{trial_id}-fold-{fold}"] = {}
        return TCNTuningResult(pd.DataFrame(), pd.DataFrame(rows), states)

    monkeypatch.setattr(
        "skill_dl_tcn_shortterm.v9_training.run_tcn_validation_sweep",
        fake_sweep,
    )
    trials = (
        Seed7Trial("horizon_skip", "v9b-horizon-skip", max_epochs=2, patience=1, min_delta=0.0, infra_enabled=True),
        Seed7Trial("rank_objective", "v9c-rank-objective", max_epochs=2, patience=1, min_delta=0.0, infra_enabled=True),
        Seed7Trial("pcgrad", "v9d-pcgrad", max_epochs=2, patience=1, min_delta=0.0, infra_enabled=True),
    )

    run_v9_candidate_sweep(_request(), registered_trials=trials, seed=7)

    assert len(calls) == 2
    phase_two = calls[1]
    assert {getattr(trial, "trial_id") for trial in phase_two} == {
        "v9c-rank-objective",
        "v9d-pcgrad",
    }
    assert {getattr(trial, "model_kind") for trial in phase_two} == {
        "horizon_skip"
    }
    assert {getattr(trial, "padding_mode") for trial in phase_two} == {"chomp"}


def test_rank_trial_is_not_spent_without_a_speed_qualified_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_sweep(
        *args: object,
        trials: tuple[object, ...],
        **kwargs: object,
    ) -> TCNTuningResult:
        del args, kwargs
        calls.append(trials)
        rows = [
            {
                "trial_id": str(getattr(trial, "trial_id")),
                "fold": fold,
                "best_mean_daily_rankic": 0.10,
                "samples_per_second": 4_999.0,
            }
            for trial in trials
            for fold in range(5)
        ]
        return TCNTuningResult(pd.DataFrame(), pd.DataFrame(rows), {})

    monkeypatch.setattr(
        "skill_dl_tcn_shortterm.v9_training.run_tcn_validation_sweep",
        fake_sweep,
    )
    result = run_v9_candidate_sweep(
        _request(),
        registered_trials=(
            Seed7Trial(
                "rank_objective",
                "v9c-rank-objective",
                max_epochs=2,
                patience=1,
                min_delta=0.0,
            ),
        ),
        seed=7,
    )

    assert len(calls) == 1
    assert set(result.leaderboard["trial_id"]) == {"lite-c16-no-dropout"}
