from __future__ import annotations

import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.v9_selection import (
    build_seed7_trials,
    select_seed7_candidate,
)


def _upstream(
    *,
    skip: str = "horizon_skip_applicable",
    rank: str = "rank_objective_not_resolvable",
    pcgrad: str = "pcgrad_not_applicable",
    infra: str = "infra_optimization_not_applicable",
) -> dict[str, dict[str, object]]:
    return {
        "horizon_skip": {"status": skip, "sealed_test_accessed": False},
        "rank_objective": {"status": rank, "sealed_test_accessed": False},
        "pcgrad": {"status": pcgrad, "sealed_test_accessed": False},
        "infra": {"status": infra, "sealed_test_accessed": False},
    }


def _leaderboard(candidate: str, *, rankic: float = 0.095, speed: float = 5500) -> pd.DataFrame:
    rows = []
    for trial_id, score, throughput, parameters in [
        ("lite-c16-no-dropout", 0.088, 5900.0, 1100),
        (candidate, rankic, speed, 1400),
    ]:
        for fold in range(5):
            rows.append(
                {
                    "trial_id": trial_id,
                    "fold": fold,
                    "seed": 7,
                    "best_mean_daily_rankic": score + fold * 0.0001,
                    "samples_per_second": throughput,
                    "parameter_count": parameters,
                    "model_step_seconds": 1.0,
                    "data_wait_seconds": 0.1,
                    "validation_seconds": 0.2,
                    "complete_cycle_seconds": 1.3,
                    "time_to_best_seconds": 2.0,
                    "precision": "float32",
                    "torch_threads": 4,
                    "batch_size": 128,
                    "data_identity": "a" * 64,
                    "fold_identity": "b" * 64,
                    "evaluation_identity": "c" * 64,
                    "sealed_test_accessed": False,
                }
            )
    return pd.DataFrame(rows)


def test_seed7_trials_are_built_only_from_triggered_receipts_without_replacement() -> None:
    trials = build_seed7_trials(_upstream())
    assert [trial.trial_id for trial in trials] == ["v9b-horizon-skip"]
    assert trials[0].seed == 7
    assert trials[0].fold_ids == (0, 1, 2, 3, 4)
    assert trials[0].max_epochs == 8
    assert trials[0].patience == 2
    assert trials[0].min_delta == pytest.approx(0.002)
    assert trials[0].infra_enabled is False

    all_triggered = build_seed7_trials(
        _upstream(
            rank="rank_objective_allowed",
            pcgrad="pcgrad_applicable",
            infra="causal_infra_acceleration_accepted",
        )
    )
    assert [trial.trial_id for trial in all_triggered] == [
        "v9b-horizon-skip",
        "v9c-rank-objective",
        "v9d-pcgrad",
    ]
    assert all(trial.infra_enabled for trial in all_triggered)


def test_seed7_selection_admits_one_deterministic_unified_pareto_candidate() -> None:
    trials = build_seed7_trials(_upstream())
    leaderboard = _leaderboard(trials[0].trial_id)

    decision = select_seed7_candidate(
        leaderboard,
        registered_trials=trials,
        control_trial_id="lite-c16-no-dropout",
    )
    replay = select_seed7_candidate(
        leaderboard,
        registered_trials=trials,
        control_trial_id="lite-c16-no-dropout",
    )

    assert decision.status == "seed7_winner_admitted"
    assert decision.winner_trial_id == "v9b-horizon-skip"
    assert decision.confirmation_seeds == (17, 27)
    pd.testing.assert_frame_equal(decision.summary, replay.summary)


def test_seed7_selection_stops_without_extra_trials_or_sealed_access() -> None:
    no_trials = build_seed7_trials(
        _upstream(skip="horizon_skip_not_applicable")
    )
    stopped = select_seed7_candidate(
        pd.DataFrame(),
        registered_trials=no_trials,
        control_trial_id="lite-c16-no-dropout",
    )
    assert stopped.status == "stop_no_pareto_gain_v9"
    assert stopped.winner_trial_id is None
    assert stopped.confirmation_seeds == ()

    trials = build_seed7_trials(_upstream())
    slow = select_seed7_candidate(
        _leaderboard(trials[0].trial_id, speed=4999),
        registered_trials=trials,
        control_trial_id="lite-c16-no-dropout",
    )
    assert slow.status == "stop_no_pareto_gain_v9"
    assert "throughput_below_5000" in slow.blockers

    injected = pd.concat(
        [_leaderboard(trials[0].trial_id), _leaderboard("unregistered").loc[lambda frame: frame["trial_id"].eq("unregistered")]],
        ignore_index=True,
    )
    with pytest.raises(ContractError, match="unregistered trial"):
        select_seed7_candidate(
            injected,
            registered_trials=trials,
            control_trial_id="lite-c16-no-dropout",
        )

    sealed = _leaderboard(trials[0].trial_id)
    sealed.loc[0, "sealed_test_accessed"] = True
    with pytest.raises(ContractError, match="sealed"):
        select_seed7_candidate(
            sealed,
            registered_trials=trials,
            control_trial_id="lite-c16-no-dropout",
        )
