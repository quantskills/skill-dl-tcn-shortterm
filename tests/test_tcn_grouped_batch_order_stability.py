from __future__ import annotations

import pandas as pd

from skill_dl_tcn_shortterm.batch_stability import (
    evaluate_grouped_batch_order_stability,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.v9_objective import DateGroupedBatchSampler


def _batches(sampler: DateGroupedBatchSampler) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(batch) for batch in sampler)


def test_v31_epoch_seeded_date_order_changes_by_epoch_and_replays() -> None:
    dates = [f"2025-08-{day:02d}" for day in range(1, 9) for _ in range(2)]
    sampler = DateGroupedBatchSampler(
        dates,
        shuffle_dates=True,
        seed=17,
        batch_size=4,
        order_policy="epoch_seeded",
    )

    epoch_zero = _batches(sampler)
    fingerprint_zero = sampler.order_fingerprint()
    sampler.set_epoch(1)
    epoch_one = _batches(sampler)
    fingerprint_one = sampler.order_fingerprint()
    sampler.set_epoch(0)

    assert epoch_zero != epoch_one
    assert fingerprint_zero != fingerprint_one
    assert _batches(sampler) == epoch_zero
    assert sampler.order_fingerprint() == fingerprint_zero


def test_v31_fixed_once_date_order_preserves_v30_behavior() -> None:
    dates = [f"2025-08-{day:02d}" for day in range(1, 9) for _ in range(2)]
    sampler = DateGroupedBatchSampler(
        dates,
        shuffle_dates=True,
        seed=27,
        batch_size=4,
        order_policy="fixed_once",
    )
    baseline = _batches(sampler)
    fingerprint = sampler.order_fingerprint()

    sampler.set_epoch(7)

    assert _batches(sampler) == baseline
    assert sampler.order_fingerprint() == fingerprint


def test_v31_real_parser_requires_explicit_epoch_seeded_identity() -> None:
    trial = parse_real_tcn_trials(
        [
            {
                "trial_id": "epoch-seeded",
                "channels": 4,
                "kernel_size": 2,
                "dilations": [1, 2, 4, 8],
                "dropout": 0.0,
                "learning_rate": 0.003,
                "batch_size": 8,
                "model_kind": "dynamic_horizon_skip",
                "strategy": "grouped_smooth_l1",
                "padding_mode": "chomp",
                "dynamic_skip_hidden": 2,
                "dynamic_skip_scale": 1.0,
                "dynamic_skip_shape_residual": True,
                "dynamic_skip_shape_residual_scale": 0.25,
                "dynamic_skip_frozen_parent": True,
                "date_batch_order": "epoch_seeded",
            }
        ]
    )[0]

    assert trial.date_batch_order == "epoch_seeded"


def _decision_evidence(
    *, candidate_delta: float = 0.001
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leaderboard_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        for fold in range(5):
            control_rankic = 0.100 + fold * 0.001
            for trial_id, policy, delta, fingerprints in (
                ("control", "fixed_once", 0.0, 1),
                ("candidate", "epoch_seeded", candidate_delta, 2),
            ):
                leaderboard_rows.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "fold": fold,
                        "best_mean_daily_rankic": control_rankic + delta,
                        "samples_per_second": 5_000.0,
                        "strategy": "grouped_smooth_l1",
                        "loss_identity": "date-grouped-smooth-l1",
                        "batching_identity": "date-grouped",
                        "date_batch_order": policy,
                        "date_order_fingerprint_count": fingerprints,
                        "median_epoch_gradient_norm_cv": 0.25,
                        "completed_epochs": 2,
                        "frozen_parent_state_drift_max": 0.0,
                        "parent_prediction_max_abs_error": 0.0,
                        "parent_checkpoint_sha256": f"{seed + fold:064x}",
                    }
                )
                for epoch in (1, 2):
                    epoch_rows.append(
                        {
                            "trial_id": trial_id,
                            "seed": seed,
                            "fold": fold,
                            "epoch": epoch,
                            "stage": "validation",
                            "date_order_fingerprint": (
                                f"{trial_id}-{seed}-{fold}-{epoch if policy == 'epoch_seeded' else 0}"
                            ),
                        }
                    )
            for horizon in (1, 2, 3, 5):
                for day in range(8):
                    daily_rows.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "horizon": horizon,
                            "signal_date": f"2025-{fold + 1:02d}-{day + 1:02d}",
                            "control_rankic": 0.1 + day * 0.001,
                            "candidate_rankic": 0.1 + day * 0.001 + candidate_delta,
                            "rankic_delta": candidate_delta,
                        }
                    )
    return (
        pd.DataFrame(leaderboard_rows),
        pd.DataFrame(epoch_rows),
        pd.DataFrame(daily_rows),
    )


def test_v31_decision_requires_integrity_mechanism_effect_and_speed() -> None:
    leaderboard, history, daily = _decision_evidence()
    decision = evaluate_grouped_batch_order_stability(
        leaderboard,
        history,
        daily,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        candidate_trial_id="candidate",
        bootstrap_draws=64,
    )
    assert decision.status == "epoch_seeded_grouped_batch_confirmed_v31"
    assert decision.integrity_passed is True
    assert decision.mechanism_passed is True
    assert decision.effect_passed is True
    assert decision.speed_passed is True

    no_gain_leaderboard, no_gain_history, no_gain_daily = _decision_evidence(
        candidate_delta=0.0
    )
    no_gain = evaluate_grouped_batch_order_stability(
        no_gain_leaderboard,
        no_gain_history,
        no_gain_daily,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        candidate_trial_id="candidate",
        bootstrap_draws=64,
    )
    assert no_gain.status == "stop_epoch_seeded_no_gain_v31"

    mechanism_leaderboard = leaderboard.copy()
    mechanism_leaderboard.loc[
        mechanism_leaderboard["trial_id"].eq("candidate"),
        "date_order_fingerprint_count",
    ] = 1
    mechanism = evaluate_grouped_batch_order_stability(
        mechanism_leaderboard,
        history,
        daily,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        candidate_trial_id="candidate",
        bootstrap_draws=64,
    )
    assert mechanism.status == "stop_epoch_seeded_mechanism_not_confirmed_v31"
