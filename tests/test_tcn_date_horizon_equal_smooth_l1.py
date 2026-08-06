from __future__ import annotations

import pytest
import pandas as pd
import torch

from skill_dl_tcn_shortterm.loss_alignment import (
    evaluate_date_horizon_equal_smooth_l1,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.training_data import masked_smooth_l1
from skill_dl_tcn_shortterm.v9_objective import date_horizon_equal_smooth_l1


def test_v32_date_horizon_equal_loss_does_not_overweight_large_dates() -> None:
    prediction = torch.zeros((4, 4), dtype=torch.float32, requires_grad=True)
    target = torch.zeros((4, 4), dtype=torch.float32)
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[:, 0] = True
    target[3, 0] = 2.0
    dates = ["2025-08-01", "2025-08-01", "2025-08-01", "2025-08-04"]

    label_mean = masked_smooth_l1(prediction, target, mask)
    equal = date_horizon_equal_smooth_l1(
        prediction, target, mask, dates
    )

    assert float(label_mean.detach()) == pytest.approx(0.375)
    assert float(equal.total.detach()) == pytest.approx(0.75)
    assert equal.group_count == 2
    assert equal.valid_label_count == 4
    equal.total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_v32_date_horizon_equal_loss_rejects_empty_groups() -> None:
    prediction = torch.zeros((2, 4), dtype=torch.float32)
    target = torch.zeros((2, 4), dtype=torch.float32)
    mask = torch.zeros((2, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="valid date/horizon"):
        date_horizon_equal_smooth_l1(
            prediction,
            target,
            mask,
            ["2025-08-01", "2025-08-04"],
        )


def test_v32_parser_records_the_only_loss_reduction_variable() -> None:
    trial = parse_real_tcn_trials(
        [
            {
                "trial_id": "date-horizon-equal",
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
                "date_batch_order": "fixed_once",
                "grouped_smooth_l1_reduction": "date_horizon_mean",
            }
        ]
    )[0]

    assert trial.grouped_smooth_l1_reduction == "date_horizon_mean"


def _decision_evidence(
    *, candidate_delta: float = 0.001
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leaderboard_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        for fold in range(5):
            parent_hash = f"{seed + fold:064x}"
            for trial_id, reduction, loss_identity, delta in (
                ("control", "label_mean", "date-grouped-smooth-l1", 0.0),
                (
                    "candidate",
                    "date_horizon_mean",
                    "date-horizon-equal-smooth-l1",
                    candidate_delta,
                ),
            ):
                leaderboard_rows.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "fold": fold,
                        "best_epoch": 1,
                        "best_mean_daily_rankic": 0.100 + fold * 0.001 + delta,
                        "samples_per_second": 5_000.0,
                        "strategy": "grouped_smooth_l1",
                        "loss_identity": loss_identity,
                        "batching_identity": "date-grouped",
                        "date_batch_order": "fixed_once",
                        "grouped_smooth_l1_reduction": reduction,
                        "date_order_fingerprint_count": 1,
                        "median_epoch_gradient_norm_cv": 0.25,
                        "median_labels_per_loss_group": 4.0,
                        "frozen_parent_state_drift_max": 0.0,
                        "parent_prediction_max_abs_error": 0.0,
                        "parent_checkpoint_sha256": parent_hash,
                    }
                )
                history_rows.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "fold": fold,
                        "stage": "validation",
                        "loss_group_count_mean": 8.0,
                        "valid_label_count_mean": 32.0,
                        "labels_per_loss_group_mean": 4.0,
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
                            "control_rankic": 0.1,
                            "candidate_rankic": 0.1 + candidate_delta,
                            "rankic_delta": candidate_delta,
                        }
                    )
    return (
        pd.DataFrame(leaderboard_rows),
        pd.DataFrame(history_rows),
        pd.DataFrame(daily_rows),
    )


def test_v32_decision_separates_mechanism_effect_and_speed() -> None:
    leaderboard, history, daily = _decision_evidence()
    decision = evaluate_date_horizon_equal_smooth_l1(
        leaderboard,
        history,
        daily,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        candidate_trial_id="candidate",
        bootstrap_draws=64,
    )
    assert decision.status == "date_horizon_equal_smooth_l1_confirmed_v32"
    assert decision.integrity_passed is True
    assert decision.mechanism_passed is True
    assert decision.effect_passed is True
    assert decision.speed_passed is True

    no_gain_leaderboard, no_gain_history, no_gain_daily = _decision_evidence(
        candidate_delta=0.0
    )
    no_gain = evaluate_date_horizon_equal_smooth_l1(
        no_gain_leaderboard,
        no_gain_history,
        no_gain_daily,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        candidate_trial_id="candidate",
        bootstrap_draws=64,
    )
    assert no_gain.status == "stop_date_horizon_equal_no_gain_v32"

    bad_mechanism = leaderboard.copy()
    bad_mechanism.loc[
        bad_mechanism["trial_id"].eq("candidate"),
        "median_labels_per_loss_group",
    ] = 1.0
    mechanism = evaluate_date_horizon_equal_smooth_l1(
        bad_mechanism,
        history,
        daily,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="control",
        candidate_trial_id="candidate",
        bootstrap_draws=64,
    )
    assert mechanism.status == "stop_date_horizon_equal_mechanism_v32"
