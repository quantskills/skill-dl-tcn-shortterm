from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.dynamic_multiscale import (
    evaluate_frozen_shape_soft_rankic_multiseed,
)
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    run_tcn_validation_sweep,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import DynamicHorizonSkipTCN


def _trial(trial_id: str, strategy: str) -> TCNTuningTrial:
    return TCNTuningTrial(
        trial_id=trial_id,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=6,
        model_kind="dynamic_horizon_skip",
        strategy=strategy,  # type: ignore[arg-type]
        padding_mode="chomp",
        dynamic_skip_hidden=2,
        dynamic_skip_scale=1.0,
        dynamic_skip_shape_residual=True,
        dynamic_skip_shape_residual_scale=0.25,
        dynamic_skip_frozen_parent=True,
        soft_rankic_weight=0.05,
        soft_rank_temperature=0.1,
    )


def _parent_state() -> dict[str, torch.Tensor]:
    model = DynamicHorizonSkipTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        dropout=0.0,
        padding_mode="chomp",
        dynamic_skip_hidden=2,
        dynamic_skip_scale=1.0,
    )
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def _tiny_data() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(301)
    features = rng.normal(size=(18, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(18),
            "sample_id": [f"rank-{value}" for value in range(18)],
            "signal_date": [f"2025-07-{2 + value // 6:02d}" for value in range(18)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"rank-{sample}",
                "signal_date": index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": float((sample % 6) / 5 * 2 - 1),
                "valid": True,
            }
            for sample in range(18)
            for horizon in (1, 2, 3, 5)
        ]
    )
    split = index[["sample_position"]].copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 12 + ["validation"] * 6
    return features, index, labels, split


def test_v30_frozen_shape_trials_share_grouped_batches_and_shape_only_optimizer() -> None:
    grouped = _trial("grouped", "grouped_smooth_l1")
    candidate = _trial("candidate", "soft_rankic")
    assert validate_tcn_tuning_plan(
        (grouped, candidate),
        input_steps=16,
        max_epochs=2,
        patience=1,
        min_delta=0.0005,
    ) == (grouped, candidate)

    nonfrozen = TCNTuningTrial(
        **{**candidate.__dict__, "trial_id": "nonfrozen", "dynamic_skip_frozen_parent": False}
    )
    with pytest.raises(ContractError, match="frozen parent shape residual"):
        validate_tcn_tuning_plan(
            (nonfrozen,),
            input_steps=16,
            max_epochs=2,
            patience=1,
            min_delta=0.0005,
        )

    features, index, labels, split = _tiny_data()
    parent = _parent_state()
    result = run_tcn_validation_sweep(
        features,
        index,
        labels,
        split,
        trials=(grouped, candidate),
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0005,
        checkpoint_min_delta=0.0,
        torch_threads=1,
        frozen_parent_states={
            "grouped-fold-0": parent,
            "candidate-fold-0": parent,
        },
    )
    rows = result.leaderboard.set_index("trial_id")
    assert set(rows["batching_identity"]) == {"date-grouped"}
    assert rows.loc["grouped", "loss_identity"] == "date-grouped-smooth-l1"
    assert (
        rows.loc["candidate", "loss_identity"]
        == "smooth-l1+0.05-soft-rankic-tau-0.1"
    )
    assert set(rows["optimizer_group_identity"]) == {
        "shape-residual-only-lr-0.003"
    }
    assert set(rows["trainable_parameter_count"]) == {22}
    assert set(rows["frozen_parent_state_drift_max"]) == {0.0}
    assert set(rows["parent_prediction_max_abs_error"]) == {0.0}


def test_v30_real_parser_accepts_registered_grouped_control() -> None:
    raw = {
        "trial_id": "grouped",
        "channels": 4,
        "kernel_size": 2,
        "dilations": [1, 2, 4, 8],
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 6,
        "model_kind": "dynamic_horizon_skip",
        "strategy": "grouped_smooth_l1",
        "padding_mode": "chomp",
        "dynamic_skip_hidden": 2,
        "dynamic_skip_scale": 1.0,
        "dynamic_skip_shape_residual": True,
        "dynamic_skip_shape_residual_scale": 0.25,
        "dynamic_skip_frozen_parent": True,
    }
    parsed = parse_real_tcn_trials([raw])
    assert parsed[0].strategy == "grouped_smooth_l1"


def _evidence(
    *, candidate_loss: str = "smooth-l1+0.05-soft-rankic-tau-0.1", control_delta: float = 0.0002
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current: list[dict[str, object]] = []
    historical: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        for fold in range(5):
            parent = 0.099
            candidate = 0.1002
            grouped = candidate - control_delta
            for trial_id, score, parameters in (
                ("static", 0.096, 6260),
                ("parent", parent, 6348),
                ("v25", 0.095, 6352),
                ("v26", 0.097, 6436),
                ("v28", 0.1000, 6436),
            ):
                historical.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "fold": fold,
                        "best_mean_daily_rankic": score,
                        "rankic_1d": score,
                        "rankic_2d": score,
                        "rankic_3d": score,
                        "rankic_5d": score,
                        "parameter_count": parameters,
                    }
                )
            for trial_id, score, strategy, loss in (
                ("grouped", grouped, "grouped_smooth_l1", "date-grouped-smooth-l1"),
                ("candidate", candidate, "soft_rankic", candidate_loss),
            ):
                current.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "fold": fold,
                        "best_epoch": 1,
                        "baseline_epoch": 0,
                        "baseline_mean_daily_rankic": parent,
                        "best_mean_daily_rankic": score,
                        "rankic_1d": score,
                        "rankic_2d": score,
                        "rankic_3d": score,
                        "rankic_5d": score,
                        "samples_per_second": 5000.0,
                        "parameter_count": 6436,
                        "trainable_parameter_count": 88,
                        "frozen_parameter_count": 6348,
                        "frozen_parent_state_drift_max": 0.0,
                        "parent_prediction_max_abs_error": 0.0,
                        "dynamic_skip_shape_output_weight_l2": 0.1,
                        "dynamic_skip_frozen_parent": True,
                        "dynamic_skip_shape_residual": True,
                        "dynamic_skip_shape_residual_scale": 0.25,
                        "dynamic_skip_raw_parameter_count": 88,
                        "dynamic_skip_shape_residual_parameter_count": 88,
                        "dynamic_skip_shape_normalization_parameter_count": 0,
                        "optimizer_group_identity": "shape-residual-only-lr-0.003",
                        "learning_rate": 0.003,
                        "parent_checkpoint_sha256": f"{seed + fold:064x}",
                        "checkpoint_min_delta": 0.0,
                        "patience_min_delta": 0.0005,
                        "checkpoint_selection_identity": (
                            "best-any-strict-improvement+patience-material-0.0005"
                        ),
                        "strategy": strategy,
                        "loss_identity": loss,
                        "batching_identity": "date-grouped",
                        "soft_rankic_weight": 0.05 if trial_id == "candidate" else None,
                        "soft_rank_temperature": 0.1 if trial_id == "candidate" else None,
                    }
                )
                diagnostics.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "fold": fold,
                        "raw_only_mean_daily_rankic": parent,
                        "shape_residual_weight_effect_max": 1e-4,
                        "simplex_error_max": 1e-7,
                    }
                )
    return pd.DataFrame(current), pd.DataFrame(historical), pd.DataFrame(diagnostics)


def _decision(
    *, candidate_loss: str = "smooth-l1+0.05-soft-rankic-tau-0.1", control_delta: float = 0.0002
):
    current, historical, diagnostics = _evidence(
        candidate_loss=candidate_loss, control_delta=control_delta
    )
    return evaluate_frozen_shape_soft_rankic_multiseed(
        current,
        historical,
        diagnostics,
        {"model_step_speed_ratio": 3.5, "end_to_end_speed_ratio": 3.2},
        control_trial_id="static",
        parent_candidate_trial_id="parent",
        v25_trial_id="v25",
        v26_trial_id="v26",
        v28_trial_id="v28",
        grouped_control_trial_id="grouped",
        candidate_trial_id="candidate",
        expected_seeds=(7, 17, 27),
        min_mean_rankic=0.0996,
        min_positive_units=15,
        min_parent_mean_rankic_delta=0.00075,
        min_control_mean_rankic_delta=0.002,
        min_v26_mean_rankic_delta=0.0015,
        min_v25_mean_rankic_delta=0.003,
        min_v28_mean_rankic_delta=0.00015,
        min_grouped_control_mean_rankic_delta=0.00015,
        min_nondegrading_folds_per_seed=5,
        min_horizon_parent_delta_1d=-0.001,
        min_horizon_parent_delta_2d=-0.001,
        min_horizon_parent_delta_3d=-0.001,
        min_horizon_parent_delta_5d=-0.001,
        max_parent_rankic_abs_error=1e-7,
        max_parent_prediction_abs_error=1e-7,
        min_trained_effect_units=8,
        min_shape_output_weight_l2=1e-12,
        min_shape_residual_weight_effect=1e-6,
        max_simplex_error=1e-6,
        min_median_samples_per_second=4500.0,
        candidate_parameter_count=6436,
        trainable_parameter_count=88,
        frozen_parameter_count=6348,
        shape_residual_scale=0.25,
        learning_rate=0.003,
        soft_rankic_weight=0.05,
        soft_rank_temperature=0.1,
        checkpoint_min_delta=0.0,
        patience_min_delta=0.0005,
        min_model_step_speed_ratio=3.0,
        min_end_to_end_speed_ratio=3.0,
    )


def test_v30_decision_separates_identity_and_incremental_rank_value() -> None:
    decision = _decision()
    assert decision.status == "shape_rank_objective_confirmed_v30"
    assert decision.integrity_passed is True
    assert decision.effect_passed is True
    assert decision.aggregate["grouped_control_mean_rankic_delta"] == pytest.approx(
        0.0002
    )

    drifted = _decision(candidate_loss="smooth-l1")
    assert drifted.status == "stop_shape_rank_integrity_v30"
    assert drifted.integrity_passed is False

    no_gain = _decision(control_delta=0.0)
    assert no_gain.status == "stop_shape_rank_no_gain_v30"
    assert no_gain.integrity_passed is True
    assert no_gain.effect_passed is False
