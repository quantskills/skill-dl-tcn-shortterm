from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.tuning import (
    TCNTuningTrial,
    build_validation_rankic_plan,
    cross_sectional_validation_rankic,
    run_tcn_validation_sweep,
    build_tcn_optimizer,
    select_tcn_candidate,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_representation import (
    DecoupledResidualTemporalContextTCN,
    StabilizedResidualTemporalContextTCN,
)


def test_researcher_can_validate_a_bounded_tcn_tuning_plan() -> None:
    valid = TCNTuningTrial(
        trial_id="control",
        channels=8,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        dropout=0.1,
        learning_rate=0.003,
        batch_size=128,
    )
    lite = TCNTuningTrial(
        trial_id="lite",
        channels=8,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.1,
        learning_rate=0.003,
        batch_size=128,
        model_kind="lite",
        head_dropout=0.1,
        dropout_kind="channel",
        weight_decay=0.0001,
    )

    assert validate_tcn_tuning_plan(
        [valid, lite], input_steps=480, max_epochs=8, patience=2, min_delta=0.002
    ) == (valid, lite)

    with pytest.raises(ContractError, match="receptive field"):
        validate_tcn_tuning_plan(
            [
                TCNTuningTrial(
                    trial_id="too-short",
                    channels=8,
                    kernel_size=3,
                    dilations=(1, 2, 4, 8, 16, 32),
                    dropout=0.1,
                    learning_rate=0.003,
                    batch_size=128,
                )
            ],
            input_steps=480,
            max_epochs=8,
            patience=2,
            min_delta=0.002,
        )

    with pytest.raises(ContractError, match="trial IDs must be unique"):
        validate_tcn_tuning_plan(
            [valid, valid],
            input_steps=480,
            max_epochs=8,
            patience=2,
            min_delta=0.002,
        )

    local_pcgrad = TCNTuningTrial(
        trial_id="local-pcgrad",
        channels=8,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=128,
        model_kind="horizon_skip",
        strategy="pcgrad",
        padding_mode="chomp",
        pcgrad_blocks=(4, 6),
        pcgrad_horizons=(1, 5),
    )
    assert validate_tcn_tuning_plan(
        [local_pcgrad],
        input_steps=480,
        max_epochs=8,
        patience=2,
        min_delta=0.002,
    ) == (local_pcgrad,)

    invalid_scope = TCNTuningTrial(
        **{
            **local_pcgrad.__dict__,
            "trial_id": "invalid-scope",
            "pcgrad_blocks": (4, 8),
        }
    )
    with pytest.raises(ContractError, match="PCGrad block scope"):
        validate_tcn_tuning_plan(
            [invalid_scope],
            input_steps=480,
            max_epochs=8,
            patience=2,
            min_delta=0.002,
        )

    temporal_context = TCNTuningTrial(
        trial_id="context-soft-rankic",
        channels=8,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=128,
        model_kind="temporal_context",
        strategy="soft_rankic",
        padding_mode="chomp",
        bars_per_day=48,
        soft_rankic_weight=0.2,
        soft_rank_temperature=0.1,
    )
    assert validate_tcn_tuning_plan(
        [temporal_context],
        input_steps=480,
        max_epochs=8,
        patience=2,
        min_delta=0.002,
    ) == (temporal_context,)


def test_stabilized_residual_optimizer_is_complete_disjoint_and_lower_lr() -> None:
    model = StabilizedResidualTemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
        residual_scale=0.05,
    )
    trial = TCNTuningTrial(
        trial_id="stable-low-lr",
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=8,
        model_kind="stabilized_temporal_context",
        strategy="smooth_l1",
        padding_mode="chomp",
        bars_per_day=4,
        residual_scale=0.05,
        adapter_learning_rate=0.0003,
    )

    bundle = build_tcn_optimizer(model, trial)

    parameter_ids = [
        id(parameter)
        for group in bundle.optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(parameter) for parameter in model.parameters()}
    assert {float(group["lr"]) for group in bundle.optimizer.param_groups} == {
        0.003,
        0.0003,
    }
    assert bundle.adapter_parameter_count == sum(
        parameter.numel() for parameter in model.temporal_adapter_parameters()
    )
    assert bundle.parameter_group_identity == "base-lr-0.003+adapter-lr-0.0003"

    invalid = TCNTuningTrial(
        **{
            **trial.__dict__,
            "trial_id": "invalid-residual",
            "residual_scale": 0.0,
        }
    )
    with pytest.raises(ContractError, match="residual scale"):
        validate_tcn_tuning_plan(
            [invalid],
            input_steps=16,
            max_epochs=2,
            patience=1,
            min_delta=0.0,
        )

    wrong_model = TCNTuningTrial(
        **{
            **trial.__dict__,
            "trial_id": "wrong-model-adapter-lr",
            "model_kind": "temporal_context",
        }
    )
    with pytest.raises(ContractError, match="adapter learning rate"):
        validate_tcn_tuning_plan(
            [wrong_model],
            input_steps=16,
            max_epochs=2,
            patience=1,
            min_delta=0.0,
        )


def test_decoupled_optimizer_keeps_base_at_full_lr_and_only_residual_low() -> None:
    model = DecoupledResidualTemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
        residual_scale=0.05,
    )
    trial = TCNTuningTrial(
        trial_id="decoupled-mid-lr",
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        residual_learning_rate=0.001,
        residual_scale=0.05,
        batch_size=8,
        model_kind="decoupled_temporal_context",
        strategy="smooth_l1",
        padding_mode="chomp",
        bars_per_day=4,
    )

    bundle = build_tcn_optimizer(model, trial)

    groups = {
        str(group["group_name"]): group for group in bundle.optimizer.param_groups
    }
    assert set(groups) == {"base", "signed_residual"}
    assert float(groups["base"]["lr"]) == pytest.approx(0.003)
    assert float(groups["signed_residual"]["lr"]) == pytest.approx(0.001)
    base_ids = {id(parameter) for parameter in groups["base"]["params"]}
    residual_ids = {
        id(parameter) for parameter in groups["signed_residual"]["params"]
    }
    assert {id(parameter) for parameter in model.temporal_adapter_parameters()} <= (
        base_ids
    )
    assert residual_ids == {
        id(parameter) for parameter in model.residual_adapter_parameters()
    }
    assert base_ids.isdisjoint(residual_ids)
    assert base_ids | residual_ids == {
        id(parameter) for parameter in model.parameters()
    }
    assert bundle.residual_parameter_count == 32
    assert bundle.parameter_group_identity == "base-lr-0.003+residual-lr-0.001"


def test_validation_rankic_is_computed_independently_by_date_and_horizon() -> None:
    index = pd.DataFrame(
        {
            "sample_position": range(6),
            "sample_id": [f"s{value}" for value in range(6)],
            "signal_date": ["2025-01-02"] * 3 + ["2025-01-03"] * 3,
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "signal_date": "2025-01-02" if sample < 3 else "2025-01-03",
                "horizon": horizon,
                "rank_target": [-1.0, 0.0, 1.0][sample % 3],
                "valid": True,
            }
            for sample in range(6)
            for horizon in [1, 2]
        ]
    )
    scores = np.zeros((6, 4), dtype="float32")
    scores[:, 1] = np.tile([-1.0, 0.0, 1.0], 2)
    scores[:3, 0] = [-1.0, 0.0, 1.0]
    scores[3:, 0] = [1.0, 0.0, -1.0]

    result = cross_sectional_validation_rankic(
        scores, np.arange(6), index, labels
    )

    assert result.mean_daily_rankic == pytest.approx(0.5)
    assert result.rankic_by_horizon == {1: pytest.approx(0.0), 2: pytest.approx(1.0)}
    assert result.valid_group_count == 4


def test_cached_validation_rankic_plan_is_position_safe_and_numerically_equivalent() -> None:
    index = pd.DataFrame(
        {
            "sample_position": [10, 11, 12, 13, 14, 15],
            "sample_id": [f"s{value}" for value in range(6)],
            "signal_date": ["2025-01-02"] * 3 + ["2025-01-03"] * 3,
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "signal_date": "2025-01-02" if sample < 3 else "2025-01-03",
                "horizon": horizon,
                "rank_target": [-1.0, 0.0, 1.0][sample % 3],
                "valid": not (sample == 5 and horizon == 5),
            }
            for sample in range(6)
            for horizon in [1, 2, 3, 5]
        ]
    )
    positions = np.array([12, 10, 11, 15, 13, 14], dtype="int64")
    scores = np.array(
        [
            [1.0, 0.0, 2.0, 2.0],
            [-1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [-1.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
        ],
        dtype="float32",
    )

    uncached = cross_sectional_validation_rankic(scores, positions, index, labels)
    plan = build_validation_rankic_plan(positions, index, labels)
    cached = plan.evaluate(scores, positions)

    assert cached.mean_daily_rankic == pytest.approx(uncached.mean_daily_rankic)
    assert cached.rankic_by_horizon == pytest.approx(uncached.rankic_by_horizon)
    assert cached.valid_group_count == uncached.valid_group_count
    with pytest.raises(ContractError, match="positions do not match"):
        plan.evaluate(scores, positions[::-1])


def test_candidate_selection_applies_deterministic_top50_gate() -> None:
    leaderboard = pd.DataFrame(
        [
            ("control", 0, 0.02, 1000, 100.0, 0.02, 0.02, 0.02, 0.02),
            ("control", 1, 0.03, 1000, 100.0, 0.03, 0.03, 0.03, 0.03),
            ("candidate", 0, 0.04, 800, 90.0, 0.04, 0.04, 0.01, 0.01),
            ("candidate", 1, 0.05, 800, 90.0, 0.05, 0.05, 0.02, 0.02),
            ("larger-tie", 0, 0.04, 1200, 120.0, 0.04, 0.04, 0.01, 0.01),
            ("larger-tie", 1, 0.05, 1200, 120.0, 0.05, 0.05, 0.02, 0.02),
        ],
        columns=[
            "trial_id",
            "fold",
            "best_mean_daily_rankic",
            "parameter_count",
            "samples_per_second",
            "rankic_1d",
            "rankic_2d",
            "rankic_3d",
            "rankic_5d",
        ],
    )

    decision = select_tcn_candidate(
        leaderboard, control_trial_id="control", min_improvement=0.01
    )

    assert decision.selected_trial_id == "candidate"
    assert decision.status == "expand_top50"
    assert decision.mean_improvement == pytest.approx(0.02)
    assert decision.non_degrading_horizon_count == 2

    failed = leaderboard.copy()
    failed.loc[
        (failed["trial_id"] == "candidate") & (failed["fold"] == 1),
        "best_mean_daily_rankic",
    ] = -0.01
    failed.loc[
        (failed["trial_id"] == "larger-tie") & (failed["fold"] == 1),
        "best_mean_daily_rankic",
    ] = -0.01
    stopped = select_tcn_candidate(
        failed, control_trial_id="control", min_improvement=0.01
    )
    assert stopped.status == "stop_no_validation_gain"


def test_tcn_sweep_uses_only_train_and_validation_rows() -> None:
    rng = np.random.default_rng(21)
    features = rng.normal(size=(24, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(24),
            "sample_id": [f"s{value}" for value in range(24)],
            "signal_date": [f"2025-01-{2 + value // 6:02d}" for value in range(24)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "signal_date": index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": float((sample % 6) / 5 * 2 - 1),
                "valid": True,
            }
            for sample in range(24)
            for horizon in [1, 2, 3, 5]
        ]
    )
    split = index[["sample_position"]].copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 12 + ["validation"] * 6 + ["test"] * 6
    trial = TCNTuningTrial(
        trial_id="tiny",
        channels=3,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=6,
    )
    twin = TCNTuningTrial(
        trial_id="tiny-twin",
        channels=3,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=6,
    )

    initial_threads = torch.get_num_threads()
    first = run_tcn_validation_sweep(
        features,
        index,
        labels,
        split,
        trials=[trial, twin],
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0,
        torch_threads=1,
    )
    changed_test = features.copy()
    changed_test[18:] = 1_000_000.0
    second = run_tcn_validation_sweep(
        changed_test,
        index,
        labels,
        split,
        trials=[trial, twin],
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0,
        torch_threads=1,
    )

    stable_columns = [
        "trial_id",
        "fold",
        "best_epoch",
        "best_mean_daily_rankic",
        "rankic_1d",
        "rankic_2d",
        "rankic_3d",
        "rankic_5d",
        "parameter_count",
        "stopping_reason",
    ]
    pd.testing.assert_frame_equal(
        first.leaderboard[stable_columns], second.leaderboard[stable_columns]
    )
    assert set(first.epoch_history["stage"]) == {"validation"}
    assert set(first.epoch_history["seed"]) == {7}
    assert first.epoch_history["model_seed"].notna().all()
    assert first.leaderboard["seed"].nunique() == 1
    assert first.leaderboard["best_mean_daily_rankic"].nunique() == 1
    assert set(first.leaderboard["torch_threads"]) == {1}
    assert torch.get_num_threads() == initial_threads


def test_temporal_context_soft_rankic_runs_a_tiny_validation_sweep() -> None:
    rng = np.random.default_rng(33)
    features = rng.normal(size=(24, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(24),
            "sample_id": [f"context-{value}" for value in range(24)],
            "signal_date": [f"2025-02-{2 + value // 6:02d}" for value in range(24)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"context-{sample}",
                "signal_date": index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": float((sample % 6) / 5 * 2 - 1),
                "valid": True,
            }
            for sample in range(24)
            for horizon in [1, 2, 3, 5]
        ]
    )
    split = index[["sample_position"]].copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 12 + ["validation"] * 6 + ["test"] * 6
    trial = TCNTuningTrial(
        trial_id="tiny-context-soft-rankic",
        channels=3,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=6,
        model_kind="temporal_context",
        strategy="soft_rankic",
        padding_mode="chomp",
        bars_per_day=4,
        soft_rankic_weight=0.2,
        soft_rank_temperature=0.1,
    )
    signed_trial = TCNTuningTrial(
        trial_id="tiny-signed-context",
        channels=3,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=6,
        model_kind="signed_temporal_context",
        strategy="smooth_l1",
        padding_mode="chomp",
        bars_per_day=4,
    )
    stabilized_trial = TCNTuningTrial(
        trial_id="tiny-stabilized-context",
        channels=3,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        adapter_learning_rate=0.0003,
        residual_scale=0.05,
        batch_size=6,
        model_kind="stabilized_temporal_context",
        strategy="smooth_l1",
        padding_mode="chomp",
        bars_per_day=4,
    )
    decoupled_trial = TCNTuningTrial(
        trial_id="tiny-decoupled-context",
        channels=3,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        learning_rate=0.003,
        residual_learning_rate=0.001,
        residual_scale=0.05,
        batch_size=6,
        model_kind="decoupled_temporal_context",
        strategy="smooth_l1",
        padding_mode="chomp",
        bars_per_day=4,
    )

    result = run_tcn_validation_sweep(
        features,
        index,
        labels,
        split,
        trials=[trial, signed_trial, stabilized_trial, decoupled_trial],
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0,
        torch_threads=1,
    )

    row = result.leaderboard.loc[
        result.leaderboard["trial_id"].eq("tiny-context-soft-rankic")
    ].iloc[0]
    assert row["model_kind"] == "temporal_context"
    assert row["loss_identity"] == "smooth-l1+0.2-soft-rankic-tau-0.1"
    assert row["batching_identity"] == "date-grouped"
    assert row["readout_identity"] == "horizon_dual_scale_full_sequence"
    assert isinstance(row["day_weights"], str)
    assert isinstance(row["intraday_weights"], str)
    assert np.isfinite(float(row["best_mean_daily_rankic"]))
    signed_row = result.leaderboard.loc[
        result.leaderboard["trial_id"].eq("tiny-signed-context")
    ].iloc[0]
    assert signed_row["model_kind"] == "signed_temporal_context"
    assert signed_row["batching_identity"] == "seeded-random"
    assert signed_row["readout_identity"] == "horizon_dual_scale_signed_adapter"
    assert isinstance(signed_row["day_negative_weight_count"], str)
    assert isinstance(signed_row["intraday_negative_weight_count"], str)
    stabilized_row = result.leaderboard.loc[
        result.leaderboard["trial_id"].eq("tiny-stabilized-context")
    ].iloc[0]
    assert stabilized_row["readout_identity"] == (
        "horizon_dual_scale_stabilized_signed_residual"
    )
    assert stabilized_row["adapter_learning_rate"] == pytest.approx(0.0003)
    assert stabilized_row["residual_scale"] == pytest.approx(0.05)
    assert stabilized_row["optimizer_group_identity"] == (
        "base-lr-0.003+adapter-lr-0.0003"
    )
    assert stabilized_row["adapter_parameter_count"] == 32
    assert isinstance(stabilized_row["day_residual_l2"], str)
    assert isinstance(stabilized_row["intraday_residual_l2"], str)
    decoupled_row = result.leaderboard.loc[
        result.leaderboard["trial_id"].eq("tiny-decoupled-context")
    ].iloc[0]
    assert decoupled_row["readout_identity"] == (
        "horizon_dual_scale_decoupled_signed_residual"
    )
    assert decoupled_row["residual_learning_rate"] == pytest.approx(0.001)
    assert decoupled_row["optimizer_group_identity"] == (
        "base-lr-0.003+residual-lr-0.001"
    )
    assert decoupled_row["optimizer_residual_parameter_count"] == 32
    assert decoupled_row["base_temporal_parameter_count"] == 32
    assert decoupled_row["residual_parameter_count"] == 32
    assert isinstance(decoupled_row["day_simplex_weights"], str)
    assert isinstance(decoupled_row["intraday_simplex_weights"], str)
    assert set(result.best_states) == {
        "tiny-context-soft-rankic-fold-0",
        "tiny-signed-context-fold-0",
        "tiny-stabilized-context-fold-0",
        "tiny-decoupled-context-fold-0",
    }


def test_tuning_task_writes_an_immutable_validation_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    np.save(
        run_dir / "feature-windows.npy",
        np.random.default_rng(3).normal(size=(18, 3, 16)).astype("float32"),
    )
    index = pd.DataFrame(
        {
            "sample_position": range(18),
            "sample_id": [f"s{value}" for value in range(18)],
            "signal_date": ["2025-01-02"] * 6
            + ["2025-01-03"] * 6
            + ["2025-01-04"] * 6,
        }
    )
    index.to_parquet(run_dir / "window-index.parquet", index=False)
    pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "signal_date": index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": float((sample % 6) / 5 * 2 - 1),
                "valid": True,
            }
            for sample in range(18)
            for horizon in [1, 2, 3, 5]
        ]
    ).to_parquet(run_dir / "labels.parquet", index=False)
    split = index[["sample_position"]].copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 12 + ["validation"] * 6
    split.to_parquet(run_dir / "split-manifest.parquet", index=False)
    validation_manifest = tmp_path / "validation-manifest.parquet"
    split.to_parquet(validation_manifest, index=False)
    config = {
        "seed": 7,
        "torch_threads": 1,
        "folds": [0],
        "max_epochs": 2,
        "patience": 1,
        "min_delta": 0.0,
        "control_trial_id": "tiny",
        "apply_scale_gate": False,
        "trials": [
            {
                "trial_id": "tiny",
                "channels": 3,
                "kernel_size": 2,
                "dilations": [1, 2, 4, 8],
                "dropout": 0.1,
                "learning_rate": 0.003,
                "batch_size": 6,
                "model_kind": "lite",
                "head_dropout": 0.1,
                "dropout_kind": "channel",
                "weight_decay": 0.0001,
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "tuning"

    completed = subprocess.run(
        [
            sys.executable,
            "tasks/tune_tcn_validation.py",
            "--run-dir",
            str(run_dir),
            "--config",
            str(config_path),
            "--split-manifest",
            str(validation_manifest),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "success"
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["selection"]["status"] == "screen_complete"
    assert receipt["source_artifacts"]["features"]["sha256"] == hashlib.sha256(
        (run_dir / "feature-windows.npy").read_bytes()
    ).hexdigest()
    assert receipt["source_artifacts"]["split_manifest"]["path"] == str(
        validation_manifest.resolve()
    )
    assert pd.read_parquet(output_dir / "leaderboard.parquet")["trial_id"].tolist() == [
        "tiny"
    ]
    assert pd.read_parquet(output_dir / "leaderboard.parquet")[
        "torch_threads"
    ].tolist() == [1]
    assert pd.read_parquet(output_dir / "leaderboard.parquet")[
        "model_kind"
    ].tolist() == ["lite"]
    recorded = pd.read_parquet(output_dir / "leaderboard.parquet").iloc[0]
    assert recorded["block_dropout"] == pytest.approx(0.1)
    assert recorded["head_dropout"] == pytest.approx(0.1)
    assert recorded["dropout_kind"] == "channel"
    assert recorded["optimizer"] == "AdamW"
    assert recorded["weight_decay"] == pytest.approx(0.0001)

    refused = subprocess.run(
        [
            sys.executable,
            "tasks/tune_tcn_validation.py",
            "--run-dir",
            str(run_dir),
            "--config",
            str(config_path),
            "--split-manifest",
            str(validation_manifest),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 2
    assert "refuses to overwrite" in json.loads(refused.stdout)["error"]
