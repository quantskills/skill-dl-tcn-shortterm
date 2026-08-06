from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.v9_confirmation import V9FinalContext
from skill_dl_tcn_shortterm.v9_protocol import V9Plan, V9TrialSpec
from skill_dl_tcn_shortterm.v9_runner import (
    _publish_checkpoint_bundle,
    _publish_seed7_receipt,
    _load_seed7_receipt,
    _validate_training_contract,
    publish_v9_upstream_receipt,
    run_v9_evidence_path,
)
from skill_dl_tcn_shortterm.v9_training import V9TrainingRequest
import skill_dl_tcn_shortterm.v9_runner as v9_runner_module


def _governance_identities() -> dict[str, str]:
    return {
        "lite_config": "f" * 64,
        "bai_config": "2" * 64,
        "rankic_predictions": "0" * 64,
        "rankic_candidate_config": "4" * 64,
        **{
            f"lite_checkpoint_fold_{fold}": "1" * 64 for fold in range(5)
        },
        **{
            f"bai_checkpoint_fold_{fold}": "3" * 64 for fold in range(5)
        },
        **{
            f"rankic_candidate_checkpoint_fold_{fold}": "5" * 64
            for fold in range(5)
        },
    }


def test_highest_v9_seam_requires_executable_diagnostic_inputs(
    tmp_path: Path,
) -> None:
    plan = V9Plan(
        run_id="synthetic-no-trigger",
        stage="diagnostic",
        seeds=(7,),
        fold_ids=(0, 1, 2, 3, 4),
        torch_threads=4,
        precision="float32",
        model_family="tcn",
        input_steps=480,
        max_epochs=8,
        patience=2,
        min_delta=0.002,
        control_trial_id="lite-c16-no-dropout",
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=128,
        trials=(
            V9TrialSpec("horizon_skip", "v9b-horizon-skip", "pending"),
            V9TrialSpec("rank_objective", "v9c-rank-objective", "pending"),
            V9TrialSpec("pcgrad", "v9d-pcgrad", "pending"),
        ),
        candidate_order=(
            "v9b-horizon-skip",
            "v9c-rank-objective",
            "v9d-pcgrad",
        ),
        source_identities={
            **_governance_identities(),
            "data": "a" * 64,
            "fold_manifest": "b" * 64,
            "window_index": "c" * 64,
            "labels": "d" * 64,
            "evaluation": "e" * 64,
            "rankic_predictions": "0" * 64,
            "lite_config": "f" * 64,
            "bai_config": "2" * 64,
            **{
                f"lite_checkpoint_fold_{fold}": "1" * 64
                for fold in range(5)
            },
            **{
                f"bai_checkpoint_fold_{fold}": "3" * 64
                for fold in range(5)
            },
        },
    )
    split = pd.DataFrame(
        [
            {
                "sample_position": position,
                "fold": fold,
                "stage": "train" if position == 0 else "validation",
                "sealed": False,
            }
            for fold in range(5)
            for position in range(2)
        ]
    )
    upstream = {
        "horizon_skip": {"status": "horizon_skip_not_applicable", "sealed_test_accessed": False},
        "rank_objective": {"status": "rank_objective_not_resolvable", "sealed_test_accessed": False},
        "pcgrad": {"status": "pcgrad_not_applicable", "sealed_test_accessed": False},
        "infra": {"status": "infra_optimization_not_applicable", "sealed_test_accessed": False},
    }
    context = V9FinalContext(
        resolved_config={"protocol": "tcn-v9", "stage": "diagnostic"},
        source_identities=plan.source_identities,
        checkpoint_identities={"controls": "c" * 64},
        environment={"hardware": "synthetic-cpu"},
        upstream_receipts={"diagnostics": "d" * 64},
    )

    with pytest.raises(ContractError, match="executable raw inputs"):
        run_v9_evidence_path(
            plan,
            split,
            upstream_receipts=upstream,
            seed7_leaderboard=pd.DataFrame(),
            confirmation_measurements=pd.DataFrame(),
            context=context,
            output_root=tmp_path / "run",
            project_root=Path(__file__).resolve().parents[1],
        )


def test_applicable_upstream_status_requires_an_immutable_receipt(
    tmp_path: Path,
) -> None:
    plan = V9Plan(
        run_id="synthetic-forged-trigger",
        stage="formal_screen",
        seeds=(7,),
        fold_ids=(0, 1, 2, 3, 4),
        torch_threads=4,
        precision="float32",
        model_family="tcn",
        input_steps=480,
        max_epochs=8,
        patience=2,
        min_delta=0.002,
        control_trial_id="lite-c16-no-dropout",
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=128,
        trials=(
            V9TrialSpec("horizon_skip", "v9b-horizon-skip", "applicable"),
            V9TrialSpec("rank_objective", "v9c-rank-objective", "not_applicable"),
            V9TrialSpec("pcgrad", "v9d-pcgrad", "not_applicable"),
        ),
        candidate_order=(
            "v9b-horizon-skip",
            "v9c-rank-objective",
            "v9d-pcgrad",
        ),
        source_identities={
            **_governance_identities(),
            "data": "a" * 64,
            "fold_manifest": "b" * 64,
            "window_index": "c" * 64,
            "labels": "d" * 64,
            "evaluation": "e" * 64,
        },
    )
    split = pd.DataFrame(
        [
            {
                "sample_position": position,
                "fold": fold,
                "stage": "train" if position == 0 else "validation",
                "sealed": False,
            }
            for fold in range(5)
            for position in range(2)
        ]
    )
    upstream = {
        "horizon_skip": {
            "status": "horizon_skip_applicable",
            "sealed_test_accessed": False,
        },
        "rank_objective": {"status": "rank_objective_not_resolvable", "sealed_test_accessed": False},
        "pcgrad": {"status": "pcgrad_not_applicable", "sealed_test_accessed": False},
        "infra": {"status": "infra_optimization_not_applicable", "sealed_test_accessed": False},
    }
    context = V9FinalContext(
        resolved_config={"protocol": "tcn-v9"},
        source_identities=plan.source_identities,
        checkpoint_identities={"controls": "c" * 64},
        environment={"hardware": "synthetic-cpu"},
        upstream_receipts={"diagnostics": "d" * 64},
    )

    with pytest.raises(ContractError, match="immutable receipt path"):
        run_v9_evidence_path(
            plan,
            split,
            upstream_receipts=upstream,
            seed7_leaderboard=pd.DataFrame(),
            confirmation_measurements=pd.DataFrame(),
            context=context,
            output_root=tmp_path / "run",
            project_root=Path(__file__).resolve().parents[1],
        )
    with pytest.raises(ContractError, match="incomplete"):
        publish_v9_upstream_receipt(
            "horizon_skip",
            "horizon_skip_applicable",
            {},
            plan.source_identities,
            output_dir=tmp_path / "forged",
            project_root=Path(__file__).resolve().parents[1],
        )


def test_formal_training_contract_binds_memmap_and_split_artifact_bytes(
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.zeros((2, 3, 480), dtype="float32"))
    features = np.load(feature_path, mmap_mode="r")
    split = pd.DataFrame(
        [
            {
                "sample_position": position,
                "fold": fold,
                "stage": "train" if position == 0 else "validation",
                "sealed": False,
            }
            for fold in range(5)
            for position in range(2)
        ]
    )
    split_path = tmp_path / "split.csv"
    split.to_csv(split_path, index=False)
    window_index = pd.DataFrame(
        {
            "sample_position": [0, 1],
            "sample_id": ["s0", "s1"],
            "signal_date": ["2025-01-01", "2025-01-02"],
        }
    )
    window_path = tmp_path / "window-index.csv"
    window_index.to_csv(window_path, index=False)
    labels = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "signal_date": ["2025-01-01", "2025-01-02"],
            "horizon": [1, 1],
            "rank_target": [-1.0, 1.0],
            "valid": [True, True],
        }
    )
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text('{"metric":"daily-rankic"}', encoding="utf-8")
    identities = {
        **_governance_identities(),
        "data": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "fold_manifest": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "window_index": hashlib.sha256(window_path.read_bytes()).hexdigest(),
        "labels": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "evaluation": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
    }
    plan = V9Plan(
        run_id="formal-contract",
        stage="formal_screen",
        seeds=(7,),
        fold_ids=(0, 1, 2, 3, 4),
        torch_threads=1,
        precision="float32",
        model_family="tcn",
        input_steps=480,
        max_epochs=8,
        patience=2,
        min_delta=0.002,
        control_trial_id="lite-c16-no-dropout",
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=128,
        trials=(
            V9TrialSpec("horizon_skip", "v9b-horizon-skip", "applicable"),
            V9TrialSpec("rank_objective", "v9c-rank-objective", "not_applicable"),
            V9TrialSpec("pcgrad", "v9d-pcgrad", "not_applicable"),
        ),
        candidate_order=("v9b-horizon-skip", "v9c-rank-objective", "v9d-pcgrad"),
        source_identities=identities,
    )
    request = V9TrainingRequest(
        features=features,
        window_index=window_index,
        labels=labels,
        split_manifest=split,
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=128,
        max_epochs=8,
        patience=2,
        min_delta=0.002,
        torch_threads=1,
        protocol_identities=identities,
        artifact_paths={
            "data": feature_path,
            "fold_manifest": split_path,
            "window_index": window_path,
            "labels": labels_path,
            "evaluation": evaluation_path,
        },
    )

    _validate_training_contract(plan, request, {"v9b-horizon-skip"}, split)
    drifted = split.copy()
    drifted.loc[0, "stage"] = "validation"
    with pytest.raises(ContractError, match="frames differ"):
        _validate_training_contract(
            plan,
            request,
            {"v9b-horizon-skip"},
            drifted,
        )


def test_seed7_receipt_persists_and_fingerprints_checkpoint_states(
    tmp_path: Path,
) -> None:
    leaderboard = pd.DataFrame(
        [{"trial_id": "candidate", "fold": 0, "seed": 7}]
    )
    receipt = _publish_seed7_receipt(
        leaderboard,
        {"candidate-fold-0": {"weight": torch.ones(2)}},
        status="stop_no_pareto_gain_v9",
        winner_trial_id=None,
        source_identities={"data": "a" * 64},
        output_dir=tmp_path / "seed7",
        project_root=Path(__file__).resolve().parents[1],
    )

    assert receipt.is_file()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    checkpoint_receipt = Path(payload["checkpoint_bundle"]["path"])
    assert checkpoint_receipt.is_file()


def test_seed7_negative_receipt_does_not_invent_a_checkpoint_bundle(
    tmp_path: Path,
) -> None:
    receipt = _publish_seed7_receipt(
        pd.DataFrame(),
        {},
        status="stop_no_pareto_gain_v9",
        winner_trial_id=None,
        source_identities={"data": "a" * 64},
        output_dir=tmp_path / "seed7-negative",
        project_root=Path(__file__).resolve().parents[1],
    )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "stop_no_pareto_gain_v9"
    assert "checkpoint_bundle" not in payload


def test_existing_checkpoint_bundle_rejects_new_state_content(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "checkpoints"
    arguments = {
        "source_identities": {"data": "a" * 64},
        "output_dir": output_dir,
        "project_root": Path(__file__).resolve().parents[1],
        "stage": "seed7",
    }
    _publish_checkpoint_bundle(
        {"candidate-fold-0": {"weight": torch.ones(2)}},
        **arguments,  # type: ignore[arg-type]
    )

    with pytest.raises(ContractError, match="content drifted"):
        _publish_checkpoint_bundle(
            {"candidate-fold-0": {"weight": torch.zeros(2)}},
            **arguments,  # type: ignore[arg-type]
        )


def _stopped_plan(stage: str) -> tuple[V9Plan, pd.DataFrame]:
    identities = {
        **_governance_identities(),
        "data": "a" * 64,
        "fold_manifest": "b" * 64,
        "window_index": "c" * 64,
        "labels": "d" * 64,
        "evaluation": "e" * 64,
    }
    plan = V9Plan(
        run_id=f"stopped-{stage}",
        stage=stage,  # type: ignore[arg-type]
        seeds=(7,) if stage == "formal_screen" else (17, 27),
        fold_ids=(0, 1, 2, 3, 4),
        torch_threads=1,
        precision="float32",
        model_family="tcn",
        input_steps=480,
        max_epochs=8,
        patience=2,
        min_delta=0.002,
        control_trial_id="lite-c16-no-dropout",
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
        learning_rate=0.003,
        batch_size=128,
        trials=(
            V9TrialSpec("horizon_skip", "v9b-horizon-skip", "not_applicable"),
            V9TrialSpec("rank_objective", "v9c-rank-objective", "not_applicable"),
            V9TrialSpec("pcgrad", "v9d-pcgrad", "not_applicable"),
        ),
        candidate_order=(
            "v9b-horizon-skip",
            "v9c-rank-objective",
            "v9d-pcgrad",
        ),
        source_identities=identities,
    )
    split = pd.DataFrame(
        [
            {
                "sample_position": position,
                "fold": fold,
                "stage": "train" if position == 0 else "validation",
                "sealed": False,
            }
            for fold in range(5)
            for position in range(2)
        ]
    )
    return plan, split


def _not_applicable_receipts(
    plan: V9Plan,
    tmp_path: Path,
) -> dict[str, dict[str, object]]:
    statuses = {
        "horizon_skip": "horizon_skip_not_applicable",
        "rank_objective": "rank_objective_not_applicable",
        "pcgrad": "pcgrad_not_applicable",
        "infra": "infra_optimization_not_applicable",
    }
    return {
        name: {
            "status": status,
            "receipt_path": publish_v9_upstream_receipt(
                name,
                status,
                {},
                plan.source_identities,
                output_dir=tmp_path / "upstream" / name,
                project_root=Path(__file__).resolve().parents[1],
            ),
        }
        for name, status in statuses.items()
    }


def test_seed7_loader_recomputes_nested_checkpoint_receipt_identity(
    tmp_path: Path,
) -> None:
    plan, _ = _stopped_plan("formal_screen")
    seed7_receipt = _publish_seed7_receipt(
        pd.DataFrame(),
        {"candidate-fold-0": {"weight": torch.ones(2)}},
        status="stop_no_pareto_gain_v9",
        winner_trial_id=None,
        source_identities=plan.source_identities,
        output_dir=tmp_path / "nested-seed7",
        project_root=Path(__file__).resolve().parents[1],
    )
    outer = json.loads(seed7_receipt.read_text(encoding="utf-8"))
    nested_path = Path(outer["checkpoint_bundle"]["path"])
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    artifact_path = nested_path.parent / nested["checkpoints"][0]["path"]
    torch.save({"weight": torch.zeros(2)}, artifact_path)
    nested["checkpoints"][0]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    nested_path.write_text(json.dumps(nested), encoding="utf-8")

    with pytest.raises(ContractError, match="checkpoint receipt identity drifted"):
        _load_seed7_receipt(
            seed7_receipt,
            plan=plan,
            project_root=Path(__file__).resolve().parents[1],
        )


def test_formal_screen_with_no_triggered_trial_publishes_a_valid_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, split = _stopped_plan("formal_screen")
    monkeypatch.setattr(v9_runner_module, "_validate_training_contract", lambda *args, **kwargs: None)
    context = V9FinalContext({}, plan.source_identities, {}, {"hardware": "cpu"}, {})

    result = run_v9_evidence_path(
        plan,
        split,
        upstream_receipts=_not_applicable_receipts(plan, tmp_path),
        seed7_leaderboard=pd.DataFrame(),
        confirmation_measurements=pd.DataFrame(),
        context=context,
        output_root=tmp_path / "formal",
        project_root=Path(__file__).resolve().parents[1],
        training_request=object(),  # type: ignore[arg-type]
    )

    assert result.status == "stop_no_pareto_gain_v9"
    payload = json.loads(result.final_receipt.read_text(encoding="utf-8"))
    assert "checkpoint_bundle" not in payload


def test_confirmation_stops_before_reference_loading_when_seed7_not_admitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen_plan, _ = _stopped_plan("formal_screen")
    seed7_receipt = _publish_seed7_receipt(
        pd.DataFrame(),
        {},
        status="stop_no_pareto_gain_v9",
        winner_trial_id=None,
        source_identities=screen_plan.source_identities,
        output_dir=tmp_path / "seed7",
        project_root=Path(__file__).resolve().parents[1],
    )
    plan, split = _stopped_plan("multi_seed_confirmation")
    monkeypatch.setattr(v9_runner_module, "_validate_training_contract", lambda *args, **kwargs: None)
    context = V9FinalContext({}, plan.source_identities, {}, {"hardware": "cpu"}, {})

    result = run_v9_evidence_path(
        plan,
        split,
        upstream_receipts=_not_applicable_receipts(plan, tmp_path),
        seed7_leaderboard=pd.DataFrame(),
        confirmation_measurements=pd.DataFrame(),
        context=context,
        output_root=tmp_path / "confirmation",
        project_root=Path(__file__).resolve().parents[1],
        training_request=object(),  # type: ignore[arg-type]
        prior_seed7_receipt=seed7_receipt,
    )

    assert result.status == "stop_no_pareto_gain_v9"
    payload = json.loads(result.final_receipt.read_text(encoding="utf-8"))
    assert payload["resolved_config"]["run_id"] == plan.run_id
    assert payload["identities"]["sources"] == plan.source_identities
    assert set(payload["identities"]["upstream_receipts"]) == {
        "plan",
        "seed7",
        "diagnostic:horizon_skip",
        "diagnostic:rank_objective",
        "diagnostic:pcgrad",
        "diagnostic:infra",
    }
