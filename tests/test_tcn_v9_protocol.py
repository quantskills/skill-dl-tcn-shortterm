from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.v9_protocol import (
    V9Plan,
    V9TrialSpec,
    execute_v9_plan,
    validate_v9_plan,
)


def _plan(**overrides: object) -> V9Plan:
    values: dict[str, object] = {
        "run_id": "synthetic-v9-diagnostic",
        "stage": "diagnostic",
        "seeds": (7,),
        "fold_ids": (0, 1, 2, 3, 4),
        "torch_threads": 4,
        "precision": "float32",
        "model_family": "tcn",
        "input_steps": 480,
        "max_epochs": 8,
        "patience": 2,
        "min_delta": 0.002,
        "control_trial_id": "lite-c16-no-dropout",
        "channels": 16,
        "kernel_size": 3,
        "dilations": (1, 2, 4, 8, 16, 32, 64, 128),
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 128,
        "trials": (
            V9TrialSpec("horizon_skip", "v9b-horizon-skip", "pending"),
            V9TrialSpec("rank_objective", "v9c-rank-objective", "pending"),
            V9TrialSpec("pcgrad", "v9d-pcgrad", "pending"),
        ),
        "candidate_order": (
            "v9b-horizon-skip",
            "v9c-rank-objective",
            "v9d-pcgrad",
        ),
        "source_identities": {
            "data": "a" * 64,
            "fold_manifest": "b" * 64,
            "window_index": "e" * 64,
            "labels": "f" * 64,
            "evaluation": "1" * 64,
            "rankic_predictions": "0" * 64,
            "rankic_candidate_config": "1" * 64,
            "lite_config": "2" * 64,
            "bai_config": "3" * 64,
            **{
                f"lite_checkpoint_fold_{fold}": f"{fold + 4:x}" * 64
                for fold in range(5)
            },
            **{
                f"bai_checkpoint_fold_{fold}": f"{fold + 9:x}" * 64
                for fold in range(5)
            },
            **{
                f"rankic_candidate_checkpoint_fold_{fold}": f"{fold + 1:x}" * 64
                for fold in range(5)
            },
        },
    }
    values.update(overrides)
    return V9Plan(**values)  # type: ignore[arg-type]


def _ordinary_split() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_position": position,
                "fold": fold,
                "stage": "train" if position < 2 else "validation",
                "sealed": False,
            }
            for fold in range(5)
            for position in range(4)
        ]
    )


def test_v9_plan_is_bounded_and_rejects_unregistered_or_sealed_stages() -> None:
    plan = _plan()
    assert validate_v9_plan(plan, _ordinary_split()) == plan

    with pytest.raises(ContractError, match="unsupported v9 stage"):
        validate_v9_plan(replace(plan, stage="explore"), _ordinary_split())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="at most three"):
        validate_v9_plan(
            replace(
                plan,
                trials=plan.trials
                + (V9TrialSpec("horizon_skip", "replacement", "pending"),),
                candidate_order=plan.candidate_order + ("replacement",),
            ),
            _ordinary_split(),
        )
    with pytest.raises(ContractError, match="control model"):
        validate_v9_plan(replace(plan, channels=12), _ordinary_split())
    with pytest.raises(ContractError, match="resolved triggers"):
        validate_v9_plan(
            replace(plan, stage="formal_screen"),
            _ordinary_split(),
        )
    test_split = _ordinary_split().copy()
    test_split.loc[3, "stage"] = "test"
    with pytest.raises(ContractError, match="ordinary train/validation"):
        validate_v9_plan(plan, test_split)
    sealed_split = _ordinary_split().copy()
    sealed_split.loc[3, "sealed"] = True
    with pytest.raises(ContractError, match="sealed"):
        validate_v9_plan(plan, sealed_split)
    undeclared = _ordinary_split().drop(columns="stage")
    with pytest.raises(ContractError, match="missing columns"):
        validate_v9_plan(plan, undeclared)

    missing_fold_checkpoint = dict(plan.source_identities)
    missing_fold_checkpoint.pop("lite_checkpoint_fold_4")
    with pytest.raises(ContractError, match="missing frozen input identities"):
        validate_v9_plan(
            replace(plan, source_identities=missing_fold_checkpoint),
            _ordinary_split(),
        )


def test_v9_public_seam_writes_a_deterministic_immutable_plan_receipt(
    tmp_path: Path,
) -> None:
    plan = _plan()
    output_dir = tmp_path / plan.run_id

    receipt_path = execute_v9_plan(
        plan,
        _ordinary_split(),
        output_dir=output_dir,
        project_root=Path(__file__).resolve().parents[1],
    )
    first_bytes = receipt_path.read_bytes()
    receipt = json.loads(first_bytes)

    assert receipt["schema_version"] == "tcn-v9-plan/v1"
    assert receipt["sealed_test_accessed"] is False
    assert receipt["resolved_plan"]["candidate_order"] == list(plan.candidate_order)
    assert receipt["resolved_plan"]["model_family"] == "tcn"
    assert receipt["identities"]["data"] == "a" * 64
    assert receipt["code_identity"]["dirty"] in {True, False, None}
    assert len(receipt["receipt_id"]) == 64

    replayed = execute_v9_plan(
        plan,
        _ordinary_split(),
        output_dir=output_dir,
        project_root=Path(__file__).resolve().parents[1],
    )
    assert replayed == receipt_path
    assert replayed.read_bytes() == first_bytes

    drifted = replace(plan, torch_threads=8)
    with pytest.raises(ContractError, match="identity drift"):
        execute_v9_plan(
            drifted,
            _ordinary_split(),
            output_dir=output_dir,
            project_root=Path(__file__).resolve().parents[1],
        )
