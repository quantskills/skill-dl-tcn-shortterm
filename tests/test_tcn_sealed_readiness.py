from __future__ import annotations

import copy

import pandas as pd
import pytest

from skill_dl_tcn_shortterm.experiment import ContractError
from skill_dl_tcn_shortterm.sealed_readiness import (
    EXACT_SEALED_AUTHORIZATION,
    EXACT_SEALED_AUTHORIZATION_SHA256,
    build_eligible_checkpoint_plan,
    require_exact_sealed_authorization,
    validate_task_aligned_freeze_config,
)


def _config() -> dict[str, object]:
    return {
        "protocol_version": "v36",
        "run_id": "v36",
        "v35_candidate_artifact": "candidate",
        "v35_receipt_id": "candidate-id",
        "v35_candidate_status": "constrained_tail_ordinary_validation_candidate_v35",
        "lstm_artifact": "lstm",
        "lstm_receipt_id": "lstm-id",
        "ordinary_split_manifest": "ordinary.parquet",
        "sealed_split_manifest": "sealed.parquet",
        "expected_sha256": {
            "ordinary_split_manifest": "a" * 64,
            "sealed_split_manifest": "b" * 64,
            "features": "c" * 64,
            "window_index": "d" * 64,
            "labels": "e" * 64,
        },
        "seeds": [7, 17, 27],
        "ordinary_folds": [0, 1, 2, 3, 4],
        "sealed_test_folds": [0, 1],
        "expected_eligible_unit_count": 24,
        "expected_changed_unit_exposures": 12,
        "top_fraction": 0.1,
        "bootstrap_seed": 36,
        "bootstrap_draws": 5000,
        "one_way_cost_bps": 10.0,
        "authorization_text_sha256": EXACT_SEALED_AUTHORIZATION_SHA256,
        "evaluation_policy": {
            "sealed_stage": "test",
            "exclude_stage": "sealed_holdout",
            "eligibility_guard": "validation_end_date < sealed_test_start_date",
            "model_unit_policy": "all_eligible_seed_fold_units",
            "metric_aggregation": "date_block_paired_unit_mean",
            "sealed_reuse": "exactly_once",
            "post_result_tuning": "forbidden",
        },
        "gates": {
            "min_mean_top_precision_delta": 0.0,
            "min_mean_ndcg_at_top_delta": 0.0,
            "min_primary_tail_ci_low": 0.0,
            "min_secondary_tail_ci_low": -0.002,
            "min_mean_rankic_delta": -0.002,
            "min_top_return_ci_low": -0.0005,
            "min_net_return_after_cost_ci_low": -0.0005,
            "max_mean_top_turnover_delta": 0.02,
            "min_model_step_speed_ratio": 3.0,
            "min_end_to_end_speed_ratio": 3.0,
        },
    }


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_ends = [
        "2023-01-06",
        "2023-05-11",
        "2023-09-04",
        "2024-06-14",
        "2024-10-15",
    ]
    ordinary = pd.DataFrame(
        [
            {
                "fold": fold,
                "sample_id": f"ordinary-{fold}",
                "signal_date": end,
                "stage": "validation",
                "sealed": False,
            }
            for fold, end in enumerate(validation_ends)
        ]
    )
    sealed = pd.DataFrame(
        [
            {
                "fold": 0,
                "sample_id": "test-0",
                "signal_date": "2023-12-13",
                "stage": "test",
                "sealed": True,
            },
            {
                "fold": 1,
                "sample_id": "repeat-0",
                "signal_date": "2023-12-13",
                "stage": "sealed_holdout",
                "sealed": True,
            },
            {
                "fold": 1,
                "sample_id": "test-1",
                "signal_date": "2024-10-30",
                "stage": "test",
                "sealed": True,
            },
        ]
    )
    changed = {(7, 0), (17, 0), (7, 1), (17, 1), (27, 1), (17, 3), (27, 4)}
    selection_rows = []
    lstm_rows = []
    for seed in (7, 17, 27):
        for fold in range(5):
            selection_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "selection_changed": (seed, fold) in changed,
                    "control_epoch": 0,
                    "candidate_epoch": 1 if (seed, fold) in changed else 0,
                    "control_checkpoint": f"control-{seed}-{fold}.pt",
                    "control_checkpoint_sha256": "a" * 64,
                    "candidate_checkpoint": f"candidate-{seed}-{fold}.pt",
                    "candidate_checkpoint_sha256": "b" * 64,
                }
            )
            lstm_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "checkpoint": f"lstm-{seed}-{fold}.pt",
                    "checkpoint_sha256": "c" * 64,
                }
            )
    return ordinary, sealed, pd.DataFrame(selection_rows), pd.DataFrame(lstm_rows)


def test_authorization_must_match_exactly() -> None:
    require_exact_sealed_authorization(EXACT_SEALED_AUTHORIZATION)
    for value in (None, "同意", "授权 sealed test", "授权执行 sealed test "):
        with pytest.raises(ContractError, match="exact authorization"):
            require_exact_sealed_authorization(value)


def test_freeze_config_rejects_protocol_or_gate_drift() -> None:
    assert validate_task_aligned_freeze_config(_config())["protocol_version"] == "v36"
    altered = copy.deepcopy(_config())
    altered["authorization_text_sha256"] = "f" * 64
    with pytest.raises(ContractError, match="authorization identity drifted"):
        validate_task_aligned_freeze_config(altered)
    altered = copy.deepcopy(_config())
    assert isinstance(altered["gates"], dict)
    altered["gates"].pop("min_mean_rankic_delta")
    with pytest.raises(ContractError, match="task-aligned gates"):
        validate_task_aligned_freeze_config(altered)


def test_checkpoint_plan_excludes_future_folds_and_duplicate_holdout() -> None:
    plan = build_eligible_checkpoint_plan(*_frames())
    assert len(plan) == 24
    assert int(plan["selection_changed"].sum()) == 12
    first = plan.loc[plan["sealed_fold"].eq(0)]
    second = plan.loc[plan["sealed_fold"].eq(1)]
    assert set(first["training_fold"]) == {0, 1, 2}
    assert set(second["training_fold"]) == {0, 1, 2, 3, 4}
    assert len(first) == 9
    assert len(second) == 15
    assert (pd.to_datetime(plan["validation_end_date"]) < pd.to_datetime(plan["sealed_test_start_date"])).all()


def test_checkpoint_plan_rejects_ordinary_sealed_overlap() -> None:
    ordinary, sealed, selection, lstm = _frames()
    first_test = sealed.index[sealed["stage"].eq("test")][0]
    sealed.loc[first_test, "sample_id"] = "ordinary-0"
    with pytest.raises(ContractError, match="identities overlap"):
        build_eligible_checkpoint_plan(ordinary, sealed, selection, lstm)


def test_checkpoint_plan_rejects_unsealed_canonical_test() -> None:
    ordinary, sealed, selection, lstm = _frames()
    sealed.loc[sealed["stage"].eq("test"), "sealed"] = False
    with pytest.raises(ContractError, match="no fully sealed test"):
        build_eligible_checkpoint_plan(ordinary, sealed, selection, lstm)
