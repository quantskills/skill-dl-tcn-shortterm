from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.v9_confirmation import (
    V9FinalContext,
    evaluate_multiseed_confirmation,
    finalize_v9_run,
)
from skill_dl_tcn_shortterm.v9_selection import Seed7Decision


def _seed7(status: str = "seed7_winner_admitted") -> Seed7Decision:
    return Seed7Decision(
        status=status,
        winner_trial_id=("v9b-horizon-skip" if status == "seed7_winner_admitted" else None),
        confirmation_seeds=((17, 27) if status == "seed7_winner_admitted" else ()),
        blockers=(() if status == "seed7_winner_admitted" else ("seed7_gate_failed",)),
        summary=pd.DataFrame(
            [{"trial_id": "v9b-horizon-skip", "mean_rankic": 0.095}]
        ),
    )


def _measurements() -> pd.DataFrame:
    rows = []
    settings = {
        "v9b-horizon-skip": (0.096, 5500.0, 6200.0, 1500),
        "tcn-lite-4": (0.086, 6100.0, 6800.0, 500),
        "lstm": (0.090, 2200.0, 2500.0, 1400),
        "gru": (0.088, 2500.0, 2800.0, 1300),
    }
    for model, (rankic, throughput, model_step, parameters) in settings.items():
        for seed in [7, 17, 27]:
            for fold in range(5):
                rows.append(
                    {
                        "model": model,
                        "fold": fold,
                        "seed": seed,
                        "rankic": rankic + fold * 0.0002 + seed * 0.000001,
                        "samples_per_second": throughput,
                        "model_step_samples_per_second": model_step,
                        "model_step_seconds": 1.25,
                        "data_wait_seconds": 0.25,
                        "validation_seconds": 0.5,
                        "complete_cycle_seconds": 2.0,
                        "time_to_best_seconds": 1.75,
                        "parameter_count": parameters,
                        "precision": "float32",
                        "torch_threads": 4,
                        "batch_size": 128,
                        "data_identity": "a" * 64,
                        "fold_identity": "b" * 64,
                        "evaluation_identity": "c" * 64,
                        "max_epochs": 8,
                        "patience": 2,
                        "min_delta": 0.002,
                        "loss_identity": "smooth-l1",
                        "infra_identity": "eager",
                        "candidate_config_identity": f"{model}-frozen",
                        "simplex_weights": (
                            "[[0.25,0.25,0.25,0.25]]"
                            if model == "v9b-horizon-skip"
                            else None
                        ),
                        "sealed_test_accessed": False,
                    }
                )
    return pd.DataFrame(rows)


def _context() -> V9FinalContext:
    return V9FinalContext(
        resolved_config={
            "protocol": "tcn-v9",
            "folds": [0, 1, 2, 3, 4],
            "seeds": [7, 17, 27],
            "precision": "float32",
            "torch_threads": 4,
        },
        source_identities={"data": "a" * 64, "fold_manifest": "b" * 64},
        checkpoint_identities={"v9b-horizon-skip": "d" * 64},
        environment={"hardware": "synthetic-cpu", "torch_threads": 4},
        upstream_receipts={
            "diagnostics": "e" * 64,
            "seed7_screen": "f" * 64,
        },
    )


def test_multiseed_confirmation_applies_all_effect_and_speed_gates() -> None:
    decision = evaluate_multiseed_confirmation(
        _measurements(),
        seed7_decision=_seed7(),
        control_trial_id="tcn-lite-4",
        lstm_model_id="lstm",
        gru_model_id="gru",
    )

    assert decision.status == "pareto_candidate_confirmed_v9"
    assert decision.blockers == ()
    assert decision.metrics["unit_count"] == 15
    assert decision.metrics["median_rankic"] >= 0.09
    assert decision.metrics["positive_rate"] == 1.0
    assert decision.metrics["paired_median_improvement"] >= 0.005
    assert decision.metrics["median_samples_per_second"] >= 5000
    assert decision.speed_ratios["vs_lstm_model_step"] > 2
    assert decision.speed_ratios["vs_gru_end_to_end"] > 2
    assert decision.sealed_test_accessed is False


def test_multiseed_confirmation_stops_on_negative_seed_or_upstream_stop() -> None:
    failed = _measurements()
    failed.loc[
        failed["model"].eq("v9b-horizon-skip") & failed["seed"].eq(27),
        "rankic",
    ] = 0.07
    decision = evaluate_multiseed_confirmation(
        failed,
        seed7_decision=_seed7(),
        control_trial_id="tcn-lite-4",
        lstm_model_id="lstm",
        gru_model_id="gru",
    )
    assert decision.status == "stop_no_pareto_gain_v9"
    assert "not_all_seed_improvements_positive" in decision.blockers

    direct_stop = evaluate_multiseed_confirmation(
        pd.DataFrame(),
        seed7_decision=_seed7("stop_no_pareto_gain_v9"),
        control_trial_id="tcn-lite-4",
        lstm_model_id="lstm",
        gru_model_id="gru",
    )
    assert direct_stop.status == "stop_no_pareto_gain_v9"
    assert direct_stop.metrics == {}

    with pytest.raises(ContractError, match="must end directly"):
        evaluate_multiseed_confirmation(
            _measurements(),
            seed7_decision=_seed7("stop_no_pareto_gain_v9"),
            control_trial_id="tcn-lite-4",
            lstm_model_id="lstm",
            gru_model_id="gru",
        )


def test_final_v9_receipt_is_complete_replayable_and_rejects_identity_drift(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "v9-final"
    receipt_path = finalize_v9_run(
        _measurements(),
        seed7_decision=_seed7(),
        context=_context(),
        output_dir=output_dir,
        project_root=Path(__file__).resolve().parents[1],
        control_trial_id="tcn-lite-4",
        lstm_model_id="lstm",
        gru_model_id="gru",
    )
    first = receipt_path.read_bytes()
    receipt = json.loads(first)
    assert receipt["schema_version"] == "tcn-v9-final/v1"
    assert receipt["status"] == "pareto_candidate_confirmed_v9"
    assert receipt["sealed_test_accessed"] is False
    assert receipt["selection"]["winner_trial_id"] == "v9b-horizon-skip"
    assert receipt["identities"]["sources"]["data"] == "a" * 64
    assert len(receipt["measurements"]) == 60
    seed17_candidate = next(
        row
        for row in receipt["measurements"]
        if row["model"] == "v9b-horizon-skip"
        and row["seed"] == 17
        and row["fold"] == 0
    )
    assert seed17_candidate["simplex_weights"] == "[[0.25,0.25,0.25,0.25]]"
    assert seed17_candidate["rankic"] > 0
    assert seed17_candidate["model_step_seconds"] == 1.25
    assert seed17_candidate["data_wait_seconds"] == 0.25
    assert seed17_candidate["validation_seconds"] == 0.5
    assert seed17_candidate["complete_cycle_seconds"] == 2.0
    assert seed17_candidate["time_to_best_seconds"] == 1.75
    assert receipt["code_identity"]["source_sha256"]
    assert len(receipt["receipt_id"]) == 64

    replay = finalize_v9_run(
        _measurements(),
        seed7_decision=_seed7(),
        context=_context(),
        output_dir=output_dir,
        project_root=Path(__file__).resolve().parents[1],
        control_trial_id="tcn-lite-4",
        lstm_model_id="lstm",
        gru_model_id="gru",
    )
    assert replay.read_bytes() == first

    drifted = replace(_context(), environment={"hardware": "other-cpu"})
    with pytest.raises(ContractError, match="identity drift"):
        finalize_v9_run(
            _measurements(),
            seed7_decision=_seed7(),
            context=drifted,
            output_dir=output_dir,
            project_root=Path(__file__).resolve().parents[1],
            control_trial_id="tcn-lite-4",
            lstm_model_id="lstm",
            gru_model_id="gru",
        )
