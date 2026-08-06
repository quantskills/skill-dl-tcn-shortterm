from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

from skill_dl_tcn_shortterm.tcn import BaiTCN
from skill_dl_tcn_shortterm.tcn_lite import TCNLite
from skill_dl_tcn_shortterm.v9_confirmation import V9FinalContext
from skill_dl_tcn_shortterm.v9_diagnostics import (
    InfraDiagnosticInput,
    LayerProbeInput,
    RankResolutionInput,
    V9DiagnosticRequest,
)
from skill_dl_tcn_shortterm.v9_protocol import V9Plan, V9TrialSpec
from skill_dl_tcn_shortterm.v9_representation import ProbeCheckpointEvidence
from skill_dl_tcn_shortterm.v9_representation import checkpoint_state_identity
from skill_dl_tcn_shortterm.v9_runner import run_v9_evidence_path
from skill_dl_tcn_shortterm.v9_training import V9TrainingRequest
import skill_dl_tcn_shortterm.v9_diagnostics as v9_diagnostics_module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_highest_v9_diagnostic_stage_derives_evidence_from_frozen_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(77)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, rng.normal(size=(60, 3, 480)).astype("float32"))
    features = np.load(feature_path, mmap_mode="r")
    window_index = pd.DataFrame(
        {
            "sample_position": range(60),
            "sample_id": [f"s{value}" for value in range(60)],
            "signal_date": [f"2025-01-{value // 5 + 1:02d}" for value in range(60)],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "signal_date": window_index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": float((sample % 5) / 2 - 1),
                "valid": True,
            }
            for sample in range(60)
            for horizon in [1, 2, 3, 5]
        ]
    )
    split = pd.concat(
        [
            pd.DataFrame(
                {
                    "sample_position": range(60),
                    "fold": fold,
                    "stage": ["train"] * 40 + ["validation"] * 20,
                    "sealed": False,
                }
            )
            for fold in range(5)
        ],
        ignore_index=True,
    )
    window_path = tmp_path / "window-index.csv"
    label_path = tmp_path / "labels.csv"
    split_path = tmp_path / "split.csv"
    evaluation_path = tmp_path / "evaluation.json"
    rankic_predictions_path = tmp_path / "rankic-predictions.csv"
    window_index.to_csv(window_path, index=False)
    labels.to_csv(label_path, index=False)
    split.to_csv(split_path, index=False)
    evaluation_path.write_text('{"metric":"daily-rankic"}', encoding="utf-8")
    lite_config_path = tmp_path / "lite-config.json"
    bai_config_path = tmp_path / "bai-config.json"
    rankic_candidate_config_path = tmp_path / "rankic-candidate-config.json"
    common_config = {
        "channels": 16,
        "kernel_size": 3,
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 128,
        "max_epochs": 8,
        "patience": 2,
        "min_delta": 0.002,
        "seed": 7,
        "torch_threads": 1,
    }
    lite_config_path.write_text(
        json.dumps(
            {
                "model": "tcn-lite-16",
                **common_config,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
            }
        ),
        encoding="utf-8",
    )
    bai_config_path.write_text(
        json.dumps(
            {
                "model": "bai-tcn-16",
                **common_config,
                "dilations": [1, 2, 4, 8, 16, 32, 64],
            }
        ),
        encoding="utf-8",
    )
    rankic_candidate_config_path.write_text(
        '{"model":"historical-tcn-candidate","protocol":"v8-frozen"}',
        encoding="utf-8",
    )
    torch.manual_seed(7)
    lite_template = TCNLite(
        feature_count=3,
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        dropout=0.0,
    )
    bai_template = BaiTCN(
        feature_count=3,
        channels=16,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        dropout=0.0,
    )
    lite_checkpoint_paths = {
        fold: tmp_path / f"lite-fold-{fold}.pt" for fold in range(5)
    }
    bai_checkpoint_paths = {
        fold: tmp_path / f"bai-fold-{fold}.pt" for fold in range(5)
    }
    for fold in range(5):
        lite_state = {
            name: value.detach().clone()
            for name, value in lite_template.state_dict().items()
        }
        bai_state = {
            name: value.detach().clone()
            for name, value in bai_template.state_dict().items()
        }
        lite_state["head.bias"][0] += fold * 1e-6
        bai_state["head.bias"][0] += fold * 1e-6
        torch.save(lite_state, lite_checkpoint_paths[fold])
        torch.save(bai_state, bai_checkpoint_paths[fold])
    rankic_candidate_checkpoints = {
        fold: hashlib.sha256(f"historical-candidate-fold-{fold}".encode()).hexdigest()
        for fold in range(5)
    }
    pd.DataFrame(
        [
            {
                "model": model,
                "fold": fold,
                "sample_position": position,
                "horizon": horizon,
                "prediction": float((position % 5) / 2 - 1)
                * (
                    -1.0
                    if (
                        model == "historical-tcn-candidate"
                        and (position // 5 + horizon) % 2 == 0
                    )
                    or (
                        model == "lite-c16-no-dropout"
                        and (position // 5 + horizon) % 3 == 0
                    )
                    else 1.0
                ),
                "config_identity": (
                    _sha256(lite_config_path)
                    if model == "lite-c16-no-dropout"
                    else _sha256(rankic_candidate_config_path)
                ),
                "checkpoint_identity": (
                    checkpoint_state_identity(lite_checkpoint_paths[fold])
                    if model == "lite-c16-no-dropout"
                    else rankic_candidate_checkpoints[fold]
                ),
                "stage": "validation",
                "sealed": False,
            }
            for model in ["lite-c16-no-dropout", "historical-tcn-candidate"]
            for fold in range(5)
            for position in range(40, 60)
            for horizon in [1, 2, 3, 5]
        ]
    ).to_csv(rankic_predictions_path, index=False)
    identities = {
        "data": _sha256(feature_path),
        "fold_manifest": _sha256(split_path),
        "window_index": _sha256(window_path),
        "labels": _sha256(label_path),
        "evaluation": _sha256(evaluation_path),
        "rankic_predictions": _sha256(rankic_predictions_path),
        "lite_config": _sha256(lite_config_path),
        "bai_config": _sha256(bai_config_path),
        "rankic_candidate_config": _sha256(rankic_candidate_config_path),
        **{
            f"lite_checkpoint_fold_{fold}": checkpoint_state_identity(
                lite_checkpoint_paths[fold]
            )
            for fold in range(5)
        },
        **{
            f"bai_checkpoint_fold_{fold}": checkpoint_state_identity(
                bai_checkpoint_paths[fold]
            )
            for fold in range(5)
        },
        **{
            f"rankic_candidate_checkpoint_fold_{fold}": (
                rankic_candidate_checkpoints[fold]
            )
            for fold in range(5)
        },
    }
    plan = V9Plan(
        run_id="artifact-bound-diagnostics",
        stage="diagnostic",
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
            V9TrialSpec("horizon_skip", "v9b-horizon-skip", "pending"),
            V9TrialSpec("rank_objective", "v9c-rank-objective", "pending"),
            V9TrialSpec("pcgrad", "v9d-pcgrad", "pending"),
        ),
        candidate_order=(
            "v9b-horizon-skip",
            "v9c-rank-objective",
            "v9d-pcgrad",
        ),
        source_identities=identities,
    )
    training = V9TrainingRequest(
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
            "labels": label_path,
            "evaluation": evaluation_path,
        },
    )
    probes = []
    for fold in range(5):
        probes.extend(
            [
                LayerProbeInput(
                    fold=fold,
                    checkpoint=ProbeCheckpointEvidence(
                        "tcn-lite-16",
                        identities["lite_config"],
                        identities["data"],
                        identities["fold_manifest"],
                        identities[f"lite_checkpoint_fold_{fold}"],
                        lite_checkpoint_paths[fold],
                        lite_config_path,
                    ),
                ),
                LayerProbeInput(
                    fold=fold,
                    checkpoint=ProbeCheckpointEvidence(
                        "bai-tcn-16",
                        identities["bai_config"],
                        identities["data"],
                        identities["fold_manifest"],
                        identities[f"bai_checkpoint_fold_{fold}"],
                        bai_checkpoint_paths[fold],
                        bai_config_path,
                    ),
                ),
            ]
        )
    request = V9DiagnosticRequest(
        training=training,
        layer_probes=tuple(probes),
        rank_resolution=RankResolutionInput(rankic_predictions_path),
        infra=InfraDiagnosticInput(warmup=1, repeats=3),
    )
    context = V9FinalContext(
        resolved_config={"protocol": "tcn-v9", "stage": "diagnostic"},
        source_identities=identities,
        checkpoint_identities={
            "lite": identities["lite_checkpoint_fold_0"],
            "bai": identities["bai_checkpoint_fold_0"],
        },
        environment={"hardware": "synthetic-cpu"},
        upstream_receipts={"diagnostics": "d" * 64},
    )

    result = run_v9_evidence_path(
        plan,
        split,
        upstream_receipts={},
        seed7_leaderboard=pd.DataFrame(),
        confirmation_measurements=pd.DataFrame(),
        context=context,
        output_root=tmp_path / "run",
        project_root=Path(__file__).resolve().parents[1],
        diagnostic_request=request,
    )

    assert result.status == "diagnostics_complete"
    assert result.final_receipt.is_file()
    summary = json.loads(result.final_receipt.read_text(encoding="utf-8"))
    upstream = summary["upstream_results"]
    detailed_keys = {
        "horizon_skip": "probe_metrics",
        "rank_objective": "resolution_summary",
        "pcgrad": "gradient_diagnostics",
        "infra": "operator_profile",
    }
    for name, detail_key in detailed_keys.items():
        receipt = json.loads(
            Path(upstream[name]["receipt_path"]).read_text(encoding="utf-8")
        )
        assert detail_key in receipt["evidence"]
        assert receipt["sealed_test_accessed"] is False
        if name == "pcgrad":
            gradients = pd.DataFrame(receipt["evidence"]["gradient_diagnostics"])
            assert gradients["signal_date"].notna().all()
            assert (
                gradients.groupby(["fold", "batch_id"], observed=True)[
                    "signal_date"
                ].nunique()
                == 1
            ).all()

    reproduced_lite_states = {
        fold: torch.load(
            lite_checkpoint_paths[fold], map_location="cpu", weights_only=True
        )
        for fold in range(5)
    }
    lite_checkpoint_paths[4].unlink()
    monkeypatch.setattr(
        v9_diagnostics_module,
        "_reproduce_control_states",
        lambda family, training, frozen_plan: reproduced_lite_states,
    )
    replay = run_v9_evidence_path(
        plan,
        split,
        upstream_receipts={},
        seed7_leaderboard=pd.DataFrame(),
        confirmation_measurements=pd.DataFrame(),
        context=context,
        output_root=tmp_path / "reproduced-run",
        project_root=Path(__file__).resolve().parents[1],
        diagnostic_request=request,
    )
    assert replay.status == "diagnostics_complete"
