"""Run the frozen v42 train-only cross-seed consensus student probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.consensus_distillation import (  # noqa: E402
    blend_training_targets,
    build_fold_consensus_rank_targets,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.neural import HORIZONS, _label_matrices  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials  # noqa: E402
from skill_dl_tcn_shortterm.task_aligned_evaluation import (  # noqa: E402
    bootstrap_task_aligned_differences,
    compare_task_aligned_models,
    evaluate_task_aligned_predictions,
    summarize_task_aligned_metrics,
    validate_prediction_contract,
)
from skill_dl_tcn_shortterm.training_data import (  # noqa: E402
    LazyWindowDataset,
    build_fold_protocols,
)
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    _predict_tcn_trial,
    build_tcn_trial_model,
    run_tcn_validation_sweep,
)
from skill_dl_tcn_shortterm.v41_validation import (  # noqa: E402
    decide_ema_holistic_gate,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_relative_feature_validation import (  # noqa: E402
    _collect_tcn_predictions,
)
from run_tcn_task_aligned_evaluation import _label_lookup  # noqa: E402
from run_tcn_v41_ema_seed7 import (  # noqa: E402
    _contains_secret_key,
    _geometric_mean,
    _sha256,
    _unit_deltas,
    _write_json,
)


EXPECTED_GATES = {
    "min_mean_rankic_delta": 0.002,
    "min_positive_folds": 3,
    "min_rankic_ci_low": -0.002,
    "min_broad_metric_count": 4,
    "min_top_return_delta": -0.0001,
    "min_top_precision_delta": -0.002,
    "min_ndcg_delta": -0.001,
    "min_quantile_monotonicity_delta": -0.002,
    "min_positive_horizons": 3,
    "min_worst_horizon_rankic_delta": -0.003,
    "min_model_step_retention": 0.95,
    "min_complete_cycle_retention": 0.90,
    "min_implied_tcn_lstm_model_step_ratio": 3.0,
}


def _teacher_specs(config: dict[str, object]) -> dict[tuple[int, int], Path]:
    raw_specs = config.get("teacher_checkpoints")
    if not isinstance(raw_specs, list):
        raise ContractError("v42 teacher checkpoint manifest is missing")
    paths: dict[tuple[int, int], Path] = {}
    for raw in raw_specs:
        if not isinstance(raw, dict):
            raise ContractError("v42 teacher checkpoint entry is invalid")
        seed = int(cast(Any, raw["seed"]))
        fold = int(cast(Any, raw["fold"]))
        path = (ROOT / str(raw["path"])).resolve()
        if not path.is_file() or _sha256(path) != str(raw["sha256"]):
            raise ContractError(f"v42 teacher checkpoint fingerprint mismatch: {seed}/{fold}")
        key = (seed, fold)
        if key in paths:
            raise ContractError("v42 teacher checkpoint manifest has duplicates")
        paths[key] = path
    expected = {(seed, fold) for seed in (7, 17, 27) for fold in range(5)}
    if set(paths) != expected:
        raise ContractError("v42 teacher checkpoint coverage drifted")
    return paths


def _build_teacher_overrides(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split: pd.DataFrame,
    *,
    teacher_trial: object,
    checkpoints: dict[tuple[int, int], Path],
    teacher_weight: float,
) -> tuple[dict[int, np.ndarray], pd.DataFrame, pd.DataFrame, float]:
    trial = cast(Any, teacher_trial)
    protocols = build_fold_protocols(features, split)
    true_targets, masks = _label_matrices(window_index, labels)
    dummy_targets = np.zeros_like(true_targets, dtype="float32")
    dummy_masks = np.ones_like(masks, dtype="bool")
    dates = window_index.set_index("sample_position")["signal_date"].astype(str)
    prediction_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    overrides: dict[int, np.ndarray] = {}
    started = time.perf_counter()
    for protocol in protocols:
        dataset = LazyWindowDataset(
            features,
            protocol.train_positions,
            dummy_targets,
            dummy_masks,
            protocol.feature_mean,
            protocol.feature_std,
        )
        fold_rows: list[dict[str, object]] = []
        expected_positions = set(int(value) for value in protocol.train_positions)
        validation_positions = set(int(value) for value in protocol.validation_positions)
        if expected_positions.intersection(validation_positions):
            raise ContractError("v42 train and validation positions overlap")
        for seed in (7, 17, 27):
            model = build_tcn_trial_model(
                trial,
                feature_count=int(features.shape[1]),
                input_steps=int(features.shape[2]),
            )
            state = torch.load(
                checkpoints[(seed, protocol.fold)],
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, dict):
                raise ContractError("v42 teacher checkpoint is not a state dict")
            model.load_state_dict(state, strict=True)
            scores, positions = _predict_tcn_trial(
                model, dataset, batch_size=int(trial.batch_size)
            )
            if set(int(value) for value in positions) != expected_positions:
                raise ContractError("v42 teacher predictions escaped train positions")
            for row_offset, position in enumerate(positions):
                sample_position = int(position)
                for horizon_offset, horizon in enumerate(HORIZONS):
                    fold_rows.append(
                        {
                            "seed": seed,
                            "fold": protocol.fold,
                            "sample_position": sample_position,
                            "signal_date": dates.loc[sample_position],
                            "horizon": horizon,
                            "score": float(scores[row_offset, horizon_offset]),
                        }
                    )
        fold_predictions = pd.DataFrame(fold_rows)
        teacher_targets = build_fold_consensus_rank_targets(
            fold_predictions,
            sample_count=len(features),
            train_positions=np.asarray(protocol.train_positions, dtype="int64"),
            expected_seeds=(7, 17, 27),
        )
        override = blend_training_targets(
            true_targets,
            masks,
            teacher_targets,
            train_positions=np.asarray(protocol.train_positions, dtype="int64"),
            teacher_weight=teacher_weight,
        )
        overrides[protocol.fold] = override
        prediction_rows.extend(fold_rows)
        for position in protocol.train_positions:
            for horizon_offset, horizon in enumerate(HORIZONS):
                target_rows.append(
                    {
                        "fold": protocol.fold,
                        "sample_position": int(position),
                        "signal_date": dates.loc[int(position)],
                        "horizon": horizon,
                        "teacher_consensus_rank": float(
                            teacher_targets[int(position), horizon_offset]
                        ),
                        "distilled_target": float(
                            override[int(position), horizon_offset]
                        ),
                        "valid": bool(masks[int(position), horizon_offset]),
                    }
                )
    return (
        overrides,
        pd.DataFrame(prediction_rows),
        pd.DataFrame(target_rows),
        time.perf_counter() - started,
    )


def _ensemble_upper_bound(
    v40_predictions: pd.DataFrame,
    student_summary: pd.DataFrame,
) -> dict[str, float]:
    selected = v40_predictions.loc[
        v40_predictions["model"].astype(str).eq("relative_tcn")
    ].copy()
    keys = ["fold", "sample_id", "instrument_id", "signal_date", "horizon"]
    fixed = [
        "rank_target",
        "raw_return",
        "stage",
        "sealed",
        "prediction_contract_id",
        "target_contract_id",
        "evaluation_contract_id",
    ]
    aggregations = {column: (column, "first") for column in fixed}
    ensemble = selected.groupby(keys, as_index=False, observed=True).agg(
        score=("score", "mean"), **aggregations
    )
    ensemble["model"] = "teacher_ensemble"
    ensemble["seed"] = 0
    ensemble["training_contract_id"] = "v40-three-seed-tcn-teacher-ensemble"
    teacher_summary = summarize_task_aligned_metrics(
        evaluate_task_aligned_predictions(ensemble)
    ).iloc[0]
    summaries = student_summary.set_index("model")
    metrics = (
        "mean_rankic",
        "mean_top_return",
        "mean_top_precision",
        "mean_ndcg_at_top",
        "mean_quantile_monotonicity",
    )
    result: dict[str, float] = {}
    for metric in metrics:
        control = float(cast(Any, summaries.loc["control_tcn", metric]))
        candidate = float(
            cast(Any, summaries.loc["consensus_student_tcn", metric])
        )
        teacher = float(cast(Any, teacher_summary[metric]))
        denominator = teacher - control
        result[f"teacher_{metric}"] = teacher
        result[f"student_recovery_{metric}"] = (
            (candidate - control) / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--relative-feature-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--v40-run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v42 seed7 run refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("protocol_version") != "v42-phase-a":
            raise ContractError("v42 phase-A config identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v42 config contains a secret-like key")
        if config.get("gates") != EXPECTED_GATES:
            raise ContractError("v42 holistic gates drifted")
        if (
            tuple(cast(list[object], config["student_seeds"])) != (7,)
            or tuple(cast(list[object], config["teacher_seeds"])) != (7, 17, 27)
            or tuple(cast(list[object], config["folds"])) != (0, 1, 2, 3, 4)
            or float(cast(Any, config["teacher_weight"])) != 0.25
            or config.get("precision") != "float32"
            or int(cast(Any, config["torch_threads"])) != 8
            or int(cast(Any, config["num_workers"])) != 0
        ):
            raise ContractError("v42 frozen execution contract drifted")

        base = arguments.base_run_dir.resolve()
        relative = arguments.relative_feature_dir.resolve()
        v40 = arguments.v40_run_dir.resolve()
        source_paths = {
            "relative_features": relative / "feature-windows.npy",
            "relative_window_index": relative / "window-index.parquet",
            "relative_manifest": relative / "manifest.json",
            "relative_receipt": relative / "receipt.json",
            "labels": base / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "v39_receipt": ROOT / "artifacts/tcn-top50-relative-seed7-screen-v39/receipt.json",
            "v40_predictions": v40 / "predictions.parquet",
            "v40_timing_summary": v40 / "timing-summary.json",
            "v40_receipt": v40 / "receipt.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v42 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v42 source SHA-256 identity drifted")
        for name in ("v39_receipt", "v40_receipt"):
            source_receipt = json.loads(
                source_paths[name].read_text(encoding="utf-8")
            )
            if not isinstance(source_receipt, dict) or source_receipt.get("sealed_test_accessed") is not False:
                raise ContractError(f"v42 source is not ordinary validation: {name}")
        checkpoints = _teacher_specs(config)

        features = np.load(
            source_paths["relative_features"], mmap_mode="r", allow_pickle=False
        )
        if features.ndim != 3 or features.shape[1:] != (10, 480):
            raise ContractError("v42 relative10 tensor shape drifted")
        window_index = pd.read_parquet(source_paths["relative_window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v42 rejects sealed split rows")
        stages = {str(stage) for stage in raw_split["stage"].tolist()}
        if unknown := sorted(stages - {"train", "validation", "purged"}):
            raise ContractError("v42 split has forbidden stages: " + ", ".join(unknown))
        split = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 2:
            raise ContractError("v42 requires exactly control and student trials")
        by_id = {trial.trial_id: trial for trial in trials}
        control_id = str(config["control_trial_id"])
        candidate_id = str(config["candidate_trial_id"])
        if set(by_id) != {control_id, candidate_id}:
            raise ContractError("v42 trial identities drifted")
        control = by_id[control_id]
        candidate = by_id[candidate_id]
        control_contract = asdict(control)
        candidate_contract = asdict(candidate)
        control_contract.pop("trial_id")
        candidate_contract.pop("trial_id")
        if control_contract != candidate_contract or control.ema_decay is not None:
            raise ContractError("v42 student trials differ by more than targets")

        temporary.mkdir(parents=True)
        overrides, teacher_predictions, teacher_targets, teacher_seconds = (
            _build_teacher_overrides(
                features,
                window_index,
                labels,
                split,
                teacher_trial=control,
                checkpoints=checkpoints,
                teacher_weight=0.25,
            )
        )
        max_epochs = int(cast(Any, config["max_epochs"]))
        shared_run: dict[str, Any] = {
            "seed": 7,
            "max_epochs": max_epochs,
            "patience": int(cast(Any, config["patience"])),
            "min_delta": float(cast(Any, config["min_delta"])),
            "checkpoint_min_delta": float(cast(Any, config["checkpoint_min_delta"])),
            "torch_threads": int(cast(Any, config["torch_threads"])),
            "capture_epoch_states": True,
            "disable_early_stopping": True,
        }
        control_tuning = run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split,
            trials=(control,),
            protocol_identities={
                "data": observed_hashes["relative_features"],
                "fold_manifest": observed_hashes["split_manifest"],
                "evaluation": observed_hashes["labels"],
            },
            **shared_run,
        )
        teacher_target_identity = hashlib.sha256(
            np.asarray(
                pd.util.hash_pandas_object(teacher_targets, index=False).values
            ).tobytes()
        ).hexdigest()
        candidate_tuning = run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split,
            trials=(candidate,),
            protocol_identities={
                "data": teacher_target_identity,
                "fold_manifest": observed_hashes["split_manifest"],
                "evaluation": observed_hashes["labels"],
            },
            training_target_overrides=overrides,
            **shared_run,
        )
        leaderboard = pd.concat(
            [control_tuning.leaderboard, candidate_tuning.leaderboard],
            ignore_index=True,
        )
        history = pd.concat(
            [control_tuning.epoch_history, candidate_tuning.epoch_history],
            ignore_index=True,
        )
        if (
            set(leaderboard.loc[leaderboard["trial_id"].eq(control_id), "training_target_override"])
            != {False}
            or set(leaderboard.loc[leaderboard["trial_id"].eq(candidate_id), "training_target_override"])
            != {True}
            or leaderboard.groupby("trial_id")["parameter_count"].first().nunique() != 1
        ):
            raise ContractError("v42 student training identity audit failed")

        lookup = _label_lookup(labels)
        contracts = cast(dict[str, str], config["contracts"])
        frames = []
        for trial, tuning, model_name in (
            (control, control_tuning, "control_tcn"),
            (candidate, candidate_tuning, "consensus_student_tcn"),
        ):
            model_contracts = dict(contracts)
            model_contracts["tcn_training_contract_id"] = (
                f"top50-relative10-{model_name}-v42"
            )
            frames.append(
                _collect_tcn_predictions(
                    features,
                    labels,
                    split,
                    trial,
                    tuning.best_states,
                    seed=7,
                    model_name=model_name,
                    lookup=lookup,
                    contracts=model_contracts,
                )
            )
        predictions = pd.concat(frames, ignore_index=True)
        validate_prediction_contract(predictions, expected_models=2)
        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(cast(Any, config["top_fraction"]))
        )
        summary = summarize_task_aligned_metrics(metrics)
        comparison = compare_task_aligned_models(
            metrics,
            reference_model="control_tcn",
            candidate_model="consensus_student_tcn",
        )
        bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="control_tcn",
            candidate_model="consensus_student_tcn",
            metric_columns=(
                "rankic",
                "pearson_ic",
                "top_return",
                "top_precision",
                "ndcg_at_top",
                "quantile_monotonicity",
            ),
            seed=int(cast(Any, config["bootstrap_seed"])),
            draws=int(cast(Any, config["bootstrap_draws"])),
        )
        fold_deltas, horizon_deltas = _unit_deltas(
            metrics.replace({"consensus_student_tcn": "ema_tcn"})
        )
        control_timing = leaderboard.loc[leaderboard["trial_id"].eq(control_id)].set_index("fold")
        candidate_timing = leaderboard.loc[leaderboard["trial_id"].eq(candidate_id)].set_index("fold")
        model_step_retention = _geometric_mean(
            (
                candidate_timing["model_step_samples_per_second"]
                / control_timing["model_step_samples_per_second"]
            ).to_numpy(dtype="float64")
        )
        complete_cycle_retention = _geometric_mean(
            (
                control_timing["complete_cycle_seconds"]
                / candidate_timing["complete_cycle_seconds"]
            ).to_numpy(dtype="float64")
        )
        v40_timing = json.loads(source_paths["v40_timing_summary"].read_text(encoding="utf-8"))
        v40_speed = float(v40_timing["model_step_speed_ratio_geomean"])
        implied_ratio = v40_speed * model_step_retention
        decision = decide_ema_holistic_gate(
            comparison,
            bootstrap,
            fold_deltas,
            horizon_deltas,
            raw_state_drift_max=0.0,
            model_step_retention=model_step_retention,
            complete_cycle_retention=complete_cycle_retention,
            implied_tcn_lstm_model_step_ratio=implied_ratio,
            min_model_step_retention=0.95,
            min_complete_cycle_retention=0.90,
            admitted_status="consensus_student_seed7_holistic_admitted_v42",
            rejected_status="stop_consensus_student_seed7_no_holistic_gain_v42",
        )
        upper_bound = _ensemble_upper_bound(
            pd.read_parquet(source_paths["v40_predictions"]), summary
        )
        timing = {
            "teacher_train_prediction_seconds": teacher_seconds,
            "model_step_retention_geomean": model_step_retention,
            "complete_cycle_retention_geomean": complete_cycle_retention,
            "implied_tcn_lstm_model_step_ratio": implied_ratio,
            "inference_forward_passes": 1,
        }

        teacher_predictions.to_parquet(
            temporary / "teacher-train-predictions.parquet", index=False
        )
        teacher_targets.to_parquet(
            temporary / "teacher-consensus-targets.parquet", index=False
        )
        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        leaderboard.to_parquet(temporary / "leaderboard.parquet", index=False)
        history.to_parquet(temporary / "epoch-history.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap.parquet", index=False)
        fold_deltas.to_parquet(temporary / "fold-deltas.parquet", index=False)
        horizon_deltas.to_parquet(temporary / "horizon-summary.parquet", index=False)
        _write_json(temporary / "comparison.json", comparison)
        _write_json(temporary / "teacher-upper-bound.json", upper_bound)
        _write_json(temporary / "timing.json", timing)
        _write_json(
            temporary / "teacher-audit.json",
            {
                "teacher_seeds": [7, 17, 27],
                "folds": [0, 1, 2, 3, 4],
                "teacher_weight": 0.25,
                "teacher_target_identity": teacher_target_identity,
                "teacher_prediction_rows": len(teacher_predictions),
                "validation_positions_used_for_teacher_training": 0,
                "sealed_test_accessed": False,
            },
        )
        _write_json(
            temporary / "model-gate.json",
            {
                "status": decision.status,
                "admitted": decision.admitted,
                "blockers": list(decision.blockers),
                "evidence": decision.evidence,
                "phase_b_authorized": decision.admitted,
                "sealed_test_accessed": False,
            },
        )
        _write_json(temporary / "config.resolved.json", config)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for tuning in (control_tuning, candidate_tuning):
            for key, state in tuning.best_states.items():
                torch.save(state, checkpoint_dir / f"{key}.pt")
        report = (
            "# TCN v42 cross-seed consensus student seed7 result\n\n"
            f"- status: `{decision.status}`\n"
            f"- mean RankIC delta: `{float(comparison['mean_rankic_delta']):+.6f}`\n"
            f"- broad improved metrics: `{decision.evidence['broad_metric_count']}/6`\n"
            f"- positive folds: `{decision.evidence['positive_folds']}/5`\n"
            f"- positive horizons: `{decision.evidence['positive_horizons']}/4`\n"
            f"- model-step retention: `{model_step_retention:.4f}`\n"
            f"- implied TCN/LSTM model-step ratio: `{implied_ratio:.4f}x`\n"
            "- inference model count: `1`\n"
            "- sealed_test_accessed: `false`\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")

        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-v42-consensus-student-seed7/v1",
            "run_id": str(config["run_id"]),
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "teacher_checkpoints": [
                {"seed": seed, "fold": fold, "path": str(path), "sha256": _sha256(path)}
                for (seed, fold), path in sorted(checkpoints.items())
            ],
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
            },
            "model_gate": {
                "status": decision.status,
                "admitted": decision.admitted,
                "blockers": list(decision.blockers),
                "evidence": decision.evidence,
            },
            "outputs": outputs,
            "sealed_test_accessed": False,
            "sealed_test_authorized": False,
            "alpha_ready": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "run_status": decision.status,
            "phase_b_authorized": decision.admitted,
            "receipt_id": receipt["receipt_id"],
            "output_dir": str(output_dir),
        }
    except (
        ContractError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
