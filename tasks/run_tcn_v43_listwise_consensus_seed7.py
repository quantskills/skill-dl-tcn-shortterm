"""Run the frozen v43 gradient-normalized listwise consensus probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
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
from skill_dl_tcn_shortterm.training_data import build_fold_protocols  # noqa: E402
from skill_dl_tcn_shortterm.tuning import run_tcn_validation_sweep  # noqa: E402
from skill_dl_tcn_shortterm.v43_validation import (  # noqa: E402
    decide_listwise_consensus_seed7_gate,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_relative_feature_validation import _collect_tcn_predictions  # noqa: E402
from run_tcn_task_aligned_evaluation import _label_lookup  # noqa: E402
from run_tcn_v41_ema_seed7 import (  # noqa: E402
    _contains_secret_key,
    _geometric_mean,
    _sha256,
    _unit_deltas,
    _write_json,
)


EXPECTED_GATES = {
    "control_min_mean_rankic_delta": 0.002,
    "control_min_positive_folds": 3,
    "control_min_rankic_ci_low": -0.002,
    "control_min_broad_metric_count": 4,
    "control_min_top_return_delta": -0.0001,
    "control_min_top_precision_delta": -0.002,
    "control_min_ndcg_delta": -0.001,
    "control_min_quantile_monotonicity_delta": -0.002,
    "control_min_positive_horizons": 3,
    "control_min_worst_horizon_rankic_delta": -0.003,
    "pointwise_min_broad_metric_count": 3,
    "pointwise_min_rankic_delta": -0.002,
    "pointwise_min_pearson_ic_delta": -0.002,
    "pointwise_min_top_return_delta": -0.0001,
    "pointwise_min_top_precision_delta": -0.001,
    "pointwise_min_ndcg_delta": -0.001,
    "pointwise_min_quantile_monotonicity_delta": -0.002,
    "min_teacher_fidelity_delta": 0.002,
    "teacher_gradient_ratio_low": 0.20,
    "teacher_gradient_ratio_high": 0.30,
    "min_model_step_retention": 0.70,
    "min_complete_cycle_retention": 0.70,
    "min_implied_tcn_lstm_model_step_ratio": 3.0,
    "inference_forward_passes": 1,
}


def _load_teacher_targets(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split: pd.DataFrame,
    target_path: Path,
    audit_path: Path,
) -> tuple[dict[int, np.ndarray], str, dict[str, object]]:
    table = pd.read_parquet(target_path)
    required = {
        "fold",
        "sample_position",
        "signal_date",
        "horizon",
        "teacher_consensus_rank",
        "valid",
    }
    if missing := sorted(required.difference(table.columns)):
        raise ContractError("v43 teacher targets missing: " + ", ".join(missing))
    if table.duplicated(["fold", "sample_position", "horizon"]).any():
        raise ContractError("v43 teacher targets contain duplicate keys")
    target_identity = hashlib.sha256(
        np.asarray(pd.util.hash_pandas_object(table, index=False).values).tobytes()
    ).hexdigest()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or not (
        audit.get("teacher_target_identity") == target_identity
        and audit.get("teacher_seeds") == [7, 17, 27]
        and audit.get("teacher_weight") == 0.25
        and audit.get("validation_positions_used_for_teacher_training") == 0
        and audit.get("sealed_test_accessed") is False
    ):
        raise ContractError("v43 frozen teacher target audit drifted")
    _, masks = _label_matrices(window_index, labels)
    dates = window_index.set_index("sample_position")["signal_date"].astype(str)
    horizon_offsets = {horizon: offset for offset, horizon in enumerate(HORIZONS)}
    matrices: dict[int, np.ndarray] = {}
    for protocol in build_fold_protocols(features, split):
        rows = table.loc[table["fold"].astype(int).eq(protocol.fold)].copy()
        expected_positions = set(int(value) for value in protocol.train_positions)
        if (
            set(rows["sample_position"].astype(int)) != expected_positions
            or len(rows) != len(expected_positions) * len(HORIZONS)
            or set(rows["horizon"].astype(int)) != set(HORIZONS)
        ):
            raise ContractError(f"v43 teacher coverage drifted in fold {protocol.fold}")
        positions = rows["sample_position"].to_numpy(dtype="int64")
        offsets = np.asarray(
            [horizon_offsets[int(value)] for value in rows["horizon"]],
            dtype="int64",
        )
        if not np.array_equal(
            dates.loc[positions].to_numpy(dtype=str),
            rows["signal_date"].astype(str).to_numpy(),
        ):
            raise ContractError(f"v43 teacher dates drifted in fold {protocol.fold}")
        if not np.array_equal(
            masks[positions, offsets], rows["valid"].astype(bool).to_numpy()
        ):
            raise ContractError(f"v43 teacher masks drifted in fold {protocol.fold}")
        values = rows["teacher_consensus_rank"].to_numpy(dtype="float64")
        if not np.isfinite(values).all() or bool((np.abs(values) > 1.0 + 1e-7).any()):
            raise ContractError("v43 teacher ranks escaped [-1, 1]")
        matrix = np.full(masks.shape, np.nan, dtype="float32")
        matrix[positions, offsets] = values.astype("float32")
        validation_positions = np.asarray(
            protocol.validation_positions, dtype="int64"
        )
        if np.isfinite(matrix[validation_positions]).any():
            raise ContractError("v43 teacher targets exposed validation positions")
        matrices[protocol.fold] = matrix
    if set(matrices) != set(range(5)):
        raise ContractError("v43 teacher fold coverage is incomplete")
    return matrices, target_identity, audit


def _teacher_ensemble(v40_predictions: pd.DataFrame) -> pd.DataFrame:
    selected = v40_predictions.loc[
        v40_predictions["model"].astype(str).eq("relative_tcn")
    ].copy()
    keys = ["fold", "sample_id", "instrument_id", "signal_date", "horizon"]
    ensemble = selected.groupby(keys, as_index=False, observed=True).agg(
        teacher_score=("score", "mean")
    )
    if ensemble.duplicated(keys).any():
        raise ContractError("v43 teacher ensemble coverage is duplicated")
    return ensemble


def _teacher_fidelity(
    predictions: pd.DataFrame,
    teacher: pd.DataFrame,
    *,
    models: tuple[str, str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    keys = ["fold", "sample_id", "instrument_id", "signal_date", "horizon"]
    rows: list[dict[str, object]] = []
    for model in models:
        selected = predictions.loc[
            predictions["model"].astype(str).eq(model), [*keys, "score"]
        ]
        merged = selected.merge(teacher, on=keys, validate="one_to_one")
        if len(merged) != len(selected):
            raise ContractError(f"v43 teacher fidelity coverage drifted for {model}")
        for (fold, signal_date, horizon), group in merged.groupby(
            ["fold", "signal_date", "horizon"], observed=True, sort=True
        ):
            correlation = group["score"].corr(group["teacher_score"], method="spearman")
            if not np.isfinite(correlation):
                raise ContractError("v43 teacher fidelity group is unresolved")
            rows.append(
                {
                    "model": model,
                    "fold": int(cast(Any, fold)),
                    "signal_date": str(signal_date),
                    "horizon": int(cast(Any, horizon)),
                    "teacher_rankic": float(correlation),
                }
            )
    frame = pd.DataFrame(rows)
    means = frame.groupby("model", observed=True)["teacher_rankic"].mean()
    if set(means.index.astype(str)) != set(models):
        raise ContractError("v43 teacher fidelity model coverage drifted")
    pointwise, candidate = models
    summary = {
        "pointwise_teacher_rankic": float(means.loc[pointwise]),
        "candidate_teacher_rankic": float(means.loc[candidate]),
        "teacher_fidelity_delta": float(means.loc[candidate] - means.loc[pointwise]),
    }
    return frame, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--relative-feature-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--v40-run-dir", required=True, type=Path)
    parser.add_argument("--v42-phase-a-run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v43 seed7 run refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("protocol_version") != "v43-phase-a":
            raise ContractError("v43 phase-A config identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v43 config contains a secret-like key")
        if config.get("gates") != EXPECTED_GATES:
            raise ContractError("v43 holistic gates drifted")
        if (
            tuple(cast(list[object], config["student_seeds"])) != (7,)
            or tuple(cast(list[object], config["teacher_seeds"])) != (7, 17, 27)
            or tuple(cast(list[object], config["folds"])) != (0, 1, 2, 3, 4)
            or float(cast(Any, config["teacher_gradient_ratio"])) != 0.25
            or float(cast(Any, config["teacher_temperature"])) != 0.1
            or config.get("precision") != "float32"
            or int(cast(Any, config["torch_threads"])) != 8
            or int(cast(Any, config["num_workers"])) != 0
        ):
            raise ContractError("v43 frozen execution contract drifted")

        base = arguments.base_run_dir.resolve()
        relative = arguments.relative_feature_dir.resolve()
        v40 = arguments.v40_run_dir.resolve()
        v42 = arguments.v42_phase_a_run_dir.resolve()
        source_paths = {
            "relative_features": relative / "feature-windows.npy",
            "relative_window_index": relative / "window-index.parquet",
            "relative_manifest": relative / "manifest.json",
            "relative_receipt": relative / "receipt.json",
            "labels": base / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "v40_predictions": v40 / "predictions.parquet",
            "v40_timing_summary": v40 / "timing-summary.json",
            "v40_receipt": v40 / "receipt.json",
            "v42_predictions": v42 / "predictions.parquet",
            "v42_leaderboard": v42 / "leaderboard.parquet",
            "v42_teacher_targets": v42 / "teacher-consensus-targets.parquet",
            "v42_teacher_audit": v42 / "teacher-audit.json",
            "v42_timing": v42 / "timing.json",
            "v42_model_gate": v42 / "model-gate.json",
            "v42_receipt": v42 / "receipt.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v43 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v43 source SHA-256 identity drifted")
        for receipt_name in ("v40_receipt", "v42_receipt"):
            source_receipt = json.loads(
                source_paths[receipt_name].read_text(encoding="utf-8")
            )
            if (
                not isinstance(source_receipt, dict)
                or source_receipt.get("sealed_test_accessed") is not False
            ):
                raise ContractError(f"v43 source is not ordinary validation: {receipt_name}")
        v42_gate = json.loads(source_paths["v42_model_gate"].read_text(encoding="utf-8"))
        if not isinstance(v42_gate, dict) or v42_gate.get("admitted") is not True:
            raise ContractError("v43 requires the frozen v42 seed7 pointwise student")

        features = np.load(
            source_paths["relative_features"], mmap_mode="r", allow_pickle=False
        )
        if features.ndim != 3 or features.shape[1:] != (10, 480):
            raise ContractError("v43 relative10 tensor shape drifted")
        window_index = pd.read_parquet(source_paths["relative_window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v43 rejects sealed split rows")
        stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(stages - {"train", "validation", "purged"}):
            raise ContractError("v43 split has forbidden stages: " + ", ".join(unknown))
        split = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].astype(str).isin(["train", "validation"])
        ].copy()
        teacher_targets, teacher_identity, teacher_audit = _load_teacher_targets(
            features,
            window_index,
            labels,
            split,
            source_paths["v42_teacher_targets"],
            source_paths["v42_teacher_audit"],
        )

        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 1:
            raise ContractError("v43 fixes exactly one listwise student trial")
        trial = trials[0]
        if (
            trial.trial_id != str(config["candidate_trial_id"])
            or trial.strategy != "teacher_listwise"
            or trial.teacher_listwise_gradient_ratio != 0.25
            or trial.teacher_listwise_temperature != 0.1
            or trial.ema_decay is not None
        ):
            raise ContractError("v43 listwise trial contract drifted")

        temporary.mkdir(parents=True)
        tuning = run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split,
            trials=(trial,),
            seed=7,
            max_epochs=int(cast(Any, config["max_epochs"])),
            patience=int(cast(Any, config["patience"])),
            min_delta=float(cast(Any, config["min_delta"])),
            checkpoint_min_delta=float(cast(Any, config["checkpoint_min_delta"])),
            torch_threads=int(cast(Any, config["torch_threads"])),
            protocol_identities={
                "data": teacher_identity,
                "fold_manifest": observed_hashes["split_manifest"],
                "evaluation": observed_hashes["labels"],
            },
            capture_epoch_states=True,
            disable_early_stopping=True,
            training_teacher_targets=teacher_targets,
        )
        if (
            set(tuning.leaderboard["training_target_override"]) != {False}
            or set(tuning.leaderboard["training_teacher_target"]) != {True}
        ):
            raise ContractError("v43 target identity audit failed")
        gradient_values = tuning.epoch_history[
            "teacher_gradient_ratio_median"
        ].dropna().to_numpy(dtype="float64")
        if (
            len(gradient_values) != 40
            or not np.isfinite(gradient_values).all()
            or bool(((gradient_values < 0.20) | (gradient_values > 0.30)).any())
        ):
            raise ContractError("v43 teacher gradient ratio audit failed")
        median_gradient_ratio = float(np.median(gradient_values))

        contracts = cast(dict[str, str], config["contracts"])
        candidate_contracts = dict(contracts)
        candidate_contracts["tcn_training_contract_id"] = (
            "top50-relative10-listwise-consensus-student-v43"
        )
        candidate_predictions = _collect_tcn_predictions(
            features,
            labels,
            split,
            trial,
            tuning.best_states,
            seed=7,
            model_name="listwise_student_tcn",
            lookup=_label_lookup(labels),
            contracts=candidate_contracts,
        )
        v42_predictions = pd.read_parquet(source_paths["v42_predictions"])
        frozen = v42_predictions.copy()
        frozen["model"] = frozen["model"].replace(
            {
                "control_tcn": "control_tcn",
                "consensus_student_tcn": "pointwise_student_tcn",
            }
        )
        for column in (
            "prediction_contract_id",
            "target_contract_id",
            "evaluation_contract_id",
        ):
            frozen[column] = contracts[column]
            candidate_predictions[column] = contracts[column]
        frozen.loc[frozen["model"].eq("control_tcn"), "training_contract_id"] = (
            "top50-relative10-control-tcn-v43"
        )
        frozen.loc[
            frozen["model"].eq("pointwise_student_tcn"), "training_contract_id"
        ] = "top50-relative10-pointwise-student-v42"
        predictions = pd.concat([frozen, candidate_predictions], ignore_index=True)
        validate_prediction_contract(predictions, expected_models=3)

        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(cast(Any, config["top_fraction"]))
        )
        summary = summarize_task_aligned_metrics(metrics)
        control_comparison = compare_task_aligned_models(
            metrics,
            reference_model="control_tcn",
            candidate_model="listwise_student_tcn",
        )
        pointwise_comparison = compare_task_aligned_models(
            metrics,
            reference_model="pointwise_student_tcn",
            candidate_model="listwise_student_tcn",
        )
        bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="control_tcn",
            candidate_model="listwise_student_tcn",
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
        gate_metrics = metrics.loc[
            metrics["model"].isin(["control_tcn", "listwise_student_tcn"])
        ].replace({"listwise_student_tcn": "ema_tcn"})
        fold_deltas, horizon_deltas = _unit_deltas(gate_metrics)
        teacher = _teacher_ensemble(
            pd.read_parquet(source_paths["v40_predictions"])
        )
        fidelity_frame, fidelity = _teacher_fidelity(
            predictions,
            teacher,
            models=("pointwise_student_tcn", "listwise_student_tcn"),
        )

        v42_leaderboard = pd.read_parquet(source_paths["v42_leaderboard"])
        pointwise_id = str(config["pointwise_trial_id"])
        pointwise_timing = v42_leaderboard.loc[
            v42_leaderboard["trial_id"].astype(str).eq(pointwise_id)
        ].set_index("fold")
        candidate_timing = tuning.leaderboard.set_index("fold")
        if set(pointwise_timing.index.astype(int)) != set(range(5)):
            raise ContractError("v43 pointwise timing coverage drifted")
        model_step_retention = _geometric_mean(
            (
                candidate_timing["model_step_samples_per_second"]
                / pointwise_timing["model_step_samples_per_second"]
            ).to_numpy(dtype="float64")
        )
        complete_cycle_retention = _geometric_mean(
            (
                pointwise_timing["complete_cycle_seconds"]
                / candidate_timing["complete_cycle_seconds"]
            ).to_numpy(dtype="float64")
        )
        v42_timing = json.loads(source_paths["v42_timing"].read_text(encoding="utf-8"))
        implied_ratio = float(v42_timing["implied_tcn_lstm_model_step_ratio"]) * model_step_retention
        decision = decide_listwise_consensus_seed7_gate(
            control_comparison,
            pointwise_comparison,
            bootstrap,
            fold_deltas,
            horizon_deltas,
            teacher_fidelity_delta=float(fidelity["teacher_fidelity_delta"]),
            median_teacher_gradient_ratio=median_gradient_ratio,
            model_step_retention=model_step_retention,
            complete_cycle_retention=complete_cycle_retention,
            implied_tcn_lstm_model_step_ratio=implied_ratio,
            inference_forward_passes=1,
        )
        timing = {
            "model_step_retention_geomean": model_step_retention,
            "complete_cycle_retention_geomean": complete_cycle_retention,
            "implied_tcn_lstm_model_step_ratio": implied_ratio,
            "inference_forward_passes": 1,
        }

        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        tuning.leaderboard.to_parquet(temporary / "leaderboard.parquet", index=False)
        tuning.epoch_history.to_parquet(temporary / "epoch-history.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap.parquet", index=False)
        fold_deltas.to_parquet(temporary / "fold-deltas.parquet", index=False)
        horizon_deltas.to_parquet(temporary / "horizon-deltas.parquet", index=False)
        fidelity_frame.to_parquet(temporary / "teacher-fidelity.parquet", index=False)
        _write_json(temporary / "control-comparison.json", control_comparison)
        _write_json(temporary / "pointwise-comparison.json", pointwise_comparison)
        _write_json(temporary / "teacher-fidelity-summary.json", fidelity)
        _write_json(temporary / "timing.json", timing)
        _write_json(
            temporary / "teacher-target-audit.json",
            {
                **teacher_audit,
                "teacher_target_identity_recomputed": teacher_identity,
                "training_target_override": False,
                "validation_teacher_cells_exposed": 0,
                "median_teacher_gradient_ratio": median_gradient_ratio,
                "sealed_test_accessed": False,
            },
        )
        model_gate = {
            "status": decision.status,
            "admitted": decision.admitted,
            "blockers": list(decision.blockers),
            "evidence": decision.evidence,
            "phase_b_authorized": decision.admitted,
            "sealed_test_accessed": False,
        }
        _write_json(temporary / "model-gate.json", model_gate)
        _write_json(temporary / "config.resolved.json", config)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for key, state in tuning.best_states.items():
            torch.save(state, checkpoint_dir / f"seed-7-{key}.pt")
        report = (
            "# TCN v43 gradient-normalized listwise consensus seed7\n\n"
            f"- status: `{decision.status}`\n"
            f"- vs control RankIC delta: `{float(control_comparison['mean_rankic_delta']):+.6f}`\n"
            f"- vs control Top precision delta: `{float(control_comparison['mean_top_precision_delta']):+.6f}`\n"
            f"- vs pointwise RankIC delta: `{float(pointwise_comparison['mean_rankic_delta']):+.6f}`\n"
            f"- vs pointwise improved metrics: `{decision.evidence['pointwise_broad_metric_count']}/6`\n"
            f"- teacher fidelity delta: `{float(fidelity['teacher_fidelity_delta']):+.6f}`\n"
            f"- teacher gradient ratio: `{median_gradient_ratio:.4f}`\n"
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
        receipt: dict[str, object] = {
            "schema_version": "tcn-v43-listwise-consensus-seed7/v1",
            "run_id": str(config["run_id"]),
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "teacher_target_identity": teacher_identity,
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
            },
            "model_gate": model_gate,
            "timing": timing,
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
