"""Run the pre-registered v41 single-model EMA seed-7 holistic probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials  # noqa: E402
from skill_dl_tcn_shortterm.stability_ema import (  # noqa: E402
    state_dict_max_abs_error,
)
from skill_dl_tcn_shortterm.task_aligned_evaluation import (  # noqa: E402
    bootstrap_task_aligned_differences,
    compare_task_aligned_models,
    evaluate_task_aligned_predictions,
    summarize_task_aligned_metrics,
    validate_prediction_contract,
)
from skill_dl_tcn_shortterm.tuning import run_tcn_validation_sweep  # noqa: E402
from skill_dl_tcn_shortterm.v41_validation import (  # noqa: E402
    decide_ema_holistic_gate,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_relative_feature_validation import (  # noqa: E402
    _collect_tcn_predictions,
)
from run_tcn_task_aligned_evaluation import _label_lookup  # noqa: E402


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
    "min_model_step_retention": 0.90,
    "min_complete_cycle_retention": 0.85,
    "min_implied_tcn_lstm_model_step_ratio": 3.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            any(
                marker in str(key).lower()
                for marker in ("password", "token", "secret", "credential")
            )
            or _contains_secret_key(nested)
            for key, nested in value.items()
        )
    return isinstance(value, list) and any(_contains_secret_key(item) for item in value)


def _geometric_mean(values: np.ndarray) -> float:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ContractError("v41 speed ratios must be finite and non-empty")
    if bool((values <= 0.0).any()):
        raise ContractError("v41 speed ratios must be positive")
    return float(np.exp(np.log(values).mean()))


def _trajectory_audit(
    epoch_states: dict[str, dict[str, torch.Tensor]],
    *,
    control_trial_id: str,
    candidate_trial_id: str,
    folds: tuple[int, ...],
    max_epochs: int,
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    for fold in folds:
        for epoch in range(max_epochs + 1):
            control_key = f"{control_trial_id}-fold-{fold}-epoch-{epoch}"
            candidate_key = f"{candidate_trial_id}-fold-{fold}-epoch-{epoch}"
            if control_key not in epoch_states or candidate_key not in epoch_states:
                raise ContractError("v41 raw epoch trajectory coverage is incomplete")
            error = state_dict_max_abs_error(
                epoch_states[control_key], epoch_states[candidate_key]
            )
            rows.append({"fold": fold, "epoch": epoch, "max_abs_error": error})
    audit = pd.DataFrame(rows)
    return audit, float(audit["max_abs_error"].max())


def _unit_deltas(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = (
        metrics.groupby(["model", "fold", "horizon"], as_index=False, observed=True)[
            "rankic"
        ]
        .mean()
        .pivot(index=["fold", "horizon"], columns="model", values="rankic")
    )
    if set(means.columns) != {"control_tcn", "ema_tcn"} or means.isna().any().any():
        raise ContractError("v41 paired unit coverage drifted")
    paired = means.reset_index()[["fold", "horizon"]].copy()
    paired["rankic_delta"] = (
        means["ema_tcn"] - means["control_tcn"]
    ).to_numpy()
    fold = paired.groupby("fold", as_index=False, observed=True).agg(
        rankic_delta=("rankic_delta", "mean")
    )
    horizon = paired.groupby("horizon", as_index=False, observed=True).agg(
        rankic_delta=("rankic_delta", "mean")
    )
    return fold, horizon


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
            raise ContractError("v41 seed7 run refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("protocol_version") != "v41-phase-a":
            raise ContractError("v41 phase-A config identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v41 config contains a secret-like key")
        if config.get("gates") != EXPECTED_GATES:
            raise ContractError("v41 holistic gates drifted")
        if (
            config.get("precision") != "float32"
            or int(cast(Any, config["torch_threads"])) != 8
            or int(cast(Any, config["num_workers"])) != 0
            or tuple(cast(list[object], config["seeds"])) != (7,)
            or tuple(cast(list[object], config["folds"])) != (0, 1, 2, 3, 4)
        ):
            raise ContractError("v41 seed7 execution budget drifted")

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
            "v40_timing_summary": v40 / "timing-summary.json",
            "v40_receipt": v40 / "receipt.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v41 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v41 source SHA-256 identity drifted")
        v40_receipt = json.loads(source_paths["v40_receipt"].read_text(encoding="utf-8"))
        if (
            not isinstance(v40_receipt, dict)
            or v40_receipt.get("sealed_test_accessed") is not False
        ):
            raise ContractError("v41 v40 source is not ordinary validation")
        v40_timing = json.loads(
            source_paths["v40_timing_summary"].read_text(encoding="utf-8")
        )
        v40_speed_ratio = float(v40_timing["model_step_speed_ratio_geomean"])

        features = np.load(
            source_paths["relative_features"], mmap_mode="r", allow_pickle=False
        )
        if features.ndim != 3 or features.shape[1:] != (10, 480):
            raise ContractError("v41 relative10 tensor shape drifted")
        window_index = pd.read_parquet(source_paths["relative_window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v41 rejects sealed split rows")
        stages = {str(stage) for stage in raw_split["stage"].tolist()}
        if unknown := sorted(stages - {"train", "validation", "purged"}):
            raise ContractError("v41 split has forbidden stages: " + ", ".join(unknown))
        folds = (0, 1, 2, 3, 4)
        split = raw_split.loc[
            raw_split["fold"].astype(int).isin(folds)
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 2:
            raise ContractError("v41 requires exactly control and EMA trials")
        by_id = {trial.trial_id: trial for trial in trials}
        control_id = str(config["control_trial_id"])
        candidate_id = str(config["candidate_trial_id"])
        if set(by_id) != {control_id, candidate_id}:
            raise ContractError("v41 trial identities drifted")
        control = by_id[control_id]
        candidate = by_id[candidate_id]
        if control.ema_decay is not None or candidate.ema_decay != 0.99:
            raise ContractError("v41 fixes control raw parameters and EMA decay 0.99")
        control_contract = asdict(control)
        candidate_contract = asdict(candidate)
        for contract in (control_contract, candidate_contract):
            contract.pop("trial_id")
            contract.pop("ema_decay")
        if control_contract != candidate_contract:
            raise ContractError("v41 trials differ by more than EMA")

        max_epochs = int(cast(Any, config["max_epochs"]))
        temporary.mkdir(parents=True)
        tuning = run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split,
            trials=trials,
            seed=7,
            max_epochs=max_epochs,
            patience=int(cast(Any, config["patience"])),
            min_delta=float(cast(Any, config["min_delta"])),
            checkpoint_min_delta=float(cast(Any, config["checkpoint_min_delta"])),
            torch_threads=int(cast(Any, config["torch_threads"])),
            protocol_identities={
                "data": observed_hashes["relative_features"],
                "fold_manifest": observed_hashes["split_manifest"],
                "evaluation": observed_hashes["labels"],
            },
            capture_epoch_states=True,
            disable_early_stopping=True,
        )
        trajectory, raw_state_drift_max = _trajectory_audit(
            tuning.epoch_states,
            control_trial_id=control_id,
            candidate_trial_id=candidate_id,
            folds=folds,
            max_epochs=max_epochs,
        )
        if raw_state_drift_max != 0.0:
            raise ContractError("v41 EMA changed the raw training trajectory")

        lookup = _label_lookup(labels)
        contracts = cast(dict[str, str], config["contracts"])
        prediction_frames = []
        for trial, model_name in ((control, "control_tcn"), (candidate, "ema_tcn")):
            model_contracts = dict(contracts)
            model_contracts["tcn_training_contract_id"] = (
                f"top50-relative10-{model_name}-smooth-l1-v41"
            )
            prediction_frames.append(
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
        predictions = pd.concat(prediction_frames, ignore_index=True)
        validate_prediction_contract(predictions, expected_models=2)
        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(cast(Any, config["top_fraction"]))
        )
        summary = summarize_task_aligned_metrics(metrics)
        comparison = compare_task_aligned_models(
            metrics, reference_model="control_tcn", candidate_model="ema_tcn"
        )
        bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="control_tcn",
            candidate_model="ema_tcn",
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
        fold_deltas, horizon_deltas = _unit_deltas(metrics)

        leaderboard = tuning.leaderboard.copy()
        control_timing = leaderboard.loc[leaderboard["trial_id"].eq(control_id)].set_index("fold")
        candidate_timing = leaderboard.loc[leaderboard["trial_id"].eq(candidate_id)].set_index("fold")
        if not control_timing.index.equals(candidate_timing.index):
            raise ContractError("v41 timing fold coverage drifted")
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
        implied_ratio = v40_speed_ratio * model_step_retention
        decision = decide_ema_holistic_gate(
            comparison,
            bootstrap,
            fold_deltas,
            horizon_deltas,
            raw_state_drift_max=raw_state_drift_max,
            model_step_retention=model_step_retention,
            complete_cycle_retention=complete_cycle_retention,
            implied_tcn_lstm_model_step_ratio=implied_ratio,
        )
        timing = {
            "model_step_retention_geomean": model_step_retention,
            "complete_cycle_retention_geomean": complete_cycle_retention,
            "v40_tcn_lstm_model_step_ratio": v40_speed_ratio,
            "implied_tcn_lstm_model_step_ratio": implied_ratio,
            "inference_forward_passes": 1,
        }
        trajectory_summary = {
            "raw_state_drift_max": raw_state_drift_max,
            "audited_units": len(trajectory),
            "fold_count": len(folds),
            "epoch_count_including_initial": max_epochs + 1,
        }

        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        tuning.leaderboard.to_parquet(temporary / "leaderboard.parquet", index=False)
        tuning.epoch_history.to_parquet(temporary / "epoch-history.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap.parquet", index=False)
        fold_deltas.to_parquet(temporary / "fold-deltas.parquet", index=False)
        horizon_deltas.to_parquet(temporary / "horizon-summary.parquet", index=False)
        trajectory.to_parquet(temporary / "trajectory-audit.parquet", index=False)
        _write_json(temporary / "comparison.json", comparison)
        _write_json(temporary / "trajectory-audit.json", trajectory_summary)
        _write_json(temporary / "timing.json", timing)
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
        checkpoints = temporary / "checkpoints"
        checkpoints.mkdir()
        for key, state in tuning.best_states.items():
            torch.save(state, checkpoints / f"{key}.pt")
        report = (
            "# TCN v41 single-model EMA seed7 result\n\n"
            f"- status: `{decision.status}`\n"
            f"- mean RankIC delta: `{float(comparison['mean_rankic_delta']):+.6f}`\n"
            f"- broad improved metrics: `{decision.evidence['broad_metric_count']}/6`\n"
            f"- positive folds: `{decision.evidence['positive_folds']}/5`\n"
            f"- positive horizons: `{decision.evidence['positive_horizons']}/4`\n"
            f"- model-step retention: `{model_step_retention:.4f}`\n"
            f"- implied TCN/LSTM model-step ratio: `{implied_ratio:.4f}x`\n"
            "- sealed_test_accessed: `false`\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")

        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-v41-single-model-ema-seed7/v1",
            "run_id": str(config["run_id"]),
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
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
