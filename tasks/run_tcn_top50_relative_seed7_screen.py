"""Run the immutable v39 top50 base8 versus relative10 TCN screen."""

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
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials  # noqa: E402
from skill_dl_tcn_shortterm.relative_features import (  # noqa: E402
    TOP50_APPENDED_SEQUENCE_FEATURE_VERSION,
)
from skill_dl_tcn_shortterm.relative_validation import (  # noqa: E402
    audit_validation_effective_breadth,
    decide_relative_feature_gate,
)
from skill_dl_tcn_shortterm.task_aligned_evaluation import (  # noqa: E402
    bootstrap_task_aligned_differences,
    compare_task_aligned_models,
    evaluate_task_aligned_predictions,
    summarize_task_aligned_metrics,
    validate_prediction_contract,
)
from skill_dl_tcn_shortterm.tuning import run_tcn_validation_sweep  # noqa: E402
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_relative_feature_validation import (  # noqa: E402
    _collect_tcn_predictions,
)
from run_tcn_task_aligned_evaluation import _label_lookup  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the v39 top50 relative-sequence seed-7 TCN screen"
    )
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--candidate-feature-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v39 seed-7 screen refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("protocol_version") != "v39-phase-a":
            raise ContractError("v39 phase-A config identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v39 config contains a secret-like key")
        if config.get("precision") != "float32" or int(
            cast(Any, config["num_workers"])
        ) != 0:
            raise ContractError("v39 requires float32 and num_workers=0")
        if tuple(cast(list[object], config["seeds"])) != (7,) or tuple(
            cast(list[object], config["folds"])
        ) != (0, 1, 2, 3, 4):
            raise ContractError("v39 phase A requires seed 7 and folds 0..4")

        base = arguments.base_run_dir.resolve()
        candidate = arguments.candidate_feature_dir.resolve()
        source_paths = {
            "base_features": base / "feature-windows.npy",
            "base_window_index": base / "window-index.parquet",
            "candidate_features": candidate / "feature-windows.npy",
            "candidate_window_index": candidate / "window-index.parquet",
            "candidate_manifest": candidate / "manifest.json",
            "candidate_receipt": candidate / "receipt.json",
            "labels": base / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v39 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v39 source SHA-256 identity drifted")

        feature_manifest = cast(
            dict[str, object],
            json.loads(source_paths["candidate_manifest"].read_text(encoding="utf-8")),
        )
        feature_receipt = cast(
            dict[str, object],
            json.loads(source_paths["candidate_receipt"].read_text(encoding="utf-8")),
        )
        if feature_manifest.get("feature_version") != (
            TOP50_APPENDED_SEQUENCE_FEATURE_VERSION
        ):
            raise ContractError("v39 candidate feature version drifted")
        if feature_manifest.get("sealed_test_accessed") is not False or (
            feature_receipt.get("sealed_test_accessed") is not False
        ):
            raise ContractError("v39 candidate feature source is not fail-closed")

        base_features = np.load(
            source_paths["base_features"], mmap_mode="r", allow_pickle=False
        )
        candidate_features = np.load(
            source_paths["candidate_features"], mmap_mode="r", allow_pickle=False
        )
        if base_features.ndim != 3 or base_features.shape[1:] != (8, 480):
            raise ContractError("v39 base tensor shape drifted")
        if candidate_features.shape != (
            base_features.shape[0],
            10,
            base_features.shape[2],
        ):
            raise ContractError("v39 candidate tensor shape drifted")
        base_index = pd.read_parquet(source_paths["base_window_index"])
        candidate_index = pd.read_parquet(source_paths["candidate_window_index"])
        identity_columns = ["sample_position", "sample_id", "instrument_id", "signal_date"]
        if any(
            not np.array_equal(
                base_index[column].astype(str).to_numpy(),
                candidate_index[column].astype(str).to_numpy(),
            )
            for column in identity_columns
        ):
            raise ContractError("v39 base and candidate sample identities drifted")

        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v39 rejects sealed split rows")
        observed_stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(observed_stages - {"train", "validation", "purged"}):
            raise ContractError("v39 split has forbidden stages: " + ", ".join(unknown))
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()
        breadth = audit_validation_effective_breadth(
            labels,
            raw_split,
            folds=range(5),
            top_fraction=float(cast(Any, config["top_fraction"])),
            min_top_count=int(cast(Any, config["min_top_count"])),
        )
        if breadth["effective_breadth_gate_passed"] is not True:
            raise ContractError("v39 validation effective breadth gate failed")

        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 1:
            raise ContractError("v39 fixes exactly one TCN trial")
        trial = trials[0]
        if trial.model_kind != "dynamic_horizon_skip" or trial.strategy != "smooth_l1":
            raise ContractError("v39 TCN isolation contract drifted")
        contracts = cast(dict[str, str], config["contracts"])
        lookup = _label_lookup(labels)
        temporary.mkdir(parents=True)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()

        prediction_frames: list[pd.DataFrame] = []
        history_frames: list[pd.DataFrame] = []
        leaderboard_frames: list[pd.DataFrame] = []
        variants = {
            "base": (base_features, base_index, observed_hashes["base_features"]),
            "relative": (
                candidate_features,
                candidate_index,
                observed_hashes["candidate_features"],
            ),
        }
        for variant, (features, _window_index, data_identity) in variants.items():
            variant_contracts = dict(contracts)
            variant_contracts["tcn_training_contract_id"] = (
                f"top50-{variant}-dynamic-horizon-skip-smooth-l1-v39"
            )
            tuning = run_tcn_validation_sweep(
                features,
                _window_index,
                labels,
                split_manifest,
                trials=trials,
                seed=7,
                max_epochs=int(cast(Any, config["max_epochs"])),
                patience=int(cast(Any, config["patience"])),
                min_delta=float(cast(Any, config["min_delta"])),
                checkpoint_min_delta=float(cast(Any, config["checkpoint_min_delta"])),
                torch_threads=int(cast(Any, config["torch_threads"])),
                protocol_identities={
                    "data": data_identity,
                    "fold_manifest": observed_hashes["split_manifest"],
                    "evaluation": observed_hashes["labels"],
                },
                capture_epoch_states=True,
                disable_early_stopping=True,
            )
            history = tuning.epoch_history.copy()
            history["variant"] = variant
            history_frames.append(history)
            leaderboard = tuning.leaderboard.copy()
            leaderboard["variant"] = variant
            leaderboard_frames.append(leaderboard)
            prediction_frames.append(
                _collect_tcn_predictions(
                    features,
                    labels,
                    split_manifest,
                    trial,
                    tuning.best_states,
                    seed=7,
                    model_name=f"{variant}_tcn",
                    lookup=lookup,
                    contracts=variant_contracts,
                )
            )
            for state_key, state in tuning.best_states.items():
                torch.save(state, checkpoint_dir / f"{variant}-seed-7-{state_key}.pt")

        predictions = pd.concat(prediction_frames, ignore_index=True)
        validate_prediction_contract(predictions, expected_models=2)
        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(cast(Any, config["top_fraction"]))
        )
        summary = summarize_task_aligned_metrics(metrics)
        comparison = compare_task_aligned_models(
            metrics, reference_model="base_tcn", candidate_model="relative_tcn"
        )
        bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="base_tcn",
            candidate_model="relative_tcn",
            metric_columns=(
                "rankic",
                "top_return",
                "top_excess_return",
                "top_precision",
                "ndcg_at_top",
                "top_turnover",
            ),
            seed=int(cast(Any, config["bootstrap_seed"])),
            draws=int(cast(Any, config["bootstrap_draws"])),
        )
        leaderboard = pd.concat(leaderboard_frames, ignore_index=True)
        history = pd.concat(history_frames, ignore_index=True)
        base_speed = float(
            leaderboard.loc[leaderboard["variant"].eq("base"), "samples_per_second"].median()
        )
        candidate_speed = float(
            leaderboard.loc[
                leaderboard["variant"].eq("relative"), "samples_per_second"
            ].median()
        )
        decision = decide_relative_feature_gate(
            leaderboard,
            comparison,
            bootstrap,
            seeds=(7,),
            folds=range(5),
            base_variant="base",
            candidate_variant="relative",
            base_median_samples_per_second=base_speed,
            candidate_median_samples_per_second=candidate_speed,
            gates=cast(dict[str, float | int], config["gates"]),
            admitted_status="top50_relative_sequence_seed7_admitted_v39",
            rejected_status="stop_top50_relative_sequence_seed7_v39",
        )
        selection: dict[str, object] = {
            "status": decision.status,
            "admitted": decision.admitted,
            "blockers": list(decision.blockers),
            "evidence": decision.evidence,
            "comparison": comparison,
            "effective_breadth": breadth,
            "phase_b_authorized": decision.admitted,
            "next_step": (
                "run_v39_multiseed_lstm_confirmation"
                if decision.admitted
                else "stop_v39_relative2_or_preregister_post_encoder_context"
            ),
            "sealed_test_accessed": False,
            "sealed_test_authorized": False,
        }

        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap-summary.parquet", index=False)
        decision.unit_deltas.to_parquet(temporary / "unit-deltas.parquet", index=False)
        history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        _write_json(temporary / "comparison.json", comparison)
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        report = "\n".join(
            [
                "# TCN top50 相对序列 seed-7 屏幕 v39",
                "",
                f"- 状态：`{decision.status}`",
                f"- 准入：`{decision.admitted}`",
                f"- 阻塞项：`{', '.join(decision.blockers) or 'none'}`",
                f"- RankIC delta：`{float(decision.evidence['mean_rankic_delta']):+.6f}`",
                f"- 正 folds：`{int(decision.evidence['positive_units'])}/5`",
                f"- Top precision delta：`{float(decision.evidence['mean_top_precision_delta']):+.6f}`",
                f"- NDCG delta：`{float(decision.evidence['mean_ndcg_delta']):+.6f}`",
                f"- Top return delta：`{float(decision.evidence['mean_top_return_delta']):+.6f}`",
                f"- TCN speed retention：`{float(decision.evidence['tcn_speed_retention']):.4f}`",
                f"- 最小有效横截面/Top count：`{breadth['minimum_member_count']}/{breadth['minimum_top_count']}`",
                "- sealed test：未访问。",
                "",
            ]
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-top50-relative-seed7-screen-v39/v1",
            "run_id": str(config["run_id"]),
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "source_config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
                "storage": "read_only_memmap",
            },
            "selection": selection,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "result": decision.status,
            "output_dir": str(output_dir),
            "receipt_id": receipt["receipt_id"],
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
