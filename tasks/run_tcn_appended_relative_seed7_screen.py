"""Run the immutable v38 seed-7 append-only relative-sequence TCN screen."""

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
    APPENDED_SEQUENCE_FEATURE_VERSION,
)
from skill_dl_tcn_shortterm.relative_validation import (  # noqa: E402
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
        for key, nested in value.items():
            if any(
                marker in str(key).lower()
                for marker in ("password", "token", "secret", "credential")
            ):
                return True
            if _contains_secret_key(nested):
                return True
    return isinstance(value, list) and any(_contains_secret_key(item) for item in value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the v38 seed-7 append-only relative sequence TCN screen"
    )
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--candidate-feature-dir", required=True, type=Path)
    parser.add_argument("--v37-parent-artifact", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v38 seed-7 screen refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("protocol_version") != "v38-phase-a":
            raise ContractError("v38 phase-A config identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v38 config contains a secret-like key")
        if config.get("precision") != "float32" or int(
            cast(Any, config["num_workers"])
        ) != 0:
            raise ContractError("v38 requires float32 and num_workers=0")
        if tuple(cast(list[object], config["seeds"])) != (7,) or tuple(
            cast(list[object], config["folds"])
        ) != (0, 1, 2, 3, 4):
            raise ContractError("v38 phase A requires seed 7 and folds 0..4")

        base = arguments.base_run_dir.resolve()
        candidate = arguments.candidate_feature_dir.resolve()
        parent = arguments.v37_parent_artifact.resolve()
        source_paths = {
            "candidate_features": candidate / "feature-windows.npy",
            "candidate_window_index": candidate / "window-index.parquet",
            "candidate_manifest": candidate / "manifest.json",
            "candidate_receipt": candidate / "receipt.json",
            "labels": base / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "parent_receipt": parent / "receipt.json",
            "parent_predictions": parent / "predictions.parquet",
            "parent_leaderboard": parent / "tcn-leaderboard.parquet",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v38 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v38 source SHA-256 identity drifted")

        candidate_manifest = cast(
            dict[str, object],
            json.loads(source_paths["candidate_manifest"].read_text(encoding="utf-8")),
        )
        candidate_receipt = cast(
            dict[str, object],
            json.loads(source_paths["candidate_receipt"].read_text(encoding="utf-8")),
        )
        if candidate_manifest.get("feature_version") != APPENDED_SEQUENCE_FEATURE_VERSION:
            raise ContractError("v38 candidate feature version drifted")
        if candidate_manifest.get("sealed_test_accessed") is not False or candidate_receipt.get(
            "sealed_test_accessed"
        ) is not False:
            raise ContractError("v38 candidate feature source is not fail-closed")

        parent_receipt = cast(
            dict[str, object],
            json.loads(source_paths["parent_receipt"].read_text(encoding="utf-8")),
        )
        if parent_receipt.get("receipt_id") != config["v37_parent_receipt_id"]:
            raise ContractError("v38 v37 parent receipt identity drifted")
        if parent_receipt.get("sealed_test_accessed") is not False:
            raise ContractError("v38 v37 parent is not sealed-test fail-closed")
        parent_outputs = parent_receipt.get("outputs")
        if not isinstance(parent_outputs, dict):
            raise ContractError("v38 v37 parent outputs are missing")
        for filename, source_name in (
            ("predictions.parquet", "parent_predictions"),
            ("tcn-leaderboard.parquet", "parent_leaderboard"),
        ):
            if parent_outputs.get(filename) != observed_hashes[source_name]:
                raise ContractError(f"v38 v37 parent output drifted: {filename}")

        features = np.load(
            source_paths["candidate_features"], mmap_mode="r", allow_pickle=False
        )
        window_index = pd.read_parquet(source_paths["candidate_window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if features.ndim != 3 or features.shape[1:] != (10, 480):
            raise ContractError("v38 candidate tensor shape drifted")
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v38 rejects sealed split rows")
        observed_stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(observed_stages - {"train", "validation", "purged"}):
            raise ContractError("v38 split has forbidden stages: " + ", ".join(unknown))
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 1:
            raise ContractError("v38 fixes exactly one TCN trial")
        trial = trials[0]
        if trial.model_kind != "dynamic_horizon_skip" or trial.strategy != "smooth_l1":
            raise ContractError("v38 TCN isolation contract drifted")
        seed = 7
        tuning = run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split_manifest,
            trials=trials,
            seed=seed,
            max_epochs=int(cast(Any, config["max_epochs"])),
            patience=int(cast(Any, config["patience"])),
            min_delta=float(cast(Any, config["min_delta"])),
            checkpoint_min_delta=float(cast(Any, config["checkpoint_min_delta"])),
            torch_threads=int(cast(Any, config["torch_threads"])),
            protocol_identities={
                "data": observed_hashes["candidate_features"],
                "fold_manifest": observed_hashes["split_manifest"],
                "evaluation": observed_hashes["labels"],
            },
            capture_epoch_states=True,
            disable_early_stopping=True,
        )
        contracts = cast(dict[str, str], config["contracts"])
        candidate_predictions = _collect_tcn_predictions(
            features,
            labels,
            split_manifest,
            trial,
            tuning.best_states,
            seed=seed,
            model_name="relative_tcn",
            lookup=_label_lookup(labels),
            contracts=contracts,
        )
        parent_predictions = pd.read_parquet(source_paths["parent_predictions"])
        base_predictions = parent_predictions.loc[
            parent_predictions["model"].astype(str).eq("base_tcn")
            & parent_predictions["seed"].astype(int).eq(seed)
        ].copy()
        predictions = pd.concat([base_predictions, candidate_predictions], ignore_index=True)
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
        parent_leaderboard = pd.read_parquet(source_paths["parent_leaderboard"])
        base_leaderboard = parent_leaderboard.loc[
            parent_leaderboard["variant"].astype(str).eq("base")
            & parent_leaderboard["seed"].astype(int).eq(seed)
        ].copy()
        candidate_leaderboard = tuning.leaderboard.copy()
        candidate_leaderboard["variant"] = "relative"
        leaderboard = pd.concat([base_leaderboard, candidate_leaderboard], ignore_index=True)
        base_speed = float(base_leaderboard["samples_per_second"].median())
        candidate_speed = float(candidate_leaderboard["samples_per_second"].median())
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
            admitted_status="append_relative_sequence_seed7_admitted_v38",
            rejected_status="stop_append_relative_sequence_seed7_v38",
        )
        selection: dict[str, object] = {
            "status": decision.status,
            "admitted": decision.admitted,
            "blockers": list(decision.blockers),
            "evidence": decision.evidence,
            "comparison": comparison,
            "confirmation_authorized": decision.admitted,
            "next_step": (
                "run_v38_multiseed_lstm_confirmation"
                if decision.admitted
                else "stop_v38_and_repair_top50_or_build_post_encoder_context"
            ),
            "sealed_test_accessed": False,
            "sealed_test_authorized": False,
        }

        temporary.mkdir(parents=True)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for key, state in tuning.best_states.items():
            torch.save(state, checkpoint_dir / f"seed-7-{key}.pt")
        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap-summary.parquet", index=False)
        decision.unit_deltas.to_parquet(temporary / "unit-deltas.parquet", index=False)
        tuning.epoch_history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        _write_json(temporary / "comparison.json", comparison)
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        report = "\n".join(
            [
                "# TCN 追加式相对时序特征 seed-7 屏幕 v38",
                "",
                f"- 状态：`{decision.status}`",
                f"- 准入：`{decision.admitted}`",
                f"- 阻塞项：`{', '.join(decision.blockers) or 'none'}`",
                f"- RankIC delta：`{float(decision.evidence['mean_rankic_delta']):+.6f}`",
                f"- 正 folds：`{int(decision.evidence['positive_units'])}/5`",
                f"- Top precision delta：`{float(decision.evidence['mean_top_precision_delta']):+.6f}`",
                f"- NDCG delta：`{float(decision.evidence['mean_ndcg_delta']):+.6f}`",
                f"- Top return delta：`{float(decision.evidence['mean_top_return_delta']):+.6f}`",
                f"- Turnover delta：`{float(decision.evidence['mean_turnover_delta']):+.6f}`",
                f"- TCN speed retention：`{float(decision.evidence['tcn_speed_retention']):.4f}`",
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
            "schema_version": "tcn-appended-relative-seed7-screen-v38/v1",
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
