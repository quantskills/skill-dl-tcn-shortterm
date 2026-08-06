"""Run the immutable ordinary-validation v37 relative-feature experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.neural import HORIZONS  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.relative_features import FEATURE_VERSION  # noqa: E402
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
from skill_dl_tcn_shortterm.training_data import (  # noqa: E402
    LazyWindowDataset,
    build_fold_protocols,
)
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    TCNTuningTrial,
    _predict_tcn_trial,
    build_tcn_trial_model,
    run_tcn_validation_sweep,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_task_aligned_evaluation import (  # noqa: E402
    _label_lookup,
    _prediction_rows,
    _train_lstm,
)


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


def _collect_tcn_predictions(
    features: np.ndarray,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    trial: TCNTuningTrial,
    best_states: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    seed: int,
    model_name: str,
    lookup: dict[tuple[int, int], tuple[object, ...]],
    contracts: dict[str, str],
) -> pd.DataFrame:
    protocols = build_fold_protocols(features, split_manifest)
    targets = np.zeros((len(features), len(HORIZONS)), dtype="float32")
    masks = np.ones_like(targets, dtype="bool")
    rows: list[dict[str, object]] = []
    for protocol in protocols:
        state_key = f"{trial.trial_id}-fold-{protocol.fold}"
        state = best_states.get(state_key)
        if state is None:
            raise ContractError(f"v37 TCN checkpoint state missing: {state_key}")
        model = build_tcn_trial_model(
            trial,
            feature_count=int(features.shape[1]),
            input_steps=int(features.shape[2]),
        )
        model.load_state_dict(state, strict=True)
        dataset = LazyWindowDataset(
            features,
            protocol.validation_positions,
            targets,
            masks,
            protocol.feature_mean,
            protocol.feature_std,
        )
        scores, positions = _predict_tcn_trial(
            model, dataset, batch_size=trial.batch_size
        )
        rows.extend(
            _prediction_rows(
                scores,
                positions,
                model=model_name,
                seed=seed,
                fold=protocol.fold,
                lookup=lookup,
                contracts=contracts,
                training_contract_id=contracts["tcn_training_contract_id"],
            )
        )
    return pd.DataFrame(rows)


def _comparison_bootstrap(
    metrics: pd.DataFrame,
    *,
    reference_model: str,
    candidate_model: str,
    scope: str,
    seed: int,
    draws: int,
) -> pd.DataFrame:
    result = bootstrap_task_aligned_differences(
        metrics,
        reference_model=reference_model,
        candidate_model=candidate_model,
        metric_columns=(
            "rankic",
            "top_return",
            "top_excess_return",
            "top_precision",
            "ndcg_at_top",
            "top_turnover",
        ),
        seed=seed,
        draws=draws,
    )
    result.insert(0, "scope", scope)
    return result


def _report(
    selection: Mapping[str, object],
    summary: pd.DataFrame,
    comparisons: Mapping[str, Mapping[str, object]],
    speed: Mapping[str, object],
    top50: Mapping[str, object],
) -> str:
    summary_lines = [
        (
            f"| {row.model} | {float(cast(Any, row.mean_rankic)):+.6f} | "
            f"{float(cast(Any, row.mean_top_return)):+.6f} | "
            f"{float(cast(Any, row.mean_top_precision)):.4f} | "
            f"{float(cast(Any, row.mean_ndcg_at_top)):.4f} | "
            f"{float(cast(Any, row.mean_top_turnover)):.4f} |"
        )
        for row in summary.itertuples(index=False)
    ]
    comparison_lines = [
        (
            f"| {name} | {float(cast(Any, value['mean_rankic_delta'])):+.6f} | "
            f"{float(cast(Any, value['mean_top_return_delta'])):+.6f} | "
            f"{float(cast(Any, value['mean_top_precision_delta'])):+.6f} | "
            f"{float(cast(Any, value['mean_ndcg_at_top_delta'])):+.6f} | "
            f"{value['winner_consensus']} |"
        )
        for name, value in comparisons.items()
    ]
    return "\n".join(
        [
            "# TCN 因果相对特征普通验证 v37",
            "",
            f"- 决策：`{selection['status']}`",
            f"- 准入：`{selection['admitted']}`",
            f"- 阻塞项：`{', '.join(cast(list[str], selection['blockers'])) or 'none'}`",
            "- sealed test：未访问；旧 v36 sealed 不参与本轮特征选择。",
            f"- top50：`{top50['status']}`，缺失 PIT 状态键 "
            f"`{top50['missing_state_key_count']}`。",
            "",
            "## 四组合多指标",
            "",
            "| 模型 | RankIC | Top return | Top precision | NDCG@Top | Turnover |",
            "|---|---:|---:|---:|---:|---:|",
            *summary_lines,
            "",
            "## 配对差（候选减参考）",
            "",
            "| 比较 | RankIC | Top return | Top precision | NDCG@Top | 共识 |",
            "|---|---:|---:|---:|---:|---|",
            *comparison_lines,
            "",
            "## 速度",
            "",
            f"- base TCN median samples/s：`{float(cast(Any, speed['base_tcn_median_samples_per_second'])):.2f}`",
            f"- relative TCN median samples/s：`{float(cast(Any, speed['relative_tcn_median_samples_per_second'])):.2f}`",
            f"- relative/base TCN 吞吐保留：`{float(cast(Any, speed['tcn_speed_retention'])):.4f}`",
            f"- relative TCN/LSTM 端到端几何平均速度比：`{float(cast(Any, speed['relative_tcn_over_lstm_geomean'])):.4f}x`",
            "",
            "结论只适用于固定 top20 ordinary validation。若未准入，下一步应补齐 "
            "top50 PIT 状态或研究显式 date-level cross-sectional adapter；不得回看旧 sealed 调参。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed base/relative TCN and LSTM ordinary validation"
    )
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--relative-feature-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v37 validation refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("protocol_version") != "v37":
            raise ContractError("v37 config identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v37 config contains a secret-like key")
        if config.get("precision") != "float32" or int(
            cast(Any, config["num_workers"])
        ) != 0:
            raise ContractError("v37 requires float32 and num_workers=0")
        seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["seeds"])
        )
        folds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["folds"])
        )
        if seeds != (7, 17, 27) or folds != (0, 1, 2, 3, 4):
            raise ContractError("v37 requires seeds 7/17/27 and folds 0..4")

        base = arguments.base_run_dir.resolve()
        relative = arguments.relative_feature_dir.resolve()
        source_paths = {
            "base_features": base / "feature-windows.npy",
            "base_window_index": base / "window-index.parquet",
            "relative_features": relative / "feature-windows.npy",
            "relative_window_index": relative / "window-index.parquet",
            "labels": base / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "relative_manifest": relative / "manifest.json",
            "top50_readiness": relative / "top50-readiness.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v37 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v37 source SHA-256 identity drifted")

        feature_manifest = cast(
            dict[str, object],
            json.loads(source_paths["relative_manifest"].read_text(encoding="utf-8")),
        )
        if feature_manifest.get("feature_version") != FEATURE_VERSION or feature_manifest.get(
            "sealed_test_accessed"
        ) is not False:
            raise ContractError("v37 relative feature manifest is not fail-closed")
        top50 = cast(
            dict[str, object],
            json.loads(source_paths["top50_readiness"].read_text(encoding="utf-8")),
        )
        if top50.get("ready") is not False or top50.get("status") != (
            "blocked_missing_pit_state"
        ):
            raise ContractError("v37 expects the preregistered top50 fail-closed gate")

        base_features = np.load(
            source_paths["base_features"], mmap_mode="r", allow_pickle=False
        )
        relative_features = np.load(
            source_paths["relative_features"], mmap_mode="r", allow_pickle=False
        )
        base_index = pd.read_parquet(source_paths["base_window_index"])
        relative_index = pd.read_parquet(source_paths["relative_window_index"])
        identity_columns = ["sample_position", "sample_id", "instrument_id", "signal_date"]
        if any(
            not np.array_equal(
                base_index[column].astype(str).to_numpy(),
                relative_index[column].astype(str).to_numpy(),
            )
            for column in identity_columns
        ):
            raise ContractError("v37 base and relative sample identities drifted")
        if base_features.shape[0] != relative_features.shape[0] or (
            base_features.shape[2] != relative_features.shape[2]
        ):
            raise ContractError("v37 base and relative tensor coverage drifted")
        if base_features.shape[1] != 8 or relative_features.shape[1] != 13:
            raise ContractError("v37 feature channel counts drifted")

        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v37 rejects sealed split rows")
        allowed_stages = {"train", "validation", "purged"}
        observed_stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(observed_stages - allowed_stages):
            raise ContractError("v37 split has forbidden stages: " + ", ".join(unknown))
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(folds)
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()
        if set(split_manifest["fold"].astype(int)) != set(folds):
            raise ContractError("v37 split fold coverage drifted")

        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 1:
            raise ContractError("v37 fixes exactly one TCN architecture")
        trial = trials[0]
        if trial.strategy != "smooth_l1" or trial.model_kind != "dynamic_horizon_skip":
            raise ContractError("v37 TCN representation-isolation contract drifted")
        lookup = _label_lookup(labels)
        contracts = cast(dict[str, str], config["contracts"])
        temporary.mkdir(parents=True)
        checkpoint_root = temporary / "checkpoints"
        checkpoint_root.mkdir()

        prediction_frames: list[pd.DataFrame] = []
        tcn_history_frames: list[pd.DataFrame] = []
        tcn_leaderboard_frames: list[pd.DataFrame] = []
        lstm_history_frames: list[pd.DataFrame] = []
        lstm_checkpoint_frames: list[pd.DataFrame] = []
        variants = {
            "base": (base_features, base_index, observed_hashes["base_features"]),
            "relative": (
                relative_features,
                relative_index,
                observed_hashes["relative_features"],
            ),
        }
        for variant, (features, window_index, data_identity) in variants.items():
            variant_contracts = dict(contracts)
            variant_contracts["tcn_training_contract_id"] = (
                f"{variant}-dynamic-horizon-skip-smooth-l1-v37"
            )
            variant_contracts["lstm_training_contract_id"] = (
                f"{variant}-lstm-smooth-l1-v37"
            )
            for seed in seeds:
                tuning = run_tcn_validation_sweep(
                    features,
                    window_index,
                    labels,
                    split_manifest,
                    trials=trials,
                    seed=seed,
                    max_epochs=int(config["max_epochs"]),
                    patience=int(config["patience"]),
                    min_delta=float(config["min_delta"]),
                    checkpoint_min_delta=float(config["checkpoint_min_delta"]),
                    torch_threads=int(config["torch_threads"]),
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
                tcn_history_frames.append(history)
                leaderboard = tuning.leaderboard.copy()
                leaderboard["variant"] = variant
                tcn_leaderboard_frames.append(leaderboard)
                prediction_frames.append(
                    _collect_tcn_predictions(
                        features,
                        labels,
                        split_manifest,
                        trial,
                        tuning.best_states,
                        seed=seed,
                        model_name=f"{variant}_tcn",
                        lookup=lookup,
                        contracts=variant_contracts,
                    )
                )
                for state_key, state in tuning.best_states.items():
                    torch.save(
                        state,
                        checkpoint_root / f"{variant}-seed-{seed}-{state_key}.pt",
                    )

            lstm_config = cast(dict[str, object], config["lstm"])
            lstm_predictions, lstm_history, lstm_checkpoints = _train_lstm(
                features,
                window_index,
                labels,
                split_manifest,
                seeds=seeds,
                hidden_size=int(cast(Any, lstm_config["hidden_size"])),
                learning_rate=float(cast(Any, lstm_config["learning_rate"])),
                batch_size=int(cast(Any, lstm_config["batch_size"])),
                epochs=int(cast(Any, lstm_config["epochs"])),
                torch_threads=int(cast(Any, config["torch_threads"])),
                lookup=lookup,
                contracts=variant_contracts,
                checkpoint_dir=checkpoint_root / f"{variant}-lstm",
            )
            lstm_predictions["model"] = f"{variant}_lstm"
            lstm_history["variant"] = variant
            lstm_checkpoints["variant"] = variant
            lstm_history_frames.append(lstm_history)
            lstm_checkpoint_frames.append(lstm_checkpoints)
            prediction_frames.append(lstm_predictions)

        predictions = pd.concat(prediction_frames, ignore_index=True)
        validate_prediction_contract(predictions, expected_models=4)
        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(config["top_fraction"])
        )
        summary = summarize_task_aligned_metrics(metrics)
        comparison_pairs = {
            "relative_tcn_minus_base_tcn": ("base_tcn", "relative_tcn"),
            "relative_lstm_minus_base_lstm": ("base_lstm", "relative_lstm"),
            "relative_tcn_minus_relative_lstm": ("relative_lstm", "relative_tcn"),
        }
        comparisons = {
            name: compare_task_aligned_models(
                metrics, reference_model=reference, candidate_model=candidate
            )
            for name, (reference, candidate) in comparison_pairs.items()
        }
        bootstrap = pd.concat(
            [
                _comparison_bootstrap(
                    metrics,
                    reference_model=reference,
                    candidate_model=candidate,
                    scope=name,
                    seed=int(config["bootstrap_seed"]),
                    draws=int(config["bootstrap_draws"]),
                )
                for name, (reference, candidate) in comparison_pairs.items()
            ],
            ignore_index=True,
        )
        tcn_history = pd.concat(tcn_history_frames, ignore_index=True)
        tcn_leaderboard = pd.concat(tcn_leaderboard_frames, ignore_index=True)
        lstm_history = pd.concat(lstm_history_frames, ignore_index=True)
        lstm_checkpoints = pd.concat(lstm_checkpoint_frames, ignore_index=True)

        protocols = build_fold_protocols(base_features, split_manifest)
        train_counts = {protocol.fold: len(protocol.train_positions) for protocol in protocols}
        lstm_checkpoints["samples_per_second"] = [
            train_counts[int(cast(Any, row.fold))]
            * int(
                cast(Any, cast(dict[str, object], config["lstm"])["epochs"])
            )
            / float(cast(Any, row.training_seconds))
            for row in lstm_checkpoints.itertuples(index=False)
        ]
        base_tcn_speed = float(
            tcn_leaderboard.loc[
                tcn_leaderboard["variant"].eq("base"), "samples_per_second"
            ].median()
        )
        relative_tcn_speed = float(
            tcn_leaderboard.loc[
                tcn_leaderboard["variant"].eq("relative"), "samples_per_second"
            ].median()
        )
        paired_speed = tcn_leaderboard.loc[
            tcn_leaderboard["variant"].eq("relative"),
            ["seed", "fold", "samples_per_second"],
        ].merge(
            lstm_checkpoints.loc[
                lstm_checkpoints["variant"].eq("relative"),
                ["seed", "fold", "samples_per_second"],
            ],
            on=["seed", "fold"],
            suffixes=("_tcn", "_lstm"),
            validate="one_to_one",
        )
        relative_tcn_over_lstm = float(
            np.exp(
                np.log(
                    paired_speed["samples_per_second_tcn"]
                    / paired_speed["samples_per_second_lstm"]
                ).mean()
            )
        )
        speed = {
            "base_tcn_median_samples_per_second": base_tcn_speed,
            "relative_tcn_median_samples_per_second": relative_tcn_speed,
            "tcn_speed_retention": relative_tcn_speed / base_tcn_speed,
            "relative_tcn_over_lstm_geomean": relative_tcn_over_lstm,
            "timing_scope": "full_training_plus_validation_cycle",
        }
        decision = decide_relative_feature_gate(
            tcn_leaderboard,
            comparisons["relative_tcn_minus_base_tcn"],
            bootstrap,
            seeds=seeds,
            folds=folds,
            base_variant="base",
            candidate_variant="relative",
            base_median_samples_per_second=base_tcn_speed,
            candidate_median_samples_per_second=relative_tcn_speed,
            gates=cast(dict[str, float | int], config["gates"]),
        )
        selection: dict[str, object] = {
            "status": decision.status,
            "admitted": decision.admitted,
            "blockers": list(decision.blockers),
            "evidence": decision.evidence,
            "comparisons": comparisons,
            "speed": speed,
            "top50_readiness": top50,
            "next_step_authorized": (
                "integrate_relative_features_with_v35_training_contract"
                if decision.admitted
                else "repair_top50_pit_state_or_test_date_level_adapter"
            ),
            "sealed_test_accessed": False,
            "sealed_test_authorized": False,
        }

        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap-summary.parquet", index=False)
        decision.unit_deltas.to_parquet(temporary / "tcn-unit-deltas.parquet", index=False)
        tcn_history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        tcn_leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        lstm_history.to_parquet(temporary / "lstm-epoch-history.parquet", index=False)
        lstm_checkpoints.to_parquet(temporary / "lstm-checkpoints.parquet", index=False)
        _write_json(temporary / "comparisons.json", comparisons)
        _write_json(temporary / "speed.json", speed)
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        (temporary / "report.md").write_text(
            _report(selection, summary, comparisons, speed, top50), encoding="utf-8"
        )

        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-relative-feature-validation-v37/v1",
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
                "cuda_available": torch.cuda.is_available(),
                "torch_threads": int(config["torch_threads"]),
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
