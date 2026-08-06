"""Run the immutable v35 constrained Top-tail checkpoint-selection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.checkpoint_selection import (  # noqa: E402
    SELECTION_METRICS,
    select_constrained_tail_checkpoints,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.neural import HORIZONS  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    build_tcn_lstm_comparison,
    parse_real_tcn_trials,
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

from run_tcn_dynamic_skip_learning_rate import (  # noqa: E402
    _historical_evidence,
    _load_parent,
)
from run_tcn_frozen_parent_shape_residual import (  # noqa: E402
    _load_frozen_parent_states,
)
from run_tcn_multiseed_confirmation import (  # noqa: E402
    _contains_secret_key,
    _sha256,
    _write_json,
)
from run_tcn_task_aligned_evaluation import (  # noqa: E402
    _label_lookup,
    _prediction_rows,
)
from run_tcn_top_tail_alignment import _bootstrap_row  # noqa: E402


EVALUATION_METRICS = (
    "rankic",
    "pearson_ic",
    "top_return",
    "top_excess_return",
    "long_short_spread",
    "top_precision",
    "ndcg_at_top",
    "quantile_monotonicity",
    "top_turnover",
)


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_v34_parent(
    config: dict[str, object],
    *,
    expected_source_hashes: Mapping[str, str],
) -> tuple[Path, dict[str, object], pd.DataFrame]:
    artifact = _resolve_project_path(config["v34_parent_artifact"])
    receipt_path = artifact / "receipt.json"
    selection_path = artifact / "selection.json"
    predictions_path = artifact / "predictions.parquet"
    if not all(path.is_file() for path in (receipt_path, selection_path, predictions_path)):
        raise ContractError("v35 v34 parent artifact is incomplete")
    receipt = cast(
        dict[str, object], json.loads(receipt_path.read_text(encoding="utf-8"))
    )
    selection = cast(
        dict[str, object], json.loads(selection_path.read_text(encoding="utf-8"))
    )
    if receipt.get("receipt_id") != config["v34_parent_receipt_id"]:
        raise ContractError("v35 v34 parent receipt identity drifted")
    if selection.get("status") != config["v34_parent_selection_status"]:
        raise ContractError("v35 v34 parent selection status drifted")
    if receipt.get("sealed_test_accessed") is not False or selection.get(
        "sealed_test_authorized"
    ) is not False:
        raise ContractError("v35 v34 parent is not sealed-test fail-closed")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("predictions.parquet") != _sha256(
        predictions_path
    ):
        raise ContractError("v35 v34 prediction output hash drifted")
    source_artifacts = receipt.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise ContractError("v35 v34 source identities are missing")
    for name, expected_hash in expected_source_hashes.items():
        source = source_artifacts.get(name)
        if not isinstance(source, dict) or source.get("sha256") != expected_hash:
            raise ContractError(f"v35 v34 source identity drifted for {name}")
    predictions = pd.read_parquet(predictions_path)
    lstm = predictions.loc[predictions["model"].astype(str).eq("lstm")].copy()
    if lstm.empty:
        raise ContractError("v35 v34 parent contains no LSTM predictions")
    return (
        artifact,
        {
            "path": str(artifact),
            "receipt_id": str(config["v34_parent_receipt_id"]),
            "selection_status": str(config["v34_parent_selection_status"]),
            "predictions_sha256": _sha256(predictions_path),
        },
        lstm,
    )


def _state_key(seed: int, trial_id: str, fold: int, epoch: int) -> str:
    return f"seed-{seed}-{trial_id}-fold-{fold}-epoch-{epoch}"


def _predict_state(
    features: np.ndarray,
    trial: TCNTuningTrial,
    state: Mapping[str, torch.Tensor],
    dataset: LazyWindowDataset,
    *,
    seed: int,
    fold: int,
    model_name: str,
    lookup: dict[tuple[int, int], tuple[object, ...]],
    contracts: dict[str, str],
    training_contract_id: str,
) -> pd.DataFrame:
    model = build_tcn_trial_model(
        trial,
        feature_count=int(features.shape[1]),
        input_steps=int(features.shape[2]),
    )
    model.load_state_dict(state, strict=True)
    scores, positions = _predict_tcn_trial(
        model, dataset, batch_size=trial.batch_size
    )
    return pd.DataFrame(
        _prediction_rows(
            scores,
            positions,
            model=model_name,
            seed=seed,
            fold=fold,
            lookup=lookup,
            contracts=contracts,
            training_contract_id=training_contract_id,
        )
    )


def _trajectory_epoch_metrics(
    features: np.ndarray,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    trial: TCNTuningTrial,
    epoch_states: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    seeds: tuple[int, ...],
    epochs: tuple[int, ...],
    contracts: dict[str, str],
    top_fraction: float,
) -> tuple[pd.DataFrame, float]:
    protocols = build_fold_protocols(features, split_manifest)
    targets = np.zeros((len(features), len(HORIZONS)), dtype="float32")
    masks = np.ones_like(targets, dtype="bool")
    lookup = _label_lookup(labels)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for seed in seeds:
        for protocol in protocols:
            dataset = LazyWindowDataset(
                features,
                protocol.validation_positions,
                targets,
                masks,
                protocol.feature_mean,
                protocol.feature_std,
            )
            for epoch in epochs:
                key = _state_key(seed, trial.trial_id, protocol.fold, epoch)
                state = epoch_states.get(key)
                if state is None:
                    raise ContractError(f"v35 epoch state missing: {key}")
                prediction = _predict_state(
                    features,
                    trial,
                    state,
                    dataset,
                    seed=seed,
                    fold=protocol.fold,
                    model_name="trajectory",
                    lookup=lookup,
                    contracts=contracts,
                    training_contract_id=contracts["trajectory_training_contract_id"],
                )
                metrics = evaluate_task_aligned_predictions(
                    prediction, top_fraction=top_fraction
                )
                row: dict[str, object] = {
                    "trial_id": trial.trial_id,
                    "seed": seed,
                    "fold": protocol.fold,
                    "epoch": epoch,
                    "metric_group_count": len(metrics),
                    "prediction_count": len(prediction),
                    "stage": "validation",
                    "sealed": False,
                }
                for metric in EVALUATION_METRICS:
                    values = metrics[metric].to_numpy(dtype="float64")
                    finite = values[np.isfinite(values)]
                    if len(finite) == 0:
                        raise ContractError(
                            f"v35 epoch metric {metric} has no finite values"
                        )
                    row[metric] = float(np.mean(finite))
                rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["seed", "fold", "epoch"], ignore_index=True
    )
    expected_count = len(seeds) * len(protocols) * len(epochs)
    if len(result) != expected_count or result.duplicated(
        ["seed", "fold", "epoch"]
    ).any():
        raise ContractError("v35 trajectory metric coverage drifted")
    return result, time.perf_counter() - started


def _selected_predictions(
    features: np.ndarray,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    trial: TCNTuningTrial,
    epoch_states: Mapping[str, Mapping[str, torch.Tensor]],
    checkpoint_selection: pd.DataFrame,
    *,
    contracts: dict[str, str],
) -> pd.DataFrame:
    protocols = {
        protocol.fold: protocol
        for protocol in build_fold_protocols(features, split_manifest)
    }
    targets = np.zeros((len(features), len(HORIZONS)), dtype="float32")
    masks = np.ones_like(targets, dtype="bool")
    lookup = _label_lookup(labels)
    rows: list[pd.DataFrame] = []
    for selection in checkpoint_selection.itertuples(index=False):
        seed = int(cast(Any, selection.seed))
        fold = int(cast(Any, selection.fold))
        protocol = protocols[fold]
        dataset = LazyWindowDataset(
            features,
            protocol.validation_positions,
            targets,
            masks,
            protocol.feature_mean,
            protocol.feature_std,
        )
        for model_name in ("control", "candidate"):
            epoch = int(cast(Any, getattr(selection, f"{model_name}_epoch")))
            key = _state_key(seed, trial.trial_id, fold, epoch)
            state = epoch_states.get(key)
            if state is None:
                raise ContractError(f"v35 selected epoch state missing: {key}")
            rows.append(
                _predict_state(
                    features,
                    trial,
                    state,
                    dataset,
                    seed=seed,
                    fold=fold,
                    model_name=model_name,
                    lookup=lookup,
                    contracts=contracts,
                    training_contract_id=contracts[
                        f"{model_name}_training_contract_id"
                    ],
                )
            )
    predictions = pd.concat(rows, ignore_index=True)
    if predictions.empty:
        raise ContractError("v35 selected TCN predictions are empty")
    return predictions


def _report(
    selection: dict[str, object],
    checkpoint_selection: pd.DataFrame,
    summary: pd.DataFrame,
    control_comparison: dict[str, object],
    lstm_comparison: dict[str, object],
    bootstrap: pd.DataFrame,
) -> str:
    summary_lines = [
        (
            f"| {row.model} | {float(cast(Any, row.mean_rankic)):.6f} | "
            f"{float(cast(Any, row.mean_top_return)):+.6f} | "
            f"{float(cast(Any, row.mean_top_precision)):.4f} | "
            f"{float(cast(Any, row.mean_ndcg_at_top)):.4f} | "
            f"{float(cast(Any, row.mean_top_turnover)):.4f} |"
        )
        for row in summary.itertuples(index=False)
    ]
    epoch_lines = [
        (
            f"| {int(cast(Any, row.seed))} | {int(cast(Any, row.fold))} | "
            f"{int(cast(Any, row.control_epoch))} | "
            f"{int(cast(Any, row.candidate_epoch))} | "
            f"{float(cast(Any, row.rankic_delta)):+.6f} | "
            f"{float(cast(Any, row.top_precision_delta)):+.6f} | "
            f"{float(cast(Any, row.ndcg_at_top_delta)):+.6f} |"
        )
        for row in checkpoint_selection.itertuples(index=False)
    ]
    bootstrap_lines = [
        (
            f"| {row.scope} | {row.metric} | "
            f"{float(cast(Any, row.paired_mean_delta)):+.6f} | "
            f"{float(cast(Any, row.bootstrap_ci_low)):+.6f} | "
            f"{float(cast(Any, row.bootstrap_ci_high)):+.6f} |"
        )
        for row in bootstrap.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# TCN 受约束 Top-tail checkpoint selection v35",
            "",
            f"- 决策：`{selection['status']}`",
            f"- 完整性门：`{selection['integrity_passed']}`",
            f"- selection 机制门：`{selection['mechanism_passed']}`",
            f"- 预测效果门：`{selection['effect_passed']}`",
            f"- 速度门：`{selection['speed_passed']}`",
            f"- changed units：`{selection['changed_selection_units']}/15`",
            "- sealed test：未访问、未授权。",
            "",
            "control 与 candidate 来自同一条固定 8-epoch Top-tail TCN 轨迹；"
            "唯一变量是 RankIC selection 与受 RankIC 约束的 Top-tail selection。",
            "",
            "## 三模型普通验证摘要",
            "",
            "| model | RankIC | top return | top precision | NDCG@top | turnover |",
            "|---|---:|---:|---:|---:|---:|",
            *summary_lines,
            "",
            "## 单元 checkpoint 选择",
            "",
            "| seed | fold | control epoch | candidate epoch | RankIC Δ | precision Δ | NDCG Δ |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            *epoch_lines,
            "",
            "## Candidate 相对基准",
            "",
            f"- 相对 control：`{json.dumps(control_comparison, ensure_ascii=False, sort_keys=True)}`",
            f"- 相对 LSTM：`{json.dumps(lstm_comparison, ensure_ascii=False, sort_keys=True)}`",
            "",
            "## 配对日期块 bootstrap",
            "",
            "| scope | metric | mean delta | 95% CI low | 95% CI high |",
            "|---|---|---:|---:|---:|",
            *bootstrap_lines,
            "",
            f"- blockers：`{selection['blockers'] or 'none'}`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the immutable v35 constrained tail checkpoint selection"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v35 refuses to overwrite experiment artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v35 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v35":
            raise ContractError("v35 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v35 config contains a secret-like key")
        if config.get("precision") != "float32" or int(
            cast(Any, config["num_workers"])
        ) != 0:
            raise ContractError("v35 requires float32 and num_workers=0")
        seeds = tuple(
            int(cast(Any, value)) for value in cast(list[object], config["seeds"])
        )
        folds = cast(list[object], config["folds"])
        epochs = tuple(range(int(cast(Any, config["max_epochs"])) + 1))
        if seeds != (7, 17, 27) or folds != [0, 1, 2, 3, 4] or epochs != tuple(
            range(9)
        ):
            raise ContractError("v35 requires seeds 7/17/27, folds 0..4, epochs 0..8")
        if (
            int(cast(Any, config["patience"])) != 2
            or float(cast(Any, config["min_delta"])) != 0.0005
            or float(cast(Any, config["checkpoint_min_delta"])) != 0.0
            or config.get("disable_early_stopping") is not True
            or config.get("capture_epoch_states") is not True
            or float(cast(Any, config["rankic_tolerance"])) != 0.002
            or float(cast(Any, config["top_precision_selection_weight"])) != 0.5
            or float(cast(Any, config["ndcg_selection_weight"])) != 0.5
        ):
            raise ContractError("v35 trajectory or selection contract drifted")

        expected_hashes_value = config.get("source_sha256")
        if not isinstance(expected_hashes_value, dict):
            raise ContractError("v35 source identities are missing")
        expected_hashes = {
            str(key): str(value) for key, value in expected_hashes_value.items()
        }
        seed7_parent, seed7_identity = _load_parent(
            config, prefix="seed7", expected_source_hashes=expected_hashes
        )
        confirmation_parent, confirmation_identity = _load_parent(
            config, prefix="confirmation", expected_source_hashes=expected_hashes
        )
        v34_parent, v34_identity, lstm_predictions = _load_v34_parent(
            config, expected_source_hashes=expected_hashes
        )

        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "labels": run_dir / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "universe": run_dir / "universe.parquet",
            "input_manifest": run_dir / "input-manifest.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v35 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if observed_hashes != expected_hashes:
            raise ContractError("v35 source SHA-256 identity drifted")
        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v35 rejects sealed split rows")
        observed_stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(
            observed_stages - {"train", "validation", "purged"}
        ):
            raise ContractError("v35 split contains forbidden stages: " + ", ".join(unknown))
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 1:
            raise ContractError("v35 must train exactly one shared trajectory")
        trial = trials[0]
        if trial.trial_id != str(config["trajectory_trial_id"]):
            raise ContractError("v35 trajectory trial identity drifted")
        if (
            trial.strategy != "top_tail"
            or trial.top_tail_weight != 0.05
            or trial.top_tail_fraction != 0.1
            or trial.top_tail_temperature != 0.1
            or trial.date_batch_order != "fixed_once"
            or not trial.dynamic_skip_frozen_parent
            or not trial.dynamic_skip_shape_residual
        ):
            raise ContractError("v35 frozen Top-tail trajectory contract drifted")

        parent_states, parent_manifest = _load_frozen_parent_states(
            seed7_parent, confirmation_parent, trial.trial_id
        )
        parent_manifest.insert(0, "trial_id", trial.trial_id)
        identities = {
            "data": observed_hashes["features"],
            "fold_manifest": observed_hashes["split_manifest"],
            "evaluation": observed_hashes["labels"],
        }
        tuning_parts = []
        epoch_states: dict[str, dict[str, torch.Tensor]] = {}
        for seed in seeds:
            tuning = run_tcn_validation_sweep(
                features,
                window_index,
                labels,
                split_manifest,
                trials=(trial,),
                seed=seed,
                max_epochs=int(cast(Any, config["max_epochs"])),
                patience=int(cast(Any, config["patience"])),
                min_delta=float(cast(Any, config["min_delta"])),
                checkpoint_min_delta=float(cast(Any, config["checkpoint_min_delta"])),
                torch_threads=int(cast(Any, config["torch_threads"])),
                protocol_identities=identities,
                frozen_parent_states=parent_states[seed],
                capture_epoch_states=True,
                disable_early_stopping=True,
            )
            tuning_parts.append(tuning)
            for key, state in tuning.epoch_states.items():
                epoch_states[f"seed-{seed}-{key}"] = state
        epoch_history = pd.concat(
            [part.epoch_history for part in tuning_parts], ignore_index=True
        )
        leaderboard = pd.concat(
            [part.leaderboard for part in tuning_parts], ignore_index=True
        ).merge(
            parent_manifest[
                ["trial_id", "seed", "fold", "parent_checkpoint_sha256"]
            ],
            on=["trial_id", "seed", "fold"],
            how="left",
            validate="one_to_one",
        )
        if len(epoch_states) != 135 or len(leaderboard) != 15:
            raise ContractError("v35 shared trajectory checkpoint coverage drifted")
        if (
            not leaderboard["completed_epochs"].astype(int).eq(8).all()
            or set(leaderboard["stopping_reason"].astype(str)) != {"max_epochs"}
        ):
            raise ContractError("v35 trajectory did not run the full fixed budget")

        contracts_value = config.get("contracts")
        if not isinstance(contracts_value, dict):
            raise ContractError("v35 prediction contracts are missing")
        contracts = {str(key): str(value) for key, value in contracts_value.items()}
        trajectory_metrics, selection_evaluation_seconds = _trajectory_epoch_metrics(
            features,
            labels,
            split_manifest,
            trial,
            epoch_states,
            seeds=seeds,
            epochs=epochs,
            contracts=contracts,
            top_fraction=float(cast(Any, config["top_fraction"])),
        )
        history_scores = epoch_history.set_index(["seed", "fold", "epoch"])[
            "mean_daily_rankic"
        ].sort_index()
        metric_scores = trajectory_metrics.set_index(["seed", "fold", "epoch"])[
            "rankic"
        ].sort_index()
        if not history_scores.index.equals(metric_scores.index):
            raise ContractError("v35 epoch RankIC coverage drifted")
        rankic_replay_max_abs_error = float(
            np.max(
                np.abs(
                    history_scores.to_numpy(dtype="float64")
                    - metric_scores.to_numpy(dtype="float64")
                )
            )
        )
        if rankic_replay_max_abs_error > 1e-12:
            raise ContractError("v35 epoch task-aligned RankIC replay drifted")

        checkpoint_selection = select_constrained_tail_checkpoints(
            trajectory_metrics,
            expected_epochs=epochs,
            rankic_tolerance=float(cast(Any, config["rankic_tolerance"])),
        )
        leaderboard_epochs = leaderboard.set_index(["seed", "fold"])["best_epoch"]
        selection_control_epochs = checkpoint_selection.set_index(["seed", "fold"])[
            "control_epoch"
        ]
        control_epoch_replay_max_abs_error = int(
            np.max(
                np.abs(
                    leaderboard_epochs.to_numpy(dtype="int64")
                    - selection_control_epochs.to_numpy(dtype="int64")
                )
            )
        )
        if control_epoch_replay_max_abs_error != 0:
            raise ContractError("v35 RankIC control selection replay drifted")

        selected_tcn_predictions = _selected_predictions(
            features,
            labels,
            split_manifest,
            trial,
            epoch_states,
            checkpoint_selection,
            contracts=contracts,
        )
        if set(lstm_predictions["training_contract_id"].astype(str)) != {
            contracts["lstm_training_contract_id"]
        }:
            raise ContractError("v35 LSTM training contract drifted")
        for column in (
            "prediction_contract_id",
            "target_contract_id",
            "evaluation_contract_id",
        ):
            if set(lstm_predictions[column].astype(str)) != {contracts[column]}:
                raise ContractError(f"v35 LSTM {column} drifted")
        predictions = pd.concat(
            [selected_tcn_predictions, lstm_predictions], ignore_index=True
        )
        validate_prediction_contract(predictions, expected_models=3)
        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(cast(Any, config["top_fraction"]))
        )
        summary = summarize_task_aligned_metrics(metrics)
        control_comparison = cast(
            dict[str, object],
            compare_task_aligned_models(
                metrics, reference_model="control", candidate_model="candidate"
            ),
        )
        lstm_comparison = cast(
            dict[str, object],
            compare_task_aligned_models(
                metrics, reference_model="lstm", candidate_model="candidate"
            ),
        )
        control_bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="control",
            candidate_model="candidate",
            metric_columns=EVALUATION_METRICS,
            seed=int(cast(Any, config["bootstrap_seed"])),
            draws=int(cast(Any, config["bootstrap_draws"])),
        )
        control_bootstrap.insert(0, "scope", "candidate-minus-control")
        lstm_bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="lstm",
            candidate_model="candidate",
            metric_columns=EVALUATION_METRICS,
            seed=int(cast(Any, config["bootstrap_seed"])) + 1,
            draws=int(cast(Any, config["bootstrap_draws"])),
        )
        lstm_bootstrap.insert(0, "scope", "candidate-minus-lstm")
        bootstrap = pd.concat(
            [control_bootstrap, lstm_bootstrap], ignore_index=True
        )

        selected_unit_metrics = (
            metrics.loc[metrics["model"].isin(["control", "candidate"])]
            .groupby(["model", "seed", "fold"], observed=True)[
                list(EVALUATION_METRICS)
            ]
            .mean()
        )
        selected_replay_errors: list[float] = []
        selection_indexed = checkpoint_selection.set_index(["seed", "fold"])
        for metric_row in selected_unit_metrics.reset_index().itertuples(
            index=False
        ):
            model_name = str(cast(Any, metric_row.model))
            seed = int(cast(Any, metric_row.seed))
            fold = int(cast(Any, metric_row.fold))
            for metric in SELECTION_METRICS:
                selected_replay_errors.append(
                    abs(
                        float(cast(Any, getattr(metric_row, metric)))
                        - float(
                            cast(
                                Any,
                                selection_indexed.loc[
                                    (seed, fold), f"{model_name}_{metric}"
                                ],
                            )
                        )
                    )
                )
        selected_metric_replay_max_abs_error = max(selected_replay_errors)
        if selected_metric_replay_max_abs_error > 1e-12:
            raise ContractError("v35 selected checkpoint metric replay drifted")

        _, _, lstm_measurements, lstm_environment = _historical_evidence(
            seed7_parent,
            confirmation_parent,
            control_trial_id=str(config["historical_control_trial_id"]),
            parent_candidate_trial_id=str(config["historical_parent_trial_id"]),
        )
        speed_comparison = build_tcn_lstm_comparison(leaderboard, lstm_measurements)
        gradient_diagnostics = epoch_history.loc[
            epoch_history["stage"].astype(str).eq("validation"),
            [
                "trial_id",
                "seed",
                "fold",
                "epoch",
                "loss_group_count_mean",
                "valid_label_count_mean",
                "top_tail_pair_count_mean",
                "top_tail_pair_count_min",
                "top_tail_pair_count_max",
                "component_gradient_cosine_mean",
                "component_gradient_cosine_median",
                "component_gradient_cosine_min",
                "gradient_norm_mean",
                "gradient_norm_cv",
                "samples_per_second",
            ],
        ].copy()

        temporary.mkdir(parents=True)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_rows: list[dict[str, object]] = []
        for seed in seeds:
            for fold in range(5):
                for epoch in epochs:
                    key = _state_key(seed, trial.trial_id, fold, epoch)
                    state = epoch_states[key]
                    relative = Path("checkpoints") / f"{key}.pt"
                    checkpoint = temporary / relative
                    torch.save(state, checkpoint)
                    checkpoint_rows.append(
                        {
                            "trajectory_id": trial.trial_id,
                            "seed": seed,
                            "fold": fold,
                            "epoch": epoch,
                            "state_key": key,
                            "checkpoint": str(relative),
                            "checkpoint_sha256": _sha256(checkpoint),
                            "tensor_count": len(state),
                        }
                    )
        checkpoint_manifest = pd.DataFrame(checkpoint_rows).sort_values(
            ["seed", "fold", "epoch"], ignore_index=True
        )
        if len(checkpoint_manifest) != 135 or checkpoint_manifest.duplicated(
            ["seed", "fold", "epoch"]
        ).any():
            raise ContractError("v35 saved checkpoint coverage drifted")
        for model_name in ("control", "candidate"):
            selected_manifest = checkpoint_manifest.rename(
                columns={
                    "epoch": f"{model_name}_epoch",
                    "checkpoint": f"{model_name}_checkpoint",
                    "checkpoint_sha256": f"{model_name}_checkpoint_sha256",
                }
            )[
                [
                    "seed",
                    "fold",
                    f"{model_name}_epoch",
                    f"{model_name}_checkpoint",
                    f"{model_name}_checkpoint_sha256",
                ]
            ]
            checkpoint_selection = checkpoint_selection.merge(
                selected_manifest,
                on=["seed", "fold", f"{model_name}_epoch"],
                validate="one_to_one",
            )

        gates = cast(dict[str, object], config["gates"])
        changed_selection_units = int(checkpoint_selection["selection_changed"].sum())
        gradient_cosine_median = float(
            gradient_diagnostics["component_gradient_cosine_median"].median()
        )
        integrity_passed = bool(
            len(trajectory_metrics) == 135
            and len(checkpoint_manifest) == 135
            and len(checkpoint_selection) == 15
            and len(leaderboard) == 15
            and set(leaderboard["trainable_parameter_count"].astype(int)) == {88}
            and leaderboard["frozen_parent_state_drift_max"].eq(0.0).all()
            and leaderboard["parent_prediction_max_abs_error"].eq(0.0).all()
            and not predictions["sealed"].astype(bool).any()
            and rankic_replay_max_abs_error <= 1e-12
            and selected_metric_replay_max_abs_error <= 1e-12
        )
        expected_loss_identity = "smooth-l1+0.05-top-tail-fraction-0.1-tau-0.1"
        mechanism_passed = bool(
            set(leaderboard["loss_identity"].astype(str)) == {expected_loss_identity}
            and set(leaderboard["strategy"].astype(str)) == {"top_tail"}
            and leaderboard["top_tail_weight"].eq(0.05).all()
            and leaderboard["top_tail_fraction"].eq(0.1).all()
            and leaderboard["top_tail_temperature"].eq(0.1).all()
            and checkpoint_selection["candidate_rankic_feasible"].all()
            and changed_selection_units
            >= int(cast(Any, gates["min_changed_selection_units"]))
            and gradient_diagnostics["top_tail_pair_count_min"].astype(float).gt(0).all()
            and np.isfinite(
                gradient_diagnostics[
                    "component_gradient_cosine_median"
                ].to_numpy(dtype="float64")
            ).all()
            and gradient_cosine_median
            >= float(cast(Any, gates["min_component_gradient_cosine"]))
        )
        precision_bootstrap = _bootstrap_row(
            bootstrap, "candidate-minus-control", "top_precision"
        )
        ndcg_bootstrap = _bootstrap_row(
            bootstrap, "candidate-minus-control", "ndcg_at_top"
        )
        top_return_bootstrap = _bootstrap_row(
            bootstrap, "candidate-minus-control", "top_return"
        )
        precision_ci_low = float(
            cast(Any, precision_bootstrap["bootstrap_ci_low"])
        )
        ndcg_ci_low = float(cast(Any, ndcg_bootstrap["bootstrap_ci_low"]))
        secondary_ci_low = float(cast(Any, gates["min_secondary_ci_low"]))
        robust_tail_improvement = bool(
            (precision_ci_low > 0 and ndcg_ci_low >= secondary_ci_low)
            or (ndcg_ci_low > 0 and precision_ci_low >= secondary_ci_low)
        )
        effect_passed = bool(
            float(cast(Any, control_comparison["mean_top_precision_delta"]))
            > float(cast(Any, gates["min_mean_top_precision_delta"]))
            and float(cast(Any, control_comparison["mean_ndcg_at_top_delta"]))
            > float(cast(Any, gates["min_mean_ndcg_delta"]))
            and robust_tail_improvement
            and float(cast(Any, control_comparison["mean_rankic_delta"]))
            >= float(cast(Any, gates["min_mean_rankic_delta"]))
            and float(cast(Any, top_return_bootstrap["bootstrap_ci_low"]))
            >= float(cast(Any, gates["min_top_return_ci_low"]))
            and float(cast(Any, control_comparison["mean_top_turnover_delta"]))
            <= float(cast(Any, gates["max_mean_turnover_delta"]))
        )
        speed_passed = bool(
            float(speed_comparison["model_step_speed_ratio"])
            >= float(cast(Any, gates["min_model_step_speed_ratio"]))
            and float(speed_comparison["end_to_end_speed_ratio"])
            >= float(cast(Any, gates["min_end_to_end_speed_ratio"]))
        )
        blockers = []
        if not integrity_passed:
            blockers.append("integrity")
        if changed_selection_units == 0:
            blockers.append("no_selection_opportunity")
        elif not mechanism_passed:
            blockers.append("selection_mechanism")
        if not effect_passed:
            blockers.append("task_aligned_prediction_effect")
        if not speed_passed:
            blockers.append("speed")
        if not integrity_passed:
            status = "stop_constrained_tail_integrity_v35"
        elif changed_selection_units == 0:
            status = "stop_no_selection_opportunity_v35"
        elif not mechanism_passed:
            status = "stop_constrained_tail_mechanism_v35"
        elif not effect_passed:
            status = "stop_constrained_tail_no_gain_v35"
        elif not speed_passed:
            status = "stop_constrained_tail_speed_v35"
        else:
            status = "constrained_tail_ordinary_validation_candidate_v35"
        selection: dict[str, object] = {
            "status": status,
            "integrity_passed": integrity_passed,
            "mechanism_passed": mechanism_passed,
            "effect_passed": effect_passed,
            "speed_passed": speed_passed,
            "next_step_authorized": bool(
                integrity_passed and mechanism_passed and effect_passed and speed_passed
            ),
            "sealed_test_authorized": False,
            "sealed_test_accessed": False,
            "trajectory_trial_id": trial.trial_id,
            "changed_selection_units": changed_selection_units,
            "rankic_replay_max_abs_error": rankic_replay_max_abs_error,
            "selected_metric_replay_max_abs_error": (
                selected_metric_replay_max_abs_error
            ),
            "control_epoch_replay_max_abs_error": (
                control_epoch_replay_max_abs_error
            ),
            "component_gradient_cosine_median": gradient_cosine_median,
            "selection_evaluation_seconds": selection_evaluation_seconds,
            "model_step_speed_ratio": float(speed_comparison["model_step_speed_ratio"]),
            "end_to_end_speed_ratio": float(speed_comparison["end_to_end_speed_ratio"]),
            "robust_tail_improvement": robust_tail_improvement,
            "blockers": blockers,
        }

        epoch_history.to_parquet(
            temporary / "trajectory-epoch-history.parquet", index=False
        )
        trajectory_metrics.to_parquet(
            temporary / "trajectory-epoch-metrics.parquet", index=False
        )
        checkpoint_manifest.to_parquet(
            temporary / "trajectory-checkpoint-manifest.parquet", index=False
        )
        checkpoint_selection.to_parquet(
            temporary / "checkpoint-selection.parquet", index=False
        )
        predictions.to_parquet(
            temporary / "selected-predictions.parquet", index=False
        )
        metrics.to_parquet(
            temporary / "selected-task-aligned-metrics.parquet", index=False
        )
        summary.to_parquet(
            temporary / "selected-model-summary.parquet", index=False
        )
        bootstrap.to_parquet(temporary / "bootstrap-summary.parquet", index=False)
        gradient_diagnostics.to_parquet(
            temporary / "gradient-conflict-diagnostics.parquet", index=False
        )
        parent_manifest.to_parquet(
            temporary / "parent-checkpoint-manifest.parquet", index=False
        )
        leaderboard.to_parquet(temporary / "trajectory-leaderboard.parquet", index=False)
        lstm_measurements.to_parquet(
            temporary / "lstm-measurements.parquet", index=False
        )
        _write_json(
            temporary / "control-candidate-comparison.json", control_comparison
        )
        _write_json(
            temporary / "candidate-lstm-context-comparison.json", lstm_comparison
        )
        _write_json(temporary / "speed-comparison.json", speed_comparison)
        _write_json(temporary / "lstm-environment.json", lstm_environment)
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        (temporary / "report.md").write_text(
            _report(
                selection,
                checkpoint_selection,
                summary,
                control_comparison,
                lstm_comparison,
                bootstrap,
            ),
            encoding="utf-8",
        )
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-constrained-tail-checkpoint-selection-v35/v1",
            "run_id": str(config["run_id"]),
            "parents": {
                "seed7": seed7_identity,
                "confirmation": confirmation_identity,
                "v34": v34_identity,
            },
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "source_config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
            },
            "selection": selection,
            "control_candidate_comparison": control_comparison,
            "candidate_lstm_context_comparison": lstm_comparison,
            "speed_comparison": speed_comparison,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "result": status,
            "output_dir": str(output_dir),
            "receipt_id": receipt["receipt_id"],
        }
    except (
        ContractError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
