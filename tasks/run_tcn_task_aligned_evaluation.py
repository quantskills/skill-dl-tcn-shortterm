"""Run the immutable v33 task-aligned TCN/LSTM validation replay."""

from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm.experiment import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.neural import (  # noqa: E402
    HORIZONS,
    RecurrentRegressor,
    _label_matrices,
)
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.runtime import torch_thread_scope  # noqa: E402
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
    masked_smooth_l1,
    predict_model,
)
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    _predict_tcn_trial,
    build_tcn_trial_model,
    build_validation_rankic_plan,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contains_secret_key(value: object) -> bool:
    secret_fragments = ("password", "passwd", "secret", "token", "credential")
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(fragment in str(key).lower() for fragment in secret_fragments):
                return True
            if _contains_secret_key(nested):
                return True
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _label_lookup(labels: pd.DataFrame) -> dict[tuple[int, int], tuple[object, ...]]:
    lookup: dict[tuple[int, int], tuple[object, ...]] = {}
    for row in labels.loc[labels["valid"].astype(bool)].itertuples(index=False):
        key = (int(cast(Any, row.sample_position)), int(cast(Any, row.horizon)))
        if key in lookup:
            raise ContractError("v33 labels contain duplicate sample/horizon keys")
        lookup[key] = (
            str(cast(Any, row.sample_id)),
            str(cast(Any, row.instrument_id)),
            str(cast(Any, row.signal_date)),
            float(cast(Any, row.rank_target)),
            float(cast(Any, row.raw_return)),
            str(cast(Any, row.label_version)),
        )
    if not lookup:
        raise ContractError("v33 valid label lookup is empty")
    return lookup


def _prediction_rows(
    scores: np.ndarray,
    positions: np.ndarray,
    *,
    model: str,
    seed: int,
    fold: int,
    lookup: dict[tuple[int, int], tuple[object, ...]],
    contracts: dict[str, str],
    training_contract_id: str,
) -> list[dict[str, object]]:
    if scores.shape != (len(positions), len(HORIZONS)):
        raise ContractError("v33 model score shape drifted")
    rows: list[dict[str, object]] = []
    for row_index, sample_position in enumerate(positions):
        for horizon_index, horizon in enumerate(HORIZONS):
            label = lookup.get((int(sample_position), int(horizon)))
            if label is None:
                continue
            sample_id, instrument_id, signal_date, rank_target, raw_return, version = (
                label
            )
            if str(version) != contracts["target_contract_id"]:
                raise ContractError("v33 label target contract drifted")
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "fold": fold,
                    "sample_id": sample_id,
                    "instrument_id": instrument_id,
                    "signal_date": signal_date,
                    "horizon": horizon,
                    "score": float(scores[row_index, horizon_index]),
                    "rank_target": rank_target,
                    "raw_return": raw_return,
                    "stage": "validation",
                    "sealed": False,
                    "prediction_contract_id": contracts[
                        "prediction_contract_id"
                    ],
                    "target_contract_id": contracts["target_contract_id"],
                    "evaluation_contract_id": contracts[
                        "evaluation_contract_id"
                    ],
                    "training_contract_id": training_contract_id,
                }
            )
    if not rows:
        raise ContractError("v33 model produced no valid prediction rows")
    return rows


def _load_parent(parent: Path, receipt_id: str) -> tuple[dict[str, object], dict[str, object]]:
    receipt_path = parent / "receipt.json"
    config_path = parent / "config.resolved.json"
    if not receipt_path.is_file() or not config_path.is_file():
        raise ContractError("v33 parent artifact is incomplete")
    receipt = cast(
        dict[str, object], json.loads(receipt_path.read_text(encoding="utf-8"))
    )
    config = cast(
        dict[str, object], json.loads(config_path.read_text(encoding="utf-8"))
    )
    if receipt.get("receipt_id") != receipt_id:
        raise ContractError("v33 parent receipt identity drifted")
    if cast(dict[str, object], receipt.get("selection", {})).get(
        "sealed_test_authorized"
    ) is not False:
        raise ContractError("v33 parent sealed-test state is not fail-closed")
    return receipt, config


def _replay_tcn(
    features: np.ndarray,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    parent: Path,
    parent_config: dict[str, object],
    trial_id: str,
    seeds: tuple[int, ...],
    lookup: dict[tuple[int, int], tuple[object, ...]],
    contracts: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = parse_real_tcn_trials(parent_config["trials"])
    matching = [trial for trial in trials if trial.trial_id == trial_id]
    if len(matching) != 1:
        raise ContractError("v33 TCN trial is missing or ambiguous in parent")
    trial = matching[0]
    parent_leaderboard = pd.read_parquet(parent / "tcn-leaderboard.parquet")
    parent_leaderboard = parent_leaderboard.loc[
        parent_leaderboard["trial_id"].astype(str).eq(trial_id)
    ].set_index(["seed", "fold"])
    protocols = build_fold_protocols(features, split_manifest)
    # Targets are not consumed during checkpoint inference; these arrays only
    # satisfy the shared lazy dataset batch contract.
    targets = np.zeros((len(features), len(HORIZONS)), dtype="float32")
    masks = np.ones_like(targets, dtype="bool")
    prediction_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    for seed in seeds:
        for protocol in protocols:
            checkpoint = (
                parent
                / "checkpoints"
                / f"seed-{seed}-{trial_id}-fold-{protocol.fold}.pt"
            )
            if not checkpoint.is_file():
                raise ContractError(f"v33 TCN checkpoint missing: {checkpoint.name}")
            model = build_tcn_trial_model(
                trial,
                feature_count=int(features.shape[1]),
                input_steps=int(features.shape[2]),
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
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
            prediction_rows.extend(
                _prediction_rows(
                    scores,
                    positions,
                    model="tcn",
                    seed=seed,
                    fold=protocol.fold,
                    lookup=lookup,
                    contracts=contracts,
                    training_contract_id=contracts["tcn_training_contract_id"],
                )
            )
            checkpoint_rows.append(
                {
                    "model": "tcn",
                    "seed": seed,
                    "fold": protocol.fold,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "trainable_parameter_count": int(
                        cast(
                            Any,
                            parent_leaderboard.loc[
                                (seed, protocol.fold),
                                "trainable_parameter_count",
                            ],
                        )
                    ),
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(checkpoint_rows)


def _train_lstm(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
    hidden_size: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    torch_threads: int,
    lookup: dict[tuple[int, int], tuple[object, ...]],
    contracts: dict[str, str],
    checkpoint_dir: Path,
    resume_existing_checkpoints: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets, masks = _label_matrices(window_index, labels)
    prediction_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=resume_existing_checkpoints)
    with torch_thread_scope(torch_threads):
        protocols = build_fold_protocols(features, split_manifest)
        for seed in seeds:
            for protocol in protocols:
                model_seed = seed + protocol.fold * 100
                torch.manual_seed(model_seed)
                generator = torch.Generator().manual_seed(model_seed)
                train_dataset = LazyWindowDataset(
                    features,
                    protocol.train_positions,
                    targets,
                    masks,
                    protocol.feature_mean,
                    protocol.feature_std,
                )
                validation_dataset = LazyWindowDataset(
                    features,
                    protocol.validation_positions,
                    targets,
                    masks,
                    protocol.feature_mean,
                    protocol.feature_std,
                )
                loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    generator=generator,
                    num_workers=0,
                )
                validation_plan = build_validation_rankic_plan(
                    protocol.validation_positions, window_index, labels
                )
                model = RecurrentRegressor("lstm", features.shape[1], hidden_size)
                checkpoint = checkpoint_dir / f"seed-{seed}-lstm-fold-{protocol.fold}.pt"
                if checkpoint.exists():
                    if not resume_existing_checkpoints:
                        raise ContractError(
                            "v33 found an unexpected pre-existing LSTM checkpoint"
                        )
                    state = torch.load(
                        checkpoint, map_location="cpu", weights_only=True
                    )
                    model.load_state_dict(state, strict=True)
                    scores, positions = predict_model(
                        model,
                        validation_dataset,
                        batch_size=batch_size,
                        num_workers=0,
                    )
                    replay_rankic = float(
                        validation_plan.evaluate(scores, positions).mean_daily_rankic
                    )
                    prediction_rows.extend(
                        _prediction_rows(
                            scores,
                            positions,
                            model="lstm",
                            seed=seed,
                            fold=protocol.fold,
                            lookup=lookup,
                            contracts=contracts,
                            training_contract_id=contracts[
                                "lstm_training_contract_id"
                            ],
                        )
                    )
                    history_rows.append(
                        {
                            "model": "lstm",
                            "seed": seed,
                            "model_seed": model_seed,
                            "fold": protocol.fold,
                            "epoch": -1,
                            "mean_train_loss": float("nan"),
                            "validation_rankic": replay_rankic,
                            "epoch_seconds": 0.0,
                            "model_step_seconds": 0.0,
                            "validation_seconds": 0.0,
                            "stage": "validation",
                            "sealed": False,
                            "history_event": "verified_checkpoint_recovery",
                            "history_complete": False,
                        }
                    )
                    checkpoint_rows.append(
                        {
                            "model": "lstm",
                            "seed": seed,
                            "model_seed": model_seed,
                            "fold": protocol.fold,
                            "best_epoch": -1,
                            "best_validation_rankic": replay_rankic,
                            "training_seconds": float("nan"),
                            "parameter_count": sum(
                                parameter.numel() for parameter in model.parameters()
                            ),
                            "checkpoint": f"checkpoints/{checkpoint.name}",
                            "checkpoint_sha256": _sha256(checkpoint),
                            "recovered_from_interruption": True,
                        }
                    )
                    continue
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                best_rankic = float("nan")
                best_epoch = 0
                best_state: dict[str, torch.Tensor] | None = None
                start = time.perf_counter()
                for epoch in range(1, epochs + 1):
                    epoch_start = time.perf_counter()
                    model.train()
                    losses: list[float] = []
                    model_step_seconds = 0.0
                    for batch_features, batch_targets, batch_masks, _ in loader:
                        model_step_start = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        prediction = model(batch_features)
                        loss = masked_smooth_l1(
                            prediction, batch_targets, batch_masks
                        )
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        model_step_seconds += time.perf_counter() - model_step_start
                    validation_start = time.perf_counter()
                    scores, positions = predict_model(
                        model,
                        validation_dataset,
                        batch_size=batch_size,
                        num_workers=0,
                    )
                    rankic = validation_plan.evaluate(
                        scores, positions
                    ).mean_daily_rankic
                    validation_seconds = time.perf_counter() - validation_start
                    if np.isnan(best_rankic) or (
                        np.isfinite(rankic) and rankic > best_rankic
                    ):
                        best_rankic = float(rankic)
                        best_epoch = epoch
                        best_state = copy.deepcopy(model.state_dict())
                    history_rows.append(
                        {
                            "model": "lstm",
                            "seed": seed,
                            "model_seed": model_seed,
                            "fold": protocol.fold,
                            "epoch": epoch,
                            "mean_train_loss": float(np.mean(losses)),
                            "validation_rankic": float(rankic),
                            "epoch_seconds": time.perf_counter() - epoch_start,
                            "model_step_seconds": model_step_seconds,
                            "validation_seconds": validation_seconds,
                            "stage": "validation",
                            "sealed": False,
                            "history_event": "training_epoch",
                            "history_complete": True,
                        }
                    )
                if best_state is None:
                    raise ContractError("v33 LSTM did not produce a valid checkpoint")
                model.load_state_dict(best_state, strict=True)
                scores, positions = predict_model(
                    model,
                    validation_dataset,
                    batch_size=batch_size,
                    num_workers=0,
                )
                replay_rankic = validation_plan.evaluate(
                    scores, positions
                ).mean_daily_rankic
                if abs(float(replay_rankic) - best_rankic) > 1e-12:
                    raise ContractError("v33 LSTM best checkpoint replay drifted")
                torch.save(best_state, checkpoint)
                prediction_rows.extend(
                    _prediction_rows(
                        scores,
                        positions,
                        model="lstm",
                        seed=seed,
                        fold=protocol.fold,
                        lookup=lookup,
                        contracts=contracts,
                        training_contract_id=contracts[
                            "lstm_training_contract_id"
                        ],
                    )
                )
                elapsed = time.perf_counter() - start
                checkpoint_rows.append(
                    {
                        "model": "lstm",
                        "seed": seed,
                        "model_seed": model_seed,
                        "fold": protocol.fold,
                        "best_epoch": best_epoch,
                        "best_validation_rankic": best_rankic,
                        "training_seconds": elapsed,
                        "parameter_count": sum(
                            parameter.numel() for parameter in model.parameters()
                        ),
                        "checkpoint": f"checkpoints/{checkpoint.name}",
                        "checkpoint_sha256": _sha256(checkpoint),
                        "recovered_from_interruption": False,
                    }
                )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(history_rows),
        pd.DataFrame(checkpoint_rows),
    )


def _report(
    selection: dict[str, object],
    comparison: dict[str, object],
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    summary_lines = [
        (
            f"| {row.model} | {float(cast(Any, row.mean_rankic)):.6f} | "
            f"{float(cast(Any, row.rankic_ir)):.4f} | "
            f"{float(cast(Any, row.mean_top_return)):+.6f} | "
            f"{float(cast(Any, row.mean_top_excess_return)):+.6f} | "
            f"{float(cast(Any, row.mean_top_precision)):.4f} | "
            f"{float(cast(Any, row.mean_ndcg_at_top)):.4f} |"
        )
        for row in summary.itertuples(index=False)
    ]
    bootstrap_lines = [
        (
            f"| {row.metric} | {float(cast(Any, row.paired_mean_delta)):+.6f} | "
            f"{float(cast(Any, row.bootstrap_ci_low)):+.6f} | "
            f"{float(cast(Any, row.bootstrap_ci_high)):+.6f} |"
        )
        for row in bootstrap.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# TCN 任务对齐多指标评测 v33",
            "",
            f"- 决策：`{selection['status']}`",
            f"- 指标赢家共识：`{comparison['winner_consensus']}`",
            f"- 输出/目标/样本契约：`{selection['contract_integrity_passed']}`",
            f"- LSTM 历史重放：`{selection['lstm_replay_passed']}`",
            "- sealed test：未访问、未授权。",
            "- 架构普适优劣推断：未授权；TCN 与 LSTM 训练协议不同。",
            "",
            "## 模型汇总",
            "",
            "| model | RankIC | RankICIR | Top return | Top excess | Top precision | NDCG@Top |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *summary_lines,
            "",
            "## TCN - LSTM 配对差值的日期块 bootstrap",
            "",
            "| metric | mean delta | 95% CI low | 95% CI high |",
            "|---|---:|---:|---:|",
            *bootstrap_lines,
            "",
            "## 解释边界",
            "",
            "RankIC 衡量全股票池排序，Top-return/precision/NDCG 衡量顶部决策。"
            "若方向冲突，本轮结论是评测目标不充分，而不是任选一个指标宣布总赢家。",
            "Top return 是未计交易约束和成本的普通验证诊断，不是可交易净收益。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run immutable v33 task-aligned TCN/LSTM evaluation"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume-temporary", action="store_true")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists():
            raise ContractError("v33 refuses to overwrite experiment artifacts")
        if temporary.exists() and not arguments.resume_temporary:
            raise ContractError(
                "v33 temporary artifact exists; explicit recovery is required"
            )
        config_path = arguments.config.resolve()
        config = cast(
            dict[str, object], json.loads(config_path.read_text(encoding="utf-8"))
        )
        if config.get("protocol_version") != "v33" or _contains_secret_key(config):
            raise ContractError("v33 config identity or secret scan failed")
        seeds = tuple(
            int(cast(Any, value)) for value in cast(list[object], config["seeds"])
        )
        folds = tuple(
            int(cast(Any, value)) for value in cast(list[object], config["folds"])
        )
        if seeds != (7, 17, 27) or folds != (0, 1, 2, 3, 4):
            raise ContractError("v33 requires seeds 7/17/27 and folds 0..4")
        if config.get("precision") != "float32" or int(
            cast(Any, config["num_workers"])
        ) != 0:
            raise ContractError("v33 requires float32 and num_workers=0")
        contracts = {
            str(key): str(value)
            for key, value in cast(dict[str, object], config["contracts"]).items()
        }
        required_contracts = {
            "prediction_contract_id",
            "target_contract_id",
            "evaluation_contract_id",
            "tcn_training_contract_id",
            "lstm_training_contract_id",
        }
        if set(contracts) != required_contracts:
            raise ContractError("v33 contract registry drifted")

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
            raise ContractError("v33 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = {
            str(key): str(value)
            for key, value in cast(dict[str, object], config["source_sha256"]).items()
        }
        if observed_hashes != expected_hashes:
            raise ContractError("v33 source SHA-256 identity drifted")

        parent = (ROOT / str(config["parent_artifact"])).resolve()
        parent_receipt, parent_config = _load_parent(
            parent, str(config["parent_receipt_id"])
        )
        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v33 rejects sealed split rows")
        observed_stages = {
            str(value) for value in raw_split["stage"].tolist()
        }
        if forbidden := sorted(observed_stages.difference({"train", "validation", "purged"})):
            raise ContractError("v33 split contains forbidden stages: " + ", ".join(forbidden))
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(folds)
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()
        if set(split_manifest["fold"].astype(int)) != set(folds):
            raise ContractError("v33 ordinary-validation fold coverage drifted")
        lookup = _label_lookup(labels)

        temporary.mkdir(parents=True, exist_ok=arguments.resume_temporary)
        tcn_predictions, tcn_checkpoints = _replay_tcn(
            features,
            labels,
            split_manifest,
            parent=parent,
            parent_config=parent_config,
            trial_id=str(config["tcn_trial_id"]),
            seeds=seeds,
            lookup=lookup,
            contracts=contracts,
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
            contracts=contracts,
            checkpoint_dir=temporary / "checkpoints",
            resume_existing_checkpoints=arguments.resume_temporary,
        )
        predictions = pd.concat(
            [tcn_predictions, lstm_predictions], ignore_index=True
        )
        validate_prediction_contract(predictions, expected_models=2)
        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(cast(Any, config["top_fraction"]))
        )
        summary = summarize_task_aligned_metrics(metrics)
        comparison = cast(
            dict[str, object],
            compare_task_aligned_models(
                metrics, reference_model="lstm", candidate_model="tcn"
            ),
        )
        bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="lstm",
            candidate_model="tcn",
            seed=int(cast(Any, config["bootstrap_seed"])),
            draws=int(cast(Any, config["bootstrap_draws"])),
        )
        bootstrap_by_metric = bootstrap.set_index("metric")
        tcn_effect_passed = bool(
            float(
                cast(
                    Any, bootstrap_by_metric.loc["rankic", "bootstrap_ci_low"]
                )
            )
            > 0
            and float(
                cast(
                    Any,
                    bootstrap_by_metric.loc[
                        "top_precision", "bootstrap_ci_low"
                    ],
                )
            )
            > 0
            and float(
                cast(
                    Any,
                    bootstrap_by_metric.loc["ndcg_at_top", "bootstrap_ci_low"],
                )
            )
            > 0
            and float(
                cast(
                    Any,
                    bootstrap_by_metric.loc["top_return", "bootstrap_ci_low"],
                )
            )
            > 0
        )

        historical_lstm_frame = pd.read_parquet(
            parent / "lstm-measurements.parquet"
        )
        historical_lstm = historical_lstm_frame.loc[
            historical_lstm_frame["model"].astype(str).eq("lstm")
        ].set_index(["base_seed", "fold"])["best_validation_rankic"].sort_index()
        replay_lstm = lstm_checkpoints.set_index(["seed", "fold"])[
            "best_validation_rankic"
        ].sort_index()
        if not historical_lstm.index.equals(replay_lstm.index):
            raise ContractError("v33 LSTM historical unit coverage drifted")
        lstm_replay_error = float(
            np.max(
                np.abs(
                    historical_lstm.to_numpy(dtype="float64")
                    - replay_lstm.to_numpy(dtype="float64")
                )
            )
        )
        lstm_replay_passed = lstm_replay_error <= float(
            cast(Any, config["lstm_replay_tolerance"])
        )
        if not lstm_replay_passed:
            raise ContractError(
                f"v33 LSTM historical replay mismatch: {lstm_replay_error:.12g}"
            )

        tcn_parent_frame = pd.read_parquet(parent / "tcn-leaderboard.parquet")
        tcn_parent = tcn_parent_frame.loc[
            tcn_parent_frame["trial_id"].astype(str).eq(
                str(config["tcn_trial_id"])
            )
        ].set_index(["seed", "fold"])["best_mean_daily_rankic"].sort_index()
        tcn_replay = (
            metrics.loc[metrics["model"].astype(str).eq("tcn")]
            .groupby(["seed", "fold"], observed=True)["rankic"]
            .mean()
            .sort_index()
        )
        if not tcn_parent.index.equals(tcn_replay.index):
            raise ContractError("v33 TCN parent unit coverage drifted")
        tcn_replay_error = float(
            np.max(
                np.abs(
                    tcn_parent.to_numpy(dtype="float64")
                    - tcn_replay.to_numpy(dtype="float64")
                )
            )
        )
        if tcn_replay_error > 1e-12:
            raise ContractError(
                f"v33 TCN checkpoint replay mismatch: {tcn_replay_error:.12g}"
            )

        consensus = str(comparison["winner_consensus"])
        status = {
            "candidate": "task_aligned_metrics_agree_tcn_v33",
            "reference": "task_aligned_metrics_agree_lstm_v33",
            "mixed": "task_aligned_metrics_mixed_v33",
        }[consensus]
        contract_audit: dict[str, object] = {
            "output_semantics_aligned": True,
            "sample_coverage_aligned": True,
            "target_contract_aligned": True,
            "evaluation_contract_aligned": True,
            "training_contract_aligned": False,
            "training_contracts_explicit": True,
            "architecture_superiority_inference_authorized": False,
            "lstm_replay_max_abs_error": lstm_replay_error,
            "tcn_replay_max_abs_error": tcn_replay_error,
            "sealed_test_accessed": False,
            "interrupted_checkpoint_recovery_used": bool(
                lstm_checkpoints["recovered_from_interruption"].astype(bool).any()
            ),
            "complete_epoch_history_available": bool(
                lstm_history["history_complete"].astype(bool).all()
            ),
        }
        selection: dict[str, object] = {
            "status": status,
            "contract_integrity_passed": True,
            "lstm_replay_passed": lstm_replay_passed,
            "winner_consensus": consensus,
            "tcn_prediction_effect_passed": tcn_effect_passed,
            "statistical_interpretation": (
                "tcn_superiority_supported"
                if tcn_effect_passed
                else "tcn_superiority_not_supported"
            ),
            "sealed_test_authorized": False,
            "next_step_authorized": False,
        }
        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "daily-metrics.parquet", index=False)
        summary.to_parquet(temporary / "model-summary.parquet", index=False)
        bootstrap.to_parquet(temporary / "paired-bootstrap.parquet", index=False)
        lstm_history.to_parquet(
            temporary / "lstm-epoch-history.parquet", index=False
        )
        pd.concat([tcn_checkpoints, lstm_checkpoints], ignore_index=True).to_parquet(
            temporary / "checkpoint-summary.parquet", index=False
        )
        lstm_checkpoints.to_parquet(
            temporary / "lstm-checkpoint-summary.parquet", index=False
        )
        _write_json(temporary / "contract-audit.json", contract_audit)
        _write_json(temporary / "comparison.json", comparison)
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        (temporary / "report.md").write_text(
            _report(selection, comparison, summary, bootstrap), encoding="utf-8"
        )
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, object] = {
            "schema_version": "tcn-task-aligned-evaluation-v33/v1",
            "run_id": str(config["run_id"]),
            "parent": {
                "path": str(parent),
                "receipt_id": parent_receipt["receipt_id"],
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
            "contract_audit": contract_audit,
            "selection": selection,
            "comparison": comparison,
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
