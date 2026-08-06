"""Run the immutable v34 TCN top-tail task-alignment experiment."""

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


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_v33_parent(
    config: dict[str, object],
    *,
    expected_source_hashes: Mapping[str, str],
) -> tuple[Path, dict[str, object], pd.DataFrame]:
    artifact = _resolve_project_path(config["v33_parent_artifact"])
    receipt_path = artifact / "receipt.json"
    selection_path = artifact / "selection.json"
    predictions_path = artifact / "predictions.parquet"
    if not all(path.is_file() for path in (receipt_path, selection_path, predictions_path)):
        raise ContractError("v34 v33 parent artifact is incomplete")
    receipt = cast(
        dict[str, object], json.loads(receipt_path.read_text(encoding="utf-8"))
    )
    selection = cast(
        dict[str, object], json.loads(selection_path.read_text(encoding="utf-8"))
    )
    if receipt.get("receipt_id") != config["v33_parent_receipt_id"]:
        raise ContractError("v34 v33 parent receipt identity drifted")
    if selection.get("status") != config["v33_parent_selection_status"]:
        raise ContractError("v34 v33 parent selection status drifted")
    if receipt.get("sealed_test_accessed") is not False or selection.get(
        "sealed_test_authorized"
    ) is not False:
        raise ContractError("v34 v33 parent is not sealed-test fail-closed")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("predictions.parquet") != _sha256(
        predictions_path
    ):
        raise ContractError("v34 v33 prediction output hash drifted")
    source_artifacts = receipt.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise ContractError("v34 v33 source identities are missing")
    for name, expected_hash in expected_source_hashes.items():
        source = source_artifacts.get(name)
        if not isinstance(source, dict) or source.get("sha256") != expected_hash:
            raise ContractError(f"v34 v33 source identity drifted for {name}")
    predictions = pd.read_parquet(predictions_path)
    lstm = predictions.loc[predictions["model"].astype(str).eq("lstm")].copy()
    if lstm.empty:
        raise ContractError("v34 v33 parent contains no LSTM predictions")
    return (
        artifact,
        {
            "path": str(artifact),
            "receipt_id": str(config["v33_parent_receipt_id"]),
            "selection_status": str(config["v33_parent_selection_status"]),
            "predictions_sha256": _sha256(predictions_path),
        },
        lstm,
    )


def _collect_tcn_predictions(
    features: np.ndarray,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    trials: tuple[TCNTuningTrial, ...],
    best_states: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    seeds: tuple[int, ...],
    contracts: dict[str, str],
    model_names: Mapping[str, str],
) -> pd.DataFrame:
    protocols = build_fold_protocols(features, split_manifest)
    targets = np.zeros((len(features), len(HORIZONS)), dtype="float32")
    masks = np.ones_like(targets, dtype="bool")
    lookup = _label_lookup(labels)
    rows: list[dict[str, object]] = []
    for trial in trials:
        model_name = model_names[trial.trial_id]
        training_contract_id = contracts[f"{model_name}_training_contract_id"]
        for seed in seeds:
            for protocol in protocols:
                state_key = f"seed-{seed}-{trial.trial_id}-fold-{protocol.fold}"
                state = best_states.get(state_key)
                if state is None:
                    raise ContractError(f"v34 checkpoint state missing: {state_key}")
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
                        training_contract_id=training_contract_id,
                    )
                )
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ContractError("v34 TCN predictions are empty")
    return predictions


def _bootstrap_row(
    bootstrap: pd.DataFrame, scope: str, metric: str
) -> pd.Series:
    rows = bootstrap.loc[
        bootstrap["scope"].astype(str).eq(scope)
        & bootstrap["metric"].astype(str).eq(metric)
    ]
    if len(rows) != 1:
        raise ContractError(f"v34 bootstrap row missing for {scope}/{metric}")
    return rows.iloc[0]


def _report(
    selection: dict[str, object],
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
            "# TCN Top-tail 任务对齐优化 v34",
            "",
            f"- 决策：`{selection['status']}`",
            f"- 完整性门：`{selection['integrity_passed']}`",
            f"- 机制/梯度门：`{selection['mechanism_passed']}`",
            f"- 预测效果门：`{selection['effect_passed']}`",
            f"- 速度门：`{selection['speed_passed']}`",
            "- sealed test：未访问、未授权。",
            "",
            "本轮唯一训练变量是给冻结 shape-residual TCN 增加固定权重的"
            " top-10% pairwise logistic 分量；checkpoint 仍按 RankIC 选择。",
            "",
            "## 三模型普通验证摘要",
            "",
            "| model | RankIC | top return | top precision | NDCG@top | turnover |",
            "|---|---:|---:|---:|---:|---:|",
            *summary_lines,
            "",
            "## Candidate 相对基准",
            "",
            f"- 相对 control TCN：`{json.dumps(control_comparison, ensure_ascii=False, sort_keys=True)}`",
            f"- 相对 v33 LSTM：`{json.dumps(lstm_comparison, ensure_ascii=False, sort_keys=True)}`",
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
        description="Run the immutable v34 TCN top-tail alignment experiment"
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
            raise ContractError("v34 refuses to overwrite experiment artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v34 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v34":
            raise ContractError("v34 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v34 config contains a secret-like key")
        if config.get("precision") != "float32" or int(
            cast(Any, config["num_workers"])
        ) != 0:
            raise ContractError("v34 requires float32 and num_workers=0")
        seeds = tuple(int(cast(Any, value)) for value in cast(list[object], config["seeds"]))
        if seeds != (7, 17, 27) or cast(list[object], config["folds"]) != [
            0,
            1,
            2,
            3,
            4,
        ]:
            raise ContractError("v34 requires seeds 7/17/27 and folds 0..4")
        if (
            int(cast(Any, config["max_epochs"])) != 8
            or int(cast(Any, config["patience"])) != 2
            or float(cast(Any, config["min_delta"])) != 0.0005
            or float(cast(Any, config["checkpoint_min_delta"])) != 0.0
        ):
            raise ContractError("v34 training budget or selection contract drifted")

        expected_hashes_value = config.get("source_sha256")
        if not isinstance(expected_hashes_value, dict):
            raise ContractError("v34 source identities are missing")
        expected_hashes = {
            str(key): str(value) for key, value in expected_hashes_value.items()
        }
        seed7_parent, seed7_identity = _load_parent(
            config, prefix="seed7", expected_source_hashes=expected_hashes
        )
        confirmation_parent, confirmation_identity = _load_parent(
            config, prefix="confirmation", expected_source_hashes=expected_hashes
        )
        v33_parent, v33_identity, lstm_predictions = _load_v33_parent(
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
            raise ContractError("v34 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if observed_hashes != expected_hashes:
            raise ContractError("v34 source SHA-256 identity drifted")
        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v34 rejects sealed split rows")
        observed_stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(
            observed_stages - {"train", "validation", "purged"}
        ):
            raise ContractError("v34 split contains forbidden stages: " + ", ".join(unknown))
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        control_trial_id = str(config["control_trial_id"])
        candidate_trial_id = str(config["candidate_trial_id"])
        if {trial.trial_id for trial in trials} != {
            control_trial_id,
            candidate_trial_id,
        }:
            raise ContractError("v34 must train exactly the registered pair")
        trials_by_id = {trial.trial_id: trial for trial in trials}
        control = trials_by_id[control_trial_id]
        candidate = trials_by_id[candidate_trial_id]
        if (
            control.strategy != "grouped_smooth_l1"
            or control.grouped_smooth_l1_reduction != "label_mean"
            or candidate.strategy != "top_tail"
            or candidate.top_tail_weight != 0.05
            or candidate.top_tail_fraction != 0.1
            or candidate.top_tail_temperature != 0.1
            or control.date_batch_order != "fixed_once"
            or candidate.date_batch_order != "fixed_once"
        ):
            raise ContractError("v34 registered loss identity drifted")
        control_contract = dict(control.__dict__)
        candidate_contract = dict(candidate.__dict__)
        for contract in (control_contract, candidate_contract):
            contract.pop("trial_id")
            contract.pop("strategy")
        if control_contract != candidate_contract:
            raise ContractError("v34 trials differ by more than the objective")

        control_states, control_manifest = _load_frozen_parent_states(
            seed7_parent, confirmation_parent, control_trial_id
        )
        candidate_states, candidate_manifest = _load_frozen_parent_states(
            seed7_parent, confirmation_parent, candidate_trial_id
        )
        control_manifest.insert(0, "trial_id", control_trial_id)
        candidate_manifest.insert(0, "trial_id", candidate_trial_id)
        checkpoint_manifest = pd.concat(
            [control_manifest, candidate_manifest], ignore_index=True
        )
        frozen_states = {
            seed: {**control_states[seed], **candidate_states[seed]} for seed in seeds
        }
        identities = {
            "data": observed_hashes["features"],
            "fold_manifest": observed_hashes["split_manifest"],
            "evaluation": observed_hashes["labels"],
        }
        tuning_parts = []
        best_states: dict[str, dict[str, torch.Tensor]] = {}
        for seed in seeds:
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
                protocol_identities=identities,
                frozen_parent_states=frozen_states[seed],
            )
            tuning_parts.append(tuning)
            for key, state in tuning.best_states.items():
                best_states[f"seed-{seed}-{key}"] = state
        epoch_history = pd.concat(
            [part.epoch_history for part in tuning_parts], ignore_index=True
        )
        leaderboard = pd.concat(
            [part.leaderboard for part in tuning_parts], ignore_index=True
        ).merge(
            checkpoint_manifest[
                ["trial_id", "seed", "fold", "parent_checkpoint_sha256"]
            ],
            on=["trial_id", "seed", "fold"],
            how="left",
            validate="one_to_one",
        )

        contracts_value = config.get("contracts")
        if not isinstance(contracts_value, dict):
            raise ContractError("v34 prediction contracts are missing")
        contracts = {str(key): str(value) for key, value in contracts_value.items()}
        tcn_predictions = _collect_tcn_predictions(
            features,
            labels,
            split_manifest,
            trials,
            best_states,
            seeds=seeds,
            contracts=contracts,
            model_names={
                control_trial_id: "control",
                candidate_trial_id: "candidate",
            },
        )
        if set(lstm_predictions["training_contract_id"].astype(str)) != {
            contracts["lstm_training_contract_id"]
        }:
            raise ContractError("v34 LSTM training contract drifted")
        for column in (
            "prediction_contract_id",
            "target_contract_id",
            "evaluation_contract_id",
        ):
            expected = contracts[column]
            if set(lstm_predictions[column].astype(str)) != {expected}:
                raise ContractError(f"v34 LSTM {column} drifted")
        predictions = pd.concat(
            [tcn_predictions, lstm_predictions], ignore_index=True
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
        metric_columns = (
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
        control_bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="control",
            candidate_model="candidate",
            metric_columns=metric_columns,
            seed=int(cast(Any, config["bootstrap_seed"])),
            draws=int(cast(Any, config["bootstrap_draws"])),
        )
        control_bootstrap.insert(0, "scope", "candidate-minus-control")
        lstm_bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="lstm",
            candidate_model="candidate",
            metric_columns=metric_columns,
            seed=int(cast(Any, config["bootstrap_seed"])) + 1,
            draws=int(cast(Any, config["bootstrap_draws"])),
        )
        lstm_bootstrap.insert(0, "scope", "candidate-minus-lstm")
        bootstrap = pd.concat(
            [control_bootstrap, lstm_bootstrap], ignore_index=True
        )

        tcn_metric_scores = (
            metrics.loc[metrics["model"].isin(["control", "candidate"])]
            .groupby(["model", "seed", "fold"], observed=True)["rankic"]
            .mean()
            .sort_index()
        )
        model_trial_map = {
            "control": control_trial_id,
            "candidate": candidate_trial_id,
        }
        leaderboard_scores = leaderboard.set_index(["trial_id", "seed", "fold"])[
            "best_mean_daily_rankic"
        ]
        replay_errors = []
        for row in tcn_metric_scores.rename("rankic").reset_index().itertuples(
            index=False
        ):
            model = str(cast(Any, row.model))
            seed = int(cast(Any, row.seed))
            fold = int(cast(Any, row.fold))
            replay_errors.append(
                abs(
                    float(cast(Any, row.rankic))
                    - float(
                        cast(
                            Any,
                            leaderboard_scores.loc[
                                (model_trial_map[model], seed, fold)
                            ],
                        )
                    )
                )
            )
        rankic_replay_max_abs_error = max(replay_errors)
        if rankic_replay_max_abs_error > 1e-12:
            raise ContractError("v34 task-aligned RankIC replay drifted")

        _, _, lstm_measurements, lstm_environment = _historical_evidence(
            seed7_parent,
            confirmation_parent,
            control_trial_id=str(config["historical_control_trial_id"]),
            parent_candidate_trial_id=str(config["historical_parent_trial_id"]),
        )
        candidate_rows = leaderboard.loc[
            leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
        ].copy()
        speed_comparison = build_tcn_lstm_comparison(
            candidate_rows, lstm_measurements
        )
        control_rows = leaderboard.loc[
            leaderboard["trial_id"].astype(str).eq(control_trial_id)
        ]
        throughput_ratio = float(candidate_rows["samples_per_second"].median()) / float(
            control_rows["samples_per_second"].median()
        )

        gradient_diagnostics = epoch_history.loc[
            epoch_history["trial_id"].astype(str).eq(candidate_trial_id)
            & epoch_history["stage"].astype(str).eq("validation"),
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
        gradient_cosine_median = float(
            candidate_rows["median_component_gradient_cosine"].median()
        )
        gates = cast(dict[str, object], config["gates"])
        expected_loss_identity = "smooth-l1+0.05-top-tail-fraction-0.1-tau-0.1"
        integrity_passed = bool(
            len(leaderboard) == 30
            and not leaderboard.duplicated(["trial_id", "seed", "fold"]).any()
            and set(leaderboard["trainable_parameter_count"].astype(int)) == {88}
            and leaderboard["frozen_parent_state_drift_max"].eq(0.0).all()
            and leaderboard["parent_prediction_max_abs_error"].eq(0.0).all()
            and not predictions["sealed"].astype(bool).any()
            and rankic_replay_max_abs_error <= 1e-12
        )
        mechanism_passed = bool(
            set(candidate_rows["loss_identity"].astype(str))
            == {expected_loss_identity}
            and set(candidate_rows["strategy"].astype(str)) == {"top_tail"}
            and candidate_rows["top_tail_weight"].eq(0.05).all()
            and candidate_rows["top_tail_fraction"].eq(0.1).all()
            and candidate_rows["top_tail_temperature"].eq(0.1).all()
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
            throughput_ratio
            >= float(
                cast(Any, gates["min_candidate_control_throughput_ratio"])
            )
            and float(speed_comparison["model_step_speed_ratio"])
            >= float(cast(Any, gates["min_model_step_speed_ratio"]))
            and float(speed_comparison["end_to_end_speed_ratio"])
            >= float(cast(Any, gates["min_end_to_end_speed_ratio"]))
        )
        blockers = []
        if not integrity_passed:
            blockers.append("integrity")
        if not mechanism_passed:
            blockers.append("mechanism_or_gradient_conflict")
        if not effect_passed:
            blockers.append("task_aligned_prediction_effect")
        if not speed_passed:
            blockers.append("speed")
        if not integrity_passed:
            status = "stop_top_tail_integrity_v34"
        elif not mechanism_passed:
            status = "stop_top_tail_mechanism_v34"
        elif not effect_passed:
            status = "stop_top_tail_no_gain_v34"
        elif not speed_passed:
            status = "stop_top_tail_speed_v34"
        else:
            status = "top_tail_ordinary_validation_candidate_v34"
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
            "control_trial_id": control_trial_id,
            "candidate_trial_id": candidate_trial_id,
            "rankic_replay_max_abs_error": rankic_replay_max_abs_error,
            "component_gradient_cosine_median": gradient_cosine_median,
            "candidate_control_throughput_ratio": throughput_ratio,
            "model_step_speed_ratio": float(speed_comparison["model_step_speed_ratio"]),
            "end_to_end_speed_ratio": float(speed_comparison["end_to_end_speed_ratio"]),
            "robust_tail_improvement": robust_tail_improvement,
            "blockers": blockers,
        }

        temporary.mkdir(parents=True)
        epoch_history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap-summary.parquet", index=False)
        gradient_diagnostics.to_parquet(
            temporary / "gradient-conflict-diagnostics.parquet", index=False
        )
        checkpoint_manifest.to_parquet(
            temporary / "parent-checkpoint-manifest.parquet", index=False
        )
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
                summary,
                control_comparison,
                lstm_comparison,
                bootstrap,
            ),
            encoding="utf-8",
        )
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for checkpoint_key, state in best_states.items():
            torch.save(state, checkpoint_dir / f"{checkpoint_key}.pt")
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-top-tail-alignment-v34/v1",
            "run_id": str(config["run_id"]),
            "parents": {
                "seed7": seed7_identity,
                "confirmation": confirmation_identity,
                "v33": v33_identity,
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
