"""Confirm the v40 relative TCN on seeds 17/27 and a same-budget LSTM."""

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

from skill_dl_tcn_shortterm.backtest import build_executable_long_only  # noqa: E402
from skill_dl_tcn_shortterm.experiment import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
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
from skill_dl_tcn_shortterm.v40_validation import (  # noqa: E402
    decide_v40_model_gate,
    decide_v40_strategy_gate,
    frozen_training_contracts,
    summarize_cost_sensitivity,
    validate_v40_frozen_predictions,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_relative_feature_validation import (  # noqa: E402
    _collect_tcn_predictions,
)
from run_tcn_task_aligned_evaluation import (  # noqa: E402
    _label_lookup,
    _train_lstm,
)
from run_tcn_v40_model_strategy_boundary import (  # noqa: E402
    _membership_diagnostics,
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
        return any(
            any(
                marker in str(key).lower()
                for marker in ("password", "token", "secret", "credential")
            )
            or _contains_secret_key(nested)
            for key, nested in value.items()
        )
    return isinstance(value, list) and any(_contains_secret_key(item) for item in value)


def _resolve(path_value: object) -> Path:
    path = Path(str(path_value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _bootstrap(
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


def _timing_comparison(
    tcn_history: pd.DataFrame,
    lstm_history: pd.DataFrame,
    train_counts: dict[int, int],
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    rows: list[dict[str, float | int]] = []
    for (seed, fold), tcn_unit in tcn_history.loc[
        tcn_history["variant"].eq("relative")
    ].groupby(["seed", "fold"], observed=True, sort=True):
        seed_int = int(cast(Any, seed))
        fold_int = int(cast(Any, fold))
        lstm_unit = lstm_history.loc[
            lstm_history["seed"].astype(int).eq(seed_int)
            & lstm_history["fold"].astype(int).eq(fold_int)
            & lstm_history["history_event"].astype(str).eq("training_epoch")
        ]
        if lstm_unit.empty:
            raise ContractError("v40 LSTM timing unit is missing")
        tcn_model_seconds = float(tcn_unit["model_step_seconds"].sum())
        tcn_cycle_seconds = float(
            (tcn_unit["training_seconds"] + tcn_unit["validation_seconds"]).sum()
        )
        lstm_model_seconds = float(lstm_unit["model_step_seconds"].sum())
        lstm_cycle_seconds = float(lstm_unit["epoch_seconds"].sum())
        epoch_count = int(len(tcn_unit))
        if epoch_count != len(lstm_unit):
            raise ContractError("v40 TCN/LSTM epoch budget drifted")
        sample_count = int(train_counts[fold_int] * epoch_count)
        times = np.asarray(
            [
                tcn_model_seconds,
                tcn_cycle_seconds,
                lstm_model_seconds,
                lstm_cycle_seconds,
            ],
            dtype="float64",
        )
        if not np.isfinite(times).all() or bool((times <= 0).any()):
            raise ContractError("v40 timing evidence is invalid")
        rows.append(
            {
                "seed": seed_int,
                "fold": fold_int,
                "epochs": epoch_count,
                "training_samples": sample_count,
                "tcn_model_step_seconds": tcn_model_seconds,
                "tcn_complete_cycle_seconds": tcn_cycle_seconds,
                "lstm_model_step_seconds": lstm_model_seconds,
                "lstm_complete_cycle_seconds": lstm_cycle_seconds,
                "tcn_model_step_samples_per_second": sample_count
                / tcn_model_seconds,
                "lstm_model_step_samples_per_second": sample_count
                / lstm_model_seconds,
                "tcn_complete_cycle_samples_per_second": sample_count
                / tcn_cycle_seconds,
                "lstm_complete_cycle_samples_per_second": sample_count
                / lstm_cycle_seconds,
                "model_step_speed_ratio": lstm_model_seconds / tcn_model_seconds,
                "end_to_end_speed_ratio": lstm_cycle_seconds / tcn_cycle_seconds,
            }
        )
    units = pd.DataFrame(rows).sort_values(["seed", "fold"], ignore_index=True)
    expected_units = {(seed, fold) for seed in (7, 17, 27) for fold in range(5)}
    if set(units[["seed", "fold"]].itertuples(index=False, name=None)) != expected_units:
        raise ContractError("v40 timing seed/fold coverage drifted")
    summary: dict[str, float | int | str] = {
        "unit_count": len(units),
        "model_step_speed_ratio_geomean": float(
            np.exp(np.log(units["model_step_speed_ratio"]).mean())
        ),
        "end_to_end_speed_ratio_geomean": float(
            np.exp(np.log(units["end_to_end_speed_ratio"]).mean())
        ),
        "model_step_speed_ratio_median": float(
            units["model_step_speed_ratio"].median()
        ),
        "end_to_end_speed_ratio_median": float(
            units["end_to_end_speed_ratio"].median()
        ),
        "timing_scope": "same-data same-fold same-seed eight-epoch CPU",
    }
    return units, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run v40 top50 relative TCN multiseed/LSTM confirmation"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v40 phase B refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v40 phase-B config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v40-phase-b":
            raise ContractError("v40 phase-B protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v40 phase-B config contains a secret-like key")
        if config.get("precision") != "float32" or int(
            cast(Any, config["num_workers"])
        ) != 0:
            raise ContractError("v40 phase B requires float32 and num_workers=0")
        confirmation_seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["confirmation_seeds"])
        )
        all_seeds = tuple(
            int(cast(Any, value)) for value in cast(list[object], config["all_seeds"])
        )
        folds = tuple(
            int(cast(Any, value)) for value in cast(list[object], config["folds"])
        )
        if confirmation_seeds != (17, 27) or all_seeds != (7, 17, 27) or folds != (
            0,
            1,
            2,
            3,
            4,
        ):
            raise ContractError("v40 phase-B seed/fold identities drifted")

        base = _resolve(config["base_run_dir"])
        relative = _resolve(config["relative_feature_dir"])
        phase_a = _resolve(config["phase_a_run_dir"])
        parent = _resolve(config["v39_parent_run_dir"])
        split_path = _resolve(config["split_manifest"])
        source_paths = {
            "base_features": base / "feature-windows.npy",
            "base_window_index": base / "window-index.parquet",
            "relative_features": relative / "feature-windows.npy",
            "relative_window_index": relative / "window-index.parquet",
            "relative_manifest": relative / "manifest.json",
            "relative_receipt": relative / "receipt.json",
            "labels": base / "labels.parquet",
            "split_manifest": split_path,
            "seed7_predictions": parent / "predictions.parquet",
            "seed7_leaderboard": parent / "tcn-leaderboard.parquet",
            "seed7_history": parent / "tcn-epoch-history.parquet",
            "v39_receipt": parent / "receipt.json",
            "phase_a_model_gate": phase_a / "model-gate.json",
            "phase_a_receipt": phase_a / "receipt.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v40 phase-B sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v40 phase-B source SHA-256 identity drifted")
        phase_a_gate = json.loads(
            source_paths["phase_a_model_gate"].read_text(encoding="utf-8")
        )
        if not isinstance(phase_a_gate, dict) or not (
            phase_a_gate.get("admitted") is True
            and phase_a_gate.get("phase_b_authorized") is True
            and phase_a_gate.get("sealed_test_accessed") is False
        ):
            raise ContractError("v40 phase B is not authorized by phase A")
        for receipt_name in ("v39_receipt", "phase_a_receipt"):
            parent_receipt = json.loads(
                source_paths[receipt_name].read_text(encoding="utf-8")
            )
            if not isinstance(parent_receipt, dict) or parent_receipt.get(
                "sealed_test_accessed"
            ) is not False:
                raise ContractError(f"v40 phase-B {receipt_name} is not fail-closed")

        base_features = np.load(
            source_paths["base_features"], mmap_mode="r", allow_pickle=False
        )
        relative_features = np.load(
            source_paths["relative_features"], mmap_mode="r", allow_pickle=False
        )
        if base_features.shape[1:] != (8, 480) or relative_features.shape != (
            base_features.shape[0],
            10,
            480,
        ):
            raise ContractError("v40 phase-B feature shapes drifted")
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
            raise ContractError("v40 phase-B sample identities drifted")
        for name in ("relative_manifest", "relative_receipt"):
            value = json.loads(source_paths[name].read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("sealed_test_accessed") is not False:
                raise ContractError(f"v40 phase-B {name} is not fail-closed")

        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v40 phase B rejects sealed split rows")
        stages = {str(stage) for stage in raw_split["stage"].tolist()}
        if unknown := sorted(stages - {"train", "validation", "purged"}):
            raise ContractError("v40 phase-B forbidden stages: " + ", ".join(unknown))
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(folds)
            & raw_split["stage"].astype(str).isin(["train", "validation"])
        ].copy()
        trials = parse_real_tcn_trials(config["trials"])
        if len(trials) != 1:
            raise ContractError("v40 phase B fixes exactly one TCN trial")
        trial = trials[0]
        if trial.model_kind != "dynamic_horizon_skip" or trial.strategy != "smooth_l1":
            raise ContractError("v40 phase-B TCN contract drifted")

        lookup = _label_lookup(labels)
        contracts = cast(dict[str, str], config["contracts"])
        temporary.mkdir(parents=True)
        checkpoint_root = temporary / "checkpoints"
        checkpoint_root.mkdir()
        seed7_predictions = pd.read_parquet(source_paths["seed7_predictions"])
        validate_prediction_contract(seed7_predictions, expected_models=2)
        parent_training_contracts = frozen_training_contracts(
            seed7_predictions,
            expected_models=("base_tcn", "relative_tcn"),
        )
        prediction_frames = [seed7_predictions]
        leaderboard_frames = [pd.read_parquet(source_paths["seed7_leaderboard"])]
        history_frames = [pd.read_parquet(source_paths["seed7_history"])]
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
            variant_contracts["tcn_training_contract_id"] = parent_training_contracts[
                f"{variant}_tcn"
            ]
            for seed in confirmation_seeds:
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
                    checkpoint_min_delta=float(
                        cast(Any, config["checkpoint_min_delta"])
                    ),
                    torch_threads=int(cast(Any, config["torch_threads"])),
                    protocol_identities={
                        "data": data_identity,
                        "fold_manifest": observed_hashes["split_manifest"],
                        "evaluation": observed_hashes["labels"],
                    },
                    capture_epoch_states=True,
                    disable_early_stopping=True,
                )
                leaderboard = tuning.leaderboard.copy()
                leaderboard["variant"] = variant
                leaderboard_frames.append(leaderboard)
                history = tuning.epoch_history.copy()
                history["variant"] = variant
                history_frames.append(history)
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
        lstm_contracts = dict(contracts)
        lstm_contracts["lstm_training_contract_id"] = (
            "top50-relative10-lstm-smooth-l1-v40"
        )
        lstm_predictions, lstm_history, lstm_checkpoints = _train_lstm(
            relative_features,
            relative_index,
            labels,
            split_manifest,
            seeds=all_seeds,
            hidden_size=int(cast(Any, lstm_config["hidden_size"])),
            learning_rate=float(cast(Any, lstm_config["learning_rate"])),
            batch_size=int(cast(Any, lstm_config["batch_size"])),
            epochs=int(cast(Any, lstm_config["epochs"])),
            torch_threads=int(cast(Any, config["torch_threads"])),
            lookup=lookup,
            contracts=lstm_contracts,
            checkpoint_dir=checkpoint_root / "relative-lstm",
        )
        lstm_predictions["model"] = "relative_lstm"
        training_cache = temporary / "training-cache"
        training_cache.mkdir()
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            training_cache / "tcn-predictions.parquet", index=False
        )
        pd.concat(leaderboard_frames, ignore_index=True).to_parquet(
            training_cache / "tcn-leaderboard.parquet", index=False
        )
        pd.concat(history_frames, ignore_index=True).to_parquet(
            training_cache / "tcn-epoch-history.parquet", index=False
        )
        lstm_predictions.to_parquet(
            training_cache / "lstm-predictions.parquet", index=False
        )
        lstm_history.to_parquet(
            training_cache / "lstm-epoch-history.parquet", index=False
        )
        lstm_checkpoints.to_parquet(
            training_cache / "lstm-checkpoints.parquet", index=False
        )
        predictions = pd.concat([*prediction_frames, lstm_predictions], ignore_index=True)
        validate_prediction_contract(predictions, expected_models=3)
        validate_v40_frozen_predictions(
            predictions,
            expected_models=("base_tcn", "relative_tcn", "relative_lstm"),
            expected_seeds=all_seeds,
            expected_folds=folds,
            expected_horizons=(1, 2, 3, 5),
        )
        metrics = evaluate_task_aligned_predictions(
            predictions, top_fraction=float(cast(Any, config["top_fraction"]))
        )
        summary = summarize_task_aligned_metrics(metrics)
        pairs = {
            "relative_tcn_minus_base_tcn": ("base_tcn", "relative_tcn"),
            "relative_tcn_minus_relative_lstm": ("relative_lstm", "relative_tcn"),
        }
        comparisons = {
            name: compare_task_aligned_models(
                metrics, reference_model=reference, candidate_model=candidate
            )
            for name, (reference, candidate) in pairs.items()
        }
        bootstrap = pd.concat(
            [
                _bootstrap(
                    metrics,
                    reference_model=reference,
                    candidate_model=candidate,
                    scope=name,
                    seed=int(cast(Any, config["bootstrap_seed"])),
                    draws=int(cast(Any, config["bootstrap_draws"])),
                )
                for name, (reference, candidate) in pairs.items()
            ],
            ignore_index=True,
        )
        leaderboard = pd.concat(leaderboard_frames, ignore_index=True)
        history = pd.concat(history_frames, ignore_index=True)
        base_speed = float(
            leaderboard.loc[leaderboard["variant"].eq("base"), "samples_per_second"].median()
        )
        relative_speed = float(
            leaderboard.loc[
                leaderboard["variant"].eq("relative"), "samples_per_second"
            ].median()
        )
        model_gate = decide_v40_model_gate(
            leaderboard,
            comparisons["relative_tcn_minus_base_tcn"],
            bootstrap,
            seeds=all_seeds,
            folds=folds,
            base_variant="base",
            candidate_variant="relative",
            base_median_samples_per_second=base_speed,
            candidate_median_samples_per_second=relative_speed,
            gates=cast(dict[str, float | int], config["model_gates"]),
            admitted_status="top50_relative_model_multiseed_admitted_v40",
            rejected_status="stop_top50_relative_model_multiseed_v40",
        )
        protocols = build_fold_protocols(relative_features, split_manifest)
        train_counts = {protocol.fold: len(protocol.train_positions) for protocol in protocols}
        timing_units, timing_summary = _timing_comparison(
            history, lstm_history, train_counts
        )

        minimal_predictions = predictions.loc[
            predictions["model"].isin(["base_tcn", "relative_tcn"]),
            [
                "model",
                "seed",
                "fold",
                "sample_id",
                "instrument_id",
                "signal_date",
                "horizon",
                "score",
            ],
        ]
        portfolio_frames: dict[str, list[pd.DataFrame]] = {
            "ledger": [],
            "holdings": [],
            "summary": [],
            "cost": [],
            "membership": [],
        }
        for policy in cast(list[dict[str, object]], config["policies"]):
            policy_name = str(policy["name"])
            buffer_fraction = float(cast(Any, policy["incumbent_buffer_fraction"]))
            for seed in all_seeds:
                seed_predictions = minimal_predictions.loc[
                    minimal_predictions["seed"].astype(int).eq(seed)
                ].drop(columns="seed")
                result = build_executable_long_only(
                    seed_predictions,
                    labels,
                    top_fraction=float(cast(Any, config["top_fraction"])),
                    incumbent_buffer_fraction=buffer_fraction,
                )
                for key, frame in {
                    "ledger": result.portfolio_ledger,
                    "holdings": result.portfolio_holdings,
                    "summary": result.metrics,
                    "cost": summarize_cost_sensitivity(
                        result,
                        cost_bps=tuple(
                            float(cast(Any, value))
                            for value in cast(list[object], config["cost_bps"])
                        ),
                    ),
                    "membership": _membership_diagnostics(
                        result.orders, policy=policy_name
                    ),
                }.items():
                    output = frame.copy()
                    output["seed"] = seed
                    output["policy"] = policy_name
                    portfolio_frames[key].append(output)
        portfolio_outputs = {
            key: pd.concat(frames, ignore_index=True)
            for key, frames in portfolio_frames.items()
        }
        strategy_config = cast(dict[str, object], config["strategy_gate"])
        strategy_gate = decide_v40_strategy_gate(
            portfolio_outputs["cost"],
            policy="raw_topk",
            reference_model="base_tcn",
            candidate_model="relative_tcn",
            one_way_cost_bps=float(cast(Any, strategy_config["one_way_cost_bps"])),
            max_mean_one_way_turnover_delta=float(
                cast(Any, strategy_config["max_mean_one_way_turnover_delta"])
            ),
            min_mean_net_return_delta=float(
                cast(Any, strategy_config["min_mean_net_return_delta"])
            ),
        )

        model_gate_value = {
            "status": model_gate.status,
            "admitted": model_gate.admitted,
            "blockers": list(model_gate.blockers),
            "evidence": model_gate.evidence,
            "sealed_test_accessed": False,
        }
        strategy_gate_value = {
            "status": strategy_gate.status,
            "admitted": strategy_gate.admitted,
            "blockers": list(strategy_gate.blockers),
            "evidence": strategy_gate.evidence,
            "sealed_test_accessed": False,
        }
        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        metrics.to_parquet(temporary / "task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "task-aligned-summary.parquet", index=False)
        bootstrap.to_parquet(temporary / "bootstrap-summary.parquet", index=False)
        leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        lstm_history.to_parquet(temporary / "lstm-epoch-history.parquet", index=False)
        lstm_checkpoints.to_parquet(temporary / "lstm-checkpoints.parquet", index=False)
        timing_units.to_parquet(temporary / "timing-units.parquet", index=False)
        model_gate.unit_deltas.to_parquet(
            temporary / "model-unit-deltas.parquet", index=False
        )
        strategy_gate.unit_deltas.to_parquet(
            temporary / "strategy-unit-deltas.parquet", index=False
        )
        for key, filename in {
            "ledger": "portfolio-ledger.parquet",
            "holdings": "portfolio-holdings.parquet",
            "summary": "policy-summary.parquet",
            "cost": "cost-sensitivity.parquet",
            "membership": "membership-diagnostics.parquet",
        }.items():
            portfolio_outputs[key].to_parquet(temporary / filename, index=False)
        _write_json(temporary / "comparisons.json", comparisons)
        _write_json(temporary / "timing-summary.json", timing_summary)
        _write_json(temporary / "model-gate.json", model_gate_value)
        _write_json(temporary / "strategy-gate.json", strategy_gate_value)
        _write_json(temporary / "config.resolved.json", config)
        report = "\n".join(
            [
                "# TCN v40 多种子与 LSTM 同预算确认",
                "",
                f"- 多种子模型门：`{model_gate.status}`",
                f"- 多种子组合门：`{strategy_gate.status}`",
                f"- TCN/LSTM model-step speed ratio：`{float(timing_summary['model_step_speed_ratio_geomean']):.4f}x`",
                f"- TCN/LSTM end-to-end speed ratio：`{float(timing_summary['end_to_end_speed_ratio_geomean']):.4f}x`",
                f"- TCN-LSTM RankIC delta：`{float(comparisons['relative_tcn_minus_relative_lstm']['mean_rankic_delta']):+.6f}`",
                "- membership turnover：仅诊断；组合门使用证券级 executable turnover。",
                "- sealed_test_accessed：`false`",
                "- 结论上限：ordinary-validation research evidence；not alpha-ready。",
                "",
            ]
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, object] = {
            "schema_version": "tcn-v40-multiseed-lstm-confirmation/v1",
            "run_id": str(config["run_id"]),
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
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
            },
            "model_gate": model_gate_value,
            "strategy_gate": strategy_gate_value,
            "comparisons": comparisons,
            "timing": timing_summary,
            "outputs": outputs,
            "sealed_test_accessed": False,
            "sealed_test_authorized": False,
            "alpha_ready": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload = {
            "status": "success",
            "model_gate": model_gate.status,
            "strategy_gate": strategy_gate.status,
            "timing": timing_summary,
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
