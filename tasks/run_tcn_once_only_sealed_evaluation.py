"""Consume the frozen v35 sealed test exactly once after exact authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, cast
import uuid

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.neural import HORIZONS, RecurrentRegressor  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials  # noqa: E402
from skill_dl_tcn_shortterm.runtime import torch_thread_scope  # noqa: E402
from skill_dl_tcn_shortterm.sealed_evaluation import (  # noqa: E402
    bootstrap_paired_daily,
    claim_sealed_consumption,
    complete_sealed_consumption,
    decide_sealed_candidate,
    paired_daily_unit_mean,
    summarize_paired_daily,
)
from skill_dl_tcn_shortterm.sealed_readiness import (  # noqa: E402
    EXACT_SEALED_AUTHORIZATION_SHA256,
    canonical_bytes,
    require_exact_sealed_authorization,
    sha256_file,
    validate_task_aligned_freeze_config,
    verify_receipt_identity,
)
from skill_dl_tcn_shortterm.task_aligned_evaluation import (  # noqa: E402
    evaluate_task_aligned_predictions,
    summarize_task_aligned_metrics,
    validate_prediction_contract,
)
from skill_dl_tcn_shortterm.training_data import (  # noqa: E402
    LazyWindowDataset,
    build_fold_protocols,
    predict_model,
)
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    TCNTuningTrial,
    _predict_tcn_trial,
    build_tcn_trial_model,
)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{description} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_readiness(readiness_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_json(readiness_dir / "receipt.json", "v36 readiness receipt")
    verify_receipt_identity(receipt, str(receipt.get("receipt_id", "")))
    if (
        receipt.get("status") != "awaiting_explicit_sealed_authorization_v36"
        or receipt.get("authorization_received") is not False
        or receipt.get("sealed_test_accessed") is not False
        or receipt.get("evaluation_executed") is not False
    ):
        raise ContractError("v36 readiness receipt is not awaiting authorization")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ContractError("v36 readiness outputs are missing")
    for name, expected in outputs.items():
        # state.json is intentionally mutable after the one-time run. Its
        # pre-consumption identity is checked below through exact state fields.
        if str(name) == "state.json":
            continue
        path = readiness_dir / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"v36 readiness output fingerprint drifted: {name}")
    frozen = _read_json(readiness_dir / "frozen-plan.json", "v36 frozen plan")
    payload = frozen.get("payload")
    if not isinstance(payload, dict):
        raise ContractError("v36 frozen payload is invalid")
    observed = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if observed != frozen.get("freeze_id") or observed != receipt.get("freeze_id"):
        raise ContractError("v36 frozen identity drifted")
    state = _read_json(readiness_dir / "state.json", "v36 readiness state")
    if (
        state.get("freeze_id") != observed
        or state.get("status") != "awaiting_explicit_sealed_authorization_v36"
        or int(state.get("attempt", -1)) != 0
        or state.get("consumed_marker_created") is not False
    ):
        raise ContractError("v36 readiness state is not consumable")
    return frozen, receipt


def _resolve_checkpoint(artifact: Path, value: object) -> Path:
    path = (artifact / Path(str(value).replace("\\", "/"))).resolve()
    try:
        path.relative_to(artifact.resolve())
    except ValueError as exc:
        raise ContractError("sealed checkpoint escapes its frozen artifact") from exc
    return path


def _preflight_models(
    plan: pd.DataFrame,
    *,
    candidate_artifact: Path,
    lstm_artifact: Path,
    trial: TCNTuningTrial,
    feature_count: int,
    input_steps: int,
    lstm_hidden_size: int,
) -> None:
    checked: set[tuple[str, str]] = set()
    for row in plan.itertuples(index=False):
        typed = cast(Any, row)
        for model_name in ("control", "candidate"):
            path = _resolve_checkpoint(
                candidate_artifact, getattr(typed, f"{model_name}_checkpoint")
            )
            expected = str(getattr(typed, f"{model_name}_checkpoint_sha256"))
            key = (str(path), expected)
            if key in checked:
                continue
            if not path.is_file() or sha256_file(path) != expected:
                raise ContractError(f"sealed {model_name} checkpoint drifted")
            model = build_tcn_trial_model(
                trial, feature_count=feature_count, input_steps=input_steps
            )
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            checked.add(key)
        path = _resolve_checkpoint(lstm_artifact, typed.lstm_checkpoint)
        expected = str(typed.lstm_checkpoint_sha256)
        key = (str(path), expected)
        if key not in checked:
            if not path.is_file() or sha256_file(path) != expected:
                raise ContractError("sealed LSTM checkpoint drifted")
            model = RecurrentRegressor("lstm", feature_count, lstm_hidden_size)
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            checked.add(key)


def _label_lookup(labels: pd.DataFrame) -> dict[tuple[int, int], tuple[object, ...]]:
    required = {
        "sample_position",
        "sample_id",
        "instrument_id",
        "signal_date",
        "horizon",
        "rank_target",
        "raw_return",
        "label_version",
        "valid",
    }
    if missing := required - set(labels):
        raise ContractError(f"sealed labels missing columns: {sorted(missing)}")
    lookup: dict[tuple[int, int], tuple[object, ...]] = {}
    for row in labels.loc[labels["valid"].astype(bool)].itertuples(index=False):
        typed = cast(Any, row)
        key = (int(typed.sample_position), int(typed.horizon))
        if key in lookup:
            raise ContractError("sealed labels contain duplicate position/horizon keys")
        lookup[key] = (
            str(typed.sample_id),
            str(typed.instrument_id),
            str(typed.signal_date),
            float(typed.rank_target),
            float(typed.raw_return),
            str(typed.label_version),
        )
    return lookup


def _prediction_rows(
    scores: np.ndarray,
    positions: np.ndarray,
    *,
    model: str,
    seed: int,
    evaluation_fold: int,
    sealed_fold: int,
    training_fold: int,
    lookup: Mapping[tuple[int, int], tuple[object, ...]],
    contracts: Mapping[str, str],
    training_contract_id: str,
) -> list[dict[str, object]]:
    if scores.shape != (len(positions), len(HORIZONS)):
        raise ContractError("sealed prediction score shape drifted")
    rows: list[dict[str, object]] = []
    for row_index, sample_position in enumerate(positions):
        for horizon_index, horizon in enumerate(HORIZONS):
            label = lookup.get((int(sample_position), int(horizon)))
            if label is None:
                continue
            sample_id, instrument_id, signal_date, rank_target, raw_return, version = label
            if str(version) != contracts["target_contract_id"]:
                raise ContractError("sealed target contract drifted")
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "fold": evaluation_fold,
                    "sealed_fold": sealed_fold,
                    "training_fold": training_fold,
                    "sample_id": sample_id,
                    "instrument_id": instrument_id,
                    "signal_date": signal_date,
                    "horizon": int(horizon),
                    "score": float(scores[row_index, horizon_index]),
                    "rank_target": rank_target,
                    "raw_return": raw_return,
                    "stage": "test",
                    "sealed": True,
                    "prediction_contract_id": contracts["prediction_contract_id"],
                    "target_contract_id": contracts["target_contract_id"],
                    "evaluation_contract_id": contracts["evaluation_contract_id"],
                    "training_contract_id": training_contract_id,
                }
            )
    if not rows:
        raise ContractError("sealed model produced no valid predictions")
    return rows


def _report(
    decision: Mapping[str, object],
    control: Mapping[str, object],
    lstm: Mapping[str, object],
    bootstrap: pd.DataFrame,
) -> str:
    selected = bootstrap.loc[
        bootstrap["reference_model"].astype(str).eq("control")
        & bootstrap["metric"].isin(
            [
                "rankic",
                "top_precision",
                "ndcg_at_top",
                "top_return",
                "net_return_after_cost",
                "top_turnover",
            ]
        )
    ]
    lines = [
        "# TCN v35 一次性 sealed test v36",
        "",
        f"- 最终状态：`{decision['status']}`",
        f"- 候选通过：`{decision['candidate_model']}`",
        "- sealed 已一次性消费；禁止调参后重试。",
        "",
        "## Candidate - control",
        "",
        f"- mean RankIC delta：`{float(cast(Any, control['mean_rankic_delta'])):+.6f}`",
        f"- mean Top precision delta：`{float(cast(Any, control['mean_top_precision_delta'])):+.6f}`",
        f"- mean NDCG delta：`{float(cast(Any, control['mean_ndcg_at_top_delta'])):+.6f}`",
        f"- mean Top return delta：`{float(cast(Any, control['mean_top_return_delta'])):+.6f}`",
        f"- mean cost-after return delta：`{float(cast(Any, control['mean_net_return_after_cost_delta'])):+.6f}`",
        f"- mean turnover delta：`{float(cast(Any, control['mean_top_turnover_delta'])):+.6f}`",
        "",
        "## Candidate - LSTM context",
        "",
        f"- mean RankIC delta：`{float(cast(Any, lstm['mean_rankic_delta'])):+.6f}`",
        f"- mean Top precision delta：`{float(cast(Any, lstm['mean_top_precision_delta'])):+.6f}`",
        f"- mean NDCG delta：`{float(cast(Any, lstm['mean_ndcg_at_top_delta'])):+.6f}`",
        f"- mean Top return delta：`{float(cast(Any, lstm['mean_top_return_delta'])):+.6f}`",
        "",
        "## Candidate - control bootstrap",
        "",
        "| metric | delta | CI low | CI high |",
        "|---|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        typed = cast(Any, row)
        lines.append(
            f"| {typed.metric} | {float(typed.paired_mean_delta):+.6f} | "
            f"{float(typed.bootstrap_ci_low):+.6f} | "
            f"{float(typed.bootstrap_ci_high):+.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen sealed test once")
    parser.add_argument("--readiness-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--authorization-text", required=True)
    arguments = parser.parse_args()
    marker: Path | None = None
    try:
        require_exact_sealed_authorization(arguments.authorization_text)
        readiness_dir = arguments.readiness_dir.resolve()
        output_dir = arguments.output_dir.resolve()
        if output_dir.exists():
            raise ContractError("sealed evaluation refuses to overwrite output")
        frozen, readiness_receipt = _verify_readiness(readiness_dir)
        freeze_id = str(frozen["freeze_id"])
        freeze_payload = cast(dict[str, Any], frozen["payload"])
        config = validate_task_aligned_freeze_config(
            cast(Mapping[str, Any], freeze_payload["config"])
        )
        if config["authorization_text_sha256"] != EXACT_SEALED_AUTHORIZATION_SHA256:
            raise ContractError("sealed authorization fingerprint drifted")

        candidate = cast(Mapping[str, Any], freeze_payload["candidate"])
        lstm_benchmark = cast(Mapping[str, Any], freeze_payload["lstm_benchmark"])
        sources = cast(Mapping[str, Mapping[str, str]], freeze_payload["sources"])
        candidate_artifact = Path(candidate["artifact"]).resolve()
        lstm_artifact = Path(lstm_benchmark["artifact"]).resolve()
        plan_path = readiness_dir / "eligible-checkpoint-plan.parquet"
        if sha256_file(plan_path) != freeze_payload["checkpoint_plan_sha256"]:
            raise ContractError("sealed checkpoint plan drifted")
        plan = pd.read_parquet(plan_path)

        for name, source in sources.items():
            path = Path(source["path"]).resolve()
            if not path.is_file() or sha256_file(path) != source["sha256"]:
                raise ContractError(f"sealed source fingerprint drifted: {name}")
        feature_path = Path(sources["features"]["path"]).resolve()
        feature_header = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if feature_header.ndim != 3:
            raise ContractError("sealed feature tensor must be rank 3")
        feature_count = int(feature_header.shape[1])
        input_steps = int(feature_header.shape[2])
        del feature_header

        v35_config = _read_json(
            candidate_artifact / "config.resolved.json", "v35 resolved config"
        )
        trials = parse_real_tcn_trials(v35_config["trials"])
        if len(trials) != 1:
            raise ContractError("sealed v35 trial identity is ambiguous")
        trial = trials[0]
        if trial.trial_id != v35_config["trajectory_trial_id"]:
            raise ContractError("sealed v35 trajectory identity drifted")
        v33_config = _read_json(
            lstm_artifact / "config.resolved.json", "v33 resolved config"
        )
        lstm_config = cast(Mapping[str, Any], v33_config["lstm"])
        _preflight_models(
            plan,
            candidate_artifact=candidate_artifact,
            lstm_artifact=lstm_artifact,
            trial=trial,
            feature_count=feature_count,
            input_steps=input_steps,
            lstm_hidden_size=int(lstm_config["hidden_size"]),
        )

        state_path = readiness_dir / "state.json"
        state = _read_json(state_path, "v36 readiness state")
        state.update(
            {
                "status": "authorized_preflight_passed_v36",
                "authorization_received": True,
                "authorization_text_sha256": EXACT_SEALED_AUTHORIZATION_SHA256,
            }
        )
        _atomic_json(state_path, state)
        sealed_sha = sources["sealed_split_manifest"]["sha256"]
        marker = claim_sealed_consumption(
            readiness_dir.parent / ".sealed-consumption-v36",
            freeze_id=freeze_id,
            sealed_data_sha256=sealed_sha,
        )
        state.update(
            {
                "status": "sealed_evaluation_running_v36",
                "attempt": 1,
                "sealed_test_accessed": True,
                "consumed_marker_created": True,
                "consumed_marker": str(marker),
            }
        )
        _atomic_json(state_path, state)

        started = time.perf_counter()
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(sources["window_index"]["path"])
        labels = pd.read_parquet(sources["labels"]["path"])
        ordinary_manifest = pd.read_parquet(
            sources["ordinary_split_manifest"]["path"]
        )
        sealed_manifest = pd.read_parquet(sources["sealed_split_manifest"]["path"])
        if len(features) != len(window_index):
            raise ContractError("sealed feature/window sample counts drifted")
        protocols = {
            protocol.fold: protocol
            for protocol in build_fold_protocols(features, ordinary_manifest)
        }
        test_positions = {
            int(cast(Any, fold)): frame["sample_position"].to_numpy(dtype="int64")
            for fold, frame in sealed_manifest.loc[
                sealed_manifest["stage"].astype(str).eq("test")
                & sealed_manifest["sealed"].astype(bool)
            ].groupby("fold", observed=True)
        }
        if set(test_positions) != {0, 1}:
            raise ContractError("sealed canonical test position coverage drifted")
        lookup = _label_lookup(labels)
        targets = np.zeros((len(features), len(HORIZONS)), dtype="float32")
        masks = np.ones_like(targets, dtype="bool")
        v35_contracts = cast(Mapping[str, str], v35_config["contracts"])
        v33_contracts = cast(Mapping[str, str], v33_config["contracts"])
        contracts = {
            "prediction_contract_id": v35_contracts["prediction_contract_id"],
            "target_contract_id": v35_contracts["target_contract_id"],
            "evaluation_contract_id": "once-only-sealed-task-aligned-v36",
            "candidate_training_contract_id": v35_contracts[
                "candidate_training_contract_id"
            ],
            "control_training_contract_id": v35_contracts[
                "control_training_contract_id"
            ],
            "lstm_training_contract_id": v33_contracts[
                "lstm_training_contract_id"
            ],
        }
        prediction_frames: list[pd.DataFrame] = []
        score_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
        with torch_thread_scope(int(v35_config["torch_threads"])):
            for row in plan.itertuples(index=False):
                typed = cast(Any, row)
                sealed_fold = int(typed.sealed_fold)
                training_fold = int(typed.training_fold)
                seed = int(typed.seed)
                evaluation_fold = sealed_fold * 10 + training_fold
                protocol = protocols[training_fold]
                dataset = LazyWindowDataset(
                    features,
                    test_positions[sealed_fold],
                    targets,
                    masks,
                    protocol.feature_mean,
                    protocol.feature_std,
                )
                for model_name in ("control", "candidate"):
                    checkpoint = _resolve_checkpoint(
                        candidate_artifact,
                        getattr(typed, f"{model_name}_checkpoint"),
                    )
                    cache_key = (str(checkpoint), sealed_fold, training_fold)
                    scores_positions = score_cache.get(cache_key)
                    if scores_positions is None:
                        model = build_tcn_trial_model(
                            trial,
                            feature_count=feature_count,
                            input_steps=input_steps,
                        )
                        state_dict = torch.load(
                            checkpoint, map_location="cpu", weights_only=True
                        )
                        model.load_state_dict(state_dict, strict=True)
                        scores_positions = _predict_tcn_trial(
                            model, dataset, batch_size=trial.batch_size
                        )
                        score_cache[cache_key] = scores_positions
                    scores, positions = scores_positions
                    prediction_frames.append(
                        pd.DataFrame(
                            _prediction_rows(
                                scores,
                                positions,
                                model=model_name,
                                seed=seed,
                                evaluation_fold=evaluation_fold,
                                sealed_fold=sealed_fold,
                                training_fold=training_fold,
                                lookup=lookup,
                                contracts=contracts,
                                training_contract_id=contracts[
                                    f"{model_name}_training_contract_id"
                                ],
                            )
                        )
                    )
                checkpoint = _resolve_checkpoint(
                    lstm_artifact, typed.lstm_checkpoint
                )
                cache_key = (str(checkpoint), sealed_fold, training_fold)
                scores_positions = score_cache.get(cache_key)
                if scores_positions is None:
                    model = RecurrentRegressor(
                        "lstm", feature_count, int(lstm_config["hidden_size"])
                    )
                    state_dict = torch.load(
                        checkpoint, map_location="cpu", weights_only=True
                    )
                    model.load_state_dict(state_dict, strict=True)
                    scores_positions = predict_model(
                        model,
                        dataset,
                        batch_size=int(lstm_config["batch_size"]),
                        num_workers=0,
                    )
                    score_cache[cache_key] = scores_positions
                scores, positions = scores_positions
                prediction_frames.append(
                    pd.DataFrame(
                        _prediction_rows(
                            scores,
                            positions,
                            model="lstm",
                            seed=seed,
                            evaluation_fold=evaluation_fold,
                            sealed_fold=sealed_fold,
                            training_fold=training_fold,
                            lookup=lookup,
                            contracts=contracts,
                            training_contract_id=contracts[
                                "lstm_training_contract_id"
                            ],
                        )
                    )
                )

        predictions = pd.concat(prediction_frames, ignore_index=True)
        validate_prediction_contract(
            predictions,
            expected_models=3,
            expected_stage="test",
            allow_sealed=True,
        )
        metrics = evaluate_task_aligned_predictions(
            predictions,
            top_fraction=float(config["top_fraction"]),
            expected_stage="test",
            allow_sealed=True,
        )
        metrics["sealed_fold"] = metrics["fold"].astype(int) // 10
        metrics["training_fold"] = metrics["fold"].astype(int) % 10
        summary = summarize_task_aligned_metrics(metrics)
        control_daily = paired_daily_unit_mean(
            metrics,
            reference_model="control",
            candidate_model="candidate",
            one_way_cost_bps=float(config["one_way_cost_bps"]),
        )
        lstm_daily = paired_daily_unit_mean(
            metrics,
            reference_model="lstm",
            candidate_model="candidate",
            one_way_cost_bps=float(config["one_way_cost_bps"]),
        )
        control_comparison = summarize_paired_daily(control_daily)
        lstm_comparison = summarize_paired_daily(lstm_daily)
        control_bootstrap = bootstrap_paired_daily(
            control_daily,
            seed=int(config["bootstrap_seed"]),
            draws=int(config["bootstrap_draws"]),
        )
        lstm_bootstrap = bootstrap_paired_daily(
            lstm_daily,
            seed=int(config["bootstrap_seed"]) + 1,
            draws=int(config["bootstrap_draws"]),
        )
        bootstrap = pd.concat([control_bootstrap, lstm_bootstrap], ignore_index=True)
        candidate_receipt = _read_json(
            candidate_artifact / "receipt.json", "v35 candidate receipt"
        )
        speed = cast(Mapping[str, object], candidate_receipt["speed_comparison"])
        decision = decide_sealed_candidate(
            control_comparison,
            control_bootstrap,
            speed=speed,
            gates=cast(Mapping[str, object], config["gates"]),
        )
        elapsed = time.perf_counter() - started

        temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir(parents=True)
        predictions.to_parquet(temporary / "sealed-predictions.parquet", index=False)
        metrics.to_parquet(temporary / "sealed-task-aligned-metrics.parquet", index=False)
        summary.to_parquet(temporary / "sealed-model-summary.parquet", index=False)
        pd.concat([control_daily, lstm_daily], ignore_index=True).to_parquet(
            temporary / "paired-daily-deltas.parquet", index=False
        )
        bootstrap.to_parquet(temporary / "paired-bootstrap.parquet", index=False)
        _write_json(temporary / "candidate-control-comparison.json", control_comparison)
        _write_json(temporary / "candidate-lstm-comparison.json", lstm_comparison)
        selection = {
            **decision,
            "freeze_id": freeze_id,
            "authorization_text_sha256": EXACT_SEALED_AUTHORIZATION_SHA256,
            "sealed_test_accessed": True,
            "evaluation_executed": True,
            "sealed_consumed_exactly_once": True,
            "prediction_count": int(len(predictions)),
            "metric_group_count": int(len(metrics)),
            "paired_market_group_count": int(len(control_daily)),
            "evaluation_seconds": elapsed,
            "model_step_speed_ratio": float(
                cast(Any, speed["model_step_speed_ratio"])
            ),
            "end_to_end_speed_ratio": float(
                cast(Any, speed["end_to_end_speed_ratio"])
            ),
        }
        _write_json(temporary / "selection.json", selection)
        (temporary / "report.md").write_text(
            _report(decision, control_comparison, lstm_comparison, bootstrap),
            encoding="utf-8",
        )
        outputs = {
            str(path.relative_to(temporary)): sha256_file(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-v35-once-only-sealed-evaluation-v36/v1",
            "run_id": "pandadata-tcn-v35-once-only-sealed-evaluation-v36",
            "freeze_id": freeze_id,
            "readiness_receipt_id": readiness_receipt["receipt_id"],
            "candidate_receipt_id": candidate["receipt_id"],
            "lstm_receipt_id": lstm_benchmark["receipt_id"],
            "authorization_text_sha256": EXACT_SEALED_AUTHORIZATION_SHA256,
            "source_sha256": {name: source["sha256"] for name, source in sources.items()},
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_threads": int(v35_config["torch_threads"]),
                "precision": "float32",
            },
            "selection": selection,
            "candidate_control_comparison": control_comparison,
            "candidate_lstm_comparison": lstm_comparison,
            "speed_comparison": dict(speed),
            "outputs": outputs,
            "sealed_test_accessed": True,
            "evaluation_executed": True,
            "sealed_consumed_exactly_once": True,
            "engineering_complete": True,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_dir)
        complete_sealed_consumption(
            marker, result_receipt=str(output_dir / "receipt.json")
        )
        state.update(
            {
                "status": decision["status"],
                "evaluation_executed": True,
                "sealed_test_accessed": True,
                "completed_receipt": str(output_dir / "receipt.json"),
                "completed_receipt_id": receipt["receipt_id"],
                "candidate_model": decision["candidate_model"],
            }
        )
        _atomic_json(state_path, state)
        payload: dict[str, object] = {
            "status": "success",
            "result": decision["status"],
            "candidate_model": decision["candidate_model"],
            "output_dir": str(output_dir),
            "receipt_id": receipt["receipt_id"],
            "sealed_test_accessed": True,
            "sealed_consumed_exactly_once": True,
        }
    except (
        ContractError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        retry_authorized = marker is None
        try:
            observed_state = _read_json(
                arguments.readiness_dir.resolve() / "state.json",
                "v36 readiness state",
            )
            if (
                int(observed_state.get("attempt", 0)) > 0
                or observed_state.get("consumed_marker_created") is True
                or observed_state.get("evaluation_executed") is True
            ):
                retry_authorized = False
        except (ContractError, OSError, ValueError, TypeError):
            retry_authorized = False
        payload = {
            "status": "error",
            "error": str(exc),
            "sealed_claimed": marker is not None,
            "retry_authorized": retry_authorized,
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
