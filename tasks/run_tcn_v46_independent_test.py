"""Run the frozen, once-only V46 utility-aligned independent test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm.experiment import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.neural import HORIZONS, RecurrentRegressor  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
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
    predict_model,
)
from skill_dl_tcn_shortterm.tuning import build_tcn_trial_model  # noqa: E402
from skill_dl_tcn_shortterm.v46_validation import (  # noqa: E402
    decide_v46_independent_gate,
    validate_v46_window_boundaries,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402


EXPECTED_GATES = {
    "min_control_mean_rankic_delta": 0.0,
    "min_control_rankic_ci_low": -0.002,
    "min_positive_seeds": 2,
    "min_positive_horizons": 3,
    "min_control_top_excess_return_delta": -0.0001,
    "min_control_ndcg_at_top_delta": -0.001,
    "min_lstm_rankic_ci_low": -0.01,
    "min_lstm_top_excess_return_ci_low": -0.0005,
    "min_lstm_ndcg_at_top_ci_low": -0.01,
    "min_tcn_lstm_model_step_speed_ratio": 3.0,
    "inference_forward_passes": 1,
    "top_membership_precision_is_gate": False,
}

BOOTSTRAP_METRICS = (
    "rankic",
    "top_excess_return",
    "ndcg_at_top",
    "top_membership_precision",
    "top_positive_return_rate",
    "top_above_cross_section_mean_rate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    if path.exists():
        raise ContractError(f"v46 refuses to overwrite {path.name}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("password", "token", "secret", "credential")):
                return True
            if _contains_secret_key(nested):
                return True
    return isinstance(value, list) and any(_contains_secret_key(item) for item in value)


def _read_receipt(root: Path) -> dict[str, object]:
    path = root / "receipt.json"
    if not path.is_file():
        raise ContractError(f"v46 parent receipt missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"v46 parent receipt is invalid: {path}")
    return cast(dict[str, object], value)


def _normalized_outputs(receipt: dict[str, object]) -> dict[str, str]:
    raw = receipt.get("outputs")
    if not isinstance(raw, dict):
        raise ContractError("v46 parent receipt has no output hashes")
    return {
        str(key).replace("\\", "/").lower(): str(value)
        for key, value in raw.items()
    }


def _verified_checkpoint(
    path: Path,
    *,
    parent_root: Path,
    parent_receipt: dict[str, object],
    model: str,
    seed: int,
) -> dict[str, object]:
    if not path.is_file():
        raise ContractError(f"v46 frozen checkpoint missing: {path}")
    relative = path.relative_to(parent_root).as_posix().lower()
    expected = _normalized_outputs(parent_receipt).get(relative)
    observed = _sha256(path)
    if expected is None or expected != observed:
        raise ContractError(f"v46 frozen checkpoint identity drifted: {model}/{seed}")
    return {
        "model": model,
        "seed": seed,
        "training_fold": 4,
        "path": str(path.resolve()),
        "sha256": observed,
        "parent_receipt_id": str(parent_receipt.get("receipt_id", "")),
    }


def _only_match(root: Path, pattern: str, *, identity: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ContractError(
            f"v46 expected one {identity} checkpoint but observed {len(matches)}"
        )
    return matches[0]


def _checkpoint_manifest(
    *,
    v39_root: Path,
    v40_root: Path,
    v42_root: Path,
    v39_receipt: dict[str, object],
    v40_receipt: dict[str, object],
    v42_receipt: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        candidate = _only_match(
            v42_root / "checkpoints",
            f"*seed-{seed}*fold-4.pt",
            identity=f"candidate seed {seed}",
        )
        rows.append(
            _verified_checkpoint(
                candidate,
                parent_root=v42_root,
                parent_receipt=v42_receipt,
                model="consensus_student_tcn",
                seed=seed,
            )
        )
        if seed == 7:
            control_root = v39_root
            control_receipt = v39_receipt
            control = _only_match(
                control_root / "checkpoints",
                "relative-seed-7-*fold-4.pt",
                identity="control seed 7",
            )
        else:
            control_root = v40_root
            control_receipt = v40_receipt
            control = _only_match(
                control_root / "checkpoints",
                f"relative-seed-{seed}-*fold-4.pt",
                identity=f"control seed {seed}",
            )
        rows.append(
            _verified_checkpoint(
                control,
                parent_root=control_root,
                parent_receipt=control_receipt,
                model="control_tcn",
                seed=seed,
            )
        )
        lstm = _only_match(
            v40_root / "checkpoints",
            f"**/seed-{seed}-lstm-fold-4.pt",
            identity=f"LSTM seed {seed}",
        )
        rows.append(
            _verified_checkpoint(
                lstm,
                parent_root=v40_root,
                parent_receipt=v40_receipt,
                model="relative_lstm",
                seed=seed,
            )
        )
    if len(rows) != 9:
        raise ContractError("v46 checkpoint manifest must contain exactly nine entries")
    return rows


def _prediction_frame(
    scores: np.ndarray,
    positions: np.ndarray,
    labels: pd.DataFrame,
    *,
    model: str,
    seed: int,
    contracts: dict[str, str],
) -> pd.DataFrame:
    if scores.shape != (len(positions), len(HORIZONS)):
        raise ContractError("v46 checkpoint output shape drifted")
    score_frame = pd.DataFrame(scores, columns=HORIZONS)
    score_frame.insert(0, "sample_position", positions.astype("int64"))
    long_scores = score_frame.melt(
        id_vars="sample_position", var_name="horizon", value_name="score"
    )
    long_scores["horizon"] = long_scores["horizon"].astype(int)
    label_columns = [
        "sample_position",
        "sample_id",
        "instrument_id",
        "signal_date",
        "horizon",
        "rank_target",
        "raw_return",
    ]
    merged = long_scores.merge(
        labels[label_columns],
        on=["sample_position", "horizon"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(labels):
        raise ContractError(f"v46 {model}/{seed} prediction-label coverage drifted")
    merged.insert(0, "model", model)
    merged.insert(1, "seed", seed)
    merged.insert(2, "fold", 4)
    merged["stage"] = "test"
    merged["sealed"] = True
    merged["prediction_contract_id"] = contracts["prediction_contract_id"]
    merged["target_contract_id"] = contracts["target_contract_id"]
    merged["evaluation_contract_id"] = contracts["evaluation_contract_id"]
    merged["training_contract_id"] = {
        "control_tcn": "frozen-true-target-relative10-tcn-v40",
        "consensus_student_tcn": "frozen-consensus-student025-relative10-tcn-v42",
        "relative_lstm": "frozen-relative10-lstm-v40",
    }[model]
    return merged


def _rankic_breadth(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = metrics.loc[
        metrics["model"].isin(["control_tcn", "consensus_student_tcn"])
    ]
    grouped = (
        selected.groupby(["model", "seed", "horizon"], observed=True)["rankic"]
        .mean()
        .unstack("model")
    )
    if set(grouped.columns.astype(str)) != {"control_tcn", "consensus_student_tcn"}:
        raise ContractError("v46 rankic breadth model coverage drifted")
    paired = grouped.reset_index()[["seed", "horizon"]].copy()
    paired["rankic_delta"] = (
        grouped["consensus_student_tcn"] - grouped["control_tcn"]
    ).to_numpy()
    seed = paired.groupby("seed", as_index=False, observed=True).agg(
        rankic_delta=("rankic_delta", "mean")
    )
    horizon = paired.groupby("horizon", as_index=False, observed=True).agg(
        rankic_delta=("rankic_delta", "mean")
    )
    return seed, horizon


def _report(
    decision: Mapping[str, object],
    control: Mapping[str, object],
    lstm: Mapping[str, object],
    window_audit: Mapping[str, object],
) -> str:
    def metric(mapping: Mapping[str, object], name: str) -> float:
        return float(cast(Any, mapping[name]))

    return "\n".join(
        [
            "# TCN V46 一次性独立外推结果",
            "",
            f"- 正式状态：`{decision['status']}`",
            f"- Research candidate：`{str(decision['admitted']).lower()}`",
            "- Alpha-ready：`false`",
            "- Deployment authorized：`false`",
            "- Trading authorized：`false`",
            "",
            "## 独立性",
            "",
            f"- 窗口：`{window_audit['evaluation_start']}..{window_audit['evaluation_end']}`",
            f"- 训练统计最晚日期：`{window_audit['training_max_date']}`",
            f"- 旧测试最晚日期：`{window_audit['prior_consumed_end']}`",
            f"- Embargo 结束：`{window_audit['embargo_end']}`",
            "- 本窗口已一次性消费：`true`",
            "",
            "## V42 student 相对 true-target TCN",
            "",
            f"- RankIC delta：`{metric(control, 'mean_rankic_delta'):+.6f}`",
            f"- Top excess-return delta：`{metric(control, 'mean_top_excess_return_delta'):+.8f}`",
            f"- NDCG@Top delta：`{metric(control, 'mean_ndcg_at_top_delta'):+.6f}`",
            f"- Top membership delta（诊断）：`{metric(control, 'mean_top_membership_precision_delta'):+.6f}`",
            f"- Top positive-return-rate delta（诊断）：`{metric(control, 'mean_top_positive_return_rate_delta'):+.6f}`",
            f"- Top above-cross-section-mean-rate delta（诊断）：`{metric(control, 'mean_top_above_cross_section_mean_rate_delta'):+.6f}`",
            "",
            "## V42 student 相对 LSTM",
            "",
            f"- RankIC delta：`{metric(lstm, 'mean_rankic_delta'):+.6f}`",
            f"- Top excess-return delta：`{metric(lstm, 'mean_top_excess_return_delta'):+.8f}`",
            f"- NDCG@Top delta：`{metric(lstm, 'mean_ndcg_at_top_delta'):+.6f}`",
            "",
            "## 解释边界",
            "",
            "`top_membership_precision` 是预测 Top10% 与实际收益 Top10% 的集合重合率；",
            "`top_positive_return_rate` 是预测 Top10% 中收益大于 0 的比例；",
            "`top_above_cross_section_mean_rate` 是预测 Top10% 中收益高于当期横截面均值的比例。",
            "三者不是同一指标，membership 不参与 V46 晋级门。V42 历史状态不追溯改写。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--relative-feature-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--v39-control-dir", required=True, type=Path)
    parser.add_argument("--v40-run-dir", required=True, type=Path)
    parser.add_argument("--v42-run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    try:
        if output.exists():
            raise ContractError("v46 once-only output identity already exists")
        config_path = arguments.config.resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("protocol_version") != "v46-independent-test":
            raise ContractError("v46 config identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v46 config contains a secret-like key")
        if config.get("gates") != EXPECTED_GATES:
            raise ContractError("v46 frozen gates drifted")
        if (
            config.get("models") != ["control_tcn", "consensus_student_tcn", "relative_lstm"]
            or config.get("seeds") != [7, 17, 27]
            or int(cast(Any, config.get("training_fold"))) != 4
            or config.get("horizons") != HORIZONS
            or float(cast(Any, config.get("top_fraction"))) != 0.1
            or config.get("precision") != "float32"
            or int(cast(Any, config.get("num_workers"))) != 0
        ):
            raise ContractError("v46 frozen execution contract drifted")

        base = arguments.base_run_dir.resolve()
        relative = arguments.relative_feature_dir.resolve()
        split_path = arguments.split_manifest.resolve()
        v39 = arguments.v39_control_dir.resolve()
        v40 = arguments.v40_run_dir.resolve()
        v42 = arguments.v42_run_dir.resolve()
        source_paths = {
            "relative_features": relative / "feature-windows.npy",
            "relative_window_index": relative / "window-index.parquet",
            "labels": base / "labels.parquet",
            "split_manifest": split_path,
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v46 sources missing: " + ", ".join(missing))
        source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if source_hashes != {
            str(key): str(value)
            for key, value in cast(dict[str, object], config["source_sha256"]).items()
        }:
            raise ContractError("v46 source SHA-256 identity drifted")

        v39_receipt = _read_receipt(v39)
        v40_receipt = _read_receipt(v40)
        v42_receipt = _read_receipt(v42)
        parent = cast(dict[str, object], config["parent_contract"])
        v42_gate = v42_receipt.get("model_gate")
        if not isinstance(v42_gate, dict):
            raise ContractError("v46 V42 parent model gate is missing")
        v42_evidence = v42_gate.get("evidence")
        if not isinstance(v42_evidence, dict):
            raise ContractError("v46 V42 parent speed evidence is missing")
        speed_ratio = float(cast(Any, v42_evidence["implied_tcn_lstm_model_step_ratio"]))
        if not (
            v42_receipt.get("receipt_id") == parent["v42_receipt_id"]
            and v42_gate.get("status") == parent["v42_historical_status"]
            and v42_gate.get("admitted") is parent["v42_historical_admitted"]
            and speed_ratio == float(cast(Any, parent["model_step_speed_ratio"]))
            and v42_receipt.get("sealed_test_accessed") is False
        ):
            raise ContractError("v46 V42 historical parent contract drifted")

        checkpoints = _checkpoint_manifest(
            v39_root=v39,
            v40_root=v40,
            v42_root=v42,
            v39_receipt=v39_receipt,
            v40_receipt=v40_receipt,
            v42_receipt=v42_receipt,
        )

        output.mkdir(parents=True)
        started_at = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
        _write_json(
            output / "consumption-start.json",
            {
                "run_id": config["run_id"],
                "started_at": started_at,
                "sealed_test_accessed": True,
                "sealed_consumed_exactly_once": True,
                "window": config["evaluation_window"],
                "status": "v46_once_only_consumption_started",
            },
        )
        _write_json(output / "config.resolved.json", config)
        _write_json(output / "checkpoint-manifest.json", checkpoints)

        torch.set_num_threads(int(cast(Any, config["torch_threads"])))
        features = np.load(
            source_paths["relative_features"], mmap_mode="r", allow_pickle=False
        )
        if features.ndim != 3 or features.shape[1:] != (10, 480):
            raise ContractError("v46 relative10 tensor shape drifted")
        window_index = pd.read_parquet(source_paths["relative_window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in split or split["sealed"].astype(bool).any():
            raise ContractError("v46 training scaler source must be ordinary validation")
        protocols = build_fold_protocols(features, split)
        protocol = next((item for item in protocols if item.fold == 4), None)
        if protocol is None:
            raise ContractError("v46 fold-4 training protocol is missing")

        window = cast(dict[str, str], config["evaluation_window"])
        dates = window_index["signal_date"].astype(str)
        evaluation_index = window_index.loc[
            dates.between(window["start"], window["end"], inclusive="both")
        ].copy()
        training_positions = set(int(value) for value in protocol.train_positions)
        training_dates = window_index.loc[
            window_index["sample_position"].astype(int).isin(training_positions),
            "signal_date",
        ]
        window_audit = validate_v46_window_boundaries(
            evaluation_dates=evaluation_index["signal_date"],
            training_dates=training_dates,
            prior_consumed_end=window["prior_consumed_end"],
            embargo_end=window["embargo_end"],
            expected_start=window["start"],
            expected_end=window["end"],
        )
        evaluation_ids = set(evaluation_index["sample_id"].astype(str))
        evaluation_labels = labels.loc[
            labels["sample_id"].astype(str).isin(evaluation_ids)
            & labels["horizon"].astype(int).isin(HORIZONS)
            & labels["valid"].astype(bool)
        ].copy()
        if (
            evaluation_labels.empty
            or evaluation_labels.duplicated(["sample_position", "horizon"]).any()
            or set(evaluation_labels["horizon"].astype(int)) != set(HORIZONS)
            or evaluation_labels["signal_date"].astype(str).min() != window["start"]
            or evaluation_labels["signal_date"].astype(str).max() != window["end"]
        ):
            raise ContractError("v46 independent label coverage drifted")
        coverage = evaluation_labels.groupby(
            ["signal_date", "horizon"], observed=True
        ).size()
        if int(coverage.min()) < 10:
            raise ContractError("v46 independent cross-section is too narrow")
        evaluation_positions = np.sort(
            evaluation_labels["sample_position"].astype("int64").unique()
        )
        if training_positions.intersection(set(map(int, evaluation_positions))):
            raise ContractError("v46 evaluation positions leaked into training scaler")
        window_audit.update(
            {
                "evaluation_sample_count": len(evaluation_positions),
                "valid_label_count": len(evaluation_labels),
                "minimum_group_member_count": int(coverage.min()),
                "maximum_group_member_count": int(coverage.max()),
                "horizons": HORIZONS,
                "sealed_test_accessed": True,
                "sealed_consumed_exactly_once": True,
            }
        )
        _write_json(output / "window-audit.json", window_audit)

        targets = np.zeros((len(features), len(HORIZONS)), dtype="float32")
        masks = np.ones_like(targets, dtype="bool")
        dataset = LazyWindowDataset(
            features,
            evaluation_positions,
            targets,
            masks,
            protocol.feature_mean,
            protocol.feature_std,
        )
        contracts = {
            str(key): str(value)
            for key, value in cast(dict[str, object], config["contracts"]).items()
        }
        trial = parse_real_tcn_trials([cast(dict[str, object], config["tcn_trial"])])[0]
        manifest_by_key = {
            (str(row["model"]), int(cast(Any, row["seed"]))): row
            for row in checkpoints
        }
        prediction_frames: list[pd.DataFrame] = []
        timing_rows: list[dict[str, object]] = []
        for model_name in ("control_tcn", "consensus_student_tcn", "relative_lstm"):
            for seed in (7, 17, 27):
                checkpoint = Path(str(manifest_by_key[(model_name, seed)]["path"]))
                if model_name == "relative_lstm":
                    model: torch.nn.Module = RecurrentRegressor(
                        "lstm",
                        int(features.shape[1]),
                        int(cast(Any, cast(dict[str, object], config["lstm"])["hidden_size"])),
                    )
                else:
                    model = build_tcn_trial_model(
                        trial,
                        feature_count=int(features.shape[1]),
                        input_steps=int(features.shape[2]),
                    )
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if not isinstance(state, dict):
                    raise ContractError(f"v46 checkpoint is not a state dict: {model_name}/{seed}")
                model.load_state_dict(state, strict=True)
                started = time.perf_counter()
                scores, positions = predict_model(
                    model,
                    dataset,
                    batch_size=int(cast(Any, config["batch_size"])),
                    num_workers=0,
                )
                elapsed = time.perf_counter() - started
                prediction_frames.append(
                    _prediction_frame(
                        scores,
                        positions,
                        evaluation_labels,
                        model=model_name,
                        seed=seed,
                        contracts=contracts,
                    )
                )
                timing_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "training_fold": 4,
                        "sample_count": len(positions),
                        "inference_seconds": elapsed,
                        "samples_per_second": len(positions) / elapsed,
                        "forward_passes": 1,
                    }
                )

        predictions = pd.concat(prediction_frames, ignore_index=True)
        validate_prediction_contract(
            predictions, expected_models=3, expected_stage="test", allow_sealed=True
        )
        if (
            set(predictions["seed"].astype(int)) != {7, 17, 27}
            or set(predictions["fold"].astype(int)) != {4}
            or set(predictions["horizon"].astype(int)) != set(HORIZONS)
        ):
            raise ContractError("v46 prediction coverage drifted")
        metrics = evaluate_task_aligned_predictions(
            predictions,
            top_fraction=0.1,
            expected_stage="test",
            allow_sealed=True,
        )
        summary = summarize_task_aligned_metrics(metrics)
        control_comparison = compare_task_aligned_models(
            metrics,
            reference_model="control_tcn",
            candidate_model="consensus_student_tcn",
        )
        lstm_comparison = compare_task_aligned_models(
            metrics,
            reference_model="relative_lstm",
            candidate_model="consensus_student_tcn",
        )
        draws = int(cast(Any, config["bootstrap_draws"]))
        bootstrap_seed = int(cast(Any, config["bootstrap_seed"]))
        control_bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="control_tcn",
            candidate_model="consensus_student_tcn",
            metric_columns=BOOTSTRAP_METRICS,
            seed=bootstrap_seed,
            draws=draws,
        )
        control_bootstrap.insert(0, "scope", "student_minus_control_tcn")
        lstm_bootstrap = bootstrap_task_aligned_differences(
            metrics,
            reference_model="relative_lstm",
            candidate_model="consensus_student_tcn",
            metric_columns=BOOTSTRAP_METRICS,
            seed=bootstrap_seed + 1,
            draws=draws,
        )
        lstm_bootstrap.insert(0, "scope", "student_minus_relative_lstm")
        seed_deltas, horizon_deltas = _rankic_breadth(metrics)
        decision = decide_v46_independent_gate(
            control_comparison,
            control_bootstrap,
            lstm_bootstrap,
            seed_deltas,
            horizon_deltas,
            contract_valid=True,
            historical_replay=False,
            model_step_speed_ratio=speed_ratio,
            inference_forward_passes=1,
        )
        decision_payload: dict[str, object] = {
            **asdict(decision),
            "alpha_ready": False,
            "deployment_authorized": False,
            "trading_authorized": False,
            "sealed_test_accessed": True,
            "sealed_consumed_exactly_once": True,
            "historical_v42_status_unchanged": str(v42_gate["status"]),
        }

        predictions.to_parquet(output / "predictions.parquet", index=False)
        metrics.to_parquet(output / "utility-metrics.parquet", index=False)
        summary.to_parquet(output / "model-summary.parquet", index=False)
        pd.concat([control_bootstrap, lstm_bootstrap], ignore_index=True).to_parquet(
            output / "paired-bootstrap.parquet", index=False
        )
        seed_deltas.to_parquet(output / "seed-deltas.parquet", index=False)
        horizon_deltas.to_parquet(output / "horizon-deltas.parquet", index=False)
        pd.DataFrame(timing_rows).to_parquet(output / "inference-timing.parquet", index=False)
        _write_json(output / "student-control-comparison.json", control_comparison)
        _write_json(output / "student-lstm-comparison.json", lstm_comparison)
        _write_json(output / "decision.json", decision_payload)
        (output / "report.md").write_text(
            _report(decision_payload, control_comparison, lstm_comparison, window_audit),
            encoding="utf-8",
        )
        _write_json(
            output / "consumption-complete.json",
            {
                "run_id": config["run_id"],
                "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                "sealed_test_accessed": True,
                "sealed_consumed_exactly_once": True,
                "status": decision.status,
            },
        )

        output_hashes = {
            path.relative_to(output).as_posix(): _sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "receipt.json"
        }
        receipt: dict[str, object] = {
            "schema_version": "tcn-v46-utility-aligned-independent-test/v1",
            "run_id": config["run_id"],
            "code_identity": code_identity(ROOT),
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "precision": config["precision"],
            },
            "source_sha256": source_hashes,
            "parent_receipts": {
                "v39": v39_receipt.get("receipt_id"),
                "v40": v40_receipt.get("receipt_id"),
                "v42": v42_receipt.get("receipt_id"),
            },
            "checkpoint_manifest": checkpoints,
            "window_audit": window_audit,
            "decision": decision_payload,
            "outputs": output_hashes,
            "sealed_test_accessed": True,
            "sealed_consumed_exactly_once": True,
            "alpha_ready": False,
            "deployment_authorized": False,
            "trading_authorized": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(output / "receipt.json", receipt)
        print(json.dumps(decision_payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        if output.exists():
            failure_path = output / "failure.json"
            if not failure_path.exists():
                failure_path.write_text(
                    json.dumps(
                        {
                            "status": "v46_once_only_run_failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "sealed_test_accessed": True,
                            "sealed_consumed_exactly_once": True,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        print(f"v46 independent test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
