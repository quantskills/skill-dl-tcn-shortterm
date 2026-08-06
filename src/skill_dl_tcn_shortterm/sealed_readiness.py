"""Fail-closed freezing for the v35 task-aligned sealed evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast
import uuid

import pandas as pd

from .experiment import ContractError
from .integrity import code_identity


V35_CANDIDATE_STATUS = "constrained_tail_ordinary_validation_candidate_v35"
EXACT_SEALED_AUTHORIZATION = "授权执行 sealed test"
EXACT_SEALED_AUTHORIZATION_SHA256 = hashlib.sha256(
    EXACT_SEALED_AUTHORIZATION.encode("utf-8")
).hexdigest()
TASK_ALIGNED_GATE_FIELDS = frozenset(
    {
        "min_mean_top_precision_delta",
        "min_mean_ndcg_at_top_delta",
        "min_primary_tail_ci_low",
        "min_secondary_tail_ci_low",
        "min_mean_rankic_delta",
        "min_top_return_ci_low",
        "min_net_return_after_cost_ci_low",
        "max_mean_top_turnover_delta",
        "min_model_step_speed_ratio",
        "min_end_to_end_speed_ratio",
    }
)


@dataclass(frozen=True)
class SealedReadinessResult:
    """Paths and immutable identity produced without consuming sealed labels."""

    freeze_id: str
    output_dir: Path
    frozen_plan_path: Path
    readiness_path: Path
    state_path: Path
    receipt_path: Path


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    """Return stable UTF-8 bytes for receipt and freeze fingerprints."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Stream a file fingerprint without loading large artifacts into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_exact_sealed_authorization(value: str | None) -> None:
    """Reject missing, approximate, or whitespace-modified authorization."""

    if value != EXACT_SEALED_AUTHORIZATION:
        raise ContractError(
            "sealed evaluation requires the exact authorization text: "
            f"{EXACT_SEALED_AUTHORIZATION}"
        )


def validate_task_aligned_freeze_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable v36 protocol before any sealed loader may run."""

    required = {
        "protocol_version",
        "run_id",
        "v35_candidate_artifact",
        "v35_receipt_id",
        "v35_candidate_status",
        "lstm_artifact",
        "lstm_receipt_id",
        "ordinary_split_manifest",
        "sealed_split_manifest",
        "expected_sha256",
        "seeds",
        "ordinary_folds",
        "sealed_test_folds",
        "expected_eligible_unit_count",
        "expected_changed_unit_exposures",
        "top_fraction",
        "bootstrap_seed",
        "bootstrap_draws",
        "one_way_cost_bps",
        "authorization_text_sha256",
        "evaluation_policy",
        "gates",
    }
    if missing := required - set(config):
        raise ContractError(f"v36 freeze config missing fields: {sorted(missing)}")
    payload = dict(config)
    if payload["protocol_version"] != "v36":
        raise ContractError("v36 freeze protocol_version must equal v36")
    if payload["v35_candidate_status"] != V35_CANDIDATE_STATUS:
        raise ContractError("v36 freeze candidate status is not admissible")
    if payload["authorization_text_sha256"] != EXACT_SEALED_AUTHORIZATION_SHA256:
        raise ContractError("v36 exact sealed authorization identity drifted")
    if list(payload["seeds"]) != [7, 17, 27]:
        raise ContractError("v36 seeds must equal [7, 17, 27]")
    if list(payload["ordinary_folds"]) != [0, 1, 2, 3, 4]:
        raise ContractError("v36 ordinary folds must equal [0, 1, 2, 3, 4]")
    if list(payload["sealed_test_folds"]) != [0, 1]:
        raise ContractError("v36 sealed test folds must equal [0, 1]")
    if int(payload["expected_eligible_unit_count"]) != 24:
        raise ContractError("v36 eligible unit count must equal 24")
    if int(payload["expected_changed_unit_exposures"]) != 12:
        raise ContractError("v36 changed unit exposures must equal 12")
    for field in ("top_fraction", "one_way_cost_bps"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"v36 {field} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ContractError(f"v36 {field} must be finite and positive")
    if int(payload["bootstrap_draws"]) < 1000:
        raise ContractError("v36 bootstrap_draws must be at least 1000")
    expected_sha256 = payload["expected_sha256"]
    required_hashes = {
        "ordinary_split_manifest",
        "sealed_split_manifest",
        "features",
        "window_index",
        "labels",
    }
    if not isinstance(expected_sha256, Mapping) or set(expected_sha256) != required_hashes:
        raise ContractError(
            f"v36 expected_sha256 must equal {sorted(required_hashes)}"
        )
    for name, value in expected_sha256.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ContractError(f"v36 expected SHA-256 is invalid: {name}")
    policy = payload["evaluation_policy"]
    expected_policy = {
        "sealed_stage": "test",
        "exclude_stage": "sealed_holdout",
        "eligibility_guard": "validation_end_date < sealed_test_start_date",
        "model_unit_policy": "all_eligible_seed_fold_units",
        "metric_aggregation": "date_block_paired_unit_mean",
        "sealed_reuse": "exactly_once",
        "post_result_tuning": "forbidden",
    }
    if policy != expected_policy:
        raise ContractError("v36 evaluation policy drifted")
    gates = payload["gates"]
    if not isinstance(gates, Mapping) or set(gates) != TASK_ALIGNED_GATE_FIELDS:
        raise ContractError(
            f"v36 task-aligned gates must equal {sorted(TASK_ALIGNED_GATE_FIELDS)}"
        )
    for name, value in gates.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ContractError(f"v36 gate must be finite: {name}")
    return payload


def verify_receipt_identity(receipt: Mapping[str, Any], expected_id: str) -> None:
    """Verify a receipt whose id hashes every other receipt field."""

    if receipt.get("receipt_id") != expected_id:
        raise ContractError("receipt identity drifted")
    payload = dict(receipt)
    payload.pop("receipt_id", None)
    observed = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if observed != expected_id:
        raise ContractError("receipt content fingerprint drifted")


def verify_receipt_outputs(artifact: Path, receipt: Mapping[str, Any]) -> None:
    """Verify every output declared by a parent receipt."""

    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ContractError("parent receipt outputs are missing")
    for raw_name, expected in outputs.items():
        relative = Path(str(raw_name).replace("\\", "/"))
        path = (artifact / relative).resolve()
        try:
            path.relative_to(artifact.resolve())
        except ValueError as exc:
            raise ContractError(f"parent output escapes artifact: {raw_name}") from exc
        if not path.is_file():
            raise ContractError(f"parent output is missing: {raw_name}")
        if sha256_file(path) != expected:
            raise ContractError(f"parent output fingerprint drifted: {raw_name}")


def build_eligible_checkpoint_plan(
    ordinary_manifest: pd.DataFrame,
    sealed_manifest: pd.DataFrame,
    checkpoint_selection: pd.DataFrame,
    lstm_checkpoint_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a no-time-travel plan using only manifest metadata and hashes."""

    ordinary_required = {"fold", "sample_id", "signal_date", "stage", "sealed"}
    sealed_required = {"fold", "sample_id", "signal_date", "stage", "sealed"}
    selection_required = {
        "seed",
        "fold",
        "selection_changed",
        "control_epoch",
        "candidate_epoch",
        "control_checkpoint",
        "control_checkpoint_sha256",
        "candidate_checkpoint",
        "candidate_checkpoint_sha256",
    }
    lstm_required = {"seed", "fold", "checkpoint", "checkpoint_sha256"}
    for label, frame, required in (
        ("ordinary manifest", ordinary_manifest, ordinary_required),
        ("sealed manifest", sealed_manifest, sealed_required),
        ("checkpoint selection", checkpoint_selection, selection_required),
        ("LSTM checkpoint summary", lstm_checkpoint_summary, lstm_required),
    ):
        if missing := required - set(frame):
            raise ContractError(f"v36 {label} missing columns: {sorted(missing)}")

    if ordinary_manifest["sealed"].astype(bool).any():
        raise ContractError("v36 ordinary manifest contains sealed rows")
    ordinary_stage = set(ordinary_manifest["stage"].astype(str))
    if not ordinary_stage <= {"train", "validation", "purged"}:
        raise ContractError("v36 ordinary manifest contains forbidden stages")
    if set(checkpoint_selection["seed"].astype(int)) != {7, 17, 27} or set(
        checkpoint_selection["fold"].astype(int)
    ) != {0, 1, 2, 3, 4}:
        raise ContractError("v36 TCN checkpoint coverage drifted")
    if len(checkpoint_selection) != 15 or checkpoint_selection.duplicated(
        ["seed", "fold"]
    ).any():
        raise ContractError("v36 TCN checkpoint selection must contain 15 units")
    if set(lstm_checkpoint_summary["seed"].astype(int)) != {7, 17, 27} or set(
        lstm_checkpoint_summary["fold"].astype(int)
    ) != {0, 1, 2, 3, 4}:
        raise ContractError("v36 LSTM checkpoint coverage drifted")
    if len(lstm_checkpoint_summary) != 15 or lstm_checkpoint_summary.duplicated(
        ["seed", "fold"]
    ).any():
        raise ContractError("v36 LSTM checkpoint summary must contain 15 units")

    tests = sealed_manifest.loc[sealed_manifest["stage"].astype(str).eq("test")].copy()
    if tests.empty or not tests["sealed"].astype(bool).all():
        raise ContractError("v36 sealed manifest has no fully sealed test rows")
    if set(tests["fold"].astype(int)) != {0, 1}:
        raise ContractError("v36 sealed test folds must equal 0 and 1")
    if tests.duplicated(["sample_id"]).any():
        raise ContractError("v36 canonical test rows contain duplicate sample ids")
    ordinary_ids = set(ordinary_manifest["sample_id"].astype(str))
    sealed_ids = set(tests["sample_id"].astype(str))
    if ordinary_ids & sealed_ids:
        raise ContractError("v36 ordinary and sealed sample identities overlap")

    ordinary = ordinary_manifest.copy()
    ordinary["signal_date"] = pd.to_datetime(ordinary["signal_date"], errors="raise")
    tests["signal_date"] = pd.to_datetime(tests["signal_date"], errors="raise")
    validation = ordinary.loc[ordinary["stage"].astype(str).eq("validation")]
    validation_ranges = validation.groupby("fold", observed=True)["signal_date"].agg(
        validation_start_date="min", validation_end_date="max"
    )
    if set(validation_ranges.index.astype(int)) != {0, 1, 2, 3, 4}:
        raise ContractError("v36 ordinary validation date coverage drifted")
    test_ranges = tests.groupby("fold", observed=True)["signal_date"].agg(
        sealed_test_start_date="min", sealed_test_end_date="max"
    )

    selection = checkpoint_selection.set_index(["seed", "fold"]).sort_index()
    lstm = lstm_checkpoint_summary.set_index(["seed", "fold"]).sort_index()
    rows: list[dict[str, object]] = []
    for sealed_fold, test_range in test_ranges.iterrows():
        sealed_start = cast(pd.Timestamp, test_range["sealed_test_start_date"])
        eligible_folds = [
            int(fold)
            for fold, row in validation_ranges.iterrows()
            if cast(pd.Timestamp, row["validation_end_date"]) < sealed_start
        ]
        if not eligible_folds:
            raise ContractError(f"v36 sealed fold {sealed_fold} has no eligible model")
        for seed in (7, 17, 27):
            for training_fold in eligible_folds:
                selected = cast(
                    dict[str, Any],
                    cast(pd.Series, selection.loc[(seed, training_fold)]).to_dict(),
                )
                recurrent = cast(
                    dict[str, Any],
                    cast(pd.Series, lstm.loc[(seed, training_fold)]).to_dict(),
                )
                rows.append(
                    {
                        "sealed_fold": int(sealed_fold),
                        "sealed_test_start_date": sealed_start.strftime("%Y-%m-%d"),
                        "sealed_test_end_date": cast(
                            pd.Timestamp, test_range["sealed_test_end_date"]
                        ).strftime("%Y-%m-%d"),
                        "seed": seed,
                        "training_fold": training_fold,
                        "validation_start_date": cast(
                            pd.Timestamp,
                            validation_ranges.loc[
                                training_fold, "validation_start_date"
                            ],
                        ).strftime("%Y-%m-%d"),
                        "validation_end_date": cast(
                            pd.Timestamp,
                            validation_ranges.loc[training_fold, "validation_end_date"],
                        ).strftime("%Y-%m-%d"),
                        "selection_changed": bool(selected["selection_changed"]),
                        "control_epoch": int(selected["control_epoch"]),
                        "candidate_epoch": int(selected["candidate_epoch"]),
                        "control_checkpoint": str(selected["control_checkpoint"]),
                        "control_checkpoint_sha256": str(
                            selected["control_checkpoint_sha256"]
                        ),
                        "candidate_checkpoint": str(selected["candidate_checkpoint"]),
                        "candidate_checkpoint_sha256": str(
                            selected["candidate_checkpoint_sha256"]
                        ),
                        "lstm_checkpoint": str(recurrent["checkpoint"]),
                        "lstm_checkpoint_sha256": str(recurrent["checkpoint_sha256"]),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["sealed_fold", "seed", "training_fold"], ignore_index=True
    )
    if len(result) != 24:
        raise ContractError("v36 time-safe eligible unit count must equal 24")
    if int(result["selection_changed"].sum()) != 12:
        raise ContractError("v36 changed checkpoint exposure count must equal 12")
    for row in result.itertuples(index=False):
        typed_row = cast(Any, row)
        if pd.Timestamp(str(typed_row.validation_end_date)) >= pd.Timestamp(
            str(typed_row.sealed_test_start_date)
        ):
            raise ContractError("v36 checkpoint plan contains time-travel leakage")
    return result


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


def _resolve_project_path(project_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _verify_checkpoint_paths(
    plan: pd.DataFrame, *, candidate_artifact: Path, lstm_artifact: Path
) -> None:
    checked: set[tuple[str, str]] = set()
    for row in plan.itertuples(index=False):
        for prefix, artifact in (
            ("control", candidate_artifact),
            ("candidate", candidate_artifact),
            ("lstm", lstm_artifact),
        ):
            raw_path = str(getattr(row, f"{prefix}_checkpoint")).replace("\\", "/")
            path = (artifact / Path(raw_path)).resolve()
            try:
                path.relative_to(artifact.resolve())
            except ValueError as exc:
                raise ContractError(f"v36 {prefix} checkpoint escapes artifact") from exc
            expected = str(getattr(row, f"{prefix}_checkpoint_sha256"))
            key = (str(path), expected)
            if key in checked:
                continue
            if not path.is_file():
                raise ContractError(f"v36 {prefix} checkpoint is missing: {raw_path}")
            if sha256_file(path) != expected:
                raise ContractError(f"v36 {prefix} checkpoint fingerprint drifted")
            checked.add(key)


def freeze_v35_sealed_readiness(
    project_root: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> SealedReadinessResult:
    """Freeze v35 and metadata-only sealed eligibility without opening labels."""

    root = Path(project_root).resolve()
    payload = validate_task_aligned_freeze_config(config)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ContractError("v36 readiness refuses to overwrite its output directory")

    candidate_artifact = _resolve_project_path(root, payload["v35_candidate_artifact"])
    lstm_artifact = _resolve_project_path(root, payload["lstm_artifact"])
    ordinary_path = _resolve_project_path(root, payload["ordinary_split_manifest"])
    sealed_path = _resolve_project_path(root, payload["sealed_split_manifest"])
    for label, path in (
        ("v35 candidate artifact", candidate_artifact),
        ("v33 LSTM artifact", lstm_artifact),
    ):
        if not path.is_dir():
            raise ContractError(f"v36 {label} is missing")
    for label, path in (
        ("ordinary split manifest", ordinary_path),
        ("sealed split manifest", sealed_path),
    ):
        if not path.is_file():
            raise ContractError(f"v36 {label} is missing")

    candidate_receipt = _read_json(
        candidate_artifact / "receipt.json", "v35 candidate receipt"
    )
    verify_receipt_identity(candidate_receipt, str(payload["v35_receipt_id"]))
    candidate_selection = candidate_receipt.get("selection")
    if not isinstance(candidate_selection, Mapping):
        raise ContractError("v36 v35 candidate selection is missing")
    if candidate_selection.get("status") != payload["v35_candidate_status"]:
        raise ContractError("v36 v35 candidate status drifted")
    for gate in ("integrity_passed", "mechanism_passed", "effect_passed", "speed_passed"):
        if candidate_selection.get(gate) is not True:
            raise ContractError(f"v36 v35 candidate gate failed: {gate}")
    if (
        candidate_receipt.get("sealed_test_accessed") is not False
        or candidate_selection.get("sealed_test_accessed") is not False
        or candidate_selection.get("sealed_test_authorized") is not False
    ):
        raise ContractError("v36 v35 candidate is not sealed fail-closed")
    verify_receipt_outputs(candidate_artifact, candidate_receipt)

    lstm_receipt = _read_json(lstm_artifact / "receipt.json", "v33 LSTM receipt")
    verify_receipt_identity(lstm_receipt, str(payload["lstm_receipt_id"]))
    if lstm_receipt.get("sealed_test_accessed") is not False:
        raise ContractError("v36 v33 LSTM benchmark accessed sealed data")
    verify_receipt_outputs(lstm_artifact, lstm_receipt)

    expected_hashes = cast(Mapping[str, str], payload["expected_sha256"])
    if sha256_file(ordinary_path) != expected_hashes["ordinary_split_manifest"]:
        raise ContractError("v36 ordinary split manifest fingerprint drifted")
    if sha256_file(sealed_path) != expected_hashes["sealed_split_manifest"]:
        raise ContractError("v36 sealed split manifest fingerprint drifted")
    source_artifacts = candidate_receipt.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise ContractError("v36 v35 source artifacts are missing")
    source_paths: dict[str, Path] = {}
    for name in ("features", "window_index", "labels"):
        source = source_artifacts.get(name)
        if not isinstance(source, Mapping):
            raise ContractError(f"v36 v35 source artifact is missing: {name}")
        if source.get("sha256") != expected_hashes[name]:
            raise ContractError(f"v36 v35 source receipt fingerprint drifted: {name}")
        source_path = Path(str(source.get("path"))).resolve()
        if not source_path.is_file() or sha256_file(source_path) != expected_hashes[name]:
            raise ContractError(f"v36 v35 source file fingerprint drifted: {name}")
        source_paths[name] = source_path

    ordinary_manifest = pd.read_parquet(
        ordinary_path, columns=["fold", "sample_id", "signal_date", "stage", "sealed"]
    )
    sealed_manifest = pd.read_parquet(
        sealed_path, columns=["fold", "sample_id", "signal_date", "stage", "sealed"]
    )
    checkpoint_selection = pd.read_parquet(candidate_artifact / "checkpoint-selection.parquet")
    lstm_summary = pd.read_parquet(lstm_artifact / "lstm-checkpoint-summary.parquet")
    checkpoint_plan = build_eligible_checkpoint_plan(
        ordinary_manifest, sealed_manifest, checkpoint_selection, lstm_summary
    )
    _verify_checkpoint_paths(
        checkpoint_plan,
        candidate_artifact=candidate_artifact,
        lstm_artifact=lstm_artifact,
    )

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        plan_path = temporary / "eligible-checkpoint-plan.parquet"
        checkpoint_plan.to_parquet(plan_path, index=False)
        test_rows = sealed_manifest.loc[sealed_manifest["stage"].astype(str).eq("test")].copy()
        test_rows["signal_date"] = pd.to_datetime(test_rows["signal_date"], errors="raise")
        test_ranges = []
        for fold, frame in test_rows.groupby("fold", observed=True):
            test_ranges.append(
                {
                    "sealed_fold": int(cast(Any, fold)),
                    "start_date": frame["signal_date"].min().strftime("%Y-%m-%d"),
                    "end_date": frame["signal_date"].max().strftime("%Y-%m-%d"),
                    "sample_count": int(len(frame)),
                    "unique_signal_dates": int(frame["signal_date"].nunique()),
                }
            )
        descriptor = {
            "schema_version": "tcn-sealed-data-descriptor-v36/v1",
            "manifest_path": str(sealed_path),
            "manifest_sha256": expected_hashes["sealed_split_manifest"],
            "manifest_row_count": int(len(sealed_manifest)),
            "stage_counts": {
                str(key): int(value)
                for key, value in sealed_manifest["stage"].astype(str).value_counts().sort_index().items()
            },
            "canonical_test_stage": "test",
            "excluded_duplicate_stage": "sealed_holdout",
            "canonical_test_sample_count": int(len(test_rows)),
            "test_ranges": test_ranges,
            "sealed_values_or_labels_read": False,
        }
        descriptor_path = temporary / "sealed-data-descriptor.json"
        _write_json(descriptor_path, descriptor)

        freeze_payload = {
            "schema_version": "tcn-v35-sealed-freeze-v36/v1",
            "config": payload,
            "candidate": {
                "artifact": str(candidate_artifact),
                "receipt_id": str(payload["v35_receipt_id"]),
                "status": str(payload["v35_candidate_status"]),
                "code_identity": candidate_receipt.get("code_identity"),
            },
            "lstm_benchmark": {
                "artifact": str(lstm_artifact),
                "receipt_id": str(payload["lstm_receipt_id"]),
            },
            "sources": {
                "ordinary_split_manifest": {"path": str(ordinary_path), "sha256": expected_hashes["ordinary_split_manifest"]},
                "sealed_split_manifest": {"path": str(sealed_path), "sha256": expected_hashes["sealed_split_manifest"]},
                **{
                    name: {"path": str(path), "sha256": expected_hashes[name]}
                    for name, path in source_paths.items()
                },
            },
            "checkpoint_plan_sha256": sha256_file(plan_path),
            "sealed_descriptor_sha256": sha256_file(descriptor_path),
            "eligible_unit_count": int(len(checkpoint_plan)),
            "changed_unit_exposures": int(checkpoint_plan["selection_changed"].sum()),
            "authorization_text_sha256": EXACT_SEALED_AUTHORIZATION_SHA256,
        }
        freeze_id = hashlib.sha256(canonical_bytes(freeze_payload)).hexdigest()
        frozen_plan = {"freeze_id": freeze_id, "payload": freeze_payload}
        frozen_plan_path = temporary / "frozen-plan.json"
        _write_json(frozen_plan_path, frozen_plan)
        state = {
            "schema_version": "tcn-v35-sealed-state-v36/v1",
            "freeze_id": freeze_id,
            "status": "awaiting_explicit_sealed_authorization_v36",
            "attempt": 0,
            "authorization_received": False,
            "sealed_test_accessed": False,
            "evaluation_executed": False,
            "consumed_marker_created": False,
        }
        state_path = temporary / "state.json"
        _write_json(state_path, state)
        readiness = {
            "schema_version": "tcn-v35-sealed-readiness-v36/v1",
            "freeze_id": freeze_id,
            "status": state["status"],
            "ready": True,
            "blockers": [],
            "eligible_unit_count": int(len(checkpoint_plan)),
            "changed_unit_exposures": int(checkpoint_plan["selection_changed"].sum()),
            "sealed_fold_unit_counts": {
                str(int(cast(Any, key))): int(value)
                for key, value in checkpoint_plan["sealed_fold"].value_counts().sort_index().items()
            },
            "authorization_received": False,
            "sealed_test_accessed": False,
            "evaluation_executed": False,
        }
        readiness_path = temporary / "readiness.json"
        _write_json(readiness_path, readiness)
        output_hashes = {
            path.name: sha256_file(path)
            for path in (plan_path, descriptor_path, frozen_plan_path, state_path, readiness_path)
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-v35-sealed-readiness-v36/v1",
            "run_id": str(payload["run_id"]),
            "freeze_id": freeze_id,
            "status": state["status"],
            "code_identity": code_identity(root),
            "outputs": output_hashes,
            "authorization_received": False,
            "sealed_test_accessed": False,
            "evaluation_executed": False,
            "engineering_complete": True,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        receipt_path = temporary / "receipt.json"
        _write_json(receipt_path, receipt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                if child.is_file():
                    child.unlink()
            temporary.rmdir()
    return SealedReadinessResult(
        freeze_id=freeze_id,
        output_dir=destination,
        frozen_plan_path=destination / frozen_plan_path.name,
        readiness_path=destination / readiness_path.name,
        state_path=destination / state_path.name,
        receipt_path=destination / receipt_path.name,
    )
