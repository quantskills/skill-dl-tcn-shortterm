"""Highest evidence seam for the bounded TCN-v9 ordinary-validation path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping, cast

import pandas as pd
import numpy as np
import torch

from .experiment import ContractError
from .integrity import code_identity
from .v9_confirmation import V9FinalContext, finalize_v9_run
from .v9_diagnostics import V9DiagnosticRequest, run_v9_diagnostics
from .v9_protocol import V9Plan, execute_v9_plan
from .v9_receipts import canonical_bytes, canonicalize, publish_immutable_receipt
from .v9_selection import build_seed7_trials, select_seed7_candidate
from .v9_training import V9TrainingRequest, run_v9_candidate_sweep


@dataclass(frozen=True)
class V9RunResult:
    status: str
    plan_receipt: Path
    final_receipt: Path
    seed7_leaderboard: pd.DataFrame


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _state_mappings_equal(left: object, right: object) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, torch.Tensor) and isinstance(right_value, torch.Tensor):
            if not torch.equal(left_value.detach().cpu(), right_value.detach().cpu()):
                return False
        elif left_value != right_value:
            return False
    return True


def _read_tabular_artifact(path: Path, *, label: str) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ContractError(f"v9 {label} artifact must be parquet or CSV")


_APPLICABLE_UPSTREAM_STATUSES = {
    "horizon_skip": "horizon_skip_applicable",
    "rank_objective": "rank_objective_allowed",
    "pcgrad": "pcgrad_applicable",
    "infra": "causal_infra_acceleration_accepted",
}
_ALLOWED_UPSTREAM_STATUSES = {
    "horizon_skip": {"horizon_skip_applicable", "horizon_skip_not_applicable"},
    "rank_objective": {
        "rank_objective_allowed",
        "rank_objective_not_resolvable",
        "rank_objective_not_applicable",
    },
    "pcgrad": {"pcgrad_applicable", "pcgrad_not_applicable"},
    "infra": {
        "causal_infra_acceleration_accepted",
        "infra_optimization_not_applicable",
    },
}


def _validate_applicable_evidence(
    diagnostic: str,
    evidence: Mapping[str, object],
) -> None:
    try:
        if diagnostic == "horizon_skip":
            valid = (
                float(cast(Any, evidence["mean_improvement"])) >= 0.002
                and int(cast(Any, evidence["positive_fold_count"])) >= 3
                and float(cast(Any, evidence["ci_low"])) > 0
                and int(cast(Any, evidence["selected_block"])) >= 0
                and evidence["model_family"] == "tcn-lite-16"
            )
        elif diagnostic == "rank_objective":
            valid = (
                int(cast(Any, evidence["minimum_paired_date_count"])) >= 40
                and float(cast(Any, evidence["maximum_degenerate_bootstrap_rate"])) <= 0.05
                and float(cast(Any, evidence["maximum_minimum_detectable_effect"])) <= 0.005
            )
        elif diagnostic == "pcgrad":
            valid = (
                int(cast(Any, evidence["conflicting_fold_count"])) >= 3
                and float(cast(Any, evidence["median_cosine"])) < 0
                and float(cast(Any, evidence["negative_batch_rate"])) >= 0.30
            )
        else:
            valid = (
                float(cast(Any, evidence["padding_self_cpu_share"])) >= 0.10
                and float(cast(Any, evidence["throughput_gain"])) >= 0.10
                and evidence["numerically_equivalent"] is True
                and evidence["gradient_equivalent"] is True
                and evidence["strictly_causal"] is True
                and float(cast(Any, evidence["learning_rate"])) == 0.003
                and float(cast(Any, evidence["eager_samples_per_second"])) > 0
                and float(cast(Any, evidence["candidate_samples_per_second"])) > 0
                and float(
                    cast(Any, evidence["eager_model_step_seconds_median"])
                )
                > 0
                and float(
                    cast(Any, evidence["candidate_model_step_seconds_median"])
                )
                > 0
                and float(cast(Any, evidence["eager_complete_cycle_seconds"])) > 0
                and float(
                    cast(Any, evidence["candidate_complete_cycle_seconds"])
                )
                > 0
                and np.isfinite(float(cast(Any, evidence["measurement_noise"])))
                and np.isfinite(
                    float(cast(Any, evidence["candidate_measurement_noise"]))
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(
            f"v9 {diagnostic} applicable evidence is incomplete"
        ) from exc
    if not valid:
        raise ContractError(f"v9 {diagnostic} evidence does not pass its frozen gate")


def publish_v9_upstream_receipt(
    diagnostic: str,
    status: str,
    evidence: Mapping[str, object],
    source_identities: Mapping[str, str],
    *,
    output_dir: Path,
    project_root: Path,
) -> Path:
    """Publish one content-addressed diagnostic receipt consumed by formal v9."""

    if diagnostic not in _APPLICABLE_UPSTREAM_STATUSES:
        raise ContractError("v9 upstream diagnostic name is unsupported")
    if status not in _ALLOWED_UPSTREAM_STATUSES[diagnostic]:
        raise ContractError("v9 upstream diagnostic status is unsupported")
    if status == _APPLICABLE_UPSTREAM_STATUSES[diagnostic]:
        _validate_applicable_evidence(diagnostic, evidence)
    payload: dict[str, object] = {
        "schema_version": "tcn-v9-upstream/v1",
        "diagnostic": diagnostic,
        "status": status,
        "source_identities": dict(sorted(source_identities.items())),
        "code_identity": code_identity(project_root.resolve()),
        "evidence": dict(evidence),
        "sealed_test_accessed": False,
    }
    payload["receipt_id"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return publish_immutable_receipt(
        payload,
        output_dir=output_dir,
        filename=f"{diagnostic}-receipt.json",
        identity_label=f"v9 {diagnostic} diagnostic",
    )


def _resolve_upstream_receipts(
    supplied: Mapping[str, Mapping[str, object]],
    *,
    plan: V9Plan,
    project_root: Path,
) -> dict[str, Mapping[str, object]]:
    if set(supplied) != set(_APPLICABLE_UPSTREAM_STATUSES):
        raise ContractError("v9 requires exactly four upstream diagnostic results")
    resolved: dict[str, Mapping[str, object]] = {}
    current_code = code_identity(project_root.resolve())
    for diagnostic, receipt in supplied.items():
        status = str(receipt.get("status", ""))
        if status not in _ALLOWED_UPSTREAM_STATUSES[diagnostic]:
            raise ContractError(f"v9 {diagnostic} status is unsupported")
        raw_path = receipt.get("receipt_path")
        if not isinstance(raw_path, (str, Path)):
            raise ContractError(
                f"applicable v9 {diagnostic} requires an immutable receipt path"
            )
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ContractError(f"v9 {diagnostic} receipt is unavailable")
        try:
            payload = cast(
                dict[str, object],
                json.loads(path.read_text(encoding="utf-8")),
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"v9 {diagnostic} receipt is unreadable") from exc
        if (
            payload.get("schema_version") != "tcn-v9-upstream/v1"
            or payload.get("diagnostic") != diagnostic
            or payload.get("status") != status
            or payload.get("sealed_test_accessed") is not False
            or not isinstance(payload.get("evidence"), Mapping)
        ):
            raise ContractError(f"v9 {diagnostic} receipt contract is invalid")
        if status == _APPLICABLE_UPSTREAM_STATUSES[diagnostic]:
            _validate_applicable_evidence(
                diagnostic,
                cast(Mapping[str, object], payload["evidence"]),
            )
        receipt_id = payload.pop("receipt_id", None)
        expected_id = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        payload["receipt_id"] = receipt_id
        if receipt_id != expected_id:
            raise ContractError(f"v9 {diagnostic} receipt identity is invalid")
        identities = payload.get("source_identities")
        if not isinstance(identities, Mapping) or dict(identities) != dict(
            plan.source_identities
        ):
            raise ContractError(f"v9 {diagnostic} receipt source identity drifted")
        receipt_code = payload.get("code_identity")
        if not isinstance(receipt_code, Mapping) or receipt_code.get(
            "source_sha256"
        ) != current_code.get("source_sha256"):
            raise ContractError(f"v9 {diagnostic} receipt code identity drifted")
        resolved[diagnostic] = payload
    return resolved


def _validate_training_contract(
    plan: V9Plan,
    request: V9TrainingRequest,
    registered_trial_ids: set[str],
    entry_split_manifest: pd.DataFrame,
    *,
    validate_trial_ids: bool = True,
) -> None:
    if tuple(sorted(plan.fold_ids)) != tuple(
        sorted(request.split_manifest["fold"].astype(int).unique())
    ):
        raise ContractError("v9 training folds drifted from the frozen plan")
    if int(request.features.shape[2]) != plan.input_steps:
        raise ContractError("v9 training input steps drifted from the frozen plan")
    if request.torch_threads != plan.torch_threads:
        raise ContractError("v9 training thread scope drifted from the frozen plan")
    if (
        request.max_epochs,
        request.patience,
        request.min_delta,
    ) != (plan.max_epochs, plan.patience, plan.min_delta):
        raise ContractError("v9 training early stopping drifted from the frozen plan")
    if (
        request.channels,
        request.kernel_size,
        request.dilations,
        request.dropout,
        request.learning_rate,
        request.batch_size,
    ) != (
        plan.channels,
        plan.kernel_size,
        plan.dilations,
        plan.dropout,
        plan.learning_rate,
        plan.batch_size,
    ):
        raise ContractError("v9 training model or optimizer drifted from the frozen plan")
    for name in (
        "data",
        "fold_manifest",
        "window_index",
        "labels",
        "evaluation",
    ):
        if request.protocol_identities.get(name) != plan.source_identities.get(name):
            raise ContractError(f"v9 training identity drift detected for {name}")
    if request.artifact_paths is None or set(request.artifact_paths) != {
        "data",
        "fold_manifest",
        "window_index",
        "labels",
        "evaluation",
    }:
        raise ContractError("v9 formal training requires exact input artifact paths")
    resolved_paths = {
        name: Path(path).resolve() for name, path in request.artifact_paths.items()
    }
    for name, path in resolved_paths.items():
        if not path.is_file():
            raise ContractError(f"v9 training artifact is unavailable: {name}")
        digest = _file_sha256(path)
        if digest != plan.source_identities.get(name):
            raise ContractError(f"v9 training artifact bytes drifted for {name}")
    if not isinstance(request.features, np.memmap):
        raise ContractError("v9 formal training requires memory-mapped feature input")
    if Path(str(request.features.filename)).resolve() != resolved_paths["data"]:
        raise ContractError("v9 feature memmap does not reference the frozen artifact")
    artifact_split = _read_tabular_artifact(
        resolved_paths["fold_manifest"], label="split"
    )
    artifact_window_index = _read_tabular_artifact(
        resolved_paths["window_index"], label="window index"
    )
    artifact_labels = _read_tabular_artifact(
        resolved_paths["labels"], label="labels"
    )
    comparable_columns = sorted(request.split_manifest.columns)
    try:
        pd.testing.assert_frame_equal(
            request.split_manifest[comparable_columns].reset_index(drop=True),
            artifact_split[comparable_columns].reset_index(drop=True),
            check_dtype=False,
        )
        window_columns = sorted(request.window_index.columns)
        pd.testing.assert_frame_equal(
            request.window_index[window_columns].reset_index(drop=True),
            artifact_window_index[window_columns].reset_index(drop=True),
            check_dtype=False,
        )
        label_columns = sorted(request.labels.columns)
        pd.testing.assert_frame_equal(
            request.labels[label_columns].reset_index(drop=True),
            artifact_labels[label_columns].reset_index(drop=True),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            request.split_manifest[comparable_columns].reset_index(drop=True),
            entry_split_manifest[comparable_columns].reset_index(drop=True),
            check_dtype=False,
        )
    except (AssertionError, KeyError) as exc:
        raise ContractError(
            "v9 training frames differ from their frozen artifacts"
        ) from exc
    planned_ids = {
        trial.trial_id
        for trial in plan.trials
        if trial.trigger_status == "applicable"
    }
    if validate_trial_ids and planned_ids != registered_trial_ids:
        raise ContractError("v9 registered trials drifted from the frozen plan")


def _publish_seed7_receipt(
    leaderboard: pd.DataFrame,
    best_states: Mapping[str, Mapping[str, object]],
    *,
    status: str,
    winner_trial_id: str | None,
    source_identities: Mapping[str, str],
    output_dir: Path,
    project_root: Path,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": "tcn-v9-seed7/v1",
        "status": status,
        "winner_trial_id": winner_trial_id,
        "source_identities": dict(sorted(source_identities.items())),
        "code_identity": code_identity(project_root.resolve()),
        "leaderboard": (
            []
            if leaderboard.empty
            else leaderboard.sort_values(
                ["trial_id", "fold", "seed"], kind="mergesort"
            ).to_dict(orient="records")
        ),
        "sealed_test_accessed": False,
    }
    if best_states:
        checkpoint_receipt = _publish_checkpoint_bundle(
            best_states,
            source_identities=source_identities,
            output_dir=output_dir.with_name(output_dir.name + "-checkpoints"),
            project_root=project_root,
            stage="seed7",
        )
        checkpoint_payload = json.loads(
            checkpoint_receipt.read_text(encoding="utf-8")
        )
        payload["checkpoint_bundle"] = {
            "path": str(checkpoint_receipt),
            "receipt_id": checkpoint_payload["receipt_id"],
        }
    payload["receipt_id"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return publish_immutable_receipt(
        payload,
        output_dir=output_dir,
        filename="seed7-receipt.json",
        identity_label="v9 seed-7 screen",
    )


def _publish_checkpoint_bundle(
    best_states: Mapping[str, Mapping[str, object]],
    *,
    source_identities: Mapping[str, str],
    output_dir: Path,
    project_root: Path,
    stage: str,
) -> Path:
    if not best_states:
        raise ContractError("v9 checkpoint bundle cannot be empty")
    destination = output_dir.resolve()
    receipt_path = destination / "checkpoint-receipt.json"
    if destination.exists():
        if not receipt_path.is_file():
            raise ContractError("v9 checkpoint bundle exists without a receipt")
        observed = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            observed.get("stage") != stage
            or dict(observed.get("source_identities", {}))
            != dict(sorted(source_identities.items()))
            or not isinstance(observed.get("code_identity"), Mapping)
            or cast(Mapping[str, object], observed["code_identity"]).get(
                "source_sha256"
            )
            != code_identity(project_root.resolve()).get("source_sha256")
        ):
            raise ContractError("v9 checkpoint bundle identity drifted")
        entries = cast(list[dict[str, object]], observed.get("checkpoints", []))
        expected_keys = set(best_states)
        observed_keys = {str(entry.get("checkpoint_key", "")) for entry in entries}
        if expected_keys != observed_keys:
            raise ContractError("v9 checkpoint bundle content drifted")
        for entry in entries:
            path = destination / str(entry["path"])
            if not path.is_file() or _file_sha256(path) != entry["sha256"]:
                raise ContractError("v9 checkpoint bundle artifact identity drifted")
            try:
                stored_state = torch.load(path, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ContractError("v9 checkpoint bundle cannot restore state") from exc
            current_state = best_states[str(entry["checkpoint_key"])]
            if not _state_mappings_equal(stored_state, current_state):
                raise ContractError("v9 checkpoint bundle content drifted")
        return receipt_path
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise ContractError("v9 checkpoint bundle temporary path already exists")
    checkpoint_dir = temporary / "artifacts"
    checkpoint_dir.mkdir(parents=True)
    entries = []
    for key, state in sorted(best_states.items()):
        if not key or Path(key).name != key:
            raise ContractError("v9 checkpoint key is not path safe")
        relative = Path("artifacts") / f"{key}.pt"
        path = temporary / relative
        torch.save(dict(state), path)
        entries.append(
            {
                "checkpoint_key": key,
                "path": relative.as_posix(),
                "sha256": _file_sha256(path),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "tcn-v9-checkpoints/v1",
        "stage": stage,
        "source_identities": dict(sorted(source_identities.items())),
        "code_identity": code_identity(project_root.resolve()),
        "checkpoints": entries,
        "sealed_test_accessed": False,
    }
    payload["receipt_id"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    (temporary / "checkpoint-receipt.json").write_bytes(
        canonical_bytes(payload) + b"\n"
    )
    temporary.replace(destination)
    return receipt_path


def _load_seed7_receipt(
    path: Path,
    *,
    plan: V9Plan,
    project_root: Path,
) -> pd.DataFrame:
    try:
        payload = cast(
            dict[str, object],
            json.loads(path.resolve().read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("v9 seed-7 receipt is unavailable or unreadable") from exc
    receipt_id = payload.pop("receipt_id", None)
    expected_id = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    payload["receipt_id"] = receipt_id
    if (
        payload.get("schema_version") != "tcn-v9-seed7/v1"
        or receipt_id != expected_id
        or payload.get("sealed_test_accessed") is not False
    ):
        raise ContractError("v9 seed-7 receipt identity is invalid")
    identities = payload.get("source_identities")
    if not isinstance(identities, Mapping) or dict(identities) != dict(
        plan.source_identities
    ):
        raise ContractError("v9 seed-7 receipt source identity drifted")
    receipt_code = payload.get("code_identity")
    current_code = code_identity(project_root.resolve())
    if not isinstance(receipt_code, Mapping) or receipt_code.get(
        "source_sha256"
    ) != current_code.get("source_sha256"):
        raise ContractError("v9 seed-7 receipt code identity drifted")
    records = payload.get("leaderboard")
    if not isinstance(records, list):
        raise ContractError("v9 seed-7 receipt leaderboard is invalid")
    checkpoint_bundle = payload.get("checkpoint_bundle")
    if not isinstance(checkpoint_bundle, Mapping):
        if (
            payload.get("status") == "stop_no_pareto_gain_v9"
            and payload.get("winner_trial_id") is None
            and not records
        ):
            return pd.DataFrame(records)
        raise ContractError("v9 seed-7 checkpoint bundle is missing")
    checkpoint_path = Path(str(checkpoint_bundle.get("path", ""))).resolve()
    if not checkpoint_path.is_file():
        raise ContractError("v9 seed-7 checkpoint receipt is unavailable")
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_receipt_id = checkpoint_payload.pop("receipt_id", None)
    expected_checkpoint_receipt_id = hashlib.sha256(
        canonical_bytes(checkpoint_payload)
    ).hexdigest()
    checkpoint_payload["receipt_id"] = checkpoint_receipt_id
    if (
        checkpoint_payload.get("schema_version") != "tcn-v9-checkpoints/v1"
        or checkpoint_payload.get("stage") != "seed7"
        or checkpoint_payload.get("sealed_test_accessed") is not False
        or checkpoint_receipt_id != expected_checkpoint_receipt_id
        or checkpoint_receipt_id != checkpoint_bundle.get("receipt_id")
    ):
        raise ContractError("v9 seed-7 checkpoint receipt identity drifted")
    checkpoint_root = checkpoint_path.parent
    if dict(checkpoint_payload.get("source_identities", {})) != dict(
        plan.source_identities
    ):
        raise ContractError("v9 seed-7 checkpoint source identity drifted")
    nested_code = checkpoint_payload.get("code_identity")
    if not isinstance(nested_code, Mapping) or nested_code.get(
        "source_sha256"
    ) != current_code.get("source_sha256"):
        raise ContractError("v9 seed-7 checkpoint code identity drifted")
    for entry in checkpoint_payload.get("checkpoints", []):
        artifact_path = checkpoint_root / str(entry["path"])
        if not artifact_path.is_file() or _file_sha256(artifact_path) != entry[
            "sha256"
        ]:
            raise ContractError("v9 seed-7 checkpoint artifact identity drifted")
    return pd.DataFrame(records)


def publish_v9_confirmation_reference_receipt(
    measurements: pd.DataFrame,
    source_identities: Mapping[str, str],
    *,
    output_dir: Path,
    project_root: Path,
    control_model_id: str = "tcn-lite-4",
    lstm_model_id: str = "lstm",
    gru_model_id: str = "gru",
) -> Path:
    """Publish frozen 3-model, 15-unit reference evidence for confirmation."""

    required = {
        "model",
        "fold",
        "seed",
        "rankic",
        "samples_per_second",
        "model_step_samples_per_second",
        "model_step_seconds",
        "data_wait_seconds",
        "validation_seconds",
        "complete_cycle_seconds",
        "time_to_best_seconds",
        "parameter_count",
        "precision",
        "torch_threads",
        "batch_size",
        "data_identity",
        "fold_identity",
        "evaluation_identity",
        "max_epochs",
        "patience",
        "min_delta",
        "loss_identity",
        "infra_identity",
        "candidate_config_identity",
        "simplex_weights",
        "sealed_test_accessed",
    }
    if missing := sorted(required.difference(measurements.columns)):
        raise ContractError(
            f"v9 reference measurements missing columns: {', '.join(missing)}"
        )
    expected_models = {control_model_id, lstm_model_id, gru_model_id}
    if set(measurements["model"].astype(str)) != expected_models:
        raise ContractError("v9 reference receipt model set is invalid")
    if measurements["sealed_test_accessed"].astype(bool).any():
        raise ContractError("v9 reference receipt rejects sealed evidence")
    for column, identity_key in {
        "data_identity": "data",
        "fold_identity": "fold_manifest",
        "evaluation_identity": "evaluation",
    }.items():
        if set(measurements[column].astype(str)) != {
            source_identities.get(identity_key)
        }:
            raise ContractError("v9 reference receipt source columns drifted")
    expected_units = {
        (fold, seed) for fold in range(5) for seed in (7, 17, 27)
    }
    for _, rows in measurements.groupby("model", observed=True):
        units = set(
            zip(rows["fold"].astype(int), rows["seed"].astype(int), strict=True)
        )
        if units != expected_units or len(rows) != 15:
            raise ContractError("v9 reference receipt requires 15 units per model")
    payload: dict[str, object] = {
        "schema_version": "tcn-v9-confirmation-reference/v1",
        "source_identities": dict(sorted(source_identities.items())),
        "code_identity": code_identity(project_root.resolve()),
        "measurements": measurements.sort_values(
            ["model", "seed", "fold"], kind="mergesort"
        ).to_dict(orient="records"),
        "sealed_test_accessed": False,
    }
    payload["receipt_id"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return publish_immutable_receipt(
        payload,
        output_dir=output_dir,
        filename="confirmation-reference-receipt.json",
        identity_label="v9 confirmation references",
    )


def _load_confirmation_reference_receipt(
    path: Path,
    *,
    plan: V9Plan,
    project_root: Path,
) -> pd.DataFrame:
    try:
        payload = cast(
            dict[str, object],
            json.loads(path.resolve().read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("v9 confirmation reference receipt is unreadable") from exc
    receipt_id = payload.pop("receipt_id", None)
    expected_id = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    payload["receipt_id"] = receipt_id
    if (
        payload.get("schema_version") != "tcn-v9-confirmation-reference/v1"
        or receipt_id != expected_id
        or payload.get("sealed_test_accessed") is not False
    ):
        raise ContractError("v9 confirmation reference receipt identity is invalid")
    identities = payload.get("source_identities")
    if not isinstance(identities, Mapping) or dict(identities) != dict(
        plan.source_identities
    ):
        raise ContractError("v9 confirmation reference source identity drifted")
    receipt_code = payload.get("code_identity")
    current_code = code_identity(project_root.resolve())
    if not isinstance(receipt_code, Mapping) or receipt_code.get(
        "source_sha256"
    ) != current_code.get("source_sha256"):
        raise ContractError("v9 confirmation reference code identity drifted")
    records = payload.get("measurements")
    if not isinstance(records, list):
        raise ContractError("v9 confirmation reference measurements are invalid")
    measurements = pd.DataFrame(records)
    if (
        "simplex_weights" not in measurements
        or measurements["simplex_weights"].notna().any()
    ):
        raise ContractError(
            "v9 confirmation reference simplex metadata is invalid"
        )
    for column, identity_key in {
        "data_identity": "data",
        "fold_identity": "fold_manifest",
        "evaluation_identity": "evaluation",
    }.items():
        if set(measurements.get(column, pd.Series(dtype=str)).astype(str)) != {
            plan.source_identities[identity_key]
        }:
            raise ContractError("v9 confirmation reference identity columns drifted")
    expected_protocol = {
        "precision": plan.precision,
        "torch_threads": plan.torch_threads,
        "batch_size": plan.batch_size,
        "max_epochs": plan.max_epochs,
        "patience": plan.patience,
        "min_delta": plan.min_delta,
    }
    for column, expected_value in expected_protocol.items():
        if set(measurements.get(column, pd.Series(dtype=object))) != {
            expected_value
        }:
            raise ContractError("v9 confirmation reference protocol drifted")
    return measurements


def _receipt_id(path: Path) -> str:
    try:
        value = json.loads(path.resolve().read_text(encoding="utf-8")).get(
            "receipt_id"
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("v9 receipt identity is unavailable") from exc
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError("v9 receipt identity is invalid")
    return value


def _checkpoint_identities_from_receipt(
    receipt_path: Path,
    *,
    prefix: str,
) -> dict[str, str]:
    payload = json.loads(receipt_path.resolve().read_text(encoding="utf-8"))
    bundle = payload.get("checkpoint_bundle")
    if not isinstance(bundle, Mapping):
        return {}
    checkpoint_receipt_path = Path(str(bundle.get("path", ""))).resolve()
    checkpoint_payload = json.loads(
        checkpoint_receipt_path.read_text(encoding="utf-8")
    )
    return {
        f"{prefix}:{entry['checkpoint_key']}": str(entry["sha256"])
        for entry in checkpoint_payload.get("checkpoints", [])
    }


def _checkpoint_identities_from_bundle(
    receipt_path: Path,
    *,
    prefix: str,
) -> dict[str, str]:
    payload = json.loads(receipt_path.resolve().read_text(encoding="utf-8"))
    return {
        f"{prefix}:{entry['checkpoint_key']}": str(entry["sha256"])
        for entry in payload.get("checkpoints", [])
    }


def _derive_final_context(
    plan: V9Plan,
    *,
    plan_receipt: Path,
    upstream_payloads: Mapping[str, Mapping[str, object]],
    seed7_receipt: Path,
    reference_receipt: Path | None = None,
    confirmation_checkpoint_receipt: Path | None = None,
) -> V9FinalContext:
    upstream_ids = {
        "plan": _receipt_id(plan_receipt),
        "seed7": _receipt_id(seed7_receipt),
        **{
            f"diagnostic:{name}": str(payload["receipt_id"])
            for name, payload in upstream_payloads.items()
        },
    }
    checkpoint_ids = _checkpoint_identities_from_receipt(
        seed7_receipt,
        prefix="seed7",
    )
    if reference_receipt is not None:
        upstream_ids["confirmation_references"] = _receipt_id(reference_receipt)
    if confirmation_checkpoint_receipt is not None:
        upstream_ids["confirmation_checkpoints"] = _receipt_id(
            confirmation_checkpoint_receipt
        )
        checkpoint_ids.update(
            _checkpoint_identities_from_bundle(
                confirmation_checkpoint_receipt,
                prefix="confirmation",
            )
        )
    return V9FinalContext(
        resolved_config=cast(Mapping[str, object], canonicalize(asdict(plan))),
        source_identities=dict(sorted(plan.source_identities.items())),
        checkpoint_identities=checkpoint_ids,
        environment={
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "active_torch_threads": torch.get_num_threads(),
            "requested_torch_threads": plan.torch_threads,
        },
        upstream_receipts=upstream_ids,
    )


def run_v9_evidence_path(
    plan: V9Plan,
    split_manifest: pd.DataFrame,
    *,
    upstream_receipts: Mapping[str, Mapping[str, object]],
    seed7_leaderboard: pd.DataFrame,
    confirmation_measurements: pd.DataFrame,
    context: V9FinalContext,
    output_root: Path,
    project_root: Path,
    training_request: V9TrainingRequest | None = None,
    diagnostic_request: V9DiagnosticRequest | None = None,
    prior_seed7_receipt: Path | None = None,
    confirmation_reference_receipt: Path | None = None,
    control_seed7_trial_id: str = "lite-c16-no-dropout",
    confirmation_control_id: str = "tcn-lite-4",
    lstm_model_id: str = "lstm",
    gru_model_id: str = "gru",
) -> V9RunResult:
    """Validate one frozen plan and consume only registered validation evidence."""

    if dict(plan.source_identities) != dict(context.source_identities):
        raise ContractError("v9 evidence context source identities drifted")
    if control_seed7_trial_id != plan.control_trial_id:
        raise ContractError("v9 control trial ID drifted from the frozen plan")
    plan_receipt = execute_v9_plan(
        plan,
        split_manifest,
        output_dir=output_root / "plan",
        project_root=project_root,
    )
    if plan.stage == "diagnostic":
        if diagnostic_request is None:
            raise ContractError("v9 diagnostic stage requires executable raw inputs")
        _validate_training_contract(
            plan,
            diagnostic_request.training,
            set(),
            split_manifest,
            validate_trial_ids=False,
        )

        def publish_diagnostic(
            diagnostic: str,
            status: str,
            evidence: Mapping[str, object],
            identities: Mapping[str, str],
        ) -> Path:
            return publish_v9_upstream_receipt(
                diagnostic,
                status,
                evidence,
                identities,
                output_dir=output_root / "diagnostics" / diagnostic,
                project_root=project_root,
            )

        diagnostics = run_v9_diagnostics(
            diagnostic_request,
            plan,
            publish_receipt=publish_diagnostic,
        )
        summary_payload: dict[str, object] = {
            "schema_version": "tcn-v9-diagnostic-summary/v1",
            "status": "diagnostics_complete",
            "source_identities": dict(sorted(plan.source_identities.items())),
            "code_identity": code_identity(project_root.resolve()),
            "upstream_results": diagnostics.upstream_receipts,
            "sealed_test_accessed": False,
        }
        summary_payload["receipt_id"] = hashlib.sha256(
            canonical_bytes(summary_payload)
        ).hexdigest()
        summary_receipt = publish_immutable_receipt(
            summary_payload,
            output_dir=output_root / "diagnostic-summary",
            filename="diagnostic-summary.json",
            identity_label="v9 diagnostic summary",
        )
        return V9RunResult(
            "diagnostics_complete",
            plan_receipt,
            summary_receipt,
            pd.DataFrame(),
        )
    resolved_upstream = _resolve_upstream_receipts(
        upstream_receipts,
        plan=plan,
        project_root=project_root,
    )
    registered_trials = build_seed7_trials(resolved_upstream)
    if not seed7_leaderboard.empty or not confirmation_measurements.empty:
        raise ContractError(
            "v9 rejects caller-supplied evidence DataFrames; use immutable receipts"
        )
    if training_request is None:
        raise ContractError("formal v9 stages require ordinary-validation inputs")
    _validate_training_contract(
        plan,
        training_request,
        {trial.trial_id for trial in registered_trials},
        split_manifest,
    )
    if plan.stage == "formal_screen":
        if prior_seed7_receipt is not None or confirmation_reference_receipt is not None:
            raise ContractError("v9 formal screen cannot consume confirmation evidence")
        best_states: Mapping[str, Mapping[str, object]] = {}
        selection_trials = registered_trials
        if registered_trials:
            training_result = run_v9_candidate_sweep(
                training_request,
                registered_trials=registered_trials,
                seed=7,
                control_trial_id=control_seed7_trial_id,
            )
            resolved_seed7_leaderboard = training_result.leaderboard
            best_states = training_result.best_states
            executed_trial_ids = set(
                resolved_seed7_leaderboard["trial_id"].astype(str)
            ) - {control_seed7_trial_id}
            selection_trials = tuple(
                trial
                for trial in registered_trials
                if trial.trial_id in executed_trial_ids
            )
        else:
            resolved_seed7_leaderboard = pd.DataFrame()
        seed7_decision = select_seed7_candidate(
            resolved_seed7_leaderboard,
            registered_trials=selection_trials,
            control_trial_id=control_seed7_trial_id,
        )
        screen_receipt = _publish_seed7_receipt(
            resolved_seed7_leaderboard,
            best_states,
            status=seed7_decision.status,
            winner_trial_id=seed7_decision.winner_trial_id,
            source_identities=plan.source_identities,
            output_dir=output_root / "seed7",
            project_root=project_root,
        )
        return V9RunResult(
            seed7_decision.status,
            plan_receipt,
            screen_receipt,
            resolved_seed7_leaderboard,
        )
    if prior_seed7_receipt is None:
        raise ContractError("v9 confirmation requires an immutable seed-7 receipt")
    resolved_seed7_leaderboard = _load_seed7_receipt(
        prior_seed7_receipt,
        plan=plan,
        project_root=project_root,
    )
    observed_seed7_trial_ids = set(
        resolved_seed7_leaderboard.get("trial_id", pd.Series(dtype=str)).astype(str)
    ) - {control_seed7_trial_id}
    confirmation_selection_trials = tuple(
        trial
        for trial in registered_trials
        if trial.trial_id in observed_seed7_trial_ids
    )
    seed7_decision = select_seed7_candidate(
        resolved_seed7_leaderboard,
        registered_trials=confirmation_selection_trials,
        control_trial_id=control_seed7_trial_id,
    )
    if seed7_decision.status != "seed7_winner_admitted":
        stopped_context = _derive_final_context(
            plan,
            plan_receipt=plan_receipt,
            upstream_payloads=resolved_upstream,
            seed7_receipt=prior_seed7_receipt,
        )
        final_receipt = finalize_v9_run(
            pd.DataFrame(),
            seed7_decision=seed7_decision,
            context=stopped_context,
            output_dir=output_root / "final",
            project_root=project_root,
            control_trial_id=confirmation_control_id,
            lstm_model_id=lstm_model_id,
            gru_model_id=gru_model_id,
        )
        return V9RunResult(
            "stop_no_pareto_gain_v9",
            plan_receipt,
            final_receipt,
            resolved_seed7_leaderboard,
        )
    if confirmation_reference_receipt is None:
        raise ContractError(
            "v9 confirmation requires an immutable reference-model receipt"
        )
    resolved_confirmation = _load_confirmation_reference_receipt(
        confirmation_reference_receipt,
        plan=plan,
        project_root=project_root,
    )
    if seed7_decision.status == "seed7_winner_admitted":
        winner_id = seed7_decision.winner_trial_id
        if winner_id is None:
            raise ContractError("seed-7 admission is missing a winner trial ID")
        winner_trials = tuple(
            trial for trial in registered_trials if trial.trial_id == winner_id
        )
        if len(winner_trials) != 1:
            raise ContractError("seed-7 winner is not a uniquely registered v9 trial")
        candidate_parts = [
            resolved_seed7_leaderboard.loc[
                resolved_seed7_leaderboard["trial_id"].eq(winner_id)
            ].copy()
        ]
        winner_parent_kind = str(
            resolved_seed7_leaderboard.loc[
                resolved_seed7_leaderboard["trial_id"].eq(winner_id),
                "model_kind",
            ].iloc[0]
        )
        confirmation_states: dict[str, Mapping[str, object]] = {}
        for confirmation_seed in plan.seeds:
            confirmation_training = run_v9_candidate_sweep(
                training_request,
                registered_trials=winner_trials,
                seed=confirmation_seed,
                control_trial_id=control_seed7_trial_id,
                frozen_parent_kind=winner_parent_kind,
            )
            candidate_parts.append(
                confirmation_training.leaderboard.loc[
                    confirmation_training.leaderboard["trial_id"].eq(winner_id)
                ].copy()
            )
            for checkpoint_key, state in confirmation_training.best_states.items():
                confirmation_states[
                    f"seed-{confirmation_seed}-{checkpoint_key}"
                ] = state
        candidate_measurements = pd.concat(candidate_parts, ignore_index=True)
        reference_models = {
            confirmation_control_id,
            lstm_model_id,
            gru_model_id,
        }
        if set(resolved_confirmation.get("model", pd.Series(dtype=str)).astype(str)) != reference_models:
            raise ContractError(
                "v9 confirmation requires exactly the frozen control/LSTM/GRU references"
            )
        resolved_confirmation = pd.concat(
            [candidate_measurements, resolved_confirmation],
            ignore_index=True,
        )
        checkpoint_receipt = _publish_checkpoint_bundle(
            confirmation_states,
            source_identities=plan.source_identities,
            output_dir=output_root / "confirmation-checkpoints",
            project_root=project_root,
            stage="confirmation",
        )
        context = _derive_final_context(
            plan,
            plan_receipt=plan_receipt,
            upstream_payloads=resolved_upstream,
            seed7_receipt=prior_seed7_receipt,
            reference_receipt=confirmation_reference_receipt,
            confirmation_checkpoint_receipt=checkpoint_receipt,
        )
    final_receipt = finalize_v9_run(
        resolved_confirmation,
        seed7_decision=seed7_decision,
        context=context,
        output_dir=output_root / "final",
        project_root=project_root,
        control_trial_id=confirmation_control_id,
        lstm_model_id=lstm_model_id,
        gru_model_id=gru_model_id,
    )
    final_status = str(
        json.loads(final_receipt.read_text(encoding="utf-8"))["status"]
    )
    return V9RunResult(
        final_status,
        plan_receipt,
        final_receipt,
        resolved_seed7_leaderboard,
    )
