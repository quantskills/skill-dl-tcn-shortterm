"""Fail-closed readiness checks for an owner-supplied real-data pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .experiment import ContractError

TOP_LEVEL_FIELDS = {
    "schema_version",
    "deliverable",
    "runtime_manifest_path",
    "data_governance",
    "evaluation_protocol",
    "compute_protocol",
    "research_budget",
}
DATA_STRING_FIELDS = {
    "provider",
    "license_reference",
    "source_version",
    "data_owner",
    "timezone",
    "trading_calendar",
    "bar_timestamp_semantics",
    "adjustment_policy",
    "availability_policy",
    "raw_schema_version",
    "canonical_schema_version",
    "universe_version",
    "feature_version",
    "label_version",
}
DATA_BOOLEAN_FIELDS = {
    "license_approved",
    "pit_instrument_state",
    "pit_corporate_actions",
    "survivorship_bias_controlled",
}
REQUIRED_MODELS = {"ridge", "lightgbm", "lstm", "gru", "bai-tcn"}
REQUIRED_METRICS = {
    "rankic",
    "icir",
    "net_long_only_return",
    "throughput_samples_per_second",
    "time_to_best_validation_seconds",
    "peak_memory_bytes",
}
RUNTIME_FILES = {
    "data_path": "data_sha256",
    "instrument_state_path": "instrument_state_sha256",
    "corporate_action_path": "corporate_action_sha256",
    "execution_state_path": "execution_state_sha256",
}
SECRET_KEY_FRAGMENTS = {
    "password",
    "secret",
    "token",
    "api_key",
    "private_key",
    "credential",
}
PLACEHOLDER_VALUES = {"todo", "tbd", "change_me", "changeme"}


@dataclass(frozen=True)
class ReadinessCheck:
    """One deterministic readiness assertion."""

    code: str
    status: str
    message: str


@dataclass(frozen=True)
class PilotReadinessReport:
    """Machine-readable result that never implies alpha or release authority."""

    ready: bool
    checks: tuple[ReadinessCheck, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    descriptor_sha256: str
    runtime_manifest_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "descriptor_sha256": self.descriptor_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
        }


class _Collector:
    def __init__(self) -> None:
        self.checks: list[ReadinessCheck] = []
        self.errors: list[str] = []

    def check(self, condition: bool, code: str, message: str) -> None:
        status = "pass" if condition else "fail"
        self.checks.append(ReadinessCheck(code=code, status=status, message=message))
        if not condition:
            self.errors.append(f"{code}: {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return payload


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        not normalized
        or normalized in PLACEHOLDER_VALUES
        or (normalized.startswith("<") and normalized.endswith(">"))
    )


def _scan_for_unsafe_values(
    value: Any, collector: _Collector, *, path: str = "descriptor"
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).lower()
            collector.check(
                not any(fragment in normalized_key for fragment in SECRET_KEY_FRAGMENTS),
                "no-secret-keys",
                f"secret-like key is forbidden at {path}.{key}",
            )
            _scan_for_unsafe_values(nested, collector, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_for_unsafe_values(nested, collector, path=f"{path}[{index}]")
    elif isinstance(value, str):
        collector.check(
            not _is_placeholder(value),
            "no-placeholders",
            f"placeholder or empty value is forbidden at {path}",
        )


def _mapping(
    payload: Mapping[str, Any], key: str, collector: _Collector
) -> Mapping[str, Any]:
    value = payload.get(key)
    collector.check(
        isinstance(value, Mapping),
        f"{key}-object",
        f"{key} must be a JSON object",
    )
    return value if isinstance(value, Mapping) else {}


def _required_strings(
    payload: Mapping[str, Any], fields: set[str], prefix: str, collector: _Collector
) -> None:
    for field in sorted(fields):
        value = payload.get(field)
        collector.check(
            isinstance(value, str) and not _is_placeholder(value),
            f"{prefix}-{field}",
            f"{prefix}.{field} must be a non-placeholder string",
        )


def _validate_data_governance(
    payload: Mapping[str, Any], collector: _Collector
) -> None:
    _required_strings(payload, DATA_STRING_FIELDS, "data_governance", collector)
    for field in sorted(DATA_BOOLEAN_FIELDS):
        collector.check(
            payload.get(field) is True,
            f"data-governance-{field}",
            f"data_governance.{field} must be true",
        )
    collector.check(
        payload.get("timezone") == "Asia/Shanghai",
        "data-governance-timezone",
        "data_governance.timezone must equal Asia/Shanghai",
    )
    collector.check(
        payload.get("bar_timestamp_semantics") in {"bar_end", "bar_start"},
        "data-governance-bar-timestamp",
        "bar timestamp semantics must equal bar_end or bar_start",
    )


def _parse_period(
    name: str, value: Any, collector: _Collector
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if not isinstance(value, Mapping):
        collector.check(False, f"period-{name}", f"period {name} must be an object")
        return None
    start_raw = value.get("start")
    end_raw = value.get("end")
    try:
        if not isinstance(start_raw, str) or not isinstance(end_raw, str):
            raise ValueError("period boundaries must be strings")
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
        valid = not pd.isna(start) and not pd.isna(end) and start <= end
    except (TypeError, ValueError):
        valid = False
        start = end = pd.Timestamp("1970-01-01")
    collector.check(
        valid,
        f"period-{name}",
        f"period {name} requires valid start <= end",
    )
    return (start, end) if valid else None


def _validate_evaluation(payload: Mapping[str, Any], collector: _Collector) -> None:
    periods = _mapping(payload, "periods", collector)
    names = ["train", "validation", "ordinary_test", "sealed_holdout"]
    parsed = [_parse_period(name, periods.get(name), collector) for name in names]
    ordered = all(
        left is not None and right is not None and left[1] < right[0]
        for left, right in zip(parsed, parsed[1:])
    )
    collector.check(
        ordered,
        "period-order",
        "train, validation, ordinary_test and sealed_holdout must be ordered and non-overlapping",
    )
    embargo = payload.get("embargo_days")
    collector.check(
        isinstance(embargo, int) and not isinstance(embargo, bool) and embargo >= 5,
        "embargo-days",
        "evaluation_protocol.embargo_days must be an integer >= 5",
    )
    collector.check(
        payload.get("purge_uses_label_end_at") is True,
        "purge-label-end",
        "purge_uses_label_end_at must be true",
    )
    collector.check(
        payload.get("sealed_holdout_accessed") is False,
        "sealed-holdout-unopened",
        "sealed_holdout_accessed must be false",
    )
    model_owner = payload.get("model_owner")
    custodian = payload.get("sealed_holdout_custodian")
    owners_valid = (
        isinstance(model_owner, str)
        and not _is_placeholder(model_owner)
        and isinstance(custodian, str)
        and not _is_placeholder(custodian)
        and model_owner != custodian
    )
    collector.check(
        owners_valid,
        "independent-holdout-custodian",
        "model_owner and sealed_holdout_custodian must be distinct non-placeholder strings",
    )
    models = payload.get("models")
    observed_models = (
        {item for item in models if isinstance(item, str)}
        if isinstance(models, list)
        else set()
    )
    collector.check(
        REQUIRED_MODELS.issubset(observed_models),
        "required-models",
        f"models must contain {sorted(REQUIRED_MODELS)}",
    )
    metrics = payload.get("metrics")
    observed_metrics = (
        {item for item in metrics if isinstance(item, str)}
        if isinstance(metrics, list)
        else set()
    )
    collector.check(
        REQUIRED_METRICS.issubset(observed_metrics),
        "required-metrics",
        f"metrics must contain {sorted(REQUIRED_METRICS)}",
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_compute(payload: Mapping[str, Any], collector: _Collector) -> None:
    _required_strings(
        payload,
        {"hardware_id", "device", "precision", "early_stopping_rule"},
        "compute_protocol",
        collector,
    )
    for field in ["batch_size", "max_epochs", "early_stopping_patience"]:
        collector.check(
            _positive_int(payload.get(field)),
            f"compute-{field}",
            f"compute_protocol.{field} must be a positive integer",
        )
    seed = payload.get("seed")
    collector.check(
        isinstance(seed, int) and not isinstance(seed, bool),
        "compute-seed",
        "compute_protocol.seed must be an integer",
    )
    collector.check(
        payload.get("deterministic_algorithms") is True,
        "compute-deterministic",
        "compute_protocol.deterministic_algorithms must be true",
    )


def _validate_budget(payload: Mapping[str, Any], collector: _Collector) -> None:
    collector.check(
        _positive_int(payload.get("max_pre_holdout_iterations")),
        "budget-iterations",
        "research_budget.max_pre_holdout_iterations must be a positive integer",
    )
    wall_clock = payload.get("max_wall_clock_hours")
    collector.check(
        isinstance(wall_clock, (int, float))
        and not isinstance(wall_clock, bool)
        and wall_clock > 0,
        "budget-wall-clock",
        "research_budget.max_wall_clock_hours must be positive",
    )
    _required_strings(
        payload,
        {"stop_rule", "model_selection_rule"},
        "research_budget",
        collector,
    )


def _resolve_relative(base: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or _is_placeholder(raw_path):
        return None
    path = Path(raw_path)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_runtime_manifest(
    descriptor: Mapping[str, Any],
    descriptor_path: Path,
    data_governance: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    collector: _Collector,
) -> str | None:
    manifest_path = _resolve_relative(
        descriptor_path.parent, descriptor.get("runtime_manifest_path")
    )
    collector.check(
        manifest_path is not None and manifest_path.is_file(),
        "runtime-manifest-exists",
        "runtime_manifest_path must reference an existing file",
    )
    if manifest_path is None or not manifest_path.is_file():
        return None
    try:
        manifest = _load_json(manifest_path, "runtime manifest")
    except ContractError as exc:
        collector.check(False, "runtime-manifest-json", str(exc))
        return None
    collector.check(
        manifest.get("schema_version") == 1,
        "runtime-manifest-schema",
        "runtime manifest schema_version must equal 1",
    )
    collector.check(
        manifest.get("dataset_kind") == "raw_1m",
        "runtime-manifest-kind",
        "real-data pilot runtime manifest dataset_kind must equal raw_1m",
    )
    collector.check(
        manifest.get("timezone") == "Asia/Shanghai",
        "runtime-manifest-timezone",
        "runtime manifest timezone must equal Asia/Shanghai",
    )
    collector.check(
        manifest.get("source_version") == data_governance.get("source_version"),
        "runtime-source-version",
        "runtime manifest and data governance source_version must match",
    )
    for field, expected in {
        "price_unit": "CNY",
        "volume_unit": "share",
        "amount_unit": "CNY",
    }.items():
        collector.check(
            manifest.get(field) == expected,
            f"runtime-{field}",
            f"runtime manifest {field} must equal {expected}",
        )
    checked_path_keys: set[str] = set()
    for path_key, hash_key in RUNTIME_FILES.items():
        checked_path_keys.add(path_key)
        data_path = _resolve_relative(manifest_path.parent, manifest.get(path_key))
        collector.check(
            data_path is not None and data_path.is_file(),
            f"runtime-file-{path_key}",
            f"runtime manifest {path_key} must reference an existing file",
        )
        if data_path is not None and data_path.is_file():
            collector.check(
                manifest.get(hash_key) == _sha256(data_path),
                f"runtime-hash-{path_key}",
                f"runtime manifest {hash_key} must match {path_key}",
            )
    extra_path_keys = sorted(
        key
        for key in manifest
        if isinstance(key, str)
        and key.endswith("_path")
        and key not in checked_path_keys
    )
    for path_key in extra_path_keys:
        hash_key = f"{path_key[:-5]}_sha256"
        data_path = _resolve_relative(manifest_path.parent, manifest.get(path_key))
        collector.check(
            data_path is not None and data_path.is_file(),
            f"runtime-file-{path_key}",
            f"runtime manifest {path_key} must reference an existing file",
        )
        if data_path is not None and data_path.is_file():
            collector.check(
                manifest.get(hash_key) == _sha256(data_path),
                f"runtime-hash-{path_key}",
                f"runtime manifest {hash_key} must match {path_key}",
            )
    promotion_path = _resolve_relative(
        descriptor_path.parent, evaluation.get("promotion_config_path")
    )
    collector.check(
        promotion_path is not None and promotion_path.is_file(),
        "promotion-config-exists",
        "promotion_config_path must reference an existing preregistration file",
    )
    if promotion_path is not None and promotion_path.is_file():
        collector.check(
            evaluation.get("promotion_config_sha256") == _sha256(promotion_path),
            "promotion-config-hash",
            "promotion_config_sha256 must match the preregistration file",
        )
    return _sha256(manifest_path)


def check_pilot_readiness(descriptor_path: str | Path) -> PilotReadinessReport:
    """Validate readiness without creating experiment outputs or opening holdouts."""

    path = Path(descriptor_path).resolve()
    descriptor = _load_json(path, "pilot readiness descriptor")
    collector = _Collector()
    collector.check(
        set(descriptor) == TOP_LEVEL_FIELDS,
        "descriptor-fields",
        f"descriptor fields must equal {sorted(TOP_LEVEL_FIELDS)}",
    )
    collector.check(
        descriptor.get("schema_version") == 1,
        "descriptor-schema",
        "descriptor schema_version must equal 1",
    )
    collector.check(
        descriptor.get("deliverable") == "engineering-research-library",
        "descriptor-deliverable",
        "deliverable must equal engineering-research-library",
    )
    _scan_for_unsafe_values(descriptor, collector)
    data_governance = _mapping(descriptor, "data_governance", collector)
    evaluation = _mapping(descriptor, "evaluation_protocol", collector)
    compute = _mapping(descriptor, "compute_protocol", collector)
    budget = _mapping(descriptor, "research_budget", collector)
    _validate_data_governance(data_governance, collector)
    _validate_evaluation(evaluation, collector)
    _validate_compute(compute, collector)
    _validate_budget(budget, collector)
    runtime_manifest_sha256 = _validate_runtime_manifest(
        descriptor,
        path,
        data_governance,
        evaluation,
        collector,
    )
    warnings = (
        "Readiness does not prove data correctness, alpha, capacity, speedup, candidate status, or release authorization.",
        "A passing descriptor authorizes no sealed-holdout access and no external write.",
    )
    return PilotReadinessReport(
        ready=not collector.errors,
        checks=tuple(collector.checks),
        errors=tuple(collector.errors),
        warnings=warnings,
        descriptor_sha256=_sha256(path),
        runtime_manifest_sha256=runtime_manifest_sha256,
    )
