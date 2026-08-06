"""Pre-registered, once-only sealed-test promotion state machine."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment import ContractError


IDENTITY_FIELDS = {"data_sha256", "code_revision", "model_fingerprint"}
THRESHOLD_FIELDS = {"rankic_min", "icir_min", "net_return_min", "speedup_min"}
RESULT_FIELDS = {"rankic", "icir", "net_return", "speedup"}


@dataclass(frozen=True)
class FrozenPromotion:
    promotion_id: str
    frozen_path: Path
    state_path: Path


@dataclass(frozen=True)
class PromotionReceipt:
    promotion_id: str
    receipt_path: Path
    candidate_model: bool


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(_canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{description} must be a JSON object")
    return value


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "candidate_model",
        "baseline_model",
        "validation_run_id",
        *IDENTITY_FIELDS,
        "sealed_data_id",
        "sealed_data_sha256",
        "thresholds",
    }
    if missing := required - set(config):
        raise ContractError(f"promotion config missing fields: {sorted(missing)}")
    payload = dict(config)
    for field in required - {"thresholds"}:
        if not isinstance(payload[field], str) or not payload[field]:
            raise ContractError(f"promotion config {field} must be a non-empty string")
    thresholds = payload["thresholds"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != THRESHOLD_FIELDS:
        raise ContractError(
            f"promotion thresholds must equal {sorted(THRESHOLD_FIELDS)}"
        )
    for field, value in thresholds.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ContractError(f"promotion threshold {field} must be finite")
    return payload


def freeze_promotion(
    registry_dir: str | Path, config: Mapping[str, Any]
) -> FrozenPromotion:
    """Freeze thresholds and identities before sealed data can be opened."""

    payload = _validate_config(config)
    payload_json = _canonical_json(payload)
    payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    promotion_id = payload_sha256[:16]
    root = Path(registry_dir).resolve()
    promotion_dir = root / "promotions" / promotion_id
    try:
        promotion_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ContractError(f"promotion is already frozen: {promotion_id}") from exc
    frozen_at = _now()
    frozen_path = promotion_dir / "frozen.json"
    state_path = promotion_dir / "state.json"
    _write_json(
        frozen_path,
        {
            "schema_version": 1,
            "promotion_id": promotion_id,
            "frozen_at": frozen_at,
            "payload_sha256": payload_sha256,
            "payload": payload,
        },
    )
    _write_json(
        state_path,
        {
            "schema_version": 1,
            "promotion_id": promotion_id,
            "status": "frozen",
            "attempt": 0,
        },
    )
    return FrozenPromotion(
        promotion_id=promotion_id, frozen_path=frozen_path, state_path=state_path
    )


def _load_frozen(root: Path, promotion_id: str) -> tuple[dict[str, Any], Path, Path]:
    promotion_dir = root / "promotions" / promotion_id
    frozen_path = promotion_dir / "frozen.json"
    state_path = promotion_dir / "state.json"
    if not frozen_path.is_file() or not state_path.is_file():
        raise ContractError(f"frozen promotion does not exist: {promotion_id}")
    envelope = _read_json(frozen_path, "frozen promotion")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ContractError("frozen promotion payload is invalid")
    observed = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if observed != envelope.get("payload_sha256") or promotion_id != observed[:16]:
        raise ContractError("frozen promotion fingerprint mismatch")
    _validate_config(payload)
    return envelope, state_path, promotion_dir


def evaluate_sealed_once(
    registry_dir: str | Path,
    *,
    promotion_id: str,
    current_identity: Mapping[str, str],
    sealed_data_id: str,
    sealed_data_sha256: str,
    evaluator: Callable[[], Mapping[str, float]],
) -> PromotionReceipt:
    """Open the sealed evaluator only after all immutable gates pass."""

    root = Path(registry_dir).resolve()
    envelope, state_path, promotion_dir = _load_frozen(root, promotion_id)
    payload = envelope["payload"]
    for field in IDENTITY_FIELDS:
        if current_identity.get(field) != payload[field]:
            raise ContractError(f"promotion identity mismatch: {field}")
    if sealed_data_id != payload["sealed_data_id"]:
        raise ContractError("promotion sealed data identity mismatch")
    if sealed_data_sha256 != payload["sealed_data_sha256"]:
        raise ContractError("promotion sealed data fingerprint mismatch")
    consumption_key = hashlib.sha256(
        f"{sealed_data_id}|{sealed_data_sha256}".encode("utf-8")
    ).hexdigest()
    consumed_dir = root / "consumed"
    consumed_path = consumed_dir / f"{consumption_key}.json"
    if consumed_path.exists():
        raise ContractError("sealed data has already been consumed")
    state = _read_json(state_path, "promotion state")
    if state.get("status") == "completed":
        raise ContractError("sealed data has already been consumed")
    if state.get("status") == "running":
        raise ContractError("sealed evaluation is already running")
    lock_dir = root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{consumption_key}.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as stream:
            stream.write(
                _canonical_json(
                    {
                        "promotion_id": promotion_id,
                        "sealed_data_id": sealed_data_id,
                        "acquired_at": _now(),
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ContractError("sealed evaluation is already running") from exc
    attempt = int(state.get("attempt", 0)) + 1
    started_at = _now()
    _write_json(
        state_path,
        {
            "schema_version": 1,
            "promotion_id": promotion_id,
            "status": "running",
            "attempt": attempt,
            "started_at": started_at,
        },
    )
    try:
        raw_results = evaluator()
        if not isinstance(raw_results, Mapping) or set(raw_results) != RESULT_FIELDS:
            raise ContractError(f"sealed results must equal {sorted(RESULT_FIELDS)}")
        results = {name: float(raw_results[name]) for name in RESULT_FIELDS}
        if not all(math.isfinite(value) for value in results.values()):
            raise ContractError("sealed results must be finite")
    except Exception as exc:
        _write_json(
            state_path,
            {
                "schema_version": 1,
                "promotion_id": promotion_id,
                "status": "failed",
                "attempt": attempt,
                "started_at": started_at,
                "ended_at": _now(),
                "error_type": type(exc).__name__,
            },
        )
        lock_path.unlink(missing_ok=True)
        raise

    thresholds = payload["thresholds"]
    outcomes = {
        "rankic": results["rankic"] >= thresholds["rankic_min"],
        "icir": results["icir"] >= thresholds["icir_min"],
        "net_return": results["net_return"] >= thresholds["net_return_min"],
        "speedup": results["speedup"] >= thresholds["speedup_min"],
    }
    is_candidate = all(outcomes.values())
    ended_at = _now()
    receipt_value = {
        "schema_version": 1,
        "promotion_id": promotion_id,
        "status": "completed",
        "attempt": attempt,
        "frozen_payload_sha256": envelope["payload_sha256"],
        "started_at": started_at,
        "ended_at": ended_at,
        "run_identity": dict(current_identity),
        "sealed_data_id": sealed_data_id,
        "sealed_data_sha256": sealed_data_sha256,
        "candidate_model_name": payload["candidate_model"],
        "baseline_model": payload["baseline_model"],
        "thresholds": thresholds,
        "results": results,
        "threshold_outcomes": outcomes,
        "candidate_model": is_candidate,
        "engineering_complete": True,
    }
    receipt_path = promotion_dir / "receipt.json"
    _write_json(receipt_path, receipt_value)
    _write_json(
        state_path,
        {
            "schema_version": 1,
            "promotion_id": promotion_id,
            "status": "completed",
            "attempt": attempt,
            "started_at": started_at,
            "ended_at": ended_at,
            "receipt": receipt_path.name,
        },
    )
    consumed_dir.mkdir(parents=True, exist_ok=True)
    try:
        with consumed_path.open("x", encoding="utf-8") as stream:
            stream.write(
                _canonical_json(
                    {
                        "promotion_id": promotion_id,
                        "receipt": receipt_path.relative_to(root).as_posix(),
                        "consumed_at": ended_at,
                    }
                )
                + "\n"
            )
    except FileExistsError as exc:
        raise ContractError("sealed data has already been consumed") from exc
    lock_path.unlink(missing_ok=True)
    return PromotionReceipt(
        promotion_id=promotion_id,
        receipt_path=receipt_path,
        candidate_model=is_candidate,
    )


def verify_promotion_receipt(receipt_path: str | Path) -> dict[str, Any]:
    """Reject partial receipts so failed writes cannot look conclusive."""

    receipt = _read_json(Path(receipt_path).resolve(), "promotion receipt")
    required = {
        "schema_version",
        "promotion_id",
        "status",
        "attempt",
        "frozen_payload_sha256",
        "started_at",
        "ended_at",
        "run_identity",
        "sealed_data_id",
        "sealed_data_sha256",
        "candidate_model_name",
        "baseline_model",
        "thresholds",
        "results",
        "threshold_outcomes",
        "candidate_model",
        "engineering_complete",
    }
    if missing := required - set(receipt):
        raise ContractError(f"receipt missing fields: {sorted(missing)}")
    if receipt["schema_version"] != 1 or receipt["status"] != "completed":
        raise ContractError("promotion receipt is not completed schema version 1")
    if set(receipt["results"]) != RESULT_FIELDS:
        raise ContractError("promotion receipt has incomplete results")
    if set(receipt["thresholds"]) != THRESHOLD_FIELDS:
        raise ContractError("promotion receipt has incomplete thresholds")
    return receipt
