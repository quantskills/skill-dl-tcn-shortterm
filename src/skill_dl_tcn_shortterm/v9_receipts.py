"""Canonical serialization and immutable atomic publication for v9 receipts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import ContractError


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"unsupported receipt value: {type(value).__name__}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def canonicalize(value: object) -> object:
    return json.loads(canonical_bytes(value).decode("utf-8"))


def publish_immutable_receipt(
    payload: dict[str, object],
    *,
    output_dir: Path,
    filename: str,
    identity_label: str,
) -> Path:
    """Replay identical content or atomically publish without overwriting drift."""

    destination = output_dir.resolve()
    receipt_path = destination / filename
    if destination.exists():
        if not receipt_path.is_file():
            raise ContractError(f"{identity_label} identity exists without a receipt")
        try:
            observed = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"{identity_label} existing receipt is unreadable") from exc
        if observed != payload:
            raise ContractError(f"{identity_label} identity drift detected; overwrite refused")
        return receipt_path
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise ContractError(f"{identity_label} temporary identity already exists")
    temporary.mkdir(parents=True)
    (temporary / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return receipt_path
