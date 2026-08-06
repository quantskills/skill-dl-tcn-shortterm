"""Flat, agent-neutral JSON interface for offline TCN research runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import tempfile
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from .experiment import ContractError, run_experiment

SCHEMA_VERSION = "1"
REQUEST_FIELDS = {
    "schema_version",
    "action",
    "config_path",
    "manifest_path",
    "output_root",
}
REQUIRED_REQUEST_FIELDS = REQUEST_FIELDS


def _engine_version() -> str:
    try:
        return importlib.metadata.version("skill-dl-tcn-shortterm")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _request_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must contain one JSON object")
    return payload


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"request.{key} must be a non-empty string")
    return value


def _validate_request(payload: Mapping[str, Any]) -> None:
    unknown = sorted(set(payload) - REQUEST_FIELDS)
    missing = sorted(REQUIRED_REQUEST_FIELDS - set(payload))
    if unknown:
        raise ContractError(f"request has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ContractError(f"request is missing fields: {', '.join(missing)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("request.schema_version must equal '1'")
    if payload.get("action") != "run":
        raise ContractError("request.action must equal 'run'")
    for key in ("config_path", "manifest_path", "output_root"):
        _require_string(payload, key)


def _resolve_path(request_file: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = request_file.parent / path
    return path.resolve()


def run_agent_request(request_path: str | Path) -> dict[str, Any]:
    """Execute one versioned request and return one portable result object."""

    request_file = Path(request_path).expanduser().resolve()
    payload = _read_json_object(request_file, label="request")
    _validate_request(payload)
    digest = _request_digest(payload)

    config_path = _resolve_path(request_file, _require_string(payload, "config_path"))
    manifest_path = _resolve_path(
        request_file, _require_string(payload, "manifest_path")
    )
    output_root = _resolve_path(request_file, _require_string(payload, "output_root"))
    if not config_path.is_file():
        raise ContractError("request config file does not exist")
    if not manifest_path.is_file():
        raise ContractError("request manifest file does not exist")

    config = _read_json_object(config_path, label="config")
    result = run_experiment(
        config=config,
        manifest_path=manifest_path,
        output_root=output_root,
    )
    run_manifest = _read_json_object(result.manifest_path, label="run manifest")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "action": "run",
        "request_digest": digest,
        "engine": {
            "name": "skill-dl-tcn-shortterm",
            "version": _engine_version(),
        },
        "run_id": result.run_id,
        "authoritative_run_manifest": run_manifest,
        "warnings": [
            "Research evidence is not Alpha, deployment, or trading authorization."
        ],
        "errors": [],
    }


def _write_example(output_dir: Path) -> list[str]:
    destination = output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ContractError("example output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)

    data_path = destination / "samples.parquet"
    pd.DataFrame(
        {
            "instrument_id": ["600000.XSHG", "000001.XSHE"],
            "signal_date": ["2024-01-02", "2024-01-02"],
            "target_1d": [-1.0, 1.0],
            "target_2d": [-1.0, 1.0],
            "target_3d": [-1.0, 1.0],
            "target_5d": [-1.0, 1.0],
        }
    ).to_parquet(data_path, index=False)
    manifest = {
        "schema_version": 1,
        "dataset_kind": "prebuilt_samples",
        "data_path": data_path.name,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    config = {
        "run_name": "tcn-shortterm-interface-example",
        "seed": 7,
        "horizons": [1, 2, 3, 5],
    }
    request = {
        "schema_version": SCHEMA_VERSION,
        "action": "run",
        "config_path": "config.json",
        "manifest_path": "manifest.json",
        "output_root": "runs",
    }
    files: dict[str, Mapping[str, Any]] = {
        "manifest.json": manifest,
        "config.json": config,
        "request.json": request,
    }
    for name, payload in files.items():
        (destination / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return ["samples.parquet", *files]


def _load_schema(kind: str) -> dict[str, Any]:
    filename = f"{kind}-v1.schema.json"
    schema_resource = resources.files("skill_dl_tcn_shortterm").joinpath(
        "resources", "agent_skill", filename
    )
    return json.loads(schema_resource.read_text(encoding="utf-8"))


def _failure(action: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "action": action,
        "engine": {
            "name": "skill-dl-tcn-shortterm",
            "version": _engine_version(),
        },
        "warnings": [],
        "errors": [str(error)],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcn-shortterm-skill",
        description="Agent-neutral JSON interface for offline TCN research",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run a temporary synthetic interface canary")
    example = subparsers.add_parser("example", help="write a minimal request bundle")
    example.add_argument("--output-dir", required=True, type=Path)
    run = subparsers.add_parser("run", help="execute one versioned request")
    run.add_argument("--request", required=True, type=Path)
    schema = subparsers.add_parser("schema", help="print a machine-readable schema")
    schema.add_argument("--kind", choices=("request", "result"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    action = str(arguments.command)
    try:
        if arguments.command == "schema":
            payload = _load_schema(str(arguments.kind))
        elif arguments.command == "example":
            files = _write_example(arguments.output_dir)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "success",
                "action": "example",
                "engine": {
                    "name": "skill-dl-tcn-shortterm",
                    "version": _engine_version(),
                },
                "files": files,
                "warnings": [
                    "The generated data only validates the interface; replace it for TCN research."
                ],
                "errors": [],
            }
        elif arguments.command == "run":
            payload = run_agent_request(arguments.request)
        else:
            with tempfile.TemporaryDirectory(prefix="tcn-shortterm-skill-") as temp:
                root = Path(temp)
                _write_example(root)
                payload = run_agent_request(root / "request.json")
                payload["action"] = "demo"
                payload["warnings"] = [
                    "Synthetic demo passed; it does not train or validate a TCN model."
                ]
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps(_failure(action, exc), ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
