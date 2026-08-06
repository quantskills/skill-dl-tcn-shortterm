from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from skill_dl_tcn_shortterm import check_pilot_readiness


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path) -> tuple[Path, dict[str, Any]]:
    files = {}
    for name in ["bars", "states", "actions", "execution"]:
        path = root / f"{name}.parquet"
        path.write_bytes(f"fixture:{name}".encode())
        files[name] = path
    promotion_path = root / "promotion.json"
    promotion_path.write_text(
        json.dumps({"status": "preregistered", "sealed_access": False}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "dataset_kind": "raw_1m",
        "timezone": "Asia/Shanghai",
        "price_unit": "CNY",
        "volume_unit": "share",
        "amount_unit": "CNY",
        "source_version": "provider-snapshot-2026-01",
        "data_path": files["bars"].name,
        "data_sha256": _sha256(files["bars"]),
        "instrument_state_path": files["states"].name,
        "instrument_state_sha256": _sha256(files["states"]),
        "corporate_action_path": files["actions"].name,
        "corporate_action_sha256": _sha256(files["actions"]),
        "execution_state_path": files["execution"].name,
        "execution_state_sha256": _sha256(files["execution"]),
    }
    manifest_path = root / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    descriptor = {
        "schema_version": 1,
        "deliverable": "engineering-research-library",
        "runtime_manifest_path": manifest_path.name,
        "data_governance": {
            "provider": "licensed-provider",
            "license_reference": "owner-approval-2026-01",
            "source_version": "provider-snapshot-2026-01",
            "data_owner": "research-data-owner",
            "timezone": "Asia/Shanghai",
            "trading_calendar": "XSHG-XSHE-v1",
            "bar_timestamp_semantics": "bar_end",
            "adjustment_policy": "point-in-time-forward-adjustment-v1",
            "availability_policy": "vendor-published-at-cutoff-v1",
            "raw_schema_version": "raw-1m-v1",
            "canonical_schema_version": "canonical-5m-v1",
            "universe_version": "pit-a-share-v1",
            "feature_version": "features-v1",
            "label_version": "next-open-rank-v1",
            "license_approved": True,
            "pit_instrument_state": True,
            "pit_corporate_actions": True,
            "survivorship_bias_controlled": True,
        },
        "evaluation_protocol": {
            "periods": {
                "train": {"start": "2018-01-01", "end": "2021-12-31"},
                "validation": {"start": "2022-01-01", "end": "2022-12-31"},
                "ordinary_test": {"start": "2023-01-01", "end": "2023-12-31"},
                "sealed_holdout": {"start": "2024-01-01", "end": "2024-12-31"},
            },
            "embargo_days": 5,
            "purge_uses_label_end_at": True,
            "sealed_holdout_accessed": False,
            "model_owner": "model-researcher",
            "sealed_holdout_custodian": "independent-custodian",
            "models": ["ridge", "lightgbm", "lstm", "gru", "bai-tcn"],
            "metrics": [
                "rankic",
                "icir",
                "net_long_only_return",
                "throughput_samples_per_second",
                "time_to_best_validation_seconds",
                "peak_memory_bytes",
            ],
            "promotion_config_path": promotion_path.name,
            "promotion_config_sha256": _sha256(promotion_path),
        },
        "compute_protocol": {
            "hardware_id": "research-node-a",
            "device": "cpu",
            "precision": "float32",
            "batch_size": 32,
            "max_epochs": 20,
            "early_stopping_patience": 3,
            "early_stopping_rule": "validation-rankic-no-improvement",
            "seed": 7,
            "deterministic_algorithms": True,
        },
        "research_budget": {
            "max_pre_holdout_iterations": 10,
            "max_wall_clock_hours": 24,
            "stop_rule": "stop-at-budget-or-no-validation-improvement",
            "model_selection_rule": "highest-validation-rankic-subject-to-net-return",
        },
    }
    descriptor_path = root / "pilot-readiness.json"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    return descriptor_path, descriptor


def _rewrite(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_complete_descriptor_passes_without_creating_outputs(tmp_path: Path) -> None:
    descriptor_path, _ = _write_bundle(tmp_path)
    before = {path.name for path in tmp_path.iterdir()}

    report = check_pilot_readiness(descriptor_path)

    assert report.ready
    assert not report.errors
    assert len(report.descriptor_sha256) == 64
    assert report.runtime_manifest_sha256 is not None
    assert {path.name for path in tmp_path.iterdir()} == before


def test_missing_field_and_placeholder_fail_closed(tmp_path: Path) -> None:
    descriptor_path, descriptor = _write_bundle(tmp_path)
    descriptor["data_governance"].pop("provider")
    descriptor["data_governance"]["license_reference"] = "<approval-id>"
    _rewrite(descriptor_path, descriptor)

    report = check_pilot_readiness(descriptor_path)

    assert not report.ready
    assert any("data_governance-provider" in error for error in report.errors)
    assert any("no-placeholders" in error for error in report.errors)


def test_secret_like_key_is_rejected_at_any_depth(tmp_path: Path) -> None:
    descriptor_path, descriptor = _write_bundle(tmp_path)
    descriptor["data_governance"]["nested"] = {"api_token": "not-a-real-value"}
    _rewrite(descriptor_path, descriptor)

    report = check_pilot_readiness(descriptor_path)

    assert not report.ready
    assert any("no-secret-keys" in error for error in report.errors)


@pytest.mark.parametrize("file_kind", ["required", "additional"])
def test_every_runtime_fingerprint_mismatch_fails(
    tmp_path: Path, file_kind: str
) -> None:
    descriptor_path, _ = _write_bundle(tmp_path)
    if file_kind == "required":
        (tmp_path / "bars.parquet").write_bytes(b"changed")
        expected = "runtime-hash-data_path"
    else:
        extra_path = tmp_path / "extra.parquet"
        extra_path.write_bytes(b"extra")
        manifest_path = tmp_path / "runtime-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["auxiliary_path"] = extra_path.name
        manifest["auxiliary_sha256"] = "wrong"
        _rewrite(manifest_path, manifest)
        expected = "runtime-hash-auxiliary_path"

    report = check_pilot_readiness(descriptor_path)

    assert not report.ready
    assert any(expected in error for error in report.errors)


@pytest.mark.parametrize("failure", ["overlap", "embargo"])
def test_split_overlap_and_short_embargo_fail(
    tmp_path: Path, failure: str
) -> None:
    descriptor_path, descriptor = _write_bundle(tmp_path)
    if failure == "overlap":
        descriptor["evaluation_protocol"]["periods"]["validation"]["start"] = (
            "2021-12-01"
        )
    else:
        descriptor["evaluation_protocol"]["embargo_days"] = 4
    _rewrite(descriptor_path, descriptor)

    report = check_pilot_readiness(descriptor_path)

    assert not report.ready
    expected = "period-order" if failure == "overlap" else "embargo-days"
    assert any(expected in error for error in report.errors)


def test_opened_holdout_and_same_custodian_fail(tmp_path: Path) -> None:
    descriptor_path, descriptor = _write_bundle(tmp_path)
    protocol = descriptor["evaluation_protocol"]
    protocol["sealed_holdout_accessed"] = True
    protocol["sealed_holdout_custodian"] = protocol["model_owner"]
    _rewrite(descriptor_path, descriptor)

    report = check_pilot_readiness(descriptor_path)

    assert not report.ready
    assert any("sealed-holdout-unopened" in error for error in report.errors)
    assert any("independent-holdout-custodian" in error for error in report.errors)


def test_required_models_and_metrics_cannot_be_omitted(tmp_path: Path) -> None:
    descriptor_path, descriptor = _write_bundle(tmp_path)
    descriptor["evaluation_protocol"]["models"].remove("lightgbm")
    descriptor["evaluation_protocol"]["metrics"].remove("peak_memory_bytes")
    _rewrite(descriptor_path, descriptor)

    report = check_pilot_readiness(descriptor_path)

    assert not report.ready
    assert any("required-models" in error for error in report.errors)
    assert any("required-metrics" in error for error in report.errors)


def test_cli_returns_json_and_zero_or_two(tmp_path: Path) -> None:
    descriptor_path, descriptor = _write_bundle(tmp_path)
    command = [
        sys.executable,
        "tasks/check_pilot_readiness.py",
        "--descriptor",
        str(descriptor_path),
    ]

    ready = subprocess.run(command, capture_output=True, text=True, check=False)
    assert ready.returncode == 0
    assert json.loads(ready.stdout)["ready"] is True

    incomplete = copy.deepcopy(descriptor)
    incomplete["evaluation_protocol"]["sealed_holdout_accessed"] = True
    _rewrite(descriptor_path, incomplete)
    not_ready = subprocess.run(command, capture_output=True, text=True, check=False)
    assert not_ready.returncode == 2
    assert json.loads(not_ready.stdout)["ready"] is False


def test_cli_reports_missing_descriptor_as_json(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tasks/check_pilot_readiness.py",
            "--descriptor",
            str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["ready"] is False
    assert payload["checks"] == []
