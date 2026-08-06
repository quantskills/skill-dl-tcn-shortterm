from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from skill_dl_tcn_shortterm.experiment import ContractError
from skill_dl_tcn_shortterm.promotion import (
    evaluate_sealed_once,
    freeze_promotion,
    verify_promotion_receipt,
)


def _config() -> dict[str, object]:
    return {
        "candidate_model": "bai-tcn",
        "baseline_model": "lightgbm",
        "validation_run_id": "run-validation",
        "data_sha256": "data-v1",
        "code_revision": "code-v1",
        "model_fingerprint": "model-v1",
        "sealed_data_id": "sealed-2024",
        "sealed_data_sha256": "sealed-sha-v1",
        "thresholds": {
            "rankic_min": 0.05,
            "icir_min": 0.5,
            "net_return_min": 0.01,
            "speedup_min": 3.0,
        },
    }


def _identity() -> dict[str, str]:
    return {
        "data_sha256": "data-v1",
        "code_revision": "code-v1",
        "model_fingerprint": "model-v1",
    }


def test_sealed_loader_is_not_called_before_freeze_or_on_identity_mismatch(
    tmp_path: Path,
) -> None:
    calls = []

    def loader() -> dict[str, float]:
        calls.append("opened")
        return {"rankic": 0.1, "icir": 1.0, "net_return": 0.02, "speedup": 3.2}

    with pytest.raises(ContractError, match="frozen promotion"):
        evaluate_sealed_once(
            tmp_path,
            promotion_id="missing",
            current_identity=_identity(),
            sealed_data_id="sealed-2024",
            sealed_data_sha256="sealed-sha-v1",
            evaluator=loader,
        )
    assert calls == []

    frozen = freeze_promotion(tmp_path, _config())
    with pytest.raises(ContractError, match="identity mismatch"):
        evaluate_sealed_once(
            tmp_path,
            promotion_id=frozen.promotion_id,
            current_identity={**_identity(), "model_fingerprint": "different"},
            sealed_data_id="sealed-2024",
            sealed_data_sha256="sealed-sha-v1",
            evaluator=loader,
        )
    assert calls == []


def test_failed_attempt_can_retry_but_successful_sealed_consumption_is_once_only(
    tmp_path: Path,
) -> None:
    frozen = freeze_promotion(tmp_path, _config())

    def failure() -> dict[str, float]:
        raise RuntimeError("offline evaluator failed")

    with pytest.raises(RuntimeError, match="offline evaluator failed"):
        evaluate_sealed_once(
            tmp_path,
            promotion_id=frozen.promotion_id,
            current_identity=_identity(),
            sealed_data_id="sealed-2024",
            sealed_data_sha256="sealed-sha-v1",
            evaluator=failure,
        )
    state = json.loads(frozen.state_path.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["attempt"] == 1

    receipt = evaluate_sealed_once(
        tmp_path,
        promotion_id=frozen.promotion_id,
        current_identity=_identity(),
        sealed_data_id="sealed-2024",
        sealed_data_sha256="sealed-sha-v1",
        evaluator=lambda: {
            "rankic": 0.04,
            "icir": 0.6,
            "net_return": 0.02,
            "speedup": 3.5,
        },
    )
    verified = verify_promotion_receipt(receipt.receipt_path)
    assert verified["candidate_model"] is False
    assert verified["engineering_complete"] is True
    assert verified["attempt"] == 2

    with pytest.raises(ContractError, match="already been consumed"):
        evaluate_sealed_once(
            tmp_path,
            promotion_id=frozen.promotion_id,
            current_identity=_identity(),
            sealed_data_id="sealed-2024",
            sealed_data_sha256="sealed-sha-v1",
            evaluator=lambda: {
                "rankic": 1.0,
                "icir": 1.0,
                "net_return": 1.0,
                "speedup": 10.0,
            },
        )


def test_incomplete_receipt_is_rejected(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    with pytest.raises(ContractError, match="receipt missing fields"):
        verify_promotion_receipt(receipt)


def test_concurrent_sealed_evaluation_is_locked_before_loader_opens(
    tmp_path: Path,
) -> None:
    frozen = freeze_promotion(tmp_path, _config())
    started = threading.Event()
    release = threading.Event()

    def slow_loader() -> dict[str, float]:
        started.set()
        assert release.wait(timeout=5)
        return {"rankic": 0.1, "icir": 1.0, "net_return": 0.02, "speedup": 3.2}

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            evaluate_sealed_once,
            tmp_path,
            promotion_id=frozen.promotion_id,
            current_identity=_identity(),
            sealed_data_id="sealed-2024",
            sealed_data_sha256="sealed-sha-v1",
            evaluator=slow_loader,
        )
        assert started.wait(timeout=5)
        with pytest.raises(ContractError, match="already running"):
            evaluate_sealed_once(
                tmp_path,
                promotion_id=frozen.promotion_id,
                current_identity=_identity(),
                sealed_data_id="sealed-2024",
                sealed_data_sha256="sealed-sha-v1",
                evaluator=lambda: {
                    "rankic": 1.0,
                    "icir": 1.0,
                    "net_return": 1.0,
                    "speedup": 10.0,
                },
            )
        release.set()
        receipt = first.result(timeout=5)

    assert verify_promotion_receipt(receipt.receipt_path)["status"] == "completed"
