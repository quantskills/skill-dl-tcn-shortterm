"""Apply the pre-registered TCN CPU speed and validation-effect gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.stability import evaluate_tcn_stability_gate  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the TCN stability gate")
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-model", default="tcn-lite")
    parser.add_argument("--recurrent-baseline", default="lstm")
    parser.add_argument("--tcn-control", default="bai-tcn")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("TCN stability gate refuses to overwrite artifacts")
        metrics_path = arguments.metrics.resolve()
        if not metrics_path.is_file():
            raise ContractError("TCN stability metrics are missing")
        decision = evaluate_tcn_stability_gate(
            pd.read_parquet(metrics_path),
            candidate_model=arguments.candidate_model,
            recurrent_baseline=arguments.recurrent_baseline,
            tcn_control=arguments.tcn_control,
            model_step_speedup_min=1.5,
            end_to_end_speedup_min=1.2,
            positive_rate_min=0.6,
            control_median_improvement_min=0.005,
            worst_fold_min=-0.01,
        )
        temporary.mkdir(parents=True)
        decision_path = temporary / "decision.json"
        decision_path.write_text(
            json.dumps(asdict(decision), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt = {
            "schema_version": 1,
            "source_metrics": {
                "path": str(metrics_path),
                "sha256": _sha256(metrics_path),
            },
            "thresholds": {
                "model_step_speedup_min": 1.5,
                "end_to_end_speedup_min": 1.2,
                "positive_rate_min": 0.6,
                "control_median_improvement_min": 0.005,
                "worst_fold_min": -0.01,
                "specific_three_x_model_step_min": 3.0,
            },
            "decision": asdict(decision),
            "outputs": {decision_path.name: _sha256(decision_path)},
            "sealed_test_accessed": False,
        }
        (temporary / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_dir)
        payload = {"status": "success", "decision": asdict(decision)}
    except (OSError, ValueError, KeyError, TypeError, ContractError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
