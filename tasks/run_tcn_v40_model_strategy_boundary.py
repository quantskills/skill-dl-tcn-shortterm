"""Replay frozen v39 predictions through separate v40 model and portfolio gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm.backtest import build_executable_long_only  # noqa: E402
from skill_dl_tcn_shortterm.experiment import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.task_aligned_evaluation import (  # noqa: E402
    validate_prediction_contract,
)
from skill_dl_tcn_shortterm.v40_validation import (  # noqa: E402
    decide_v40_model_gate,
    decide_v40_strategy_gate,
    summarize_cost_sensitivity,
    validate_v40_frozen_predictions,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            any(
                marker in str(key).lower()
                for marker in ("password", "token", "secret", "credential")
            )
            or _contains_secret_key(nested)
            for key, nested in value.items()
        )
    return isinstance(value, list) and any(_contains_secret_key(item) for item in value)


def _resolve(path_value: object) -> Path:
    path = Path(str(path_value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _membership_diagnostics(orders: pd.DataFrame, *, policy: str) -> pd.DataFrame:
    selected = orders.loc[
        orders["portfolio_type"].astype(str).eq("executable_long_only")
    ].copy()
    rows: list[dict[str, object]] = []
    for keys, group in selected.groupby(
        ["model", "fold", "horizon"], observed=True, sort=True
    ):
        model, fold, horizon = cast(tuple[Any, Any, Any], keys)
        previous: set[str] | None = None
        for signal_date, date_group in group.groupby(
            "signal_date", observed=True, sort=True
        ):
            current = {
                str(instrument_id)
                for instrument_id in date_group["instrument_id"].tolist()
            }
            turnover = float("nan")
            if previous is not None:
                denominator = min(len(previous), len(current))
                turnover = 1.0 - len(previous & current) / denominator
            rows.append(
                {
                    "policy": policy,
                    "model": model,
                    "fold": int(fold),
                    "horizon": int(horizon),
                    "signal_date": signal_date,
                    "membership_turnover_diagnostic": turnover,
                    "selected_count": len(current),
                    "is_executable_turnover": False,
                }
            )
            previous = current
    return pd.DataFrame(rows)


def _jsonable_decision(decision: object) -> dict[str, object]:
    return {
        "status": cast(Any, decision).status,
        "admitted": cast(Any, decision).admitted,
        "blockers": list(cast(Any, decision).blockers),
        "evidence": cast(Any, decision).evidence,
        "sealed_test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the v40 frozen model/portfolio boundary replay"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v40 replay refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v40 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v40-phase-a":
            raise ContractError("v40 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v40 config contains a secret-like key")

        parent = _resolve(config["parent_run_dir"])
        labels_path = _resolve(config["labels_path"])
        source_paths = {
            "predictions": parent / "predictions.parquet",
            "leaderboard": parent / "tcn-leaderboard.parquet",
            "bootstrap": parent / "bootstrap-summary.parquet",
            "comparison": parent / "comparison.json",
            "selection": parent / "selection.json",
            "parent_receipt": parent / "receipt.json",
            "labels": labels_path,
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v40 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v40 source SHA-256 identity drifted")

        parent_receipt_value = json.loads(
            source_paths["parent_receipt"].read_text(encoding="utf-8")
        )
        if not isinstance(parent_receipt_value, dict) or parent_receipt_value.get(
            "sealed_test_accessed"
        ) is not False:
            raise ContractError("v40 parent receipt is not fail-closed")
        parent_outputs = parent_receipt_value.get("outputs")
        if not isinstance(parent_outputs, dict):
            raise ContractError("v40 parent output manifest is missing")
        for name, artifact_name in {
            "predictions": "predictions.parquet",
            "leaderboard": "tcn-leaderboard.parquet",
            "bootstrap": "bootstrap-summary.parquet",
            "comparison": "comparison.json",
            "selection": "selection.json",
        }.items():
            if parent_outputs.get(artifact_name) != observed_hashes[name]:
                raise ContractError(f"v40 parent receipt output drifted: {artifact_name}")

        predictions = pd.read_parquet(source_paths["predictions"])
        expected = cast(dict[str, list[object]], config["expected_panel"])
        validate_prediction_contract(predictions, expected_models=2)
        validate_v40_frozen_predictions(
            predictions,
            expected_models=tuple(str(value) for value in expected["models"]),
            expected_seeds=tuple(int(cast(Any, value)) for value in expected["seeds"]),
            expected_folds=tuple(int(cast(Any, value)) for value in expected["folds"]),
            expected_horizons=tuple(
                int(cast(Any, value)) for value in expected["horizons"]
            ),
        )
        labels = pd.read_parquet(source_paths["labels"])
        leaderboard = pd.read_parquet(source_paths["leaderboard"])
        bootstrap = pd.read_parquet(source_paths["bootstrap"])
        comparison_value = json.loads(
            source_paths["comparison"].read_text(encoding="utf-8")
        )
        if not isinstance(comparison_value, dict):
            raise ContractError("v40 comparison must contain an object")
        comparison = cast(dict[str, object], comparison_value)
        base_speed = float(
            leaderboard.loc[leaderboard["variant"].eq("base"), "samples_per_second"].median()
        )
        candidate_speed = float(
            leaderboard.loc[
                leaderboard["variant"].eq("relative"), "samples_per_second"
            ].median()
        )
        model_gate = decide_v40_model_gate(
            leaderboard,
            comparison,
            bootstrap,
            seeds=tuple(int(cast(Any, value)) for value in expected["seeds"]),
            folds=tuple(int(cast(Any, value)) for value in expected["folds"]),
            base_variant="base",
            candidate_variant="relative",
            base_median_samples_per_second=base_speed,
            candidate_median_samples_per_second=candidate_speed,
            gates=cast(dict[str, float | int], config["model_gates"]),
        )

        minimal_predictions = predictions[
            [
                "model",
                "fold",
                "sample_id",
                "instrument_id",
                "signal_date",
                "horizon",
                "score",
            ]
        ].copy()
        policies = cast(list[dict[str, object]], config["policies"])
        if not policies or {str(policy["name"]) for policy in policies} != {
            "raw_topk",
            "incumbent_buffer_20pct",
        }:
            raise ContractError("v40 policy identities drifted")
        all_orders: list[pd.DataFrame] = []
        all_ledgers: list[pd.DataFrame] = []
        all_holdings: list[pd.DataFrame] = []
        all_policy_summaries: list[pd.DataFrame] = []
        all_cost_summaries: list[pd.DataFrame] = []
        all_membership: list[pd.DataFrame] = []
        for policy in policies:
            policy_name = str(policy["name"])
            buffer_fraction = float(cast(Any, policy["incumbent_buffer_fraction"]))
            result = build_executable_long_only(
                minimal_predictions,
                labels,
                top_fraction=float(cast(Any, config["top_fraction"])),
                incumbent_buffer_fraction=buffer_fraction,
            )
            orders = result.orders.copy()
            orders["policy"] = policy_name
            ledger = result.portfolio_ledger.copy()
            ledger["policy"] = policy_name
            holdings = result.portfolio_holdings.copy()
            holdings["policy"] = policy_name
            policy_summary = result.metrics.copy()
            policy_summary["policy"] = policy_name
            costs = summarize_cost_sensitivity(
                result,
                cost_bps=tuple(
                    float(cast(Any, value))
                    for value in cast(list[object], config["cost_bps"])
                ),
            )
            costs["policy"] = policy_name
            all_orders.append(orders)
            all_ledgers.append(ledger)
            all_holdings.append(holdings)
            all_policy_summaries.append(policy_summary)
            all_cost_summaries.append(costs)
            all_membership.append(_membership_diagnostics(orders, policy=policy_name))

        orders = pd.concat(all_orders, ignore_index=True)
        ledgers = pd.concat(all_ledgers, ignore_index=True)
        holdings = pd.concat(all_holdings, ignore_index=True)
        policy_summary = pd.concat(all_policy_summaries, ignore_index=True)
        cost_summary = pd.concat(all_cost_summaries, ignore_index=True)
        membership = pd.concat(all_membership, ignore_index=True)
        strategy_config = cast(dict[str, object], config["strategy_gate"])
        strategy_gate = decide_v40_strategy_gate(
            cost_summary,
            policy="raw_topk",
            reference_model="base_tcn",
            candidate_model="relative_tcn",
            one_way_cost_bps=float(cast(Any, strategy_config["one_way_cost_bps"])),
            max_mean_one_way_turnover_delta=float(
                cast(Any, strategy_config["max_mean_one_way_turnover_delta"])
            ),
            min_mean_net_return_delta=float(
                cast(Any, strategy_config["min_mean_net_return_delta"])
            ),
        )

        temporary.mkdir(parents=True)
        model_gate_value = _jsonable_decision(model_gate)
        model_gate_value["phase_b_authorized"] = model_gate.admitted
        strategy_gate_value = _jsonable_decision(strategy_gate)
        _write_json(temporary / "model-gate.json", model_gate_value)
        _write_json(temporary / "strategy-gate.json", strategy_gate_value)
        model_gate.unit_deltas.to_parquet(
            temporary / "model-unit-deltas.parquet", index=False
        )
        strategy_gate.unit_deltas.to_parquet(
            temporary / "strategy-unit-deltas.parquet", index=False
        )
        orders.to_parquet(temporary / "orders.parquet", index=False)
        ledgers.to_parquet(temporary / "portfolio-ledger.parquet", index=False)
        holdings.to_parquet(temporary / "portfolio-holdings.parquet", index=False)
        policy_summary.to_parquet(temporary / "policy-summary.parquet", index=False)
        cost_summary.to_parquet(temporary / "cost-sensitivity.parquet", index=False)
        membership.to_parquet(
            temporary / "membership-diagnostics.parquet", index=False
        )
        _write_json(temporary / "config.resolved.json", config)
        report = "\n".join(
            [
                "# TCN v40 模型—组合责任边界校正结果",
                "",
                f"- 模型门：`{model_gate.status}`",
                f"- 组合研究门：`{strategy_gate.status}`",
                "- membership turnover：仅诊断，不进入模型门。",
                "- executable turnover：按证券目标权重变化净额计算。",
                "- transaction cost：按 buy + sell traded notional 计提。",
                f"- mean executable turnover delta：`{float(strategy_gate.evidence['mean_one_way_turnover_delta']):+.6f}`",
                f"- mean 10bps net-return delta：`{float(strategy_gate.evidence['mean_net_return_delta']):+.6f}`",
                f"- Phase B authorized：`{str(model_gate.admitted).lower()}`",
                "- sealed_test_accessed：`false`",
                "- 结论上限：ordinary-validation research evidence；not alpha-ready。",
                "",
            ]
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, object] = {
            "schema_version": "tcn-v40-model-strategy-boundary/v1",
            "run_id": str(config["run_id"]),
            "parent_receipt_id": parent_receipt_value.get("receipt_id"),
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "source_config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "model_gate": model_gate_value,
            "strategy_gate": strategy_gate_value,
            "outputs": outputs,
            "sealed_test_accessed": False,
            "sealed_test_authorized": False,
            "alpha_ready": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload = {
            "status": "success",
            "model_gate": model_gate.status,
            "strategy_gate": strategy_gate.status,
            "phase_b_authorized": model_gate.admitted,
            "output_dir": str(output_dir),
            "receipt_id": receipt["receipt_id"],
        }
    except (
        ContractError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
