"""Run the immutable v17 seed-7 PIT market-conditioned TCN screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.market_context import (  # noqa: E402
    build_pit_market_context,
    fit_market_context_standardizer,
)
from skill_dl_tcn_shortterm.performance import benchmark_sequence_models  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    build_tcn_lstm_comparison,
    evaluate_pit_market_conditioning_seed7,
    finalize_pit_market_conditioning_seed7,
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.training_data import build_fold_protocols  # noqa: E402
from skill_dl_tcn_shortterm.tuning import run_tcn_validation_sweep  # noqa: E402
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            any(
                marker in str(key).lower()
                for marker in ["password", "token", "secret", "credential"]
            )
            or _contains_secret_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else ROOT / path).resolve()


def _validate_parent(config: dict[str, object]) -> dict[str, object]:
    parent = _resolve_project_path(config["parent_artifact"])
    receipt_path = parent / "receipt.json"
    selection_path = parent / "selection.json"
    if not receipt_path.is_file() or not selection_path.is_file():
        raise ContractError("v17 parent artifact is incomplete")
    receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
    selection_value = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(receipt_value, dict) or not isinstance(selection_value, dict):
        raise ContractError("v17 parent evidence must contain objects")
    receipt = cast(dict[str, object], receipt_value)
    selection = cast(dict[str, object], selection_value)
    expected_receipt_id = str(config["parent_receipt_id"])
    if receipt.get("receipt_id") != expected_receipt_id:
        raise ContractError("v17 parent receipt identity drifted")
    if receipt.get("sealed_test_accessed") is not False:
        raise ContractError("v17 parent must not have accessed sealed test")
    if selection.get("status") != "stop_decoupled_residual_seed7_effect_v16":
        raise ContractError("v17 parent status drifted")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ContractError("v17 parent receipt outputs are missing")
    for relative, expected_hash in outputs.items():
        path = parent / str(relative)
        if not path.is_file() or _sha256(path) != str(expected_hash):
            raise ContractError("v17 parent output hash drifted")
    return {
        "path": str(parent),
        "receipt_id": expected_receipt_id,
        "selection_status": str(selection["status"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded seed-7 PIT market-conditioned TCN validation"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError(
                "v17 market-conditioning screen refuses to overwrite artifacts"
            )
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v17 market-conditioning config must contain an object")
        config = cast(dict[str, object], config_value)
        if _contains_secret_key(config):
            raise ContractError(
                "v17 market-conditioning config contains a secret-like key"
            )
        if config.get("precision") != "float32" or int(cast(Any, config["seed"])) != 7:
            raise ContractError(
                "v17 market-conditioning screen requires seed 7 float32"
            )
        parent = _validate_parent(config)

        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "labels": run_dir / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "universe": run_dir / "universe.parquet",
            "input_manifest": run_dir / "input-manifest.json",
        }
        if missing := [
            name for name, path in source_paths.items() if not path.is_file()
        ]:
            raise ContractError(
                f"v17 market-conditioning sources missing: {', '.join(missing)}"
            )
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError(
                "v17 market-conditioning source SHA-256 identity drifted"
            )

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        universe = pd.read_parquet(source_paths["universe"])
        input_manifest_value = json.loads(
            source_paths["input_manifest"].read_text(encoding="utf-8")
        )
        if not isinstance(input_manifest_value, dict):
            raise ContractError("v17 input manifest must contain an object")
        enrichment = input_manifest_value.get("enrichment")
        if (
            "industry" not in universe
            or set(universe["industry"].astype(str)) != {"unavailable"}
            or not isinstance(enrichment, dict)
            or enrichment.get("industry_history") != "unavailable"
        ):
            raise ContractError("v17 industry-history availability drifted")
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError(
                "v17 market-conditioning screen rejects sealed split rows"
            )
        allowed_stages = {"train", "validation", "purged"}
        observed_stages = {
            str(value) for value in raw_split["stage"].astype(str).tolist()
        }
        if unknown := sorted(observed_stages - allowed_stages):
            raise ContractError(
                "v17 market-conditioning split contains forbidden stages: "
                + ", ".join(unknown)
            )
        folds = tuple(
            int(cast(Any, value)) for value in cast(list[object], config["folds"])
        )
        if folds != tuple(range(5)):
            raise ContractError(
                "v17 market-conditioning folds must be exactly 0 through 4"
            )
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(folds)
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        context_config = cast(dict[str, object], config["market_context"])
        feature_indices = tuple(
            int(cast(Any, value))
            for value in cast(list[object], context_config["feature_indices"])
        )
        if (
            feature_indices != tuple(range(6))
            or int(cast(Any, context_config["bars_per_day"])) != 48
            or str(context_config["industry_context_status"])
            != "blocked_historical_industry_unavailable"
        ):
            raise ContractError("v17 market context construction contract drifted")
        allowed_positions = np.sort(
            raw_split["sample_position"].drop_duplicates().to_numpy(dtype="int64")
        )
        market_context = build_pit_market_context(
            features,
            window_index,
            allowed_positions=allowed_positions,
            feature_indices=feature_indices,
            bars_per_day=48,
        )
        if market_context.values.shape[1] != 24:
            raise ContractError("v17 market context dimension drifted")

        trials = parse_real_tcn_trials(config["trials"])
        control_trial_id = str(config["control_trial_id"])
        candidate_trial_id = str(config["candidate_trial_id"])
        if {trial.trial_id for trial in trials} != {
            control_trial_id,
            candidate_trial_id,
        } or len(trials) != 2:
            raise ContractError("v17 market-conditioning trial identities drifted")
        by_trial = {trial.trial_id: trial for trial in trials}
        candidate = by_trial[candidate_trial_id]
        if by_trial[control_trial_id].model_kind != "temporal_context" or (
            candidate.model_kind != "market_conditioned_temporal_context"
            or candidate.market_context_dim != 24
            or candidate.market_context_hidden != 4
            or candidate.market_gate_scale != 0.25
        ):
            raise ContractError("v17 market-conditioning mechanism drifted")

        protocol_identities = {
            "data": observed_hashes["features"],
            "fold_manifest": observed_hashes["split_manifest"],
            "evaluation": observed_hashes["labels"],
            "market_context": market_context.identity,
        }
        tuning = run_tcn_validation_sweep(
            features,
            window_index,
            labels,
            split_manifest,
            trials=trials,
            seed=7,
            max_epochs=int(cast(Any, config["max_epochs"])),
            patience=int(cast(Any, config["patience"])),
            min_delta=float(cast(Any, config["min_delta"])),
            torch_threads=int(cast(Any, config["torch_threads"])),
            protocol_identities=protocol_identities,
            market_context=market_context,
        )
        gates = cast(dict[str, object], config["gates"])
        effect = evaluate_pit_market_conditioning_seed7(
            tuning.leaderboard,
            control_trial_id=control_trial_id,
            candidate_trial_id=candidate_trial_id,
            min_mean_rankic=float(cast(Any, gates["min_mean_rankic"])),
            min_mean_rankic_delta=float(cast(Any, gates["min_mean_rankic_delta"])),
            min_positive_folds=int(cast(Any, gates["min_positive_folds"])),
            min_nondegrading_folds=int(cast(Any, gates["min_nondegrading_folds"])),
            min_horizon_delta_1d=float(cast(Any, gates["min_horizon_delta_1d"])),
            min_horizon_delta_2d=float(cast(Any, gates["min_horizon_delta_2d"])),
            min_horizon_delta_3d=float(cast(Any, gates["min_horizon_delta_3d"])),
            min_horizon_delta_5d=float(cast(Any, gates["min_horizon_delta_5d"])),
            min_median_samples_per_second=float(
                cast(Any, gates["min_median_samples_per_second"])
            ),
            min_market_gate_output_l2=float(
                cast(Any, gates["min_market_gate_output_l2"])
            ),
            control_parameter_count=int(cast(Any, gates["control_parameter_count"])),
            candidate_parameter_count=int(
                cast(Any, gates["candidate_parameter_count"])
            ),
        )

        benchmark_config = cast(dict[str, object], config["lstm_benchmark"])
        reference_trial = by_trial[control_trial_id]
        lstm = benchmark_sequence_models(
            features,
            window_index,
            labels,
            split_manifest,
            seed=7,
            seeds=(7,),
            hidden_size=int(cast(Any, benchmark_config["hidden_size"])),
            tcn_channels=reference_trial.channels,
            tcn_kernel_size=reference_trial.kernel_size,
            tcn_dilations=reference_trial.dilations[:-1],
            epochs=int(cast(Any, benchmark_config["epochs"])),
            batch_size=int(cast(Any, benchmark_config["batch_size"])),
            device="cpu",
            num_workers=int(cast(Any, config["num_workers"])),
            torch_threads=int(cast(Any, config["torch_threads"])),
            learning_rate=float(cast(Any, benchmark_config["learning_rate"])),
            models=("lstm",),
        )
        comparison_rows = tuning.leaderboard.loc[
            tuning.leaderboard["trial_id"].eq(candidate_trial_id)
        ].copy()
        comparison: dict[str, float | int] = build_tcn_lstm_comparison(
            comparison_rows, lstm.measurements
        )
        final = finalize_pit_market_conditioning_seed7(
            effect,
            comparison,
            min_model_step_speed_ratio=float(
                cast(Any, gates["min_model_step_speed_ratio"])
            ),
            min_end_to_end_speed_ratio=float(
                cast(Any, gates["min_end_to_end_speed_ratio"])
            ),
        )

        protocols = build_fold_protocols(features, split_manifest)
        scaler_evidence = []
        for protocol in protocols:
            scaler = fit_market_context_standardizer(
                market_context,
                window_index,
                train_positions=protocol.train_positions,
            )
            scaler_evidence.append(
                {
                    "fold": protocol.fold,
                    "identity": scaler.identity,
                    "fit_date_count": scaler.fit_date_count,
                    "mean": scaler.mean.tolist(),
                    "std": scaler.std.tolist(),
                }
            )

        temporary.mkdir(parents=True)
        np.save(
            temporary / "market-context.npy", market_context.values, allow_pickle=False
        )
        _write_json(
            temporary / "market-context-manifest.json",
            {
                "schema_version": "pit-market-context-v17/v1",
                "identity": market_context.identity,
                "field_names": list(market_context.field_names),
                "feature_indices": list(market_context.feature_indices),
                "bars_per_day": market_context.bars_per_day,
                "available_positions": market_context.available_positions.tolist(),
                "date_sample_counts": market_context.date_sample_counts,
                "industry_context_status": "blocked_historical_industry_unavailable",
                "fold_scalers": scaler_evidence,
            },
        )
        tuning.epoch_history.to_parquet(
            temporary / "tcn-epoch-history.parquet", index=False
        )
        tuning.leaderboard.to_parquet(
            temporary / "tcn-leaderboard.parquet", index=False
        )
        effect.summary.to_parquet(temporary / "tcn-summary.parquet", index=False)
        effect.horizon_summary.to_parquet(
            temporary / "horizon-summary.parquet", index=False
        )
        lstm.measurements.to_parquet(
            temporary / "lstm-measurements.parquet", index=False
        )
        _write_json(temporary / "lstm-environment.json", lstm.environment)
        comparison_output: dict[str, object] = {
            **comparison,
            "tcn_trial_id": candidate_trial_id,
            "benchmark_feature_parity": False,
            "benchmark_note": "fixed stock-only LSTM reference; TCN control isolates context effect",
        }
        _write_json(temporary / "comparison.json", comparison_output)
        selection = {
            "status": final.status,
            "effect_gate_status": effect.status,
            "effect_winner_trial_id": effect.winner_trial_id,
            "winner_trial_id": final.winner_trial_id,
            "comparison_trial_id": candidate_trial_id,
            "relative_speed_gate_passed": final.relative_speed_gate_passed,
            "confirmation_seeds_authorized": list(final.confirmation_seeds_authorized),
            "capacity_delta": int(cast(Any, gates["candidate_parameter_count"]))
            - int(cast(Any, gates["control_parameter_count"])),
            "industry_context_status": "blocked_historical_industry_unavailable",
            "sealed_test_authorized": False,
        }
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for checkpoint_key, state in tuning.best_states.items():
            torch.save(state, checkpoint_dir / f"{checkpoint_key}.pt")
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-pit-market-conditioning-v17/v1",
            "run_id": str(config["run_id"]),
            "parent": parent,
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "source_config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "code_identity": code_identity(ROOT),
            "market_context": {
                "identity": market_context.identity,
                "field_names": list(market_context.field_names),
                "industry_context_status": "blocked_historical_industry_unavailable",
            },
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
            },
            "selection": selection,
            "comparison": comparison_output,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "result": final.status,
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
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
