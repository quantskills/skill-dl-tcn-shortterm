"""Run the immutable v22 dynamic-skip learning-rate probe."""

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
from skill_dl_tcn_shortterm.dynamic_multiscale import (  # noqa: E402
    evaluate_dynamic_skip_lr_multiseed,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    build_tcn_lstm_comparison,
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    run_tcn_validation_sweep,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_multiseed_confirmation import (  # noqa: E402
    _contains_secret_key,
    _multiscale_diagnostics,
    _resolve_project_path,
    _sha256,
    _write_json,
)


def _load_parent(
    config: dict[str, object],
    *,
    prefix: str,
    expected_source_hashes: dict[str, str],
) -> tuple[Path, dict[str, object]]:
    artifact = _resolve_project_path(config[f"{prefix}_parent_artifact"])
    receipt_path = artifact / "receipt.json"
    selection_path = artifact / "selection.json"
    if not receipt_path.is_file() or not selection_path.is_file():
        raise ContractError(f"v22 {prefix} parent artifact is incomplete")
    receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
    selection_value = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(receipt_value, dict) or not isinstance(selection_value, dict):
        raise ContractError(f"v22 {prefix} parent evidence must contain objects")
    receipt = cast(dict[str, object], receipt_value)
    selection = cast(dict[str, object], selection_value)
    expected_receipt_id = str(config[f"{prefix}_parent_receipt_id"])
    expected_status = str(config[f"{prefix}_parent_selection_status"])
    if receipt.get("receipt_id") != expected_receipt_id:
        raise ContractError(f"v22 {prefix} parent receipt identity drifted")
    if receipt.get("sealed_test_accessed") is not False:
        raise ContractError(f"v22 {prefix} parent accessed sealed evidence")
    if selection.get("status") != expected_status:
        raise ContractError(f"v22 {prefix} parent status drifted")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ContractError(f"v22 {prefix} parent outputs are missing")
    for relative, expected_hash in outputs.items():
        path = artifact / str(relative)
        if not path.is_file() or _sha256(path) != str(expected_hash):
            raise ContractError(f"v22 {prefix} parent output hash drifted")
    source_artifacts = receipt.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise ContractError(f"v22 {prefix} parent sources are missing")
    for name, expected_hash in expected_source_hashes.items():
        source = source_artifacts.get(name)
        if not isinstance(source, dict) or source.get("sha256") != expected_hash:
            raise ContractError(f"v22 {prefix} parent source identity drifted")
    return artifact, {
        "path": str(artifact),
        "receipt_id": expected_receipt_id,
        "selection_status": expected_status,
    }


def _historical_evidence(
    seed7_parent: Path,
    confirmation_parent: Path,
    *,
    control_trial_id: str,
    parent_candidate_trial_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    seed7_leaderboard = pd.read_parquet(seed7_parent / "tcn-leaderboard.parquet")
    confirmation_leaderboard = pd.read_parquet(
        confirmation_parent / "tcn-leaderboard.parquet"
    )
    historical = pd.concat(
        [seed7_leaderboard, confirmation_leaderboard], ignore_index=True
    )
    historical = historical.loc[
        historical["trial_id"].astype(str).isin(
            [control_trial_id, parent_candidate_trial_id]
        )
    ].copy()
    expected_units = {
        (trial_id, seed, fold)
        for trial_id in (control_trial_id, parent_candidate_trial_id)
        for seed in (7, 17, 27)
        for fold in range(5)
    }
    observed_units = {
        (str(row.trial_id), int(cast(Any, row.seed)), int(cast(Any, row.fold)))
        for row in historical.itertuples(index=False)
    }
    if observed_units != expected_units or historical.duplicated(
        ["trial_id", "seed", "fold"]
    ).any():
        raise ContractError("v22 historical TCN coverage drifted")

    seed7_diagnostics = pd.read_parquet(
        seed7_parent / "attention-diagnostics.parquet"
    ).copy()
    seed7_diagnostics["seed"] = 7
    confirmation_diagnostics = pd.read_parquet(
        confirmation_parent / "attention-diagnostics.parquet"
    )
    parent_diagnostics = pd.concat(
        [seed7_diagnostics, confirmation_diagnostics], ignore_index=True
    )
    if set(parent_diagnostics["trial_id"].astype(str)) != {
        parent_candidate_trial_id
    }:
        raise ContractError("v22 parent diagnostic identity drifted")

    seed7_lstm = pd.read_parquet(seed7_parent / "lstm-measurements.parquet")
    confirmation_lstm = pd.read_parquet(
        confirmation_parent / "lstm-measurements.parquet"
    )
    lstm = pd.concat([seed7_lstm, confirmation_lstm], ignore_index=True)
    expected_lstm_units = {
        (seed, fold) for seed in (7, 17, 27) for fold in range(5)
    }
    observed_lstm_units = {
        (int(cast(Any, row.base_seed)), int(cast(Any, row.fold)))
        for row in lstm.itertuples(index=False)
    }
    if (
        observed_lstm_units != expected_lstm_units
        or lstm.duplicated(["base_seed", "fold"]).any()
        or set(lstm["model"].astype(str)) != {"lstm"}
        or set(lstm["parameter_count"].astype(int)) != {6124}
    ):
        raise ContractError("v22 fixed LSTM evidence coverage drifted")
    environment = {
        "models": ["lstm"],
        "base_seeds": [7, 17, 27],
        "fold_count": 5,
        "device": "cpu",
        "evidence_mode": "immutable-v20-v21-composition",
        "seed7_environment_sha256": _sha256(seed7_parent / "lstm-environment.json"),
        "confirmation_environment_sha256": _sha256(
            confirmation_parent / "lstm-environment.json"
        ),
    }
    return historical, parent_diagnostics, lstm, environment


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the immutable v22 dynamic-skip optimizer probe"
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
            raise ContractError("v22 refuses to overwrite experiment artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v22 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v22":
            raise ContractError("v22 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v22 config contains a secret-like key")
        if config.get("precision") != "float32":
            raise ContractError("v22 precision must remain float32")
        seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["seeds"])
        )
        if seeds != (7, 17, 27):
            raise ContractError("v22 seeds must be exactly 7, 17 and 27")
        if cast(list[object], config["folds"]) != [0, 1, 2, 3, 4]:
            raise ContractError("v22 folds must be exactly 0 through 4")

        expected_hashes_value = config.get("source_sha256")
        if not isinstance(expected_hashes_value, dict):
            raise ContractError("v22 source identities are missing")
        expected_hashes = {
            str(key): str(value) for key, value in expected_hashes_value.items()
        }
        seed7_parent, seed7_parent_identity = _load_parent(
            config, prefix="seed7", expected_source_hashes=expected_hashes
        )
        confirmation_parent, confirmation_parent_identity = _load_parent(
            config, prefix="confirmation", expected_source_hashes=expected_hashes
        )

        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "labels": run_dir / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "universe": run_dir / "universe.parquet",
            "input_manifest": run_dir / "input-manifest.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v22 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if observed_hashes != expected_hashes:
            raise ContractError("v22 source SHA-256 identity drifted")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v22 rejects sealed split rows")
        allowed_stages = {"train", "validation", "purged"}
        observed_stages: set[str] = {
            str(value) for value in raw_split["stage"].astype(str).tolist()
        }
        if unknown := sorted(observed_stages - allowed_stages):
            raise ContractError("v22 split contains forbidden stages: " + ", ".join(unknown))
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        candidate_trial_id = str(config["candidate_trial_id"])
        control_trial_id = str(config["control_trial_id"])
        parent_candidate_trial_id = str(config["parent_candidate_trial_id"])
        if len(trials) != 1 or trials[0].trial_id != candidate_trial_id:
            raise ContractError("v22 must train exactly one candidate")
        candidate = trials[0]
        if (
            candidate.model_kind != "dynamic_horizon_skip"
            or candidate.channels != 16
            or candidate.kernel_size != 3
            or candidate.dilations != (1, 2, 4, 8, 16, 32, 64, 128)
            or candidate.dynamic_skip_hidden != 4
            or candidate.dynamic_skip_scale != 1.0
            or candidate.learning_rate != 0.003
            or candidate.dynamic_skip_learning_rate != 0.01
            or candidate.batch_size != 128
            or candidate.strategy != "smooth_l1"
            or candidate.padding_mode != "chomp"
        ):
            raise ContractError("v22 single-variable candidate contract drifted")

        protocol_identities = {
            "data": observed_hashes["features"],
            "fold_manifest": observed_hashes["split_manifest"],
            "evaluation": observed_hashes["labels"],
        }
        tuning_parts = []
        best_states: dict[str, dict[str, torch.Tensor]] = {}
        for seed in seeds:
            tuning = run_tcn_validation_sweep(
                features,
                window_index,
                labels,
                split_manifest,
                trials=trials,
                seed=seed,
                max_epochs=int(cast(Any, config["max_epochs"])),
                patience=int(cast(Any, config["patience"])),
                min_delta=float(cast(Any, config["min_delta"])),
                torch_threads=int(cast(Any, config["torch_threads"])),
                protocol_identities=protocol_identities,
            )
            tuning_parts.append(tuning)
            for key, state in tuning.best_states.items():
                best_states[f"seed-{seed}-{key}"] = state
        epoch_history = pd.concat(
            [part.epoch_history for part in tuning_parts], ignore_index=True
        )
        leaderboard = pd.concat(
            [part.leaderboard for part in tuning_parts], ignore_index=True
        )
        diagnostics = _multiscale_diagnostics(
            features,
            split_manifest,
            candidate,
            best_states,
            seeds=seeds,
            batch_size=candidate.batch_size,
        )
        historical, parent_diagnostics, lstm, lstm_environment = (
            _historical_evidence(
                seed7_parent,
                confirmation_parent,
                control_trial_id=control_trial_id,
                parent_candidate_trial_id=parent_candidate_trial_id,
            )
        )
        comparison = build_tcn_lstm_comparison(leaderboard, lstm)
        gates = cast(dict[str, object], config["gates"])
        decision = evaluate_dynamic_skip_lr_multiseed(
            leaderboard,
            historical,
            diagnostics,
            parent_diagnostics,
            comparison,
            control_trial_id=control_trial_id,
            parent_candidate_trial_id=parent_candidate_trial_id,
            candidate_trial_id=candidate_trial_id,
            expected_seeds=seeds,
            min_mean_rankic=float(cast(Any, gates["min_mean_rankic"])),
            min_positive_units=int(cast(Any, gates["min_positive_units"])),
            min_mean_rankic_delta=float(cast(Any, gates["min_mean_rankic_delta"])),
            min_parent_mean_rankic_delta=float(
                cast(Any, gates["min_parent_mean_rankic_delta"])
            ),
            min_nondegrading_folds_per_seed=int(
                cast(Any, gates["min_nondegrading_folds_per_seed"])
            ),
            min_horizon_delta_1d=float(cast(Any, gates["min_horizon_delta_1d"])),
            min_horizon_delta_2d=float(cast(Any, gates["min_horizon_delta_2d"])),
            min_horizon_delta_3d=float(cast(Any, gates["min_horizon_delta_3d"])),
            min_horizon_delta_5d=float(cast(Any, gates["min_horizon_delta_5d"])),
            min_median_samples_per_second=float(
                cast(Any, gates["min_median_samples_per_second"])
            ),
            min_dynamic_skip_output_weight_l2=float(
                cast(Any, gates["min_dynamic_skip_output_weight_l2"])
            ),
            min_block_weight_variation=float(
                cast(Any, gates["min_block_weight_variation"])
            ),
            min_parent_variation_ratio=float(
                cast(Any, gates["min_parent_variation_ratio"])
            ),
            max_simplex_error=float(cast(Any, gates["max_simplex_error"])),
            control_parameter_count=int(cast(Any, gates["control_parameter_count"])),
            candidate_parameter_count=int(
                cast(Any, gates["candidate_parameter_count"])
            ),
            dynamic_parameter_count=int(cast(Any, gates["dynamic_parameter_count"])),
            base_learning_rate=float(cast(Any, gates["base_learning_rate"])),
            dynamic_skip_learning_rate=float(
                cast(Any, gates["dynamic_skip_learning_rate"])
            ),
            min_model_step_speed_ratio=float(
                cast(Any, gates["min_model_step_speed_ratio"])
            ),
            min_end_to_end_speed_ratio=float(
                cast(Any, gates["min_end_to_end_speed_ratio"])
            ),
        )

        temporary.mkdir(parents=True)
        epoch_history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        diagnostics.to_parquet(temporary / "attention-diagnostics.parquet", index=False)
        historical.to_parquet(temporary / "historical-controls.parquet", index=False)
        parent_diagnostics.to_parquet(
            temporary / "parent-attention-diagnostics.parquet", index=False
        )
        decision.seed_summary.to_parquet(temporary / "seed-summary.parquet", index=False)
        decision.horizon_summary.to_parquet(
            temporary / "horizon-summary.parquet", index=False
        )
        lstm.to_parquet(temporary / "lstm-measurements.parquet", index=False)
        _write_json(temporary / "lstm-environment.json", lstm_environment)
        _write_json(temporary / "comparison.json", comparison)
        selection = {
            "status": decision.status,
            "effect_passed": decision.effect_passed,
            "speed_passed": decision.speed_passed,
            "candidate_trial_id": candidate_trial_id,
            "control_trial_id": control_trial_id,
            "parent_candidate_trial_id": parent_candidate_trial_id,
            "seeds": list(seeds),
            "aggregate": decision.aggregate,
            "sealed_test_authorized": False,
        }
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for checkpoint_key, state in best_states.items():
            torch.save(state, checkpoint_dir / f"{checkpoint_key}.pt")
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-dynamic-skip-learning-rate-v22/v1",
            "run_id": str(config["run_id"]),
            "parents": {
                "seed7": seed7_parent_identity,
                "confirmation": confirmation_parent_identity,
            },
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
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
            },
            "selection": selection,
            "comparison": comparison,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "result": decision.status,
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
