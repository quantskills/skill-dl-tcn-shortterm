"""Run the immutable v23 dynamic-skip warm-up probe."""

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
    evaluate_dynamic_skip_warmup_multiseed,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    build_tcn_lstm_comparison,
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    dynamic_skip_learning_rate_for_epoch,
    run_tcn_validation_sweep,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_dynamic_skip_learning_rate import (  # noqa: E402
    _historical_evidence,
    _load_parent,
)
from run_tcn_multiseed_confirmation import (  # noqa: E402
    _contains_secret_key,
    _multiscale_diagnostics,
    _sha256,
    _write_json,
)


def _audit_epoch_schedule(
    epoch_history: pd.DataFrame,
    trial: Any,
    *,
    expected_seeds: tuple[int, ...],
) -> None:
    required = {
        "trial_id",
        "seed",
        "fold",
        "epoch",
        "dynamic_skip_epoch_learning_rate",
    }
    if missing := sorted(required.difference(epoch_history.columns)):
        raise ContractError(
            "v23 epoch history missing schedule columns: " + ", ".join(missing)
        )
    if set(epoch_history["trial_id"].astype(str)) != {trial.trial_id}:
        raise ContractError("v23 epoch history trial identity drifted")
    if set(epoch_history["seed"].astype(int)) != set(expected_seeds):
        raise ContractError("v23 epoch history seed coverage drifted")
    if epoch_history.duplicated(["trial_id", "seed", "fold", "epoch"]).any():
        raise ContractError("v23 epoch history contains duplicate units")
    for row in epoch_history.itertuples(index=False):
        expected = dynamic_skip_learning_rate_for_epoch(
            trial, int(cast(Any, row.epoch))
        )
        observed = float(cast(Any, row.dynamic_skip_epoch_learning_rate))
        if expected is None or not np.isclose(observed, expected, rtol=0, atol=1e-12):
            raise ContractError("v23 applied learning-rate schedule drifted")
    for seed in expected_seeds:
        for fold in range(5):
            rows = epoch_history.loc[
                epoch_history["seed"].astype(int).eq(seed)
                & epoch_history["fold"].astype(int).eq(fold)
            ]
            observed_epochs = sorted(rows["epoch"].astype(int).tolist())
            if not observed_epochs or observed_epochs != list(
                range(1, max(observed_epochs) + 1)
            ):
                raise ContractError("v23 epoch schedule coverage drifted")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the immutable v23 dynamic-skip warm-up probe"
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
            raise ContractError("v23 refuses to overwrite experiment artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v23 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v23":
            raise ContractError("v23 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v23 config contains a secret-like key")
        if config.get("precision") != "float32":
            raise ContractError("v23 precision must remain float32")
        seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["seeds"])
        )
        if seeds != (7, 17, 27):
            raise ContractError("v23 seeds must be exactly 7, 17 and 27")
        if cast(list[object], config["folds"]) != [0, 1, 2, 3, 4]:
            raise ContractError("v23 folds must be exactly 0 through 4")

        expected_hashes_value = config.get("source_sha256")
        if not isinstance(expected_hashes_value, dict):
            raise ContractError("v23 source identities are missing")
        expected_hashes = {
            str(key): str(value) for key, value in expected_hashes_value.items()
        }
        seed7_parent, seed7_identity = _load_parent(
            config, prefix="seed7", expected_source_hashes=expected_hashes
        )
        confirmation_parent, confirmation_identity = _load_parent(
            config, prefix="confirmation", expected_source_hashes=expected_hashes
        )
        high_lr_parent, high_lr_identity = _load_parent(
            config, prefix="high_lr", expected_source_hashes=expected_hashes
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
            raise ContractError("v23 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if observed_hashes != expected_hashes:
            raise ContractError("v23 source SHA-256 identity drifted")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v23 rejects sealed split rows")
        observed_stages: set[str] = {
            str(value) for value in raw_split["stage"].astype(str).tolist()
        }
        if unknown := sorted(observed_stages - {"train", "validation", "purged"}):
            raise ContractError(
                "v23 split contains forbidden stages: " + ", ".join(unknown)
            )
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        candidate_trial_id = str(config["candidate_trial_id"])
        control_trial_id = str(config["control_trial_id"])
        parent_candidate_trial_id = str(config["parent_candidate_trial_id"])
        high_lr_candidate_trial_id = str(config["high_lr_candidate_trial_id"])
        if len(trials) != 1 or trials[0].trial_id != candidate_trial_id:
            raise ContractError("v23 must train exactly one candidate")
        candidate = trials[0]
        if (
            candidate.model_kind != "dynamic_horizon_skip"
            or candidate.channels != 16
            or candidate.kernel_size != 3
            or candidate.dilations != (1, 2, 4, 8, 16, 32, 64, 128)
            or candidate.dynamic_skip_hidden != 4
            or candidate.dynamic_skip_scale != 1.0
            or candidate.learning_rate != 0.003
            or candidate.dynamic_skip_learning_rate != 0.005
            or candidate.dynamic_skip_warmup_epochs != 2
            or candidate.batch_size != 128
            or candidate.strategy != "smooth_l1"
            or candidate.padding_mode != "chomp"
        ):
            raise ContractError("v23 single-variable warm-up contract drifted")

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
        _audit_epoch_schedule(epoch_history, candidate, expected_seeds=seeds)
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
        high_lr_diagnostics = pd.read_parquet(
            high_lr_parent / "attention-diagnostics.parquet"
        )
        comparison = build_tcn_lstm_comparison(leaderboard, lstm)
        gates = cast(dict[str, object], config["gates"])
        decision = evaluate_dynamic_skip_warmup_multiseed(
            leaderboard,
            historical,
            diagnostics,
            parent_diagnostics,
            high_lr_diagnostics,
            comparison,
            control_trial_id=control_trial_id,
            parent_candidate_trial_id=parent_candidate_trial_id,
            high_lr_candidate_trial_id=high_lr_candidate_trial_id,
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
            max_parent_variation_ratio=float(
                cast(Any, gates["max_parent_variation_ratio"])
            ),
            max_high_lr_variation_ratio=float(
                cast(Any, gates["max_high_lr_variation_ratio"])
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
            dynamic_skip_warmup_epochs=int(
                cast(Any, gates["dynamic_skip_warmup_epochs"])
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
        high_lr_diagnostics.to_parquet(
            temporary / "high-lr-attention-diagnostics.parquet", index=False
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
            "high_lr_candidate_trial_id": high_lr_candidate_trial_id,
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
            "schema_version": "tcn-dynamic-skip-warmup-v23/v1",
            "run_id": str(config["run_id"]),
            "parents": {
                "seed7": seed7_identity,
                "confirmation": confirmation_identity,
                "high_lr": high_lr_identity,
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
