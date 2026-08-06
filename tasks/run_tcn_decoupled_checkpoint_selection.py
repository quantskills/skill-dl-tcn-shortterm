"""Run the immutable v28 decoupled checkpoint-selection experiment."""

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
    evaluate_decoupled_checkpoint_selection_multiseed,
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

from run_tcn_dynamic_skip_learning_rate import (  # noqa: E402
    _historical_evidence,
    _load_parent,
)
from run_tcn_frozen_parent_shape_residual import (  # noqa: E402
    _frozen_shape_diagnostics,
    _history_rows,
    _load_frozen_parent_states,
)
from run_tcn_multiseed_confirmation import (  # noqa: E402
    _contains_secret_key,
    _sha256,
    _write_json,
)


def _trajectory_comparison(
    current: pd.DataFrame, previous: pd.DataFrame
) -> pd.DataFrame:
    key = ["seed", "fold", "epoch"]
    current_values = current[key + ["mean_daily_rankic"]].rename(
        columns={"mean_daily_rankic": "v28_mean_daily_rankic"}
    )
    previous_values = previous[key + ["mean_daily_rankic"]].rename(
        columns={"mean_daily_rankic": "v27_mean_daily_rankic"}
    )
    comparison = current_values.merge(
        previous_values,
        on=key,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    comparison["rankic_abs_error"] = np.abs(
        comparison["v28_mean_daily_rankic"]
        - comparison["v27_mean_daily_rankic"]
    )
    return comparison.sort_values(key, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the immutable v28 decoupled checkpoint-selection probe"
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
            raise ContractError("v28 refuses to overwrite experiment artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v28 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v28":
            raise ContractError("v28 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v28 config contains a secret-like key")
        if config.get("precision") != "float32":
            raise ContractError("v28 precision must remain float32")
        checkpoint_min_delta = float(
            cast(Any, config["checkpoint_min_delta"])
        )
        patience_min_delta = float(cast(Any, config["min_delta"]))
        if checkpoint_min_delta != 0.0 or patience_min_delta != 0.0005:
            raise ContractError("v28 decoupled selection thresholds drifted")
        seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["seeds"])
        )
        if seeds != (7, 17, 27):
            raise ContractError("v28 seeds must be exactly 7, 17 and 27")
        if cast(list[object], config["folds"]) != [0, 1, 2, 3, 4]:
            raise ContractError("v28 folds must be exactly 0 through 4")

        expected_hashes_value = config.get("source_sha256")
        if not isinstance(expected_hashes_value, dict):
            raise ContractError("v28 source identities are missing")
        expected_hashes = {
            str(key): str(value) for key, value in expected_hashes_value.items()
        }
        seed7_parent, seed7_identity = _load_parent(
            config, prefix="seed7", expected_source_hashes=expected_hashes
        )
        confirmation_parent, confirmation_identity = _load_parent(
            config, prefix="confirmation", expected_source_hashes=expected_hashes
        )
        v25_parent, v25_identity = _load_parent(
            config, prefix="v25", expected_source_hashes=expected_hashes
        )
        v26_parent, v26_identity = _load_parent(
            config, prefix="v26", expected_source_hashes=expected_hashes
        )
        v27_parent, v27_identity = _load_parent(
            config, prefix="v27", expected_source_hashes=expected_hashes
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
            raise ContractError("v28 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if observed_hashes != expected_hashes:
            raise ContractError("v28 source SHA-256 identity drifted")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v28 rejects sealed split rows")
        observed_stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(observed_stages - {"train", "validation", "purged"}):
            raise ContractError(
                "v28 split contains forbidden stages: " + ", ".join(unknown)
            )
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        candidate_trial_id = str(config["candidate_trial_id"])
        control_trial_id = str(config["control_trial_id"])
        parent_candidate_trial_id = str(config["parent_candidate_trial_id"])
        v25_trial_id = str(config["v25_trial_id"])
        v26_trial_id = str(config["v26_trial_id"])
        v27_trial_id = str(config["v27_trial_id"])
        if len(trials) != 1 or trials[0].trial_id != candidate_trial_id:
            raise ContractError("v28 must train exactly one candidate")
        candidate = trials[0]
        if (
            candidate.model_kind != "dynamic_horizon_skip"
            or candidate.channels != 16
            or candidate.kernel_size != 3
            or candidate.dilations != (1, 2, 4, 8, 16, 32, 64, 128)
            or candidate.dynamic_skip_hidden != 4
            or candidate.dynamic_skip_scale != 1.0
            or candidate.dynamic_skip_token_normalization != "none"
            or candidate.dynamic_skip_shape_residual is not True
            or candidate.dynamic_skip_shape_residual_scale != 0.25
            or candidate.dynamic_skip_frozen_parent is not True
            or candidate.learning_rate != 0.003
            or candidate.dynamic_skip_learning_rate is not None
            or candidate.dynamic_skip_warmup_epochs != 0
            or candidate.weight_decay != 0
            or candidate.batch_size != 128
            or candidate.strategy != "smooth_l1"
            or candidate.padding_mode != "chomp"
        ):
            raise ContractError("v28 frozen shape-only contract drifted")

        frozen_states, checkpoint_manifest = _load_frozen_parent_states(
            seed7_parent,
            confirmation_parent,
            candidate_trial_id,
        )
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
                min_delta=patience_min_delta,
                checkpoint_min_delta=checkpoint_min_delta,
                torch_threads=int(cast(Any, config["torch_threads"])),
                protocol_identities=protocol_identities,
                frozen_parent_states=frozen_states[seed],
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
        leaderboard = leaderboard.merge(
            checkpoint_manifest[["seed", "fold", "parent_checkpoint_sha256"]],
            on=["seed", "fold"],
            how="left",
            validate="one_to_one",
        )
        diagnostics = _frozen_shape_diagnostics(
            features,
            window_index,
            labels,
            split_manifest,
            candidate,
            best_states,
            seeds=seeds,
            batch_size=candidate.batch_size,
        )
        diagnostic_scores = diagnostics.set_index(["seed", "fold"])[
            "full_mean_daily_rankic"
        ]
        leaderboard_scores = leaderboard.set_index(["seed", "fold"])[
            "best_mean_daily_rankic"
        ]
        if float(np.max(np.abs(diagnostic_scores - leaderboard_scores))) > 1e-12:
            raise ContractError("v28 diagnostic RankIC drifted from leaderboard")

        historical, parent_diagnostics, lstm, lstm_environment = (
            _historical_evidence(
                seed7_parent,
                confirmation_parent,
                control_trial_id=control_trial_id,
                parent_candidate_trial_id=parent_candidate_trial_id,
            )
        )
        v25_rows = _history_rows(v25_parent, v25_trial_id, label="v25")
        v26_rows = _history_rows(v26_parent, v26_trial_id, label="v26")
        v27_rows = _history_rows(v27_parent, v27_trial_id, label="v27")
        historical = pd.concat(
            [historical, v25_rows, v26_rows, v27_rows], ignore_index=True
        )
        v27_epoch_history = pd.read_parquet(
            v27_parent / "tcn-epoch-history.parquet"
        )
        v27_epoch_history = v27_epoch_history.loc[
            v27_epoch_history["trial_id"].astype(str).eq(v27_trial_id)
        ].copy()
        trajectory = _trajectory_comparison(epoch_history, v27_epoch_history)
        comparison = build_tcn_lstm_comparison(leaderboard, lstm)
        gates = cast(dict[str, object], config["gates"])
        decision = evaluate_decoupled_checkpoint_selection_multiseed(
            leaderboard,
            historical,
            diagnostics,
            epoch_history,
            v27_epoch_history,
            comparison,
            control_trial_id=control_trial_id,
            parent_candidate_trial_id=parent_candidate_trial_id,
            v25_trial_id=v25_trial_id,
            v26_trial_id=v26_trial_id,
            v27_trial_id=v27_trial_id,
            candidate_trial_id=candidate_trial_id,
            expected_seeds=seeds,
            min_mean_rankic=float(cast(Any, gates["min_mean_rankic"])),
            min_positive_units=int(cast(Any, gates["min_positive_units"])),
            min_parent_mean_rankic_delta=float(
                cast(Any, gates["min_parent_mean_rankic_delta"])
            ),
            min_control_mean_rankic_delta=float(
                cast(Any, gates["min_control_mean_rankic_delta"])
            ),
            min_v26_mean_rankic_delta=float(
                cast(Any, gates["min_v26_mean_rankic_delta"])
            ),
            min_v25_mean_rankic_delta=float(
                cast(Any, gates["min_v25_mean_rankic_delta"])
            ),
            min_v27_mean_rankic_delta=float(
                cast(Any, gates["min_v27_mean_rankic_delta"])
            ),
            min_nondegrading_folds_per_seed=int(
                cast(Any, gates["min_nondegrading_folds_per_seed"])
            ),
            min_horizon_parent_delta_1d=float(
                cast(Any, gates["min_horizon_parent_delta_1d"])
            ),
            min_horizon_parent_delta_2d=float(
                cast(Any, gates["min_horizon_parent_delta_2d"])
            ),
            min_horizon_parent_delta_3d=float(
                cast(Any, gates["min_horizon_parent_delta_3d"])
            ),
            min_horizon_parent_delta_5d=float(
                cast(Any, gates["min_horizon_parent_delta_5d"])
            ),
            max_trajectory_rankic_abs_error=float(
                cast(Any, gates["max_trajectory_rankic_abs_error"])
            ),
            max_selected_best_abs_error=float(
                cast(Any, gates["max_selected_best_abs_error"])
            ),
            max_parent_rankic_abs_error=float(
                cast(Any, gates["max_parent_rankic_abs_error"])
            ),
            max_parent_prediction_abs_error=float(
                cast(Any, gates["max_parent_prediction_abs_error"])
            ),
            min_trained_effect_units=int(
                cast(Any, gates["min_trained_effect_units"])
            ),
            min_shape_output_weight_l2=float(
                cast(Any, gates["min_shape_output_weight_l2"])
            ),
            min_shape_residual_weight_effect=float(
                cast(Any, gates["min_shape_residual_weight_effect"])
            ),
            max_simplex_error=float(cast(Any, gates["max_simplex_error"])),
            min_median_samples_per_second=float(
                cast(Any, gates["min_median_samples_per_second"])
            ),
            candidate_parameter_count=int(
                cast(Any, gates["candidate_parameter_count"])
            ),
            trainable_parameter_count=int(
                cast(Any, gates["trainable_parameter_count"])
            ),
            frozen_parameter_count=int(
                cast(Any, gates["frozen_parameter_count"])
            ),
            shape_residual_scale=float(
                cast(Any, gates["shape_residual_scale"])
            ),
            learning_rate=float(cast(Any, gates["learning_rate"])),
            checkpoint_min_delta=checkpoint_min_delta,
            patience_min_delta=patience_min_delta,
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
        diagnostics.to_parquet(temporary / "shape-diagnostics.parquet", index=False)
        checkpoint_manifest.to_parquet(
            temporary / "parent-checkpoint-manifest.parquet", index=False
        )
        trajectory.to_parquet(
            temporary / "v27-trajectory-comparison.parquet", index=False
        )
        historical.to_parquet(temporary / "historical-controls.parquet", index=False)
        parent_diagnostics.to_parquet(
            temporary / "parent-attention-diagnostics.parquet", index=False
        )
        decision.seed_summary.to_parquet(
            temporary / "seed-summary.parquet", index=False
        )
        decision.horizon_summary.to_parquet(
            temporary / "horizon-summary.parquet", index=False
        )
        lstm.to_parquet(temporary / "lstm-measurements.parquet", index=False)
        _write_json(temporary / "lstm-environment.json", lstm_environment)
        _write_json(temporary / "comparison.json", comparison)
        selection = {
            "status": decision.status,
            "integrity_passed": decision.integrity_passed,
            "effect_passed": decision.effect_passed,
            "speed_passed": decision.speed_passed,
            "candidate_trial_id": candidate_trial_id,
            "control_trial_id": control_trial_id,
            "parent_candidate_trial_id": parent_candidate_trial_id,
            "v25_trial_id": v25_trial_id,
            "v26_trial_id": v26_trial_id,
            "v27_trial_id": v27_trial_id,
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
            "schema_version": "tcn-decoupled-checkpoint-selection-v28/v1",
            "run_id": str(config["run_id"]),
            "parents": {
                "seed7": seed7_identity,
                "confirmation": confirmation_identity,
                "v25": v25_identity,
                "v26": v26_identity,
                "v27": v27_identity,
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
