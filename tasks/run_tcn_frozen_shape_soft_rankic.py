"""Run the immutable v30 frozen shape soft-RankIC experiment."""

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
    evaluate_frozen_shape_soft_rankic_multiseed,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    build_tcn_lstm_comparison,
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    TCNTuningTrial,
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


def _paired_objective_comparison(
    current: pd.DataFrame,
    v28: pd.DataFrame,
    *,
    candidate_trial_id: str,
    grouped_control_trial_id: str,
) -> pd.DataFrame:
    key = ["seed", "fold"]
    value_columns = key + [
        "best_epoch",
        "best_mean_daily_rankic",
        "dynamic_skip_shape_output_weight_l2",
    ]

    def _trial(trial_id: str, prefix: str) -> pd.DataFrame:
        rows = current.loc[current["trial_id"].astype(str).eq(trial_id)]
        return rows[value_columns].rename(
            columns={
                "best_epoch": f"{prefix}_best_epoch",
                "best_mean_daily_rankic": f"{prefix}_mean_daily_rankic",
                "dynamic_skip_shape_output_weight_l2": (
                    f"{prefix}_shape_output_weight_l2"
                ),
            }
        )

    candidate = _trial(candidate_trial_id, "candidate")
    grouped = _trial(grouped_control_trial_id, "grouped_control")
    historical = v28[value_columns].rename(
        columns={
            "best_epoch": "v28_best_epoch",
            "best_mean_daily_rankic": "v28_mean_daily_rankic",
            "dynamic_skip_shape_output_weight_l2": "v28_shape_output_weight_l2",
        }
    )
    paired = candidate.merge(
        grouped, on=key, how="outer", validate="one_to_one", indicator="candidate_grouped"
    ).merge(
        historical,
        on=key,
        how="outer",
        validate="one_to_one",
        indicator="current_v28",
    )
    if (
        len(paired) != 15
        or set(paired["candidate_grouped"].astype(str)) != {"both"}
        or set(paired["current_v28"].astype(str)) != {"both"}
    ):
        raise ContractError("v30 objective paired comparison coverage drifted")
    paired = paired.drop(columns=["candidate_grouped", "current_v28"])
    paired["candidate_grouped_control_rankic_delta"] = (
        paired["candidate_mean_daily_rankic"]
        - paired["grouped_control_mean_daily_rankic"]
    )
    paired["candidate_v28_rankic_delta"] = (
        paired["candidate_mean_daily_rankic"] - paired["v28_mean_daily_rankic"]
    )
    paired["grouped_control_v28_rankic_delta"] = (
        paired["grouped_control_mean_daily_rankic"]
        - paired["v28_mean_daily_rankic"]
    )
    return paired.sort_values(key, ignore_index=True)


def _validate_common_trial(trial: TCNTuningTrial) -> bool:
    return bool(
        trial.model_kind == "dynamic_horizon_skip"
        and trial.channels == 16
        and trial.kernel_size == 3
        and trial.dilations == (1, 2, 4, 8, 16, 32, 64, 128)
        and trial.dynamic_skip_hidden == 4
        and trial.dynamic_skip_scale == 1.0
        and trial.dynamic_skip_token_normalization == "none"
        and trial.dynamic_skip_shape_residual is True
        and trial.dynamic_skip_shape_residual_scale == 0.25
        and trial.dynamic_skip_frozen_parent is True
        and trial.learning_rate == 0.003
        and trial.dynamic_skip_learning_rate is None
        and trial.dynamic_skip_warmup_epochs == 0
        and trial.weight_decay == 0
        and trial.batch_size == 128
        and trial.padding_mode == "chomp"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the immutable v30 frozen shape soft-RankIC probe"
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
            raise ContractError("v30 refuses to overwrite experiment artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v30 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v30":
            raise ContractError("v30 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v30 config contains a secret-like key")
        if config.get("precision") != "float32":
            raise ContractError("v30 precision must remain float32")
        if int(cast(Any, config["num_workers"])) != 0:
            raise ContractError("v30 num_workers must remain zero")
        checkpoint_min_delta = float(cast(Any, config["checkpoint_min_delta"]))
        patience_min_delta = float(cast(Any, config["min_delta"]))
        if checkpoint_min_delta != 0.0 or patience_min_delta != 0.0005:
            raise ContractError("v30 decoupled selection thresholds drifted")
        seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["seeds"])
        )
        if seeds != (7, 17, 27):
            raise ContractError("v30 seeds must be exactly 7, 17 and 27")
        if cast(list[object], config["folds"]) != [0, 1, 2, 3, 4]:
            raise ContractError("v30 folds must be exactly 0 through 4")

        expected_hashes_value = config.get("source_sha256")
        if not isinstance(expected_hashes_value, dict):
            raise ContractError("v30 source identities are missing")
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
        v28_parent, v28_identity = _load_parent(
            config, prefix="v28", expected_source_hashes=expected_hashes
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
            raise ContractError("v30 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if observed_hashes != expected_hashes:
            raise ContractError("v30 source SHA-256 identity drifted")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v30 rejects sealed split rows")
        observed_stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(observed_stages - {"train", "validation", "purged"}):
            raise ContractError(
                "v30 split contains forbidden stages: " + ", ".join(unknown)
            )
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        candidate_trial_id = str(config["candidate_trial_id"])
        grouped_control_trial_id = str(config["grouped_control_trial_id"])
        control_trial_id = str(config["control_trial_id"])
        parent_candidate_trial_id = str(config["parent_candidate_trial_id"])
        v25_trial_id = str(config["v25_trial_id"])
        v26_trial_id = str(config["v26_trial_id"])
        v28_trial_id = str(config["v28_trial_id"])
        trials_by_id = {trial.trial_id: trial for trial in trials}
        if set(trials_by_id) != {candidate_trial_id, grouped_control_trial_id}:
            raise ContractError("v30 must train exactly the two registered trials")
        candidate = trials_by_id[candidate_trial_id]
        grouped_control = trials_by_id[grouped_control_trial_id]
        if not _validate_common_trial(candidate) or not _validate_common_trial(
            grouped_control
        ):
            raise ContractError("v30 frozen shape common contract drifted")
        if (
            grouped_control.strategy != "grouped_smooth_l1"
            or candidate.strategy != "soft_rankic"
            or candidate.soft_rankic_weight != 0.05
            or candidate.soft_rank_temperature != 0.1
        ):
            raise ContractError("v30 objective contract drifted")

        grouped_states, grouped_manifest = _load_frozen_parent_states(
            seed7_parent, confirmation_parent, grouped_control_trial_id
        )
        candidate_states, candidate_manifest = _load_frozen_parent_states(
            seed7_parent, confirmation_parent, candidate_trial_id
        )
        grouped_manifest.insert(0, "trial_id", grouped_control_trial_id)
        candidate_manifest.insert(0, "trial_id", candidate_trial_id)
        checkpoint_manifest = pd.concat(
            [grouped_manifest, candidate_manifest], ignore_index=True
        )
        if len(checkpoint_manifest) != 30 or checkpoint_manifest.duplicated(
            ["trial_id", "seed", "fold"]
        ).any():
            raise ContractError("v30 parent checkpoint manifest drifted")
        frozen_states = {
            seed: {**grouped_states[seed], **candidate_states[seed]}
            for seed in seeds
        }
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
            checkpoint_manifest[
                ["trial_id", "seed", "fold", "parent_checkpoint_sha256"]
            ],
            on=["trial_id", "seed", "fold"],
            how="left",
            validate="one_to_one",
        )
        diagnostic_parts = [
            _frozen_shape_diagnostics(
                features,
                window_index,
                labels,
                split_manifest,
                trial,
                best_states,
                seeds=seeds,
                batch_size=trial.batch_size,
            )
            for trial in (grouped_control, candidate)
        ]
        diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
        diagnostic_scores = diagnostics.set_index(["trial_id", "seed", "fold"])[
            "full_mean_daily_rankic"
        ]
        leaderboard_scores = leaderboard.set_index(["trial_id", "seed", "fold"])[
            "best_mean_daily_rankic"
        ]
        if float(np.max(np.abs(diagnostic_scores - leaderboard_scores))) > 1e-12:
            raise ContractError("v30 diagnostic RankIC drifted from leaderboard")

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
        v28_rows = _history_rows(v28_parent, v28_trial_id, label="v28")
        historical = pd.concat(
            [historical, v25_rows, v26_rows, v28_rows], ignore_index=True
        )
        paired_objective = _paired_objective_comparison(
            leaderboard,
            v28_rows,
            candidate_trial_id=candidate_trial_id,
            grouped_control_trial_id=grouped_control_trial_id,
        )
        candidate_rows = leaderboard.loc[
            leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
        ].copy()
        comparison = build_tcn_lstm_comparison(candidate_rows, lstm)
        gates = cast(dict[str, object], config["gates"])
        decision = evaluate_frozen_shape_soft_rankic_multiseed(
            leaderboard,
            historical,
            diagnostics,
            comparison,
            control_trial_id=control_trial_id,
            parent_candidate_trial_id=parent_candidate_trial_id,
            v25_trial_id=v25_trial_id,
            v26_trial_id=v26_trial_id,
            v28_trial_id=v28_trial_id,
            grouped_control_trial_id=grouped_control_trial_id,
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
            min_v28_mean_rankic_delta=float(
                cast(Any, gates["min_v28_mean_rankic_delta"])
            ),
            min_grouped_control_mean_rankic_delta=float(
                cast(Any, gates["min_grouped_control_mean_rankic_delta"])
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
            max_parent_rankic_abs_error=float(
                cast(Any, gates["max_parent_rankic_abs_error"])
            ),
            max_parent_prediction_abs_error=float(
                cast(Any, gates["max_parent_prediction_abs_error"])
            ),
            min_trained_effect_units=int(cast(Any, gates["min_trained_effect_units"])),
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
            candidate_parameter_count=int(cast(Any, gates["candidate_parameter_count"])),
            trainable_parameter_count=int(cast(Any, gates["trainable_parameter_count"])),
            frozen_parameter_count=int(cast(Any, gates["frozen_parameter_count"])),
            shape_residual_scale=float(cast(Any, gates["shape_residual_scale"])),
            learning_rate=float(cast(Any, gates["learning_rate"])),
            soft_rankic_weight=float(cast(Any, gates["soft_rankic_weight"])),
            soft_rank_temperature=float(cast(Any, gates["soft_rank_temperature"])),
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
        paired_objective.to_parquet(
            temporary / "objective-paired-comparison.parquet", index=False
        )
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
            "integrity_passed": decision.integrity_passed,
            "effect_passed": decision.effect_passed,
            "speed_passed": decision.speed_passed,
            "candidate_trial_id": candidate_trial_id,
            "grouped_control_trial_id": grouped_control_trial_id,
            "control_trial_id": control_trial_id,
            "parent_candidate_trial_id": parent_candidate_trial_id,
            "v25_trial_id": v25_trial_id,
            "v26_trial_id": v26_trial_id,
            "v28_trial_id": v28_trial_id,
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
            "schema_version": "tcn-frozen-shape-soft-rankic-v30/v1",
            "run_id": str(config["run_id"]),
            "parents": {
                "seed7": seed7_identity,
                "confirmation": confirmation_identity,
                "v25": v25_identity,
                "v26": v26_identity,
                "v28": v28_identity,
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
