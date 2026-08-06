"""Run the immutable v26 dynamic-skip raw/shape residual probe."""

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.dynamic_multiscale import (  # noqa: E402
    evaluate_dynamic_skip_raw_shape_residual_multiseed,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    build_tcn_lstm_comparison,
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.training_data import (  # noqa: E402
    LazyWindowDataset,
    build_fold_protocols,
)
from skill_dl_tcn_shortterm.tuning import (  # noqa: E402
    TCNTuningTrial,
    build_tcn_trial_model,
    run_tcn_validation_sweep,
)
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402
from skill_dl_tcn_shortterm.v9_representation import (  # noqa: E402
    ShapeResidualDynamicHorizonSkipTCN,
)

from run_tcn_dynamic_skip_learning_rate import (  # noqa: E402
    _historical_evidence,
    _load_parent,
)
from run_tcn_multiseed_confirmation import (  # noqa: E402
    _contains_secret_key,
    _sha256,
    _write_json,
)


def _shape_residual_diagnostics(
    features: np.ndarray,
    split_manifest: pd.DataFrame,
    candidate: TCNTuningTrial,
    best_states: dict[str, dict[str, torch.Tensor]],
    *,
    seeds: tuple[int, ...],
    batch_size: int,
) -> pd.DataFrame:
    protocols = build_fold_protocols(features, split_manifest)
    dummy_targets = np.zeros((len(features), 4), dtype="float32")
    dummy_masks = np.ones((len(features), 4), dtype="bool")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for protocol in protocols:
            model = build_tcn_trial_model(
                candidate,
                feature_count=int(features.shape[1]),
                input_steps=int(features.shape[2]),
            )
            if not isinstance(model, ShapeResidualDynamicHorizonSkipTCN):
                raise ContractError("v26 diagnostics require the shape residual TCN")
            checkpoint_key = (
                f"seed-{seed}-{candidate.trial_id}-fold-{protocol.fold}"
            )
            try:
                model.load_state_dict(best_states[checkpoint_key])
            except KeyError as exc:
                raise ContractError("v26 dynamic checkpoint is missing") from exc
            dataset = LazyWindowDataset(
                features,
                protocol.validation_positions,
                dummy_targets,
                dummy_masks,
                protocol.feature_mean,
                protocol.feature_std,
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            weight_batches: list[np.ndarray] = []
            raw_weight_batches: list[np.ndarray] = []
            model.eval()
            observed = 0
            with torch.no_grad():
                for batch_features, _, _, _ in loader:
                    sequences = model.encode_blocks(batch_features)
                    weights = model.dynamic_skip_weights(sequences)
                    raw_weights = model.dynamic_skip_weights_without_shape_residual(
                        sequences
                    )
                    weight_batches.append(weights.cpu().numpy())
                    raw_weight_batches.append(raw_weights.cpu().numpy())
                    observed += len(batch_features)
                    if observed >= 512:
                        break
            if not weight_batches:
                raise ContractError("v26 diagnostics found no validation samples")
            observed_weights = np.concatenate(weight_batches, axis=0)[:512]
            observed_raw_weights = np.concatenate(raw_weight_batches, axis=0)[:512]
            rows.append(
                {
                    "trial_id": candidate.trial_id,
                    "seed": seed,
                    "fold": protocol.fold,
                    "sample_count": int(len(observed_weights)),
                    "block_weight_variation": float(
                        np.std(observed_weights, axis=0).max()
                    ),
                    "shape_residual_weight_effect_max": float(
                        np.abs(observed_weights - observed_raw_weights).max()
                    ),
                    "simplex_error_max": float(
                        np.abs(observed_weights.sum(axis=2) - 1.0).max()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "fold"], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the immutable v26 raw/shape residual probe"
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
            raise ContractError("v26 refuses to overwrite experiment artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v26 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v26":
            raise ContractError("v26 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v26 config contains a secret-like key")
        if config.get("precision") != "float32":
            raise ContractError("v26 precision must remain float32")
        seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["seeds"])
        )
        if seeds != (7, 17, 27):
            raise ContractError("v26 seeds must be exactly 7, 17 and 27")
        if cast(list[object], config["folds"]) != [0, 1, 2, 3, 4]:
            raise ContractError("v26 folds must be exactly 0 through 4")

        expected_hashes_value = config.get("source_sha256")
        if not isinstance(expected_hashes_value, dict):
            raise ContractError("v26 source identities are missing")
        expected_hashes = {
            str(key): str(value) for key, value in expected_hashes_value.items()
        }
        seed7_parent, seed7_identity = _load_parent(
            config, prefix="seed7", expected_source_hashes=expected_hashes
        )
        confirmation_parent, confirmation_identity = _load_parent(
            config, prefix="confirmation", expected_source_hashes=expected_hashes
        )
        ablation_parent, ablation_identity = _load_parent(
            config, prefix="ablation", expected_source_hashes=expected_hashes
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
            raise ContractError("v26 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if observed_hashes != expected_hashes:
            raise ContractError("v26 source SHA-256 identity drifted")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v26 rejects sealed split rows")
        observed_stages: set[str] = {
            str(value) for value in raw_split["stage"].astype(str).tolist()
        }
        if unknown := sorted(observed_stages - {"train", "validation", "purged"}):
            raise ContractError(
                "v26 split contains forbidden stages: " + ", ".join(unknown)
            )
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        candidate_trial_id = str(config["candidate_trial_id"])
        control_trial_id = str(config["control_trial_id"])
        parent_candidate_trial_id = str(config["parent_candidate_trial_id"])
        ablation_trial_id = str(config["ablation_trial_id"])
        if len(trials) != 1 or trials[0].trial_id != candidate_trial_id:
            raise ContractError("v26 must train exactly one candidate")
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
            or candidate.learning_rate != 0.003
            or candidate.dynamic_skip_learning_rate is not None
            or candidate.dynamic_skip_warmup_epochs != 0
            or candidate.weight_decay != 0
            or candidate.batch_size != 128
            or candidate.strategy != "smooth_l1"
            or candidate.padding_mode != "chomp"
        ):
            raise ContractError("v26 single-variable raw/shape residual contract drifted")

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
        diagnostics = _shape_residual_diagnostics(
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
        ablation_rows = pd.read_parquet(
            ablation_parent / "tcn-leaderboard.parquet"
        )
        ablation_rows = ablation_rows.loc[
            ablation_rows["trial_id"].astype(str).eq(ablation_trial_id)
        ].copy()
        expected_ablation_units = {
            (seed, fold) for seed in seeds for fold in range(5)
        }
        observed_ablation_units = {
            (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
            for row in ablation_rows.itertuples(index=False)
        }
        if (
            observed_ablation_units != expected_ablation_units
            or ablation_rows.duplicated(["trial_id", "seed", "fold"]).any()
        ):
            raise ContractError("v26 failed-ablation coverage drifted")
        historical = pd.concat([historical, ablation_rows], ignore_index=True)
        comparison = build_tcn_lstm_comparison(leaderboard, lstm)
        gates = cast(dict[str, object], config["gates"])
        decision = evaluate_dynamic_skip_raw_shape_residual_multiseed(
            leaderboard,
            historical,
            diagnostics,
            comparison,
            control_trial_id=control_trial_id,
            parent_candidate_trial_id=parent_candidate_trial_id,
            ablation_trial_id=ablation_trial_id,
            candidate_trial_id=candidate_trial_id,
            expected_seeds=seeds,
            min_mean_rankic=float(cast(Any, gates["min_mean_rankic"])),
            min_positive_units=int(cast(Any, gates["min_positive_units"])),
            min_mean_rankic_delta=float(cast(Any, gates["min_mean_rankic_delta"])),
            min_parent_mean_rankic_delta=float(
                cast(Any, gates["min_parent_mean_rankic_delta"])
            ),
            min_ablation_mean_rankic_delta=float(
                cast(Any, gates["min_ablation_mean_rankic_delta"])
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
            min_shape_output_weight_l2=float(
                cast(Any, gates["min_shape_output_weight_l2"])
            ),
            min_shape_residual_weight_effect=float(
                cast(Any, gates["min_shape_residual_weight_effect"])
            ),
            min_block_weight_variation=float(
                cast(Any, gates["min_block_weight_variation"])
            ),
            max_simplex_error=float(cast(Any, gates["max_simplex_error"])),
            control_parameter_count=int(cast(Any, gates["control_parameter_count"])),
            parent_parameter_count=int(
                cast(Any, gates["parent_parameter_count"])
            ),
            ablation_parameter_count=int(
                cast(Any, gates["ablation_parameter_count"])
            ),
            candidate_parameter_count=int(
                cast(Any, gates["candidate_parameter_count"])
            ),
            dynamic_parameter_count=int(cast(Any, gates["dynamic_parameter_count"])),
            raw_parameter_count=int(cast(Any, gates["raw_parameter_count"])),
            shape_parameter_count=int(
                cast(Any, gates["shape_parameter_count"])
            ),
            shape_residual_scale=float(
                cast(Any, gates["shape_residual_scale"])
            ),
            learning_rate=float(cast(Any, gates["learning_rate"])),
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
            "ablation_trial_id": ablation_trial_id,
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
            "schema_version": "tcn-dynamic-skip-raw-shape-residual-v26/v1",
            "run_id": str(config["run_id"]),
            "parents": {
                "seed7": seed7_identity,
                "confirmation": confirmation_identity,
                "ablation": ablation_identity,
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
