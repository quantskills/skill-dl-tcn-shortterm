"""Run the immutable v27 frozen-parent shape-residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.dynamic_multiscale import (  # noqa: E402
    evaluate_frozen_parent_shape_residual_multiseed,
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
    build_validation_rankic_plan,
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


PARENT_TRIAL_ID = "dynamic-horizon-skip-c16-chomp-smooth-h4-s1"


def _checkpoint_relative(seed: int, fold: int) -> Path:
    name = f"{PARENT_TRIAL_ID}-fold-{fold}.pt"
    if seed != 7:
        name = f"seed-{seed}-{name}"
    return Path("checkpoints") / name


def _load_frozen_parent_states(
    seed7_parent: Path,
    confirmation_parent: Path,
    candidate_trial_id: str,
) -> tuple[
    dict[int, dict[str, Mapping[str, torch.Tensor]]], pd.DataFrame
]:
    by_seed: dict[int, dict[str, Mapping[str, torch.Tensor]]] = {}
    manifest_rows: list[dict[str, object]] = []
    for seed in (7, 17, 27):
        artifact = seed7_parent if seed == 7 else confirmation_parent
        receipt_value = json.loads(
            (artifact / "receipt.json").read_text(encoding="utf-8")
        )
        if not isinstance(receipt_value, dict):
            raise ContractError("v27 parent receipt must contain an object")
        outputs = receipt_value.get("outputs")
        if not isinstance(outputs, dict):
            raise ContractError("v27 parent checkpoint hashes are missing")
        states: dict[str, Mapping[str, torch.Tensor]] = {}
        for fold in range(5):
            relative = _checkpoint_relative(seed, fold)
            checkpoint = artifact / relative
            relative_key = str(relative)
            expected_sha = outputs.get(relative_key)
            if not isinstance(expected_sha, str):
                raise ContractError(
                    f"v27 parent checkpoint receipt is missing {relative_key}"
                )
            observed_sha = _sha256(checkpoint)
            if observed_sha != expected_sha:
                raise ContractError(
                    f"v27 parent checkpoint hash drifted for seed {seed} fold {fold}"
                )
            value = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(value, dict) or not value:
                raise ContractError("v27 parent checkpoint state is invalid")
            state: dict[str, torch.Tensor] = {}
            for name, tensor in value.items():
                if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                    raise ContractError("v27 parent checkpoint tensor is invalid")
                state[name] = tensor.detach().cpu().clone()
            key = f"{candidate_trial_id}-fold-{fold}"
            states[key] = state
            manifest_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "candidate_checkpoint_key": key,
                    "parent_artifact": str(artifact),
                    "parent_checkpoint": relative_key,
                    "parent_checkpoint_sha256": observed_sha,
                    "parent_tensor_count": len(state),
                }
            )
        by_seed[seed] = states
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["seed", "fold"], ignore_index=True
    )
    if len(manifest) != 15 or manifest.duplicated(["seed", "fold"]).any():
        raise ContractError("v27 parent checkpoint coverage drifted")
    return by_seed, manifest


def _frozen_shape_diagnostics(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
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
                raise ContractError("v27 diagnostics require shape residual TCN")
            checkpoint_key = (
                f"seed-{seed}-{candidate.trial_id}-fold-{protocol.fold}"
            )
            try:
                model.load_state_dict(best_states[checkpoint_key])
            except KeyError as exc:
                raise ContractError("v27 candidate checkpoint is missing") from exc
            dataset = LazyWindowDataset(
                features,
                protocol.validation_positions,
                dummy_targets,
                dummy_masks,
                protocol.feature_mean,
                protocol.feature_std,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            full_scores: list[np.ndarray] = []
            raw_scores: list[np.ndarray] = []
            positions: list[np.ndarray] = []
            full_weights: list[np.ndarray] = []
            raw_weights: list[np.ndarray] = []
            weight_samples = 0
            model.eval()
            with torch.no_grad():
                for batch_features, _, _, batch_positions in loader:
                    full_scores.append(model(batch_features).cpu().numpy())
                    raw_scores.append(
                        model.forward_without_shape_residual(batch_features)
                        .cpu()
                        .numpy()
                    )
                    positions.append(batch_positions.numpy())
                    if weight_samples < 512:
                        sequences = model.encode_blocks(batch_features)
                        full_weights.append(
                            model.dynamic_skip_weights(sequences).cpu().numpy()
                        )
                        raw_weights.append(
                            model.dynamic_skip_weights_without_shape_residual(
                                sequences
                            )
                            .cpu()
                            .numpy()
                        )
                        weight_samples += len(batch_features)
            if not full_scores or not full_weights:
                raise ContractError("v27 diagnostics found no validation samples")
            observed_positions = np.concatenate(positions)
            observed_full_scores = np.concatenate(full_scores)
            observed_raw_scores = np.concatenate(raw_scores)
            validation_plan = build_validation_rankic_plan(
                protocol.validation_positions, window_index, labels
            )
            full_rankic = validation_plan.evaluate(
                observed_full_scores, observed_positions
            )
            raw_rankic = validation_plan.evaluate(
                observed_raw_scores, observed_positions
            )
            observed_full_weights = np.concatenate(full_weights)[:512]
            observed_raw_weights = np.concatenate(raw_weights)[:512]
            rows.append(
                {
                    "trial_id": candidate.trial_id,
                    "seed": seed,
                    "fold": protocol.fold,
                    "sample_count": len(observed_positions),
                    "full_mean_daily_rankic": full_rankic.mean_daily_rankic,
                    "raw_only_mean_daily_rankic": raw_rankic.mean_daily_rankic,
                    "raw_only_rankic_1d": raw_rankic.rankic_by_horizon.get(
                        1, float("nan")
                    ),
                    "raw_only_rankic_2d": raw_rankic.rankic_by_horizon.get(
                        2, float("nan")
                    ),
                    "raw_only_rankic_3d": raw_rankic.rankic_by_horizon.get(
                        3, float("nan")
                    ),
                    "raw_only_rankic_5d": raw_rankic.rankic_by_horizon.get(
                        5, float("nan")
                    ),
                    "block_weight_variation": float(
                        np.std(observed_full_weights, axis=0).max()
                    ),
                    "shape_residual_weight_effect_max": float(
                        np.abs(
                            observed_full_weights - observed_raw_weights
                        ).max()
                    ),
                    "prediction_effect_max": float(
                        np.abs(observed_full_scores - observed_raw_scores).max()
                    ),
                    "simplex_error_max": float(
                        np.abs(
                            observed_full_weights.sum(axis=2) - 1.0
                        ).max()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "fold"], ignore_index=True)


def _history_rows(
    artifact: Path, trial_id: str, *, label: str
) -> pd.DataFrame:
    rows = pd.read_parquet(artifact / "tcn-leaderboard.parquet")
    rows = rows.loc[rows["trial_id"].astype(str).eq(trial_id)].copy()
    expected = {(seed, fold) for seed in (7, 17, 27) for fold in range(5)}
    observed = {
        (int(cast(Any, row.seed)), int(cast(Any, row.fold)))
        for row in rows.itertuples(index=False)
    }
    if observed != expected or rows.duplicated(["trial_id", "seed", "fold"]).any():
        raise ContractError(f"v27 {label} historical coverage drifted")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the immutable v27 frozen-parent shape residual probe"
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
            raise ContractError("v27 refuses to overwrite experiment artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v27 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v27":
            raise ContractError("v27 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v27 config contains a secret-like key")
        if config.get("precision") != "float32":
            raise ContractError("v27 precision must remain float32")
        seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["seeds"])
        )
        if seeds != (7, 17, 27):
            raise ContractError("v27 seeds must be exactly 7, 17 and 27")
        if cast(list[object], config["folds"]) != [0, 1, 2, 3, 4]:
            raise ContractError("v27 folds must be exactly 0 through 4")

        expected_hashes_value = config.get("source_sha256")
        if not isinstance(expected_hashes_value, dict):
            raise ContractError("v27 source identities are missing")
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
            raise ContractError("v27 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if observed_hashes != expected_hashes:
            raise ContractError("v27 source SHA-256 identity drifted")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v27 rejects sealed split rows")
        observed_stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(observed_stages - {"train", "validation", "purged"}):
            raise ContractError(
                "v27 split contains forbidden stages: " + ", ".join(unknown)
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
        if len(trials) != 1 or trials[0].trial_id != candidate_trial_id:
            raise ContractError("v27 must train exactly one candidate")
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
            raise ContractError("v27 frozen shape-only contract drifted")

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
                min_delta=float(cast(Any, config["min_delta"])),
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
        observed_scores = leaderboard.set_index(["seed", "fold"])[
            "best_mean_daily_rankic"
        ]
        diagnostic_scores = diagnostics.set_index(["seed", "fold"])[
            "full_mean_daily_rankic"
        ]
        if float(np.max(np.abs(observed_scores - diagnostic_scores))) > 1e-12:
            raise ContractError("v27 diagnostic RankIC drifted from leaderboard")

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
        historical = pd.concat(
            [historical, v25_rows, v26_rows], ignore_index=True
        )
        comparison = build_tcn_lstm_comparison(leaderboard, lstm)
        gates = cast(dict[str, object], config["gates"])
        decision = evaluate_frozen_parent_shape_residual_multiseed(
            leaderboard,
            historical,
            diagnostics,
            comparison,
            control_trial_id=control_trial_id,
            parent_candidate_trial_id=parent_candidate_trial_id,
            v25_trial_id=v25_trial_id,
            v26_trial_id=v26_trial_id,
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
            "schema_version": "tcn-frozen-parent-shape-residual-v27/v1",
            "run_id": str(config["run_id"]),
            "parents": {
                "seed7": seed7_identity,
                "confirmation": confirmation_identity,
                "v25": v25_identity,
                "v26": v26_identity,
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
