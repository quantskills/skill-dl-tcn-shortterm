"""Run immutable v18-v20 stock-conditioned TCN readout screens."""

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
from skill_dl_tcn_shortterm.dynamic_readout import (  # noqa: E402
    evaluate_dynamic_lr_seed7,
    evaluate_dynamic_readout_seed7,
    finalize_dynamic_lr_seed7,
    finalize_dynamic_readout_seed7,
)
from skill_dl_tcn_shortterm.dynamic_multiscale import (  # noqa: E402
    evaluate_dynamic_multiscale_seed7,
    finalize_dynamic_multiscale_seed7,
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
    DynamicHorizonSkipTCN,
    DynamicTemporalContextTCN,
)


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


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else ROOT / path).resolve()


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


def _validate_parent(config: dict[str, object]) -> tuple[dict[str, object], Path]:
    parent = _resolve_project_path(config["parent_artifact"])
    receipt_path = parent / "receipt.json"
    selection_path = parent / "selection.json"
    if not receipt_path.is_file() or not selection_path.is_file():
        raise ContractError("v18 parent artifact is incomplete")
    receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
    selection_value = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(receipt_value, dict) or not isinstance(selection_value, dict):
        raise ContractError("v18 parent evidence must contain objects")
    receipt = cast(dict[str, object], receipt_value)
    selection = cast(dict[str, object], selection_value)
    expected_receipt_id = str(config["parent_receipt_id"])
    expected_status = str(config["parent_selection_status"])
    if receipt.get("receipt_id") != expected_receipt_id:
        raise ContractError("v18 parent receipt identity drifted")
    if receipt.get("sealed_test_accessed") is not False:
        raise ContractError("v18 parent must not have accessed sealed test")
    if selection.get("status") != expected_status:
        raise ContractError("v18 parent selection status drifted")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ContractError("v18 parent receipt outputs are missing")
    for relative, expected_hash in outputs.items():
        path = parent / str(relative)
        if not path.is_file() or _sha256(path) != str(expected_hash):
            raise ContractError("v18 parent output hash drifted")
    for required in ("lstm-measurements.parquet", "lstm-environment.json"):
        if not (parent / required).is_file():
            raise ContractError("v18 parent fixed LSTM evidence is missing")
    if config.get("protocol_version") == "v19" and not (
        parent / "attention-diagnostics.parquet"
    ).is_file():
        raise ContractError("v19 parent attention diagnostics are missing")
    return (
        {
            "path": str(parent),
            "receipt_id": expected_receipt_id,
            "selection_status": expected_status,
        },
        parent,
    )


def _attention_diagnostics(
    features: np.ndarray,
    split_manifest: pd.DataFrame,
    candidate: TCNTuningTrial,
    best_states: dict[str, dict[str, torch.Tensor]],
    *,
    batch_size: int,
) -> pd.DataFrame:
    protocols = build_fold_protocols(features, split_manifest)
    dummy_targets = np.zeros((len(features), 4), dtype="float32")
    dummy_masks = np.ones((len(features), 4), dtype="bool")
    rows: list[dict[str, object]] = []
    for protocol in protocols:
        model = build_tcn_trial_model(
            candidate,
            feature_count=int(features.shape[1]),
            input_steps=int(features.shape[2]),
        )
        if not isinstance(model, DynamicTemporalContextTCN):
            raise ContractError("v18 attention diagnostics require the dynamic TCN")
        checkpoint_key = f"{candidate.trial_id}-fold-{protocol.fold}"
        try:
            model.load_state_dict(best_states[checkpoint_key])
        except KeyError as exc:
            raise ContractError("v18 dynamic checkpoint is missing") from exc
        dataset = LazyWindowDataset(
            features,
            protocol.validation_positions,
            dummy_targets,
            dummy_masks,
            protocol.feature_mean,
            protocol.feature_std,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        day_batches: list[np.ndarray] = []
        intraday_batches: list[np.ndarray] = []
        model.eval()
        observed = 0
        with torch.no_grad():
            for batch_features, _, _, _ in loader:
                sequence = model.encode_sequence(batch_features)
                day_weights, intraday_weights = model.dynamic_weights(sequence)
                day_batches.append(day_weights.cpu().numpy())
                intraday_batches.append(intraday_weights.cpu().numpy())
                observed += len(batch_features)
                if observed >= 512:
                    break
        if not day_batches or not intraday_batches:
            raise ContractError("v18 attention diagnostics found no validation samples")
        day = np.concatenate(day_batches, axis=0)[:512]
        intraday = np.concatenate(intraday_batches, axis=0)[:512]
        rows.append(
            {
                "trial_id": candidate.trial_id,
                "fold": protocol.fold,
                "sample_count": int(len(day)),
                "day_weight_variation": float(np.std(day, axis=0).max()),
                "intraday_weight_variation": float(
                    np.std(intraday, axis=0).max()
                ),
                "day_simplex_error_max": float(
                    np.abs(day.sum(axis=2) - 1.0).max()
                ),
                "intraday_simplex_error_max": float(
                    np.abs(intraday.sum(axis=2) - 1.0).max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("fold", ignore_index=True)


def _multiscale_diagnostics(
    features: np.ndarray,
    split_manifest: pd.DataFrame,
    candidate: TCNTuningTrial,
    best_states: dict[str, dict[str, torch.Tensor]],
    *,
    batch_size: int,
) -> pd.DataFrame:
    protocols = build_fold_protocols(features, split_manifest)
    dummy_targets = np.zeros((len(features), 4), dtype="float32")
    dummy_masks = np.ones((len(features), 4), dtype="bool")
    rows: list[dict[str, object]] = []
    for protocol in protocols:
        model = build_tcn_trial_model(
            candidate,
            feature_count=int(features.shape[1]),
            input_steps=int(features.shape[2]),
        )
        if not isinstance(model, DynamicHorizonSkipTCN):
            raise ContractError("v20 diagnostics require the dynamic horizon-skip TCN")
        checkpoint_key = f"{candidate.trial_id}-fold-{protocol.fold}"
        try:
            model.load_state_dict(best_states[checkpoint_key])
        except KeyError as exc:
            raise ContractError("v20 dynamic checkpoint is missing") from exc
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
        model.eval()
        observed = 0
        with torch.no_grad():
            for batch_features, _, _, _ in loader:
                blocks = model.encode_blocks(batch_features)
                weights = model.dynamic_skip_weights(blocks)
                weight_batches.append(weights.cpu().numpy())
                observed += len(batch_features)
                if observed >= 512:
                    break
        if not weight_batches:
            raise ContractError("v20 diagnostics found no validation samples")
        observed_weights = np.concatenate(weight_batches, axis=0)[:512]
        rows.append(
            {
                "trial_id": candidate.trial_id,
                "fold": protocol.fold,
                "sample_count": int(len(observed_weights)),
                "block_weight_variation": float(
                    np.std(observed_weights, axis=0).max()
                ),
                "simplex_error_max": float(
                    np.abs(observed_weights.sum(axis=2) - 1.0).max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("fold", ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded seed-7 stock-conditioned dynamic TCN validation"
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
            raise ContractError("v18 dynamic-readout screen refuses to overwrite artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v18 dynamic-readout config must contain an object")
        config = cast(dict[str, object], config_value)
        protocol_version = str(config.get("protocol_version", "v18"))
        if protocol_version not in {"v18", "v19", "v20"}:
            raise ContractError("dynamic-readout protocol version is unsupported")
        if _contains_secret_key(config):
            raise ContractError("v18 dynamic-readout config contains a secret-like key")
        if config.get("precision") != "float32" or int(cast(Any, config["seed"])) != 7:
            raise ContractError("v18 dynamic-readout screen requires seed 7 float32")
        parent, parent_path = _validate_parent(config)

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
            raise ContractError("v18 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        expected_hashes = config.get("source_sha256")
        if not isinstance(expected_hashes, dict) or observed_hashes != {
            str(key): str(value) for key, value in expected_hashes.items()
        }:
            raise ContractError("v18 source SHA-256 identity drifted")

        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v18 dynamic-readout screen rejects sealed split rows")
        allowed_stages = {"train", "validation", "purged"}
        observed_stages = {
            str(value) for value in raw_split["stage"].astype(str).tolist()
        }
        if unknown := sorted(observed_stages - allowed_stages):
            raise ContractError("v18 split contains forbidden stages: " + ", ".join(unknown))
        folds = tuple(int(cast(Any, value)) for value in cast(list[object], config["folds"]))
        if folds != tuple(range(5)):
            raise ContractError("v18 folds must be exactly 0 through 4")
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(folds)
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        control_trial_id = str(config["control_trial_id"])
        candidate_trial_id = str(config["candidate_trial_id"])
        if len(trials) != 2 or {trial.trial_id for trial in trials} != {
            control_trial_id,
            candidate_trial_id,
        }:
            raise ContractError("v18 trial identities drifted")
        by_trial = {trial.trial_id: trial for trial in trials}
        control = by_trial[control_trial_id]
        candidate = by_trial[candidate_trial_id]
        final: Any
        if protocol_version == "v20":
            if control.model_kind != "horizon_skip" or (
                candidate.model_kind != "dynamic_horizon_skip"
                or candidate.dynamic_skip_hidden != 4
                or candidate.dynamic_skip_scale != 1.0
            ):
                raise ContractError("v20 dynamic multiscale mechanism drifted")
        else:
            if control.model_kind != "temporal_context" or (
                candidate.model_kind != "dynamic_temporal_context"
                or candidate.dynamic_attention_hidden != 4
                or candidate.dynamic_attention_scale != 1.0
            ):
                raise ContractError("v18 dynamic-readout mechanism drifted")
            if protocol_version == "v18" and (
                candidate.dynamic_attention_learning_rate is not None
            ):
                raise ContractError("v18 dynamic-readout learning rate contract drifted")
            if protocol_version == "v19" and (
                candidate.dynamic_attention_learning_rate != 0.01
            ):
                raise ContractError("v19 dynamic attention learning rate must be 0.01")
        immutable_fields = (
            "channels",
            "kernel_size",
            "dilations",
            "dropout",
            "learning_rate",
            "batch_size",
            "strategy",
            "padding_mode",
            "bars_per_day",
        )
        if any(getattr(control, field) != getattr(candidate, field) for field in immutable_fields):
            raise ContractError("v18 control and candidate contracts differ outside readout")

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
            protocol_identities={
                "data": observed_hashes["features"],
                "fold_manifest": observed_hashes["split_manifest"],
                "evaluation": observed_hashes["labels"],
            },
        )
        attention = (
            _multiscale_diagnostics(
                features,
                split_manifest,
                candidate,
                tuning.best_states,
                batch_size=candidate.batch_size,
            )
            if protocol_version == "v20"
            else _attention_diagnostics(
                features,
                split_manifest,
                candidate,
                tuning.best_states,
                batch_size=candidate.batch_size,
            )
        )
        gates = cast(dict[str, object], config["gates"])
        min_mean_rankic = float(cast(Any, gates["min_mean_rankic"]))
        min_mean_rankic_delta = float(
            cast(Any, gates["min_mean_rankic_delta"])
        )
        min_positive_folds = int(cast(Any, gates["min_positive_folds"]))
        min_nondegrading_folds = int(
            cast(Any, gates["min_nondegrading_folds"])
        )
        min_horizon_delta_1d = float(
            cast(Any, gates["min_horizon_delta_1d"])
        )
        min_horizon_delta_2d = float(
            cast(Any, gates["min_horizon_delta_2d"])
        )
        min_horizon_delta_3d = float(
            cast(Any, gates["min_horizon_delta_3d"])
        )
        min_horizon_delta_5d = float(
            cast(Any, gates["min_horizon_delta_5d"])
        )
        min_median_samples_per_second = float(
            cast(Any, gates["min_median_samples_per_second"])
        )
        min_dynamic_weight_variation = float(
            cast(Any, gates.get("min_dynamic_weight_variation", 1e-6))
        )
        control_parameter_count = int(
            cast(Any, gates["control_parameter_count"])
        )
        candidate_parameter_count = int(
            cast(Any, gates["candidate_parameter_count"])
        )
        effect: Any
        if protocol_version == "v20":
            effect = evaluate_dynamic_multiscale_seed7(
                tuning.leaderboard,
                attention,
                control_trial_id=control_trial_id,
                candidate_trial_id=candidate_trial_id,
                min_mean_rankic=min_mean_rankic,
                min_mean_rankic_delta=min_mean_rankic_delta,
                min_positive_folds=min_positive_folds,
                min_nondegrading_folds=min_nondegrading_folds,
                min_horizon_delta_1d=min_horizon_delta_1d,
                min_horizon_delta_2d=min_horizon_delta_2d,
                min_horizon_delta_3d=min_horizon_delta_3d,
                min_horizon_delta_5d=min_horizon_delta_5d,
                min_median_samples_per_second=min_median_samples_per_second,
                min_dynamic_skip_output_weight_l2=float(
                    cast(Any, gates["min_dynamic_skip_output_weight_l2"])
                ),
                min_block_weight_variation=float(
                    cast(Any, gates["min_block_weight_variation"])
                ),
                max_simplex_error=float(cast(Any, gates["max_simplex_error"])),
                control_parameter_count=control_parameter_count,
                candidate_parameter_count=candidate_parameter_count,
                dynamic_parameter_count=int(
                    cast(Any, gates["dynamic_parameter_count"])
                ),
            )
        elif protocol_version == "v19":
            parent_attention = pd.read_parquet(
                parent_path / "attention-diagnostics.parquet"
            )
            parent_variation = parent_attention.set_index("fold")[[
                "day_weight_variation",
                "intraday_weight_variation",
            ]].max(axis=1)
            current_variation = attention.set_index("fold")[[
                "day_weight_variation",
                "intraday_weight_variation",
            ]].max(axis=1)
            attention["parent_weight_variation"] = attention["fold"].map(
                parent_variation
            )
            attention["parent_variation_ratio"] = attention["fold"].map(
                current_variation / parent_variation
            )
            effect = evaluate_dynamic_lr_seed7(
                tuning.leaderboard,
                attention,
                parent_attention,
                control_trial_id=control_trial_id,
                candidate_trial_id=candidate_trial_id,
                min_mean_rankic=min_mean_rankic,
                min_mean_rankic_delta=min_mean_rankic_delta,
                min_positive_folds=min_positive_folds,
                min_nondegrading_folds=min_nondegrading_folds,
                min_horizon_delta_1d=min_horizon_delta_1d,
                min_horizon_delta_2d=min_horizon_delta_2d,
                min_horizon_delta_3d=min_horizon_delta_3d,
                min_horizon_delta_5d=min_horizon_delta_5d,
                min_median_samples_per_second=min_median_samples_per_second,
                min_dynamic_weight_variation=min_dynamic_weight_variation,
                control_parameter_count=control_parameter_count,
                candidate_parameter_count=candidate_parameter_count,
                min_dynamic_attention_output_weight_l2=float(
                    cast(Any, gates["min_dynamic_attention_output_weight_l2"])
                ),
                min_parent_variation_ratio=float(
                    cast(Any, gates["min_parent_variation_ratio"])
                ),
                dynamic_parameter_count=int(
                    cast(Any, gates["dynamic_parameter_count"])
                ),
            )
        else:
            effect = evaluate_dynamic_readout_seed7(
                tuning.leaderboard,
                attention,
                control_trial_id=control_trial_id,
                candidate_trial_id=candidate_trial_id,
                min_mean_rankic=min_mean_rankic,
                min_mean_rankic_delta=min_mean_rankic_delta,
                min_positive_folds=min_positive_folds,
                min_nondegrading_folds=min_nondegrading_folds,
                min_horizon_delta_1d=min_horizon_delta_1d,
                min_horizon_delta_2d=min_horizon_delta_2d,
                min_horizon_delta_3d=min_horizon_delta_3d,
                min_horizon_delta_5d=min_horizon_delta_5d,
                min_median_samples_per_second=min_median_samples_per_second,
                min_dynamic_weight_variation=min_dynamic_weight_variation,
                control_parameter_count=control_parameter_count,
                candidate_parameter_count=candidate_parameter_count,
                min_dynamic_attention_output_l2=float(
                    cast(Any, gates["min_dynamic_attention_output_l2"])
                ),
            )

        lstm_measurements = pd.read_parquet(parent_path / "lstm-measurements.parquet")
        lstm_environment_value = json.loads(
            (parent_path / "lstm-environment.json").read_text(encoding="utf-8")
        )
        if not isinstance(lstm_environment_value, dict):
            raise ContractError("v18 fixed LSTM environment must contain an object")
        comparison = build_tcn_lstm_comparison(
            tuning.leaderboard.loc[
                tuning.leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
            ],
            lstm_measurements,
        )
        min_model_step_speed_ratio = float(
            cast(Any, gates["min_model_step_speed_ratio"])
        )
        min_end_to_end_speed_ratio = float(
            cast(Any, gates["min_end_to_end_speed_ratio"])
        )
        if protocol_version == "v20":
            final = finalize_dynamic_multiscale_seed7(
                effect,
                comparison,
                min_model_step_speed_ratio=min_model_step_speed_ratio,
                min_end_to_end_speed_ratio=min_end_to_end_speed_ratio,
            )
        elif protocol_version == "v19":
            final = finalize_dynamic_lr_seed7(
                effect,
                comparison,
                min_model_step_speed_ratio=min_model_step_speed_ratio,
                min_end_to_end_speed_ratio=min_end_to_end_speed_ratio,
            )
        else:
            final = finalize_dynamic_readout_seed7(
                effect,
                comparison,
                min_model_step_speed_ratio=min_model_step_speed_ratio,
                min_end_to_end_speed_ratio=min_end_to_end_speed_ratio,
            )

        temporary.mkdir(parents=True)
        tuning.epoch_history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        tuning.leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        effect.summary.to_parquet(temporary / "tcn-summary.parquet", index=False)
        effect.horizon_summary.to_parquet(temporary / "horizon-summary.parquet", index=False)
        attention.to_parquet(temporary / "attention-diagnostics.parquet", index=False)
        lstm_measurements.to_parquet(temporary / "lstm-measurements.parquet", index=False)
        _write_json(temporary / "lstm-environment.json", lstm_environment_value)
        comparison_output: dict[str, object] = {
            **comparison,
            "tcn_trial_id": candidate_trial_id,
            "benchmark_feature_parity": True,
            "benchmark_note": (
                "verified frozen parent stock-only LSTM; identical data folds and seed"
            ),
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
            "schema_version": {
                "v18": "tcn-stock-conditioned-dynamic-readout-v18/v1",
                "v19": "tcn-dynamic-readout-learning-rate-v19/v1",
                "v20": "tcn-stock-conditioned-multiscale-v20/v1",
            }[protocol_version],
            "run_id": str(config["run_id"]),
            "parent": parent,
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "source_config": {"path": str(config_path), "sha256": _sha256(config_path)},
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
