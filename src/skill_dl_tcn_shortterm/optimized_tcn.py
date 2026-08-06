"""Portable, frozen optimized TCN profile admitted by the V40 research gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .baselines import _summarize_predictions
from .experiment import ContractError
from .neural import HORIZONS, NeuralResult, _label_matrices
from .training_data import LazyWindowDataset, build_fold_protocols
from .tuning import (
    TCNTuningTrial,
    build_tcn_trial_model,
    predict_tcn_trial,
    run_tcn_validation_sweep,
)

V40_PORTABLE_PROFILE = "v40-portable"
V40_PORTABLE_MODEL = "optimized-tcn-v40-portable"
V40_PORTABLE_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class OptimizedTCNProfile:
    """Resolved portable profile shared by training and performance evidence."""

    name: str
    model_name: str
    channels: int
    kernel_size: int
    dilations: tuple[int, ...]
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    torch_threads: int
    dynamic_skip_hidden: int
    dynamic_skip_scale: float


def resolve_optimized_tcn_profile(
    profile: str = V40_PORTABLE_PROFILE,
    *,
    learning_rate: float = 0.003,
    batch_size: int = 128,
    epochs: int = 8,
    torch_threads: int = 8,
) -> OptimizedTCNProfile:
    """Resolve the only portable profile and fail closed on contract drift."""

    if profile != V40_PORTABLE_PROFILE:
        raise ContractError("optimized_tcn.profile must equal v40-portable")
    if learning_rate <= 0:
        raise ContractError("optimized_tcn.learning_rate must be positive")
    if batch_size <= 0 or epochs <= 0 or torch_threads <= 0:
        raise ContractError(
            "optimized_tcn batch_size, epochs, and torch_threads must be positive"
        )
    return OptimizedTCNProfile(
        name=profile,
        model_name=V40_PORTABLE_MODEL,
        channels=16,
        kernel_size=3,
        dilations=V40_PORTABLE_DILATIONS,
        dropout=0.0,
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        epochs=int(epochs),
        torch_threads=int(torch_threads),
        dynamic_skip_hidden=4,
        dynamic_skip_scale=1.0,
    )


def optimized_tcn_trial(profile: OptimizedTCNProfile) -> TCNTuningTrial:
    """Create the frozen trial used by both the CLI and benchmark factory."""

    return TCNTuningTrial(
        trial_id=profile.model_name,
        model_kind="dynamic_horizon_skip",
        channels=profile.channels,
        kernel_size=profile.kernel_size,
        dilations=profile.dilations,
        dropout=profile.dropout,
        learning_rate=profile.learning_rate,
        batch_size=profile.batch_size,
        strategy="smooth_l1",
        padding_mode="chomp",
        dynamic_skip_hidden=profile.dynamic_skip_hidden,
        dynamic_skip_scale=profile.dynamic_skip_scale,
        date_batch_order="fixed_once",
    )


def build_optimized_tcn_model(
    *,
    feature_count: int,
    input_steps: int,
    profile: OptimizedTCNProfile,
) -> torch.nn.Module:
    """Build the exact portable optimized model from the canonical TCN factory."""

    return build_tcn_trial_model(
        optimized_tcn_trial(profile),
        feature_count=feature_count,
        input_steps=input_steps,
    )


def run_optimized_tcn(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    seed: int,
    profile: OptimizedTCNProfile,
    num_workers: int = 0,
) -> NeuralResult:
    """Train the frozen V40 portable TCN on ordinary validation only."""

    if len(features) != len(window_index):
        raise ContractError("features and window index sample counts must match")
    if profile.epochs < 2:
        raise ContractError("optimized_tcn training requires at least two epochs")
    if num_workers != 0:
        raise ContractError("optimized_tcn currently requires data_loader.num_workers=0")
    forbidden = set(
        split_manifest.loc[
            split_manifest["stage"].isin(["test", "sealed_holdout"]),
            "sample_position",
        ].astype(int)
    )
    tuning_manifest = split_manifest.loc[
        split_manifest["stage"].isin(["train", "validation", "test"])
    ].copy()
    trial = optimized_tcn_trial(profile)
    tuning = run_tcn_validation_sweep(
        features,
        window_index,
        labels,
        tuning_manifest,
        trials=(trial,),
        seed=seed,
        max_epochs=profile.epochs,
        patience=min(2, profile.epochs - 1),
        min_delta=0.0005,
        checkpoint_min_delta=0.0,
        torch_threads=profile.torch_threads,
        capture_epoch_states=True,
        disable_early_stopping=True,
    )

    targets, masks = _label_matrices(window_index, labels)
    protocols = build_fold_protocols(features, split_manifest)
    sample_by_position = window_index.set_index("sample_position")[
        "sample_id"
    ].to_dict()
    labels_by_key = labels.set_index(["sample_id", "horizon"])
    prediction_rows: list[dict[str, object]] = []
    for protocol in protocols:
        if forbidden.intersection(map(int, protocol.train_positions)):
            raise ContractError("optimized_tcn training accessed sealed/test samples")
        if forbidden.intersection(map(int, protocol.validation_positions)):
            raise ContractError("optimized_tcn validation accessed sealed/test samples")
        state_key = f"{trial.trial_id}-fold-{protocol.fold}"
        state = tuning.best_states.get(state_key)
        if state is None:
            raise ContractError(f"optimized_tcn checkpoint state missing: {state_key}")
        model = build_optimized_tcn_model(
            feature_count=int(features.shape[1]),
            input_steps=int(features.shape[2]),
            profile=profile,
        )
        model.load_state_dict(state, strict=True)
        validation_dataset = LazyWindowDataset(
            features,
            protocol.validation_positions,
            targets,
            masks,
            protocol.feature_mean,
            protocol.feature_std,
        )
        scores, positions = predict_tcn_trial(
            model, validation_dataset, batch_size=profile.batch_size
        )
        for row_position, sample_position in enumerate(positions):
            sample_id = sample_by_position[int(sample_position)]
            for column, horizon in enumerate(HORIZONS):
                key = (sample_id, horizon)
                if key not in labels_by_key.index or not bool(
                    labels_by_key.loc[key, "valid"]
                ):
                    continue
                label = labels_by_key.loc[key]
                prediction_rows.append(
                    {
                        "model": profile.model_name,
                        "fold": protocol.fold,
                        "stage": "validation",
                        "sample_id": sample_id,
                        "instrument_id": label["instrument_id"],
                        "signal_date": label["signal_date"],
                        "horizon": horizon,
                        "score": float(scores[row_position, column]),
                        "target": float(label["rank_target"]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metadata = tuning.leaderboard.copy()
    metadata["model"] = profile.model_name
    metadata["profile"] = profile.name
    metadata["seed"] = seed
    metadata["epochs"] = profile.epochs
    metadata["torch_threads"] = profile.torch_threads
    metadata["sealed_test_accessed"] = False
    return NeuralResult(
        predictions=predictions,
        metrics=_summarize_predictions(predictions, seed),
        training_metadata=metadata,
    )
