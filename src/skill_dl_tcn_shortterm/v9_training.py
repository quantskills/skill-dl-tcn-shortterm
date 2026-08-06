"""Bounded TCN-v9 adapter over the project's canonical validation trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import pandas as pd
import torch

from .experiment import ContractError
from .tcn_lite import lite_receptive_field
from .tuning import TCNTuningResult, TCNTuningTrial, run_tcn_validation_sweep
from .v9_selection import Seed7Trial


V9ParentKind = Literal["lite", "horizon_skip"]


@dataclass(frozen=True)
class V9TrainingRequest:
    features: Any
    window_index: pd.DataFrame
    labels: pd.DataFrame
    split_manifest: pd.DataFrame
    channels: int
    kernel_size: int
    dilations: tuple[int, ...]
    dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    torch_threads: int
    protocol_identities: Mapping[str, str]
    artifact_paths: Mapping[str, Path] | None = None


@dataclass(frozen=True)
class V9TrainingResult:
    epoch_history: pd.DataFrame
    leaderboard: pd.DataFrame
    best_states: dict[str, dict[str, torch.Tensor]]


def _validate_training_request(
    request: V9TrainingRequest,
    trials: Sequence[Seed7Trial],
) -> None:
    if len(request.features) != len(request.window_index):
        raise ContractError("v9 features and window index counts differ")
    required_split = {"fold", "sample_position", "stage", "sealed"}
    if missing := sorted(required_split.difference(request.split_manifest.columns)):
        raise ContractError(f"v9 training split missing columns: {', '.join(missing)}")
    if request.split_manifest["sealed"].astype(bool).any():
        raise ContractError("v9 training rejects sealed rows")
    if not set(request.split_manifest["stage"].astype(str)) <= {
        "train",
        "validation",
    }:
        raise ContractError("v9 training accepts only ordinary train/validation rows")
    if set(request.split_manifest["fold"].astype(int)) != {0, 1, 2, 3, 4}:
        raise ContractError("v9 formal training requires five folds")
    if request.max_epochs <= 0 or request.max_epochs > 8:
        raise ContractError("v9 training max_epochs must be between 1 and 8")
    if request.patience <= 0 or request.patience >= request.max_epochs:
        raise ContractError("v9 training patience must be smaller than max_epochs")
    if request.min_delta < 0 or request.channels <= 0 or request.batch_size <= 0:
        raise ContractError("v9 training numeric protocol is invalid")
    if lite_receptive_field(
        kernel_size=request.kernel_size,
        dilations=request.dilations,
    ) < int(request.features.shape[2]):
        raise ContractError("v9 training receptive field is smaller than input steps")
    required_identities = {"data", "fold_manifest", "evaluation"}
    if not required_identities <= set(request.protocol_identities):
        raise ContractError("v9 training protocol identities are incomplete")
    if any(trial.fold_ids != (0, 1, 2, 3, 4) for trial in trials):
        raise ContractError("v9 registered trial fold protocol drifted")
    if any(
        (
            trial.max_epochs,
            trial.patience,
            trial.min_delta,
        )
        != (request.max_epochs, request.patience, request.min_delta)
        for trial in trials
    ):
        raise ContractError("v9 registered trial early-stopping protocol drifted")


def _trial(
    request: V9TrainingRequest,
    *,
    trial_id: str,
    model_kind: V9ParentKind,
    strategy: Literal["smooth_l1", "rank_objective", "pcgrad"],
    infra_enabled: bool,
) -> TCNTuningTrial:
    return TCNTuningTrial(
        trial_id=trial_id,
        channels=request.channels,
        kernel_size=request.kernel_size,
        dilations=request.dilations,
        dropout=request.dropout,
        learning_rate=request.learning_rate,
        batch_size=request.batch_size,
        model_kind=model_kind,
        strategy=strategy,
        padding_mode="chomp" if infra_enabled else "explicit",
    )


def _run(
    request: V9TrainingRequest,
    trials: Sequence[TCNTuningTrial],
    *,
    seed: int,
) -> TCNTuningResult:
    return run_tcn_validation_sweep(
        request.features,
        request.window_index,
        request.labels,
        request.split_manifest,
        trials=tuple(trials),
        seed=seed,
        max_epochs=request.max_epochs,
        patience=request.patience,
        min_delta=request.min_delta,
        torch_threads=request.torch_threads,
        protocol_identities=request.protocol_identities,
    )


def _select_parent(
    phase_one: pd.DataFrame,
    *,
    control_trial_id: str,
    horizon_trial_id: str | None,
) -> V9ParentKind | None:
    summaries = phase_one.groupby("trial_id", observed=True).agg(
        mean_rankic=("best_mean_daily_rankic", "mean"),
        median_throughput=("samples_per_second", "median"),
    )
    control = summaries.loc[control_trial_id]
    control_eligible = float(cast(float, control["median_throughput"])) >= 5_000.0
    if horizon_trial_id is not None:
        horizon = summaries.loc[horizon_trial_id]
        if (
            float(cast(float, horizon["mean_rankic"]))
            >= float(cast(float, control["mean_rankic"]))
            and float(cast(float, horizon["median_throughput"])) >= 5_000.0
        ):
            return "horizon_skip"
    return "lite" if control_eligible else None


def run_v9_candidate_sweep(
    request: V9TrainingRequest,
    *,
    registered_trials: Sequence[Seed7Trial],
    seed: int,
    control_trial_id: str = "lite-c16-no-dropout",
    frozen_parent_kind: str | None = None,
) -> V9TrainingResult:
    """Run control/skip selection, then train objectives on the selected parent."""

    _validate_training_request(request, registered_trials)
    global_infra = any(trial.infra_enabled for trial in registered_trials)
    by_kind = {trial.kind: trial for trial in registered_trials}
    horizon = by_kind.get("horizon_skip")
    phase_one_trials = [
        _trial(
            request,
            trial_id=control_trial_id,
            model_kind="lite",
            strategy="smooth_l1",
            infra_enabled=global_infra,
        )
    ]
    if horizon is not None:
        phase_one_trials.append(
            _trial(
                request,
                trial_id=horizon.trial_id,
                model_kind="horizon_skip",
                strategy="smooth_l1",
                infra_enabled=global_infra,
            )
        )
    phase_one = _run(request, phase_one_trials, seed=seed)
    parent_kind: V9ParentKind | None
    if frozen_parent_kind is not None:
        if frozen_parent_kind not in {"lite", "horizon_skip"}:
            raise ContractError("v9 confirmation parent model kind is unsupported")
        parent_kind = cast(V9ParentKind, frozen_parent_kind)
    else:
        parent_kind = _select_parent(
            phase_one.leaderboard,
            control_trial_id=control_trial_id,
            horizon_trial_id=None if horizon is None else horizon.trial_id,
        )

    phase_two_trials = [
        _trial(
            request,
            trial_id=trial.trial_id,
            model_kind=parent_kind,
            strategy=cast(
                Literal["rank_objective", "pcgrad"],
                trial.kind,
            ),
            infra_enabled=global_infra,
        )
        for trial in registered_trials
        if trial.kind in {"rank_objective", "pcgrad"} and parent_kind is not None
    ]
    parts = [phase_one]
    if phase_two_trials:
        parts.append(_run(request, phase_two_trials, seed=seed))

    history = pd.concat(
        [part.epoch_history for part in parts],
        ignore_index=True,
    )
    leaderboard = pd.concat(
        [part.leaderboard for part in parts],
        ignore_index=True,
    )
    best_states = {
        key: state
        for part in parts
        for key, state in part.best_states.items()
    }
    return V9TrainingResult(history, leaderboard, best_states)
