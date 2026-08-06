"""Validation-only TCN tuning with explicit scale gates."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from torch import nn
from torch.utils.data import DataLoader

from .experiment import ContractError
from .market_context import (
    ContextualLazyWindowDataset,
    PITMarketContext,
    fit_market_context_standardizer,
)
from .neural import HORIZONS, _label_matrices
from .runtime import torch_thread_scope
from .stability_ema import EpochParameterAverage, ParameterEMA
from .tcn import BaiTCN, receptive_field
from .tcn_lite import TCNLite, lite_receptive_field
from .training_data import (
    LazyWindowDataset,
    build_fold_protocols,
    masked_smooth_l1,
)
from .v9_gradients import PCGradBackwardReceipt, pcgrad_backward
from .v9_infra import TCNLiteChomp
from .v9_objective import (
    DateGroupedBatchSampler,
    date_horizon_equal_smooth_l1,
    date_horizon_group_counts,
    mixed_date_grouped_rank_loss,
    mixed_date_grouped_soft_rankic_loss,
    mixed_date_grouped_top_tail_loss,
    teacher_listwise_components,
)
from .v9_representation import (
    DecoupledResidualTemporalContextTCN,
    DynamicHorizonSkipTCN,
    DynamicTemporalContextTCN,
    HorizonSkipTCN,
    MarketConditionedTemporalContextTCN,
    ShapeResidualDynamicHorizonSkipTCN,
    SignedTemporalContextTCN,
    StabilizedResidualTemporalContextTCN,
    TemporalContextTCN,
)


@dataclass(frozen=True)
class TCNTuningTrial:
    """One immutable Bai TCN validation trial."""

    trial_id: str
    channels: int
    kernel_size: int
    dilations: tuple[int, ...]
    dropout: float
    learning_rate: float
    batch_size: int
    model_kind: Literal[
        "bai",
        "lite",
        "horizon_skip",
        "temporal_context",
        "signed_temporal_context",
        "stabilized_temporal_context",
        "decoupled_temporal_context",
        "market_conditioned_temporal_context",
        "dynamic_temporal_context",
        "dynamic_horizon_skip",
    ] = "bai"
    head_dropout: float = 0.0
    dropout_kind: Literal["element", "channel"] = "element"
    weight_decay: float = 0.0
    strategy: Literal[
        "smooth_l1",
        "grouped_smooth_l1",
        "rank_objective",
        "soft_rankic",
        "top_tail",
        "teacher_listwise",
        "pcgrad",
    ] = "smooth_l1"
    padding_mode: Literal["explicit", "chomp"] = "explicit"
    pcgrad_blocks: tuple[int, ...] | None = None
    pcgrad_horizons: tuple[int, ...] | None = None
    bars_per_day: int = 48
    soft_rankic_weight: float = 0.2
    soft_rank_temperature: float = 0.1
    top_tail_weight: float = 0.05
    top_tail_fraction: float = 0.1
    top_tail_temperature: float = 0.1
    teacher_listwise_gradient_ratio: float = 0.25
    teacher_listwise_temperature: float = 0.1
    residual_scale: float = 0.05
    adapter_learning_rate: float | None = None
    residual_learning_rate: float | None = None
    market_context_dim: int = 24
    market_context_hidden: int = 4
    market_gate_scale: float = 0.25
    dynamic_attention_hidden: int = 4
    dynamic_attention_scale: float = 1.0
    dynamic_attention_learning_rate: float | None = None
    dynamic_skip_hidden: int = 4
    dynamic_skip_scale: float = 1.0
    dynamic_skip_learning_rate: float | None = None
    dynamic_skip_warmup_epochs: int = 0
    dynamic_skip_token_normalization: Literal[
        "none", "layer_norm", "shape_log_rms"
    ] = "none"
    dynamic_skip_shape_residual: bool = False
    dynamic_skip_shape_residual_scale: float = 0.25
    dynamic_skip_frozen_parent: bool = False
    date_batch_order: Literal["fixed_once", "epoch_seeded"] = "fixed_once"
    grouped_smooth_l1_reduction: Literal[
        "label_mean", "date_horizon_mean"
    ] = "label_mean"
    ema_decay: float | None = None
    epoch_average_start: int | None = None
    teacher_blend_start_weight: float | None = None
    teacher_blend_end_weight: float = 0.0


@dataclass(frozen=True)
class _BackwardTrialResult:
    loss: torch.Tensor
    pcgrad_receipt: PCGradBackwardReceipt | None = None
    loss_group_count: int | None = None
    valid_label_count: int | None = None
    auxiliary_pair_count: int | None = None
    component_gradient_cosine: float | None = None
    teacher_gradient_ratio: float | None = None


@dataclass(frozen=True)
class ValidationSelectionState:
    """Best-checkpoint and material-patience state for validation selection."""

    best_score: float
    patience_anchor_score: float
    epochs_without_material_improvement: int
    has_checkpoint: bool


@dataclass(frozen=True)
class ValidationSelectionResult:
    """One immutable transition of the validation selection state machine."""

    state: ValidationSelectionState
    checkpoint_improved: bool
    patience_improved: bool


def advance_validation_selection(
    state: ValidationSelectionState,
    *,
    score: float,
    checkpoint_min_delta: float,
    patience_min_delta: float,
) -> ValidationSelectionResult:
    """Advance checkpoint and patience decisions without coupling their deltas."""

    deltas = np.asarray(
        [checkpoint_min_delta, patience_min_delta], dtype="float64"
    )
    if (
        not np.isfinite(deltas).all()
        or bool((deltas < 0).any())
        or checkpoint_min_delta > patience_min_delta
    ):
        raise ContractError(
            "checkpoint selection deltas must be finite, non-negative and ordered"
        )
    if state.epochs_without_material_improvement < 0:
        raise ContractError("validation selection patience count cannot be negative")
    if state.has_checkpoint and not np.isfinite(
        [state.best_score, state.patience_anchor_score]
    ).all():
        raise ContractError("validation selection checkpoint scores must be finite")
    if not np.isfinite(score):
        return ValidationSelectionResult(
            state=ValidationSelectionState(
                best_score=state.best_score,
                patience_anchor_score=state.patience_anchor_score,
                epochs_without_material_improvement=(
                    state.epochs_without_material_improvement + 1
                ),
                has_checkpoint=state.has_checkpoint,
            ),
            checkpoint_improved=False,
            patience_improved=False,
        )
    checkpoint_improved = bool(
        not state.has_checkpoint
        or score > state.best_score + checkpoint_min_delta
    )
    patience_improved = bool(
        not state.has_checkpoint
        or score > state.patience_anchor_score + patience_min_delta
    )
    return ValidationSelectionResult(
        state=ValidationSelectionState(
            best_score=score if checkpoint_improved else state.best_score,
            patience_anchor_score=(
                score if patience_improved else state.patience_anchor_score
            ),
            epochs_without_material_improvement=(
                0
                if patience_improved
                else state.epochs_without_material_improvement + 1
            ),
            has_checkpoint=state.has_checkpoint or checkpoint_improved,
        ),
        checkpoint_improved=checkpoint_improved,
        patience_improved=patience_improved,
    )


@dataclass(frozen=True)
class TCNOptimizerBundle:
    """Optimizer plus an auditable account of its parameter groups."""

    optimizer: torch.optim.Optimizer
    optimizer_name: str
    parameter_group_identity: str
    adapter_parameter_count: int
    residual_parameter_count: int
    dynamic_attention_parameter_count: int = 0
    dynamic_skip_parameter_count: int = 0


def dynamic_skip_learning_rate_for_epoch(
    trial: TCNTuningTrial, epoch: int
) -> float | None:
    """Return the public dynamic-skip learning rate for a one-based epoch."""

    if epoch < 1:
        raise ContractError("dynamic skip learning-rate epoch must start at one")
    target = trial.dynamic_skip_learning_rate
    if target is None:
        return None
    warmup_epochs = trial.dynamic_skip_warmup_epochs
    if warmup_epochs <= 0:
        return float(target)
    progress = min((epoch - 1) / warmup_epochs, 1.0)
    return float(trial.learning_rate + (target - trial.learning_rate) * progress)


def apply_tcn_epoch_learning_rates(
    bundle: TCNOptimizerBundle,
    trial: TCNTuningTrial,
    epoch: int,
) -> float | None:
    """Apply and return the auditable dynamic-skip rate for one epoch."""

    dynamic_rate = dynamic_skip_learning_rate_for_epoch(trial, epoch)
    if dynamic_rate is None:
        return None
    groups = {
        str(group.get("group_name")): group
        for group in bundle.optimizer.param_groups
    }
    if set(groups) != {"base", "dynamic_skip"}:
        raise ContractError("dynamic skip learning-rate groups drifted")
    groups["base"]["lr"] = trial.learning_rate
    groups["dynamic_skip"]["lr"] = dynamic_rate
    return dynamic_rate


def build_tcn_optimizer(
    model: nn.Module,
    trial: TCNTuningTrial,
) -> TCNOptimizerBundle:
    """Build complete, disjoint optimizer groups for one frozen TCN trial."""

    all_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not all_parameters:
        raise ContractError("TCN optimizer requires trainable parameters")
    optimizer_name = "AdamW" if trial.weight_decay > 0 else "Adam"

    if (
        trial.adapter_learning_rate is None
        and trial.residual_learning_rate is None
        and trial.dynamic_attention_learning_rate is None
        and trial.dynamic_skip_learning_rate is None
    ):
        optimizer: torch.optim.Optimizer
        if trial.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                all_parameters,
                lr=trial.learning_rate,
                weight_decay=trial.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                all_parameters,
                lr=trial.learning_rate,
            )
        return TCNOptimizerBundle(
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            parameter_group_identity=(
                f"shape-residual-only-lr-{trial.learning_rate:g}"
                if trial.dynamic_skip_frozen_parent
                else f"all-lr-{trial.learning_rate:g}"
            ),
            adapter_parameter_count=0,
            residual_parameter_count=0,
        )

    if trial.dynamic_skip_learning_rate is not None:
        if (
            trial.adapter_learning_rate is not None
            or trial.residual_learning_rate is not None
            or trial.dynamic_attention_learning_rate is not None
        ):
            raise ContractError(
                "dynamic skip learning rate cannot share another special group"
            )
        dynamic_builder = getattr(model, "dynamic_skip_parameters", None)
        if not callable(dynamic_builder):
            raise ContractError(
                "dynamic skip learning rate requires explicit dynamic parameters"
            )
        dynamic_parameters = tuple(
            parameter
            for parameter in cast(Any, dynamic_builder)()
            if parameter.requires_grad
        )
        dynamic_ids = {id(parameter) for parameter in dynamic_parameters}
        if not dynamic_parameters or len(dynamic_ids) != len(dynamic_parameters):
            raise ContractError("dynamic skip parameters must be non-empty and unique")
        base_parameters = tuple(
            parameter for parameter in all_parameters if id(parameter) not in dynamic_ids
        )
        grouped_ids = {
            id(parameter) for parameter in (*base_parameters, *dynamic_parameters)
        }
        if grouped_ids != {id(parameter) for parameter in all_parameters}:
            raise ContractError("dynamic skip optimizer groups are incomplete")
        initial_dynamic_skip_learning_rate = dynamic_skip_learning_rate_for_epoch(
            trial, 1
        )
        assert initial_dynamic_skip_learning_rate is not None
        dynamic_skip_parameter_groups: list[dict[str, Any]] = [
            {
                "params": base_parameters,
                "lr": trial.learning_rate,
                "group_name": "base",
            },
            {
                "params": dynamic_parameters,
                "lr": initial_dynamic_skip_learning_rate,
                "group_name": "dynamic_skip",
            },
        ]
        if trial.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                dynamic_skip_parameter_groups,
                lr=trial.learning_rate,
                weight_decay=trial.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                dynamic_skip_parameter_groups,
                lr=trial.learning_rate,
            )
        return TCNOptimizerBundle(
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            parameter_group_identity=(
                (
                    f"base-lr-{trial.learning_rate:g}"
                    f"+dynamic-skip-linear-warmup-"
                    f"{trial.dynamic_skip_warmup_epochs}"
                    f"-lr-{trial.learning_rate:g}"
                    f"-to-{trial.dynamic_skip_learning_rate:g}"
                )
                if trial.dynamic_skip_warmup_epochs > 0
                else (
                    f"base-lr-{trial.learning_rate:g}"
                    f"+dynamic-skip-lr-{trial.dynamic_skip_learning_rate:g}"
                )
            ),
            adapter_parameter_count=0,
            residual_parameter_count=0,
            dynamic_skip_parameter_count=sum(
                parameter.numel() for parameter in dynamic_parameters
            ),
        )

    if trial.dynamic_attention_learning_rate is not None:
        if (
            trial.adapter_learning_rate is not None
            or trial.residual_learning_rate is not None
        ):
            raise ContractError(
                "dynamic attention learning rate cannot share another special group"
            )
        dynamic_builder = getattr(model, "dynamic_attention_parameters", None)
        if not callable(dynamic_builder):
            raise ContractError(
                "dynamic attention learning rate requires explicit dynamic parameters"
            )
        dynamic_parameters = tuple(cast(Any, dynamic_builder)())
        dynamic_ids = {id(parameter) for parameter in dynamic_parameters}
        if not dynamic_parameters or len(dynamic_ids) != len(dynamic_parameters):
            raise ContractError(
                "dynamic attention parameters must be non-empty and unique"
            )
        base_parameters = tuple(
            parameter
            for parameter in all_parameters
            if id(parameter) not in dynamic_ids
        )
        grouped_ids = {
            id(parameter) for parameter in (*base_parameters, *dynamic_parameters)
        }
        if grouped_ids != {id(parameter) for parameter in all_parameters}:
            raise ContractError("dynamic attention optimizer groups are incomplete")
        dynamic_parameter_groups: list[dict[str, Any]] = [
            {
                "params": base_parameters,
                "lr": trial.learning_rate,
                "group_name": "base",
            },
            {
                "params": dynamic_parameters,
                "lr": trial.dynamic_attention_learning_rate,
                "group_name": "dynamic_attention",
            },
        ]
        if trial.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                dynamic_parameter_groups,
                lr=trial.learning_rate,
                weight_decay=trial.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                dynamic_parameter_groups,
                lr=trial.learning_rate,
            )
        return TCNOptimizerBundle(
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            parameter_group_identity=(
                f"base-lr-{trial.learning_rate:g}"
                f"+dynamic-attention-lr-{trial.dynamic_attention_learning_rate:g}"
            ),
            adapter_parameter_count=0,
            residual_parameter_count=0,
            dynamic_attention_parameter_count=sum(
                parameter.numel() for parameter in dynamic_parameters
            ),
        )

    if trial.residual_learning_rate is not None:
        residual_builder = getattr(model, "residual_adapter_parameters", None)
        if not callable(residual_builder):
            raise ContractError(
                "residual learning rate requires explicit residual parameters"
            )
        residual_parameters = tuple(cast(Any, residual_builder)())
        residual_ids = {id(parameter) for parameter in residual_parameters}
        if not residual_parameters or len(residual_ids) != len(residual_parameters):
            raise ContractError(
                "signed residual parameters must be non-empty and unique"
            )
        base_parameters = tuple(
            parameter
            for parameter in all_parameters
            if id(parameter) not in residual_ids
        )
        grouped_ids = {
            id(parameter) for parameter in (*base_parameters, *residual_parameters)
        }
        if grouped_ids != {id(parameter) for parameter in all_parameters}:
            raise ContractError("TCN residual optimizer groups are incomplete")
        parameter_groups: list[dict[str, Any]] = [
            {
                "params": base_parameters,
                "lr": trial.learning_rate,
                "group_name": "base",
            },
            {
                "params": residual_parameters,
                "lr": trial.residual_learning_rate,
                "group_name": "signed_residual",
            },
        ]
        if trial.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                parameter_groups,
                lr=trial.learning_rate,
                weight_decay=trial.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                parameter_groups,
                lr=trial.learning_rate,
            )
        return TCNOptimizerBundle(
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            parameter_group_identity=(
                f"base-lr-{trial.learning_rate:g}"
                f"+residual-lr-{trial.residual_learning_rate:g}"
            ),
            adapter_parameter_count=0,
            residual_parameter_count=sum(
                parameter.numel() for parameter in residual_parameters
            ),
        )

    adapter_builder = getattr(model, "temporal_adapter_parameters", None)
    if not callable(adapter_builder):
        raise ContractError(
            "adapter learning rate requires explicit temporal adapter parameters"
        )
    adapter_parameters = tuple(cast(Any, adapter_builder)())
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    if not adapter_parameters or len(adapter_ids) != len(adapter_parameters):
        raise ContractError("temporal adapter parameters must be non-empty and unique")
    base_parameters = tuple(
        parameter for parameter in all_parameters if id(parameter) not in adapter_ids
    )
    grouped_ids = {
        id(parameter) for parameter in (*base_parameters, *adapter_parameters)
    }
    if grouped_ids != {id(parameter) for parameter in all_parameters}:
        raise ContractError("TCN optimizer parameter groups are incomplete")
    adapter_parameter_groups: list[dict[str, Any]] = [
        {
            "params": base_parameters,
            "lr": trial.learning_rate,
            "group_name": "base",
        },
        {
            "params": adapter_parameters,
            "lr": trial.adapter_learning_rate,
            "group_name": "temporal_adapter",
        },
    ]
    if trial.weight_decay > 0:
        optimizer = torch.optim.AdamW(
            adapter_parameter_groups,
            lr=trial.learning_rate,
            weight_decay=trial.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            adapter_parameter_groups,
            lr=trial.learning_rate,
        )
    return TCNOptimizerBundle(
        optimizer=optimizer,
        optimizer_name=optimizer_name,
        parameter_group_identity=(
            f"base-lr-{trial.learning_rate:g}"
            f"+adapter-lr-{trial.adapter_learning_rate:g}"
        ),
        adapter_parameter_count=sum(
            parameter.numel() for parameter in adapter_parameters
        ),
        residual_parameter_count=0,
    )


@dataclass(frozen=True)
class ValidationRankIC:
    """Date/horizon grouped validation score used for model selection."""

    mean_daily_rankic: float
    rankic_by_horizon: dict[int, float]
    valid_group_count: int


@dataclass(frozen=True)
class _ValidationRankICGroup:
    """One immutable date/horizon group with pre-ranked validation targets."""

    signal_date: str
    horizon: int
    score_column: int
    score_rows: np.ndarray
    centered_target_ranks: np.ndarray
    target_sum_squares: float


@dataclass(frozen=True)
class ValidationRankICPlan:
    """Cached label joins and target ranks for repeated epoch validation."""

    sample_positions: np.ndarray
    groups: tuple[_ValidationRankICGroup, ...]

    def evaluate(
        self, scores: np.ndarray, sample_positions: np.ndarray
    ) -> ValidationRankIC:
        """Evaluate changing model scores against one position-safe fixed plan."""

        daily = self.evaluate_daily(scores, sample_positions)
        if daily.empty:
            return ValidationRankIC(float("nan"), {}, 0)
        by_horizon = {
            int(cast(Any, horizon)): float(rows["rankic"].mean())
            for horizon, rows in daily.groupby("horizon", observed=True)
        }
        return ValidationRankIC(
            mean_daily_rankic=float(daily["rankic"].mean()),
            rankic_by_horizon=by_horizon,
            valid_group_count=len(daily),
        )

    def evaluate_daily(
        self, scores: np.ndarray, sample_positions: np.ndarray
    ) -> pd.DataFrame:
        """Return the exact date/horizon RankIC groups used by selection."""

        if scores.ndim != 2 or scores.shape[1] != len(HORIZONS):
            raise ContractError("validation scores must have four horizon columns")
        observed_positions = np.asarray(sample_positions, dtype="int64")
        if len(scores) != len(observed_positions):
            raise ContractError(
                "validation scores and positions must have equal length"
            )
        if not np.array_equal(observed_positions, self.sample_positions):
            raise ContractError("validation positions do not match cached plan")

        rows: list[dict[str, object]] = []
        for group in self.groups:
            values = scores[group.score_rows, group.score_column]
            if np.unique(values).size < 2:
                continue
            score_ranks = np.asarray(rankdata(values), dtype="float64")
            centered_scores = score_ranks - score_ranks.mean()
            score_sum_squares = float(np.dot(centered_scores, centered_scores))
            denominator = np.sqrt(score_sum_squares * group.target_sum_squares)
            if not np.isfinite(denominator) or denominator <= 0:
                continue
            rankic = float(
                np.dot(centered_scores, group.centered_target_ranks) / denominator
            )
            if np.isfinite(rankic):
                rows.append(
                    {
                        "signal_date": group.signal_date,
                        "horizon": group.horizon,
                        "rankic": rankic,
                        "valid_member_count": len(group.score_rows),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "signal_date",
                "horizon",
                "rankic",
                "valid_member_count",
            ],
        )


@dataclass(frozen=True)
class TCNTuningDecision:
    """Pre-registered candidate selection and data-scale decision."""

    selected_trial_id: str
    status: str
    mean_improvement: float
    non_degrading_horizon_count: int


@dataclass(frozen=True)
class TCNTuningResult:
    """Epoch history, fold leaderboard, and restorable best checkpoints."""

    epoch_history: pd.DataFrame
    leaderboard: pd.DataFrame
    best_states: dict[str, dict[str, torch.Tensor]]
    epoch_states: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)


def build_tcn_trial_model(
    trial: TCNTuningTrial,
    *,
    feature_count: int,
    input_steps: int,
    market_context_dim: int | None = None,
) -> torch.nn.Module:
    if trial.model_kind == "bai":
        return BaiTCN(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
        )
    if trial.model_kind == "horizon_skip":
        return HorizonSkipTCN(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
            input_steps=input_steps,
            padding_mode=trial.padding_mode,
        )
    if trial.model_kind == "dynamic_horizon_skip":
        if trial.dynamic_skip_shape_residual:
            return ShapeResidualDynamicHorizonSkipTCN(
                feature_count=feature_count,
                channels=trial.channels,
                kernel_size=trial.kernel_size,
                dilations=trial.dilations,
                dropout=trial.dropout,
                input_steps=input_steps,
                dynamic_skip_hidden=trial.dynamic_skip_hidden,
                dynamic_skip_scale=trial.dynamic_skip_scale,
                dynamic_skip_shape_residual_scale=(
                    trial.dynamic_skip_shape_residual_scale
                ),
                padding_mode=trial.padding_mode,
            )
        return DynamicHorizonSkipTCN(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
            input_steps=input_steps,
            dynamic_skip_hidden=trial.dynamic_skip_hidden,
            dynamic_skip_scale=trial.dynamic_skip_scale,
            dynamic_skip_token_normalization=(
                trial.dynamic_skip_token_normalization
            ),
            padding_mode=trial.padding_mode,
        )
    if trial.model_kind == "temporal_context":
        return TemporalContextTCN(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
            input_steps=input_steps,
            bars_per_day=trial.bars_per_day,
            padding_mode=trial.padding_mode,
        )
    if trial.model_kind == "signed_temporal_context":
        return SignedTemporalContextTCN(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
            input_steps=input_steps,
            bars_per_day=trial.bars_per_day,
            padding_mode=trial.padding_mode,
        )
    if trial.model_kind == "stabilized_temporal_context":
        return StabilizedResidualTemporalContextTCN(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
            input_steps=input_steps,
            bars_per_day=trial.bars_per_day,
            residual_scale=trial.residual_scale,
            padding_mode=trial.padding_mode,
        )
    if trial.model_kind == "decoupled_temporal_context":
        return DecoupledResidualTemporalContextTCN(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
            input_steps=input_steps,
            bars_per_day=trial.bars_per_day,
            residual_scale=trial.residual_scale,
            padding_mode=trial.padding_mode,
        )
    if trial.model_kind == "market_conditioned_temporal_context":
        if market_context_dim is None:
            raise ContractError("market-conditioned TCN requires PIT market context")
        if market_context_dim != trial.market_context_dim:
            raise ContractError("market context dimension drifted from the trial")
        return MarketConditionedTemporalContextTCN(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
            input_steps=input_steps,
            bars_per_day=trial.bars_per_day,
            market_context_dim=trial.market_context_dim,
            market_context_hidden=trial.market_context_hidden,
            market_gate_scale=trial.market_gate_scale,
            padding_mode=trial.padding_mode,
        )
    if trial.model_kind == "dynamic_temporal_context":
        return DynamicTemporalContextTCN(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
            input_steps=input_steps,
            bars_per_day=trial.bars_per_day,
            dynamic_attention_hidden=trial.dynamic_attention_hidden,
            dynamic_attention_scale=trial.dynamic_attention_scale,
            padding_mode=trial.padding_mode,
        )
    if trial.padding_mode == "chomp":
        return TCNLiteChomp(
            feature_count=feature_count,
            channels=trial.channels,
            kernel_size=trial.kernel_size,
            dilations=trial.dilations,
            dropout=trial.dropout,
        )
    return TCNLite(
        feature_count=feature_count,
        channels=trial.channels,
        kernel_size=trial.kernel_size,
        dilations=trial.dilations,
        dropout=trial.dropout,
        head_dropout=trial.head_dropout,
        dropout_kind=trial.dropout_kind,
    )


def _training_loader(
    dataset: Any,
    trial: TCNTuningTrial,
    *,
    train_positions: np.ndarray,
    dates_by_position: pd.Series,
    model_seed: int,
) -> DataLoader[Any]:
    if trial.strategy in {
        "grouped_smooth_l1",
        "rank_objective",
        "soft_rankic",
        "top_tail",
        "teacher_listwise",
    }:
        if dates_by_position.empty:
            raise ContractError("date-grouped v9 training requires signal_date")
        dates = [dates_by_position.loc[int(value)] for value in train_positions]
        return DataLoader(
            dataset,
            batch_sampler=DateGroupedBatchSampler(
                dates,
                shuffle_dates=True,
                seed=model_seed,
                batch_size=trial.batch_size,
                order_policy=trial.date_batch_order,
            ),
            num_workers=0,
        )
    return DataLoader(
        dataset,
        batch_size=trial.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(model_seed),
        num_workers=0,
    )


def _unpack_tcn_batch(
    batch: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    if not isinstance(batch, (tuple, list)) or len(batch) not in {4, 5}:
        raise ContractError("TCN loader returned an unsupported batch contract")
    features, targets, masks, positions = batch[:4]
    context = batch[4] if len(batch) == 5 else None
    return features, targets, masks, positions, context


def _forward_tcn_trial(
    model: torch.nn.Module,
    features: torch.Tensor,
    market_context: torch.Tensor | None,
) -> torch.Tensor:
    if isinstance(model, MarketConditionedTemporalContextTCN):
        if market_context is None:
            raise ContractError("market-conditioned TCN batch is missing context")
        return model(features, market_context)
    if market_context is not None:
        raise ContractError("stock-only TCN unexpectedly received market context")
    return model(features)


def _predict_tcn_trial(
    model: torch.nn.Module,
    dataset: Any,
    *,
    batch_size: int,
    without_shape_residual: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    score_batches: list[np.ndarray] = []
    position_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features, _, _, positions, context = _unpack_tcn_batch(batch)
            if without_shape_residual:
                if not isinstance(model, ShapeResidualDynamicHorizonSkipTCN):
                    raise ContractError(
                        "shape-residual counterfactual requires the shape TCN"
                    )
                if context is not None:
                    raise ContractError(
                        "shape-residual counterfactual forbids market context"
                    )
                prediction = model.forward_without_shape_residual(features)
            else:
                prediction = _forward_tcn_trial(model, features, context)
            score_batches.append(
                prediction.cpu().numpy()
            )
            position_batches.append(positions.numpy())
    return np.concatenate(score_batches), np.concatenate(position_batches)


def predict_tcn_trial(
    model: torch.nn.Module,
    dataset: Any,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict one public TCN trial without exposing task-script internals."""

    return _predict_tcn_trial(model, dataset, batch_size=batch_size)


def _backward_trial_loss(
    trial: TCNTuningTrial,
    model: torch.nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    dates_by_position: pd.Series,
    *,
    gradient_seed: int,
    teacher_target: torch.Tensor | None = None,
) -> _BackwardTrialResult:
    safe_target = torch.where(mask, target, prediction.detach())
    if trial.strategy == "smooth_l1":
        loss = masked_smooth_l1(prediction, safe_target, mask)
        loss.backward()
        return _BackwardTrialResult(loss)
    if trial.strategy == "grouped_smooth_l1":
        dates = [dates_by_position.loc[int(value)] for value in positions]
        if trial.grouped_smooth_l1_reduction == "date_horizon_mean":
            equal = date_horizon_equal_smooth_l1(
                prediction, target, mask, dates
            )
            equal.total.backward()
            return _BackwardTrialResult(
                equal.total,
                loss_group_count=equal.group_count,
                valid_label_count=equal.valid_label_count,
            )
        group_count, valid_label_count = date_horizon_group_counts(mask, dates)
        loss = masked_smooth_l1(prediction, safe_target, mask)
        loss.backward()
        return _BackwardTrialResult(
            loss,
            loss_group_count=group_count,
            valid_label_count=valid_label_count,
        )
    if trial.strategy == "rank_objective":
        dates = [dates_by_position.loc[int(value)] for value in positions]
        rank_result = mixed_date_grouped_rank_loss(
            prediction,
            target,
            mask,
            dates,
            rank_objective_allowed=True,
        )
        rank_result.total.backward()
        return _BackwardTrialResult(rank_result.total)
    if trial.strategy == "soft_rankic":
        dates = [dates_by_position.loc[int(value)] for value in positions]
        soft_rankic_result = mixed_date_grouped_soft_rankic_loss(
            prediction,
            target,
            mask,
            dates,
            soft_rankic_weight=trial.soft_rankic_weight,
            temperature=trial.soft_rank_temperature,
        )
        soft_rankic_result.total.backward()
        return _BackwardTrialResult(soft_rankic_result.total)
    if trial.strategy == "top_tail":
        dates = [dates_by_position.loc[int(value)] for value in positions]
        top_tail_result = mixed_date_grouped_top_tail_loss(
            prediction,
            target,
            mask,
            dates,
            top_tail_weight=trial.top_tail_weight,
            top_fraction=trial.top_tail_fraction,
            temperature=trial.top_tail_temperature,
        )
        smooth_gradient = torch.autograd.grad(
            top_tail_result.smooth_l1,
            prediction,
            retain_graph=True,
        )[0]
        top_tail_gradient = torch.autograd.grad(
            top_tail_result.top_tail,
            prediction,
            retain_graph=True,
        )[0]
        denominator = float(
            torch.linalg.vector_norm(smooth_gradient)
            * torch.linalg.vector_norm(top_tail_gradient)
        )
        component_gradient_cosine = (
            float(
                torch.dot(
                    smooth_gradient.reshape(-1),
                    top_tail_gradient.reshape(-1),
                )
                / denominator
            )
            if np.isfinite(denominator) and denominator > 0
            else float("nan")
        )
        top_tail_result.total.backward()
        return _BackwardTrialResult(
            top_tail_result.total,
            loss_group_count=top_tail_result.group_count,
            valid_label_count=int(mask.sum()),
            auxiliary_pair_count=top_tail_result.pair_count,
            component_gradient_cosine=component_gradient_cosine,
        )
    if trial.strategy == "teacher_listwise":
        if teacher_target is None:
            raise ContractError("teacher listwise training target is missing")
        dates = [dates_by_position.loc[int(value)] for value in positions]
        components = teacher_listwise_components(
            prediction,
            target,
            teacher_target,
            mask,
            dates,
            temperature=trial.teacher_listwise_temperature,
        )
        true_gradient = torch.autograd.grad(
            components.smooth_l1,
            prediction,
            retain_graph=True,
        )[0]
        teacher_gradient = torch.autograd.grad(
            components.teacher_listwise,
            prediction,
            retain_graph=True,
        )[0]
        true_norm = torch.linalg.vector_norm(true_gradient)
        teacher_norm = torch.linalg.vector_norm(teacher_gradient)
        if not (
            bool(torch.isfinite(true_norm))
            and bool(torch.isfinite(teacher_norm))
            and float(true_norm) > 0.0
            and float(teacher_norm) > 0.0
        ):
            raise ContractError("teacher listwise prediction gradients are invalid")
        scale = (true_norm / teacher_norm.clamp_min(1e-12)).detach()
        ratio = trial.teacher_listwise_gradient_ratio
        total = components.smooth_l1 + ratio * scale * components.teacher_listwise
        cosine = float(
            torch.dot(true_gradient.reshape(-1), teacher_gradient.reshape(-1))
            / (true_norm * teacher_norm)
        )
        realized_ratio = float(ratio * scale * teacher_norm / true_norm)
        total.backward()
        return _BackwardTrialResult(
            total,
            loss_group_count=components.group_count,
            valid_label_count=components.valid_label_count,
            component_gradient_cosine=cosine,
            teacher_gradient_ratio=realized_ratio,
        )

    horizon_losses = {
        int(horizon): masked_smooth_l1(
            prediction[:, column],
            safe_target[:, column],
            mask[:, column],
        )
        for column, horizon in enumerate(HORIZONS)
    }
    trunk = cast(Any, model).trunk
    all_block_parameters = {
        f"block-{offset}": tuple(block.parameters())
        for offset, block in enumerate(trunk)
    }
    selected_blocks = (
        tuple(range(len(trunk))) if trial.pcgrad_blocks is None else trial.pcgrad_blocks
    )
    block_parameters = {
        f"block-{offset}": all_block_parameters[f"block-{offset}"]
        for offset in selected_blocks
    }
    loss = torch.stack(list(horizon_losses.values())).mean()
    receipt = pcgrad_backward(
        horizon_losses,
        block_parameters,
        seed=gradient_seed,
        total_loss=loss,
        selected_horizons=(
            HORIZONS if trial.pcgrad_horizons is None else trial.pcgrad_horizons
        ),
    )
    return _BackwardTrialResult(loss, receipt)


def _trial_config_identity(
    trial: TCNTuningTrial,
    *,
    max_epochs: int,
    patience: int,
    min_delta: float,
    checkpoint_min_delta: float | None = None,
) -> str:
    payload = {
        **trial.__dict__,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "checkpoint_min_delta": checkpoint_min_delta,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_tcn_validation_sweep(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    trials: Sequence[TCNTuningTrial],
    seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    checkpoint_min_delta: float | None = None,
    torch_threads: int | None = None,
    protocol_identities: Mapping[str, str] | None = None,
    market_context: PITMarketContext | None = None,
    frozen_parent_states: Mapping[
        str, Mapping[str, torch.Tensor]
    ] | None = None,
    training_target_overrides: Mapping[int, np.ndarray] | None = None,
    training_teacher_targets: Mapping[int, np.ndarray] | None = None,
    capture_epoch_states: bool = False,
    disable_early_stopping: bool = False,
) -> TCNTuningResult:
    """Run the bounded TCN sweep with a restored, auditable thread scope."""

    with torch_thread_scope(torch_threads) as effective_torch_threads:
        result = _run_tcn_validation_sweep_unscoped(
            features,
            window_index,
            labels,
            split_manifest,
            trials=trials,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            checkpoint_min_delta=checkpoint_min_delta,
            protocol_identities=protocol_identities,
            market_context=market_context,
            frozen_parent_states=frozen_parent_states,
            training_target_overrides=training_target_overrides,
            training_teacher_targets=training_teacher_targets,
            capture_epoch_states=capture_epoch_states,
            disable_early_stopping=disable_early_stopping,
        )
    epoch_history = result.epoch_history.copy()
    leaderboard = result.leaderboard.copy()
    epoch_history["torch_threads"] = effective_torch_threads
    leaderboard["torch_threads"] = effective_torch_threads
    return TCNTuningResult(
        epoch_history,
        leaderboard,
        result.best_states,
        result.epoch_states,
    )


def _run_tcn_validation_sweep_unscoped(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    trials: Sequence[TCNTuningTrial],
    seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    checkpoint_min_delta: float | None,
    protocol_identities: Mapping[str, str] | None,
    market_context: PITMarketContext | None,
    frozen_parent_states: Mapping[
        str, Mapping[str, torch.Tensor]
    ] | None,
    training_target_overrides: Mapping[int, np.ndarray] | None,
    training_teacher_targets: Mapping[int, np.ndarray] | None,
    capture_epoch_states: bool,
    disable_early_stopping: bool,
) -> TCNTuningResult:
    """Train Bai TCN trials with ordinary validation-only early stopping."""

    effective_checkpoint_min_delta = (
        min_delta if checkpoint_min_delta is None else checkpoint_min_delta
    )
    if disable_early_stopping and not capture_epoch_states:
        raise ContractError(
            "disabling early stopping requires auditable epoch checkpoints"
        )
    selection_deltas = np.asarray(
        [effective_checkpoint_min_delta, min_delta], dtype="float64"
    )
    if (
        not np.isfinite(selection_deltas).all()
        or bool((selection_deltas < 0).any())
        or effective_checkpoint_min_delta > min_delta
    ):
        raise ContractError(
            "checkpoint selection deltas must be finite, non-negative and ordered"
        )
    if len(features) != len(window_index):
        raise ContractError("features and window index sample counts must match")
    resolved_trials = validate_tcn_tuning_plan(
        trials,
        input_steps=int(features.shape[2]),
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
    )
    if any(trial.epoch_average_start is not None for trial in resolved_trials) and not (
        disable_early_stopping and capture_epoch_states
    ):
        raise ContractError(
            "final epoch averaging requires disabled early stopping and auditable epoch states"
        )
    if (
        any(trial.teacher_blend_start_weight is not None for trial in resolved_trials)
        and training_target_overrides is not None
    ):
        raise ContractError(
            "linear teacher blending cannot be combined with static target overrides"
        )
    requires_market_context = any(
        trial.model_kind == "market_conditioned_temporal_context"
        for trial in resolved_trials
    )
    if requires_market_context and market_context is None:
        raise ContractError("market-conditioned TCN sweep requires PIT market context")
    if market_context is not None:
        if market_context.values.shape[0] != len(features):
            raise ContractError("market context and feature sample counts differ")
        if market_context.values.ndim != 2:
            raise ContractError("market context values must be two-dimensional")
    if "stage" not in split_manifest or "fold" not in split_manifest:
        raise ContractError("split manifest requires stage and fold columns")
    allowed_stages = {"train", "validation", "test"}
    observed_stages = set(split_manifest["stage"].astype(str))
    if not observed_stages <= allowed_stages:
        raise ContractError("split manifest contains unsupported tuning stages")
    targets, masks = _label_matrices(window_index, labels)
    protocols = build_fold_protocols(features, split_manifest)
    validation_plans = {
        protocol.fold: build_validation_rankic_plan(
            protocol.validation_positions, window_index, labels
        )
        for protocol in protocols
    }
    if training_target_overrides is not None and set(training_target_overrides) != {
        protocol.fold for protocol in protocols
    }:
        raise ContractError("training target override fold coverage drifted")
    requires_teacher_targets = any(
        trial.strategy == "teacher_listwise"
        or trial.teacher_blend_start_weight is not None
        for trial in resolved_trials
    )
    if requires_teacher_targets and training_teacher_targets is None:
        raise ContractError("teacher listwise trials require fold-scoped teacher targets")
    if training_teacher_targets is not None and set(training_teacher_targets) != {
        protocol.fold for protocol in protocols
    }:
        raise ContractError("training teacher target fold coverage drifted")
    epoch_rows = []
    leaderboard_rows = []
    best_states: dict[str, dict[str, torch.Tensor]] = {}
    epoch_states: dict[str, dict[str, torch.Tensor]] = {}
    identities = dict(protocol_identities or {})
    dates_by_position = (
        window_index.set_index("sample_position")["signal_date"].astype(str)
        if "signal_date" in window_index
        else pd.Series(dtype=str)
    )
    for trial in resolved_trials:
        observed_receptive_field = (
            receptive_field(kernel_size=trial.kernel_size, dilations=trial.dilations)
            if trial.model_kind == "bai"
            else lite_receptive_field(
                kernel_size=trial.kernel_size, dilations=trial.dilations
            )
        )
        for protocol in protocols:
            training_targets = targets
            fold_teacher_targets: np.ndarray | None = None
            if training_target_overrides is not None:
                override = np.asarray(training_target_overrides[protocol.fold])
                if override.shape != targets.shape:
                    raise ContractError("training target override shape drifted")
                train_positions = np.asarray(protocol.train_positions, dtype="int64")
                outside = np.ones(len(targets), dtype="bool")
                outside[train_positions] = False
                if not np.array_equal(override[outside], targets[outside], equal_nan=True):
                    raise ContractError(
                        "training target override changed a non-train position"
                    )
                valid_train = masks[train_positions]
                if not np.isfinite(override[train_positions][valid_train]).all():
                    raise ContractError(
                        "training target override is incomplete on valid train cells"
                    )
                training_targets = override.astype(targets.dtype, copy=False)
            if training_teacher_targets is not None:
                teacher = np.asarray(training_teacher_targets[protocol.fold])
                if teacher.shape != targets.shape:
                    raise ContractError("training teacher target shape drifted")
                train_positions = np.asarray(protocol.train_positions, dtype="int64")
                outside = np.ones(len(targets), dtype="bool")
                outside[train_positions] = False
                if np.isfinite(teacher[outside]).any():
                    raise ContractError(
                        "training teacher target exposed a non-train position"
                    )
                valid_train = masks[train_positions]
                teacher_valid = teacher[train_positions][valid_train]
                if (
                    not np.isfinite(teacher_valid).all()
                    or bool((np.abs(teacher_valid) > 1.0 + 1e-7).any())
                ):
                    raise ContractError(
                        "training teacher target is invalid on train cells"
                    )
                fold_teacher_targets = teacher.astype(targets.dtype, copy=False)
            model_seed = seed + protocol.fold * 100
            torch.manual_seed(model_seed)
            model = build_tcn_trial_model(
                trial,
                feature_count=int(features.shape[1]),
                input_steps=int(features.shape[2]),
                market_context_dim=(
                    None
                    if market_context is None
                    else int(market_context.values.shape[1])
                ),
            )
            parent_state: Mapping[str, torch.Tensor] | None = None
            if trial.dynamic_skip_frozen_parent:
                if not isinstance(model, ShapeResidualDynamicHorizonSkipTCN):
                    raise ContractError(
                        "dynamic skip frozen parent requires shape residual TCN"
                    )
                checkpoint_key = f"{trial.trial_id}-fold-{protocol.fold}"
                if frozen_parent_states is None or checkpoint_key not in frozen_parent_states:
                    raise ContractError(
                        f"frozen parent state is missing for {checkpoint_key}"
                    )
                parent_state = frozen_parent_states[checkpoint_key]
                model.load_frozen_raw_parent(parent_state)
            context_standardizer = None
            if trial.model_kind == "market_conditioned_temporal_context":
                assert market_context is not None
                context_standardizer = fit_market_context_standardizer(
                    market_context,
                    window_index,
                    train_positions=protocol.train_positions,
                )
                train_dataset: Any = ContextualLazyWindowDataset(
                    features,
                    protocol.train_positions,
                    training_targets,
                    masks,
                    protocol.feature_mean,
                    protocol.feature_std,
                    market_context,
                    context_standardizer,
                )
                validation_dataset: Any = ContextualLazyWindowDataset(
                    features,
                    protocol.validation_positions,
                    targets,
                    masks,
                    protocol.feature_mean,
                    protocol.feature_std,
                    market_context,
                    context_standardizer,
                )
            else:
                train_dataset = LazyWindowDataset(
                    features,
                    protocol.train_positions,
                    training_targets,
                    masks,
                    protocol.feature_mean,
                    protocol.feature_std,
                )
                validation_dataset = LazyWindowDataset(
                    features,
                    protocol.validation_positions,
                    targets,
                    masks,
                    protocol.feature_mean,
                    protocol.feature_std,
                )
            loader = _training_loader(
                train_dataset,
                trial,
                train_positions=protocol.train_positions,
                dates_by_position=dates_by_position,
                model_seed=model_seed,
            )
            optimizer_bundle = build_tcn_optimizer(model, trial)
            optimizer = optimizer_bundle.optimizer
            optimizer_name = optimizer_bundle.optimizer_name
            parameter_ema = (
                ParameterEMA(decay=trial.ema_decay)
                if trial.ema_decay is not None
                else None
            )
            epoch_parameter_average = (
                EpochParameterAverage()
                if trial.epoch_average_start is not None
                else None
            )
            best_score = float("-inf")
            best_epoch = 0
            best_rankic_by_horizon: dict[int, float] = {}
            best_state: dict[str, torch.Tensor] | None = None
            total_model_step_seconds = 0.0
            total_data_wait_seconds = 0.0
            total_validation_seconds = 0.0
            total_pcgrad_projection_seconds = 0.0
            total_pcgrad_horizon_backward_seconds = 0.0
            total_samples = 0
            observed_batch_sizes: list[int] = []
            epoch_gradient_norm_cvs: list[float] = []
            epoch_labels_per_loss_group: list[float] = []
            epoch_component_gradient_cosine_medians: list[float] = []
            epoch_top_tail_pair_count_medians: list[float] = []
            epoch_teacher_gradient_ratio_medians: list[float] = []
            date_order_fingerprints: set[str] = set()
            time_to_best_seconds = 0.0
            completed_epochs = 0
            stopping_reason = "max_epochs"
            cycle_start = time.perf_counter()
            baseline_score = float("nan")
            baseline_rankic_by_horizon: dict[int, float] = {}
            parent_prediction_max_abs_error = float("nan")
            checkpoint_only_improvement_count = 0
            material_patience_improvement_count = 0
            if capture_epoch_states:
                epoch_states[
                    f"{trial.trial_id}-fold-{protocol.fold}-epoch-0"
                ] = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in copy.deepcopy(model.state_dict()).items()
                }
            if trial.dynamic_skip_frozen_parent:
                baseline_validation_start = time.perf_counter()
                baseline_scores, baseline_positions = _predict_tcn_trial(
                    model,
                    validation_dataset,
                    batch_size=trial.batch_size,
                )
                parent_scores, parent_positions = _predict_tcn_trial(
                    model,
                    validation_dataset,
                    batch_size=trial.batch_size,
                    without_shape_residual=True,
                )
                if not np.array_equal(baseline_positions, parent_positions):
                    raise ContractError(
                        "frozen parent validation positions drifted"
                    )
                parent_prediction_max_abs_error = float(
                    np.max(np.abs(baseline_scores - parent_scores))
                )
                if parent_prediction_max_abs_error != 0.0:
                    raise ContractError(
                        "frozen parent prediction is not exactly preserved"
                    )
                baseline_rankic = validation_plans[protocol.fold].evaluate(
                    baseline_scores, baseline_positions
                )
                if not np.isfinite(baseline_rankic.mean_daily_rankic):
                    raise ContractError(
                        "frozen parent baseline has no finite validation RankIC"
                    )
                baseline_validation_seconds = (
                    time.perf_counter() - baseline_validation_start
                )
                total_validation_seconds += baseline_validation_seconds
                baseline_score = baseline_rankic.mean_daily_rankic
                baseline_rankic_by_horizon = dict(
                    baseline_rankic.rankic_by_horizon
                )
                best_score = baseline_score
                best_epoch = 0
                best_rankic_by_horizon = dict(baseline_rankic_by_horizon)
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in copy.deepcopy(model.state_dict()).items()
                }
                epoch_rows.append(
                    {
                        "trial_id": trial.trial_id,
                        "fold": protocol.fold,
                        "seed": seed,
                        "model_seed": model_seed,
                        "epoch": 0,
                        "dynamic_skip_epoch_learning_rate": None,
                        "stage": "frozen_parent_baseline",
                        "train_loss": float("nan"),
                        "training_seconds": 0.0,
                        "model_step_seconds": 0.0,
                        "data_wait_seconds": 0.0,
                        "pcgrad_projection_seconds": 0.0,
                        "pcgrad_horizon_backward_seconds": 0.0,
                        "validation_seconds": baseline_validation_seconds,
                        "samples_per_second": float("nan"),
                        "mean_daily_rankic": baseline_score,
                        "rankic_1d": baseline_rankic_by_horizon.get(
                            1, float("nan")
                        ),
                        "rankic_2d": baseline_rankic_by_horizon.get(
                            2, float("nan")
                        ),
                        "rankic_3d": baseline_rankic_by_horizon.get(
                            3, float("nan")
                        ),
                        "rankic_5d": baseline_rankic_by_horizon.get(
                            5, float("nan")
                        ),
                        "valid_group_count": baseline_rankic.valid_group_count,
                        "improved": True,
                        "checkpoint_improved": True,
                        "patience_improved": True,
                        "epochs_without_material_improvement": 0,
                        "date_order_epoch": None,
                        "date_order_fingerprint": None,
                        "optimizer_step_count": 0,
                        "gradient_norm_mean": float("nan"),
                        "gradient_norm_std": float("nan"),
                        "gradient_norm_cv": float("nan"),
                        "gradient_norm_max": float("nan"),
                        "batch_size_mean": float("nan"),
                        "batch_size_std": float("nan"),
                        "batch_size_cv": float("nan"),
                        "batch_size_min": float("nan"),
                        "batch_size_max": float("nan"),
                        "loss_group_count_mean": float("nan"),
                        "loss_group_count_min": float("nan"),
                        "loss_group_count_max": float("nan"),
                        "valid_label_count_mean": float("nan"),
                        "valid_label_count_min": float("nan"),
                        "valid_label_count_max": float("nan"),
                        "labels_per_loss_group_mean": float("nan"),
                        "top_tail_pair_count_mean": float("nan"),
                        "top_tail_pair_count_min": float("nan"),
                        "top_tail_pair_count_max": float("nan"),
                        "component_gradient_cosine_mean": float("nan"),
                        "component_gradient_cosine_median": float("nan"),
                        "component_gradient_cosine_min": float("nan"),
                    }
                )
            selection_state = ValidationSelectionState(
                best_score=best_score,
                patience_anchor_score=best_score,
                epochs_without_material_improvement=0,
                has_checkpoint=best_state is not None,
            )
            for epoch in range(1, max_epochs + 1):
                teacher_blend_weight: float | None = None
                if trial.teacher_blend_start_weight is not None:
                    teacher_blend_weight = (
                        trial.teacher_blend_start_weight
                        + (
                            trial.teacher_blend_end_weight
                            - trial.teacher_blend_start_weight
                        )
                        * float(epoch - 1)
                        / float(max_epochs - 1)
                    )
                dynamic_skip_epoch_learning_rate = apply_tcn_epoch_learning_rates(
                    optimizer_bundle, trial, epoch
                )
                model.train()
                losses = []
                epoch_model_step_seconds = 0.0
                epoch_data_wait_seconds = 0.0
                epoch_pcgrad_projection_seconds = 0.0
                epoch_pcgrad_horizon_backward_seconds = 0.0
                epoch_gradient_norms: list[float] = []
                epoch_batch_sizes: list[int] = []
                epoch_loss_group_counts: list[int] = []
                epoch_valid_label_counts: list[int] = []
                epoch_top_tail_pair_counts: list[int] = []
                epoch_component_gradient_cosines: list[float] = []
                epoch_teacher_gradient_ratios: list[float] = []
                date_order_epoch: int | None = None
                date_order_fingerprint: str | None = None
                if isinstance(loader.batch_sampler, DateGroupedBatchSampler):
                    loader.batch_sampler.set_epoch(epoch - 1)
                    date_order_epoch = loader.batch_sampler.epoch
                    date_order_fingerprint = (
                        loader.batch_sampler.order_fingerprint()
                    )
                    date_order_fingerprints.add(date_order_fingerprint)
                iterator = iter(loader)
                batch_number = 0
                while True:
                    data_wait_start = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    (
                        batch_features,
                        batch_targets,
                        batch_masks,
                        batch_positions,
                        batch_context,
                    ) = _unpack_tcn_batch(batch)
                    epoch_data_wait_seconds += time.perf_counter() - data_wait_start
                    model_step_start = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    prediction = _forward_tcn_trial(
                        model, batch_features, batch_context
                    )
                    batch_teacher_target = (
                        None
                        if fold_teacher_targets is None
                        else torch.as_tensor(
                            fold_teacher_targets[
                                batch_positions.detach().cpu().numpy()
                            ],
                            dtype=prediction.dtype,
                            device=prediction.device,
                        )
                    )
                    effective_batch_targets = batch_targets
                    if teacher_blend_weight is not None:
                        if batch_teacher_target is None:
                            raise ContractError(
                                "linear teacher blending has no train teacher target"
                            )
                        if teacher_blend_weight != 0.0:
                            blended_targets = (
                                (1.0 - teacher_blend_weight) * batch_targets
                                + teacher_blend_weight * batch_teacher_target
                            )
                            effective_batch_targets = torch.where(
                                batch_masks, blended_targets, batch_targets
                            )
                    backward = _backward_trial_loss(
                        trial,
                        model,
                        prediction,
                        effective_batch_targets,
                        batch_masks,
                        batch_positions,
                        dates_by_position,
                        gradient_seed=(model_seed + epoch * 10_000 + batch_number),
                        teacher_target=(
                            batch_teacher_target
                            if trial.strategy == "teacher_listwise"
                            else None
                        ),
                    )
                    if isinstance(loader.batch_sampler, DateGroupedBatchSampler):
                        squared_gradient_norm = sum(
                            float(parameter.grad.detach().square().sum())
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        )
                        epoch_gradient_norms.append(
                            float(np.sqrt(squared_gradient_norm))
                        )
                    if backward.loss_group_count is not None:
                        if (
                            backward.loss_group_count <= 0
                            or backward.valid_label_count is None
                            or backward.valid_label_count <= 0
                        ):
                            raise ContractError(
                                "date-grouped loss diagnostics are invalid"
                            )
                        epoch_loss_group_counts.append(backward.loss_group_count)
                        epoch_valid_label_counts.append(backward.valid_label_count)
                    if backward.auxiliary_pair_count is not None:
                        if backward.auxiliary_pair_count <= 0:
                            raise ContractError(
                                "top-tail loss produced no valid training pairs"
                            )
                        epoch_top_tail_pair_counts.append(
                            backward.auxiliary_pair_count
                        )
                    if backward.component_gradient_cosine is not None:
                        if not np.isfinite(backward.component_gradient_cosine):
                            raise ContractError(
                                "top-tail component gradient cosine is non-finite"
                            )
                        epoch_component_gradient_cosines.append(
                            backward.component_gradient_cosine
                        )
                    if backward.teacher_gradient_ratio is not None:
                        if not np.isfinite(backward.teacher_gradient_ratio):
                            raise ContractError(
                                "teacher listwise gradient ratio is non-finite"
                            )
                        epoch_teacher_gradient_ratios.append(
                            backward.teacher_gradient_ratio
                        )
                    optimizer.step()
                    if parameter_ema is not None:
                        parameter_ema.update(model)
                    if backward.pcgrad_receipt is not None:
                        epoch_pcgrad_projection_seconds += (
                            backward.pcgrad_receipt.projection_seconds
                        )
                        epoch_pcgrad_horizon_backward_seconds += (
                            backward.pcgrad_receipt.horizon_backward_seconds
                        )
                    epoch_model_step_seconds += time.perf_counter() - model_step_start
                    total_samples += len(batch_features)
                    observed_batch_sizes.append(len(batch_features))
                    epoch_batch_sizes.append(len(batch_features))
                    losses.append(float(backward.loss.detach()))
                    batch_number += 1
                training_seconds = epoch_model_step_seconds + epoch_data_wait_seconds
                total_model_step_seconds += epoch_model_step_seconds
                total_data_wait_seconds += epoch_data_wait_seconds
                total_pcgrad_projection_seconds += epoch_pcgrad_projection_seconds
                total_pcgrad_horizon_backward_seconds += (
                    epoch_pcgrad_horizon_backward_seconds
                )
                if (
                    epoch_parameter_average is not None
                    and trial.epoch_average_start is not None
                    and epoch >= trial.epoch_average_start
                ):
                    epoch_parameter_average.update(model)
                validation_start = time.perf_counter()
                if parameter_ema is not None:
                    with parameter_ema.average_parameters(model):
                        validation_scores, validation_positions = _predict_tcn_trial(
                            model,
                            validation_dataset,
                            batch_size=trial.batch_size,
                        )
                        checkpoint_state = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in copy.deepcopy(model.state_dict()).items()
                        }
                    validation_parameter_source = "ema"
                elif (
                    epoch_parameter_average is not None
                    and epoch_parameter_average.update_count > 0
                ):
                    with epoch_parameter_average.average_parameters(model):
                        validation_scores, validation_positions = _predict_tcn_trial(
                            model,
                            validation_dataset,
                            batch_size=trial.batch_size,
                        )
                        checkpoint_state = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in copy.deepcopy(model.state_dict()).items()
                        }
                    validation_parameter_source = "epoch_uniform_average"
                else:
                    validation_scores, validation_positions = _predict_tcn_trial(
                        model,
                        validation_dataset,
                        batch_size=trial.batch_size,
                    )
                    checkpoint_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in copy.deepcopy(model.state_dict()).items()
                    }
                    validation_parameter_source = "raw"
                rankic = validation_plans[protocol.fold].evaluate(
                    validation_scores, validation_positions
                )
                validation_seconds = time.perf_counter() - validation_start
                total_validation_seconds += validation_seconds
                score = rankic.mean_daily_rankic
                if capture_epoch_states:
                    epoch_states[
                        f"{trial.trial_id}-fold-{protocol.fold}-epoch-{epoch}"
                    ] = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in copy.deepcopy(model.state_dict()).items()
                    }
                selection = advance_validation_selection(
                    selection_state,
                    score=score,
                    checkpoint_min_delta=effective_checkpoint_min_delta,
                    patience_min_delta=min_delta,
                )
                selection_state = selection.state
                if selection.checkpoint_improved:
                    best_epoch = epoch
                    best_rankic_by_horizon = dict(rankic.rankic_by_horizon)
                    best_state = checkpoint_state
                    time_to_best_seconds = time.perf_counter() - cycle_start
                best_score = selection_state.best_score
                if selection.checkpoint_improved and not selection.patience_improved:
                    checkpoint_only_improvement_count += 1
                if selection.patience_improved:
                    material_patience_improvement_count += 1
                completed_epochs = epoch
                gradient_norm_mean = (
                    float(np.mean(epoch_gradient_norms))
                    if epoch_gradient_norms
                    else float("nan")
                )
                gradient_norm_std = (
                    float(np.std(epoch_gradient_norms))
                    if epoch_gradient_norms
                    else float("nan")
                )
                gradient_norm_cv = (
                    gradient_norm_std / gradient_norm_mean
                    if np.isfinite(gradient_norm_mean) and gradient_norm_mean > 0
                    else float("nan")
                )
                if np.isfinite(gradient_norm_cv):
                    epoch_gradient_norm_cvs.append(gradient_norm_cv)
                batch_size_mean = float(np.mean(epoch_batch_sizes))
                batch_size_std = float(np.std(epoch_batch_sizes))
                labels_per_loss_group_mean = (
                    float(sum(epoch_valid_label_counts) / sum(epoch_loss_group_counts))
                    if epoch_loss_group_counts
                    else float("nan")
                )
                if np.isfinite(labels_per_loss_group_mean):
                    epoch_labels_per_loss_group.append(labels_per_loss_group_mean)
                component_gradient_cosine_median = (
                    float(np.median(epoch_component_gradient_cosines))
                    if epoch_component_gradient_cosines
                    else float("nan")
                )
                if np.isfinite(component_gradient_cosine_median):
                    epoch_component_gradient_cosine_medians.append(
                        component_gradient_cosine_median
                    )
                teacher_gradient_ratio_median = (
                    float(np.median(epoch_teacher_gradient_ratios))
                    if epoch_teacher_gradient_ratios
                    else float("nan")
                )
                if np.isfinite(teacher_gradient_ratio_median):
                    epoch_teacher_gradient_ratio_medians.append(
                        teacher_gradient_ratio_median
                    )
                top_tail_pair_count_median = (
                    float(np.median(epoch_top_tail_pair_counts))
                    if epoch_top_tail_pair_counts
                    else float("nan")
                )
                if np.isfinite(top_tail_pair_count_median):
                    epoch_top_tail_pair_count_medians.append(
                        top_tail_pair_count_median
                    )
                epoch_rows.append(
                    {
                        "trial_id": trial.trial_id,
                        "fold": protocol.fold,
                        "seed": seed,
                        "model_seed": model_seed,
                        "epoch": epoch,
                        "dynamic_skip_epoch_learning_rate": (
                            dynamic_skip_epoch_learning_rate
                        ),
                        "stage": "validation",
                        "train_loss": float(np.mean(losses)),
                        "training_seconds": training_seconds,
                        "model_step_seconds": epoch_model_step_seconds,
                        "data_wait_seconds": epoch_data_wait_seconds,
                        "pcgrad_projection_seconds": epoch_pcgrad_projection_seconds,
                        "pcgrad_horizon_backward_seconds": (
                            epoch_pcgrad_horizon_backward_seconds
                        ),
                        "validation_seconds": validation_seconds,
                        "samples_per_second": len(train_dataset) / training_seconds,
                        "mean_daily_rankic": score,
                        "validation_parameter_source": validation_parameter_source,
                        "teacher_blend_weight": teacher_blend_weight,
                        "rankic_1d": rankic.rankic_by_horizon.get(1, float("nan")),
                        "rankic_2d": rankic.rankic_by_horizon.get(2, float("nan")),
                        "rankic_3d": rankic.rankic_by_horizon.get(3, float("nan")),
                        "rankic_5d": rankic.rankic_by_horizon.get(5, float("nan")),
                        "valid_group_count": rankic.valid_group_count,
                        "improved": selection.checkpoint_improved,
                        "checkpoint_improved": selection.checkpoint_improved,
                        "patience_improved": selection.patience_improved,
                        "epochs_without_material_improvement": (
                            selection_state.epochs_without_material_improvement
                        ),
                        "date_order_epoch": date_order_epoch,
                        "date_order_fingerprint": date_order_fingerprint,
                        "optimizer_step_count": batch_number,
                        "gradient_norm_mean": gradient_norm_mean,
                        "gradient_norm_std": gradient_norm_std,
                        "gradient_norm_cv": gradient_norm_cv,
                        "gradient_norm_max": (
                            max(epoch_gradient_norms)
                            if epoch_gradient_norms
                            else float("nan")
                        ),
                        "batch_size_mean": batch_size_mean,
                        "batch_size_std": batch_size_std,
                        "batch_size_cv": (
                            batch_size_std / batch_size_mean
                            if batch_size_mean > 0
                            else float("nan")
                        ),
                        "batch_size_min": min(epoch_batch_sizes),
                        "batch_size_max": max(epoch_batch_sizes),
                        "loss_group_count_mean": (
                            float(np.mean(epoch_loss_group_counts))
                            if epoch_loss_group_counts
                            else float("nan")
                        ),
                        "loss_group_count_min": (
                            min(epoch_loss_group_counts)
                            if epoch_loss_group_counts
                            else float("nan")
                        ),
                        "loss_group_count_max": (
                            max(epoch_loss_group_counts)
                            if epoch_loss_group_counts
                            else float("nan")
                        ),
                        "valid_label_count_mean": (
                            float(np.mean(epoch_valid_label_counts))
                            if epoch_valid_label_counts
                            else float("nan")
                        ),
                        "valid_label_count_min": (
                            min(epoch_valid_label_counts)
                            if epoch_valid_label_counts
                            else float("nan")
                        ),
                        "valid_label_count_max": (
                            max(epoch_valid_label_counts)
                            if epoch_valid_label_counts
                            else float("nan")
                        ),
                        "labels_per_loss_group_mean": labels_per_loss_group_mean,
                        "top_tail_pair_count_mean": (
                            float(np.mean(epoch_top_tail_pair_counts))
                            if epoch_top_tail_pair_counts
                            else float("nan")
                        ),
                        "top_tail_pair_count_min": (
                            min(epoch_top_tail_pair_counts)
                            if epoch_top_tail_pair_counts
                            else float("nan")
                        ),
                        "top_tail_pair_count_max": (
                            max(epoch_top_tail_pair_counts)
                            if epoch_top_tail_pair_counts
                            else float("nan")
                        ),
                        "component_gradient_cosine_mean": (
                            float(np.mean(epoch_component_gradient_cosines))
                            if epoch_component_gradient_cosines
                            else float("nan")
                        ),
                        "component_gradient_cosine_median": (
                            component_gradient_cosine_median
                        ),
                        "component_gradient_cosine_min": (
                            min(epoch_component_gradient_cosines)
                            if epoch_component_gradient_cosines
                            else float("nan")
                        ),
                        "teacher_gradient_ratio_mean": (
                            float(np.mean(epoch_teacher_gradient_ratios))
                            if epoch_teacher_gradient_ratios
                            else float("nan")
                        ),
                        "teacher_gradient_ratio_median": (
                            teacher_gradient_ratio_median
                        ),
                        "teacher_gradient_ratio_min": (
                            min(epoch_teacher_gradient_ratios)
                            if epoch_teacher_gradient_ratios
                            else float("nan")
                        ),
                        "teacher_gradient_ratio_max": (
                            max(epoch_teacher_gradient_ratios)
                            if epoch_teacher_gradient_ratios
                            else float("nan")
                        ),
                    }
                )
                if (
                    not disable_early_stopping
                    and
                    selection_state.epochs_without_material_improvement
                    >= patience
                ):
                    stopping_reason = "validation_early_stopping"
                    break
            if epoch_parameter_average is not None:
                assert trial.epoch_average_start is not None
                expected_updates = max_epochs - trial.epoch_average_start + 1
                if (
                    completed_epochs != max_epochs
                    or epoch_parameter_average.update_count != expected_updates
                ):
                    raise ContractError("final epoch average coverage drifted")
                best_state = epoch_parameter_average.averaged_state_dict(model)
                best_epoch = completed_epochs
                best_score = score
                best_rankic_by_horizon = dict(rankic.rankic_by_horizon)
                time_to_best_seconds = time.perf_counter() - cycle_start
                stopping_reason = "final_epoch_uniform_average"
            if best_state is None or (
                best_epoch == 0 and not trial.dynamic_skip_frozen_parent
            ):
                raise ContractError(
                    f"trial {trial.trial_id} fold {protocol.fold} has no finite validation RankIC"
                )
            model.load_state_dict(best_state)
            trainable_parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            frozen_parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if not parameter.requires_grad
            )
            frozen_parent_state_drift_max = float("nan")
            if trial.dynamic_skip_frozen_parent:
                if not isinstance(model, ShapeResidualDynamicHorizonSkipTCN):
                    raise ContractError(
                        "frozen parent audit requires shape residual TCN"
                    )
                frozen_parent_state_drift_max = (
                    model.frozen_parent_state_drift_max()
                )
                if frozen_parent_state_drift_max != 0.0:
                    raise ContractError("frozen parent state changed during training")
            metadata_builder = getattr(model, "receipt_metadata", None)
            model_metadata = metadata_builder() if callable(metadata_builder) else {}
            simplex_weights = model_metadata.get("simplex_weights")
            day_weights = model_metadata.get("day_weights")
            intraday_weights = model_metadata.get("intraday_weights")
            readout_identity = model_metadata.get("readout")
            day_negative_weight_count = model_metadata.get("day_negative_weight_count")
            intraday_negative_weight_count = model_metadata.get(
                "intraday_negative_weight_count"
            )
            day_weight_sum = model_metadata.get("day_weight_sum")
            intraday_weight_sum = model_metadata.get("intraday_weight_sum")
            day_residual_l2 = model_metadata.get("day_residual_l2")
            intraday_residual_l2 = model_metadata.get("intraday_residual_l2")
            day_simplex_weights = model_metadata.get("day_simplex_weights")
            intraday_simplex_weights = model_metadata.get("intraday_simplex_weights")
            base_temporal_parameter_count = model_metadata.get(
                "base_temporal_parameter_count"
            )
            residual_parameter_count = model_metadata.get("residual_parameter_count")
            market_gate_output_l2 = model_metadata.get("market_gate_output_l2")
            dynamic_attention_output_l2 = model_metadata.get(
                "dynamic_attention_output_l2"
            )
            dynamic_attention_output_weight_l2 = model_metadata.get(
                "dynamic_attention_output_weight_l2"
            )
            dynamic_attention_output_bias_l2 = model_metadata.get(
                "dynamic_attention_output_bias_l2"
            )
            dynamic_skip_output_weight_l2 = model_metadata.get(
                "dynamic_skip_output_weight_l2"
            )
            dynamic_skip_output_bias_l2 = model_metadata.get(
                "dynamic_skip_output_bias_l2"
            )
            dynamic_skip_amplitude_feature = model_metadata.get(
                "dynamic_skip_amplitude_feature"
            )
            dynamic_skip_scorer_input_width = model_metadata.get(
                "dynamic_skip_scorer_input_width"
            )
            dynamic_skip_amplitude_projection_weight_l2 = model_metadata.get(
                "dynamic_skip_amplitude_projection_weight_l2"
            )
            dynamic_skip_normalization_parameter_count = model_metadata.get(
                "dynamic_skip_normalization_parameter_count"
            )
            dynamic_skip_raw_parameter_count = model_metadata.get(
                "dynamic_skip_raw_parameter_count"
            )
            dynamic_skip_shape_residual_parameter_count = model_metadata.get(
                "dynamic_skip_shape_residual_parameter_count"
            )
            dynamic_skip_shape_normalization_parameter_count = model_metadata.get(
                "dynamic_skip_shape_normalization_parameter_count"
            )
            dynamic_skip_shape_output_weight_l2 = model_metadata.get(
                "dynamic_skip_shape_output_weight_l2"
            )
            dynamic_skip_shape_output_bias_l2 = model_metadata.get(
                "dynamic_skip_shape_output_bias_l2"
            )
            checkpoint_key = f"{trial.trial_id}-fold-{protocol.fold}"
            best_states[checkpoint_key] = best_state
            complete_cycle_seconds = time.perf_counter() - cycle_start
            total_training_seconds = total_model_step_seconds + total_data_wait_seconds
            loss_identity = {
                "smooth_l1": "smooth-l1",
                "grouped_smooth_l1": "date-grouped-smooth-l1",
                "rank_objective": "smooth-l1+0.1-pairwise-logistic",
                "soft_rankic": (
                    f"smooth-l1+{trial.soft_rankic_weight:g}-soft-rankic"
                    f"-tau-{trial.soft_rank_temperature:g}"
                ),
                "top_tail": (
                    f"smooth-l1+{trial.top_tail_weight:g}-top-tail"
                    f"-fraction-{trial.top_tail_fraction:g}"
                    f"-tau-{trial.top_tail_temperature:g}"
                ),
                "teacher_listwise": (
                    "smooth-l1+gradient-normalized-teacher-listwise"
                    f"-ratio-{trial.teacher_listwise_gradient_ratio:g}"
                    f"-tau-{trial.teacher_listwise_temperature:g}"
                ),
                "pcgrad": "smooth-l1-pcgrad",
            }[trial.strategy]
            if (
                trial.strategy == "grouped_smooth_l1"
                and trial.grouped_smooth_l1_reduction == "date_horizon_mean"
            ):
                loss_identity = "date-horizon-equal-smooth-l1"
            if trial.strategy == "pcgrad" and trial.pcgrad_blocks is not None:
                loss_identity = "smooth-l1-local-pcgrad:block-" + ",block-".join(
                    map(str, trial.pcgrad_blocks)
                )
            if trial.teacher_blend_start_weight is not None:
                loss_identity = (
                    "smooth-l1+linear-teacher-blend-"
                    f"{trial.teacher_blend_start_weight:g}-to-"
                    f"{trial.teacher_blend_end_weight:g}"
                )
            leaderboard_rows.append(
                {
                    "trial_id": trial.trial_id,
                    "model": trial.trial_id,
                    "model_kind": trial.model_kind,
                    "fold": protocol.fold,
                    "best_epoch": best_epoch,
                    "completed_epochs": completed_epochs,
                    "best_mean_daily_rankic": best_score,
                    "rankic": best_score,
                    "rankic_1d": best_rankic_by_horizon.get(1, float("nan")),
                    "rankic_2d": best_rankic_by_horizon.get(2, float("nan")),
                    "rankic_3d": best_rankic_by_horizon.get(3, float("nan")),
                    "rankic_5d": best_rankic_by_horizon.get(5, float("nan")),
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "trainable_parameter_count": trainable_parameter_count,
                    "frozen_parameter_count": frozen_parameter_count,
                    "frozen_parent_state_drift_max": (
                        frozen_parent_state_drift_max
                    ),
                    "parent_prediction_max_abs_error": (
                        parent_prediction_max_abs_error
                    ),
                    "baseline_epoch": (
                        0 if trial.dynamic_skip_frozen_parent else None
                    ),
                    "baseline_mean_daily_rankic": baseline_score,
                    "baseline_rankic_1d": baseline_rankic_by_horizon.get(
                        1, float("nan")
                    ),
                    "baseline_rankic_2d": baseline_rankic_by_horizon.get(
                        2, float("nan")
                    ),
                    "baseline_rankic_3d": baseline_rankic_by_horizon.get(
                        3, float("nan")
                    ),
                    "baseline_rankic_5d": baseline_rankic_by_horizon.get(
                        5, float("nan")
                    ),
                    "checkpoint_min_delta": effective_checkpoint_min_delta,
                    "patience_min_delta": min_delta,
                    "checkpoint_selection_identity": (
                        "best-any-strict-improvement+patience-material-"
                        f"{min_delta:g}"
                        if effective_checkpoint_min_delta == 0
                        and min_delta > 0
                        else f"coupled-best-and-patience-{min_delta:g}"
                    ),
                    "checkpoint_only_improvement_count": (
                        checkpoint_only_improvement_count
                    ),
                    "material_patience_improvement_count": (
                        material_patience_improvement_count
                    ),
                    "samples_per_second": (total_samples / total_training_seconds),
                    "model_step_samples_per_second": (
                        total_samples / total_model_step_seconds
                    ),
                    "model_step_seconds": total_model_step_seconds,
                    "data_wait_seconds": total_data_wait_seconds,
                    "validation_seconds": total_validation_seconds,
                    "pcgrad_projection_seconds": total_pcgrad_projection_seconds,
                    "pcgrad_horizon_backward_seconds": (
                        total_pcgrad_horizon_backward_seconds
                    ),
                    "complete_cycle_seconds": complete_cycle_seconds,
                    "time_to_best_seconds": time_to_best_seconds,
                    "stopping_reason": stopping_reason,
                    "channels": trial.channels,
                    "kernel_size": trial.kernel_size,
                    "dilations": ",".join(map(str, trial.dilations)),
                    "receptive_field": observed_receptive_field,
                    "dropout": trial.dropout,
                    "block_dropout": trial.dropout,
                    "head_dropout": trial.head_dropout,
                    "dropout_kind": trial.dropout_kind,
                    "optimizer": optimizer_name,
                    "weight_decay": trial.weight_decay,
                    "learning_rate": trial.learning_rate,
                    "adapter_learning_rate": trial.adapter_learning_rate,
                    "residual_learning_rate": trial.residual_learning_rate,
                    "market_context_dim": (
                        trial.market_context_dim
                        if trial.model_kind == "market_conditioned_temporal_context"
                        else None
                    ),
                    "market_context_hidden": (
                        trial.market_context_hidden
                        if trial.model_kind == "market_conditioned_temporal_context"
                        else None
                    ),
                    "market_gate_scale": (
                        trial.market_gate_scale
                        if trial.model_kind == "market_conditioned_temporal_context"
                        else None
                    ),
                    "market_gate_output_l2": market_gate_output_l2,
                    "dynamic_attention_hidden": (
                        trial.dynamic_attention_hidden
                        if trial.model_kind == "dynamic_temporal_context"
                        else None
                    ),
                    "dynamic_attention_scale": (
                        trial.dynamic_attention_scale
                        if trial.model_kind == "dynamic_temporal_context"
                        else None
                    ),
                    "dynamic_attention_learning_rate": (
                        trial.dynamic_attention_learning_rate
                        if trial.model_kind == "dynamic_temporal_context"
                        else None
                    ),
                    "dynamic_attention_output_l2": dynamic_attention_output_l2,
                    "dynamic_attention_output_weight_l2": (
                        dynamic_attention_output_weight_l2
                    ),
                    "dynamic_attention_output_bias_l2": (
                        dynamic_attention_output_bias_l2
                    ),
                    "dynamic_skip_hidden": (
                        trial.dynamic_skip_hidden
                        if trial.model_kind == "dynamic_horizon_skip"
                        else None
                    ),
                    "dynamic_skip_scale": (
                        trial.dynamic_skip_scale
                        if trial.model_kind == "dynamic_horizon_skip"
                        else None
                    ),
                    "dynamic_skip_learning_rate": (
                        trial.dynamic_skip_learning_rate
                        if trial.model_kind == "dynamic_horizon_skip"
                        else None
                    ),
                    "dynamic_skip_warmup_epochs": (
                        trial.dynamic_skip_warmup_epochs
                        if trial.model_kind == "dynamic_horizon_skip"
                        else None
                    ),
                    "dynamic_skip_token_normalization": (
                        trial.dynamic_skip_token_normalization
                        if trial.model_kind == "dynamic_horizon_skip"
                        else None
                    ),
                    "dynamic_skip_output_weight_l2": (
                        dynamic_skip_output_weight_l2
                    ),
                    "dynamic_skip_output_bias_l2": dynamic_skip_output_bias_l2,
                    "dynamic_skip_amplitude_feature": (
                        dynamic_skip_amplitude_feature
                    ),
                    "dynamic_skip_scorer_input_width": (
                        dynamic_skip_scorer_input_width
                    ),
                    "dynamic_skip_amplitude_projection_weight_l2": (
                        dynamic_skip_amplitude_projection_weight_l2
                    ),
                    "dynamic_skip_normalization_parameter_count": (
                        dynamic_skip_normalization_parameter_count
                    ),
                    "dynamic_skip_raw_parameter_count": (
                        dynamic_skip_raw_parameter_count
                    ),
                    "dynamic_skip_shape_residual": (
                        trial.dynamic_skip_shape_residual
                        if trial.model_kind == "dynamic_horizon_skip"
                        else None
                    ),
                    "dynamic_skip_frozen_parent": (
                        trial.dynamic_skip_frozen_parent
                        if trial.model_kind == "dynamic_horizon_skip"
                        else None
                    ),
                    "dynamic_skip_shape_residual_scale": (
                        trial.dynamic_skip_shape_residual_scale
                        if trial.model_kind == "dynamic_horizon_skip"
                        and trial.dynamic_skip_shape_residual
                        else None
                    ),
                    "dynamic_skip_shape_residual_parameter_count": (
                        dynamic_skip_shape_residual_parameter_count
                    ),
                    "dynamic_skip_shape_normalization_parameter_count": (
                        dynamic_skip_shape_normalization_parameter_count
                    ),
                    "dynamic_skip_shape_output_weight_l2": (
                        dynamic_skip_shape_output_weight_l2
                    ),
                    "dynamic_skip_shape_output_bias_l2": (
                        dynamic_skip_shape_output_bias_l2
                    ),
                    "market_context_identity": (
                        market_context.identity
                        if trial.model_kind == "market_conditioned_temporal_context"
                        and market_context is not None
                        else None
                    ),
                    "market_context_scaler_identity": (
                        context_standardizer.identity
                        if context_standardizer is not None
                        else None
                    ),
                    "optimizer_group_identity": (
                        optimizer_bundle.parameter_group_identity
                    ),
                    "adapter_parameter_count": (
                        optimizer_bundle.adapter_parameter_count
                    ),
                    "optimizer_residual_parameter_count": (
                        optimizer_bundle.residual_parameter_count
                    ),
                    "optimizer_dynamic_attention_parameter_count": (
                        optimizer_bundle.dynamic_attention_parameter_count
                    ),
                    "optimizer_dynamic_skip_parameter_count": (
                        optimizer_bundle.dynamic_skip_parameter_count
                    ),
                    "base_temporal_parameter_count": (base_temporal_parameter_count),
                    "residual_parameter_count": residual_parameter_count,
                    "batch_size": trial.batch_size,
                    "observed_batch_size_median": float(
                        np.median(observed_batch_sizes)
                    ),
                    "observed_batch_size_min": min(observed_batch_sizes),
                    "observed_batch_size_max": max(observed_batch_sizes),
                    "date_batch_order": trial.date_batch_order,
                    "date_order_fingerprint_count": len(
                        date_order_fingerprints
                    ),
                    "median_epoch_gradient_norm_cv": (
                        float(np.median(epoch_gradient_norm_cvs))
                        if epoch_gradient_norm_cvs
                        else float("nan")
                    ),
                    "grouped_smooth_l1_reduction": (
                        trial.grouped_smooth_l1_reduction
                        if trial.strategy == "grouped_smooth_l1"
                        else None
                    ),
                    "ema_decay": trial.ema_decay,
                    "ema_update_count": (
                        parameter_ema.update_count
                        if parameter_ema is not None
                        else 0
                    ),
                    "epoch_average_start": trial.epoch_average_start,
                    "epoch_average_update_count": (
                        epoch_parameter_average.update_count
                        if epoch_parameter_average is not None
                        else 0
                    ),
                    "teacher_blend_start_weight": (
                        trial.teacher_blend_start_weight
                    ),
                    "teacher_blend_end_weight": trial.teacher_blend_end_weight,
                    "checkpoint_parameter_source": (
                        "ema"
                        if parameter_ema is not None
                        else (
                            "epoch_uniform_average_final"
                            if epoch_parameter_average is not None
                            else "raw"
                        )
                    ),
                    "training_target_override": (
                        training_target_overrides is not None
                    ),
                    "training_teacher_target": (
                        trial.strategy == "teacher_listwise"
                    ),
                    "training_target_schedule": (
                        trial.teacher_blend_start_weight is not None
                    ),
                    "median_labels_per_loss_group": (
                        float(np.median(epoch_labels_per_loss_group))
                        if epoch_labels_per_loss_group
                        else float("nan")
                    ),
                    "median_top_tail_pair_count": (
                        float(np.median(epoch_top_tail_pair_count_medians))
                        if epoch_top_tail_pair_count_medians
                        else float("nan")
                    ),
                    "median_component_gradient_cosine": (
                        float(np.median(epoch_component_gradient_cosine_medians))
                        if epoch_component_gradient_cosine_medians
                        else float("nan")
                    ),
                    "median_teacher_gradient_ratio": (
                        float(np.median(epoch_teacher_gradient_ratio_medians))
                        if epoch_teacher_gradient_ratio_medians
                        else float("nan")
                    ),
                    "seed": seed,
                    "model_seed": model_seed,
                    "strategy": trial.strategy,
                    "padding_mode": trial.padding_mode,
                    "pcgrad_blocks": (
                        json.dumps(
                            list(
                                range(len(trial.dilations))
                                if trial.pcgrad_blocks is None
                                else trial.pcgrad_blocks
                            ),
                            separators=(",", ":"),
                        )
                        if trial.strategy == "pcgrad"
                        else None
                    ),
                    "pcgrad_horizons": (
                        json.dumps(
                            list(
                                HORIZONS
                                if trial.pcgrad_horizons is None
                                else trial.pcgrad_horizons
                            ),
                            separators=(",", ":"),
                        )
                        if trial.strategy == "pcgrad"
                        else None
                    ),
                    "bars_per_day": trial.bars_per_day,
                    "residual_scale": (
                        trial.residual_scale
                        if trial.model_kind
                        in {
                            "stabilized_temporal_context",
                            "decoupled_temporal_context",
                        }
                        else None
                    ),
                    "soft_rankic_weight": (
                        trial.soft_rankic_weight
                        if trial.strategy == "soft_rankic"
                        else None
                    ),
                    "soft_rank_temperature": (
                        trial.soft_rank_temperature
                        if trial.strategy == "soft_rankic"
                        else None
                    ),
                    "top_tail_weight": (
                        trial.top_tail_weight
                        if trial.strategy == "top_tail"
                        else None
                    ),
                    "top_tail_fraction": (
                        trial.top_tail_fraction
                        if trial.strategy == "top_tail"
                        else None
                    ),
                    "top_tail_temperature": (
                        trial.top_tail_temperature
                        if trial.strategy == "top_tail"
                        else None
                    ),
                    "teacher_listwise_gradient_ratio": (
                        trial.teacher_listwise_gradient_ratio
                        if trial.strategy == "teacher_listwise"
                        else None
                    ),
                    "teacher_listwise_temperature": (
                        trial.teacher_listwise_temperature
                        if trial.strategy == "teacher_listwise"
                        else None
                    ),
                    "precision": "float32",
                    "data_identity": identities.get("data", "unregistered"),
                    "fold_identity": identities.get("fold_manifest", "unregistered"),
                    "evaluation_identity": identities.get("evaluation", "unregistered"),
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "min_delta": min_delta,
                    "loss_identity": loss_identity,
                    "batching_identity": (
                        "date-grouped"
                        if trial.strategy
                        in {
                            "grouped_smooth_l1",
                            "rank_objective",
                            "soft_rankic",
                            "top_tail",
                            "teacher_listwise",
                        }
                        else "seeded-random"
                    ),
                    "readout_identity": readout_identity,
                    "infra_identity": (
                        "padding-chomp"
                        if trial.padding_mode == "chomp"
                        else "explicit-left-pad"
                    ),
                    "candidate_config_identity": _trial_config_identity(
                        trial,
                        max_epochs=max_epochs,
                        patience=patience,
                        min_delta=min_delta,
                        checkpoint_min_delta=(
                            effective_checkpoint_min_delta
                        ),
                    ),
                    "simplex_weights": (
                        json.dumps(simplex_weights, separators=(",", ":"))
                        if simplex_weights is not None
                        else None
                    ),
                    "day_weights": (
                        json.dumps(day_weights, separators=(",", ":"))
                        if day_weights is not None
                        else None
                    ),
                    "intraday_weights": (
                        json.dumps(intraday_weights, separators=(",", ":"))
                        if intraday_weights is not None
                        else None
                    ),
                    "day_simplex_weights": (
                        json.dumps(day_simplex_weights, separators=(",", ":"))
                        if day_simplex_weights is not None
                        else None
                    ),
                    "intraday_simplex_weights": (
                        json.dumps(intraday_simplex_weights, separators=(",", ":"))
                        if intraday_simplex_weights is not None
                        else None
                    ),
                    "day_negative_weight_count": (
                        json.dumps(day_negative_weight_count, separators=(",", ":"))
                        if day_negative_weight_count is not None
                        else None
                    ),
                    "intraday_negative_weight_count": (
                        json.dumps(
                            intraday_negative_weight_count, separators=(",", ":")
                        )
                        if intraday_negative_weight_count is not None
                        else None
                    ),
                    "day_weight_sum": (
                        json.dumps(day_weight_sum, separators=(",", ":"))
                        if day_weight_sum is not None
                        else None
                    ),
                    "intraday_weight_sum": (
                        json.dumps(intraday_weight_sum, separators=(",", ":"))
                        if intraday_weight_sum is not None
                        else None
                    ),
                    "day_residual_l2": (
                        json.dumps(day_residual_l2, separators=(",", ":"))
                        if day_residual_l2 is not None
                        else None
                    ),
                    "intraday_residual_l2": (
                        json.dumps(intraday_residual_l2, separators=(",", ":"))
                        if intraday_residual_l2 is not None
                        else None
                    ),
                    "sealed_test_accessed": False,
                }
            )
    return TCNTuningResult(
        epoch_history=pd.DataFrame(epoch_rows),
        leaderboard=pd.DataFrame(leaderboard_rows),
        best_states=best_states,
        epoch_states=epoch_states,
    )


def select_tcn_candidate(
    leaderboard: pd.DataFrame,
    *,
    control_trial_id: str,
    min_improvement: float,
) -> TCNTuningDecision:
    """Select deterministically and open top50 only on cross-fold validation gain."""

    horizon_columns = ["rankic_1d", "rankic_2d", "rankic_3d", "rankic_5d"]
    required = {
        "trial_id",
        "fold",
        "best_mean_daily_rankic",
        "parameter_count",
        "samples_per_second",
        *horizon_columns,
    }
    if missing := sorted(required.difference(leaderboard.columns)):
        raise ContractError(f"TCN leaderboard missing columns: {', '.join(missing)}")
    if min_improvement < 0:
        raise ContractError("minimum improvement cannot be negative")
    if control_trial_id not in set(leaderboard["trial_id"]):
        raise ContractError("TCN leaderboard is missing the control trial")
    if not np.isfinite(
        leaderboard[
            ["best_mean_daily_rankic", "parameter_count", "samples_per_second"]
            + horizon_columns
        ].to_numpy(dtype="float64")
    ).all():
        raise ContractError("TCN leaderboard contains non-finite selection evidence")
    fold_sets = {
        str(trial_id): set(group["fold"].astype(int))
        for trial_id, group in leaderboard.groupby("trial_id", observed=True)
    }
    expected_folds = fold_sets[control_trial_id]
    if not expected_folds or any(
        folds != expected_folds for folds in fold_sets.values()
    ):
        raise ContractError("TCN leaderboard trials must cover identical folds")
    summary_rows = []
    for trial_id, group in leaderboard.groupby("trial_id", observed=True):
        if group["parameter_count"].nunique() != 1:
            raise ContractError("TCN parameter count must be constant across folds")
        summary_rows.append(
            {
                "trial_id": str(trial_id),
                "mean_rankic": float(group["best_mean_daily_rankic"].mean()),
                "parameter_count": int(group["parameter_count"].iloc[0]),
                "samples_per_second": float(group["samples_per_second"].mean()),
            }
        )
    ranked = pd.DataFrame(summary_rows).sort_values(
        ["mean_rankic", "parameter_count", "samples_per_second", "trial_id"],
        ascending=[False, True, False, True],
        kind="mergesort",
    )
    selected_id = str(ranked.iloc[0]["trial_id"])
    selected = leaderboard.loc[leaderboard["trial_id"] == selected_id]
    control = leaderboard.loc[leaderboard["trial_id"] == control_trial_id]
    improvement = float(
        selected["best_mean_daily_rankic"].mean()
        - control["best_mean_daily_rankic"].mean()
    )
    non_degrading = sum(
        float(selected[column].mean()) >= float(control[column].mean())
        for column in horizon_columns
    )
    pass_gate = (
        selected["best_mean_daily_rankic"].gt(0).all()
        and improvement >= min_improvement
        and non_degrading >= 2
    )
    return TCNTuningDecision(
        selected_trial_id=selected_id,
        status="expand_top50" if pass_gate else "stop_no_validation_gain",
        mean_improvement=improvement,
        non_degrading_horizon_count=non_degrading,
    )


def build_validation_rankic_plan(
    sample_positions: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
) -> ValidationRankICPlan:
    """Resolve fixed validation joins and target ranks once per walk-forward fold."""

    required_index = {"sample_position", "sample_id"}
    if missing := sorted(required_index.difference(window_index.columns)):
        raise ContractError(f"window index missing columns: {', '.join(missing)}")
    required_labels = {
        "sample_id",
        "signal_date",
        "horizon",
        "rank_target",
        "valid",
    }
    if missing := sorted(required_labels.difference(labels.columns)):
        raise ContractError(f"labels missing columns: {', '.join(missing)}")
    if window_index["sample_position"].duplicated().any():
        raise ContractError("window index sample positions must be unique")
    if labels.duplicated(["sample_id", "horizon"]).any():
        raise ContractError("validation labels must be unique by sample and horizon")

    resolved_positions = np.asarray(sample_positions, dtype="int64").copy()
    resolved_positions.setflags(write=False)
    sample_by_position = window_index.set_index("sample_position")[
        "sample_id"
    ].to_dict()
    label_by_key = labels.set_index(["sample_id", "horizon"])
    grouped_rows: dict[tuple[Any, int], list[int]] = {}
    grouped_targets: dict[tuple[Any, int], list[float]] = {}
    for row_number, position in enumerate(resolved_positions):
        sample_id = sample_by_position.get(int(position))
        if sample_id is None:
            raise ContractError(f"validation sample position is unknown: {position}")
        for column, horizon in enumerate(HORIZONS):
            key = (sample_id, horizon)
            if key not in label_by_key.index:
                continue
            label = label_by_key.loc[key]
            if not bool(label["valid"]):
                continue
            group_key = (label["signal_date"], int(horizon))
            grouped_rows.setdefault(group_key, []).append(row_number)
            grouped_targets.setdefault(group_key, []).append(
                float(label["rank_target"])
            )

    horizon_columns = {horizon: column for column, horizon in enumerate(HORIZONS)}
    groups = []
    for group_key, rows in grouped_rows.items():
        signal_date, horizon = group_key
        targets = np.asarray(grouped_targets[group_key], dtype="float64")
        if len(rows) < 2 or np.unique(targets).size < 2:
            continue
        target_ranks = np.asarray(rankdata(targets), dtype="float64")
        centered_targets = target_ranks - target_ranks.mean()
        score_rows = np.asarray(rows, dtype="int64")
        score_rows.setflags(write=False)
        centered_targets.setflags(write=False)
        groups.append(
            _ValidationRankICGroup(
                signal_date=str(signal_date),
                horizon=int(horizon),
                score_column=horizon_columns[int(horizon)],
                score_rows=score_rows,
                centered_target_ranks=centered_targets,
                target_sum_squares=float(np.dot(centered_targets, centered_targets)),
            )
        )
    return ValidationRankICPlan(
        sample_positions=resolved_positions,
        groups=tuple(groups),
    )


def cross_sectional_validation_rankic(
    scores: np.ndarray,
    sample_positions: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
) -> ValidationRankIC:
    """Average Spearman RankIC only after grouping by signal date and horizon."""

    return build_validation_rankic_plan(
        sample_positions, window_index, labels
    ).evaluate(scores, sample_positions)


def validate_tcn_tuning_plan(
    trials: Sequence[TCNTuningTrial],
    *,
    input_steps: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> tuple[TCNTuningTrial, ...]:
    """Fail closed on invalid or unbounded tuning plans."""

    if not trials:
        raise ContractError("TCN tuning plan requires at least one trial")
    if input_steps <= 0 or max_epochs <= 0:
        raise ContractError("input_steps and max_epochs must be positive")
    if patience <= 0 or patience >= max_epochs:
        raise ContractError("patience must be positive and smaller than max_epochs")
    if min_delta < 0:
        raise ContractError("min_delta cannot be negative")
    trial_ids = [trial.trial_id for trial in trials]
    if any(not trial_id for trial_id in trial_ids):
        raise ContractError("trial IDs must be non-empty")
    if len(set(trial_ids)) != len(trial_ids):
        raise ContractError("trial IDs must be unique")
    for trial in trials:
        if trial.channels <= 0 or trial.kernel_size <= 1 or trial.batch_size <= 0:
            raise ContractError(
                "channels, kernel_size, and batch_size must be positive"
            )
        if not trial.dilations or any(value <= 0 for value in trial.dilations):
            raise ContractError("dilations must be non-empty and positive")
        if not 0 <= trial.dropout < 1:
            raise ContractError("dropout must be in [0, 1)")
        if not 0 <= trial.head_dropout < 1:
            raise ContractError("head dropout must be in [0, 1)")
        if trial.dropout_kind not in {"element", "channel"}:
            raise ContractError("dropout kind must be element or channel")
        if trial.learning_rate <= 0:
            raise ContractError("learning_rate must be positive")
        if trial.weight_decay < 0:
            raise ContractError("weight decay cannot be negative")
        if trial.model_kind not in {
            "bai",
            "lite",
            "horizon_skip",
            "temporal_context",
            "signed_temporal_context",
            "stabilized_temporal_context",
            "decoupled_temporal_context",
            "market_conditioned_temporal_context",
            "dynamic_temporal_context",
            "dynamic_horizon_skip",
        }:
            raise ContractError(
                "TCN model kind must be bai, lite, horizon_skip, temporal_context, "
                "signed_temporal_context, stabilized_temporal_context, "
                "decoupled_temporal_context, market_conditioned_temporal_context, "
                "dynamic_temporal_context, or dynamic_horizon_skip"
            )
        if trial.strategy not in {
            "smooth_l1",
            "grouped_smooth_l1",
            "rank_objective",
            "soft_rankic",
            "top_tail",
            "teacher_listwise",
            "pcgrad",
        }:
            raise ContractError("TCN strategy is unsupported")
        if trial.strategy != "pcgrad" and (
            trial.pcgrad_blocks is not None or trial.pcgrad_horizons is not None
        ):
            raise ContractError("PCGrad scope is only valid for PCGrad trials")
        if trial.pcgrad_blocks is not None and (
            not trial.pcgrad_blocks
            or tuple(sorted(set(trial.pcgrad_blocks))) != trial.pcgrad_blocks
            or min(trial.pcgrad_blocks) < 0
            or max(trial.pcgrad_blocks) >= len(trial.dilations)
        ):
            raise ContractError("PCGrad block scope is invalid")
        if trial.pcgrad_horizons is not None and (
            len(trial.pcgrad_horizons) < 2
            or tuple(sorted(set(trial.pcgrad_horizons))) != trial.pcgrad_horizons
            or not set(trial.pcgrad_horizons).issubset(set(HORIZONS))
        ):
            raise ContractError("PCGrad horizon scope is invalid")
        if trial.padding_mode not in {"explicit", "chomp"}:
            raise ContractError("TCN padding mode must be explicit or chomp")
        if trial.model_kind == "bai" and trial.head_dropout != 0:
            raise ContractError("Bai TCN does not support head dropout")
        if trial.model_kind == "bai" and trial.dropout_kind != "element":
            raise ContractError("Bai TCN does not support channel dropout")
        if trial.model_kind == "bai" and (
            trial.strategy != "smooth_l1" or trial.padding_mode != "explicit"
        ):
            raise ContractError("Bai TCN does not support v9 strategies or chomp")
        if trial.model_kind == "horizon_skip" and (
            trial.head_dropout != 0 or trial.dropout_kind != "element"
        ):
            raise ContractError("horizon skip requires the frozen lite dropout path")
        if trial.model_kind == "dynamic_horizon_skip" and (
            trial.head_dropout != 0 or trial.dropout_kind != "element"
        ):
            raise ContractError(
                "dynamic horizon skip requires the frozen lite dropout path"
            )
        if trial.model_kind == "temporal_context" and (
            trial.head_dropout != 0 or trial.dropout_kind != "element"
        ):
            raise ContractError(
                "temporal context requires the frozen lite dropout path"
            )
        if trial.model_kind == "signed_temporal_context" and (
            trial.head_dropout != 0 or trial.dropout_kind != "element"
        ):
            raise ContractError(
                "signed temporal context requires the frozen lite dropout path"
            )
        if trial.model_kind == "stabilized_temporal_context" and (
            trial.head_dropout != 0 or trial.dropout_kind != "element"
        ):
            raise ContractError(
                "stabilized temporal context requires the frozen lite dropout path"
            )
        if trial.model_kind == "decoupled_temporal_context" and (
            trial.head_dropout != 0 or trial.dropout_kind != "element"
        ):
            raise ContractError(
                "decoupled temporal context requires the frozen lite dropout path"
            )
        if trial.model_kind == "market_conditioned_temporal_context" and (
            trial.head_dropout != 0 or trial.dropout_kind != "element"
        ):
            raise ContractError(
                "market-conditioned temporal context requires the frozen lite dropout path"
            )
        if trial.model_kind == "dynamic_temporal_context" and (
            trial.head_dropout != 0 or trial.dropout_kind != "element"
        ):
            raise ContractError(
                "dynamic temporal context requires the frozen lite dropout path"
            )
        if trial.model_kind in {
            "temporal_context",
            "signed_temporal_context",
            "stabilized_temporal_context",
            "decoupled_temporal_context",
            "market_conditioned_temporal_context",
            "dynamic_temporal_context",
        } and (trial.bars_per_day <= 0 or input_steps % trial.bars_per_day != 0):
            raise ContractError(
                "temporal context input steps must be divisible by bars per day"
            )
        if trial.model_kind == "stabilized_temporal_context":
            if (
                not np.isfinite(trial.residual_scale)
                or not 0 < trial.residual_scale <= 0.5
            ):
                raise ContractError(
                    "stabilized temporal residual scale must be in (0, 0.5]"
                )
            if (
                trial.adapter_learning_rate is None
                or not np.isfinite(trial.adapter_learning_rate)
                or trial.adapter_learning_rate <= 0
                or trial.adapter_learning_rate > trial.learning_rate
            ):
                raise ContractError(
                    "stabilized adapter learning rate must be positive and no greater "
                    "than the base learning rate"
                )
            if trial.strategy != "smooth_l1":
                raise ContractError("stabilized temporal context requires smooth_l1")
        elif trial.adapter_learning_rate is not None:
            raise ContractError(
                "adapter learning rate is only valid for stabilized temporal context"
            )
        if trial.model_kind == "decoupled_temporal_context":
            if (
                not np.isfinite(trial.residual_scale)
                or not 0 < trial.residual_scale <= 0.5
            ):
                raise ContractError(
                    "decoupled temporal residual scale must be in (0, 0.5]"
                )
            if (
                trial.residual_learning_rate is None
                or not np.isfinite(trial.residual_learning_rate)
                or trial.residual_learning_rate <= 0
                or trial.residual_learning_rate > trial.learning_rate
            ):
                raise ContractError(
                    "decoupled residual learning rate must be positive and no greater "
                    "than the base learning rate"
                )
            if trial.strategy != "smooth_l1":
                raise ContractError("decoupled temporal context requires smooth_l1")
        elif trial.residual_learning_rate is not None:
            raise ContractError(
                "residual learning rate is only valid for decoupled temporal context"
            )
        if trial.model_kind == "market_conditioned_temporal_context":
            if trial.market_context_dim <= 0 or trial.market_context_hidden <= 0:
                raise ContractError("market context dimensions must be positive")
            if (
                not np.isfinite(trial.market_gate_scale)
                or not 0 < trial.market_gate_scale <= 0.5
            ):
                raise ContractError("market gate scale must be in (0, 0.5]")
            if trial.strategy != "smooth_l1":
                raise ContractError(
                    "market-conditioned temporal context requires smooth_l1"
                )
        if trial.model_kind == "dynamic_temporal_context":
            if trial.dynamic_attention_hidden <= 0:
                raise ContractError("dynamic attention hidden size must be positive")
            if (
                not np.isfinite(trial.dynamic_attention_scale)
                or not 0 < trial.dynamic_attention_scale <= 1.0
            ):
                raise ContractError("dynamic attention scale must be in (0, 1]")
            if trial.strategy != "smooth_l1":
                raise ContractError("dynamic temporal context requires smooth_l1")
            if trial.dynamic_attention_learning_rate is not None and (
                not np.isfinite(trial.dynamic_attention_learning_rate)
                or trial.dynamic_attention_learning_rate <= 0
                or trial.dynamic_attention_learning_rate
                > 10 * trial.learning_rate
            ):
                raise ContractError(
                    "dynamic attention learning rate must be positive and no greater "
                    "than ten times the base learning rate"
                )
        elif trial.dynamic_attention_learning_rate is not None:
            raise ContractError(
                "dynamic attention learning rate is only valid for dynamic temporal context"
            )
        if trial.model_kind == "dynamic_horizon_skip":
            if trial.dynamic_skip_hidden <= 0:
                raise ContractError("dynamic skip hidden size must be positive")
            if (
                not np.isfinite(trial.dynamic_skip_scale)
                or not 0 < trial.dynamic_skip_scale <= 1.0
            ):
                raise ContractError("dynamic skip scale must be in (0, 1]")
            if trial.strategy not in {"smooth_l1", "teacher_listwise"} and not (
                trial.dynamic_skip_frozen_parent
                and trial.dynamic_skip_shape_residual
                and trial.strategy
                in {"grouped_smooth_l1", "soft_rankic", "top_tail"}
            ):
                raise ContractError(
                    "dynamic horizon skip grouped objectives require frozen parent "
                    "shape residual"
                )
            if trial.dynamic_skip_token_normalization not in {
                "none",
                "layer_norm",
                "shape_log_rms",
            }:
                raise ContractError(
                    "dynamic skip token normalization is unsupported"
                )
            if trial.dynamic_skip_shape_residual:
                if trial.dynamic_skip_token_normalization != "none":
                    raise ContractError(
                        "dynamic skip shape residual requires raw scorer inputs"
                    )
                if (
                    not np.isfinite(trial.dynamic_skip_shape_residual_scale)
                    or not 0 < trial.dynamic_skip_shape_residual_scale <= 1.0
                ):
                    raise ContractError(
                        "dynamic skip shape residual scale must be in (0, 1]"
                    )
            elif trial.dynamic_skip_shape_residual_scale != 0.25:
                raise ContractError(
                    "dynamic skip shape residual scale requires the residual branch"
                )
            if trial.dynamic_skip_learning_rate is not None and (
                not np.isfinite(trial.dynamic_skip_learning_rate)
                or trial.dynamic_skip_learning_rate <= 0
                or trial.dynamic_skip_learning_rate > 10 * trial.learning_rate
            ):
                raise ContractError(
                    "dynamic skip learning rate must be positive and no greater "
                    "than ten times the base learning rate"
                )
            if trial.dynamic_skip_warmup_epochs < 0:
                raise ContractError(
                    "dynamic skip warmup epochs must be non-negative"
                )
            if trial.dynamic_skip_warmup_epochs > 0:
                if trial.dynamic_skip_learning_rate is None:
                    raise ContractError(
                        "dynamic skip warmup requires a target learning rate"
                    )
                if trial.dynamic_skip_learning_rate <= trial.learning_rate:
                    raise ContractError(
                        "dynamic skip warmup target must exceed the base learning rate"
                    )
                if trial.dynamic_skip_warmup_epochs >= max_epochs:
                    raise ContractError(
                        "dynamic skip warmup epochs must be smaller than max_epochs"
                    )
            if trial.dynamic_skip_frozen_parent:
                if not trial.dynamic_skip_shape_residual:
                    raise ContractError(
                        "dynamic skip frozen parent requires shape residual"
                    )
                if trial.dynamic_skip_learning_rate is not None:
                    raise ContractError(
                        "dynamic skip frozen parent uses the common shape-only rate"
                    )
                if trial.weight_decay != 0:
                    raise ContractError(
                        "dynamic skip frozen parent forbids weight decay"
                    )
        elif trial.dynamic_skip_learning_rate is not None:
            raise ContractError(
                "dynamic skip learning rate is only valid for dynamic horizon skip"
            )
        elif trial.dynamic_skip_warmup_epochs != 0:
            raise ContractError(
                "dynamic skip warmup is only valid for dynamic horizon skip"
            )
        elif trial.dynamic_skip_token_normalization != "none":
            raise ContractError(
                "dynamic skip token normalization is only valid for dynamic horizon skip"
            )
        elif trial.dynamic_skip_shape_residual:
            raise ContractError(
                "dynamic skip shape residual is only valid for dynamic horizon skip"
            )
        elif trial.dynamic_skip_shape_residual_scale != 0.25:
            raise ContractError(
                "dynamic skip shape residual scale is only valid for dynamic horizon skip"
            )
        elif trial.dynamic_skip_frozen_parent:
            raise ContractError(
                "dynamic skip frozen parent is only valid for dynamic horizon skip"
            )
        if (
            trial.adapter_learning_rate is not None
            and trial.residual_learning_rate is not None
        ):
            raise ContractError(
                "adapter and residual learning rates cannot both be configured"
            )
        if trial.strategy == "grouped_smooth_l1" and not (
            trial.model_kind == "dynamic_horizon_skip"
            and trial.dynamic_skip_frozen_parent
            and trial.dynamic_skip_shape_residual
        ):
            raise ContractError(
                "grouped SmoothL1 requires frozen parent shape residual"
            )
        if trial.date_batch_order not in {"fixed_once", "epoch_seeded"}:
            raise ContractError("date batch order policy is unsupported")
        if trial.date_batch_order == "epoch_seeded" and trial.strategy not in {
            "grouped_smooth_l1",
            "rank_objective",
            "soft_rankic",
            "top_tail",
            "teacher_listwise",
        }:
            raise ContractError(
                "epoch-seeded date order requires a date-grouped strategy"
            )
        if trial.grouped_smooth_l1_reduction not in {
            "label_mean",
            "date_horizon_mean",
        }:
            raise ContractError("grouped SmoothL1 reduction is unsupported")
        if (
            trial.grouped_smooth_l1_reduction != "label_mean"
            and trial.strategy != "grouped_smooth_l1"
        ):
            raise ContractError(
                "date/horizon mean reduction requires grouped SmoothL1"
            )
        if trial.ema_decay is not None and (
            not np.isfinite(trial.ema_decay)
            or not 0.0 < trial.ema_decay < 1.0
        ):
            raise ContractError("EMA decay must be finite and in (0, 1)")
        if trial.epoch_average_start is not None and (
            isinstance(trial.epoch_average_start, bool)
            or trial.epoch_average_start < 1
            or trial.epoch_average_start > max_epochs
        ):
            raise ContractError(
                "epoch average start must be an integer in [1, max_epochs]"
            )
        if trial.ema_decay is not None and trial.epoch_average_start is not None:
            raise ContractError("EMA and epoch averaging are mutually exclusive")
        if trial.teacher_blend_start_weight is not None:
            blend_values = np.asarray(
                [
                    trial.teacher_blend_start_weight,
                    trial.teacher_blend_end_weight,
                ],
                dtype="float64",
            )
            if (
                not np.isfinite(blend_values).all()
                or not 0.0
                <= trial.teacher_blend_end_weight
                < trial.teacher_blend_start_weight
                < 1.0
            ):
                raise ContractError(
                    "teacher blend weights must satisfy 0 <= end < start < 1"
                )
            if max_epochs < 2 or trial.strategy != "smooth_l1":
                raise ContractError(
                    "linear teacher blending requires multi-epoch SmoothL1"
                )
            if trial.ema_decay is not None or trial.epoch_average_start is not None:
                raise ContractError(
                    "linear teacher blending cannot use EMA or epoch averaging"
                )
        elif trial.teacher_blend_end_weight != 0.0:
            raise ContractError(
                "teacher blend end weight requires a configured start weight"
            )
        if trial.strategy == "soft_rankic" and not (
            trial.model_kind == "temporal_context"
            or (
                trial.model_kind == "dynamic_horizon_skip"
                and trial.dynamic_skip_frozen_parent
                and trial.dynamic_skip_shape_residual
            )
        ):
            raise ContractError(
                "soft RankIC requires temporal context or frozen parent shape residual"
            )
        if trial.strategy == "soft_rankic" and (
            not np.isfinite(trial.soft_rankic_weight)
            or trial.soft_rankic_weight <= 0
            or not np.isfinite(trial.soft_rank_temperature)
            or trial.soft_rank_temperature <= 0
        ):
            raise ContractError("soft RankIC parameters must be finite and positive")
        if trial.strategy == "top_tail" and not (
            trial.model_kind == "dynamic_horizon_skip"
            and trial.dynamic_skip_frozen_parent
            and trial.dynamic_skip_shape_residual
        ):
            raise ContractError(
                "top-tail objective requires frozen parent shape residual"
            )
        if trial.strategy == "top_tail" and (
            not np.isfinite(trial.top_tail_weight)
            or trial.top_tail_weight <= 0
            or not np.isfinite(trial.top_tail_fraction)
            or not 0 < trial.top_tail_fraction <= 0.5
            or not np.isfinite(trial.top_tail_temperature)
            or trial.top_tail_temperature <= 0
        ):
            raise ContractError(
                "top-tail parameters must be finite with fraction in (0, 0.5]"
            )
        if trial.strategy != "top_tail" and (
            trial.top_tail_weight != 0.05
            or trial.top_tail_fraction != 0.1
            or trial.top_tail_temperature != 0.1
        ):
            raise ContractError(
                "top-tail parameters are only valid for top-tail trials"
            )
        if trial.strategy == "teacher_listwise" and (
            trial.model_kind != "dynamic_horizon_skip"
            or not np.isfinite(trial.teacher_listwise_gradient_ratio)
            or not 0.0 < trial.teacher_listwise_gradient_ratio <= 1.0
            or not np.isfinite(trial.teacher_listwise_temperature)
            or trial.teacher_listwise_temperature <= 0.0
        ):
            raise ContractError(
                "teacher listwise requires dynamic horizon skip and positive parameters"
            )
        if trial.strategy != "teacher_listwise" and (
            trial.teacher_listwise_gradient_ratio != 0.25
            or trial.teacher_listwise_temperature != 0.1
        ):
            raise ContractError(
                "teacher listwise parameters are only valid for teacher listwise trials"
            )
        if trial.padding_mode == "chomp" and (
            trial.head_dropout != 0 or trial.dropout_kind != "element"
        ):
            raise ContractError("padding chomp requires the frozen lite dropout path")
        observed = (
            receptive_field(kernel_size=trial.kernel_size, dilations=trial.dilations)
            if trial.model_kind == "bai"
            else lite_receptive_field(
                kernel_size=trial.kernel_size, dilations=trial.dilations
            )
        )
        if observed < input_steps:
            raise ContractError(
                f"trial {trial.trial_id} receptive field {observed} is smaller "
                f"than input window {input_steps}"
            )
    return tuple(trials)
