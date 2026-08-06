"""Train-only multi-horizon gradient diagnostics and deterministic PCGrad."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import time
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from .experiment import ContractError
from .neural import HORIZONS


@dataclass(frozen=True)
class PCGradTrigger:
    status: str
    horizon_pair: tuple[int, int] | None
    conflicting_fold_count: int


@dataclass(frozen=True)
class PCGradProjection:
    projected: dict[int, torch.Tensor]
    merged: torch.Tensor
    task_order: tuple[int, ...]


@dataclass(frozen=True)
class PCGradBackwardReceipt:
    task_order: tuple[int, ...]
    selected_horizons: tuple[int, ...]
    projection_seconds: float
    horizon_backward_seconds: float
    merged_gradient_norm: float


def _flatten_gradients(
    losses: Mapping[int, torch.Tensor],
    block_parameters: Mapping[str, Sequence[nn.Parameter]],
    *,
    selected_horizons: Sequence[int] = HORIZONS,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[nn.Parameter]]:
    if set(losses) != set(HORIZONS):
        raise ContractError("gradient diagnostic requires all four horizons")
    resolved_horizons = tuple(int(value) for value in selected_horizons)
    if (
        len(resolved_horizons) < 2
        or len(set(resolved_horizons)) != len(resolved_horizons)
        or not set(resolved_horizons).issubset(set(HORIZONS))
    ):
        raise ContractError("PCGrad selected horizons must be a unique HORIZONS subset")
    if not block_parameters or any(not name for name in block_parameters):
        raise ContractError("gradient diagnostic requires named shared-trunk blocks")
    parameters: list[nn.Parameter] = []
    block_slices: dict[str, slice] = {}
    observed_ids: set[int] = set()
    for block_name, values in block_parameters.items():
        start = len(parameters)
        block_values = list(values)
        if not block_values:
            raise ContractError("gradient diagnostic blocks cannot be empty")
        for parameter in block_values:
            if id(parameter) in observed_ids:
                raise ContractError("shared parameter appears in more than one block")
            observed_ids.add(id(parameter))
            parameters.append(parameter)
        block_slices[block_name] = slice(start, len(parameters))

    by_horizon: dict[int, dict[str, torch.Tensor]] = {}
    for horizon in resolved_horizons:
        loss = losses[int(horizon)]
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ContractError("each horizon gradient loss must be a finite scalar")
        raw = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        vectors = [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for parameter, gradient in zip(parameters, raw, strict=True)
        ]
        by_horizon[int(horizon)] = {
            block_name: torch.cat(vectors[block_slice])
            for block_name, block_slice in block_slices.items()
        }
    return by_horizon, parameters


def _cosine(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float, float]:
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    denominator = left_norm * right_norm
    cosine = float(torch.dot(left, right) / denominator) if denominator > 0 else 0.0
    norm_ratio = left_norm / right_norm if right_norm > 0 else float("inf")
    return cosine, left_norm, norm_ratio


def diagnose_task_gradients(
    losses: Mapping[int, torch.Tensor],
    block_parameters: Mapping[str, Sequence[nn.Parameter]],
    *,
    fold: int,
    seed: int,
    batch_id: int,
) -> pd.DataFrame:
    """Measure horizon gradients at one shared snapshot without an optimizer step."""

    by_horizon, _ = _flatten_gradients(losses, block_parameters)
    scopes = tuple(block_parameters) + ("global",)
    rows = []
    for left_horizon, right_horizon in combinations(HORIZONS, 2):
        for scope in scopes:
            if scope == "global":
                left = torch.cat(list(by_horizon[int(left_horizon)].values()))
                right = torch.cat(list(by_horizon[int(right_horizon)].values()))
            else:
                left = by_horizon[int(left_horizon)][scope]
                right = by_horizon[int(right_horizon)][scope]
            cosine, left_norm, norm_ratio = _cosine(left, right)
            right_norm = float(torch.linalg.vector_norm(right))
            rows.append(
                {
                    "fold": int(fold),
                    "seed": int(seed),
                    "batch_id": int(batch_id),
                    "scope": scope,
                    "left_horizon": int(left_horizon),
                    "right_horizon": int(right_horizon),
                    "cosine": cosine,
                    "left_gradient_norm": left_norm,
                    "right_gradient_norm": right_norm,
                    "norm_ratio": norm_ratio,
                    "negative_cosine": cosine < 0,
                }
            )
    return pd.DataFrame(rows)


def evaluate_pcgrad_trigger(diagnostics: pd.DataFrame) -> PCGradTrigger:
    """Require median-negative, >=30% negative batches in at least three folds."""

    required = {
        "fold",
        "batch_id",
        "scope",
        "left_horizon",
        "right_horizon",
        "cosine",
        "negative_cosine",
    }
    if missing := sorted(required.difference(diagnostics.columns)):
        raise ContractError(f"gradient diagnostics missing columns: {', '.join(missing)}")
    global_rows = diagnostics.loc[diagnostics["scope"].eq("global")].copy()
    if global_rows.empty or not np.isfinite(global_rows["cosine"].to_numpy(dtype="float64")).all():
        raise ContractError("global gradient diagnostics are empty or non-finite")
    if global_rows.duplicated(
        ["fold", "batch_id", "left_horizon", "right_horizon"]
    ).any():
        raise ContractError("gradient diagnostics contain duplicate batch pairs")
    candidates: list[tuple[tuple[int, int], int]] = []
    for pair_values, pair_rows in global_rows.groupby(
        ["left_horizon", "right_horizon"], observed=True
    ):
        left_value, right_value = pair_values
        pair = (int(cast(int, left_value)), int(cast(int, right_value)))
        conflicting_folds = 0
        for _, fold_rows in pair_rows.groupby("fold", observed=True):
            median = float(fold_rows["cosine"].median())
            negative_rate = float(fold_rows["negative_cosine"].astype(bool).mean())
            if median < 0 and negative_rate >= 0.30:
                conflicting_folds += 1
        if conflicting_folds >= 3:
            candidates.append((pair, conflicting_folds))
    if not candidates:
        return PCGradTrigger("pcgrad_not_applicable", None, 0)
    pair, fold_count = sorted(candidates, key=lambda value: (-value[1], value[0]))[0]
    return PCGradTrigger("pcgrad_applicable", pair, fold_count)


def project_conflicting_gradients(
    gradients: Mapping[int, torch.Tensor],
    *,
    seed: int,
) -> PCGradProjection:
    """Project negative task components using a seed-determined traversal."""

    if set(gradients) != set(HORIZONS):
        raise ContractError("PCGrad requires gradients for all four horizons")
    return _project_gradient_set(gradients, seed=seed)


def _project_gradient_set(
    gradients: Mapping[int, torch.Tensor],
    *,
    seed: int,
) -> PCGradProjection:
    """Project a prevalidated set of at least two task gradients."""

    if len(gradients) < 2:
        raise ContractError("PCGrad requires at least two task gradients")
    horizons = np.asarray(sorted(int(value) for value in gradients), dtype="int64")
    task_order = tuple(int(value) for value in np.random.default_rng(seed).permutation(horizons))
    resolved = {horizon: gradients[horizon].detach().clone().reshape(-1) for horizon in task_order}
    shapes = {tuple(value.shape) for value in resolved.values()}
    devices = {str(value.device) for value in resolved.values()}
    dtypes = {value.dtype for value in resolved.values()}
    if len(shapes) != 1 or len(devices) != 1 or len(dtypes) != 1:
        raise ContractError("PCGrad task gradients must share shape, device, and dtype")
    projected: dict[int, torch.Tensor] = {}
    rng = np.random.default_rng(seed)
    for horizon in task_order:
        candidate = resolved[horizon].clone()
        others = [value for value in task_order if value != horizon]
        for other in rng.permutation(np.asarray(others, dtype="int64")):
            reference = resolved[int(other)]
            inner = torch.dot(candidate, reference)
            denominator = torch.dot(reference, reference)
            if float(inner) < 0 and float(denominator) > 0:
                candidate = candidate - inner / denominator * reference
        projected[horizon] = candidate
    merged = torch.stack([projected[horizon] for horizon in task_order]).mean(dim=0)
    return PCGradProjection(projected, merged, task_order)


def pcgrad_backward(
    losses: Mapping[int, torch.Tensor],
    block_parameters: Mapping[str, Sequence[nn.Parameter]],
    *,
    seed: int,
    total_loss: torch.Tensor | None = None,
    selected_horizons: Sequence[int] = HORIZONS,
) -> PCGradBackwardReceipt:
    """Populate full or localized PCGrad while preserving ordinary gradients."""

    resolved_horizons = tuple(int(value) for value in selected_horizons)
    backward_start = time.perf_counter()
    by_horizon, parameters = _flatten_gradients(
        losses,
        block_parameters,
        selected_horizons=resolved_horizons,
    )
    horizon_backward_seconds = time.perf_counter() - backward_start
    flat = {
        horizon: torch.cat(list(block_gradients.values()))
        for horizon, block_gradients in by_horizon.items()
    }
    projection_start = time.perf_counter()
    result = _project_gradient_set(flat, seed=seed)
    projection_seconds = time.perf_counter() - projection_start
    if total_loss is not None:
        if total_loss.ndim != 0 or not bool(torch.isfinite(total_loss)):
            raise ContractError("PCGrad total loss must be a finite scalar")
        total_loss.backward()
        raw_sum = torch.stack([flat[horizon] for horizon in resolved_horizons]).sum(0)
        projected_sum = torch.stack(
            [result.projected[horizon] for horizon in resolved_horizons]
        ).sum(0)
        correction = (projected_sum - raw_sum) / len(HORIZONS)
    else:
        if set(resolved_horizons) != set(HORIZONS):
            raise ContractError(
                "localized PCGrad requires total_loss to preserve other gradients"
            )
        correction = None
    offset = 0
    applied_gradients: list[torch.Tensor] = []
    for parameter in parameters:
        size = parameter.numel()
        if correction is None:
            parameter.grad = result.merged[
                offset : offset + size
            ].reshape_as(parameter).clone()
        else:
            baseline = (
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad.detach()
            )
            parameter.grad = baseline + correction[offset : offset + size].reshape_as(
                parameter
            )
        applied_gradients.append(parameter.grad.detach().reshape(-1))
        offset += size
    return PCGradBackwardReceipt(
        task_order=result.task_order,
        selected_horizons=resolved_horizons,
        projection_seconds=projection_seconds,
        horizon_backward_seconds=horizon_backward_seconds,
        merged_gradient_norm=float(
            torch.linalg.vector_norm(torch.cat(applied_gradients))
        ),
    )
