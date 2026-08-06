"""Date-grouped TCN-v9 ranking sampler and fixed mixed objective."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import hashlib
from math import ceil
from typing import Literal

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import Sampler

from .experiment import ContractError
from .training_data import masked_smooth_l1


class DateGroupedBatchSampler(Sampler[list[int]]):
    """Yield every member of one signal date as one deterministic batch."""

    def __init__(
        self,
        signal_dates: Sequence[object],
        *,
        shuffle_dates: bool,
        seed: int,
        batch_size: int | None = None,
        order_policy: Literal["fixed_once", "epoch_seeded"] = "fixed_once",
    ) -> None:
        if not signal_dates:
            raise ContractError("date-grouped sampler requires signal dates")
        groups: dict[str, list[int]] = {}
        for index, signal_date in enumerate(signal_dates):
            date = str(signal_date)
            if not date:
                raise ContractError("date-grouped sampler contains an empty date")
            groups.setdefault(date, []).append(index)
        self._groups = tuple(groups[date] for date in sorted(groups))
        self._shuffle_dates = bool(shuffle_dates)
        self._seed = int(seed)
        if order_policy not in {"fixed_once", "epoch_seeded"}:
            raise ContractError("date-grouped sampler order policy is unsupported")
        self._order_policy = order_policy
        self._epoch = 0
        if batch_size is not None and batch_size <= 0:
            raise ContractError("date-grouped sampler batch size must be positive")
        self._batch_size = batch_size

    def _ordered_batches(self) -> tuple[list[int], ...]:
        order = np.arange(len(self._groups), dtype="int64")
        if self._shuffle_dates:
            if self._order_policy == "fixed_once" or self._epoch == 0:
                rng = np.random.default_rng(self._seed)
            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence([self._seed, self._epoch])
                )
            order = rng.permutation(order)
        ordered_groups = [self._groups[int(offset)] for offset in order]
        if self._batch_size is None:
            return tuple(list(group) for group in ordered_groups)
        batches: list[list[int]] = []
        current: list[int] = []
        for group in ordered_groups:
            for start in range(0, len(group), self._batch_size):
                chunk = list(group[start : start + self._batch_size])
                if current and len(current) + len(chunk) > self._batch_size:
                    batches.append(current)
                    current = []
                current.extend(chunk)
                if len(current) == self._batch_size:
                    batches.append(current)
                    current = []
        if current:
            batches.append(current)
        return tuple(batches)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._ordered_batches()

    def __len__(self) -> int:
        return len(self._ordered_batches())

    def set_epoch(self, epoch: int) -> None:
        """Select a replayable epoch-specific date order when configured."""

        if epoch < 0:
            raise ContractError("date-grouped sampler epoch cannot be negative")
        if self._order_policy == "epoch_seeded":
            self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def order_policy(self) -> str:
        return self._order_policy

    def order_fingerprint(self) -> str:
        """Hash the exact physical-batch order without exposing sample labels."""

        digest = hashlib.sha256()
        for batch in self._ordered_batches():
            values = np.asarray(batch, dtype="int64")
            digest.update(np.asarray([len(values)], dtype="int64").tobytes())
            digest.update(values.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class MixedRankLoss:
    status: str
    total: torch.Tensor
    smooth_l1: torch.Tensor
    pairwise: torch.Tensor
    group_count: int
    pair_count: int


@dataclass(frozen=True)
class MixedSoftRankICLoss:
    status: str
    total: torch.Tensor
    smooth_l1: torch.Tensor
    soft_rankic: torch.Tensor
    group_count: int


@dataclass(frozen=True)
class TeacherListwiseLoss:
    """True-target SmoothL1 plus an unscaled teacher ordering component."""

    smooth_l1: torch.Tensor
    teacher_listwise: torch.Tensor
    group_count: int
    valid_label_count: int


@dataclass(frozen=True)
class MixedTopTailLoss:
    """SmoothL1 plus equal-date/horizon realized-top separation loss."""

    status: str
    total: torch.Tensor
    smooth_l1: torch.Tensor
    top_tail: torch.Tensor
    group_count: int
    pair_count: int


@dataclass(frozen=True)
class DateHorizonSmoothL1Loss:
    """Equal-weight SmoothL1 across non-empty date/horizon groups."""

    total: torch.Tensor
    group_count: int
    valid_label_count: int


def date_horizon_group_counts(
    mask: torch.Tensor,
    signal_dates: Sequence[object],
) -> tuple[int, int]:
    """Count non-empty date/horizon groups and their valid labels."""

    if mask.ndim != 2 or mask.shape[1] != 4 or mask.dtype != torch.bool:
        raise ContractError("date/horizon SmoothL1 mask must be boolean [N, 4]")
    if len(signal_dates) != len(mask):
        raise ContractError("date/horizon SmoothL1 dates do not match the batch")
    dates = np.asarray([str(value) for value in signal_dates])
    if len(dates) == 0 or any(not value for value in dates):
        raise ContractError("date/horizon SmoothL1 requires non-empty dates")
    group_count = 0
    valid_label_count = 0
    for signal_date in sorted(set(dates)):
        date_rows = torch.as_tensor(
            np.flatnonzero(dates == signal_date),
            dtype=torch.long,
            device=mask.device,
        )
        for horizon_column in range(4):
            valid_count = int(mask[date_rows, horizon_column].sum())
            if valid_count > 0:
                group_count += 1
                valid_label_count += valid_count
    return group_count, valid_label_count


def date_horizon_equal_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    signal_dates: Sequence[object],
) -> DateHorizonSmoothL1Loss:
    """Average SmoothL1 first within dates/horizons and then across groups."""

    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ContractError("date/horizon SmoothL1 requires four prediction horizons")
    if target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ContractError(
            "date/horizon SmoothL1 prediction, target, and mask shapes differ"
        )
    if mask.dtype != torch.bool:
        raise ContractError("date/horizon SmoothL1 mask must be boolean")
    if bool((mask & ~torch.isfinite(target)).any()):
        raise ContractError("a valid date/horizon SmoothL1 target is non-finite")
    group_count, valid_label_count = date_horizon_group_counts(mask, signal_dates)
    if group_count == 0:
        raise ContractError("date/horizon SmoothL1 has no valid date/horizon groups")

    dates = np.asarray([str(value) for value in signal_dates])
    group_losses: list[torch.Tensor] = []
    for signal_date in sorted(set(dates)):
        date_rows = torch.as_tensor(
            np.flatnonzero(dates == signal_date),
            dtype=torch.long,
            device=prediction.device,
        )
        for horizon_column in range(4):
            valid_rows = date_rows[mask[date_rows, horizon_column]]
            if len(valid_rows) == 0:
                continue
            group_losses.append(
                functional.smooth_l1_loss(
                    prediction[valid_rows, horizon_column],
                    target[valid_rows, horizon_column],
                    reduction="mean",
                )
            )
    return DateHorizonSmoothL1Loss(
        total=torch.stack(group_losses).mean(),
        group_count=group_count,
        valid_label_count=valid_label_count,
    )


def mixed_date_grouped_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    signal_dates: Sequence[object],
    *,
    rank_objective_allowed: bool,
    pairwise_weight: float = 0.1,
) -> MixedRankLoss:
    """Combine masked SmoothL1 with equal-weight date/horizon pairwise loss."""

    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ContractError("mixed rank objective requires four prediction horizons")
    if target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ContractError("mixed rank objective prediction, target, and mask shapes differ")
    if len(signal_dates) != len(prediction):
        raise ContractError("mixed rank objective dates do not match the batch")
    if mask.dtype != torch.bool:
        raise ContractError("mixed rank objective mask must be boolean")
    if bool((mask & ~torch.isfinite(target)).any()):
        raise ContractError("a valid mixed-objective target is non-finite")
    if abs(pairwise_weight - 0.1) > 1e-12:
        raise ContractError("TCN-v9 pairwise weight is fixed at 0.1")
    safe_target = torch.where(mask, target, prediction.detach())
    smooth = masked_smooth_l1(prediction, safe_target, mask)
    zero = prediction.sum() * 0.0
    if not rank_objective_allowed:
        return MixedRankLoss(
            status="rank_objective_not_applicable",
            total=smooth,
            smooth_l1=smooth,
            pairwise=zero,
            group_count=0,
            pair_count=0,
        )

    dates = np.asarray([str(value) for value in signal_dates])
    group_losses = []
    pair_count = 0
    for signal_date in sorted(set(dates)):
        date_rows = torch.as_tensor(
            np.flatnonzero(dates == signal_date),
            dtype=torch.long,
            device=prediction.device,
        )
        for horizon_column in range(4):
            valid_rows = date_rows[mask[date_rows, horizon_column]]
            if len(valid_rows) < 2:
                continue
            pairs = torch.triu_indices(
                len(valid_rows),
                len(valid_rows),
                offset=1,
                device=prediction.device,
            )
            left_rows = valid_rows[pairs[0]]
            right_rows = valid_rows[pairs[1]]
            target_delta = (
                target[left_rows, horizon_column] - target[right_rows, horizon_column]
            )
            directional = target_delta.ne(0)
            if not bool(directional.any()):
                continue
            left_rows = left_rows[directional]
            right_rows = right_rows[directional]
            directions = torch.sign(target_delta[directional])
            score_delta = (
                prediction[left_rows, horizon_column]
                - prediction[right_rows, horizon_column]
            )
            losses = functional.softplus(-directions * score_delta)
            group_losses.append(losses.mean())
            pair_count += int(losses.numel())
    pairwise = torch.stack(group_losses).mean() if group_losses else zero
    total = smooth + pairwise_weight * pairwise
    return MixedRankLoss(
        status="rank_objective_active",
        total=total,
        smooth_l1=smooth,
        pairwise=pairwise,
        group_count=len(group_losses),
        pair_count=pair_count,
    )


def mixed_date_grouped_soft_rankic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    signal_dates: Sequence[object],
    *,
    soft_rankic_weight: float,
    temperature: float,
) -> MixedSoftRankICLoss:
    """Mix SmoothL1 with an equal-date differentiable Spearman surrogate."""

    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ContractError("soft RankIC objective requires four prediction horizons")
    if target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ContractError(
            "soft RankIC objective prediction, target, and mask shapes differ"
        )
    if len(signal_dates) != len(prediction):
        raise ContractError("soft RankIC objective dates do not match the batch")
    if mask.dtype != torch.bool:
        raise ContractError("soft RankIC objective mask must be boolean")
    if bool((mask & ~torch.isfinite(target)).any()):
        raise ContractError("a valid soft RankIC target is non-finite")
    if not np.isfinite(soft_rankic_weight) or soft_rankic_weight <= 0:
        raise ContractError("soft RankIC weight must be finite and positive")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ContractError("soft RankIC temperature must be finite and positive")

    safe_target = torch.where(mask, target, prediction.detach())
    smooth = masked_smooth_l1(prediction, safe_target, mask)
    zero = prediction.sum() * 0.0
    dates = np.asarray([str(value) for value in signal_dates])
    group_losses: list[torch.Tensor] = []
    for signal_date in sorted(set(dates)):
        date_rows = torch.as_tensor(
            np.flatnonzero(dates == signal_date),
            dtype=torch.long,
            device=prediction.device,
        )
        for horizon_column in range(4):
            valid_rows = date_rows[mask[date_rows, horizon_column]]
            if len(valid_rows) < 2:
                continue
            scores = prediction[valid_rows, horizon_column]
            targets = target[valid_rows, horizon_column]
            centered_targets = targets - targets.mean()
            target_norm = torch.linalg.vector_norm(centered_targets)
            if not bool(torch.isfinite(target_norm)) or float(target_norm) <= 0:
                continue
            score_differences = (
                scores[:, None] - scores[None, :]
            ) / temperature
            soft_ranks = torch.sigmoid(score_differences).sum(dim=1)
            centered_ranks = soft_ranks - soft_ranks.mean()
            rank_norm = torch.linalg.vector_norm(centered_ranks).clamp_min(1e-12)
            correlation = torch.dot(centered_ranks, centered_targets) / (
                rank_norm * target_norm
            )
            group_losses.append(1.0 - correlation)
    soft_rankic = torch.stack(group_losses).mean() if group_losses else zero
    total = smooth + soft_rankic_weight * soft_rankic
    return MixedSoftRankICLoss(
        status=(
            "soft_rankic_active" if group_losses else "soft_rankic_no_valid_groups"
        ),
        total=total,
        smooth_l1=smooth,
        soft_rankic=soft_rankic,
        group_count=len(group_losses),
    )


def teacher_listwise_components(
    prediction: torch.Tensor,
    true_target: torch.Tensor,
    teacher_target: torch.Tensor,
    mask: torch.Tensor,
    signal_dates: Sequence[object],
    *,
    temperature: float,
) -> TeacherListwiseLoss:
    """Build scale-independent true and full-cross-section teacher losses."""

    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ContractError("teacher listwise objective requires four horizons")
    if (
        true_target.shape != prediction.shape
        or teacher_target.shape != prediction.shape
        or mask.shape != prediction.shape
    ):
        raise ContractError("teacher listwise prediction/target/mask shapes differ")
    if len(signal_dates) != len(prediction):
        raise ContractError("teacher listwise dates do not match the batch")
    if mask.dtype != torch.bool:
        raise ContractError("teacher listwise mask must be boolean")
    if bool((mask & ~torch.isfinite(true_target)).any()):
        raise ContractError("a valid true listwise target is non-finite")
    if bool((mask & ~torch.isfinite(teacher_target)).any()):
        raise ContractError("a valid teacher listwise target is non-finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ContractError("teacher listwise temperature must be finite and positive")

    safe_true = torch.where(mask, true_target, prediction.detach())
    smooth = masked_smooth_l1(prediction, safe_true, mask)
    dates = np.asarray([str(value) for value in signal_dates])
    group_losses: list[torch.Tensor] = []
    for signal_date in sorted(set(dates)):
        date_rows = torch.as_tensor(
            np.flatnonzero(dates == signal_date),
            dtype=torch.long,
            device=prediction.device,
        )
        for horizon_column in range(4):
            valid_rows = date_rows[mask[date_rows, horizon_column]]
            if len(valid_rows) < 2:
                continue
            scores = prediction[valid_rows, horizon_column]
            teacher = teacher_target[valid_rows, horizon_column]
            centered_teacher = teacher - teacher.mean()
            teacher_norm = torch.linalg.vector_norm(centered_teacher)
            if not bool(torch.isfinite(teacher_norm)) or float(teacher_norm) <= 0:
                continue
            score_differences = (
                scores[:, None] - scores[None, :]
            ) / temperature
            soft_ranks = torch.sigmoid(score_differences).sum(dim=1)
            centered_ranks = soft_ranks - soft_ranks.mean()
            rank_norm = torch.linalg.vector_norm(centered_ranks).clamp_min(1e-12)
            correlation = torch.dot(centered_ranks, centered_teacher) / (
                rank_norm * teacher_norm
            )
            group_losses.append(1.0 - correlation)
    if not group_losses:
        raise ContractError("teacher listwise objective has no valid groups")
    return TeacherListwiseLoss(
        smooth_l1=smooth,
        teacher_listwise=torch.stack(group_losses).mean(),
        group_count=len(group_losses),
        valid_label_count=int(mask.sum()),
    )


def mixed_date_grouped_top_tail_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    signal_dates: Sequence[object],
    *,
    top_tail_weight: float,
    top_fraction: float,
    temperature: float,
) -> MixedTopTailLoss:
    """Separate the realized top tail from every non-top member per group.

    Pair losses are averaged inside each non-empty ``(signal_date, horizon)``
    cross-section and those cross-sections are then weighted equally.  The
    target determines membership only; gradients flow through scores alone.
    """

    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ContractError("top-tail objective requires four prediction horizons")
    if target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ContractError(
            "top-tail objective prediction, target, and mask shapes differ"
        )
    if len(signal_dates) != len(prediction):
        raise ContractError("top-tail objective dates do not match the batch")
    if mask.dtype != torch.bool:
        raise ContractError("top-tail objective mask must be boolean")
    if bool((mask & ~torch.isfinite(target)).any()):
        raise ContractError("a valid top-tail target is non-finite")
    if not np.isfinite(top_tail_weight) or top_tail_weight <= 0:
        raise ContractError("top-tail weight must be finite and positive")
    if not np.isfinite(top_fraction) or not 0 < top_fraction <= 0.5:
        raise ContractError("top-tail fraction must be in (0, 0.5]")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ContractError("top-tail temperature must be finite and positive")

    dates = np.asarray([str(value) for value in signal_dates])
    if len(dates) == 0 or any(not value for value in dates):
        raise ContractError("top-tail objective requires non-empty dates")
    safe_target = torch.where(mask, target, prediction.detach())
    smooth = masked_smooth_l1(prediction, safe_target, mask)
    zero = prediction.sum() * 0.0
    group_losses: list[torch.Tensor] = []
    pair_count = 0
    for signal_date in sorted(set(dates)):
        date_rows = torch.as_tensor(
            np.flatnonzero(dates == signal_date),
            dtype=torch.long,
            device=prediction.device,
        )
        for horizon_column in range(4):
            valid_rows = date_rows[mask[date_rows, horizon_column]]
            member_count = len(valid_rows)
            if member_count < 2:
                continue
            top_count = max(1, int(ceil(member_count * top_fraction)))
            target_order = torch.argsort(
                target[valid_rows, horizon_column],
                descending=True,
                stable=True,
            )
            top_rows = valid_rows[target_order[:top_count]]
            non_top_rows = valid_rows[target_order[top_count:]]
            if len(non_top_rows) == 0:
                continue
            score_delta = (
                prediction[top_rows, horizon_column][:, None]
                - prediction[non_top_rows, horizon_column][None, :]
            )
            pair_losses = functional.softplus(-score_delta / temperature)
            group_losses.append(pair_losses.mean())
            pair_count += int(pair_losses.numel())
    top_tail = torch.stack(group_losses).mean() if group_losses else zero
    return MixedTopTailLoss(
        status="top_tail_active" if group_losses else "top_tail_no_valid_groups",
        total=smooth + top_tail_weight * top_tail,
        smooth_l1=smooth,
        top_tail=top_tail,
        group_count=len(group_losses),
        pair_count=pair_count,
    )
