from __future__ import annotations

import math

import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.v9_objective import (
    DateGroupedBatchSampler,
    mixed_date_grouped_soft_rankic_loss,
    mixed_date_grouped_rank_loss,
)


def test_mixed_rank_loss_never_crosses_date_or_horizon_and_weights_groups_equally() -> None:
    prediction = torch.tensor(
        [
            [-1.0, -0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [1.0, 9.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [-1.0, -1.0, 0.0, float("nan")],
            [0.0, 1.0, 0.0, float("nan")],
            [1.0, 1.0, 0.0, float("nan")],
            [-1.0, 0.0, 0.0, float("nan")],
            [1.0, 0.0, 0.0, float("nan")],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, False, True, False],
            [True, False, True, False],
            [True, False, True, False],
        ]
    )
    dates = ["2025-01-02"] * 3 + ["2025-01-03"] * 2

    result = mixed_date_grouped_rank_loss(
        prediction,
        target,
        mask,
        dates,
        rank_objective_allowed=True,
    )

    date_a_h1 = sum(math.log1p(math.exp(-value)) for value in [1.0, 2.0, 1.0]) / 3
    date_a_h2 = math.log1p(math.exp(-1.0))
    date_b_h1 = math.log1p(math.exp(2.0))
    expected_pairwise = (date_a_h1 + date_a_h2 + date_b_h1) / 3
    assert result.status == "rank_objective_active"
    assert result.group_count == 3
    assert result.pair_count == 5
    assert float(result.pairwise.detach()) == pytest.approx(expected_pairwise)
    assert float(result.total.detach()) == pytest.approx(
        float(result.smooth_l1.detach()) + 0.1 * expected_pairwise
    )
    result.total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_rank_objective_gate_preserves_default_smooth_l1_and_fixed_weight() -> None:
    prediction = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    target = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    skipped = mixed_date_grouped_rank_loss(
        prediction,
        target,
        mask,
        ["2025-01-02", "2025-01-02"],
        rank_objective_allowed=False,
    )
    assert skipped.status == "rank_objective_not_applicable"
    assert float(skipped.total) == pytest.approx(float(skipped.smooth_l1))
    assert float(skipped.pairwise) == 0.0

    with pytest.raises(ContractError, match="fixed at 0.1"):
        mixed_date_grouped_rank_loss(
            prediction,
            target,
            mask,
            ["2025-01-02", "2025-01-02"],
            rank_objective_allowed=True,
            pairwise_weight=0.2,
        )


def test_date_grouped_sampler_preserves_every_member_with_seeded_date_order() -> None:
    dates = ["2025-01-03", "2025-01-02", "2025-01-03", "2025-01-04"]
    ordered = list(DateGroupedBatchSampler(dates, shuffle_dates=False, seed=7))
    assert ordered == [[1], [0, 2], [3]]
    shuffled = list(DateGroupedBatchSampler(dates, shuffle_dates=True, seed=17))
    replayed = list(DateGroupedBatchSampler(dates, shuffle_dates=True, seed=17))
    assert shuffled == replayed
    assert sorted(index for batch in shuffled for index in batch) == [0, 1, 2, 3]

    packed = list(
        DateGroupedBatchSampler(
            dates,
            shuffle_dates=False,
            seed=7,
            batch_size=3,
        )
    )
    assert packed == [[1, 0, 2], [3]]


def test_soft_rankic_prefers_correct_cross_section_and_backpropagates() -> None:
    target = torch.tensor(
        [[-1.0] * 4, [-0.3] * 4, [0.4] * 4, [1.0] * 4]
    )
    mask = torch.ones_like(target, dtype=torch.bool)
    correct = target.clone().requires_grad_(True)
    reversed_scores = target.flip(0).clone().requires_grad_(True)
    dates = ["2025-01-02"] * 4

    aligned = mixed_date_grouped_soft_rankic_loss(
        correct,
        target,
        mask,
        dates,
        soft_rankic_weight=0.2,
        temperature=0.1,
    )
    reversed_result = mixed_date_grouped_soft_rankic_loss(
        reversed_scores,
        target,
        mask,
        dates,
        soft_rankic_weight=0.2,
        temperature=0.1,
    )
    reversed_result.total.backward()

    assert aligned.soft_rankic < reversed_result.soft_rankic
    assert aligned.group_count == 4
    assert reversed_scores.grad is not None
    assert torch.isfinite(reversed_scores.grad).all()
    assert float(reversed_scores.grad.abs().sum()) > 0


def test_soft_rankic_has_stable_fallback_and_validates_public_parameters() -> None:
    prediction = torch.tensor([[0.2, 0.1, 0.0, -0.1]], requires_grad=True)
    target = torch.tensor([[0.1, 0.0, -0.1, -0.2]])
    mask = torch.ones_like(target, dtype=torch.bool)
    result = mixed_date_grouped_soft_rankic_loss(
        prediction,
        target,
        mask,
        ["2025-01-02"],
        soft_rankic_weight=0.2,
        temperature=0.1,
    )

    assert result.group_count == 0
    assert float(result.soft_rankic.detach()) == 0
    torch.testing.assert_close(result.total, result.smooth_l1)
    with pytest.raises(ContractError, match="temperature"):
        mixed_date_grouped_soft_rankic_loss(
            prediction,
            target,
            mask,
            ["2025-01-02"],
            soft_rankic_weight=0.2,
            temperature=0.0,
        )
    with pytest.raises(ContractError, match="weight"):
        mixed_date_grouped_soft_rankic_loss(
            prediction,
            target,
            mask,
            ["2025-01-02"],
            soft_rankic_weight=-0.1,
            temperature=0.1,
        )
