from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import nn

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.v9_gradients import (
    diagnose_task_gradients,
    evaluate_pcgrad_trigger,
    pcgrad_backward,
    project_conflicting_gradients,
)


def test_gradient_diagnostic_observes_global_and_block_conflict_without_step() -> None:
    torch.manual_seed(2)
    trunk = nn.Linear(2, 1, bias=False)
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    shared = trunk(inputs)
    losses = {
        1: shared.sum(),
        2: -shared.sum(),
        3: (shared.square()).sum(),
        5: (shared.square()).sum(),
    }
    before = trunk.weight.detach().clone()

    diagnostics = diagnose_task_gradients(
        losses,
        {"block-0": tuple(trunk.parameters())},
        fold=1,
        seed=7,
        batch_id=3,
    )

    opposed = diagnostics.loc[
        diagnostics["scope"].eq("global")
        & diagnostics["left_horizon"].eq(1)
        & diagnostics["right_horizon"].eq(2)
    ].iloc[0]
    assert opposed["cosine"] == pytest.approx(-1.0)
    assert bool(opposed["negative_cosine"])
    assert set(diagnostics["scope"]) == {"global", "block-0"}
    assert set(diagnostics["fold"]) == {1}
    assert trunk.weight.grad is None
    torch.testing.assert_close(trunk.weight, before)

    with pytest.raises(ContractError, match="four horizons"):
        diagnose_task_gradients(
            {1: losses[1], 2: losses[2]},
            {"block-0": tuple(trunk.parameters())},
            fold=1,
            seed=7,
            batch_id=4,
        )


def test_pcgrad_trigger_requires_persistent_conflict_in_three_folds() -> None:
    rows = []
    for fold in range(5):
        for batch in range(10):
            rows.append(
                {
                    "fold": fold,
                    "batch_id": batch,
                    "scope": "global",
                    "left_horizon": 1,
                    "right_horizon": 2,
                    "cosine": -0.5 if fold < 3 and batch < 6 else 0.25,
                    "negative_cosine": fold < 3 and batch < 6,
                }
            )
    triggered = evaluate_pcgrad_trigger(pd.DataFrame(rows))
    assert triggered.status == "pcgrad_applicable"
    assert triggered.horizon_pair == (1, 2)
    assert triggered.conflicting_fold_count == 3

    insufficient = pd.DataFrame(rows).loc[lambda frame: frame["fold"].ne(2)]
    insufficient.loc[insufficient["fold"].eq(3), "cosine"] = 0.5
    insufficient.loc[insufficient["fold"].eq(3), "negative_cosine"] = False
    skipped = evaluate_pcgrad_trigger(insufficient)
    assert skipped.status == "pcgrad_not_applicable"


def test_pcgrad_projects_only_negative_components_with_seeded_order() -> None:
    aligned = {
        horizon: torch.tensor([float(horizon), 1.0])
        for horizon in [1, 2, 3, 5]
    }
    aligned_result = project_conflicting_gradients(aligned, seed=7)
    for horizon, gradient in aligned.items():
        torch.testing.assert_close(aligned_result.projected[horizon], gradient)

    conflicting = {
        1: torch.tensor([1.0, 0.0]),
        2: torch.tensor([-1.0, 0.0]),
        3: torch.tensor([0.0, 1.0]),
        5: torch.tensor([0.0, 0.0]),
    }
    first = project_conflicting_gradients(conflicting, seed=17)
    replay = project_conflicting_gradients(conflicting, seed=17)
    other_seed = project_conflicting_gradients(conflicting, seed=27)

    assert first.task_order == replay.task_order
    assert first.task_order != other_seed.task_order
    for horizon in conflicting:
        torch.testing.assert_close(first.projected[horizon], replay.projected[horizon])
        assert bool(torch.isfinite(first.projected[horizon]).all())
    assert torch.dot(first.projected[1], conflicting[2]) >= -1e-7
    assert torch.dot(first.projected[2], conflicting[1]) >= -1e-7


def test_pcgrad_localizes_one_horizon_pair_without_replacing_other_gradients() -> None:
    selected = nn.Parameter(torch.tensor([1.0, 1.0]))
    ordinary = nn.Parameter(torch.tensor([1.0]))
    losses = {
        1: selected[0] + ordinary[0],
        2: selected[1] + 2.0 * ordinary[0],
        3: selected[1] + 3.0 * ordinary[0],
        5: -selected[0] + selected[1] + 5.0 * ordinary[0],
    }
    total_loss = torch.stack(list(losses.values())).mean()

    receipt = pcgrad_backward(
        losses,
        {"block-4": (selected,)},
        seed=7,
        total_loss=total_loss,
        selected_horizons=(1, 5),
    )

    # The unselected parameter keeps the ordinary mean-loss gradient.
    torch.testing.assert_close(ordinary.grad, torch.tensor([2.75]))
    # Only the conflicting pair's contribution is projected on the selected block.
    torch.testing.assert_close(selected.grad, torch.tensor([0.125, 0.875]))
    assert receipt.selected_horizons == (1, 5)
    assert receipt.horizon_backward_seconds >= 0
    assert receipt.projection_seconds >= 0
