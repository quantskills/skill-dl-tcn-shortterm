from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.real_validation import parse_real_tcn_trials
from skill_dl_tcn_shortterm.tuning import (
    run_tcn_validation_sweep,
    validate_tcn_tuning_plan,
)
from skill_dl_tcn_shortterm.v9_objective import mixed_date_grouped_top_tail_loss
from skill_dl_tcn_shortterm.v9_representation import DynamicHorizonSkipTCN


def _top_tail_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    prediction = torch.zeros((10, 4), dtype=torch.float32, requires_grad=True)
    target = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 4)
    mask = torch.ones_like(target, dtype=torch.bool)
    return prediction, target, mask, ["2025-08-01"] * 10


def test_v34_top_tail_loss_pairs_realized_top_decile_against_the_rest() -> None:
    prediction, target, mask, dates = _top_tail_inputs()

    result = mixed_date_grouped_top_tail_loss(
        prediction,
        target,
        mask,
        dates,
        top_tail_weight=0.05,
        top_fraction=0.1,
        temperature=0.1,
    )

    assert result.status == "top_tail_active"
    assert result.group_count == 4
    assert result.pair_count == 36
    assert torch.isfinite(result.total)
    result.top_tail.backward()
    assert prediction.grad is not None
    assert torch.all(prediction.grad[9] < 0)
    assert torch.all(prediction.grad[:9] > 0)


def test_v34_top_tail_loss_is_invariant_to_non_top_score_permutation() -> None:
    prediction, target, mask, dates = _top_tail_inputs()
    with torch.no_grad():
        prediction[:, :] = torch.linspace(-1.0, 1.0, 10)[:, None]
    first = mixed_date_grouped_top_tail_loss(
        prediction,
        target,
        mask,
        dates,
        top_tail_weight=0.05,
        top_fraction=0.1,
        temperature=0.1,
    )
    permuted = prediction.detach().clone()
    permuted[:9] = permuted[torch.tensor([8, 0, 7, 1, 6, 2, 5, 3, 4])]
    second = mixed_date_grouped_top_tail_loss(
        permuted,
        target,
        mask,
        dates,
        top_tail_weight=0.05,
        top_fraction=0.1,
        temperature=0.1,
    )

    assert float(first.top_tail.detach()) == pytest.approx(
        float(second.top_tail.detach()), abs=1e-7
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_tail_weight": 0.0}, "weight"),
        ({"top_fraction": 0.0}, "fraction"),
        ({"top_fraction": 0.75}, "fraction"),
        ({"temperature": 0.0}, "temperature"),
    ],
)
def test_v34_top_tail_loss_rejects_unregistered_parameters(
    overrides: dict[str, float], message: str
) -> None:
    prediction, target, mask, dates = _top_tail_inputs()
    parameters = {
        "top_tail_weight": 0.05,
        "top_fraction": 0.1,
        "temperature": 0.1,
        **overrides,
    }
    with pytest.raises(ContractError, match=message):
        mixed_date_grouped_top_tail_loss(
            prediction, target, mask, dates, **parameters
        )


def _raw_top_tail_trial() -> dict[str, object]:
    return {
        "trial_id": "top-tail",
        "channels": 4,
        "kernel_size": 2,
        "dilations": [1, 2, 4, 8],
        "dropout": 0.0,
        "learning_rate": 0.003,
        "batch_size": 8,
        "model_kind": "dynamic_horizon_skip",
        "strategy": "top_tail",
        "padding_mode": "chomp",
        "dynamic_skip_hidden": 2,
        "dynamic_skip_scale": 1.0,
        "dynamic_skip_shape_residual": True,
        "dynamic_skip_shape_residual_scale": 0.25,
        "dynamic_skip_frozen_parent": True,
        "date_batch_order": "fixed_once",
        "top_tail_weight": 0.05,
        "top_tail_fraction": 0.1,
        "top_tail_temperature": 0.1,
    }


def test_v34_parser_and_plan_require_explicit_frozen_shape_top_tail_contract() -> None:
    trial = parse_real_tcn_trials([_raw_top_tail_trial()])[0]
    assert trial.strategy == "top_tail"
    assert trial.top_tail_weight == pytest.approx(0.05)
    assert trial.top_tail_fraction == pytest.approx(0.1)
    assert trial.top_tail_temperature == pytest.approx(0.1)
    assert validate_tcn_tuning_plan(
        (trial,), input_steps=16, max_epochs=2, patience=1, min_delta=0.0005
    ) == (trial,)

    missing = _raw_top_tail_trial()
    missing.pop("top_tail_weight")
    with pytest.raises(ContractError, match="explicit public parameters"):
        parse_real_tcn_trials([missing])

    nonfrozen = _raw_top_tail_trial()
    nonfrozen["dynamic_skip_frozen_parent"] = False
    with pytest.raises(ContractError, match="frozen parent shape residual"):
        parsed = parse_real_tcn_trials([nonfrozen])[0]
        validate_tcn_tuning_plan(
            (parsed,), input_steps=16, max_epochs=2, patience=1, min_delta=0.0005
        )


def test_v34_training_records_top_tail_pairs_and_component_gradient_cosine() -> None:
    trial = parse_real_tcn_trials([_raw_top_tail_trial()])[0]
    rng = np.random.default_rng(34)
    features = rng.normal(size=(18, 3, 16)).astype("float32")
    index = pd.DataFrame(
        {
            "sample_position": range(18),
            "sample_id": [f"top-{value}" for value in range(18)],
            "signal_date": [
                f"2025-08-{1 + value // 6:02d}" for value in range(18)
            ],
        }
    )
    labels = pd.DataFrame(
        [
            {
                "sample_id": f"top-{sample}",
                "signal_date": index.loc[sample, "signal_date"],
                "horizon": horizon,
                "rank_target": float((sample % 6) / 5 * 2 - 1),
                "valid": True,
            }
            for sample in range(18)
            for horizon in (1, 2, 3, 5)
        ]
    )
    split = index[["sample_position"]].copy()
    split["fold"] = 0
    split["stage"] = ["train"] * 12 + ["validation"] * 6
    parent = DynamicHorizonSkipTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        dropout=0.0,
        padding_mode="chomp",
        dynamic_skip_hidden=2,
        dynamic_skip_scale=1.0,
    )
    parent_state = {
        name: tensor.detach().clone() for name, tensor in parent.state_dict().items()
    }

    result = run_tcn_validation_sweep(
        features,
        index,
        labels,
        split,
        trials=(trial,),
        seed=7,
        max_epochs=2,
        patience=1,
        min_delta=0.0005,
        checkpoint_min_delta=0.0,
        torch_threads=1,
        frozen_parent_states={"top-tail-fold-0": parent_state},
    )

    row = result.leaderboard.iloc[0]
    assert row["loss_identity"] == "smooth-l1+0.05-top-tail-fraction-0.1-tau-0.1"
    assert row["batching_identity"] == "date-grouped"
    assert float(row["median_top_tail_pair_count"]) > 0
    assert np.isfinite(float(row["median_component_gradient_cosine"]))
    history = result.epoch_history.loc[
        result.epoch_history["stage"].eq("validation")
    ]
    assert history["top_tail_pair_count_min"].astype(float).gt(0).all()
    assert history["component_gradient_cosine_median"].astype(float).map(
        np.isfinite
    ).all()
