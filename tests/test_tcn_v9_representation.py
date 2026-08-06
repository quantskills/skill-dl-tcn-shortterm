from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.v9_representation import (
    DecoupledResidualTemporalContextTCN,
    HorizonSkipTCN,
    ProbeCheckpointEvidence,
    SignedTemporalContextTCN,
    StabilizedResidualTemporalContextTCN,
    TemporalContextTCN,
    checkpoint_state_identity,
    evaluate_horizon_skip_trigger,
    run_layer_probes,
)


def test_temporal_context_tcn_reads_full_days_and_intraday_sequence() -> None:
    torch.manual_seed(14)
    model = TemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 3, 16)

    outputs = model(inputs)
    day_weights = model.day_weights()
    intraday_weights = model.intraday_weights()

    assert outputs.shape == (2, 4)
    assert day_weights.shape == (4, 4)
    assert intraday_weights.shape == (4, 4)
    torch.testing.assert_close(day_weights, torch.full((4, 4), 0.25))
    torch.testing.assert_close(intraday_weights, torch.full((4, 4), 0.25))

    hidden = torch.zeros(1, 4, 16)
    changed = hidden.clone()
    changed[:, :, :4] = 1.0
    with torch.no_grad():
        for head in model.heads:
            assert isinstance(head, torch.nn.Linear)
            torch.nn.init.constant_(head.weight, 1.0)
            torch.nn.init.zeros_(head.bias)
    baseline = model.readout_sequence(hidden)
    historical_change = model.readout_sequence(changed)
    assert bool((historical_change - baseline).abs().gt(0).all())

    metadata = model.receipt_metadata()
    assert metadata["readout"] == "horizon_dual_scale_full_sequence"
    assert metadata["bars_per_day"] == 4
    assert metadata["day_count"] == 4
    assert metadata["normalization"] == "weight_norm"


def test_temporal_context_tcn_rejects_partial_trading_days() -> None:
    with pytest.raises(ContractError, match="divisible"):
        TemporalContextTCN(
            feature_count=3,
            channels=4,
            kernel_size=2,
            dilations=(1, 2, 4, 8),
            input_steps=15,
            bars_per_day=4,
            dropout=0.0,
        )


def test_signed_temporal_context_matches_simplex_initially_and_allows_negatives() -> None:
    torch.manual_seed(29)
    control = TemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
    ).eval()
    torch.manual_seed(29)
    candidate = SignedTemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 3, 16)

    torch.testing.assert_close(candidate(inputs), control(inputs))
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    torch.testing.assert_close(candidate.day_weights(), torch.full((4, 4), 0.25))
    torch.testing.assert_close(
        candidate.intraday_weights(), torch.full((4, 4), 0.25)
    )

    with torch.no_grad():
        candidate.day_adapter.weight[0, 0] = -0.75
        candidate.intraday_adapter.weight[0, 0] = -0.5
    assert float(candidate.day_weights()[0, 0].detach()) == pytest.approx(-0.75)
    assert float(candidate.intraday_weights()[0, 0].detach()) == pytest.approx(-0.5)
    changed = candidate(inputs)
    assert not torch.allclose(changed[:, 0], control(inputs)[:, 0])
    metadata = candidate.receipt_metadata()
    assert metadata["readout"] == "horizon_dual_scale_signed_adapter"
    assert metadata["signed_temporal_weights"] is True
    day_negative = metadata["day_negative_weight_count"]
    intraday_negative = metadata["intraday_negative_weight_count"]
    assert isinstance(day_negative, list)
    assert isinstance(intraday_negative, list)
    assert day_negative[0] == 1
    assert intraday_negative[0] == 1


def test_stabilized_residual_context_is_bounded_parameter_matched_and_signed() -> None:
    torch.manual_seed(41)
    control = TemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
    ).eval()
    torch.manual_seed(41)
    candidate = StabilizedResidualTemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
        residual_scale=0.05,
    ).eval()
    inputs = torch.randn(2, 3, 16)

    torch.testing.assert_close(candidate(inputs), control(inputs))
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    assert set(map(id, candidate.temporal_adapter_parameters())) == {
        id(candidate.day_logits),
        id(candidate.intraday_logits),
    }

    with torch.no_grad():
        extreme = torch.tensor([10.0, -10.0, -10.0, -10.0])
        candidate.day_logits[0].copy_(extreme)
        candidate.intraday_logits[0].copy_(extreme)
    day_simplex = torch.softmax(candidate.day_logits, dim=1)
    intraday_simplex = torch.softmax(candidate.intraday_logits, dim=1)
    day_residual = candidate.day_weights() - day_simplex
    intraday_residual = candidate.intraday_weights() - intraday_simplex

    torch.testing.assert_close(candidate.day_weights().sum(dim=1), torch.ones(4))
    torch.testing.assert_close(
        candidate.intraday_weights().sum(dim=1), torch.ones(4)
    )
    assert float(day_residual.detach().abs().max()) <= 0.1 + 1e-7
    assert float(intraday_residual.detach().abs().max()) <= 0.1 + 1e-7
    assert bool(candidate.day_weights().lt(0).any())
    assert bool(candidate.intraday_weights().lt(0).any())
    metadata = candidate.receipt_metadata()
    assert metadata["readout"] == "horizon_dual_scale_stabilized_signed_residual"
    assert metadata["residual_scale"] == pytest.approx(0.05)
    assert metadata["signed_residual_bounded"] is True


def test_decoupled_residual_preserves_base_and_adds_only_independent_residual() -> None:
    torch.manual_seed(53)
    control = TemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
    ).eval()
    torch.manual_seed(53)
    candidate = DecoupledResidualTemporalContextTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        bars_per_day=4,
        dropout=0.0,
        residual_scale=0.05,
    ).eval()
    inputs = torch.randn(2, 3, 16)

    torch.testing.assert_close(candidate(inputs), control(inputs))
    control_count = sum(parameter.numel() for parameter in control.parameters())
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    assert candidate_count - control_count == 32
    base = candidate.temporal_adapter_parameters()
    residual = candidate.residual_adapter_parameters()
    assert set(map(id, base)).isdisjoint(map(id, residual))
    assert all(
        left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
        for left in base
        for right in residual
    )

    original_simplex = candidate.day_simplex_weights().detach().clone()
    with torch.no_grad():
        candidate.day_residual_logits[0].copy_(
            torch.tensor([10.0, -10.0, -10.0, -10.0])
        )
    torch.testing.assert_close(candidate.day_simplex_weights(), original_simplex)
    torch.testing.assert_close(candidate.day_weights().sum(dim=1), torch.ones(4))
    assert float(candidate.day_residual().detach().abs().max()) <= 0.1 + 1e-7
    with torch.no_grad():
        signed_pattern = torch.tensor([-10.0, 10.0, 10.0, 10.0])
        candidate.day_logits[0].copy_(signed_pattern)
        candidate.day_residual_logits[0].copy_(signed_pattern)
    assert bool(candidate.day_weights().lt(0).any())
    metadata = candidate.receipt_metadata()
    assert metadata["readout"] == "horizon_dual_scale_decoupled_signed_residual"
    assert metadata["residual_scale"] == pytest.approx(0.05)
    assert metadata["base_temporal_parameter_count"] == 32
    assert metadata["residual_parameter_count"] == 32


def test_horizon_skip_is_simplex_weighted_shape_safe_and_strictly_causal() -> None:
    torch.manual_seed(4)
    model = HorizonSkipTCN(
        feature_count=3,
        channels=4,
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        input_steps=16,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 3, 16)

    weights = model.simplex_weights()
    outputs = model(inputs)
    original_blocks = model.encode_blocks(inputs)
    changed = inputs.clone()
    changed[:, :, 10:] += 10_000
    changed_blocks = model.encode_blocks(changed)

    assert outputs.shape == (2, 4)
    assert weights.shape == (4, 4)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))
    assert bool((weights >= 0).all())
    for original, perturbed in zip(original_blocks, changed_blocks, strict=True):
        torch.testing.assert_close(original[:, :, :10], perturbed[:, :, :10])
    assert model.receptive_field == 16
    assert model.receipt_metadata()["normalization"] == "weight_norm"
    assert model.receipt_metadata()["readout"] == "horizon_simplex_last_valid"


def test_layer_probe_fits_on_train_and_reports_validation_by_fold_horizon_block(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(8)
    train_count = 60
    validation_count = 40
    channels = 3
    train_signal = rng.normal(size=train_count)
    validation_signal = rng.normal(size=validation_count)
    train_blocks = rng.normal(size=(train_count, 2, channels)) * 0.05
    validation_blocks = rng.normal(size=(validation_count, 2, channels)) * 0.05
    train_blocks[:, 0, 0] = train_signal
    validation_blocks[:, 0, 0] = validation_signal
    train_targets = np.column_stack(
        [train_signal, train_signal, train_signal, train_signal]
    ).astype("float32")
    validation_targets = np.column_stack(
        [validation_signal, validation_signal, validation_signal, validation_signal]
    ).astype("float32")
    train_masks = np.ones_like(train_targets, dtype=bool)
    validation_masks = np.ones_like(validation_targets, dtype=bool)
    train_dates = np.repeat([f"2025-01-{day:02d}" for day in range(1, 7)], 10)
    validation_dates = np.repeat(
        [f"2025-02-{day:02d}" for day in range(1, 5)], 10
    )

    checkpoint_path = tmp_path / "lite-checkpoint.pt"
    torch.save({"weight": torch.ones(1)}, checkpoint_path)
    config_path = tmp_path / "lite-config.json"
    config_path.write_text('{"model":"tcn-lite-16"}', encoding="utf-8")
    checkpoint_identity = checkpoint_state_identity(checkpoint_path)
    checkpoint = ProbeCheckpointEvidence(
        model_family="tcn-lite-16",
        config_identity=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        data_identity="d" * 64,
        fold_identity="e" * 64,
        checkpoint_identity=checkpoint_identity,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    expected_identities = {
        "data": "d" * 64,
        "fold_manifest": "e" * 64,
        "lite_config": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "lite_checkpoint_fold_2": checkpoint_identity,
    }
    result = run_layer_probes(
        train_blocks,
        train_targets,
        train_masks,
        train_dates,
        validation_blocks,
        validation_targets,
        validation_masks,
        validation_dates,
        fold=2,
        checkpoint=checkpoint,
        expected_identities=expected_identities,
    )
    mutated_validation = run_layer_probes(
        train_blocks,
        train_targets,
        train_masks,
        train_dates,
        validation_blocks,
        -validation_targets,
        validation_masks,
        validation_dates,
        fold=2,
        checkpoint=checkpoint,
        expected_identities=expected_identities,
    )

    assert len(result.metrics) == 8
    assert set(result.metrics["fold"]) == {2}
    assert set(result.metrics["horizon"]) == {1, 2, 3, 5}
    assert set(result.metrics["block"]) == {0, 1}
    assert set(result.metrics["checkpoint_identity"]) == {checkpoint_identity}
    assert result.metrics.loc[result.metrics["block"].eq(0), "rankic"].min() > 0.9
    assert result.metrics.loc[result.metrics["block"].eq(1), "rankic"].max() < 0.5
    for key in result.coefficients:
        np.testing.assert_allclose(
            result.coefficients[key], mutated_validation.coefficients[key]
        )


def test_horizon_skip_trigger_requires_effect_folds_and_bootstrap_lower_bound() -> None:
    rows = []
    for fold in range(5):
        for day in range(12):
            for block, rankic in [(0, 0.07 + day * 0.0001), (1, 0.05)]:
                rows.append(
                    {
                        "model_family": "tcn-lite-16",
                        "fold": fold,
                        "horizon": 1,
                        "block": block,
                        "signal_date": f"2025-{fold + 1:02d}-{day + 1:02d}",
                        "rankic": rankic,
                    }
                )
            for block in [0, 1, 2]:
                rows.append(
                    {
                        "model_family": "bai-tcn-16",
                        "fold": fold,
                        "horizon": 1,
                        "block": block,
                        "signal_date": f"2025-{fold + 1:02d}-{day + 1:02d}",
                        "rankic": 0.08 if block == 0 else 0.04,
                    }
                )
    triggered = evaluate_horizon_skip_trigger(pd.DataFrame(rows), seed=7)
    assert triggered.status == "horizon_skip_applicable"
    assert triggered.selected_block == 0
    assert triggered.model_family == "tcn-lite-16"
    assert triggered.positive_fold_count == 5
    assert triggered.ci_low > 0

    unchanged = pd.DataFrame(rows)
    unchanged.loc[
        unchanged["model_family"].eq("tcn-lite-16")
        & unchanged["block"].eq(0),
        "rankic",
    ] = 0.0505
    skipped = evaluate_horizon_skip_trigger(unchanged, seed=7)
    assert skipped.status == "horizon_skip_not_applicable"
    assert skipped.selected_block is None
