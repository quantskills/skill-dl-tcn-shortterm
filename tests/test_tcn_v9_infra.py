from __future__ import annotations

import pandas as pd
import pytest
import torch

from skill_dl_tcn_shortterm.tcn_lite import CausalLiteBlock
from skill_dl_tcn_shortterm.v9_infra import (
    CausalLiteBlockChomp,
    benchmark_throughput,
    compile_with_eager_fallback,
    evaluate_infra_gate,
    profile_model_step,
)


@pytest.mark.parametrize(("kernel_size", "dilation"), [(2, 1), (3, 4), (5, 8)])
def test_padding_chomp_matches_explicit_left_padding_and_input_gradients(
    kernel_size: int,
    dilation: int,
) -> None:
    torch.manual_seed(kernel_size * 10 + dilation)
    explicit = CausalLiteBlock(
        3,
        4,
        kernel_size=kernel_size,
        dilation=dilation,
        dropout=0.0,
        dropout_kind="element",
    ).eval()
    chomp = CausalLiteBlockChomp(
        3,
        4,
        kernel_size=kernel_size,
        dilation=dilation,
        dropout=0.0,
        dropout_kind="element",
    ).eval()
    chomp.load_state_dict(explicit.state_dict(), strict=True)
    explicit_input = torch.randn(2, 3, 64, requires_grad=True)
    chomp_input = explicit_input.detach().clone().requires_grad_(True)

    explicit_output = explicit(explicit_input)
    chomp_output = chomp(chomp_input)
    explicit_output.square().sum().backward()
    chomp_output.square().sum().backward()

    torch.testing.assert_close(chomp_output, explicit_output, rtol=1e-5, atol=1e-6)
    assert explicit_input.grad is not None and chomp_input.grad is not None
    torch.testing.assert_close(chomp_input.grad, explicit_input.grad, rtol=1e-5, atol=1e-6)
    assert chomp_output.shape[-1] == explicit_input.shape[-1]
    assert set(chomp.state_dict()) == set(explicit.state_dict())

    changed = chomp_input.detach().clone()
    changed[:, :, 40:] += 1000
    with torch.no_grad():
        torch.testing.assert_close(chomp(chomp_input.detach())[:, :, :40], chomp(changed)[:, :, :40])


def test_profile_records_operator_shapes_memory_and_restores_threads() -> None:
    original_threads = torch.get_num_threads()
    model = CausalLiteBlock(
        2,
        2,
        kernel_size=3,
        dilation=2,
        dropout=0.0,
        dropout_kind="element",
    )
    inputs = torch.randn(4, 2, 32)
    target = torch.zeros(4, 2, 32)
    mask = torch.ones_like(target, dtype=torch.bool)

    profile = profile_model_step(
        model,
        inputs,
        target,
        mask,
        torch_threads=1,
        learning_rate=0.003,
        warmup=1,
        repeats=2,
        formal_protocol=False,
    )

    assert torch.get_num_threads() == original_threads
    assert not profile.operators.empty
    assert {
        "operator",
        "family",
        "input_shapes",
        "self_cpu_time_us",
        "cpu_time_us",
        "self_cpu_memory_bytes",
        "self_cpu_share",
    } <= set(profile.operators)
    assert profile.model_step_seconds_median > 0
    assert profile.samples_per_second > 0
    assert profile.learning_rate == 0.003
    assert profile.measurement_noise >= 0
    assert profile.hardware_identity

    with pytest.raises(Exception, match="480"):
        profile_model_step(
            model,
            inputs,
            target,
            mask,
            torch_threads=1,
            learning_rate=0.003,
            warmup=0,
            repeats=1,
            formal_protocol=True,
        )


def test_infra_gate_requires_profile_share_equivalence_causality_and_ten_percent_gain() -> None:
    operators = pd.DataFrame(
        {
            "operator": ["aten::constant_pad_nd", "aten::convolution"],
            "family": ["padding", "convolution"],
            "input_shapes": ["[]", "[]"],
            "self_cpu_time_us": [20.0, 80.0],
            "cpu_time_us": [20.0, 80.0],
            "self_cpu_memory_bytes": [0, 0],
            "self_cpu_share": [0.20, 0.80],
        }
    )
    accepted = evaluate_infra_gate(
        operators,
        eager_samples_per_second=5000,
        candidate_samples_per_second=5600,
        numerically_equivalent=True,
        strictly_causal=True,
    )
    assert accepted.status == "causal_infra_acceleration_accepted"
    assert accepted.throughput_gain == pytest.approx(0.12)

    too_small = evaluate_infra_gate(
        operators,
        eager_samples_per_second=5000,
        candidate_samples_per_second=5400,
        numerically_equivalent=True,
        strictly_causal=True,
    )
    assert too_small.status == "infra_optimization_not_applicable"


def test_compile_failure_falls_back_cleanly_and_benchmark_restores_on_error() -> None:
    model = torch.nn.Linear(2, 2)
    example = torch.ones(1, 2)

    def broken_compiler(module: torch.nn.Module) -> torch.nn.Module:
        raise RuntimeError("unsupported")

    selected, decision = compile_with_eager_fallback(
        model,
        example,
        enabled=True,
        compiler=broken_compiler,
    )
    assert selected is model
    assert decision.status == "compile_fallback_eager"

    original_threads = torch.get_num_threads()

    def failing_operation() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        benchmark_throughput(
            failing_operation,
            samples_per_call=1,
            warmup=0,
            repeats=1,
            torch_threads=1,
        )
    assert torch.get_num_threads() == original_threads


def test_compile_candidate_with_wrong_input_gradient_falls_back_to_eager() -> None:
    model = torch.nn.Linear(2, 2)
    example = torch.ones(1, 2)

    class WrongGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
            return value.clone()

        @staticmethod
        def backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(gradient)

    class Compiled(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return WrongGradient.apply(self.wrapped(value))

    selected, decision = compile_with_eager_fallback(
        model,
        example,
        enabled=True,
        compiler=Compiled,
        warmup=1,
        repeats=2,
    )

    assert selected is model
    assert decision.status == "compile_fallback_eager"
    assert decision.reason == "compiled input gradient mismatch"
