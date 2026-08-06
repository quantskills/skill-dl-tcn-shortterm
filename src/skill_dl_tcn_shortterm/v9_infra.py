"""Profile-gated, causal infrastructure options for TCN-v9."""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import dataclass
import json
import platform
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch.nn.utils.parametrizations import weight_norm

from .experiment import ContractError
from .runtime import torch_thread_scope
from .training_data import masked_smooth_l1


class CausalLiteBlockChomp(nn.Module):
    """State-compatible Conv1d overpadding with right chomp for strict causality."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        dropout_kind: Literal["element", "channel"],
    ) -> None:
        super().__init__()
        if kernel_size <= 1 or dilation <= 0:
            raise ContractError("padding-chomp kernel and dilation are invalid")
        if dropout_kind not in {"element", "channel"}:
            raise ContractError("padding-chomp dropout kind is invalid")
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = weight_norm(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                dilation=dilation,
                padding=self.left_padding,
            )
        )
        self.dropout: nn.Module = (
            nn.Dropout(dropout)
            if dropout_kind == "element"
            else nn.Dropout1d(dropout)
        )
        self.projection: nn.Module = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        overpadded = self.convolution(inputs)
        causal = overpadded[..., : inputs.shape[-1]]
        outputs = self.dropout(functional.relu(causal))
        return functional.relu(outputs + residual)


class TCNLiteChomp(nn.Module):
    """TCN-lite with the state-compatible causal padding/chomp blocks."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        input_channels = feature_count
        for dilation in dilations:
            blocks.append(
                CausalLiteBlockChomp(
                    input_channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    dropout_kind="element",
                )
            )
            input_channels = channels
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 4)

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.trunk(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_sequence(inputs)[:, :, -1])


@dataclass(frozen=True)
class OperatorProfile:
    operators: pd.DataFrame
    model_step_seconds_median: float
    samples_per_second: float
    measurement_noise: float
    torch_threads: int
    data_wait_seconds: float | None
    validation_seconds: float | None
    complete_cycle_seconds: float
    hardware_identity: str
    learning_rate: float


@dataclass(frozen=True)
class ThroughputBenchmark:
    samples_per_second_median: float
    seconds_per_call: tuple[float, ...]
    measurement_noise: float
    torch_threads: int


@dataclass(frozen=True)
class InfraGateDecision:
    status: str
    padding_self_cpu_share: float
    throughput_gain: float


@dataclass(frozen=True)
class CompileDecision:
    status: str
    reason: str | None
    throughput_gain: float | None = None
    graph_break_count: int = 0
    compiled_graph_count: int = 0


def _operator_family(operator: str) -> str:
    value = operator.lower()
    if "pad" in value:
        return "padding"
    if "weight_norm" in value or "weightnorm" in value:
        return "weight_norm"
    if "conv" in value:
        return "convolution"
    if "backward" in value or "autograd" in value:
        return "backward"
    if "optimizer" in value or "sgd" in value or "adam" in value:
        return "optimizer"
    if "relu" in value or "activation" in value:
        return "activation"
    return "other"


def _training_step(
    model: nn.Module,
    inputs: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = masked_smooth_l1(prediction, target, mask)
    loss.backward()
    optimizer.step()


def profile_model_step(
    model: nn.Module,
    inputs: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    torch_threads: int,
    learning_rate: float,
    warmup: int,
    repeats: int,
    formal_protocol: bool = True,
    data_wait_seconds: float | None = None,
    validation_seconds: float | None = None,
) -> OperatorProfile:
    """Record operator/shape/time/memory plus repeated model-step wall time."""

    if warmup < 0 or repeats <= 0 or len(inputs) <= 0 or learning_rate <= 0:
        raise ContractError("profile warmup, repeats, and batch are invalid")
    if formal_protocol and (
        inputs.dtype != torch.float32
        or target.dtype != torch.float32
        or inputs.ndim != 3
        or inputs.shape[-1] != 480
    ):
        raise ContractError(
            "formal TCN profile requires float32 [batch, feature, 480] inputs"
        )
    if formal_protocol and (
        data_wait_seconds is None
        or validation_seconds is None
        or data_wait_seconds < 0
        or validation_seconds < 0
    ):
        raise ContractError(
            "formal TCN profile requires measured data-wait and validation time"
        )
    if target.shape != model(inputs.detach()).shape or mask.shape != target.shape:
        raise ContractError("profile target and mask do not match model output")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    durations = []
    with torch_thread_scope(torch_threads) as effective_threads:
        for _ in range(warmup):
            _training_step(model, inputs, target, mask, optimizer)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as profile:
            for _ in range(repeats):
                start = time.perf_counter()
                _training_step(model, inputs, target, mask, optimizer)
                durations.append(time.perf_counter() - start)
    events: list[dict[str, object]] = []
    for event in profile.key_averages(group_by_input_shape=True):
        events.append(
            {
                "operator": str(event.key),
                "family": _operator_family(str(event.key)),
                "input_shapes": json.dumps(event.input_shapes, default=str),
                "self_cpu_time_us": float(event.self_cpu_time_total),
                "cpu_time_us": float(event.cpu_time_total),
                "self_cpu_memory_bytes": int(event.self_cpu_memory_usage),
            }
        )
    operators = pd.DataFrame(events)
    if operators.empty:
        raise ContractError("PyTorch profiler returned no CPU operators")
    total_self_cpu = float(operators["self_cpu_time_us"].sum())
    if total_self_cpu <= 0:
        raise ContractError("PyTorch profiler returned no positive self CPU time")
    operators["self_cpu_share"] = operators["self_cpu_time_us"] / total_self_cpu
    median = float(np.median(durations))
    q25, q75 = np.quantile(durations, [0.25, 0.75]).tolist()
    noise = float((q75 - q25) / median) if median > 0 else float("inf")
    return OperatorProfile(
        operators=operators.sort_values("self_cpu_time_us", ascending=False, kind="mergesort").reset_index(drop=True),
        model_step_seconds_median=median,
        samples_per_second=len(inputs) / median,
        measurement_noise=noise,
        torch_threads=effective_threads,
        data_wait_seconds=data_wait_seconds,
        validation_seconds=validation_seconds,
        complete_cycle_seconds=(
            float(sum(durations))
            + float(data_wait_seconds or 0.0)
            + float(validation_seconds or 0.0)
        ),
        hardware_identity=json.dumps(
            {
                "machine": platform.machine(),
                "processor": platform.processor(),
                "system": platform.system(),
                "torch": torch.__version__,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        learning_rate=float(learning_rate),
    )


def benchmark_throughput(
    operation: Callable[[], None],
    *,
    samples_per_call: int,
    warmup: int,
    repeats: int,
    torch_threads: int,
) -> ThroughputBenchmark:
    """Measure repeated throughput under a caller-restoring thread scope."""

    if samples_per_call <= 0 or warmup < 0 or repeats <= 0:
        raise ContractError("throughput benchmark arguments are invalid")
    durations = []
    with torch_thread_scope(torch_threads) as effective_threads:
        for _ in range(warmup):
            operation()
        for _ in range(repeats):
            start = time.perf_counter()
            operation()
            durations.append(time.perf_counter() - start)
    median = float(np.median(durations))
    q25, q75 = np.quantile(durations, [0.25, 0.75]).tolist()
    return ThroughputBenchmark(
        samples_per_second_median=samples_per_call / median,
        seconds_per_call=tuple(float(value) for value in durations),
        measurement_noise=float((q75 - q25) / median),
        torch_threads=effective_threads,
    )


def evaluate_infra_gate(
    operators: pd.DataFrame,
    *,
    eager_samples_per_second: float,
    candidate_samples_per_second: float,
    numerically_equivalent: bool,
    strictly_causal: bool,
) -> InfraGateDecision:
    """Accept padding-chomp only behind the frozen 10% profile and gain gates."""

    required = {"family", "self_cpu_share"}
    if missing := sorted(required.difference(operators.columns)):
        raise ContractError(f"operator profile missing columns: {', '.join(missing)}")
    if eager_samples_per_second <= 0 or candidate_samples_per_second <= 0:
        raise ContractError("infra throughput measurements must be positive")
    shares = operators.groupby("family", observed=True)["self_cpu_share"].sum()
    padding_share = float(shares.get("padding", 0.0))
    gain = candidate_samples_per_second / eager_samples_per_second - 1.0
    accepted = (
        padding_share >= 0.10
        and gain >= 0.10
        and numerically_equivalent
        and strictly_causal
    )
    return InfraGateDecision(
        status=(
            "causal_infra_acceleration_accepted"
            if accepted
            else "infra_optimization_not_applicable"
        ),
        padding_self_cpu_share=padding_share,
        throughput_gain=float(gain),
    )


def compile_with_eager_fallback(
    model: nn.Module,
    example_input: torch.Tensor,
    *,
    enabled: bool,
    compiler: Callable[[nn.Module], nn.Module] | None = None,
    warmup: int = 1,
    repeats: int = 3,
    minimum_gain: float = 0.10,
) -> tuple[nn.Module, CompileDecision]:
    """Bound compile capture and retain eager as the only required path."""

    if not enabled:
        return model, CompileDecision("compile_disabled_eager", None)
    if warmup < 1 or repeats < 2 or minimum_gain < 0:
        raise ContractError("compile A/B benchmark arguments are invalid")
    selected_compiler: Any = compiler if compiler is not None else torch.compile
    try:
        counters = getattr(getattr(torch, "_dynamo", None), "utils", None)
        counter_map = getattr(counters, "counters", {})
        breaks_before = sum(counter_map.get("graph_break", {}).values())
        graphs_before = int(counter_map.get("stats", {}).get("unique_graphs", 0))
        candidate_model = copy.deepcopy(model)
        compiled = selected_compiler(candidate_model)
        eager_input = example_input.detach().clone().requires_grad_(True)
        compiled_input = example_input.detach().clone().requires_grad_(True)
        eager_output = model(eager_input)
        compiled_output = compiled(compiled_input)
        if not torch.allclose(eager_output, compiled_output, rtol=1e-5, atol=1e-6):
            return model, CompileDecision(
                "compile_fallback_eager", "compiled output mismatch"
            )
        eager_gradient = torch.autograd.grad(
            eager_output.sum(), eager_input, retain_graph=True
        )[0]
        compiled_gradient = torch.autograd.grad(
            compiled_output.sum(), compiled_input, retain_graph=True
        )[0]
        if not torch.allclose(
            eager_gradient,
            compiled_gradient,
            rtol=1e-5,
            atol=1e-6,
        ):
            return model, CompileDecision(
                "compile_fallback_eager", "compiled input gradient mismatch"
            )
        eager_parameter_gradients = torch.autograd.grad(
            eager_output.sum(), tuple(model.parameters()), allow_unused=True
        )
        compiled_parameter_gradients = torch.autograd.grad(
            compiled_output.sum(),
            tuple(candidate_model.parameters()),
            allow_unused=True,
        )
        if any(
            (left is None) != (right is None)
            or (
                left is not None
                and right is not None
                and not torch.allclose(left, right, rtol=1e-5, atol=1e-6)
            )
            for left, right in zip(
                eager_parameter_gradients,
                compiled_parameter_gradients,
                strict=True,
            )
        ):
            return model, CompileDecision(
                "compile_fallback_eager", "compiled parameter gradient mismatch"
            )
        eager_optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
        compiled_optimizer = torch.optim.Adam(candidate_model.parameters(), lr=0.0)

        def train_step(
            selected_model: nn.Module,
            optimizer: torch.optim.Optimizer,
        ) -> None:
            optimizer.zero_grad(set_to_none=True)
            output = selected_model(example_input)
            output.square().mean().backward()
            optimizer.step()

        for _ in range(warmup):
            train_step(model, eager_optimizer)
            train_step(compiled, compiled_optimizer)
        eager_durations = []
        compiled_durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            train_step(model, eager_optimizer)
            eager_durations.append(time.perf_counter() - start)
            start = time.perf_counter()
            train_step(compiled, compiled_optimizer)
            compiled_durations.append(time.perf_counter() - start)
        breaks_after = sum(counter_map.get("graph_break", {}).values())
        graphs_after = int(counter_map.get("stats", {}).get("unique_graphs", 0))
        graph_breaks = max(0, breaks_after - breaks_before)
        compiled_graphs = max(0, graphs_after - graphs_before)
        gain = float(
            np.median(eager_durations) / np.median(compiled_durations) - 1.0
        )
        if graph_breaks > 0 or compiled_graphs > 1 or gain < minimum_gain:
            reason = (
                "graph break"
                if graph_breaks > 0
                else "recompilation"
                if compiled_graphs > 1
                else "insufficient stable throughput gain"
            )
            return model, CompileDecision(
                "compile_fallback_eager",
                reason,
                gain,
                graph_breaks,
                compiled_graphs,
            )
    except Exception as exc:  # compile support is deliberately best-effort
        return model, CompileDecision("compile_fallback_eager", type(exc).__name__)
    return compiled, CompileDecision(
        "compile_candidate_ready",
        None,
        gain,
        graph_breaks,
        compiled_graphs,
    )
