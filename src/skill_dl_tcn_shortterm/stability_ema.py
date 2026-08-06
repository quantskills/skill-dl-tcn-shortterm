"""Single-model parameter EMA with auditable raw-trajectory isolation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from .experiment import ContractError


class ParameterEMA:
    """Track trainable floating parameters without changing model execution."""

    def __init__(self, *, decay: float) -> None:
        if not np.isfinite(decay) or not 0.0 < decay < 1.0:
            raise ContractError("EMA decay must be finite and in (0, 1)")
        self.decay = float(decay)
        self._shadow: dict[str, torch.Tensor] = {}
        self._update_count = 0

    @property
    def update_count(self) -> int:
        return self._update_count

    @staticmethod
    def _tracked_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        }
        if not parameters:
            raise ContractError("EMA model has no trainable floating parameters")
        return parameters

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Copy on the first optimizer step, then update the shadow in place."""

        parameters = self._tracked_parameters(model)
        if self._update_count == 0:
            self._shadow = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
        else:
            if set(parameters) != set(self._shadow):
                raise ContractError("EMA parameter identity drifted")
            for name, parameter in parameters.items():
                shadow = self._shadow[name]
                if (
                    shadow.shape != parameter.shape
                    or shadow.dtype != parameter.dtype
                    or shadow.device != parameter.device
                ):
                    raise ContractError(f"EMA parameter metadata drifted: {name}")
                shadow.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        self._update_count += 1

    def averaged_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Return a normal state dict with EMA parameters and current buffers."""

        if self._update_count == 0:
            raise ContractError("EMA state requested before the first update")
        parameters = self._tracked_parameters(model)
        if set(parameters) != set(self._shadow):
            raise ContractError("EMA parameter identity drifted")
        state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        for name, shadow in self._shadow.items():
            if name not in state:
                raise ContractError(f"EMA parameter is absent from state dict: {name}")
            state[name] = shadow.detach().clone()
        return state

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap EMA parameters into a model and always restore raw values."""

        if self._update_count == 0:
            raise ContractError("EMA parameters requested before the first update")
        parameters = self._tracked_parameters(model)
        if set(parameters) != set(self._shadow):
            raise ContractError("EMA parameter identity drifted")
        backup = {
            name: parameter.detach().clone()
            for name, parameter in parameters.items()
        }
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(self._shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(backup[name])


class EpochParameterAverage:
    """Track the exact uniform mean of epoch-boundary trainable parameters."""

    def __init__(self) -> None:
        self._average: dict[str, torch.Tensor] = {}
        self._update_count = 0

    @property
    def update_count(self) -> int:
        return self._update_count

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Include one epoch-boundary state in the online arithmetic mean."""

        parameters = ParameterEMA._tracked_parameters(model)
        if self._update_count == 0:
            self._average = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
        else:
            if set(parameters) != set(self._average):
                raise ContractError("epoch average parameter identity drifted")
            next_count = self._update_count + 1
            for name, parameter in parameters.items():
                average = self._average[name]
                if (
                    average.shape != parameter.shape
                    or average.dtype != parameter.dtype
                    or average.device != parameter.device
                ):
                    raise ContractError(
                        f"epoch average parameter metadata drifted: {name}"
                    )
                average.add_(
                    (parameter.detach() - average) / float(next_count)
                )
        self._update_count += 1

    def averaged_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Return the uniform parameter mean with current non-parameter buffers."""

        if self._update_count == 0:
            raise ContractError("epoch average requested before the first update")
        parameters = ParameterEMA._tracked_parameters(model)
        if set(parameters) != set(self._average):
            raise ContractError("epoch average parameter identity drifted")
        state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        for name, average in self._average.items():
            if name not in state:
                raise ContractError(
                    f"epoch average parameter is absent from state dict: {name}"
                )
            state[name] = average.detach().clone()
        return state

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap the uniform epoch mean into a raw model."""

        if self._update_count == 0:
            raise ContractError("epoch average requested before the first update")
        parameters = ParameterEMA._tracked_parameters(model)
        if set(parameters) != set(self._average):
            raise ContractError("epoch average parameter identity drifted")
        backup = {
            name: parameter.detach().clone()
            for name, parameter in parameters.items()
        }
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(self._average[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(backup[name])


def state_dict_max_abs_error(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    """Return exact-state drift, failing closed on identity or metadata changes."""

    if set(left) != set(right):
        raise ContractError("state dict identities differ")
    maximum = 0.0
    for name in sorted(left):
        left_tensor = left[name]
        right_tensor = right[name]
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            raise ContractError(f"state dict tensor metadata differs: {name}")
        if left_tensor.is_floating_point() or left_tensor.is_complex():
            if left_tensor.numel():
                error = float(
                    torch.max(
                        torch.abs(left_tensor.detach().cpu() - right_tensor.detach().cpu())
                    )
                )
                if not np.isfinite(error):
                    raise ContractError(f"state dict drift is non-finite: {name}")
                maximum = max(maximum, error)
        elif not torch.equal(left_tensor.detach().cpu(), right_tensor.detach().cpu()):
            raise ContractError(f"non-floating state dict tensor differs: {name}")
    return maximum
