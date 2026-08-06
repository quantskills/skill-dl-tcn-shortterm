"""TCN-v9 block probes and horizon-conditioned causal skip readout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import torch
from torch import nn

from .baselines import _block_bootstrap_means
from .experiment import ContractError
from .neural import HORIZONS
from .tcn_lite import CausalLiteBlock, lite_receptive_field
from .v9_infra import CausalLiteBlockChomp


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HorizonSkipTCN(nn.Module):
    """TCN-lite trunk with one simplex block mixture per forecast horizon."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        input_steps: int,
        dropout: float,
        padding_mode: Literal["explicit", "chomp"] = "explicit",
    ) -> None:
        super().__init__()
        if feature_count <= 0 or channels <= 0 or kernel_size <= 1:
            raise ContractError("horizon skip TCN dimensions are invalid")
        if not dilations or any(int(value) <= 0 for value in dilations):
            raise ContractError("horizon skip TCN dilations must be positive")
        if not 0 <= dropout < 1:
            raise ContractError("horizon skip TCN dropout must be in [0, 1)")
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(value) for value in dilations)
        self.input_steps = int(input_steps)
        self.receptive_field = lite_receptive_field(
            kernel_size=self.kernel_size,
            dilations=self.dilations,
        )
        if self.receptive_field < self.input_steps:
            raise ContractError(
                "horizon skip TCN receptive field is smaller than the input window"
            )
        if padding_mode not in {"explicit", "chomp"}:
            raise ContractError("horizon skip TCN padding mode is invalid")
        self.padding_mode = padding_mode
        blocks: list[nn.Module] = []
        input_channels = int(feature_count)
        for dilation in self.dilations:
            block_type = (
                CausalLiteBlock if padding_mode == "explicit" else CausalLiteBlockChomp
            )
            blocks.append(
                block_type(
                    input_channels,
                    channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    dropout_kind="element",
                )
            )
            input_channels = channels
        self.trunk = nn.ModuleList(blocks)
        self.skip_logits = nn.Parameter(torch.zeros(len(HORIZONS), len(blocks)))
        self.heads = nn.ModuleList(nn.Linear(channels, 1) for _ in HORIZONS)

    def encode_blocks(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return every causal block sequence for diagnostics and readout."""

        outputs = inputs
        representations = []
        for block in self.trunk:
            outputs = block(outputs)
            representations.append(outputs)
        return tuple(representations)

    def simplex_weights(self) -> torch.Tensor:
        return torch.softmax(self.skip_logits, dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        block_sequences = self.encode_blocks(inputs)
        last_valid = torch.stack(
            [representation[:, :, -1] for representation in block_sequences],
            dim=1,
        )
        mixed = torch.einsum("hk,bkc->bhc", self.simplex_weights(), last_valid)
        return torch.cat(
            [head(mixed[:, offset, :]) for offset, head in enumerate(self.heads)],
            dim=1,
        )

    def receipt_metadata(self) -> dict[str, object]:
        return {
            "model_family": "tcn",
            "readout": "horizon_simplex_last_valid",
            "normalization": "weight_norm",
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
            "receptive_field": self.receptive_field,
            "input_steps": self.input_steps,
            "padding_mode": self.padding_mode,
            "simplex_weights": self.simplex_weights().detach().cpu().tolist(),
        }


class DynamicHorizonSkipTCN(HorizonSkipTCN):
    """TCN with sample- and horizon-conditioned dilation-block mixtures."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        input_steps: int,
        dropout: float,
        dynamic_skip_hidden: int,
        dynamic_skip_scale: float,
        dynamic_skip_token_normalization: Literal[
            "none", "layer_norm", "shape_log_rms"
        ] = "none",
        padding_mode: Literal["explicit", "chomp"] = "explicit",
    ) -> None:
        if dynamic_skip_hidden <= 0:
            raise ContractError("dynamic skip hidden size must be positive")
        if (
            not np.isfinite(dynamic_skip_scale)
            or not 0 < dynamic_skip_scale <= 1.0
        ):
            raise ContractError("dynamic skip scale must be in (0, 1]")
        if dynamic_skip_token_normalization not in {
            "none",
            "layer_norm",
            "shape_log_rms",
        }:
            raise ContractError("dynamic skip token normalization is unsupported")
        super().__init__(
            feature_count=feature_count,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            input_steps=input_steps,
            dropout=dropout,
            padding_mode=padding_mode,
        )
        self.channels = int(channels)
        self.dynamic_skip_hidden_size = int(dynamic_skip_hidden)
        self.dynamic_skip_scale = float(dynamic_skip_scale)
        self.dynamic_skip_token_normalization = dynamic_skip_token_normalization
        self.dynamic_skip_normalizer: nn.Module = (
            nn.LayerNorm(self.channels, elementwise_affine=False)
            if dynamic_skip_token_normalization in {"layer_norm", "shape_log_rms"}
            else nn.Identity()
        )
        self.dynamic_skip_scorer_input_width = self.channels + int(
            dynamic_skip_token_normalization == "shape_log_rms"
        )
        self.dynamic_skip_hidden = nn.Linear(
            self.dynamic_skip_scorer_input_width, self.dynamic_skip_hidden_size
        )
        self.dynamic_skip_output = nn.Linear(
            self.dynamic_skip_hidden_size, len(HORIZONS)
        )
        nn.init.zeros_(self.dynamic_skip_output.weight)
        nn.init.zeros_(self.dynamic_skip_output.bias)

    def dynamic_skip_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return exactly the scorer parameters for optimizer audits."""

        return self.raw_dynamic_skip_parameters()

    def raw_dynamic_skip_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the v21-compatible raw scorer parameters."""

        return tuple(
            parameter
            for module in (self.dynamic_skip_hidden, self.dynamic_skip_output)
            for parameter in module.parameters()
        )

    def _last_valid_tokens(
        self, block_sequences: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        if len(block_sequences) != len(self.trunk):
            raise ContractError("dynamic skip block sequence count drifted")
        if any(
            sequence.ndim != 3
            or sequence.shape[1] != self.channels
            or sequence.shape[2] != self.input_steps
            for sequence in block_sequences
        ):
            raise ContractError("dynamic skip block sequence shape drifted")
        return torch.stack(
            [representation[:, :, -1] for representation in block_sequences],
            dim=1,
        )

    def dynamic_skip_weights(
        self, block_sequences: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Return ``[batch, horizon, block]`` sample-conditioned simplexes."""

        tokens = self._last_valid_tokens(block_sequences)
        dynamic_logits = self.raw_dynamic_skip_logits(tokens)
        logits = self.skip_logits.unsqueeze(0) + self.dynamic_skip_scale * torch.tanh(
            dynamic_logits
        )
        return torch.softmax(logits, dim=2)

    def raw_dynamic_skip_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, horizon, block]`` logits from the primary scorer."""

        scorer_inputs = self.dynamic_skip_scorer_inputs(tokens)
        return self.dynamic_skip_output(
            torch.tanh(self.dynamic_skip_hidden(scorer_inputs))
        ).permute(0, 2, 1)

    def normalize_dynamic_skip_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Normalize scorer tokens without changing the TCN readout tokens."""

        if tokens.ndim != 3 or tokens.shape[2] != self.channels:
            raise ContractError("dynamic skip token shape drifted")
        return self.dynamic_skip_normalizer(tokens)

    def dynamic_skip_scorer_inputs(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the causal token features consumed by the dynamic scorer."""

        normalized = self.normalize_dynamic_skip_tokens(tokens)
        if self.dynamic_skip_token_normalization != "shape_log_rms":
            return normalized
        amplitude = torch.log1p(torch.sqrt(torch.mean(tokens.square(), dim=2)))
        return torch.cat((normalized, amplitude.unsqueeze(2)), dim=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        block_sequences = self.encode_blocks(inputs)
        tokens = self._last_valid_tokens(block_sequences)
        mixed = torch.einsum(
            "bhk,bkc->bhc", self.dynamic_skip_weights(block_sequences), tokens
        )
        return torch.cat(
            [head(mixed[:, offset, :]) for offset, head in enumerate(self.heads)],
            dim=1,
        )

    def receipt_metadata(self) -> dict[str, object]:
        metadata = super().receipt_metadata()
        dynamic_parameters = self.dynamic_skip_parameters()
        metadata.update(
            {
                "readout": "horizon_stock_conditioned_dilation_block_simplex",
                "dynamic_skip_hidden": self.dynamic_skip_hidden_size,
                "dynamic_skip_scale": self.dynamic_skip_scale,
                "dynamic_skip_token_normalization": (
                    self.dynamic_skip_token_normalization
                ),
                "dynamic_skip_normalization_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.dynamic_skip_normalizer.parameters()
                ),
                "dynamic_skip_amplitude_feature": (
                    "log1p_rms"
                    if self.dynamic_skip_token_normalization == "shape_log_rms"
                    else "none"
                ),
                "dynamic_skip_scorer_input_width": (
                    self.dynamic_skip_scorer_input_width
                ),
                "dynamic_skip_parameter_count": sum(
                    parameter.numel() for parameter in dynamic_parameters
                ),
                "dynamic_skip_raw_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.raw_dynamic_skip_parameters()
                ),
                "dynamic_skip_output_weight_l2": float(
                    torch.linalg.vector_norm(
                        self.dynamic_skip_output.weight.detach().reshape(-1)
                    ).cpu()
                ),
                "dynamic_skip_output_bias_l2": float(
                    torch.linalg.vector_norm(
                        self.dynamic_skip_output.bias.detach().reshape(-1)
                    ).cpu()
                ),
                "dynamic_skip_amplitude_projection_weight_l2": (
                    float(
                        torch.linalg.vector_norm(
                            self.dynamic_skip_hidden.weight.detach()[:, -1]
                        ).cpu()
                    )
                    if self.dynamic_skip_token_normalization == "shape_log_rms"
                    else 0.0
                ),
            }
        )
        return metadata


class ShapeResidualDynamicHorizonSkipTCN(DynamicHorizonSkipTCN):
    """v21-compatible raw scorer plus a bounded channel-shape residual."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        input_steps: int,
        dropout: float,
        dynamic_skip_hidden: int,
        dynamic_skip_scale: float,
        dynamic_skip_shape_residual_scale: float,
        padding_mode: Literal["explicit", "chomp"] = "explicit",
    ) -> None:
        if (
            not np.isfinite(dynamic_skip_shape_residual_scale)
            or not 0 < dynamic_skip_shape_residual_scale <= 1.0
        ):
            raise ContractError("dynamic skip shape residual scale must be in (0, 1]")
        super().__init__(
            feature_count=feature_count,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            input_steps=input_steps,
            dropout=dropout,
            dynamic_skip_hidden=dynamic_skip_hidden,
            dynamic_skip_scale=dynamic_skip_scale,
            dynamic_skip_token_normalization="none",
            padding_mode=padding_mode,
        )
        self.dynamic_skip_shape_residual_scale = float(
            dynamic_skip_shape_residual_scale
        )
        self.dynamic_skip_shape_normalizer = nn.LayerNorm(
            self.channels, elementwise_affine=False
        )
        self.dynamic_skip_shape_hidden = nn.Linear(
            self.channels, self.dynamic_skip_hidden_size
        )
        self.dynamic_skip_shape_output = nn.Linear(
            self.dynamic_skip_hidden_size, len(HORIZONS)
        )
        nn.init.zeros_(self.dynamic_skip_shape_output.weight)
        nn.init.zeros_(self.dynamic_skip_shape_output.bias)
        self._frozen_raw_parent = False
        self._frozen_raw_parent_reference: dict[str, torch.Tensor] = {}

    def shape_residual_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return exactly the auxiliary shape scorer parameters."""

        return tuple(
            parameter
            for module in (
                self.dynamic_skip_shape_hidden,
                self.dynamic_skip_shape_output,
            )
            for parameter in module.parameters()
        )

    def load_frozen_raw_parent(
        self, parent_state: Mapping[str, torch.Tensor]
    ) -> None:
        """Load an exact v21 parent and expose only the shape branch for training."""

        candidate_state = self.state_dict()
        shape_state_names = {
            name
            for name in candidate_state
            if name.startswith("dynamic_skip_shape_")
        }
        expected_parent_names = set(candidate_state).difference(shape_state_names)
        observed_parent_names = set(parent_state)
        if observed_parent_names != expected_parent_names:
            missing = sorted(expected_parent_names.difference(observed_parent_names))
            unexpected = sorted(observed_parent_names.difference(expected_parent_names))
            raise ContractError(
                "frozen raw parent state keys drifted"
                f"; missing={missing}; unexpected={unexpected}"
            )
        for name in sorted(expected_parent_names):
            source = parent_state[name]
            target = candidate_state[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ContractError(
                    f"frozen raw parent tensor contract drifted for {name}"
                )
        incompatible = self.load_state_dict(dict(parent_state), strict=False)
        if (
            set(incompatible.missing_keys) != shape_state_names
            or incompatible.unexpected_keys
        ):
            raise ContractError("frozen raw parent state loading drifted")
        shape_parameter_ids = {
            id(parameter) for parameter in self.shape_residual_parameters()
        }
        for parameter in self.parameters():
            parameter.requires_grad_(id(parameter) in shape_parameter_ids)
        self._frozen_raw_parent_reference = {
            name: parent_state[name].detach().cpu().clone()
            for name in sorted(expected_parent_names)
        }
        self._frozen_raw_parent = True

    def frozen_parent_state_drift_max(self) -> float:
        """Return the maximum absolute drift from the loaded v21 parent state."""

        if not self._frozen_raw_parent_reference:
            raise ContractError("frozen raw parent has not been loaded")
        current = self.state_dict()
        maxima = [
            float(
                torch.max(
                    torch.abs(
                        current[name].detach().cpu()
                        - reference.to(dtype=current[name].dtype)
                    )
                )
            )
            for name, reference in self._frozen_raw_parent_reference.items()
        ]
        return max(maxima, default=0.0)

    def dynamic_skip_parameters(self) -> tuple[nn.Parameter, ...]:
        return (*self.raw_dynamic_skip_parameters(), *self.shape_residual_parameters())

    def shape_residual_inputs(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return parameter-free channel shapes for the residual scorer."""

        if tokens.ndim != 3 or tokens.shape[2] != self.channels:
            raise ContractError("dynamic skip shape residual token shape drifted")
        return self.dynamic_skip_shape_normalizer(tokens)

    def shape_residual_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, horizon, block]`` auxiliary residual logits."""

        return self.dynamic_skip_shape_output(
            torch.tanh(
                self.dynamic_skip_shape_hidden(self.shape_residual_inputs(tokens))
            )
        ).permute(0, 2, 1)

    def dynamic_skip_weights_without_shape_residual(
        self, block_sequences: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Return the v21-compatible raw-only simplex counterfactual."""

        tokens = self._last_valid_tokens(block_sequences)
        logits = self.skip_logits.unsqueeze(0) + self.dynamic_skip_scale * torch.tanh(
            self.raw_dynamic_skip_logits(tokens)
        )
        return torch.softmax(logits, dim=2)

    def forward_without_shape_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the exact frozen-parent counterfactual prediction."""

        block_sequences = self.encode_blocks(inputs)
        tokens = self._last_valid_tokens(block_sequences)
        mixed = torch.einsum(
            "bhk,bkc->bhc",
            self.dynamic_skip_weights_without_shape_residual(block_sequences),
            tokens,
        )
        return torch.cat(
            [head(mixed[:, offset, :]) for offset, head in enumerate(self.heads)],
            dim=1,
        )

    def dynamic_skip_weights(
        self, block_sequences: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        tokens = self._last_valid_tokens(block_sequences)
        combined = self.raw_dynamic_skip_logits(tokens) + (
            self.dynamic_skip_shape_residual_scale
            * torch.tanh(self.shape_residual_logits(tokens))
        )
        logits = self.skip_logits.unsqueeze(0) + self.dynamic_skip_scale * torch.tanh(
            combined
        )
        return torch.softmax(logits, dim=2)

    def receipt_metadata(self) -> dict[str, object]:
        metadata = super().receipt_metadata()
        shape_parameters = self.shape_residual_parameters()
        metadata.update(
            {
                "readout": "horizon_stock_conditioned_raw_plus_shape_residual",
                "dynamic_skip_shape_residual": True,
                "dynamic_skip_shape_residual_scale": (
                    self.dynamic_skip_shape_residual_scale
                ),
                "dynamic_skip_shape_residual_parameter_count": sum(
                    parameter.numel() for parameter in shape_parameters
                ),
                "dynamic_skip_shape_normalization_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.dynamic_skip_shape_normalizer.parameters()
                ),
                "dynamic_skip_shape_output_weight_l2": float(
                    torch.linalg.vector_norm(
                        self.dynamic_skip_shape_output.weight.detach().reshape(-1)
                    ).cpu()
                ),
                "dynamic_skip_shape_output_bias_l2": float(
                    torch.linalg.vector_norm(
                        self.dynamic_skip_shape_output.bias.detach().reshape(-1)
                    ).cpu()
                ),
                "frozen_parent": self._frozen_raw_parent,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ),
                "frozen_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if not parameter.requires_grad
                ),
            }
        )
        return metadata


class TemporalContextTCN(nn.Module):
    """Causal TCN with horizon-specific intraday and cross-day context."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        input_steps: int,
        bars_per_day: int,
        dropout: float,
        padding_mode: Literal["explicit", "chomp"] = "chomp",
    ) -> None:
        super().__init__()
        if feature_count <= 0 or channels <= 0 or kernel_size <= 1:
            raise ContractError("temporal context TCN dimensions are invalid")
        if not dilations or any(int(value) <= 0 for value in dilations):
            raise ContractError("temporal context TCN dilations must be positive")
        if not 0 <= dropout < 1:
            raise ContractError("temporal context TCN dropout must be in [0, 1)")
        if bars_per_day <= 0:
            raise ContractError("temporal context bars per day must be positive")
        if input_steps <= 0 or input_steps % bars_per_day != 0:
            raise ContractError(
                "temporal context input steps must be divisible by bars per day"
            )
        if padding_mode not in {"explicit", "chomp"}:
            raise ContractError("temporal context TCN padding mode is invalid")
        self.kernel_size = int(kernel_size)
        self.channels = int(channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.input_steps = int(input_steps)
        self.bars_per_day = int(bars_per_day)
        self.day_count = self.input_steps // self.bars_per_day
        self.padding_mode = padding_mode
        self.receptive_field = lite_receptive_field(
            kernel_size=self.kernel_size,
            dilations=self.dilations,
        )
        if self.receptive_field < self.input_steps:
            raise ContractError(
                "temporal context TCN receptive field is smaller than the input window"
            )

        blocks: list[nn.Module] = []
        input_channels = int(feature_count)
        for dilation in self.dilations:
            block_type = (
                CausalLiteBlock if padding_mode == "explicit" else CausalLiteBlockChomp
            )
            blocks.append(
                block_type(
                    input_channels,
                    channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    dropout_kind="element",
                )
            )
            input_channels = channels
        self.trunk = nn.ModuleList(blocks)
        self.day_logits = nn.Parameter(torch.zeros(len(HORIZONS), self.day_count))
        self.intraday_logits = nn.Parameter(
            torch.zeros(len(HORIZONS), self.bars_per_day)
        )
        self.heads = nn.ModuleList(nn.Linear(2 * channels, 1) for _ in HORIZONS)

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for block in self.trunk:
            outputs = block(outputs)
        return outputs

    def day_weights(self) -> torch.Tensor:
        return torch.softmax(self.day_logits, dim=1)

    def intraday_weights(self) -> torch.Tensor:
        return torch.softmax(self.intraday_logits, dim=1)

    def temporal_adapter_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return exactly the parameters controlling temporal mixtures."""

        return self.day_logits, self.intraday_logits

    def readout_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        context = self.temporal_context_sequence(sequence)
        return torch.cat(
            [head(context[:, offset, :]) for offset, head in enumerate(self.heads)],
            dim=1,
        )

    def temporal_context_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return the horizon-specific representation before the linear heads."""

        if sequence.ndim != 3 or sequence.shape[2] != self.input_steps:
            raise ContractError(
                "temporal context hidden sequence does not match input steps"
            )
        batch_size, channels, _ = sequence.shape
        by_day = sequence.reshape(
            batch_size,
            channels,
            self.day_count,
            self.bars_per_day,
        )
        daily_states = by_day.mean(dim=3)
        cross_day = torch.einsum("hd,bcd->bhc", self.day_weights(), daily_states)
        last_day = by_day[:, :, -1, :]
        intraday = torch.einsum("ht,bct->bhc", self.intraday_weights(), last_day)
        context = torch.cat([cross_day, intraday], dim=2)
        return context

    def forward(
        self, inputs: torch.Tensor, market_context: torch.Tensor | None = None
    ) -> torch.Tensor:
        if market_context is not None:
            raise ContractError(
                "stock-only temporal context TCN rejects market context"
            )
        return self.readout_sequence(self.encode_sequence(inputs))

    def receipt_metadata(self) -> dict[str, object]:
        return {
            "model_family": "tcn",
            "readout": "horizon_dual_scale_full_sequence",
            "normalization": "weight_norm",
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
            "receptive_field": self.receptive_field,
            "input_steps": self.input_steps,
            "bars_per_day": self.bars_per_day,
            "day_count": self.day_count,
            "padding_mode": self.padding_mode,
            "day_weights": self.day_weights().detach().cpu().tolist(),
            "intraday_weights": self.intraday_weights().detach().cpu().tolist(),
        }


class DynamicTemporalContextTCN(TemporalContextTCN):
    """TCN with bounded, stock-conditioned day and intraday attention."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        input_steps: int,
        bars_per_day: int,
        dropout: float,
        dynamic_attention_hidden: int,
        dynamic_attention_scale: float,
        padding_mode: Literal["explicit", "chomp"] = "chomp",
    ) -> None:
        if dynamic_attention_hidden <= 0:
            raise ContractError("dynamic attention hidden size must be positive")
        if (
            not np.isfinite(dynamic_attention_scale)
            or not 0 < dynamic_attention_scale <= 1.0
        ):
            raise ContractError("dynamic attention scale must be in (0, 1]")
        super().__init__(
            feature_count=feature_count,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            input_steps=input_steps,
            bars_per_day=bars_per_day,
            dropout=dropout,
            padding_mode=padding_mode,
        )
        self.dynamic_attention_hidden_size = int(dynamic_attention_hidden)
        self.dynamic_attention_scale = float(dynamic_attention_scale)
        self.day_attention_hidden = nn.Linear(
            self.channels, self.dynamic_attention_hidden_size
        )
        self.day_attention_output = nn.Linear(
            self.dynamic_attention_hidden_size, len(HORIZONS)
        )
        self.intraday_attention_hidden = nn.Linear(
            self.channels, self.dynamic_attention_hidden_size
        )
        self.intraday_attention_output = nn.Linear(
            self.dynamic_attention_hidden_size, len(HORIZONS)
        )
        nn.init.zeros_(self.day_attention_output.weight)
        nn.init.zeros_(self.day_attention_output.bias)
        nn.init.zeros_(self.intraday_attention_output.weight)
        nn.init.zeros_(self.intraday_attention_output.bias)

    def dynamic_attention_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return exactly the parameters controlling sample-conditioned weights."""

        return tuple(
            parameter
            for module in (
                self.day_attention_hidden,
                self.day_attention_output,
                self.intraday_attention_hidden,
                self.intraday_attention_output,
            )
            for parameter in module.parameters()
        )

    def _temporal_tokens(
        self, sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or sequence.shape[2] != self.input_steps:
            raise ContractError(
                "dynamic temporal context hidden sequence does not match input steps"
            )
        batch_size, channels, _ = sequence.shape
        if channels != self.channels:
            raise ContractError(
                "dynamic temporal context hidden channels do not match the model"
            )
        by_day = sequence.reshape(
            batch_size,
            channels,
            self.day_count,
            self.bars_per_day,
        )
        daily_tokens = by_day.mean(dim=3).transpose(1, 2)
        intraday_tokens = by_day[:, :, -1, :].transpose(1, 2)
        return daily_tokens, intraday_tokens

    def dynamic_weights(
        self, sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sample- and horizon-specific temporal simplex weights."""

        daily_tokens, intraday_tokens = self._temporal_tokens(sequence)
        day_dynamic = self.day_attention_output(
            torch.tanh(self.day_attention_hidden(daily_tokens))
        ).permute(0, 2, 1)
        intraday_dynamic = self.intraday_attention_output(
            torch.tanh(self.intraday_attention_hidden(intraday_tokens))
        ).permute(0, 2, 1)
        day_logits = self.day_logits.unsqueeze(0) + self.dynamic_attention_scale * (
            torch.tanh(day_dynamic)
        )
        intraday_logits = self.intraday_logits.unsqueeze(
            0
        ) + self.dynamic_attention_scale * torch.tanh(intraday_dynamic)
        return torch.softmax(day_logits, dim=2), torch.softmax(
            intraday_logits, dim=2
        )

    def temporal_context_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        daily_tokens, intraday_tokens = self._temporal_tokens(sequence)
        day_weights, intraday_weights = self.dynamic_weights(sequence)
        cross_day = torch.einsum("bhd,bdc->bhc", day_weights, daily_tokens)
        intraday = torch.einsum(
            "bht,btc->bhc", intraday_weights, intraday_tokens
        )
        return torch.cat([cross_day, intraday], dim=2)

    def receipt_metadata(self) -> dict[str, object]:
        metadata = super().receipt_metadata()
        dynamic_parameters = self.dynamic_attention_parameters()
        output_parameters = (
            self.day_attention_output.weight,
            self.day_attention_output.bias,
            self.intraday_attention_output.weight,
            self.intraday_attention_output.bias,
        )
        output_weights = (
            self.day_attention_output.weight,
            self.intraday_attention_output.weight,
        )
        output_biases = (
            self.day_attention_output.bias,
            self.intraday_attention_output.bias,
        )
        metadata.update(
            {
                "readout": "horizon_dual_scale_stock_conditioned_attention",
                "dynamic_attention_hidden": self.dynamic_attention_hidden_size,
                "dynamic_attention_scale": self.dynamic_attention_scale,
                "dynamic_attention_parameter_count": sum(
                    parameter.numel() for parameter in dynamic_parameters
                ),
                "dynamic_attention_output_l2": float(
                    torch.linalg.vector_norm(
                        torch.cat(
                            [
                                parameter.detach().reshape(-1)
                                for parameter in output_parameters
                            ]
                        )
                    ).cpu()
                ),
                "dynamic_attention_output_weight_l2": float(
                    torch.linalg.vector_norm(
                        torch.cat(
                            [
                                parameter.detach().reshape(-1)
                                for parameter in output_weights
                            ]
                        )
                    ).cpu()
                ),
                "dynamic_attention_output_bias_l2": float(
                    torch.linalg.vector_norm(
                        torch.cat(
                            [
                                parameter.detach().reshape(-1)
                                for parameter in output_biases
                            ]
                        )
                    ).cpu()
                ),
            }
        )
        return metadata


class MarketConditionedTemporalContextTCN(TemporalContextTCN):
    """Condition a causal stock TCN on a shared, bounded PIT market gate."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        input_steps: int,
        bars_per_day: int,
        dropout: float,
        market_context_dim: int,
        market_context_hidden: int,
        market_gate_scale: float,
        padding_mode: Literal["explicit", "chomp"] = "chomp",
    ) -> None:
        if market_context_dim <= 0 or market_context_hidden <= 0:
            raise ContractError("market context dimensions must be positive")
        if not np.isfinite(market_gate_scale) or not 0 < market_gate_scale <= 0.5:
            raise ContractError("market gate scale must be in (0, 0.5]")
        super().__init__(
            feature_count=feature_count,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            input_steps=input_steps,
            bars_per_day=bars_per_day,
            dropout=dropout,
            padding_mode=padding_mode,
        )
        self.market_context_dim = int(market_context_dim)
        self.market_context_hidden_size = int(market_context_hidden)
        self.market_gate_scale = float(market_gate_scale)
        self.market_gate_hidden = nn.Linear(
            self.market_context_dim, self.market_context_hidden_size
        )
        self.market_gate_output = nn.Linear(
            self.market_context_hidden_size, 2 * self.channels
        )
        nn.init.zeros_(self.market_gate_output.weight)
        nn.init.zeros_(self.market_gate_output.bias)

    def market_gate(self, market_context: torch.Tensor) -> torch.Tensor:
        if (
            market_context.ndim != 2
            or market_context.shape[1] != self.market_context_dim
        ):
            raise ContractError(
                "market context shape does not match the model contract"
            )
        if not bool(torch.isfinite(market_context).all()):
            raise ContractError("market context contains non-finite values")
        hidden = torch.tanh(self.market_gate_hidden(market_context))
        raw_gate = self.market_gate_output(hidden)
        return 1.0 + self.market_gate_scale * torch.tanh(raw_gate)

    def forward(
        self, inputs: torch.Tensor, market_context: torch.Tensor | None = None
    ) -> torch.Tensor:
        if market_context is None:
            raise ContractError("market-conditioned TCN requires market context")
        context = self.temporal_context_sequence(self.encode_sequence(inputs))
        gate = self.market_gate(market_context).unsqueeze(1)
        conditioned = context * gate
        return torch.cat(
            [head(conditioned[:, offset, :]) for offset, head in enumerate(self.heads)],
            dim=1,
        )

    def receipt_metadata(self) -> dict[str, object]:
        metadata = super().receipt_metadata()
        output_parameters = (
            self.market_gate_output.weight,
            self.market_gate_output.bias,
        )
        metadata.update(
            {
                "readout": "horizon_dual_scale_pit_market_conditioned",
                "market_context_dim": self.market_context_dim,
                "market_context_hidden": self.market_context_hidden_size,
                "market_gate_scale": self.market_gate_scale,
                "market_gate_min": 1.0 - self.market_gate_scale,
                "market_gate_max": 1.0 + self.market_gate_scale,
                "market_gate_parameter_count": sum(
                    parameter.numel()
                    for parameter in (
                        *self.market_gate_hidden.parameters(),
                        *self.market_gate_output.parameters(),
                    )
                ),
                "market_gate_output_l2": float(
                    torch.linalg.vector_norm(
                        torch.cat(
                            [
                                parameter.detach().reshape(-1)
                                for parameter in output_parameters
                            ]
                        )
                    ).cpu()
                ),
            }
        )
        return metadata


class SignedTemporalContextTCN(TemporalContextTCN):
    """Parameter-matched temporal context with unconstrained signed weights."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        input_steps: int,
        bars_per_day: int,
        dropout: float,
        padding_mode: Literal["explicit", "chomp"] = "chomp",
    ) -> None:
        super().__init__(
            feature_count=feature_count,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            input_steps=input_steps,
            bars_per_day=bars_per_day,
            dropout=dropout,
            padding_mode=padding_mode,
        )
        del self.day_logits
        del self.intraday_logits
        self.day_adapter = nn.Linear(self.day_count, len(HORIZONS), bias=False)
        self.intraday_adapter = nn.Linear(self.bars_per_day, len(HORIZONS), bias=False)
        nn.init.constant_(self.day_adapter.weight, 1.0 / self.day_count)
        nn.init.constant_(self.intraday_adapter.weight, 1.0 / self.bars_per_day)

    def day_weights(self) -> torch.Tensor:
        return self.day_adapter.weight

    def intraday_weights(self) -> torch.Tensor:
        return self.intraday_adapter.weight

    def receipt_metadata(self) -> dict[str, object]:
        metadata = super().receipt_metadata()
        day = self.day_weights().detach().cpu()
        intraday = self.intraday_weights().detach().cpu()
        metadata.update(
            {
                "readout": "horizon_dual_scale_signed_adapter",
                "signed_temporal_weights": True,
                "day_negative_weight_count": day.lt(0).sum(dim=1).tolist(),
                "intraday_negative_weight_count": (intraday.lt(0).sum(dim=1).tolist()),
                "day_weight_sum": day.sum(dim=1).tolist(),
                "intraday_weight_sum": intraday.sum(dim=1).tolist(),
            }
        )
        return metadata


class StabilizedResidualTemporalContextTCN(TemporalContextTCN):
    """Parameter-matched simplex readout plus a bounded zero-sum signed residual."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        input_steps: int,
        bars_per_day: int,
        dropout: float,
        residual_scale: float,
        padding_mode: Literal["explicit", "chomp"] = "chomp",
    ) -> None:
        if not np.isfinite(residual_scale) or not 0 < residual_scale <= 0.5:
            raise ContractError(
                "stabilized temporal residual scale must be in (0, 0.5]"
            )
        super().__init__(
            feature_count=feature_count,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            input_steps=input_steps,
            bars_per_day=bars_per_day,
            dropout=dropout,
            padding_mode=padding_mode,
        )
        self.residual_scale = float(residual_scale)

    def _stabilized_weights(self, logits: torch.Tensor) -> torch.Tensor:
        simplex = torch.softmax(logits, dim=1)
        bounded = torch.tanh(logits)
        centered = bounded - bounded.mean(dim=1, keepdim=True)
        return simplex + self.residual_scale * centered

    def day_weights(self) -> torch.Tensor:
        return self._stabilized_weights(self.day_logits)

    def intraday_weights(self) -> torch.Tensor:
        return self._stabilized_weights(self.intraday_logits)

    def receipt_metadata(self) -> dict[str, object]:
        metadata = super().receipt_metadata()
        day = self.day_weights().detach().cpu()
        intraday = self.intraday_weights().detach().cpu()
        day_simplex = torch.softmax(self.day_logits.detach().cpu(), dim=1)
        intraday_simplex = torch.softmax(self.intraday_logits.detach().cpu(), dim=1)
        day_residual = day - day_simplex
        intraday_residual = intraday - intraday_simplex
        metadata.update(
            {
                "readout": "horizon_dual_scale_stabilized_signed_residual",
                "residual_scale": self.residual_scale,
                "signed_residual_bounded": True,
                "day_negative_weight_count": day.lt(0).sum(dim=1).tolist(),
                "intraday_negative_weight_count": (intraday.lt(0).sum(dim=1).tolist()),
                "day_weight_sum": day.sum(dim=1).tolist(),
                "intraday_weight_sum": intraday.sum(dim=1).tolist(),
                "day_residual_l2": day_residual.norm(dim=1).tolist(),
                "intraday_residual_l2": intraday_residual.norm(dim=1).tolist(),
            }
        )
        return metadata


class DecoupledResidualTemporalContextTCN(TemporalContextTCN):
    """Learn a normal simplex base and an independent bounded signed residual."""

    def __init__(
        self,
        *,
        feature_count: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        input_steps: int,
        bars_per_day: int,
        dropout: float,
        residual_scale: float,
        padding_mode: Literal["explicit", "chomp"] = "chomp",
    ) -> None:
        if not np.isfinite(residual_scale) or not 0 < residual_scale <= 0.5:
            raise ContractError("decoupled temporal residual scale must be in (0, 0.5]")
        super().__init__(
            feature_count=feature_count,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            input_steps=input_steps,
            bars_per_day=bars_per_day,
            dropout=dropout,
            padding_mode=padding_mode,
        )
        self.residual_scale = float(residual_scale)
        self.day_residual_logits = nn.Parameter(
            torch.zeros(len(HORIZONS), self.day_count)
        )
        self.intraday_residual_logits = nn.Parameter(
            torch.zeros(len(HORIZONS), self.bars_per_day)
        )

    def residual_adapter_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return only the independently optimized signed residual parameters."""

        return self.day_residual_logits, self.intraday_residual_logits

    def day_simplex_weights(self) -> torch.Tensor:
        return torch.softmax(self.day_logits, dim=1)

    def intraday_simplex_weights(self) -> torch.Tensor:
        return torch.softmax(self.intraday_logits, dim=1)

    def _bounded_residual(self, logits: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(logits)
        centered = bounded - bounded.mean(dim=1, keepdim=True)
        return self.residual_scale * centered

    def day_residual(self) -> torch.Tensor:
        return self._bounded_residual(self.day_residual_logits)

    def intraday_residual(self) -> torch.Tensor:
        return self._bounded_residual(self.intraday_residual_logits)

    def day_weights(self) -> torch.Tensor:
        return self.day_simplex_weights() + self.day_residual()

    def intraday_weights(self) -> torch.Tensor:
        return self.intraday_simplex_weights() + self.intraday_residual()

    def receipt_metadata(self) -> dict[str, object]:
        metadata = super().receipt_metadata()
        day = self.day_weights().detach().cpu()
        intraday = self.intraday_weights().detach().cpu()
        day_simplex = self.day_simplex_weights().detach().cpu()
        intraday_simplex = self.intraday_simplex_weights().detach().cpu()
        day_residual = self.day_residual().detach().cpu()
        intraday_residual = self.intraday_residual().detach().cpu()
        base_parameters = self.temporal_adapter_parameters()
        residual_parameters = self.residual_adapter_parameters()
        metadata.update(
            {
                "readout": "horizon_dual_scale_decoupled_signed_residual",
                "residual_scale": self.residual_scale,
                "signed_residual_bounded": True,
                "base_temporal_parameter_count": sum(
                    parameter.numel() for parameter in base_parameters
                ),
                "residual_parameter_count": sum(
                    parameter.numel() for parameter in residual_parameters
                ),
                "day_simplex_weights": day_simplex.tolist(),
                "intraday_simplex_weights": intraday_simplex.tolist(),
                "day_negative_weight_count": day.lt(0).sum(dim=1).tolist(),
                "intraday_negative_weight_count": (intraday.lt(0).sum(dim=1).tolist()),
                "day_weight_sum": day.sum(dim=1).tolist(),
                "intraday_weight_sum": intraday.sum(dim=1).tolist(),
                "day_residual_l2": day_residual.norm(dim=1).tolist(),
                "intraday_residual_l2": intraday_residual.norm(dim=1).tolist(),
            }
        )
        return metadata


@dataclass(frozen=True)
class LayerProbeResult:
    """Validation metrics and train-derived coefficients for an auditable probe."""

    metrics: pd.DataFrame
    daily_rankic: pd.DataFrame
    coefficients: dict[tuple[int, int, int], np.ndarray]


@dataclass(frozen=True)
class ProbeCheckpointEvidence:
    model_family: str
    config_identity: str
    data_identity: str
    fold_identity: str
    checkpoint_identity: str
    checkpoint_path: Path
    config_path: Path


@dataclass(frozen=True)
class HorizonSkipTrigger:
    status: str
    selected_block: int | None
    model_family: str | None
    horizon: int | None
    mean_improvement: float
    positive_fold_count: int
    ci_low: float
    ci_high: float


def _validate_probe_arrays(
    representations: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    dates: Sequence[object] | np.ndarray,
    *,
    stage: str,
) -> None:
    if representations.ndim != 3:
        raise ContractError(
            f"{stage} block representations must be [sample, block, channel]"
        )
    if targets.shape != masks.shape or targets.ndim != 2 or targets.shape[1] != 4:
        raise ContractError(f"{stage} probe targets and masks must have four horizons")
    if len(representations) != len(targets) or len(dates) != len(targets):
        raise ContractError(f"{stage} probe inputs have inconsistent sample counts")
    if not np.isfinite(representations).all():
        raise ContractError(f"{stage} probe representations contain non-finite values")


def run_layer_probes(
    train_representations: np.ndarray,
    train_targets: np.ndarray,
    train_masks: np.ndarray,
    train_dates: Sequence[object] | np.ndarray,
    validation_representations: np.ndarray,
    validation_targets: np.ndarray,
    validation_masks: np.ndarray,
    validation_dates: Sequence[object] | np.ndarray,
    *,
    fold: int,
    checkpoint: ProbeCheckpointEvidence,
    expected_identities: Mapping[str, str],
    allow_missing_checkpoint: bool = False,
) -> LayerProbeResult:
    """Fit only on train rows and evaluate only on ordinary validation rows."""

    validate_probe_checkpoint(
        checkpoint,
        expected_identities,
        fold=fold,
        allow_missing_checkpoint=allow_missing_checkpoint,
    )
    model_family = checkpoint.model_family
    checkpoint_identity = checkpoint.checkpoint_identity
    train_blocks = np.asarray(train_representations, dtype="float64")
    validation_blocks = np.asarray(validation_representations, dtype="float64")
    train_y = np.asarray(train_targets, dtype="float64")
    validation_y = np.asarray(validation_targets, dtype="float64")
    train_valid = np.asarray(train_masks, dtype=bool)
    validation_valid = np.asarray(validation_masks, dtype=bool)
    _validate_probe_arrays(
        train_blocks, train_y, train_valid, train_dates, stage="train"
    )
    _validate_probe_arrays(
        validation_blocks,
        validation_y,
        validation_valid,
        validation_dates,
        stage="validation",
    )
    if train_blocks.shape[1:] != validation_blocks.shape[1:]:
        raise ContractError("train and validation probe representation shapes differ")
    metrics = []
    daily_rows = []
    coefficients: dict[tuple[int, int, int], np.ndarray] = {}
    dates = np.asarray(validation_dates).astype(str)
    for block in range(train_blocks.shape[1]):
        for column, horizon in enumerate(HORIZONS):
            train_mask = train_valid[:, column] & np.isfinite(train_y[:, column])
            validation_mask = validation_valid[:, column] & np.isfinite(
                validation_y[:, column]
            )
            if int(train_mask.sum()) < 2:
                raise ContractError("a layer probe requires at least two train labels")
            estimator = Ridge(alpha=1.0)
            estimator.fit(
                train_blocks[train_mask, block, :], train_y[train_mask, column]
            )
            coefficients[(int(fold), int(horizon), block)] = np.concatenate(
                [
                    np.asarray(estimator.coef_, dtype="float64"),
                    [float(estimator.intercept_)],
                ]
            )
            scores = estimator.predict(validation_blocks[:, block, :])
            observed_daily = []
            for signal_date in sorted(set(dates[validation_mask])):
                group = validation_mask & dates.__eq__(signal_date)
                group_scores = scores[group]
                group_targets = validation_y[group, column]
                if (
                    len(group_scores) < 2
                    or np.unique(group_scores).size < 2
                    or np.unique(group_targets).size < 2
                ):
                    continue
                rankic = float(spearmanr(group_scores, group_targets).statistic)
                if np.isfinite(rankic):
                    observed_daily.append(rankic)
                    daily_rows.append(
                        {
                            "model_family": model_family,
                            "fold": int(fold),
                            "horizon": int(horizon),
                            "block": block,
                            "signal_date": str(signal_date),
                            "rankic": rankic,
                        }
                    )
            metrics.append(
                {
                    "model_family": model_family,
                    "fold": int(fold),
                    "horizon": int(horizon),
                    "block": block,
                    "rankic": float(np.mean(observed_daily))
                    if observed_daily
                    else float("nan"),
                    "valid_date_count": len(observed_daily),
                    "fit_sample_count": int(train_mask.sum()),
                    "checkpoint_identity": checkpoint_identity,
                }
            )
    return LayerProbeResult(
        metrics=pd.DataFrame(metrics),
        daily_rankic=pd.DataFrame(daily_rows),
        coefficients=coefficients,
    )


def validate_probe_checkpoint(
    checkpoint: ProbeCheckpointEvidence,
    expected_identities: Mapping[str, str],
    *,
    fold: int,
    allow_missing_checkpoint: bool = False,
) -> None:
    """Bind a probe to the frozen config, data, fold, and checkpoint bytes."""

    family_keys = {
        "tcn-lite-16": "lite",
        "bai-tcn-16": "bai",
    }
    if checkpoint.model_family not in family_keys:
        raise ContractError("layer probes accept only the two frozen v9 controls")
    family_key = family_keys[checkpoint.model_family]
    config_key = f"{family_key}_config"
    checkpoint_key = f"{family_key}_checkpoint_fold_{int(fold)}"
    required = {"data", "fold_manifest", config_key, checkpoint_key}
    if missing := sorted(required.difference(expected_identities)):
        raise ContractError(
            f"layer-probe expected identities missing keys: {', '.join(missing)}"
        )
    observed = {
        "data": checkpoint.data_identity,
        "fold_manifest": checkpoint.fold_identity,
        config_key: checkpoint.config_identity,
        checkpoint_key: checkpoint.checkpoint_identity,
    }
    if any(not _SHA256.fullmatch(str(value)) for value in observed.values()):
        raise ContractError("layer-probe identities must be SHA-256 digests")
    if any(observed[name] != expected_identities[name] for name in required):
        raise ContractError("layer-probe checkpoint contract identity drift detected")
    if not checkpoint.checkpoint_path.is_file() and not allow_missing_checkpoint:
        raise ContractError("layer-probe checkpoint file is unavailable")
    if not checkpoint.config_path.is_file():
        raise ContractError("layer-probe config file is unavailable")
    actual_config_identity = hashlib.sha256(
        checkpoint.config_path.read_bytes()
    ).hexdigest()
    if actual_config_identity != checkpoint.config_identity:
        raise ContractError("layer-probe config bytes do not match the receipt")
    if checkpoint.checkpoint_path.is_file():
        actual_checkpoint_identity = checkpoint_state_identity(
            checkpoint.checkpoint_path
        )
        if actual_checkpoint_identity != checkpoint.checkpoint_identity:
            raise ContractError(
                "layer-probe checkpoint state does not match the receipt"
            )


def checkpoint_mapping_identity(state: Mapping[str, object]) -> str:
    """Hash tensor content independently of a torch.save container's metadata."""

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise ContractError(
                "layer-probe checkpoint state must contain only tensors"
            )
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="int64").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_state_identity(path: Path) -> str:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ContractError("layer-probe checkpoint cannot be fingerprinted") from exc
    if not isinstance(state, Mapping):
        raise ContractError("layer-probe checkpoint is not a state mapping")
    return checkpoint_mapping_identity(cast(Mapping[str, object], state))


def evaluate_horizon_skip_trigger(
    daily_rankic: pd.DataFrame,
    *,
    seed: int,
    candidate_model_family: str = "tcn-lite-16",
) -> HorizonSkipTrigger:
    """Apply the frozen effect, fold-consistency, and paired-bootstrap gate."""

    required = {"model_family", "fold", "horizon", "block", "signal_date", "rankic"}
    if missing := sorted(required.difference(daily_rankic.columns)):
        raise ContractError(
            f"layer-probe daily evidence missing columns: {', '.join(missing)}"
        )
    if daily_rankic.empty or daily_rankic[list(required)].isna().any().any():
        raise ContractError("layer-probe daily evidence is empty or incomplete")
    if daily_rankic.duplicated(
        ["model_family", "fold", "horizon", "block", "signal_date"]
    ).any():
        raise ContractError("layer-probe daily evidence contains duplicate units")
    candidate_evidence = daily_rankic.loc[
        daily_rankic["model_family"].astype(str).eq(candidate_model_family)
    ].copy()
    final_blocks = {
        str(model): int(rows["block"].max())
        for model, rows in candidate_evidence.groupby("model_family", observed=True)
    }
    rng = np.random.default_rng(seed)
    candidates: list[HorizonSkipTrigger] = []
    for keys, group in candidate_evidence.groupby(
        ["model_family", "horizon", "block"], observed=True
    ):
        model_value, horizon_value, block_value = keys
        model_family = str(model_value)
        horizon = int(cast(int, horizon_value))
        block = int(cast(int, block_value))
        final_block = final_blocks[model_family]
        if block == final_block:
            continue
        final_mask = (
            candidate_evidence["model_family"].astype(str).eq(model_family)
            & candidate_evidence["horizon"].eq(horizon)
            & candidate_evidence["block"].eq(final_block)
        )
        final = candidate_evidence[final_mask][["fold", "signal_date", "rankic"]].copy()
        final = final.rename(columns={"rankic": "final_rankic"})
        paired = group[["fold", "signal_date", "rankic"]].merge(
            final,
            on=["fold", "signal_date"],
            validate="one_to_one",
        )
        if paired.empty:
            continue
        paired = paired.sort_values(["fold", "signal_date"], kind="mergesort")
        deltas = (paired["rankic"] - paired["final_rankic"]).to_numpy(dtype="float64")
        draws = _block_bootstrap_means(deltas, rng)
        low, high = np.quantile(draws, [0.025, 0.975]).tolist()
        fold_deltas = (
            paired.assign(delta=deltas).groupby("fold", observed=True)["delta"].mean()
        )
        mean_improvement = float(deltas.mean())
        positive_folds = int(fold_deltas.gt(0).sum())
        applicable = mean_improvement >= 0.002 and positive_folds >= 3 and low > 0
        candidates.append(
            HorizonSkipTrigger(
                status=(
                    "horizon_skip_applicable"
                    if applicable
                    else "horizon_skip_not_applicable"
                ),
                selected_block=block if applicable else None,
                model_family=model_family if applicable else None,
                horizon=horizon if applicable else None,
                mean_improvement=mean_improvement,
                positive_fold_count=positive_folds,
                ci_low=float(low),
                ci_high=float(high),
            )
        )
    passing = [item for item in candidates if item.status == "horizon_skip_applicable"]
    if passing:
        return sorted(
            passing,
            key=lambda item: (
                -item.mean_improvement,
                str(item.model_family),
                int(item.horizon or 0),
                int(item.selected_block or 0),
            ),
        )[0]
    if candidates:
        strongest = sorted(candidates, key=lambda item: -item.mean_improvement)[0]
        return strongest
    return HorizonSkipTrigger(
        status="horizon_skip_not_applicable",
        selected_block=None,
        model_family=None,
        horizon=None,
        mean_improvement=float("nan"),
        positive_fold_count=0,
        ci_low=float("nan"),
        ci_high=float("nan"),
    )
