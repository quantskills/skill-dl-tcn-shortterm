"""Comparable wall-clock benchmark for sequence models."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

from .experiment import ContractError
from .neural import RecurrentRegressor, _label_matrices
from .optimized_tcn import (
    V40_PORTABLE_MODEL,
    build_optimized_tcn_model,
    resolve_optimized_tcn_profile,
)
from .runtime import torch_thread_scope
from .tcn import BaiTCN, validate_receptive_field
from .tcn_lite import TCNLite, lite_receptive_field
from .training_data import (
    LazyWindowDataset,
    build_fold_protocols,
    masked_smooth_l1,
)
from .tuning import (
    ValidationRankICPlan,
    build_validation_rankic_plan,
)


@dataclass(frozen=True)
class PerformanceBenchmarkResult:
    measurements: pd.DataFrame
    environment: dict[str, object]


def _fingerprint(
    features: np.ndarray, train: np.ndarray, validation: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for array in [features, train, validation]:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        for position in range(len(array)):
            digest.update(np.ascontiguousarray(array[position]).tobytes())
    return digest.hexdigest()


def _validation_rankic(
    model: nn.Module,
    validation_dataset: LazyWindowDataset,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    validation_plan: ValidationRankICPlan | None = None,
    validation_loader: DataLoader[Any] | None = None,
) -> float:
    loader = validation_loader
    if loader is None:
        loader_options: dict[str, Any] = (
            {"prefetch_factor": 2, "persistent_workers": True}
            if num_workers > 0
            else {}
        )
        loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **loader_options,
        )
    score_batches = []
    position_batches = []
    model.eval()
    with torch.no_grad():
        for features, _, _, positions in loader:
            score_batches.append(model(features.to(device)).cpu())
            position_batches.append(positions)
    scores = torch.cat(score_batches)
    positions = torch.cat(position_batches).numpy()
    plan = validation_plan or build_validation_rankic_plan(
        positions, window_index, labels
    )
    return plan.evaluate(scores.numpy(), positions).mean_daily_rankic


def benchmark_sequence_models(
    features: np.ndarray,
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    seed: int,
    seeds: Sequence[int] | None = None,
    hidden_size: int,
    tcn_channels: int,
    tcn_kernel_size: int,
    tcn_dilations: Sequence[int],
    epochs: int,
    batch_size: int,
    device: str = "cpu",
    num_workers: int = 0,
    torch_threads: int | None = None,
    include_tcn_lite: bool = False,
    tcn_lite_channels: int = 4,
    tcn_lite_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    learning_rate: float = 0.01,
    models: Sequence[str] | None = None,
    optimized_tcn_profile: str = "v40-portable",
) -> PerformanceBenchmarkResult:
    """Measure recurrent and TCN models under one fixed, auditable protocol.

    This reports observed timings without asserting that any model must achieve a
    speedup. CI therefore verifies the measurement contract, while real hardware
    conclusions remain evidence from an explicitly executed benchmark.
    """

    if device not in {"cpu", "cuda"}:
        raise ContractError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ContractError("CUDA was requested but is unavailable")
    if epochs <= 0 or batch_size <= 0:
        raise ContractError("epochs and batch_size must be positive")
    if learning_rate <= 0:
        raise ContractError("learning rate must be positive")
    if num_workers < 0:
        raise ContractError("num_workers cannot be negative")
    if len(features) != len(window_index):
        raise ContractError("features and window index sample counts must match")
    validate_receptive_field(
        input_steps=features.shape[2],
        kernel_size=tcn_kernel_size,
        dilations=tcn_dilations,
    )
    if include_tcn_lite:
        observed_lite_receptive_field = lite_receptive_field(
            kernel_size=tcn_kernel_size, dilations=tcn_lite_dilations
        )
        if observed_lite_receptive_field < features.shape[2]:
            raise ContractError(
                "TCN-lite receptive field "
                f"{observed_lite_receptive_field} is smaller than input window "
                f"{features.shape[2]}"
            )
        if tcn_lite_channels <= 0:
            raise ContractError("TCN-lite channels must be positive")
    resolved_seeds = tuple(int(value) for value in (seeds or [seed]))
    if not resolved_seeds or len(set(resolved_seeds)) != len(resolved_seeds):
        raise ContractError("benchmark seeds must be non-empty and unique")
    portable_profile = resolve_optimized_tcn_profile(
        optimized_tcn_profile,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        torch_threads=torch_threads or torch.get_num_threads(),
    )
    default_models = ["lstm", "gru", "bai-tcn"]
    available_models = [*default_models, V40_PORTABLE_MODEL]
    if include_tcn_lite:
        available_models.append("tcn-lite")
        default_models.append("tcn-lite")
    resolved_models = tuple(models or default_models)
    if not resolved_models or len(set(resolved_models)) != len(resolved_models):
        raise ContractError("benchmark models must be non-empty and unique")
    if unknown_models := sorted(set(resolved_models).difference(available_models)):
        raise ContractError(
            f"benchmark models are unavailable: {', '.join(unknown_models)}"
        )

    with torch_thread_scope(torch_threads) as effective_torch_threads:
        protocols = build_fold_protocols(features, split_manifest)
        targets, masks = _label_matrices(window_index, labels)
        resolved_device = torch.device(device)
        process = psutil.Process()
        measurements = []
        for protocol in protocols:
            fold = protocol.fold
            train_positions = protocol.train_positions
            validation_positions = protocol.validation_positions
            train_dataset = LazyWindowDataset(
                features,
                train_positions,
                targets,
                masks,
                protocol.feature_mean,
                protocol.feature_std,
            )
            validation_dataset = LazyWindowDataset(
                features,
                validation_positions,
                targets,
                masks,
                protocol.feature_mean,
                protocol.feature_std,
            )
            validation_plan = build_validation_rankic_plan(
                validation_positions, window_index, labels
            )
            validation_loader_options: dict[str, Any] = (
                {"prefetch_factor": 2, "persistent_workers": True}
                if num_workers > 0
                else {}
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                **validation_loader_options,
            )
            fingerprint = _fingerprint(features, train_positions, validation_positions)
            factories = {
                "lstm": lambda: RecurrentRegressor(
                    "lstm", features.shape[1], hidden_size
                ),
                "gru": lambda: RecurrentRegressor(
                    "gru", features.shape[1], hidden_size
                ),
                "bai-tcn": lambda: BaiTCN(
                    feature_count=features.shape[1],
                    channels=tcn_channels,
                    kernel_size=tcn_kernel_size,
                    dilations=tcn_dilations,
                    dropout=0.0,
                ),
                V40_PORTABLE_MODEL: lambda: build_optimized_tcn_model(
                    feature_count=features.shape[1],
                    input_steps=features.shape[2],
                    profile=portable_profile,
                ),
            }
            if include_tcn_lite:
                factories["tcn-lite"] = lambda: TCNLite(
                    feature_count=features.shape[1],
                    channels=tcn_lite_channels,
                    kernel_size=tcn_kernel_size,
                    dilations=tcn_lite_dilations,
                    dropout=0.0,
                )
            factories = {
                model_name: factories[model_name]
                for model_name in resolved_models
            }
            for base_seed in resolved_seeds:
                model_seed = base_seed + fold * 100
                for model_name, factory in factories.items():
                    torch.manual_seed(model_seed)
                    if device == "cuda":
                        torch.cuda.manual_seed_all(model_seed)
                        torch.cuda.reset_peak_memory_stats()
                    generator = torch.Generator().manual_seed(model_seed)
                    loader_options: dict[str, Any] = (
                        {"prefetch_factor": 2, "persistent_workers": True}
                        if num_workers > 0
                        else {}
                    )
                    loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        generator=generator,
                        num_workers=num_workers,
                        pin_memory=False,
                        **loader_options,
                    )
                    model = factory().to(resolved_device)
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=learning_rate
                    )
                    start = time.perf_counter()
                    epoch_times = []
                    data_wait_seconds = 0.0
                    model_step_seconds = 0.0
                    train_pipeline_seconds = 0.0
                    validation_seconds = 0.0
                    processed_samples = 0
                    best_rankic = float("nan")
                    time_to_best = 0.0
                    peak_ram = process.memory_info().rss
                    for _ in range(epochs):
                        epoch_start = time.perf_counter()
                        train_start = time.perf_counter()
                        model.train()
                        iterator = iter(loader)
                        while True:
                            wait_start = time.perf_counter()
                            try:
                                batch_features, batch_targets, batch_masks, _ = next(
                                    iterator
                                )
                            except StopIteration:
                                break
                            data_wait_seconds += time.perf_counter() - wait_start
                            batch_features = batch_features.to(resolved_device)
                            batch_targets = batch_targets.to(resolved_device)
                            batch_masks = batch_masks.to(resolved_device)
                            step_start = time.perf_counter()
                            optimizer.zero_grad(set_to_none=True)
                            prediction = model(batch_features)
                            loss = masked_smooth_l1(
                                prediction, batch_targets, batch_masks
                            )
                            loss.backward()
                            optimizer.step()
                            if device == "cuda":
                                torch.cuda.synchronize()
                            model_step_seconds += time.perf_counter() - step_start
                            processed_samples += int(batch_features.shape[0])
                            peak_ram = max(peak_ram, process.memory_info().rss)
                        train_pipeline_seconds += time.perf_counter() - train_start
                        validation_start = time.perf_counter()
                        rankic = _validation_rankic(
                            model,
                            validation_dataset,
                            batch_size=batch_size,
                            device=resolved_device,
                            num_workers=num_workers,
                            window_index=window_index,
                            labels=labels,
                            validation_plan=validation_plan,
                            validation_loader=validation_loader,
                        )
                        validation_seconds += time.perf_counter() - validation_start
                        elapsed = time.perf_counter() - start
                        if np.isnan(best_rankic) or (
                            not np.isnan(rankic) and rankic > best_rankic
                        ):
                            best_rankic = rankic
                            time_to_best = elapsed
                        epoch_times.append(time.perf_counter() - epoch_start)
                    total_seconds = time.perf_counter() - start
                    measurements.append(
                        {
                            "model": model_name,
                            "data_fingerprint": fingerprint,
                            "fold": fold,
                            "base_seed": base_seed,
                            "model_seed": model_seed,
                            "batch_size": batch_size,
                            "precision": "float32",
                            "loader": "lazy_window_dataset",
                            "storage": (
                                "read_only_memmap"
                                if isinstance(features, np.memmap)
                                else "ndarray"
                            ),
                            "epochs": epochs,
                            "train_sample_count": len(train_positions),
                            "validation_sample_count": len(validation_positions),
                            "parameter_count": sum(
                                parameter.numel() for parameter in model.parameters()
                            ),
                            "samples_per_second": processed_samples / total_seconds,
                            "model_step_samples_per_second": (
                                processed_samples / model_step_seconds
                            ),
                            "train_pipeline_samples_per_second": (
                                processed_samples / train_pipeline_seconds
                            ),
                            "mean_epoch_seconds": float(np.mean(epoch_times)),
                            "model_step_seconds": model_step_seconds,
                            "validation_seconds": validation_seconds,
                            "end_to_end_seconds": total_seconds,
                            "time_to_best_seconds": time_to_best,
                            "best_validation_rankic": best_rankic,
                            "peak_ram_bytes": peak_ram,
                            "peak_vram_bytes": (
                                int(torch.cuda.max_memory_allocated())
                                if device == "cuda"
                                else 0
                            ),
                            "data_wait_seconds": data_wait_seconds,
                        }
                    )
        measurement_frame = pd.DataFrame(measurements)
        ratio_index = ["fold", "base_seed"]
        end_to_end = measurement_frame.pivot(
            index=ratio_index, columns="model", values="samples_per_second"
        )
        model_step = measurement_frame.pivot(
            index=ratio_index,
            columns="model",
            values="model_step_samples_per_second",
        )

        def geomean_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
            return float(
                np.exp(np.log(frame[numerator] / frame[denominator]).mean())
            )

        speed_ratios = {}
        for candidate, prefix in [
            ("bai-tcn", "tcn"),
            ("tcn-lite", "tcn_lite"),
            (V40_PORTABLE_MODEL, "optimized_tcn"),
        ]:
            if candidate not in resolved_models:
                continue
            for baseline in ["lstm", "gru"]:
                if baseline not in resolved_models:
                    continue
                speed_ratios[f"{prefix}_over_{baseline}_geomean"] = geomean_ratio(
                    end_to_end, candidate, baseline
                )
                speed_ratios[
                    f"{prefix}_over_{baseline}_model_step_geomean"
                ] = geomean_ratio(model_step, candidate, baseline)
        environment: dict[str, object] = {
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(0)
                if device == "cuda"
                else platform.processor()
            ),
            "cpu_count": os.cpu_count(),
            "torch_threads": effective_torch_threads,
            "data_workers": num_workers,
            "prefetch_factor": 2 if num_workers > 0 else None,
            "pin_memory": False,
            "amp": False,
            "window_storage": (
                "read_only_memmap" if isinstance(features, np.memmap) else "ndarray"
            ),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "fold_count": len(protocols),
            "base_seeds": list(resolved_seeds),
            "models": list(resolved_models),
            "learning_rate": learning_rate,
            "optimized_tcn_profile": portable_profile.name,
            "observed_tcn_speed_ratios": speed_ratios,
        }
        return PerformanceBenchmarkResult(
            measurements=measurement_frame, environment=environment
        )
