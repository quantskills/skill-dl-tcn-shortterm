"""Executable, artifact-bound diagnostic stage for governed TCN-v9 evidence."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .experiment import ContractError
from .neural import HORIZONS, _label_matrices
from .tcn import BaiTCN
from .tcn_lite import TCNLite
from .training_data import (
    LazyWindowDataset,
    build_fold_protocols,
    masked_smooth_l1,
    predict_model,
)
from .v9_gradients import diagnose_task_gradients, evaluate_pcgrad_trigger
from .v9_infra import TCNLiteChomp, evaluate_infra_gate, profile_model_step
from .v9_objective import DateGroupedBatchSampler
from .v9_protocol import V9Plan
from .v9_representation import (
    ProbeCheckpointEvidence,
    checkpoint_mapping_identity,
    evaluate_horizon_skip_trigger,
    run_layer_probes,
    validate_probe_checkpoint,
)
from .v9_statistics import audit_rankic_resolution
from .v9_training import V9TrainingRequest


@dataclass(frozen=True)
class LayerProbeInput:
    fold: int
    checkpoint: ProbeCheckpointEvidence


@dataclass(frozen=True)
class InfraDiagnosticInput:
    warmup: int = 1
    repeats: int = 3


@dataclass(frozen=True)
class RankResolutionInput:
    prediction_path: Path


@dataclass(frozen=True)
class V9DiagnosticRequest:
    training: V9TrainingRequest
    layer_probes: tuple[LayerProbeInput, ...]
    rank_resolution: RankResolutionInput
    infra: InfraDiagnosticInput


@dataclass(frozen=True)
class V9DiagnosticResult:
    upstream_receipts: dict[str, Mapping[str, object]]
    receipt_paths: dict[str, Path]


ReceiptPublisher = Callable[
    [str, str, Mapping[str, object], Mapping[str, str]], Path
]


def _load_checkpoint_model(
    probe: LayerProbeInput,
    identities: Mapping[str, str],
    *,
    training: V9TrainingRequest,
    plan: V9Plan,
    reproduced_states: dict[str, dict[int, Mapping[str, object]]],
) -> nn.Module:
    validate_probe_checkpoint(
        probe.checkpoint,
        identities,
        fold=probe.fold,
        allow_missing_checkpoint=True,
    )
    _validate_control_config(probe, plan)
    model = _build_control_model(
        probe.checkpoint.model_family,
        feature_count=int(training.features.shape[1]),
        plan=plan,
    )
    try:
        if probe.checkpoint.checkpoint_path.is_file():
            state = torch.load(
                probe.checkpoint.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        else:
            family = probe.checkpoint.model_family
            if family not in reproduced_states:
                reproduced_states[family] = _reproduce_control_states(
                    family,
                    training,
                    plan,
                )
            state = reproduced_states[family][probe.fold]
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint is not a state mapping")
        if checkpoint_mapping_identity(cast(Mapping[str, object], state)) != (
            probe.checkpoint.checkpoint_identity
        ):
            raise ContractError(
                "v9 reproduced checkpoint does not match the frozen identity"
            )
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ContractError("v9 diagnostic checkpoint cannot restore its model") from exc
    return model.eval()


def _control_config(plan: V9Plan, model_family: str) -> dict[str, object]:
    return {
        "model": model_family,
        "channels": plan.channels,
        "kernel_size": plan.kernel_size,
        "dilations": list(
            plan.dilations
            if model_family == "tcn-lite-16"
            else (1, 2, 4, 8, 16, 32, 64)
        ),
        "dropout": plan.dropout,
        "learning_rate": plan.learning_rate,
        "batch_size": plan.batch_size,
        "max_epochs": plan.max_epochs,
        "patience": plan.patience,
        "min_delta": plan.min_delta,
        "seed": 7,
        "torch_threads": plan.torch_threads,
    }


def _validate_control_config(probe: LayerProbeInput, plan: V9Plan) -> None:
    try:
        observed = json.loads(
            probe.checkpoint.config_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("v9 diagnostic control config is unreadable") from exc
    if observed != _control_config(plan, probe.checkpoint.model_family):
        raise ContractError("v9 diagnostic control config drifted")


def _build_control_model(
    model_family: str,
    *,
    feature_count: int,
    plan: V9Plan,
) -> nn.Module:
    if model_family == "tcn-lite-16":
        return TCNLite(
            feature_count=feature_count,
            channels=plan.channels,
            kernel_size=plan.kernel_size,
            dilations=plan.dilations,
            dropout=plan.dropout,
        )
    if model_family == "bai-tcn-16":
        return BaiTCN(
            feature_count=feature_count,
            channels=plan.channels,
            kernel_size=plan.kernel_size,
            dilations=(1, 2, 4, 8, 16, 32, 64),
            dropout=plan.dropout,
        )
    raise ContractError("v9 diagnostic control family is unsupported")


def _reproduce_control_states(
    model_family: str,
    training: V9TrainingRequest,
    plan: V9Plan,
) -> dict[int, Mapping[str, object]]:
    """Exactly replay a missing frozen control through the canonical trainer."""

    from .tuning import TCNTuningTrial, run_tcn_validation_sweep

    trial_id = f"reproduce-{model_family}"
    trial = TCNTuningTrial(
        trial_id=trial_id,
        channels=plan.channels,
        kernel_size=plan.kernel_size,
        dilations=(
            plan.dilations
            if model_family == "tcn-lite-16"
            else (1, 2, 4, 8, 16, 32, 64)
        ),
        dropout=plan.dropout,
        learning_rate=plan.learning_rate,
        batch_size=plan.batch_size,
        model_kind="lite" if model_family == "tcn-lite-16" else "bai",
    )
    result = run_tcn_validation_sweep(
        training.features,
        training.window_index,
        training.labels,
        training.split_manifest,
        trials=(trial,),
        seed=7,
        max_epochs=plan.max_epochs,
        patience=plan.patience,
        min_delta=plan.min_delta,
        torch_threads=plan.torch_threads,
        protocol_identities=plan.source_identities,
    )
    return {
        fold: cast(
            Mapping[str, object],
            result.best_states[f"{trial_id}-fold-{fold}"],
        )
        for fold in plan.fold_ids
    }


def _block_representations(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    trunk = getattr(model, "trunk", None)
    if not isinstance(trunk, (nn.Sequential, nn.ModuleList)):
        raise ContractError("v9 layer probe model requires a residual block trunk")
    outputs = inputs
    blocks = []
    with torch.no_grad():
        for block in trunk:
            outputs = block(outputs)
            blocks.append(outputs[:, :, -1].detach().cpu().numpy())
    if not blocks:
        raise ContractError("v9 layer probe model has no residual blocks")
    return np.stack(blocks, axis=1)


def _normalized_fold_inputs(
    features: np.ndarray,
    positions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    values = np.asarray(features[positions], dtype="float32")
    normalized = (values - mean[None, :, None]) / std[None, :, None]
    return torch.from_numpy(np.asarray(normalized, dtype="float32"))


def _rank_resolution_evidence(
    rank_input: RankResolutionInput,
    training: V9TrainingRequest,
    plan: V9Plan,
    targets: np.ndarray,
    masks: np.ndarray,
) -> pd.DataFrame:
    path = rank_input.prediction_path.resolve()
    if not path.is_file():
        raise ContractError("v9 historical RankIC predictions are unavailable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != plan.source_identities.get("rankic_predictions"):
        raise ContractError("v9 historical RankIC prediction identity drifted")
    if path.suffix.lower() in {".parquet", ".pq"}:
        predictions = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        predictions = pd.read_csv(path)
    else:
        raise ContractError("v9 historical RankIC predictions must be parquet or CSV")
    required = {
        "model",
        "fold",
        "sample_position",
        "horizon",
        "prediction",
        "config_identity",
        "checkpoint_identity",
        "stage",
        "sealed",
    }
    if missing := sorted(required.difference(predictions.columns)):
        raise ContractError(
            f"v9 historical RankIC predictions missing columns: {', '.join(missing)}"
        )
    control_model_id = plan.control_trial_id
    candidate_model_id = plan.rankic_candidate_model_id
    expected_models = {control_model_id, candidate_model_id}
    if (
        set(predictions["model"].astype(str)) != expected_models
        or set(predictions["fold"].astype(int)) != set(plan.fold_ids)
        or set(predictions["horizon"].astype(int)) != set(HORIZONS)
        or set(predictions["stage"].astype(str)) != {"validation"}
        or predictions["sealed"].astype(bool).any()
    ):
        raise ContractError("v9 historical RankIC prediction contract drifted")
    if predictions.duplicated(
        ["model", "fold", "sample_position", "horizon"]
    ).any():
        raise ContractError("v9 historical RankIC predictions contain duplicates")
    dates_by_position = training.window_index.set_index("sample_position")[
        "signal_date"
    ].astype(str)
    daily_rows = []
    for (fold, horizon), unit in predictions.groupby(
        ["fold", "horizon"], observed=True
    ):
        by_model = {
            str(model): rows.sort_values("sample_position", kind="mergesort")
            for model, rows in unit.groupby("model", observed=True)
        }
        control = by_model[control_model_id]
        candidate = by_model[candidate_model_id]
        if not np.array_equal(
            control["sample_position"].to_numpy(dtype="int64"),
            candidate["sample_position"].to_numpy(dtype="int64"),
        ):
            raise ContractError("v9 historical RankIC prediction units are unpaired")
        positions = control["sample_position"].to_numpy(dtype="int64")
        expected_positions = set(
            training.split_manifest.loc[
                training.split_manifest["fold"].eq(int(cast(Any, fold)))
                & training.split_manifest["stage"].eq("validation"),
                "sample_position",
            ].astype(int)
        )
        if set(int(value) for value in positions) != expected_positions:
            raise ContractError(
                "v9 historical RankIC predictions drifted from validation folds"
            )
        for model_id, rows in by_model.items():
            family = "lite" if model_id == control_model_id else "rankic_candidate"
            expected_config = plan.source_identities[
                "lite_config"
                if family == "lite"
                else "rankic_candidate_config"
            ]
            expected_checkpoint = plan.source_identities[
                f"{family}_checkpoint_fold_{int(cast(Any, fold))}"
            ]
            if (
                set(rows["config_identity"].astype(str)) != {expected_config}
                or set(rows["checkpoint_identity"].astype(str))
                != {expected_checkpoint}
            ):
                raise ContractError(
                    "v9 historical RankIC prediction model identity drifted"
                )
        horizon_int = int(cast(Any, horizon))
        column = HORIZONS.index(horizon_int)
        if np.any(positions < 0) or np.any(positions >= len(targets)):
            raise ContractError("v9 historical RankIC prediction position is invalid")
        dates = np.asarray(
            [dates_by_position.loc[int(position)] for position in positions]
        )
        for signal_date in sorted(set(dates)):
            date_mask = dates == signal_date
            valid = masks[positions, column] & np.isfinite(targets[positions, column])
            valid &= date_mask
            valid &= np.isfinite(control["prediction"].to_numpy(dtype="float64"))
            valid &= np.isfinite(candidate["prediction"].to_numpy(dtype="float64"))
            if int(valid.sum()) < 2:
                continue
            target = targets[positions[valid], column]
            control_score = control["prediction"].to_numpy(dtype="float64")[valid]
            candidate_score = candidate["prediction"].to_numpy(dtype="float64")[valid]
            control_rankic = pd.Series(control_score).corr(
                pd.Series(target), method="spearman"
            )
            candidate_rankic = pd.Series(candidate_score).corr(
                pd.Series(target), method="spearman"
            )
            if not np.isfinite(control_rankic) or not np.isfinite(candidate_rankic):
                continue
            daily_rows.append(
                {
                    "fold": int(cast(Any, fold)),
                    "horizon": horizon_int,
                    "signal_date": str(signal_date),
                    "control_rankic": float(control_rankic),
                    "candidate_rankic": float(candidate_rankic),
                    "valid_member_count": int(valid.sum()),
                    "label_overlap_days": horizon_int - 1,
                    "valid": True,
                    "stage": "validation",
                    "sealed": False,
                }
            )
    evidence = pd.DataFrame(daily_rows)
    if evidence.empty:
        raise ContractError("v9 diagnostic cannot derive paired RankIC evidence")
    return evidence


def _diagnose_gradients(
    models_by_fold: Mapping[int, nn.Module],
    training: V9TrainingRequest,
    protocols: tuple[object, ...],
    targets: np.ndarray,
    masks: np.ndarray,
) -> pd.DataFrame:
    dates_by_position = training.window_index.set_index("sample_position")[
        "signal_date"
    ].astype(str)
    rows = []
    for protocol_value in protocols:
        fold = int(getattr(protocol_value, "fold"))
        train_positions = cast(
            np.ndarray, getattr(protocol_value, "train_positions")
        )
        dataset = LazyWindowDataset(
            training.features,
            train_positions,
            targets,
            masks,
            cast(np.ndarray, getattr(protocol_value, "feature_mean")),
            cast(np.ndarray, getattr(protocol_value, "feature_std")),
        )
        dates = [dates_by_position.loc[int(value)] for value in train_positions]
        loader = DataLoader(
            dataset,
            batch_sampler=DateGroupedBatchSampler(
                dates,
                shuffle_dates=True,
                seed=7 + fold,
            ),
            num_workers=0,
        )
        model = models_by_fold[fold].train()
        trunk = cast(nn.Sequential | nn.ModuleList, getattr(model, "trunk"))
        block_parameters = {
            f"block-{offset}": tuple(block.parameters())
            for offset, block in enumerate(trunk)
        }
        for batch_id, (features, batch_targets, batch_masks, positions) in enumerate(loader):
            batch_dates = {
                str(dates_by_position.loc[int(position)]) for position in positions
            }
            if len(batch_dates) != 1:
                raise ContractError(
                    "v9 gradient diagnostic batch crosses signal dates"
                )
            prediction = model(features)
            safe_targets = torch.where(
                batch_masks,
                batch_targets,
                prediction.detach(),
            )
            losses = {
                int(horizon): masked_smooth_l1(
                    prediction[:, column],
                    safe_targets[:, column],
                    batch_masks[:, column],
                )
                for column, horizon in enumerate(HORIZONS)
            }
            rows.append(
                diagnose_task_gradients(
                    losses,
                    block_parameters,
                    fold=fold,
                    seed=7,
                    batch_id=batch_id,
                ).assign(signal_date=next(iter(batch_dates)))
            )
    return pd.concat(rows, ignore_index=True)


def _equivalence_and_causality(
    eager: nn.Module,
    candidate: nn.Module,
    inputs: torch.Tensor,
) -> tuple[bool, bool, bool]:
    eager_input = inputs.detach().clone().requires_grad_(True)
    candidate_input = inputs.detach().clone().requires_grad_(True)
    eager_output = eager(eager_input)
    candidate_output = candidate(candidate_input)
    numerical = bool(
        torch.allclose(eager_output, candidate_output, rtol=1e-5, atol=1e-6)
    )
    eager_gradient = torch.autograd.grad(eager_output.sum(), eager_input)[0]
    candidate_gradient = torch.autograd.grad(candidate_output.sum(), candidate_input)[0]
    gradient = bool(
        torch.allclose(eager_gradient, candidate_gradient, rtol=1e-5, atol=1e-6)
    )
    eager_encoder = getattr(eager, "encode_sequence", None)
    candidate_encoder = getattr(candidate, "encode_sequence", None)
    causal = False
    if callable(eager_encoder) and callable(candidate_encoder):
        cutoff = max(1, inputs.shape[-1] // 2)
        changed = inputs.detach().clone()
        changed[..., cutoff:] += 10_000
        with torch.no_grad():
            causal = bool(
                torch.allclose(
                    candidate_encoder(inputs.detach())[..., :cutoff],
                    candidate_encoder(changed)[..., :cutoff],
                    rtol=1e-5,
                    atol=1e-6,
                )
            )
    return numerical, gradient, causal


def run_v9_diagnostics(
    request: V9DiagnosticRequest,
    plan: V9Plan,
    *,
    publish_receipt: ReceiptPublisher,
) -> V9DiagnosticResult:
    """Derive every diagnostic from frozen artifacts and restored checkpoints."""

    training = request.training
    if request.infra.warmup < 1 or request.infra.repeats < 3:
        raise ContractError(
            "v9 formal infra profile requires warm-up and at least three repeats"
        )
    protocols = build_fold_protocols(training.features, training.split_manifest)
    protocol_by_fold = {int(protocol.fold): protocol for protocol in protocols}
    expected_probe_units = {
        (fold, family)
        for fold in plan.fold_ids
        for family in ("tcn-lite-16", "bai-tcn-16")
    }
    observed_probe_units = {
        (probe.fold, probe.checkpoint.model_family) for probe in request.layer_probes
    }
    if observed_probe_units != expected_probe_units:
        raise ContractError("v9 diagnostic requires both frozen controls in all five folds")
    for family in ("tcn-lite-16", "bai-tcn-16"):
        family_probes = [
            probe
            for probe in request.layer_probes
            if probe.checkpoint.model_family == family
        ]
        family_paths = {
            probe.checkpoint.checkpoint_path.resolve()
            for probe in family_probes
        }
        family_identities = {
            probe.checkpoint.checkpoint_identity for probe in family_probes
        }
        if (
            len(family_paths) != len(plan.fold_ids)
            or len(family_identities) != len(plan.fold_ids)
        ):
            raise ContractError(
                "v9 diagnostic requires distinct fold-specific checkpoints per control"
            )

    targets, masks = _label_matrices(training.window_index, training.labels)
    dates_by_position = training.window_index.set_index("sample_position")[
        "signal_date"
    ].astype(str)
    probe_metrics = []
    probe_daily = []
    lite_models: dict[int, nn.Module] = {}
    reproduced_states: dict[str, dict[int, Mapping[str, object]]] = {}
    for probe in request.layer_probes:
        model = _load_checkpoint_model(
            probe,
            plan.source_identities,
            training=training,
            plan=plan,
            reproduced_states=reproduced_states,
        )
        protocol = protocol_by_fold[probe.fold]
        train_inputs = _normalized_fold_inputs(
            training.features,
            protocol.train_positions,
            protocol.feature_mean,
            protocol.feature_std,
        )
        validation_inputs = _normalized_fold_inputs(
            training.features,
            protocol.validation_positions,
            protocol.feature_mean,
            protocol.feature_std,
        )
        result = run_layer_probes(
            _block_representations(model, train_inputs),
            targets[protocol.train_positions],
            masks[protocol.train_positions],
            [dates_by_position.loc[int(value)] for value in protocol.train_positions],
            _block_representations(model, validation_inputs),
            targets[protocol.validation_positions],
            masks[protocol.validation_positions],
            [dates_by_position.loc[int(value)] for value in protocol.validation_positions],
            fold=probe.fold,
            checkpoint=probe.checkpoint,
            expected_identities=plan.source_identities,
            allow_missing_checkpoint=True,
        )
        probe_metrics.append(result.metrics)
        probe_daily.append(result.daily_rankic)
        if probe.checkpoint.model_family == "tcn-lite-16":
            lite_models[probe.fold] = model
    combined_metrics = pd.concat(probe_metrics, ignore_index=True)
    combined_daily = pd.concat(probe_daily, ignore_index=True)
    horizon = evaluate_horizon_skip_trigger(combined_daily, seed=7)

    gradient_rows = _diagnose_gradients(
        lite_models,
        training,
        cast(tuple[object, ...], protocols),
        targets,
        masks,
    )
    pcgrad = evaluate_pcgrad_trigger(gradient_rows)
    rank_evidence = _rank_resolution_evidence(
        request.rank_resolution,
        training,
        plan,
        targets,
        masks,
    )
    rank_audit = audit_rankic_resolution(
        rank_evidence,
        seed=7,
        expected_folds=plan.fold_ids,
    )

    first_protocol = protocol_by_fold[0]
    first_dataset = LazyWindowDataset(
        training.features,
        first_protocol.train_positions,
        targets,
        masks,
        first_protocol.feature_mean,
        first_protocol.feature_std,
    )
    loader = DataLoader(
        first_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    wait_start = time.perf_counter()
    profile_inputs, profile_targets, profile_masks, _ = next(iter(loader))
    data_wait_seconds = time.perf_counter() - wait_start
    validation_dataset = LazyWindowDataset(
        training.features,
        first_protocol.validation_positions,
        targets,
        masks,
        first_protocol.feature_mean,
        first_protocol.feature_std,
    )
    validation_start = time.perf_counter()
    eager_model = copy.deepcopy(lite_models[0])
    candidate_model = TCNLiteChomp(
        feature_count=int(training.features.shape[1]),
        channels=plan.channels,
        kernel_size=plan.kernel_size,
        dilations=plan.dilations,
        dropout=plan.dropout,
    )
    candidate_model.load_state_dict(eager_model.state_dict(), strict=True)
    predict_model(
        eager_model,
        validation_dataset,
        batch_size=training.batch_size,
        num_workers=0,
    )
    eager_validation_seconds = time.perf_counter() - validation_start
    candidate_validation_start = time.perf_counter()
    predict_model(
        candidate_model,
        validation_dataset,
        batch_size=training.batch_size,
        num_workers=0,
    )
    candidate_validation_seconds = (
        time.perf_counter() - candidate_validation_start
    )
    numerical, gradient, causal = _equivalence_and_causality(
        eager_model,
        candidate_model,
        profile_inputs,
    )
    eager_profile = profile_model_step(
        copy.deepcopy(eager_model),
        profile_inputs,
        profile_targets,
        profile_masks,
        torch_threads=plan.torch_threads,
        learning_rate=plan.learning_rate,
        warmup=request.infra.warmup,
        repeats=request.infra.repeats,
        formal_protocol=True,
        data_wait_seconds=data_wait_seconds,
        validation_seconds=eager_validation_seconds,
    )
    candidate_profile = profile_model_step(
        copy.deepcopy(candidate_model),
        profile_inputs,
        profile_targets,
        profile_masks,
        torch_threads=plan.torch_threads,
        learning_rate=plan.learning_rate,
        warmup=request.infra.warmup,
        repeats=request.infra.repeats,
        formal_protocol=True,
        data_wait_seconds=data_wait_seconds,
        validation_seconds=candidate_validation_seconds,
    )
    infra_gate = evaluate_infra_gate(
        eager_profile.operators,
        eager_samples_per_second=eager_profile.samples_per_second,
        candidate_samples_per_second=candidate_profile.samples_per_second,
        numerically_equivalent=numerical and gradient,
        strictly_causal=causal,
    )

    global_gradients = gradient_rows.loc[gradient_rows["scope"].eq("global")]
    selected_pair = pcgrad.horizon_pair
    selected_gradients = (
        global_gradients.loc[
            global_gradients["left_horizon"].eq(selected_pair[0])
            & global_gradients["right_horizon"].eq(selected_pair[1])
        ]
        if selected_pair is not None
        else global_gradients.iloc[0:0]
    )
    evidence_by_name: dict[str, dict[str, object]] = {
        "horizon_skip": {
            "model_family": horizon.model_family,
            "mean_improvement": horizon.mean_improvement,
            "positive_fold_count": horizon.positive_fold_count,
            "ci_low": horizon.ci_low,
            "ci_high": horizon.ci_high,
            "selected_block": horizon.selected_block,
            "probe_metrics": combined_metrics.to_dict(orient="records"),
            "daily_rankic": combined_daily.to_dict(orient="records"),
        },
        "rank_objective": {
            "control_model_id": plan.control_trial_id,
            "candidate_model_id": plan.rankic_candidate_model_id,
            "historical_prediction_identity": plan.source_identities[
                "rankic_predictions"
            ],
            "minimum_paired_date_count": int(
                rank_audit.summary["paired_date_count"].min()
            ),
            "maximum_degenerate_bootstrap_rate": float(
                rank_audit.summary["degenerate_bootstrap_rate"].max()
            ),
            "maximum_minimum_detectable_effect": float(
                rank_audit.summary["minimum_detectable_effect"].max()
            ),
            "resolution_summary": rank_audit.summary.to_dict(orient="records"),
        },
        "pcgrad": {
            "conflicting_fold_count": pcgrad.conflicting_fold_count,
            "median_cosine": (
                float(selected_gradients["cosine"].median())
                if not selected_gradients.empty
                else 0.0
            ),
            "negative_batch_rate": (
                float(selected_gradients["negative_cosine"].mean())
                if not selected_gradients.empty
                else 0.0
            ),
            "gradient_diagnostics": gradient_rows.to_dict(orient="records"),
        },
        "infra": {
            "padding_self_cpu_share": infra_gate.padding_self_cpu_share,
            "throughput_gain": infra_gate.throughput_gain,
            "numerically_equivalent": numerical,
            "gradient_equivalent": gradient,
            "strictly_causal": causal,
            "hardware_identity": eager_profile.hardware_identity,
            "measurement_noise": eager_profile.measurement_noise,
            "candidate_measurement_noise": candidate_profile.measurement_noise,
            "learning_rate": eager_profile.learning_rate,
            "eager_model_step_seconds_median": (
                eager_profile.model_step_seconds_median
            ),
            "candidate_model_step_seconds_median": (
                candidate_profile.model_step_seconds_median
            ),
            "eager_samples_per_second": eager_profile.samples_per_second,
            "candidate_samples_per_second": candidate_profile.samples_per_second,
            "data_wait_seconds": data_wait_seconds,
            "validation_seconds": eager_validation_seconds,
            "candidate_validation_seconds": candidate_validation_seconds,
            "eager_complete_cycle_seconds": eager_profile.complete_cycle_seconds,
            "candidate_complete_cycle_seconds": (
                candidate_profile.complete_cycle_seconds
            ),
            "operator_profile": eager_profile.operators.to_dict(orient="records"),
            "candidate_operator_profile": candidate_profile.operators.to_dict(
                orient="records"
            ),
        },
    }
    statuses = {
        "horizon_skip": horizon.status,
        "rank_objective": rank_audit.status,
        "pcgrad": pcgrad.status,
        "infra": infra_gate.status,
    }
    receipts: dict[str, Mapping[str, object]] = {}
    paths: dict[str, Path] = {}
    for name, status in statuses.items():
        path = publish_receipt(
            name,
            status,
            evidence_by_name[name],
            plan.source_identities,
        )
        paths[name] = path
        receipts[name] = {
            "status": status,
            "receipt_path": str(path),
            "sealed_test_accessed": False,
        }
    return V9DiagnosticResult(receipts, paths)
