"""TCN-v9 multi-seed confirmation and immutable final evidence receipt."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Mapping, cast

import numpy as np
import pandas as pd

from .experiment import ContractError
from .integrity import code_identity
from .v9_receipts import canonical_bytes, canonicalize, publish_immutable_receipt
from .v9_selection import Seed7Decision


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class V9ConfirmationDecision:
    status: str
    winner_trial_id: str | None
    blockers: tuple[str, ...]
    metrics: dict[str, float | int]
    speed_ratios: dict[str, float]
    sealed_test_accessed: bool = False


@dataclass(frozen=True)
class V9FinalContext:
    resolved_config: Mapping[str, object]
    source_identities: Mapping[str, str]
    checkpoint_identities: Mapping[str, str]
    environment: Mapping[str, object]
    upstream_receipts: Mapping[str, str]


def _geometric_median_ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    ratios = numerator.to_numpy(dtype="float64") / denominator.to_numpy(dtype="float64")
    return float(np.exp(np.median(np.log(ratios))))


def evaluate_multiseed_confirmation(
    measurements: pd.DataFrame,
    *,
    seed7_decision: Seed7Decision,
    control_trial_id: str,
    lstm_model_id: str,
    gru_model_id: str,
) -> V9ConfirmationDecision:
    """Apply the frozen 15-unit effect gate and report fair recurrent speed ratios."""

    if seed7_decision.status != "seed7_winner_admitted":
        if not measurements.empty:
            raise ContractError(
                "v9 confirmation must end directly when seed 7 is not admitted"
            )
        return V9ConfirmationDecision(
            status="stop_no_pareto_gain_v9",
            winner_trial_id=None,
            blockers=("seed7_winner_not_admitted",),
            metrics={},
            speed_ratios={},
        )
    candidate_id = seed7_decision.winner_trial_id
    if candidate_id is None or seed7_decision.confirmation_seeds != (17, 27):
        raise ContractError("multi-seed confirmation has an invalid seed-7 admission")
    required = {
        "model",
        "fold",
        "seed",
        "rankic",
        "samples_per_second",
        "model_step_samples_per_second",
        "model_step_seconds",
        "data_wait_seconds",
        "validation_seconds",
        "complete_cycle_seconds",
        "time_to_best_seconds",
        "parameter_count",
        "precision",
        "torch_threads",
        "batch_size",
        "data_identity",
        "fold_identity",
        "evaluation_identity",
        "max_epochs",
        "patience",
        "min_delta",
        "loss_identity",
        "infra_identity",
        "candidate_config_identity",
        "simplex_weights",
        "sealed_test_accessed",
    }
    if missing := sorted(required.difference(measurements.columns)):
        raise ContractError(f"v9 confirmation measurements missing columns: {', '.join(missing)}")
    if measurements.empty:
        raise ContractError("admitted v9 confirmation requires measurements")
    if measurements["sealed_test_accessed"].astype(bool).any():
        raise ContractError("v9 confirmation rejects sealed evidence")
    expected_models = {candidate_id, control_trial_id, lstm_model_id, gru_model_id}
    if set(measurements["model"].astype(str)) != expected_models:
        raise ContractError("v9 confirmation models do not match the frozen comparison set")
    if set(measurements["seed"].astype(int)) != {7, 17, 27}:
        raise ContractError("v9 confirmation requires seeds 7, 17, and 27")
    if measurements.duplicated(["model", "fold", "seed"]).any():
        raise ContractError("v9 confirmation contains duplicate model units")
    for column in [
        "precision",
        "torch_threads",
        "batch_size",
        "data_identity",
        "fold_identity",
        "evaluation_identity",
    ]:
        if measurements[column].nunique(dropna=False) != 1:
            raise ContractError(f"v9 confirmation protocol drift detected in {column}")
    numeric = [
        "rankic",
        "samples_per_second",
        "model_step_samples_per_second",
        "model_step_seconds",
        "data_wait_seconds",
        "validation_seconds",
        "complete_cycle_seconds",
        "time_to_best_seconds",
        "parameter_count",
    ]
    if not np.isfinite(measurements[numeric].to_numpy(dtype="float64")).all():
        raise ContractError("v9 confirmation contains non-finite measurements")
    if measurements[["samples_per_second", "model_step_samples_per_second", "parameter_count"]].le(0).any().any():
        raise ContractError("v9 confirmation speed and parameter values must be positive")
    if measurements[
        [
            "model_step_seconds",
            "data_wait_seconds",
            "validation_seconds",
            "complete_cycle_seconds",
            "time_to_best_seconds",
        ]
    ].lt(0).any().any():
        raise ContractError("v9 confirmation timings must be non-negative")
    expected_units = {(fold, seed) for seed in [7, 17, 27] for fold in range(5)}
    indexed: dict[str, pd.DataFrame] = {}
    for model_value, rows in measurements.groupby("model", observed=True):
        model = str(model_value)
        units = set(
            zip(rows["fold"].astype(int), rows["seed"].astype(int), strict=True)
        )
        if units != expected_units or len(rows) != 15:
            raise ContractError("each v9 confirmation model must cover 15 fold-seed units")
        if rows["parameter_count"].nunique() != 1:
            raise ContractError("v9 confirmation parameter count must be stable per model")
        for column in [
            "max_epochs",
            "patience",
            "min_delta",
            "loss_identity",
            "infra_identity",
            "candidate_config_identity",
        ]:
            if rows[column].nunique(dropna=False) != 1:
                raise ContractError(
                    f"v9 confirmation cross-seed drift detected in {column}"
                )
        indexed[model] = rows.set_index(["fold", "seed"]).sort_index()
    candidate = indexed[candidate_id]
    control = indexed[control_trial_id]
    deltas = candidate["rankic"] - control["rankic"]
    seed_improvements = deltas.groupby(level="seed").mean()
    metrics: dict[str, float | int] = {
        "unit_count": int(len(candidate)),
        "median_rankic": float(candidate["rankic"].median()),
        "positive_rate": float(candidate["rankic"].gt(0).mean()),
        "worst_fold_seed_mean_rankic": float(
            candidate["rankic"].groupby(level="fold").mean().min()
        ),
        "paired_median_improvement": float(deltas.median()),
        "minimum_seed_mean_improvement": float(seed_improvements.min()),
        "median_samples_per_second": float(candidate["samples_per_second"].median()),
        "parameter_count": int(candidate["parameter_count"].iloc[0]),
    }
    speed_ratios = {
        "vs_lstm_model_step": _geometric_median_ratio(
            candidate["model_step_samples_per_second"],
            indexed[lstm_model_id]["model_step_samples_per_second"],
        ),
        "vs_lstm_end_to_end": _geometric_median_ratio(
            candidate["samples_per_second"],
            indexed[lstm_model_id]["samples_per_second"],
        ),
        "vs_gru_model_step": _geometric_median_ratio(
            candidate["model_step_samples_per_second"],
            indexed[gru_model_id]["model_step_samples_per_second"],
        ),
        "vs_gru_end_to_end": _geometric_median_ratio(
            candidate["samples_per_second"],
            indexed[gru_model_id]["samples_per_second"],
        ),
    }
    blockers = []
    if float(metrics["median_rankic"]) < 0.09:
        blockers.append("median_rankic_below_0.09")
    if float(metrics["positive_rate"]) < 0.80:
        blockers.append("positive_rate_below_0.80")
    if float(metrics["worst_fold_seed_mean_rankic"]) < -0.01:
        blockers.append("worst_fold_below_minus_0.01")
    if float(metrics["paired_median_improvement"]) < 0.005:
        blockers.append("paired_improvement_below_0.005")
    if not seed_improvements.gt(0).all():
        blockers.append("not_all_seed_improvements_positive")
    if float(metrics["median_samples_per_second"]) < 5000:
        blockers.append("throughput_below_5000")
    return V9ConfirmationDecision(
        status=(
            "pareto_candidate_confirmed_v9"
            if not blockers
            else "stop_no_pareto_gain_v9"
        ),
        winner_trial_id=candidate_id,
        blockers=tuple(blockers),
        metrics=metrics,
        speed_ratios=speed_ratios,
    )


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if any(marker in str(key).lower() for marker in ["password", "token", "secret", "credential"]):
                return True
            if _contains_secret_key(nested):
                return True
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_key(item) for item in value)
    return False


def _validate_context(context: V9FinalContext) -> None:
    if _contains_secret_key(context.resolved_config) or _contains_secret_key(context.environment):
        raise ContractError("v9 final context contains a forbidden secret-like key")
    for identities in [context.source_identities, context.upstream_receipts]:
        if not identities or any(
            not name or not _SHA256.fullmatch(str(digest))
            for name, digest in identities.items()
        ):
            raise ContractError("v9 final context identities must be named SHA-256 digests")
    if any(
        not name or not _SHA256.fullmatch(str(digest))
        for name, digest in context.checkpoint_identities.items()
    ):
        raise ContractError("v9 final checkpoint identities must be SHA-256 digests")


def _measurement_receipt_records(measurements: pd.DataFrame) -> list[dict[str, object]]:
    """Return a stable, JSON-safe snapshot of every confirmation unit."""

    if measurements.empty:
        return []
    sorted_measurements = measurements.sort_values(
        ["model", "seed", "fold"], kind="mergesort"
    )
    normalized = sorted_measurements.astype(object).where(
        pd.notna(sorted_measurements), None
    )
    return cast(list[dict[str, object]], normalized.to_dict("records"))


def finalize_v9_run(
    measurements: pd.DataFrame,
    *,
    seed7_decision: Seed7Decision,
    context: V9FinalContext,
    output_dir: Path,
    project_root: Path,
    control_trial_id: str,
    lstm_model_id: str,
    gru_model_id: str,
) -> Path:
    """Evaluate and atomically publish the final bounded v9 result."""

    _validate_context(context)
    decision = evaluate_multiseed_confirmation(
        measurements,
        seed7_decision=seed7_decision,
        control_trial_id=control_trial_id,
        lstm_model_id=lstm_model_id,
        gru_model_id=gru_model_id,
    )
    measurement_records = _measurement_receipt_records(measurements)
    payload: dict[str, object] = {
        "schema_version": "tcn-v9-final/v1",
        "status": decision.status,
        "stop_reasons": list(decision.blockers),
        "resolved_config": context.resolved_config,
        "identities": {
            "sources": dict(sorted(context.source_identities.items())),
            "checkpoints": dict(sorted(context.checkpoint_identities.items())),
            "upstream_receipts": dict(sorted(context.upstream_receipts.items())),
            "measurements_sha256": hashlib.sha256(
                canonical_bytes(measurement_records)
            ).hexdigest(),
        },
        "code_identity": code_identity(project_root.resolve()),
        "environment": context.environment,
        "selection": {
            "seed7_status": seed7_decision.status,
            "winner_trial_id": decision.winner_trial_id,
            "confirmation_seeds": list(seed7_decision.confirmation_seeds),
            "seed7_summary": seed7_decision.summary.to_dict("records"),
        },
        "metrics": decision.metrics,
        "performance": decision.speed_ratios,
        "measurements": measurement_records,
        "sealed_test_accessed": False,
    }
    payload = cast(dict[str, object], canonicalize(payload))
    payload["receipt_id"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return publish_immutable_receipt(
        payload,
        output_dir=output_dir,
        filename="receipt.json",
        identity_label="v9 final",
    )
