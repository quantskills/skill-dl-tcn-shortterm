"""Leakage-safe ordinary-validation manifests for stability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

import numpy as np
import pandas as pd

from .experiment import ContractError


@dataclass(frozen=True)
class ValidationStabilityManifest:
    """Derived folds, fold summary, and deterministic protocol fingerprint."""

    manifest: pd.DataFrame
    summary: pd.DataFrame
    fingerprint: str


@dataclass(frozen=True)
class TCNStabilityGateDecision:
    """Pre-registered CPU speed and ordinary-validation effect decision."""

    candidate_model: str
    speed_status: str
    effect_status: str
    model_step_speedup: float
    end_to_end_speedup: float
    candidate_median_rankic: float
    baseline_median_rankic: float
    control_median_rankic: float
    control_median_improvement: float
    positive_rate: float
    worst_fold_rankic: float
    model_step_three_x: bool


def evaluate_tcn_stability_gate(
    measurements: pd.DataFrame,
    *,
    candidate_model: str,
    recurrent_baseline: str,
    tcn_control: str,
    model_step_speedup_min: float,
    end_to_end_speedup_min: float,
    positive_rate_min: float,
    control_median_improvement_min: float,
    worst_fold_min: float,
) -> TCNStabilityGateDecision:
    """Apply speed and prediction gates to identical fold/seed units."""

    required = {
        "model",
        "fold",
        "base_seed",
        "best_validation_rankic",
        "model_step_samples_per_second",
        "samples_per_second",
    }
    if missing := sorted(required.difference(measurements.columns)):
        raise ContractError(f"stability measurements missing columns: {', '.join(missing)}")
    if (
        model_step_speedup_min <= 0
        or end_to_end_speedup_min <= 0
        or not 0 <= positive_rate_min <= 1
        or control_median_improvement_min < 0
    ):
        raise ContractError("stability gate thresholds are invalid")
    models = {candidate_model, recurrent_baseline, tcn_control}
    selected = measurements.loc[measurements["model"].isin(models)].copy()
    if set(selected["model"].astype(str)) != models:
        raise ContractError("stability measurements are missing a required model")
    if selected.duplicated(["model", "fold", "base_seed"]).any():
        raise ContractError("stability measurements contain duplicate model units")
    unit_sets = {
        str(model): set(zip(group["fold"], group["base_seed"], strict=True))
        for model, group in selected.groupby("model", observed=True)
    }
    expected_units = unit_sets[recurrent_baseline]
    if not expected_units or any(units != expected_units for units in unit_sets.values()):
        raise ContractError("stability models must cover identical fold/seed units")
    numeric_columns = [
        "best_validation_rankic",
        "model_step_samples_per_second",
        "samples_per_second",
    ]
    if not np.isfinite(selected[numeric_columns].to_numpy(dtype="float64")).all():
        raise ContractError("stability measurements contain non-finite values")
    if selected[["model_step_samples_per_second", "samples_per_second"]].le(0).any().any():
        raise ContractError("stability throughput must be positive")

    indexed = selected.set_index(["fold", "base_seed", "model"])
    candidate = indexed.xs(candidate_model, level="model")
    baseline = indexed.xs(recurrent_baseline, level="model")
    control = indexed.xs(tcn_control, level="model")

    def geomean_ratio(numerator: pd.Series, denominator: pd.Series) -> float:
        return float(np.exp(np.log(numerator / denominator).mean()))

    model_step_speedup = geomean_ratio(
        candidate["model_step_samples_per_second"],
        baseline["model_step_samples_per_second"],
    )
    end_to_end_speedup = geomean_ratio(
        candidate["samples_per_second"], baseline["samples_per_second"]
    )
    model_speed_pass = model_step_speedup >= model_step_speedup_min
    end_to_end_pass = end_to_end_speedup >= end_to_end_speedup_min
    if model_speed_pass and end_to_end_pass:
        speed_status = "cpu_end_to_end_speedup_confirmed"
    elif model_speed_pass:
        speed_status = "cpu_model_step_speedup_confirmed"
    else:
        speed_status = "no_cpu_speedup"

    candidate_rankic = candidate["best_validation_rankic"]
    baseline_median = float(baseline["best_validation_rankic"].median())
    control_median = float(control["best_validation_rankic"].median())
    candidate_median = float(candidate_rankic.median())
    control_improvement = candidate_median - control_median
    positive_rate = float(candidate_rankic.gt(0).mean())
    worst_fold = float(candidate_rankic.groupby(level="fold").mean().min())
    effect_pass = (
        positive_rate >= positive_rate_min
        and candidate_median > baseline_median
        and control_improvement >= control_median_improvement_min
        and worst_fold >= worst_fold_min
    )
    return TCNStabilityGateDecision(
        candidate_model=candidate_model,
        speed_status=speed_status,
        effect_status=(
            "validation_effect_confirmed"
            if effect_pass
            else "stop_unstable_validation"
        ),
        model_step_speedup=model_step_speedup,
        end_to_end_speedup=end_to_end_speedup,
        candidate_median_rankic=candidate_median,
        baseline_median_rankic=baseline_median,
        control_median_rankic=control_median,
        control_median_improvement=control_improvement,
        positive_rate=positive_rate,
        worst_fold_rankic=worst_fold,
        model_step_three_x=model_step_speedup >= 3.0,
    )


def build_validation_stability_manifest(
    window_index: pd.DataFrame,
    labels: pd.DataFrame,
    source_split_manifest: pd.DataFrame,
    *,
    source_fold: int,
    train_days: int,
    validation_days: int,
    fold_count: int,
    window_kind: Literal["expanding", "sliding"],
) -> ValidationStabilityManifest:
    """Build expanding or sliding folds only from ordinary train/validation rows."""

    if train_days <= 0 or validation_days <= 0 or fold_count <= 0:
        raise ContractError("stability train, validation, and fold counts must be positive")
    if window_kind not in {"expanding", "sliding"}:
        raise ContractError("stability window kind must be expanding or sliding")
    index_columns = {
        "sample_position",
        "sample_id",
        "signal_date",
        "instrument_id",
    }
    if missing := sorted(index_columns.difference(window_index.columns)):
        raise ContractError(f"window index missing columns: {', '.join(missing)}")
    label_columns = {"sample_id", "valid", "label_end_at"}
    if missing := sorted(label_columns.difference(labels.columns)):
        raise ContractError(f"labels missing columns: {', '.join(missing)}")
    split_columns = {"fold", "sample_position", "stage", "sealed"}
    if missing := sorted(split_columns.difference(source_split_manifest.columns)):
        raise ContractError(f"source split manifest missing columns: {', '.join(missing)}")

    source_rows = source_split_manifest.loc[
        source_split_manifest["fold"].astype(int).eq(source_fold)
    ].copy()
    if source_rows.empty:
        raise ContractError(f"source fold {source_fold} is unavailable")
    ordinary = source_rows.loc[
        source_rows["stage"].isin(["train", "validation"])
        & ~source_rows["sealed"].astype(bool),
        ["sample_position", "stage"],
    ].copy()
    if ordinary["sample_position"].duplicated().any():
        raise ContractError("ordinary-validation source positions must be unique")
    eligible = window_index.merge(
        ordinary.rename(columns={"stage": "source_stage"}),
        on="sample_position",
        how="inner",
        validate="one_to_one",
    )
    eligible["signal_date"] = eligible["signal_date"].astype(str)
    eligible_dates = sorted(eligible["signal_date"].unique())
    required_days = train_days + validation_days * fold_count
    if len(eligible_dates) < required_days:
        raise ContractError(
            "insufficient ordinary-validation dates: "
            f"need {required_days}, observed {len(eligible_dates)}"
        )

    valid_labels = labels.loc[labels["valid"].astype(bool)].copy()
    valid_labels["label_end_at"] = pd.to_datetime(
        valid_labels["label_end_at"], utc=True, errors="coerce"
    )
    if valid_labels["label_end_at"].isna().any():
        raise ContractError("valid labels contain invalid label_end_at values")
    valid_labels["label_end_date"] = (
        valid_labels["label_end_at"].dt.tz_convert("Asia/Shanghai").dt.date
    )
    max_label_end = valid_labels.groupby("sample_id", observed=True)[
        "label_end_date"
    ].max()

    manifest_parts = []
    summary_rows = []
    for fold in range(fold_count):
        validation_start_index = train_days + fold * validation_days
        validation_date_values = eligible_dates[
            validation_start_index : validation_start_index + validation_days
        ]
        if window_kind == "expanding":
            train_date_values = eligible_dates[:validation_start_index]
        else:
            train_date_values = eligible_dates[
                validation_start_index - train_days : validation_start_index
            ]
        validation_start = pd.Timestamp(validation_date_values[0]).date()
        train_candidates = eligible.loc[
            eligible["signal_date"].isin(train_date_values)
        ].copy()
        candidate_label_end = train_candidates["sample_id"].map(max_label_end)
        overlap = candidate_label_end.isna() | candidate_label_end.ge(validation_start)
        train = train_candidates.loc[~overlap].copy()
        purged = train_candidates.loc[overlap].copy()
        validation = eligible.loc[
            eligible["signal_date"].isin(validation_date_values)
        ].copy()
        if train.empty or validation.empty:
            raise ContractError(f"stability fold {fold} has an empty train or validation stage")
        for rows, stage, reason in [
            (train, "train", f"{window_kind}_ordinary_train"),
            (purged, "purged", "label_end_overlaps_validation"),
            (validation, "validation", "ordinary_validation"),
        ]:
            rows = rows.copy()
            rows["fold"] = fold
            rows["stage"] = stage
            rows["stage_reason"] = reason
            rows["source_fold"] = source_fold
            rows["window_kind"] = window_kind
            rows["validation_start_date"] = validation_date_values[0]
            rows["validation_end_date"] = validation_date_values[-1]
            rows["sealed"] = False
            manifest_parts.append(rows)
        summary_rows.append(
            {
                "fold": fold,
                "window_kind": window_kind,
                "train_days": train["signal_date"].nunique(),
                "validation_days": validation["signal_date"].nunique(),
                "train_sample_count": len(train),
                "validation_sample_count": len(validation),
                "purged_sample_count": len(purged),
                "train_start_date": train["signal_date"].min(),
                "train_end_date": train["signal_date"].max(),
                "validation_start_date": validation_date_values[0],
                "validation_end_date": validation_date_values[-1],
            }
        )

    manifest = pd.concat(manifest_parts, ignore_index=True).sort_values(
        ["fold", "stage", "signal_date", "sample_position"], kind="mergesort"
    )
    fingerprint_payload = {
        "source_fold": source_fold,
        "train_days": train_days,
        "validation_days": validation_days,
        "fold_count": fold_count,
        "window_kind": window_kind,
        "rows": manifest[
            ["fold", "stage", "sample_position", "signal_date"]
        ].to_dict("records"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    manifest["data_fingerprint"] = fingerprint
    summary = pd.DataFrame(summary_rows)
    summary["data_fingerprint"] = fingerprint
    return ValidationStabilityManifest(manifest, summary, fingerprint)
