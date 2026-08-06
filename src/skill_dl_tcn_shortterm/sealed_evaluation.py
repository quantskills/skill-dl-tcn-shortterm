"""Pure statistics and once-only consumption state for sealed TCN evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast
import uuid

import numpy as np
import pandas as pd

from .experiment import ContractError
from .sealed_readiness import TASK_ALIGNED_GATE_FIELDS
from .v9_statistics import _paired_block_bootstrap


SEALED_METRICS = (
    "rankic",
    "pearson_ic",
    "top_return",
    "top_excess_return",
    "long_short_spread",
    "top_precision",
    "ndcg_at_top",
    "quantile_monotonicity",
    "top_turnover",
    "net_return_after_cost",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def claim_sealed_consumption(
    registry_dir: str | Path,
    *,
    freeze_id: str,
    sealed_data_sha256: str,
) -> Path:
    """Irreversibly claim one sealed identity before its values are loaded."""

    if len(freeze_id) != 64 or len(sealed_data_sha256) != 64:
        raise ContractError("sealed consumption identities must be SHA-256 values")
    root = Path(registry_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        f"{freeze_id}|{sealed_data_sha256}".encode("utf-8")
    ).hexdigest()
    marker = root / f"{key}.json"
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "schema_version": "once-only-sealed-consumption-v36/v1",
                        "freeze_id": freeze_id,
                        "sealed_data_sha256": sealed_data_sha256,
                        "status": "running",
                        "claimed_at": _now(),
                        "result_receipt": None,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ContractError("sealed data has already been consumed or claimed") from exc
    return marker


def complete_sealed_consumption(marker: str | Path, *, result_receipt: str) -> None:
    """Complete a claim; a failed claim intentionally remains non-retryable."""

    path = Path(marker).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read sealed consumption marker: {exc}") from exc
    if value.get("status") != "running" or value.get("result_receipt") is not None:
        raise ContractError("sealed consumption marker is not running")
    value["status"] = "completed"
    value["completed_at"] = _now()
    value["result_receipt"] = result_receipt
    _atomic_json(path, value)


def paired_daily_unit_mean(
    metrics: pd.DataFrame,
    *,
    reference_model: str,
    candidate_model: str,
    one_way_cost_bps: float,
) -> pd.DataFrame:
    """Average paired model-unit deltas before treating dates as observations."""

    if not math.isfinite(one_way_cost_bps) or one_way_cost_bps <= 0:
        raise ContractError("one-way cost must be finite and positive")
    required = {
        "model",
        "seed",
        "fold",
        "signal_date",
        "horizon",
        *SEALED_METRICS[:-1],
    }
    if missing := required - set(metrics):
        raise ContractError(f"sealed metrics missing columns: {sorted(missing)}")
    selected = metrics.loc[
        metrics["model"].astype(str).isin([reference_model, candidate_model])
    ].copy()
    if set(selected["model"].astype(str)) != {reference_model, candidate_model}:
        raise ContractError("sealed paired model coverage is incomplete")
    index = ["seed", "fold", "signal_date", "horizon"]
    value_columns = list(SEALED_METRICS[:-1])
    reference = selected.loc[
        selected["model"].astype(str).eq(reference_model)
    ].set_index(index)[value_columns].sort_index()
    candidate = selected.loc[
        selected["model"].astype(str).eq(candidate_model)
    ].set_index(index)[value_columns].sort_index()
    if reference.empty or not reference.index.equals(candidate.index):
        raise ContractError("sealed paired sample coverage drifted")
    cost_rate = float(one_way_cost_bps) / 10_000.0
    reference["net_return_after_cost"] = reference["top_return"] - (
        reference["top_turnover"].fillna(1.0) * cost_rate
    )
    candidate["net_return_after_cost"] = candidate["top_return"] - (
        candidate["top_turnover"].fillna(1.0) * cost_rate
    )
    paired = (candidate - reference).reset_index()
    paired["sealed_fold"] = paired["fold"].astype(int) // 10
    paired["training_fold"] = paired["fold"].astype(int) % 10
    grouped = (
        paired.groupby(
            ["sealed_fold", "signal_date", "horizon"], observed=True, sort=True
        )[list(SEALED_METRICS)]
        .mean()
        .reset_index()
    )
    grouped.insert(0, "candidate_model", candidate_model)
    grouped.insert(0, "reference_model", reference_model)
    if grouped.empty or grouped.duplicated(
        ["sealed_fold", "signal_date", "horizon"]
    ).any():
        raise ContractError("sealed daily paired aggregation is invalid")
    return grouped


def summarize_paired_daily(paired_daily: pd.DataFrame) -> dict[str, object]:
    """Summarize the market-date weighted deltas used by the frozen gates."""

    if paired_daily.empty:
        raise ContractError("sealed paired daily deltas cannot be empty")
    result: dict[str, object] = {
        "reference_model": str(paired_daily["reference_model"].iloc[0]),
        "candidate_model": str(paired_daily["candidate_model"].iloc[0]),
        "paired_group_count": int(len(paired_daily)),
        "sealed_fold_count": int(paired_daily["sealed_fold"].nunique()),
        "date_count": int(paired_daily["signal_date"].nunique()),
    }
    for metric in SEALED_METRICS:
        values = paired_daily[metric].to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            raise ContractError(f"sealed paired metric has no finite values: {metric}")
        result[f"mean_{metric}_delta"] = float(finite.mean())
    primary = [
        cast(float, result["mean_rankic_delta"]),
        cast(float, result["mean_top_return_delta"]),
        cast(float, result["mean_top_precision_delta"]),
    ]
    result["winner_consensus"] = (
        "candidate"
        if all(value > 0 for value in primary)
        else "reference"
        if all(value < 0 for value in primary)
        else "mixed"
    )
    return result


def bootstrap_paired_daily(
    paired_daily: pd.DataFrame,
    *,
    seed: int,
    draws: int,
) -> pd.DataFrame:
    """Block-bootstrap dates inside each sealed-segment/horizon unit."""

    if draws < 1000:
        raise ContractError("sealed bootstrap requires at least 1000 draws")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for metric in SEALED_METRICS:
        unit_draws: list[np.ndarray] = []
        unit_means: list[float] = []
        for _, group in paired_daily.groupby(
            ["sealed_fold", "horizon"], observed=True, sort=True
        ):
            values = group.sort_values("signal_date", kind="mergesort")[
                metric
            ].to_numpy(dtype="float64")
            values = values[np.isfinite(values)]
            if len(values) < 20:
                raise ContractError(f"sealed bootstrap unit is too short: {metric}")
            sampled, _ = _paired_block_bootstrap(values, rng, draws=draws)
            unit_draws.append(sampled)
            unit_means.append(float(values.mean()))
        aggregate = np.stack(unit_draws).mean(axis=0)
        low, high = np.quantile(aggregate, [0.025, 0.975]).tolist()
        rows.append(
            {
                "metric": metric,
                "reference_model": str(paired_daily["reference_model"].iloc[0]),
                "candidate_model": str(paired_daily["candidate_model"].iloc[0]),
                "unit_count": len(unit_draws),
                "paired_group_count": len(paired_daily),
                "paired_mean_delta": float(np.mean(unit_means)),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(rows)


def decide_sealed_candidate(
    comparison: Mapping[str, object],
    bootstrap: pd.DataFrame,
    *,
    speed: Mapping[str, object],
    gates: Mapping[str, object],
) -> dict[str, object]:
    """Apply only the task-aligned gates frozen before sealed access."""

    if set(gates) != TASK_ALIGNED_GATE_FIELDS:
        raise ContractError("sealed decision gate identity drifted")
    boot = bootstrap.set_index("metric")
    for metric in (
        "top_precision",
        "ndcg_at_top",
        "top_return",
        "net_return_after_cost",
    ):
        if metric not in boot.index:
            raise ContractError(f"sealed bootstrap metric is missing: {metric}")
    precision_low = float(
        cast(Any, boot.loc["top_precision", "bootstrap_ci_low"])
    )
    ndcg_low = float(cast(Any, boot.loc["ndcg_at_top", "bootstrap_ci_low"]))
    robust_tail = bool(
        (
            precision_low >= float(cast(Any, gates["min_primary_tail_ci_low"]))
            and ndcg_low
            >= float(cast(Any, gates["min_secondary_tail_ci_low"]))
        )
        or (
            ndcg_low >= float(cast(Any, gates["min_primary_tail_ci_low"]))
            and precision_low
            >= float(cast(Any, gates["min_secondary_tail_ci_low"]))
        )
    )
    outcomes = {
        "mean_top_precision": float(
            cast(Any, comparison["mean_top_precision_delta"])
        )
        >= float(cast(Any, gates["min_mean_top_precision_delta"])),
        "mean_ndcg_at_top": float(
            cast(Any, comparison["mean_ndcg_at_top_delta"])
        )
        >= float(cast(Any, gates["min_mean_ndcg_at_top_delta"])),
        "robust_tail": robust_tail,
        "mean_rankic": float(cast(Any, comparison["mean_rankic_delta"]))
        >= float(cast(Any, gates["min_mean_rankic_delta"])),
        "top_return_ci": float(
            cast(Any, boot.loc["top_return", "bootstrap_ci_low"])
        )
        >= float(cast(Any, gates["min_top_return_ci_low"])),
        "net_return_after_cost_ci": float(
            cast(Any, boot.loc["net_return_after_cost", "bootstrap_ci_low"])
        )
        >= float(cast(Any, gates["min_net_return_after_cost_ci_low"])),
        "mean_top_turnover": float(
            cast(Any, comparison["mean_top_turnover_delta"])
        )
        <= float(cast(Any, gates["max_mean_top_turnover_delta"])),
        "model_step_speed": float(cast(Any, speed["model_step_speed_ratio"]))
        >= float(cast(Any, gates["min_model_step_speed_ratio"])),
        "end_to_end_speed": float(cast(Any, speed["end_to_end_speed_ratio"]))
        >= float(cast(Any, gates["min_end_to_end_speed_ratio"])),
    }
    passed = all(outcomes.values())
    return {
        "status": (
            "sealed_confirmed_tcn_candidate_v36"
            if passed
            else "sealed_rejected_tcn_candidate_v36"
        ),
        "candidate_model": passed,
        "threshold_outcomes": outcomes,
        "robust_tail_improvement": robust_tail,
        "precision_ci_low": precision_low,
        "ndcg_ci_low": ndcg_low,
        "post_result_tuning_authorized": False,
        "deployment_authorized": False,
        "trading_authorized": False,
    }
