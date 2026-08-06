"""Reproducible evidence indexes and report rendering."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .experiment import ContractError
from .integrity import code_identity


SENSITIVE_KEY_PARTS = ("password", "secret", "token", "api_key", "private_key")
RUN_SCOPED_PARQUET = {
    "predictions",
    "metrics",
    "portfolio_metrics",
    "portfolio_ledger",
    "orders",
    "vintages",
    "diagnostic",
    "execution_ledger",
    "execution_metrics",
    "performance_metrics",
    "training_metadata",
    "tcn_comparison",
    "risk_exposures",
}


@dataclass(frozen=True)
class EvidenceBundleResult:
    index_path: Path
    environment_path: Path


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_secrets(value: Any, prefix: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            empty_secret = child is None or child == "" or child is False
            if (
                any(part in normalized for part in SENSITIVE_KEY_PARTS)
                and not empty_secret
            ):
                raise ContractError(
                    f"secret-like value cannot be persisted: {prefix}.{key}"
                )
            _assert_no_secrets(child, f"{prefix}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for position, child in enumerate(value):
            _assert_no_secrets(child, f"{prefix}[{position}]")


def _environment(root: Path) -> dict[str, object]:
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "torch"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "code": code_identity(Path(__file__).resolve().parents[2]),
    }


def _assert_run_identity(name: str, path: Path, run_id: str) -> None:
    if name not in RUN_SCOPED_PARQUET or path.suffix != ".parquet":
        return
    try:
        frame = pd.read_parquet(path, columns=["run_id"])
    except Exception as exc:
        raise ContractError(f"cannot validate run identity for {name}: {exc}") from exc
    if frame.empty or set(frame["run_id"].astype(str)) != {run_id}:
        raise ContractError(f"artifact {name} does not share run identity {run_id}")


def write_evidence_bundle(
    run_dir: str | Path,
    *,
    run_id: str,
    config: Mapping[str, Any],
    seed: int,
    artifacts: Mapping[str, str | Path],
    required_artifacts: set[str],
    engineering_complete: bool,
) -> EvidenceBundleResult:
    """Freeze an evidence index after checking completeness and run identity."""

    _assert_no_secrets(config)
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    missing = sorted(required_artifacts - set(artifacts))
    if engineering_complete and missing:
        raise ContractError(
            f"engineering-complete evidence is missing: {', '.join(missing)}"
        )
    resolved_artifacts: dict[str, Path] = {}
    for name, raw_path in artifacts.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ContractError(f"evidence artifact does not exist: {name}")
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ContractError(
                f"evidence artifact is outside run directory: {name}"
            ) from exc
        _assert_run_identity(name, path, run_id)
        resolved_artifacts[name] = path

    environment_path = root / "environment.json"
    environment_path.write_text(
        _canonical_json(_environment(root)) + "\n",
        encoding="utf-8",
    )
    resolved_artifacts["environment"] = environment_path
    artifact_index = {
        name: {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(resolved_artifacts.items())
    }
    index = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": seed,
        "engineering_complete": engineering_complete and not missing,
        "config_sha256": hashlib.sha256(
            _canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "required_artifacts": sorted(required_artifacts),
        "artifacts": artifact_index,
    }
    index_path = root / "evidence-index.json"
    index_path.write_text(_canonical_json(index) + "\n", encoding="utf-8")
    return EvidenceBundleResult(
        index_path=index_path, environment_path=environment_path
    )


def verify_evidence_bundle(index_path: str | Path) -> None:
    """Recompute every artifact digest recorded in an evidence index."""

    path = Path(index_path).resolve()
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read evidence index: {exc}") from exc
    if index.get("schema_version") != 1:
        raise ContractError("evidence index schema_version must equal 1")
    for name, entry in index.get("artifacts", {}).items():
        artifact_path = (path.parent / entry["path"]).resolve()
        if not artifact_path.is_file():
            raise ContractError(f"evidence artifact is missing: {name}")
        if _sha256(artifact_path) != entry["sha256"]:
            raise ContractError(f"artifact fingerprint mismatch: {name}")


def _frame_block(title: str, frame: pd.DataFrame, columns: Sequence[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    if frame.empty or not available:
        return f"## {title}\n\n无可用证据。\n"
    rendered = frame[available].to_csv(index=False, float_format="%.6g").strip()
    return f"## {title}\n\n```text\n{rendered}\n```\n"


def _data_quality_summary(
    *,
    data_quality: Mapping[str, Any] | None,
    canonical_bars: pd.DataFrame | None,
    universe: pd.DataFrame | None,
    window_index: pd.DataFrame | None,
    window_rejections: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any) -> None:
        rows.append({"section": section, "metric": metric, "value": value})

    for metric, value in sorted(dict(data_quality or {}).items()):
        add(
            "aggregation",
            metric,
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else value,
        )
    if canonical_bars is not None and "quality_flag" in canonical_bars:
        for reason, count in (
            canonical_bars["quality_flag"].astype(str).value_counts().items()
        ):
            add("aggregation_rejection_reason", str(reason), int(count))
    if universe is not None and not universe.empty:
        for column in ["universe_version", "admission_config"]:
            if column in universe:
                add(
                    "pit_universe",
                    column,
                    ";".join(sorted(universe[column].astype(str).unique())),
                )
        eligible_count = (
            int(universe["eligible"].fillna(False).astype(bool).sum())
            if "eligible" in universe
            else 0
        )
        add("pit_universe", "snapshot_count", len(universe))
        add("pit_universe", "eligible_count", eligible_count)
        add(
            "pit_universe",
            "eligibility_coverage",
            eligible_count / len(universe) if len(universe) else 0.0,
        )
        if "exclusion_reasons" in universe:
            reasons = pd.Series(
                [
                    reason
                    for value in universe["exclusion_reasons"].fillna("").tolist()
                    for reason in str(value).split(";")
                ],
                dtype="object",
            )
            for reason, count in reasons.loc[reasons.ne("")].value_counts().items():
                add("pit_exclusion_reason", str(reason), int(count))
    valid_count = len(window_index) if window_index is not None else 0
    rejected_count = len(window_rejections) if window_rejections is not None else 0
    if window_index is not None or window_rejections is not None:
        add("feature_window", "valid_count", valid_count)
        add("feature_window", "rejected_count", rejected_count)
        add(
            "feature_window",
            "coverage",
            valid_count / (valid_count + rejected_count)
            if valid_count + rejected_count
            else 0.0,
        )
    if window_index is not None and not window_index.empty:
        for column in ["window_version", "time_steps"]:
            if column in window_index:
                add(
                    "feature_window",
                    f"actual_{column}",
                    ";".join(sorted(window_index[column].astype(str).unique())),
                )
    if window_rejections is not None and not window_rejections.empty:
        for reason, count in (
            window_rejections["rejection_reason"].value_counts().items()
        ):
            add("window_rejection_reason", str(reason), int(count))
        if "observed_steps" in window_rejections:
            add(
                "feature_window",
                "rejected_observed_steps_range",
                f"{int(window_rejections['observed_steps'].min())}..{int(window_rejections['observed_steps'].max())}",
            )
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def render_research_report(
    *,
    engineering_complete: bool,
    metrics: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    execution_metrics: pd.DataFrame,
    performance_metrics: pd.DataFrame,
    execution_ledger: pd.DataFrame,
    risk_exposures: pd.DataFrame | None = None,
    data_quality: Mapping[str, Any] | None = None,
    canonical_bars: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    window_index: pd.DataFrame | None = None,
    window_rejections: pd.DataFrame | None = None,
) -> str:
    """Render conclusions exclusively from structured artifacts."""

    tcn_models = (
        metrics["model"].map(lambda value: "tcn" in str(value).lower())
        if "model" in metrics
        else pd.Series(False, index=metrics.index)
    )
    positive_alpha = bool(
        not metrics.empty
        and {"model", "paired_delta_ci_low", "comparison_baseline"}.issubset(
            metrics.columns
        )
        and (
            pd.to_numeric(
                metrics.loc[tcn_models, "paired_delta_ci_low"], errors="coerce"
            )
            > 0
        ).any()
    )
    alpha_conclusion = (
        "存在相对最强非 TCN 基准的初步增量证据（尚非封存测试结论）"
        if positive_alpha
        else "未发现相对最强非 TCN 基准的增量证据"
    )
    speed_conclusion = "未评估"
    if not performance_metrics.empty and {
        "model",
        "samples_per_second",
    } <= set(performance_metrics.columns):
        tcn = performance_metrics.loc[
            performance_metrics["model"] == "bai-tcn", "samples_per_second"
        ]
        recurrent = performance_metrics.loc[
            performance_metrics["model"].isin(["lstm", "gru"]), "samples_per_second"
        ]
        if not tcn.empty and not recurrent.empty and recurrent.max() > 0:
            ratio = float(tcn.iloc[0] / recurrent.max())
            speed_conclusion = (
                f"获支持（观察到 {ratio:.2f}×）"
                if ratio >= 3.0
                else f"未获支持（观察到 {ratio:.2f}×）"
            )
    unfilled_count = (
        int(execution_ledger["unfilled_reason"].astype(str).ne("").sum())
        if "unfilled_reason" in execution_ledger
        else 0
    )
    delayed_count = (
        int(execution_ledger["sell_delay_sessions"].fillna(0).gt(0).sum())
        if "sell_delay_sessions" in execution_ledger
        else 0
    )
    capacity_loss = (
        float(execution_ledger["capacity_clipped_amount"].fillna(0).sum())
        if "capacity_clipped_amount" in execution_ledger
        else 0.0
    )
    quality_summary = _data_quality_summary(
        data_quality=data_quality,
        canonical_bars=canonical_bars,
        universe=universe,
        window_index=window_index,
        window_rejections=window_rejections,
    )
    sections = [
        "# TCN 短线研究证据报告\n",
        "## 结论边界\n\n"
        f"- 工程完成：{'是' if engineering_complete else '否'}\n"
        f"- Alpha 证据：{alpha_conclusion}\n"
        f"- 3–5× 速度假设：{speed_conclusion}\n"
        "- 诊断性多空不属于可实现多头组合，二者不合并展示。\n",
        _frame_block(
            "数据、PIT 与窗口质量证据",
            quality_summary,
            ["section", "metric", "value"],
        ),
        _frame_block(
            "统计预测证据",
            metrics,
            [
                "model",
                "fold",
                "horizon",
                "rankic",
                "icir",
                "rankic_ci_low",
                "rankic_ci_high",
                "comparison_baseline",
                "paired_delta_rankic",
                "paired_delta_ci_low",
                "paired_delta_ci_high",
            ],
        ),
        _frame_block(
            "可实现多头组合",
            portfolio_metrics,
            [
                "model",
                "fold",
                "horizon",
                "cumulative_gross_return_contribution",
                "max_drawdown",
                "mean_one_way_turnover",
                "mean_gross_exposure",
            ],
        ),
        _frame_block(
            "成交与成本情景",
            execution_metrics,
            [
                "portfolio_type",
                "model",
                "fold",
                "horizon",
                "slippage_scenario",
                "slippage_bps",
                "net_return",
                "benchmark_net_return",
                "excess_net_return",
                "gross_pnl",
                "net_pnl",
                "unused_cash",
                "commission_cost",
                "sell_tax_cost",
                "slippage_cost",
            ],
        ),
        f"## 成交异常摘要\n\n- 未成交/未完全退出：{unfilled_count}\n"
        f"- 延迟退出记录：{delayed_count}\n- 容量裁剪金额：{capacity_loss:.6g}\n",
        _frame_block(
            "训练性能",
            performance_metrics,
            [
                "model",
                "samples_per_second",
                "mean_epoch_seconds",
                "time_to_best_seconds",
                "peak_ram_bytes",
                "peak_vram_bytes",
                "data_wait_seconds",
            ],
        ),
        _frame_block(
            "PIT risk exposure stratification (scores unchanged)",
            risk_exposures if risk_exposures is not None else pd.DataFrame(),
            [
                "model",
                "fold",
                "horizon",
                "dimension",
                "bucket",
                "sample_count",
                "mean_daily_rankic",
                "scores_unchanged",
            ],
        ),
    ]
    return "\n".join(sections)
