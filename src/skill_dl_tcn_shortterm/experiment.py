"""Public offline experiment seam."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .integrity import code_identity


class ContractError(ValueError):
    """Raised before training when an experiment contract is invalid."""


REQUIRED_ENGINEERING_MODELS = frozenset(
    {
        "constant-zero",
        "ridge",
        "lightgbm",
        "lstm",
        "gru",
        "bai-tcn",
        "tcn-lite",
        "bai-tcn-1d",
        "bai-tcn-2d",
        "bai-tcn-3d",
        "bai-tcn-5d",
    }
)


def _assert_engineering_complete(
    components: Mapping[str, Any],
    predictions: pd.DataFrame | None,
    metrics: pd.DataFrame | None,
    tcn_comparison: pd.DataFrame | None,
    split_manifest: pd.DataFrame | None,
) -> None:
    missing_components = sorted(
        name for name, value in components.items() if value is None
    )
    if missing_components:
        raise ContractError(
            "engineering-complete run is missing components: "
            + ", ".join(missing_components)
        )
    for artifact_name, frame in [("predictions", predictions), ("metrics", metrics)]:
        observed_models = (
            set(frame["model"].astype(str))
            if frame is not None and "model" in frame
            else set()
        )
        if missing_models := sorted(REQUIRED_ENGINEERING_MODELS - observed_models):
            raise ContractError(
                f"engineering-complete {artifact_name} are missing required models: "
                + ", ".join(missing_models)
            )
    if split_manifest is None or predictions is None or metrics is None:
        raise ContractError("engineering-complete comparison matrix is unavailable")
    folds = sorted(int(value) for value in split_manifest["fold"].unique())
    shared_models = REQUIRED_ENGINEERING_MODELS - {
        "bai-tcn-1d",
        "bai-tcn-2d",
        "bai-tcn-3d",
        "bai-tcn-5d",
    }
    for fold in folds:
        for horizon in [1, 2, 3, 5]:
            expected_models = shared_models | {f"bai-tcn-{horizon}d"}
            reference_samples = set(
                predictions.loc[
                    (predictions["model"] == "constant-zero")
                    & (predictions["fold"] == fold)
                    & (predictions["horizon"] == horizon),
                    "sample_id",
                ].astype(str)
            )
            if not reference_samples:
                raise ContractError(
                    f"engineering-complete reference samples missing for fold {fold} h{horizon}"
                )
            for model in sorted(expected_models):
                model_samples = set(
                    predictions.loc[
                        (predictions["model"] == model)
                        & (predictions["fold"] == fold)
                        & (predictions["horizon"] == horizon),
                        "sample_id",
                    ].astype(str)
                )
                if model_samples != reference_samples:
                    raise ContractError(
                        "engineering-complete models do not share validation samples: "
                        f"{model}/fold{fold}/h{horizon}"
                    )
                metric_rows = metrics.loc[
                    (metrics["model"] == model)
                    & (metrics["fold"] == fold)
                    & (metrics["horizon"] == horizon)
                ]
                if len(metric_rows) != 1:
                    raise ContractError(
                        "engineering-complete metric matrix is incomplete: "
                        f"{model}/fold{fold}/h{horizon}"
                    )
    required_comparisons = {
        (comparison_type, horizon)
        for comparison_type in {"single-vs-shared", "lite-vs-shared"}
        for horizon in {1, 2, 3, 5}
    }
    observed_comparisons = (
        set(
            zip(
                tcn_comparison["comparison_type"].astype(str),
                tcn_comparison["horizon"].astype(int),
                strict=True,
            )
        )
        if tcn_comparison is not None
        and {"comparison_type", "horizon"}.issubset(tcn_comparison.columns)
        else set()
    )
    if missing_comparisons := sorted(required_comparisons - observed_comparisons):
        raise ContractError(
            "engineering-complete run is missing TCN comparison rows: "
            + ", ".join(f"{kind}/h{horizon}" for kind, horizon in missing_comparisons)
        )
    assert tcn_comparison is not None
    if (
        "paired_date_count" not in tcn_comparison
        or (
            pd.to_numeric(tcn_comparison["paired_date_count"], errors="coerce").fillna(
                0
            )
            <= 0
        ).any()
    ):
        raise ContractError(
            "engineering-complete TCN comparisons require paired validation dates"
        )


@dataclass(frozen=True)
class RunResult:
    """Paths belonging to one immutable experiment run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    predictions_path: Path
    metrics_path: Path
    report_path: Path
    canonical_bars_path: Path | None = None
    quality_path: Path | None = None
    universe_path: Path | None = None
    window_index_path: Path | None = None
    windows_path: Path | None = None
    labels_path: Path | None = None
    split_manifest_path: Path | None = None
    preprocessing_path: Path | None = None
    memmap_path: Path | None = None
    window_cache_manifest_path: Path | None = None
    training_metadata_path: Path | None = None
    orders_path: Path | None = None
    vintages_path: Path | None = None
    portfolio_metrics_path: Path | None = None
    portfolio_ledger_path: Path | None = None
    diagnostic_path: Path | None = None
    execution_ledger_path: Path | None = None
    execution_metrics_path: Path | None = None
    performance_metrics_path: Path | None = None
    performance_environment_path: Path | None = None
    evidence_index_path: Path | None = None
    environment_path: Path | None = None
    window_rejections_path: Path | None = None
    tcn_comparison_path: Path | None = None
    risk_exposures_path: Path | None = None


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    run_name = resolved.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ContractError("config.run_name must be a non-empty string")
    seed = resolved.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ContractError("config.seed must be an integer")
    horizons = resolved.get("horizons")
    if horizons != [1, 2, 3, 5]:
        raise ContractError("config.horizons must equal [1, 2, 3, 5]")
    return resolved


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read data manifest: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ContractError("manifest.schema_version must equal 1")
    if manifest.get("dataset_kind") not in {"prebuilt_samples", "raw_1m"}:
        raise ContractError(
            "manifest.dataset_kind must equal prebuilt_samples or raw_1m"
        )
    raw_path = manifest.get("data_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractError("manifest.data_path must be a non-empty string")
    data_path = Path(raw_path)
    if not data_path.is_absolute():
        data_path = manifest_path.parent / data_path
    data_path = data_path.resolve()
    if not data_path.is_file():
        raise ContractError(f"manifest data file does not exist: {data_path}")
    expected = manifest.get("data_sha256")
    actual = _sha256(data_path)
    if expected != actual:
        raise ContractError(
            f"data fingerprint mismatch: expected {expected}, observed {actual}"
        )
    return manifest, data_path


def _load_optional_parquet(
    manifest: Mapping[str, Any], manifest_path: Path, path_key: str, sha_key: str
) -> pd.DataFrame | None:
    raw_path = manifest.get(path_key)
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractError(f"manifest.{path_key} must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"manifest optional data file does not exist: {path}")
    expected = manifest.get(sha_key)
    actual = _sha256(path)
    if expected != actual:
        raise ContractError(f"optional data fingerprint mismatch for {path_key}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise ContractError(f"cannot read optional data {path_key}: {exc}") from exc


def _validate_samples(samples: pd.DataFrame, horizons: list[int]) -> None:
    required = {"instrument_id", "signal_date", *(f"target_{h}d" for h in horizons)}
    missing = sorted(required.difference(samples.columns))
    if missing:
        raise ContractError(f"prebuilt samples missing columns: {', '.join(missing)}")
    if samples.empty:
        raise ContractError("prebuilt samples must contain at least one row")
    if samples[["instrument_id", "signal_date"]].isna().any(axis=None):
        raise ContractError("sample identity columns cannot contain null values")


def _resolve_cost_schedule(
    execution_config: Mapping[str, Any], orders: pd.DataFrame
) -> list[dict[str, Any]]:
    schedule = execution_config.get("cost_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ContractError("execution.cost_schedule must be a non-empty list")
    required = {
        "version",
        "effective_from",
        "commission_bps",
        "sell_tax_bps",
        "slippage_bps",
    }
    normalized = []
    for item in schedule:
        if not isinstance(item, Mapping):
            raise ContractError(
                f"execution cost schedule entry missing fields: {sorted(required)}"
            )
        if missing := required - set(item):
            raise ContractError(
                f"execution cost schedule entry missing fields: {sorted(missing)}"
            )
        entry = dict(item)
        if not isinstance(entry["version"], str) or not entry["version"]:
            raise ContractError("execution cost schedule version must be non-empty")
        try:
            effective_from = pd.Timestamp(entry["effective_from"])
            if pd.isna(effective_from):
                raise ValueError("effective_from")
            if effective_from.tzinfo is None:
                effective_from = effective_from.tz_localize("Asia/Shanghai")
            else:
                effective_from = effective_from.tz_convert("Asia/Shanghai")
            entry["effective_from"] = effective_from.tz_localize(None)
            for field in ["commission_bps", "sell_tax_bps", "slippage_bps"]:
                entry[field] = float(entry[field])
                if not np.isfinite(entry[field]) or entry[field] < 0:
                    raise ValueError(field)
        except (TypeError, ValueError) as exc:
            raise ContractError("execution cost schedule values are invalid") from exc
        normalized.append(entry)
    normalized.sort(key=lambda item: item["effective_from"])
    for entry_at in pd.to_datetime(orders["entry_at"], utc=True):
        entry_date = (
            pd.Timestamp(entry_at).tz_convert("Asia/Shanghai").tz_localize(None)
        )
        candidates = [
            item for item in normalized if item["effective_from"] <= entry_date
        ]
        if not candidates:
            raise ContractError("no execution cost schedule is effective for an order")
    return normalized


def run_experiment(
    *,
    config: Mapping[str, Any],
    manifest_path: str | Path,
    output_root: str | Path,
) -> RunResult:
    """Run one immutable, fully offline research experiment."""

    resolved = _validate_config(config)
    manifest_file = Path(manifest_path).resolve()
    manifest, data_path = _load_manifest(manifest_file)
    try:
        dataset = pd.read_parquet(data_path)
    except Exception as exc:
        raise ContractError(
            f"cannot read {manifest['dataset_kind']} data: {exc}"
        ) from exc
    horizons = list(resolved["horizons"])
    canonical_bars: pd.DataFrame | None = None
    quality: dict[str, Any] | None = None
    universe: pd.DataFrame | None = None
    feature_windows: np.ndarray | None = None
    window_index: pd.DataFrame | None = None
    labels: pd.DataFrame | None = None
    split_manifest: pd.DataFrame | None = None
    preprocessing: dict[str, Any] | None = None
    baseline_predictions: pd.DataFrame | None = None
    baseline_metrics: pd.DataFrame | None = None
    training_metadata: pd.DataFrame | None = None
    orders: pd.DataFrame | None = None
    vintages: pd.DataFrame | None = None
    portfolio_metrics: pd.DataFrame | None = None
    portfolio_ledger: pd.DataFrame | None = None
    diagnostic: pd.DataFrame | None = None
    execution_ledger: pd.DataFrame | None = None
    execution_metrics: pd.DataFrame | None = None
    performance_metrics: pd.DataFrame | None = None
    performance_environment: dict[str, object] | None = None
    corporate_actions: pd.DataFrame | None = None
    window_rejections: pd.DataFrame | None = None
    tcn_comparison: pd.DataFrame | None = None
    risk_exposures: pd.DataFrame | None = None
    shared_tcn_result: Any | None = None
    temporary_window_store: Any | None = None
    temporary_relative_store: Any | None = None
    optimized_feature_windows: np.ndarray | None = None
    optimized_window_index: pd.DataFrame | None = None
    if manifest["dataset_kind"] == "prebuilt_samples":
        samples = dataset
        _validate_samples(samples, horizons)
        prediction_identities = samples[["instrument_id", "signal_date"]].copy()
    else:
        from .market_data import aggregate_five_minute_bars
        from .features import FEATURE_NAMES, build_feature_windows_with_quality
        from .labels import build_labels
        from .splits import build_walk_forward_splits
        from .universe import build_pit_universe

        canonical_bars, quality = aggregate_five_minute_bars(dataset, manifest)
        states = _load_optional_parquet(
            manifest, manifest_file, "instrument_state_path", "instrument_state_sha256"
        )
        if states is not None:
            universe = build_pit_universe(
                canonical_bars, states, resolved.get("universe")
            )
            universe["universe_version"] = "pit-a-share-v1"
            universe["state_fingerprint"] = manifest["instrument_state_sha256"]
            universe["admission_config"] = _canonical_json(
                dict(resolved.get("universe", {}))
            )
            lookback_days = int(resolved.get("lookback_days", 10))
            if lookback_days not in {5, 10, 20}:
                raise ContractError("config.lookback_days must be one of 5, 10, or 20")
            feature_windows, window_index, window_rejections = (
                build_feature_windows_with_quality(
                    canonical_bars,
                    universe,
                    lookback_days=lookback_days,
                    source_fingerprint=manifest["data_sha256"],
                )
            )
            from .streaming import TemporaryWindowMemmap, TemporaryWindowOutput

            temporary_window_store = TemporaryWindowMemmap(feature_windows)
            feature_windows = temporary_window_store.array
            quality["training_window_storage"] = "read_only_memmap"
            quality["valid_window_count"] = len(window_index)
            quality["rejected_window_count"] = len(window_rejections)
            quality["window_rejection_reasons"] = (
                window_rejections["rejection_reason"].value_counts().to_dict()
                if not window_rejections.empty
                else {}
            )
            corporate_actions = _load_optional_parquet(
                manifest,
                manifest_file,
                "corporate_action_path",
                "corporate_action_sha256",
            )
            labels = build_labels(
                window_index,
                canonical_bars,
                horizons=horizons,
                corporate_actions=corporate_actions,
            )
            if "walk_forward" in resolved:
                split_config = dict(resolved["walk_forward"])
                split_result = build_walk_forward_splits(
                    window_index,
                    labels,
                    feature_windows,
                    train_days=int(split_config["train_days"]),
                    validation_days=int(split_config["validation_days"]),
                    embargo_days=int(split_config.get("embargo_days", 5)),
                    test_days=int(split_config["test_days"]),
                    max_folds=split_config.get("max_folds"),
                )
                split_manifest = split_result.manifest
                preprocessing = split_result.preprocessing
                from .baselines import run_statistical_baselines

                baseline_result = run_statistical_baselines(
                    feature_windows,
                    window_index,
                    labels,
                    split_manifest,
                    seed=int(resolved["seed"]),
                )
                baseline_predictions = baseline_result.predictions
                baseline_metrics = baseline_result.metrics
                loader_config = dict(resolved.get("data_loader", {}))
                num_workers = int(loader_config.get("num_workers", 0))
                if num_workers < 0:
                    raise ContractError("data_loader.num_workers cannot be negative")
                sequence_config = dict(resolved.get("sequence_models", {}))
                if sequence_config.get("enabled", False):
                    from .neural import run_sequence_baselines

                    sequence_result = run_sequence_baselines(
                        feature_windows,
                        window_index,
                        labels,
                        split_manifest,
                        seed=int(resolved["seed"]),
                        hidden_size=int(sequence_config.get("hidden_size", 32)),
                        epochs=int(sequence_config.get("epochs", 10)),
                        batch_size=int(sequence_config.get("batch_size", 64)),
                        num_workers=num_workers,
                    )
                    baseline_predictions = pd.concat(
                        [baseline_predictions, sequence_result.predictions],
                        ignore_index=True,
                    )
                    baseline_metrics = pd.concat(
                        [baseline_metrics, sequence_result.metrics], ignore_index=True
                    )
                    training_metadata = sequence_result.training_metadata
                tcn_config = dict(resolved.get("tcn", {}))
                if tcn_config.get("enabled", False):
                    from .tcn import run_bai_tcn

                    tcn_result = run_bai_tcn(
                        feature_windows,
                        window_index,
                        labels,
                        split_manifest,
                        seed=int(resolved["seed"]),
                        channels=int(tcn_config.get("channels", 64)),
                        kernel_size=int(tcn_config.get("kernel_size", 3)),
                        dilations=tuple(
                            tcn_config.get("dilations", [1, 2, 4, 8, 16, 32, 64])
                        ),
                        dropout=float(tcn_config.get("dropout", 0.1)),
                        epochs=int(tcn_config.get("epochs", 10)),
                        batch_size=int(tcn_config.get("batch_size", 64)),
                        num_workers=num_workers,
                    )
                    shared_tcn_result = tcn_result
                    baseline_predictions = pd.concat(
                        [baseline_predictions, tcn_result.predictions],
                        ignore_index=True,
                    )
                    baseline_metrics = pd.concat(
                        [baseline_metrics, tcn_result.metrics], ignore_index=True
                    )
                    training_metadata = pd.concat(
                        [
                            training_metadata
                            if training_metadata is not None
                            else pd.DataFrame(),
                            tcn_result.training_metadata,
                        ],
                        ignore_index=True,
                    )
                optimized_config = dict(resolved.get("optimized_tcn", {}))
                if optimized_config.get("enabled", False):
                    from .optimized_tcn import (
                        resolve_optimized_tcn_profile,
                        run_optimized_tcn,
                    )

                    profile = resolve_optimized_tcn_profile(
                        str(optimized_config.get("profile", "v40-portable")),
                        learning_rate=float(
                            optimized_config.get("learning_rate", 0.003)
                        ),
                        batch_size=int(optimized_config.get("batch_size", 128)),
                        epochs=int(optimized_config.get("epochs", 8)),
                        torch_threads=int(
                            optimized_config.get("torch_threads", 8)
                        ),
                    )
                    optimized_feature_windows = feature_windows
                    optimized_window_index = window_index
                    if bool(optimized_config.get("relative_features", True)):
                        if universe is None:
                            raise ContractError(
                                "optimized_tcn relative features require a PIT universe"
                            )
                        from .relative_features import (
                            materialize_top50_appended_relative_sequence_features,
                        )

                        temporary_relative_store = TemporaryWindowOutput(
                            "relative-feature-windows.npy"
                        )
                        relative_path = temporary_relative_store.path
                        relative_result = (
                            materialize_top50_appended_relative_sequence_features(
                                feature_windows,
                                window_index,
                                universe,
                                output_path=relative_path,
                                bars_per_day=48,
                                min_cross_section=int(
                                    optimized_config.get("min_cross_section", 31)
                                ),
                            )
                        )
                        optimized_feature_windows = (
                            temporary_relative_store.adopt_read_only(
                                relative_result.features
                            )
                        )
                        optimized_window_index = relative_result.window_index
                        if quality is not None:
                            quality["optimized_tcn_features"] = relative_result.quality
                    optimized_result = run_optimized_tcn(
                        optimized_feature_windows,
                        optimized_window_index,
                        labels,
                        split_manifest,
                        seed=int(resolved["seed"]),
                        profile=profile,
                        num_workers=num_workers,
                    )
                    baseline_predictions = pd.concat(
                        [baseline_predictions, optimized_result.predictions],
                        ignore_index=True,
                    )
                    baseline_metrics = pd.concat(
                        [baseline_metrics, optimized_result.metrics],
                        ignore_index=True,
                    )
                    training_metadata = pd.concat(
                        [
                            training_metadata
                            if training_metadata is not None
                            else pd.DataFrame(),
                            optimized_result.training_metadata,
                        ],
                        ignore_index=True,
                    )
                ablation_config = dict(resolved.get("tcn_ablations", {}))
                if ablation_config.get("enabled", False):
                    if shared_tcn_result is None:
                        raise ContractError(
                            "tcn_ablations.enabled requires the shared Bai TCN"
                        )
                    from .tcn_lite import run_tcn_ablations

                    ablation_result, tcn_comparison = run_tcn_ablations(
                        feature_windows,
                        window_index,
                        labels,
                        split_manifest,
                        shared_tcn_result.metrics,
                        shared_predictions=shared_tcn_result.predictions,
                        seed=int(resolved["seed"]),
                        channels=int(ablation_config.get("channels", 32)),
                        kernel_size=int(ablation_config.get("kernel_size", 3)),
                        lite_dilations=tuple(
                            ablation_config.get(
                                "lite_dilations", [1, 2, 4, 8, 16, 32, 64, 128]
                            )
                        ),
                        bai_dilations=tuple(
                            ablation_config.get(
                                "bai_dilations", [1, 2, 4, 8, 16, 32, 64]
                            )
                        ),
                        dropout=float(ablation_config.get("dropout", 0.1)),
                        epochs=int(ablation_config.get("epochs", 10)),
                        batch_size=int(ablation_config.get("batch_size", 64)),
                        num_workers=num_workers,
                    )
                    baseline_predictions = pd.concat(
                        [baseline_predictions, ablation_result.predictions],
                        ignore_index=True,
                    )
                    baseline_metrics = pd.concat(
                        [baseline_metrics, ablation_result.metrics],
                        ignore_index=True,
                    )
                    training_metadata = pd.concat(
                        [
                            training_metadata
                            if training_metadata is not None
                            else pd.DataFrame(),
                            ablation_result.training_metadata,
                        ],
                        ignore_index=True,
                    )
                modern_config = dict(resolved.get("moderntcn_experiment", {}))
                if modern_config.get("enabled", False):
                    from .moderntcn import run_moderntcn_experiment

                    modern_result = run_moderntcn_experiment(
                        feature_windows,
                        window_index,
                        labels,
                        split_manifest,
                        seed=int(resolved["seed"]),
                        d_model=int(modern_config.get("d_model", 32)),
                        ffn_ratio=int(modern_config.get("ffn_ratio", 2)),
                        patch_size=int(modern_config.get("patch_size", 16)),
                        patch_stride=int(modern_config.get("patch_stride", 8)),
                        large_kernel_size=int(
                            modern_config.get("large_kernel_size", 31)
                        ),
                        small_kernel_size=int(
                            modern_config.get("small_kernel_size", 5)
                        ),
                        block_count=int(modern_config.get("block_count", 4)),
                        dropout=float(modern_config.get("dropout", 0.1)),
                        epochs=int(modern_config.get("epochs", 10)),
                        batch_size=int(modern_config.get("batch_size", 64)),
                        num_workers=num_workers,
                    )
                    baseline_predictions = pd.concat(
                        [baseline_predictions, modern_result.predictions],
                        ignore_index=True,
                    )
                    baseline_metrics = pd.concat(
                        [baseline_metrics, modern_result.metrics], ignore_index=True
                    )
                    training_metadata = pd.concat(
                        [
                            training_metadata
                            if training_metadata is not None
                            else pd.DataFrame(),
                            modern_result.training_metadata,
                        ],
                        ignore_index=True,
                    )
                from .baselines import _summarize_predictions

                baseline_metrics = _summarize_predictions(
                    baseline_predictions, int(resolved["seed"])
                )
                from .baselines import build_risk_exposure_report

                risk_exposures = build_risk_exposure_report(
                    baseline_predictions, universe
                )
                from .backtest import build_executable_long_only

                backtest_result = build_executable_long_only(
                    baseline_predictions,
                    labels,
                    top_fraction=float(
                        resolved.get("portfolio", {}).get("top_fraction", 0.10)
                    ),
                )
                orders = backtest_result.orders
                vintages = backtest_result.vintages
                portfolio_ledger = backtest_result.portfolio_ledger
                portfolio_metrics = backtest_result.metrics
                diagnostic = backtest_result.diagnostic

                execution_state = _load_optional_parquet(
                    manifest,
                    manifest_file,
                    "execution_state_path",
                    "execution_state_sha256",
                )
                if execution_state is not None:
                    from .execution import simulate_a_share_execution

                    execution_config = dict(resolved.get("execution", {}))
                    cost_schedule = _resolve_cost_schedule(execution_config, orders)
                    trading_rules = execution_config.get("rules")
                    if not isinstance(trading_rules, Mapping):
                        raise ContractError(
                            "execution.rules with versioned T+1 entries is required"
                        )
                    execution_result = simulate_a_share_execution(
                        orders,
                        execution_state,
                        capital=float(execution_config.get("capital", 1_000_000.0)),
                        capacity_fraction=float(
                            execution_config.get("capacity_fraction", 0.05)
                        ),
                        rules=trading_rules,
                        cost_schedule=cost_schedule,
                    )
                    execution_ledger = execution_result.ledger
                    execution_metrics = execution_result.scenario_metrics

                performance_config = dict(resolved.get("performance", {}))
                if performance_config.get("enabled", False):
                    from .performance import benchmark_sequence_models

                    benchmark_models_value = performance_config.get("models")
                    benchmark_models = (
                        tuple(str(value) for value in benchmark_models_value)
                        if isinstance(benchmark_models_value, list)
                        else None
                    )
                    benchmark_features = feature_windows
                    benchmark_index = window_index
                    if (
                        benchmark_models is not None
                        and "optimized-tcn-v40-portable" in benchmark_models
                        and optimized_feature_windows is not None
                        and optimized_window_index is not None
                    ):
                        benchmark_features = optimized_feature_windows
                        benchmark_index = optimized_window_index

                    performance_result = benchmark_sequence_models(
                        benchmark_features,
                        benchmark_index,
                        labels,
                        split_manifest,
                        seed=int(resolved["seed"]),
                        hidden_size=int(performance_config.get("hidden_size", 32)),
                        tcn_channels=int(performance_config.get("tcn_channels", 64)),
                        tcn_kernel_size=int(performance_config.get("kernel_size", 3)),
                        tcn_dilations=tuple(
                            performance_config.get(
                                "dilations", [1, 2, 4, 8, 16, 32, 64]
                            )
                        ),
                        epochs=int(performance_config.get("epochs", 10)),
                        batch_size=int(performance_config.get("batch_size", 64)),
                        device=str(performance_config.get("device", "cpu")),
                        num_workers=num_workers,
                        torch_threads=(
                            int(performance_config["torch_threads"])
                            if "torch_threads" in performance_config
                            else None
                        ),
                        learning_rate=float(
                            performance_config.get("learning_rate", 0.01)
                        ),
                        models=benchmark_models,
                        optimized_tcn_profile=str(
                            performance_config.get(
                                "optimized_tcn_profile", "v40-portable"
                            )
                        ),
                    )
                    performance_metrics = performance_result.measurements
                    performance_environment = performance_result.environment
        latest_date = canonical_bars["trade_date"].max()
        if universe is not None:
            prediction_identities = universe.loc[
                (universe["signal_date"] == latest_date) & universe["eligible"],
                ["instrument_id", "signal_date"],
            ]
        else:
            prediction_identities = (
                canonical_bars.loc[
                    canonical_bars["trade_date"] == latest_date, ["instrument_id"]
                ]
                .drop_duplicates()
                .assign(signal_date=latest_date)
            )
        samples = prediction_identities.copy()

    source_identity = code_identity(Path(__file__).resolve().parents[2])
    identity = {
        "config": resolved,
        "manifest": manifest,
        "data_sha256": manifest["data_sha256"],
        "code": source_identity,
    }
    run_id = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:16]
    engineering_complete = bool(resolved.get("engineering_complete", False))
    if engineering_complete:
        complete_components = {
            "canonical_bars": canonical_bars,
            "data_quality": quality,
            "universe": universe,
            "feature_windows": feature_windows,
            "window_index": window_index,
            "labels": labels,
            "split_manifest": split_manifest,
            "preprocessing": preprocessing,
            "training_metadata": training_metadata,
            "orders": orders,
            "portfolio_ledger": portfolio_ledger,
            "portfolio_metrics": portfolio_metrics,
            "execution_ledger": execution_ledger,
            "execution_metrics": execution_metrics,
            "performance_metrics": performance_metrics,
            "corporate_actions": corporate_actions,
            "window_rejections": window_rejections,
            "risk_exposures": risk_exposures,
        }
        _assert_engineering_complete(
            complete_components,
            baseline_predictions,
            baseline_metrics,
            tcn_comparison,
            split_manifest,
        )
    root = Path(output_root).resolve()
    run_dir = root / run_id
    if run_dir.exists():
        raise ContractError(f"run already exists and will not be overwritten: {run_id}")

    if baseline_predictions is not None and baseline_metrics is not None:
        predictions_frame = baseline_predictions.copy()
        predictions_frame.insert(0, "run_id", run_id)
        metrics_frame = baseline_metrics.copy()
        metrics_frame.insert(0, "run_id", run_id)
    else:
        predictions = []
        metrics = []
        for horizon in horizons:
            target_column = f"target_{horizon}d"
            if target_column in samples:
                target = pd.to_numeric(samples[target_column], errors="coerce")
            else:
                target = pd.Series(float("nan"), index=samples.index, dtype="float64")
            for identity_row, target_value in zip(
                prediction_identities.to_dict("records"),
                target,
                strict=True,
            ):
                predictions.append(
                    {
                        "run_id": run_id,
                        **identity_row,
                        "horizon": horizon,
                        "score": 0.0,
                        "target": target_value,
                        "model": "constant-zero",
                    }
                )
            metrics.append(
                {
                    "run_id": run_id,
                    "model": "constant-zero",
                    "horizon": horizon,
                    "metric": "mean_absolute_error",
                    "value": float(target.abs().mean())
                    if target.notna().any()
                    else float("nan"),
                }
            )
        predictions_frame = pd.DataFrame(predictions)
        metrics_frame = pd.DataFrame(metrics)

    def add_run_identity(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None:
            return None
        scoped = frame.copy()
        if "run_id" in scoped:
            scoped["run_id"] = run_id
        else:
            scoped.insert(0, "run_id", run_id)
        return scoped

    orders = add_run_identity(orders)
    vintages = add_run_identity(vintages)
    portfolio_metrics = add_run_identity(portfolio_metrics)
    portfolio_ledger = add_run_identity(portfolio_ledger)
    diagnostic = add_run_identity(diagnostic)
    execution_ledger = add_run_identity(execution_ledger)
    execution_metrics = add_run_identity(execution_metrics)
    performance_metrics = add_run_identity(performance_metrics)
    training_metadata = add_run_identity(training_metadata)
    risk_exposures = add_run_identity(risk_exposures)

    root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    resolved_path = run_dir / "config.resolved.json"
    predictions_path = run_dir / "predictions.parquet"
    metrics_path = run_dir / "metrics.parquet"
    report_path = run_dir / "report.md"
    run_manifest_path = run_dir / "run.json"
    input_manifest_path = run_dir / "input-manifest.json"
    canonical_bars_path = (
        run_dir / "bars_5m.parquet" if canonical_bars is not None else None
    )
    quality_path = run_dir / "data-quality.json" if quality is not None else None
    universe_path = run_dir / "universe.parquet" if universe is not None else None
    window_index_path = (
        run_dir / "window-index.parquet" if window_index is not None else None
    )
    windows_path = (
        run_dir / "feature-windows.npz" if feature_windows is not None else None
    )
    labels_path = run_dir / "labels.parquet" if labels is not None else None
    split_manifest_path = (
        run_dir / "split-manifest.parquet" if split_manifest is not None else None
    )
    preprocessing_path = (
        run_dir / "preprocessing.json" if preprocessing is not None else None
    )
    training_metadata_path = (
        run_dir / "training-metadata.parquet" if training_metadata is not None else None
    )
    window_rejections_path = (
        run_dir / "window-rejections.parquet" if window_rejections is not None else None
    )
    tcn_comparison_path = (
        run_dir / "tcn-comparison.parquet" if tcn_comparison is not None else None
    )
    risk_exposures_path = (
        run_dir / "risk-exposures.parquet" if risk_exposures is not None else None
    )
    orders_path = run_dir / "orders.parquet" if orders is not None else None
    vintages_path = run_dir / "vintages.parquet" if vintages is not None else None
    portfolio_metrics_path = (
        run_dir / "portfolio-metrics.parquet" if portfolio_metrics is not None else None
    )
    portfolio_ledger_path = (
        run_dir / "portfolio-ledger.parquet" if portfolio_ledger is not None else None
    )
    diagnostic_path = (
        run_dir / "diagnostic-long-short.parquet" if diagnostic is not None else None
    )
    execution_ledger_path = (
        run_dir / "execution-ledger.parquet" if execution_ledger is not None else None
    )
    execution_metrics_path = (
        run_dir / "execution-metrics.parquet" if execution_metrics is not None else None
    )
    performance_metrics_path = (
        run_dir / "performance-metrics.parquet"
        if performance_metrics is not None
        else None
    )
    performance_environment_path = (
        run_dir / "performance-environment.json"
        if performance_environment is not None
        else None
    )

    resolved_path.write_text(_canonical_json(resolved) + "\n", encoding="utf-8")
    input_manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    predictions_frame.to_parquet(predictions_path, index=False)
    metrics_frame.to_parquet(metrics_path, index=False)
    if (
        canonical_bars_path is not None
        and canonical_bars is not None
        and quality_path is not None
        and quality is not None
    ):
        canonical_bars.to_parquet(canonical_bars_path, index=False)
        quality_path.write_text(_canonical_json(quality) + "\n", encoding="utf-8")
    if universe_path is not None and universe is not None:
        universe.to_parquet(universe_path, index=False)
    if window_index_path is not None and window_index is not None:
        window_index.to_parquet(window_index_path, index=False)
    if windows_path is not None and feature_windows is not None:
        np.savez_compressed(
            windows_path, features=feature_windows, feature_names=FEATURE_NAMES
        )
    if labels_path is not None and labels is not None:
        labels.to_parquet(labels_path, index=False)
    if split_manifest_path is not None and split_manifest is not None:
        split_manifest.to_parquet(split_manifest_path, index=False)
    if preprocessing_path is not None and preprocessing is not None:
        preprocessing_path.write_text(
            _canonical_json(preprocessing) + "\n", encoding="utf-8"
        )
    if training_metadata_path is not None and training_metadata is not None:
        training_metadata.to_parquet(training_metadata_path, index=False)
    if window_rejections_path is not None and window_rejections is not None:
        window_rejections.to_parquet(window_rejections_path, index=False)
    if tcn_comparison_path is not None and tcn_comparison is not None:
        scoped_tcn_comparison = add_run_identity(tcn_comparison)
        assert scoped_tcn_comparison is not None
        scoped_tcn_comparison.to_parquet(tcn_comparison_path, index=False)
        tcn_comparison = scoped_tcn_comparison
    if risk_exposures_path is not None and risk_exposures is not None:
        risk_exposures.to_parquet(risk_exposures_path, index=False)
    if orders_path is not None and orders is not None:
        orders.to_parquet(orders_path, index=False)
    if vintages_path is not None and vintages is not None:
        vintages.to_parquet(vintages_path, index=False)
    if portfolio_metrics_path is not None and portfolio_metrics is not None:
        portfolio_metrics.to_parquet(portfolio_metrics_path, index=False)
    if portfolio_ledger_path is not None and portfolio_ledger is not None:
        portfolio_ledger.to_parquet(portfolio_ledger_path, index=False)
    if diagnostic_path is not None and diagnostic is not None:
        diagnostic.to_parquet(diagnostic_path, index=False)
    if execution_ledger_path is not None and execution_ledger is not None:
        execution_ledger.to_parquet(execution_ledger_path, index=False)
    if execution_metrics_path is not None and execution_metrics is not None:
        execution_metrics.to_parquet(execution_metrics_path, index=False)
    if performance_metrics_path is not None and performance_metrics is not None:
        performance_metrics.to_parquet(performance_metrics_path, index=False)
    if performance_environment_path is not None and performance_environment is not None:
        performance_environment_path.write_text(
            _canonical_json(performance_environment) + "\n", encoding="utf-8"
        )
    cache = None
    if feature_windows is not None and window_index is not None:
        from .streaming import write_window_cache

        cache = write_window_cache(
            run_dir,
            feature_windows,
            window_index,
            source_fingerprint=manifest["data_sha256"],
            feature_version="causal-basic-v1",
        )
    data_summary = ""
    if quality is not None:
        data_summary = (
            f"- 标准分钟条：{quality['canonical_bar_count']}\n"
            f"- 完整交易日：{quality['complete_session_count']}\n"
        )
    from .evidence import render_research_report, write_evidence_bundle

    structured_report = render_research_report(
        engineering_complete=engineering_complete,
        metrics=metrics_frame,
        portfolio_metrics=(
            portfolio_metrics if portfolio_metrics is not None else pd.DataFrame()
        ),
        execution_metrics=(
            execution_metrics if execution_metrics is not None else pd.DataFrame()
        ),
        performance_metrics=(
            performance_metrics if performance_metrics is not None else pd.DataFrame()
        ),
        execution_ledger=(
            execution_ledger if execution_ledger is not None else pd.DataFrame()
        ),
        risk_exposures=(
            risk_exposures if risk_exposures is not None else pd.DataFrame()
        ),
        data_quality=quality,
        canonical_bars=canonical_bars,
        universe=universe,
        window_index=window_index,
        window_rejections=window_rejections,
    )
    minimal_summary = ""
    if not engineering_complete:
        minimal_summary = (
            "- 工程状态：完成最小离线运行\n"
            "- Alpha 证据：未评估\n"
            "- 速度假设：未评估\n" + data_summary + "\n"
        )
    report_path.write_text(
        "# 最小离线实验报告\n\n" + minimal_summary + structured_report,
        encoding="utf-8",
    )
    evidence_artifacts: dict[str, Path] = {
        "config": resolved_path,
        "input_manifest": input_manifest_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "report": report_path,
    }
    optional_evidence = {
        "canonical_bars": canonical_bars_path,
        "data_quality": quality_path,
        "universe": universe_path,
        "window_index": window_index_path,
        "feature_windows": windows_path,
        "labels": labels_path,
        "split_manifest": split_manifest_path,
        "preprocessing": preprocessing_path,
        "training_metadata": training_metadata_path,
        "window_rejections": window_rejections_path,
        "tcn_comparison": tcn_comparison_path,
        "risk_exposures": risk_exposures_path,
        "orders": orders_path,
        "vintages": vintages_path,
        "portfolio_metrics": portfolio_metrics_path,
        "portfolio_ledger": portfolio_ledger_path,
        "diagnostic": diagnostic_path,
        "execution_ledger": execution_ledger_path,
        "execution_metrics": execution_metrics_path,
        "performance_metrics": performance_metrics_path,
        "performance_environment": performance_environment_path,
        "feature_memmap": cache.data_path if cache is not None else None,
        "window_cache": cache.manifest_path if cache is not None else None,
    }
    evidence_artifacts.update(
        {name: path for name, path in optional_evidence.items() if path is not None}
    )
    complete_required = {
        "config",
        "input_manifest",
        "predictions",
        "metrics",
        "report",
        "canonical_bars",
        "data_quality",
        "universe",
        "window_index",
        "feature_windows",
        "labels",
        "split_manifest",
        "preprocessing",
        "training_metadata",
        "orders",
        "vintages",
        "portfolio_metrics",
        "portfolio_ledger",
        "diagnostic",
        "execution_ledger",
        "execution_metrics",
        "performance_metrics",
        "performance_environment",
        "feature_memmap",
        "window_cache",
        "window_rejections",
        "tcn_comparison",
        "risk_exposures",
    }
    bundle = write_evidence_bundle(
        run_dir,
        run_id=run_id,
        config=resolved,
        seed=int(resolved["seed"]),
        artifacts=evidence_artifacts,
        required_artifacts=complete_required if engineering_complete else set(),
        engineering_complete=engineering_complete,
    )
    model_ids = sorted(predictions_frame["model"].astype(str).unique().tolist())
    model_label = model_ids[0] if len(model_ids) == 1 else "multi-model-comparison"
    run_manifest = {
        "run_id": run_id,
        "status": "engineering_complete" if engineering_complete else "success",
        "model": model_label,
        "models": model_ids,
        "data_sha256": manifest["data_sha256"],
        "code": source_identity,
        "config_sha256": hashlib.sha256(
            _canonical_json(resolved).encode("utf-8")
        ).hexdigest(),
        "artifacts": {
            "config": resolved_path.name,
            "input_manifest": input_manifest_path.name,
            "predictions": predictions_path.name,
            "metrics": metrics_path.name,
            "report": report_path.name,
            "evidence_index": bundle.index_path.name,
            "environment": bundle.environment_path.name,
        },
    }
    if canonical_bars_path is not None and quality_path is not None:
        run_manifest["artifacts"]["canonical_bars"] = canonical_bars_path.name
        run_manifest["artifacts"]["data_quality"] = quality_path.name
    if universe_path is not None:
        run_manifest["artifacts"]["universe"] = universe_path.name
    if window_index_path is not None and windows_path is not None:
        run_manifest["artifacts"]["window_index"] = window_index_path.name
        run_manifest["artifacts"]["feature_windows"] = windows_path.name
    if labels_path is not None:
        run_manifest["artifacts"]["labels"] = labels_path.name
    if split_manifest_path is not None and preprocessing_path is not None:
        run_manifest["artifacts"]["split_manifest"] = split_manifest_path.name
        run_manifest["artifacts"]["preprocessing"] = preprocessing_path.name
    if cache is not None:
        run_manifest["artifacts"]["feature_memmap"] = cache.data_path.name
        run_manifest["artifacts"]["window_cache"] = cache.manifest_path.name
    if training_metadata_path is not None:
        run_manifest["artifacts"]["training_metadata"] = training_metadata_path.name
    if window_rejections_path is not None:
        run_manifest["artifacts"]["window_rejections"] = window_rejections_path.name
    if tcn_comparison_path is not None:
        run_manifest["artifacts"]["tcn_comparison"] = tcn_comparison_path.name
    if risk_exposures_path is not None:
        run_manifest["artifacts"]["risk_exposures"] = risk_exposures_path.name
    for name, path in {
        "orders": orders_path,
        "vintages": vintages_path,
        "portfolio_metrics": portfolio_metrics_path,
        "portfolio_ledger": portfolio_ledger_path,
        "diagnostic": diagnostic_path,
        "execution_ledger": execution_ledger_path,
        "execution_metrics": execution_metrics_path,
        "performance_metrics": performance_metrics_path,
        "performance_environment": performance_environment_path,
    }.items():
        if path is not None:
            run_manifest["artifacts"][name] = path.name
    run_manifest_path.write_text(_canonical_json(run_manifest) + "\n", encoding="utf-8")

    result = RunResult(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=run_manifest_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        report_path=report_path,
        canonical_bars_path=canonical_bars_path,
        quality_path=quality_path,
        universe_path=universe_path,
        window_index_path=window_index_path,
        windows_path=windows_path,
        labels_path=labels_path,
        split_manifest_path=split_manifest_path,
        preprocessing_path=preprocessing_path,
        memmap_path=cache.data_path if cache is not None else None,
        window_cache_manifest_path=cache.manifest_path if cache is not None else None,
        training_metadata_path=training_metadata_path,
        orders_path=orders_path,
        vintages_path=vintages_path,
        portfolio_metrics_path=portfolio_metrics_path,
        portfolio_ledger_path=portfolio_ledger_path,
        diagnostic_path=diagnostic_path,
        execution_ledger_path=execution_ledger_path,
        execution_metrics_path=execution_metrics_path,
        performance_metrics_path=performance_metrics_path,
        performance_environment_path=performance_environment_path,
        evidence_index_path=bundle.index_path,
        environment_path=bundle.environment_path,
        window_rejections_path=window_rejections_path,
        tcn_comparison_path=tcn_comparison_path,
        risk_exposures_path=risk_exposures_path,
    )
    if temporary_window_store is not None:
        temporary_window_store.close()
    if temporary_relative_store is not None:
        temporary_relative_store.close()
    return result
