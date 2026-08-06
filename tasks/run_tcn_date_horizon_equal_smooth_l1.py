"""Run the immutable v32 equal-date/equal-horizon SmoothL1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_dl_tcn_shortterm import ContractError  # noqa: E402
from skill_dl_tcn_shortterm.batch_stability import (  # noqa: E402
    collect_frozen_shape_daily_rankic,
    pair_daily_rankic,
)
from skill_dl_tcn_shortterm.integrity import code_identity  # noqa: E402
from skill_dl_tcn_shortterm.loss_alignment import (  # noqa: E402
    evaluate_date_horizon_equal_smooth_l1,
)
from skill_dl_tcn_shortterm.real_validation import (  # noqa: E402
    build_tcn_lstm_comparison,
    parse_real_tcn_trials,
)
from skill_dl_tcn_shortterm.tuning import run_tcn_validation_sweep  # noqa: E402
from skill_dl_tcn_shortterm.v9_receipts import canonical_bytes  # noqa: E402

from run_tcn_dynamic_skip_learning_rate import (  # noqa: E402
    _historical_evidence,
    _load_parent,
)
from run_tcn_frozen_parent_shape_residual import (  # noqa: E402
    _frozen_shape_diagnostics,
    _load_frozen_parent_states,
)
from run_tcn_frozen_shape_soft_rankic import _validate_common_trial  # noqa: E402
from run_tcn_multiseed_confirmation import (  # noqa: E402
    _contains_secret_key,
    _sha256,
    _write_json,
)


def _paired_units(
    leaderboard: pd.DataFrame,
    v31_control: pd.DataFrame,
    *,
    control_trial_id: str,
    candidate_trial_id: str,
) -> pd.DataFrame:
    key = ["seed", "fold"]
    columns = [
        "best_epoch",
        "best_mean_daily_rankic",
        "completed_epochs",
        "median_epoch_gradient_norm_cv",
        "median_labels_per_loss_group",
        "samples_per_second",
    ]

    def _side(trial_id: str, prefix: str) -> pd.DataFrame:
        return leaderboard.loc[
            leaderboard["trial_id"].astype(str).eq(trial_id), key + columns
        ].rename(columns={column: f"{prefix}_{column}" for column in columns})

    paired = _side(control_trial_id, "control").merge(
        _side(candidate_trial_id, "candidate"), on=key, validate="one_to_one"
    )
    historical = v31_control[key + ["best_mean_daily_rankic"]].rename(
        columns={"best_mean_daily_rankic": "v31_best_mean_daily_rankic"}
    )
    paired = paired.merge(historical, on=key, validate="one_to_one")
    if len(paired) != 15:
        raise ContractError("v32 paired unit coverage drifted")
    paired["candidate_control_rankic_delta"] = (
        paired["candidate_best_mean_daily_rankic"]
        - paired["control_best_mean_daily_rankic"]
    )
    paired["control_v31_replay_delta"] = (
        paired["control_best_mean_daily_rankic"]
        - paired["v31_best_mean_daily_rankic"]
    )
    return paired.sort_values(key, ignore_index=True)


def _report(
    selection: dict[str, object],
    seed_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    aggregate = cast(dict[str, object], selection["aggregate"])
    seed_lines = [
        (
            f"| {int(cast(Any, row.seed))} | "
            f"{float(cast(Any, row.control_mean_rankic)):.6f} | "
            f"{float(cast(Any, row.candidate_mean_rankic)):.6f} | "
            f"{float(cast(Any, row.rankic_delta)):+.6f} |"
        )
        for row in seed_summary.itertuples(index=False)
    ]
    bootstrap_lines = [
        (
            f"| {row.scope} | "
            f"{float(cast(Any, row.paired_mean_delta)):+.6f} | "
            f"{float(cast(Any, row.bootstrap_ci_low)):+.6f} | "
            f"{float(cast(Any, row.bootstrap_ci_high)):+.6f} |"
        )
        for row in bootstrap.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# TCN date/horizon equal SmoothL1 v32",
            "",
            f"- 决策：`{selection['status']}`",
            f"- 完整性门：`{selection['integrity_passed']}`",
            f"- loss 机制门：`{selection['mechanism_passed']}`",
            f"- 预测效果门：`{selection['effect_passed']}`",
            f"- 速度门：`{selection['speed_passed']}`",
            "- sealed test：未访问、未授权。",
            "",
            "## 结论",
            "",
            "本轮唯一变量是把 grouped SmoothL1 从逐标签均值改为先按"
            " `(signal_date,horizon)` 求组内均值、再对组等权。TCN、父 checkpoint、"
            "日期顺序和训练预算均保持不变。",
            "",
            f"- candidate mean RankIC：`{float(cast(Any, aggregate['candidate_mean_rankic'])):.6f}`",
            f"- control mean RankIC：`{float(cast(Any, aggregate['control_mean_rankic'])):.6f}`",
            f"- paired mean delta：`{float(cast(Any, aggregate['mean_rankic_delta'])):+.6f}`",
            f"- trained units：`{aggregate['control_trained_effect_units']} -> {aggregate['candidate_trained_effect_units']}`",
            f"- median labels/loss-group：`{float(cast(Any, aggregate['candidate_median_labels_per_loss_group'])):.3f}`",
            f"- model-step TCN/LSTM：`{float(cast(Any, aggregate['model_step_speed_ratio'])):.3f}x`",
            f"- end-to-end TCN/LSTM：`{float(cast(Any, aggregate['end_to_end_speed_ratio'])):.3f}x`",
            f"- candidate/control throughput：`{float(cast(Any, aggregate['candidate_control_throughput_ratio'])):.3f}x`",
            f"- blockers：`{aggregate['blockers'] or 'none'}`",
            "",
            "## Seed 配对结果",
            "",
            "| seed | control | candidate | delta |",
            "|---:|---:|---:|---:|",
            *seed_lines,
            "",
            "## 日期块 bootstrap",
            "",
            "| scope | paired delta | 95% CI low | 95% CI high |",
            "|---|---:|---:|---:|",
            *bootstrap_lines,
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the immutable v32 date/horizon equal SmoothL1 probe"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    try:
        if output_dir.exists() or temporary.exists():
            raise ContractError("v32 refuses to overwrite experiment artifacts")
        config_path = arguments.config.resolve()
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ContractError("v32 config must contain an object")
        config = cast(dict[str, object], config_value)
        if config.get("protocol_version") != "v32":
            raise ContractError("v32 protocol identity drifted")
        if _contains_secret_key(config):
            raise ContractError("v32 config contains a secret-like key")
        if config.get("precision") != "float32" or int(
            cast(Any, config["num_workers"])
        ) != 0:
            raise ContractError("v32 requires float32 and num_workers=0")
        seeds = tuple(
            int(cast(Any, value))
            for value in cast(list[object], config["seeds"])
        )
        if seeds != (7, 17, 27) or cast(list[object], config["folds"]) != [
            0,
            1,
            2,
            3,
            4,
        ]:
            raise ContractError("v32 requires seeds 7/17/27 and folds 0..4")
        checkpoint_min_delta = float(cast(Any, config["checkpoint_min_delta"]))
        patience_min_delta = float(cast(Any, config["min_delta"]))
        if checkpoint_min_delta != 0.0 or patience_min_delta != 0.0005:
            raise ContractError("v32 checkpoint selection contract drifted")

        expected_hashes_value = config.get("source_sha256")
        if not isinstance(expected_hashes_value, dict):
            raise ContractError("v32 source identities are missing")
        expected_hashes = {
            str(key): str(value) for key, value in expected_hashes_value.items()
        }
        seed7_parent, seed7_identity = _load_parent(
            config, prefix="seed7", expected_source_hashes=expected_hashes
        )
        confirmation_parent, confirmation_identity = _load_parent(
            config, prefix="confirmation", expected_source_hashes=expected_hashes
        )
        v31_parent, v31_identity = _load_parent(
            config, prefix="v31", expected_source_hashes=expected_hashes
        )

        run_dir = arguments.run_dir.resolve()
        source_paths = {
            "features": run_dir / "feature-windows.npy",
            "window_index": run_dir / "window-index.parquet",
            "labels": run_dir / "labels.parquet",
            "split_manifest": arguments.split_manifest.resolve(),
            "universe": run_dir / "universe.parquet",
            "input_manifest": run_dir / "input-manifest.json",
        }
        if missing := [name for name, path in source_paths.items() if not path.is_file()]:
            raise ContractError("v32 sources missing: " + ", ".join(missing))
        observed_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if observed_hashes != expected_hashes:
            raise ContractError("v32 source SHA-256 identity drifted")
        features = np.load(source_paths["features"], mmap_mode="r", allow_pickle=False)
        window_index = pd.read_parquet(source_paths["window_index"])
        labels = pd.read_parquet(source_paths["labels"])
        raw_split = pd.read_parquet(source_paths["split_manifest"])
        if "sealed" not in raw_split or raw_split["sealed"].astype(bool).any():
            raise ContractError("v32 rejects sealed split rows")
        observed_stages = {str(value) for value in raw_split["stage"].tolist()}
        if unknown := sorted(
            observed_stages - {"train", "validation", "purged"}
        ):
            raise ContractError(
                "v32 split contains forbidden stages: " + ", ".join(unknown)
            )
        split_manifest = raw_split.loc[
            raw_split["fold"].astype(int).isin(range(5))
            & raw_split["stage"].isin(["train", "validation"])
        ].copy()

        trials = parse_real_tcn_trials(config["trials"])
        control_trial_id = str(config["control_trial_id"])
        candidate_trial_id = str(config["candidate_trial_id"])
        if {trial.trial_id for trial in trials} != {
            control_trial_id,
            candidate_trial_id,
        }:
            raise ContractError("v32 must train exactly the registered pair")
        trials_by_id = {trial.trial_id: trial for trial in trials}
        control = trials_by_id[control_trial_id]
        candidate = trials_by_id[candidate_trial_id]
        if not _validate_common_trial(control) or not _validate_common_trial(candidate):
            raise ContractError("v32 frozen TCN common contract drifted")
        if (
            control.strategy != "grouped_smooth_l1"
            or candidate.strategy != "grouped_smooth_l1"
            or control.date_batch_order != "fixed_once"
            or candidate.date_batch_order != "fixed_once"
            or control.grouped_smooth_l1_reduction != "label_mean"
            or candidate.grouped_smooth_l1_reduction != "date_horizon_mean"
        ):
            raise ContractError("v32 loss reduction identity drifted")
        control_contract = dict(control.__dict__)
        candidate_contract = dict(candidate.__dict__)
        for contract in (control_contract, candidate_contract):
            contract.pop("trial_id")
            contract.pop("grouped_smooth_l1_reduction")
        if control_contract != candidate_contract:
            raise ContractError("v32 trials differ by more than loss reduction")

        control_states, control_manifest = _load_frozen_parent_states(
            seed7_parent, confirmation_parent, control_trial_id
        )
        candidate_states, candidate_manifest = _load_frozen_parent_states(
            seed7_parent, confirmation_parent, candidate_trial_id
        )
        control_manifest.insert(0, "trial_id", control_trial_id)
        candidate_manifest.insert(0, "trial_id", candidate_trial_id)
        checkpoint_manifest = pd.concat(
            [control_manifest, candidate_manifest], ignore_index=True
        )
        frozen_states = {
            seed: {**control_states[seed], **candidate_states[seed]}
            for seed in seeds
        }
        identities = {
            "data": observed_hashes["features"],
            "fold_manifest": observed_hashes["split_manifest"],
            "evaluation": observed_hashes["labels"],
        }
        tuning_parts = []
        best_states: dict[str, dict[str, torch.Tensor]] = {}
        for seed in seeds:
            tuning = run_tcn_validation_sweep(
                features,
                window_index,
                labels,
                split_manifest,
                trials=trials,
                seed=seed,
                max_epochs=int(cast(Any, config["max_epochs"])),
                patience=int(cast(Any, config["patience"])),
                min_delta=patience_min_delta,
                checkpoint_min_delta=checkpoint_min_delta,
                torch_threads=int(cast(Any, config["torch_threads"])),
                protocol_identities=identities,
                frozen_parent_states=frozen_states[seed],
            )
            tuning_parts.append(tuning)
            for key, state in tuning.best_states.items():
                best_states[f"seed-{seed}-{key}"] = state
        epoch_history = pd.concat(
            [part.epoch_history for part in tuning_parts], ignore_index=True
        )
        leaderboard = pd.concat(
            [part.leaderboard for part in tuning_parts], ignore_index=True
        ).merge(
            checkpoint_manifest[
                ["trial_id", "seed", "fold", "parent_checkpoint_sha256"]
            ],
            on=["trial_id", "seed", "fold"],
            how="left",
            validate="one_to_one",
        )
        diagnostics = pd.concat(
            [
                _frozen_shape_diagnostics(
                    features,
                    window_index,
                    labels,
                    split_manifest,
                    trial,
                    best_states,
                    seeds=seeds,
                    batch_size=trial.batch_size,
                )
                for trial in (control, candidate)
            ],
            ignore_index=True,
        )
        leaderboard_scores = leaderboard.set_index(["trial_id", "seed", "fold"])[
            "best_mean_daily_rankic"
        ].sort_index()
        diagnostic_scores = diagnostics.set_index(["trial_id", "seed", "fold"])[
            "full_mean_daily_rankic"
        ].sort_index()
        if float(np.max(np.abs(diagnostic_scores - leaderboard_scores))) > 1e-12:
            raise ContractError("v32 diagnostics drifted from leaderboard")

        daily = collect_frozen_shape_daily_rankic(
            features,
            window_index,
            labels,
            split_manifest,
            (control, candidate),
            best_states,
            seeds=seeds,
        )
        paired_daily = pair_daily_rankic(
            daily,
            control_trial_id=control_trial_id,
            candidate_trial_id=candidate_trial_id,
        )
        daily_means = daily.groupby(["trial_id", "seed", "fold"], observed=True)[
            "rankic"
        ].mean().sort_index()
        if float(np.max(np.abs(daily_means - leaderboard_scores))) > 1e-12:
            raise ContractError("v32 daily RankIC drifted from leaderboard")

        _, _, lstm, lstm_environment = _historical_evidence(
            seed7_parent,
            confirmation_parent,
            control_trial_id=str(config["historical_control_trial_id"]),
            parent_candidate_trial_id=str(config["historical_parent_trial_id"]),
        )
        candidate_rows = leaderboard.loc[
            leaderboard["trial_id"].astype(str).eq(candidate_trial_id)
        ].copy()
        comparison = build_tcn_lstm_comparison(candidate_rows, lstm)
        v31_leaderboard = pd.read_parquet(v31_parent / "tcn-leaderboard.parquet")
        v31_control = v31_leaderboard.loc[
            v31_leaderboard["trial_id"].astype(str).eq(
                str(config["v31_control_trial_id"])
            )
        ].copy()
        paired_units = _paired_units(
            leaderboard,
            v31_control,
            control_trial_id=control_trial_id,
            candidate_trial_id=candidate_trial_id,
        )
        replay_error = float(paired_units["control_v31_replay_delta"].abs().max())
        if replay_error > 1e-12:
            raise ContractError(
                f"v32 label-mean control failed v31 replay: {replay_error:.12g}"
            )

        gates = cast(dict[str, object], config["gates"])
        decision = evaluate_date_horizon_equal_smooth_l1(
            leaderboard,
            epoch_history,
            paired_daily,
            comparison,
            control_trial_id=control_trial_id,
            candidate_trial_id=candidate_trial_id,
            expected_seeds=seeds,
            min_mean_rankic_delta=float(cast(Any, gates["min_mean_rankic_delta"])),
            min_nondegrading_units=int(
                cast(Any, gates["min_nondegrading_units"])
            ),
            max_unit_degradation=float(cast(Any, gates["max_unit_degradation"])),
            min_candidate_mean_rankic=float(
                cast(Any, gates["min_candidate_mean_rankic"])
            ),
            min_throughput_ratio=float(cast(Any, gates["min_throughput_ratio"])),
            min_model_step_speed_ratio=float(
                cast(Any, gates["min_model_step_speed_ratio"])
            ),
            min_end_to_end_speed_ratio=float(
                cast(Any, gates["min_end_to_end_speed_ratio"])
            ),
            bootstrap_seed=int(cast(Any, config["bootstrap_seed"])),
            bootstrap_draws=int(cast(Any, config["bootstrap_draws"])),
        )
        selection: dict[str, object] = {
            "status": decision.status,
            "integrity_passed": decision.integrity_passed,
            "mechanism_passed": decision.mechanism_passed,
            "effect_passed": decision.effect_passed,
            "speed_passed": decision.speed_passed,
            "candidate_trial_id": candidate_trial_id,
            "control_trial_id": control_trial_id,
            "aggregate": decision.aggregate,
            "v31_control_replay_max_abs_error": replay_error,
            "sealed_test_authorized": False,
        }
        loss_diagnostics = epoch_history.loc[
            epoch_history["stage"].astype(str).eq("validation"),
            [
                "trial_id",
                "seed",
                "fold",
                "epoch",
                "loss_group_count_mean",
                "loss_group_count_min",
                "loss_group_count_max",
                "valid_label_count_mean",
                "valid_label_count_min",
                "valid_label_count_max",
                "labels_per_loss_group_mean",
                "gradient_norm_mean",
                "gradient_norm_std",
                "gradient_norm_cv",
                "gradient_norm_max",
                "optimizer_step_count",
                "date_order_fingerprint",
            ],
        ].copy()

        temporary.mkdir(parents=True)
        epoch_history.to_parquet(temporary / "tcn-epoch-history.parquet", index=False)
        leaderboard.to_parquet(temporary / "tcn-leaderboard.parquet", index=False)
        loss_diagnostics.to_parquet(
            temporary / "loss-reduction-diagnostics.parquet", index=False
        )
        daily.to_parquet(temporary / "daily-rankic-long.parquet", index=False)
        paired_daily.to_parquet(
            temporary / "paired-rankic-by-date.parquet", index=False
        )
        decision.bootstrap_summary.to_parquet(
            temporary / "bootstrap-summary.parquet", index=False
        )
        paired_units.to_parquet(
            temporary / "paired-unit-comparison.parquet", index=False
        )
        diagnostics.to_parquet(temporary / "shape-diagnostics.parquet", index=False)
        checkpoint_manifest.to_parquet(
            temporary / "parent-checkpoint-manifest.parquet", index=False
        )
        decision.seed_summary.to_parquet(
            temporary / "seed-summary.parquet", index=False
        )
        lstm.to_parquet(temporary / "lstm-measurements.parquet", index=False)
        _write_json(temporary / "lstm-environment.json", lstm_environment)
        _write_json(temporary / "comparison.json", comparison)
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "config.resolved.json", config)
        (temporary / "report.md").write_text(
            _report(selection, decision.seed_summary, decision.bootstrap_summary),
            encoding="utf-8",
        )
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        for checkpoint_key, state in best_states.items():
            torch.save(state, checkpoint_dir / f"{checkpoint_key}.pt")
        outputs = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        receipt: dict[str, Any] = {
            "schema_version": "tcn-date-horizon-equal-smooth-l1-v32/v1",
            "run_id": str(config["run_id"]),
            "parents": {
                "seed7": seed7_identity,
                "confirmation": confirmation_identity,
                "v31": v31_identity,
            },
            "source_artifacts": {
                name: {"path": str(path), "sha256": observed_hashes[name]}
                for name, path in source_paths.items()
            },
            "source_config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "code_identity": code_identity(ROOT),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_threads": int(cast(Any, config["torch_threads"])),
                "precision": "float32",
            },
            "selection": selection,
            "comparison": comparison,
            "outputs": outputs,
            "sealed_test_accessed": False,
        }
        receipt["receipt_id"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        _write_json(temporary / "receipt.json", receipt)
        temporary.replace(output_dir)
        payload: dict[str, object] = {
            "status": "success",
            "result": decision.status,
            "output_dir": str(output_dir),
            "receipt_id": receipt["receipt_id"],
        }
    except (
        ContractError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
