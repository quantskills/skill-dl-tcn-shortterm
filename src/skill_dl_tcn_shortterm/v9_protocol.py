"""Governed TCN-v9 plan validation and immutable plan receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Literal, Mapping, cast

import pandas as pd

from .experiment import ContractError
from .integrity import code_identity
from .v9_receipts import canonical_bytes, canonicalize, publish_immutable_receipt


V9Stage = Literal["diagnostic", "formal_screen", "multi_seed_confirmation"]
V9TrialKind = Literal["horizon_skip", "rank_objective", "pcgrad"]
V9TriggerStatus = Literal["pending", "applicable", "not_applicable"]

_STAGES = {"diagnostic", "formal_screen", "multi_seed_confirmation"}
_TRIAL_KINDS = {"horizon_skip", "rank_objective", "pcgrad"}
_TRIGGER_STATUSES = {"pending", "applicable", "not_applicable"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class V9TrialSpec:
    """One pre-registered conditional v9 candidate slot."""

    kind: V9TrialKind
    trial_id: str
    trigger_status: V9TriggerStatus


@dataclass(frozen=True)
class V9Plan:
    """Immutable plan submitted to the ordinary-validation seam."""

    run_id: str
    stage: V9Stage
    seeds: tuple[int, ...]
    fold_ids: tuple[int, ...]
    torch_threads: int
    precision: str
    model_family: str
    input_steps: int
    max_epochs: int
    patience: int
    min_delta: float
    control_trial_id: str
    channels: int
    kernel_size: int
    dilations: tuple[int, ...]
    dropout: float
    learning_rate: float
    batch_size: int
    trials: tuple[V9TrialSpec, ...]
    candidate_order: tuple[str, ...]
    source_identities: Mapping[str, str]
    rankic_candidate_model_id: str = "historical-tcn-candidate"


def parse_v9_plan(config: Mapping[str, object]) -> V9Plan:
    """Parse a fail-closed JSON-compatible public v9 configuration."""

    expected = {
        "run_id",
        "stage",
        "seeds",
        "fold_ids",
        "torch_threads",
        "precision",
        "model_family",
        "input_steps",
        "max_epochs",
        "patience",
        "min_delta",
        "control_trial_id",
        "channels",
        "kernel_size",
        "dilations",
        "dropout",
        "learning_rate",
        "batch_size",
        "trials",
        "candidate_order",
        "source_identities",
        "rankic_candidate_model_id",
    }
    if set(config) != expected:
        missing = sorted(expected.difference(config))
        unknown = sorted(set(config).difference(expected))
        raise ContractError(
            f"v9 config keys mismatch; missing={missing}, unknown={unknown}"
        )
    raw_trials = config["trials"]
    if not isinstance(raw_trials, list):
        raise ContractError("v9 config trials must be a list")
    trials = []
    for raw_trial in raw_trials:
        if not isinstance(raw_trial, Mapping) or set(raw_trial) != {
            "kind",
            "trial_id",
            "trigger_status",
        }:
            raise ContractError("each v9 config trial must contain the registered fields")
        trials.append(
            V9TrialSpec(
                kind=cast(V9TrialKind, str(raw_trial["kind"])),
                trial_id=str(raw_trial["trial_id"]),
                trigger_status=cast(
                    V9TriggerStatus, str(raw_trial["trigger_status"])
                ),
            )
        )
    identities = config["source_identities"]
    if not isinstance(identities, Mapping):
        raise ContractError("v9 config source_identities must be an object")
    return V9Plan(
        run_id=str(config["run_id"]),
        stage=cast(V9Stage, str(config["stage"])),
        seeds=tuple(int(value) for value in cast(list[int], config["seeds"])),
        fold_ids=tuple(
            int(value) for value in cast(list[int], config["fold_ids"])
        ),
        torch_threads=int(cast(int, config["torch_threads"])),
        precision=str(config["precision"]),
        model_family=str(config["model_family"]),
        input_steps=int(cast(int, config["input_steps"])),
        max_epochs=int(cast(int, config["max_epochs"])),
        patience=int(cast(int, config["patience"])),
        min_delta=float(cast(float, config["min_delta"])),
        control_trial_id=str(config["control_trial_id"]),
        channels=int(cast(int, config["channels"])),
        kernel_size=int(cast(int, config["kernel_size"])),
        dilations=tuple(
            int(value) for value in cast(list[int], config["dilations"])
        ),
        dropout=float(cast(float, config["dropout"])),
        learning_rate=float(cast(float, config["learning_rate"])),
        batch_size=int(cast(int, config["batch_size"])),
        trials=tuple(trials),
        candidate_order=tuple(
            str(value) for value in cast(list[str], config["candidate_order"])
        ),
        source_identities={str(key): str(value) for key, value in identities.items()},
        rankic_candidate_model_id=str(config["rankic_candidate_model_id"]),
    )


def _validate_split_manifest(plan: V9Plan, split_manifest: pd.DataFrame) -> None:
    required = {"sample_position", "fold", "stage", "sealed"}
    if missing := sorted(required.difference(split_manifest.columns)):
        raise ContractError(f"v9 split manifest missing columns: {', '.join(missing)}")
    if split_manifest.empty:
        raise ContractError("v9 split manifest cannot be empty")
    if split_manifest[list(required)].isna().any().any():
        raise ContractError("v9 split manifest contains undeclared values")
    if split_manifest["sealed"].astype(bool).any():
        raise ContractError("v9 ordinary validation rejects sealed rows")
    stages = set(split_manifest["stage"].astype(str))
    if not stages <= {"train", "validation"}:
        raise ContractError("v9 accepts only ordinary train/validation rows")
    if stages != {"train", "validation"}:
        raise ContractError("each v9 run requires train and validation rows")
    observed_folds = set(split_manifest["fold"].astype(int))
    if observed_folds != set(plan.fold_ids):
        raise ContractError("v9 plan fold IDs do not match the split manifest")
    for fold, rows in split_manifest.groupby("fold", observed=True):
        if set(rows["stage"].astype(str)) != {"train", "validation"}:
            raise ContractError(f"v9 fold {fold} requires train and validation rows")
    if split_manifest.duplicated(["fold", "sample_position"]).any():
        raise ContractError("v9 split manifest contains duplicate fold positions")


def validate_v9_plan(plan: V9Plan, split_manifest: pd.DataFrame) -> V9Plan:
    """Fail closed before a diagnostic or optimizer can observe any row."""

    if str(plan.stage) not in _STAGES:
        raise ContractError(f"unsupported v9 stage: {plan.stage}")
    if not plan.run_id or Path(plan.run_id).name != plan.run_id:
        raise ContractError("v9 run_id must be a non-empty path-safe name")
    if plan.model_family != "tcn":
        raise ContractError("v9 permits only the TCN model family")
    if plan.precision != "float32":
        raise ContractError("v9 formal protocol requires float32 precision")
    if plan.torch_threads <= 0 or plan.input_steps <= 0:
        raise ContractError("v9 threads and input steps must be positive")
    if not plan.seeds or len(set(plan.seeds)) != len(plan.seeds):
        raise ContractError("v9 seeds must be non-empty and unique")
    if plan.fold_ids != (0, 1, 2, 3, 4):
        raise ContractError("v9 protocol is frozen to ordinary-validation folds 0-4")
    if plan.max_epochs <= 0 or plan.max_epochs > 8:
        raise ContractError("v9 max_epochs must be between 1 and 8")
    if plan.patience != 2 or plan.min_delta != 0.002:
        raise ContractError("v9 early stopping is frozen to patience 2 and min_delta 0.002")
    frozen_control = (
        plan.control_trial_id == "lite-c16-no-dropout"
        and plan.channels == 16
        and plan.kernel_size == 3
        and plan.dilations == (1, 2, 4, 8, 16, 32, 64, 128)
        and plan.dropout == 0.0
        and plan.learning_rate == 0.003
        and plan.batch_size == 128
    )
    if not frozen_control:
        raise ContractError("v9 control model and optimizer configuration drifted")
    if len(plan.trials) > 3:
        raise ContractError("v9 permits at most three pre-registered new trials")
    trial_ids = [trial.trial_id for trial in plan.trials]
    trial_kinds = [str(trial.kind) for trial in plan.trials]
    if any(not trial_id for trial_id in trial_ids) or len(set(trial_ids)) != len(trial_ids):
        raise ContractError("v9 trial IDs must be non-empty and unique")
    if any(kind not in _TRIAL_KINDS for kind in trial_kinds):
        raise ContractError("v9 plan contains an unsupported trial kind")
    if len(set(trial_kinds)) != len(trial_kinds):
        raise ContractError("v9 permits at most one trial of each registered kind")
    if any(str(trial.trigger_status) not in _TRIGGER_STATUSES for trial in plan.trials):
        raise ContractError("v9 trial contains an unsupported trigger status")
    if tuple(trial_ids) != plan.candidate_order:
        raise ContractError("v9 candidate order must match the frozen trial order")
    if plan.stage == "diagnostic":
        if plan.seeds != (7,) or any(
            trial.trigger_status != "pending" for trial in plan.trials
        ):
            raise ContractError("v9 diagnostic requires seed 7 and pending trials")
    if plan.stage == "formal_screen":
        if plan.seeds != (7,) or any(
            trial.trigger_status == "pending" for trial in plan.trials
        ):
            raise ContractError(
                "v9 formal screen requires seed 7 and resolved triggers"
            )
    if plan.stage == "multi_seed_confirmation":
        if plan.seeds != (17, 27) or any(
            trial.trigger_status == "pending" for trial in plan.trials
        ):
            raise ContractError(
                "v9 confirmation requires seeds 17/27 and resolved triggers"
            )
    required_artifacts = {
        "data",
        "fold_manifest",
        "window_index",
        "labels",
        "evaluation",
    }
    required_artifacts |= {
        "lite_config",
        "bai_config",
        "rankic_predictions",
        "rankic_candidate_config",
    }
    required_artifacts |= {
        f"{family}_checkpoint_fold_{fold}"
        for family in ("lite", "bai", "rankic_candidate")
        for fold in plan.fold_ids
    }
    if not required_artifacts <= set(plan.source_identities):
        raise ContractError("v9 plan is missing frozen input identities")
    if not plan.source_identities:
        raise ContractError("v9 plan requires source identities")
    if (
        not plan.rankic_candidate_model_id
        or plan.rankic_candidate_model_id == plan.control_trial_id
    ):
        raise ContractError("v9 RankIC candidate model identity is invalid")
    for name, digest in plan.source_identities.items():
        if not name or not _SHA256.fullmatch(str(digest)):
            raise ContractError("v9 source identities must be named SHA-256 digests")
    _validate_split_manifest(plan, split_manifest)
    return plan


def build_v9_plan_receipt(
    plan: V9Plan,
    split_manifest: pd.DataFrame,
    *,
    project_root: Path,
) -> dict[str, object]:
    """Build the deterministic receipt content for a validated plan."""

    validate_v9_plan(plan, split_manifest)
    resolved_plan = cast(dict[str, object], canonicalize(asdict(plan)))
    resolved_plan["source_identities"] = dict(sorted(plan.source_identities.items()))
    payload: dict[str, object] = {
        "schema_version": "tcn-v9-plan/v1",
        "resolved_plan": resolved_plan,
        "identities": dict(sorted(plan.source_identities.items())),
        "code_identity": code_identity(project_root.resolve()),
        "split_summary": {
            "folds": sorted(int(value) for value in split_manifest["fold"].unique()),
            "train_rows": int(split_manifest["stage"].eq("train").sum()),
            "validation_rows": int(split_manifest["stage"].eq("validation").sum()),
        },
        "sealed_test_accessed": False,
    }
    payload["receipt_id"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def execute_v9_plan(
    plan: V9Plan,
    split_manifest: pd.DataFrame,
    *,
    output_dir: Path,
    project_root: Path,
) -> Path:
    """Validate and atomically publish, or deterministically replay, a v9 plan."""

    receipt = build_v9_plan_receipt(
        plan,
        split_manifest,
        project_root=project_root,
    )
    return publish_immutable_receipt(
        receipt,
        output_dir=output_dir,
        filename="plan-receipt.json",
        identity_label="v9 plan",
    )
