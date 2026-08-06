"""Offline TCN short-horizon research package."""

from .duckdb_source import (
    TrainingCoverageAudit,
    audit_duckdb_training_coverage,
    write_training_coverage_receipt,
)
from .agent_cli import run_agent_request
from .experiment import ContractError, RunResult, run_experiment
from .readiness import PilotReadinessReport, ReadinessCheck, check_pilot_readiness

__version__ = "0.1.0"
__all__ = [
    "ContractError",
    "PilotReadinessReport",
    "ReadinessCheck",
    "RunResult",
    "TrainingCoverageAudit",
    "audit_duckdb_training_coverage",
    "check_pilot_readiness",
    "run_agent_request",
    "run_experiment",
    "write_training_coverage_receipt",
]
