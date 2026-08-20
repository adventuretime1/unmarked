"""Candidate and run result contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from unmark.core.budgets import BudgetUsage
from unmark.core.operations import Operation
from unmark.core.spans import StrictModel
from unmark.core.targets import ScoreEvidence

ResultState = Literal[
    "sanitized",
    "verified_below_threshold",
    "verified_reduction_only",
    "surrogate_reduced",
    "rewritten_unverified",
    "abstained",
    "unsupported",
]
"""Result states. Never report a generic "clean" result.

This iteration may only emit ``sanitized``, ``rewritten_unverified``,
``abstained``, or ``unsupported``. ``verified_below_threshold`` requires a named,
versioned detector and is unreachable until Phase 2.
"""

VERIFIED_STATES: frozenset[str] = frozenset({"verified_below_threshold", "verified_reduction_only"})

STATES_AVAILABLE_NOW: frozenset[str] = frozenset(
    {"sanitized", "rewritten_unverified", "abstained", "unsupported"}
)


class CandidateResult(StrictModel):
    """One candidate produced by a strategy."""

    schema_version: Literal["1"] = "1"
    candidate_id: str
    parent_id: str | None = None
    strategy_id: str
    operations: tuple[Operation, ...] = ()
    content_sha256: str
    char_edit_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    token_edit_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    length_drift_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    invariants_passed: bool = True
    evidence: tuple[ScoreEvidence, ...] = ()
    accepted: bool = True
    rejection_reason: str | None = None


class RunResult(StrictModel):
    """The terminal record of a run."""

    schema_version: Literal["1"] = "1"
    run_id: str
    state: ResultState
    started_at: datetime
    finished_at: datetime
    source_sha256: str
    media_type: str
    preset: str
    selected: CandidateResult | None = None
    alternatives: tuple[CandidateResult, ...] = ()
    usage: BudgetUsage = BudgetUsage()
    residual_risk: str = Field(
        default="",
        description="Plain-language description of which signal the run changed or measured.",
    )
    notes: tuple[str, ...] = ()

    @property
    def claims_verification(self) -> bool:
        return self.state in VERIFIED_STATES
