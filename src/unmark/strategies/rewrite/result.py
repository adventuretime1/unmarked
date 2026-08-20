"""The result record shared by the rewrite baselines."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from unmark.core.budgets import BudgetUsage
from unmark.core.results import CandidateResult, ResultState
from unmark.core.spans import StrictModel
from unmark.detectors.protocols import DetectorScore
from unmark.strategies.rewrite.candidates import RewriteCandidate


class RewriteHopRecord(StrictModel):
    """One recursive hop, retained even after a later hop supersedes it."""

    round_index: int = Field(ge=0)
    style: str
    base_content_hash: str
    best_candidate_id: str | None = None
    best_residual_risk: float | None = None
    accepted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    note: str = ""


class RewriteTrace(StrictModel):
    strategy_id: str
    backend_id: str
    style: str
    prompt_only: str | None = None
    candidates_generated: int = Field(default=0, ge=0)
    candidates_rejected: int = Field(default=0, ge=0)
    rejection_reasons: tuple[str, ...] = ()
    hops: tuple[RewriteHopRecord, ...] = ()
    stopping_reason: str = ""
    selection_reason: str = ""
    warnings: tuple[str, ...] = ()


class RewriteResult(StrictModel):
    schema_version: Literal["1"] = "1"
    state: ResultState
    baseline: DetectorScore | None = None
    selected: RewriteCandidate | None = None
    alternatives: tuple[RewriteCandidate, ...] = ()
    rejected: tuple[RewriteCandidate, ...] = ()
    usage: BudgetUsage
    trace: RewriteTrace
    residual_risk: str

    def selected_candidate_result(self, strategy_id: str) -> CandidateResult | None:
        if self.selected is None:
            return None
        selected = self.selected
        return CandidateResult(
            candidate_id=selected.candidate_id,
            parent_id=selected.parent_id,
            strategy_id=strategy_id,
            operations=selected.operations,
            content_sha256=selected.content_hash,
            char_edit_ratio=selected.char_edit_ratio,
            token_edit_ratio=selected.token_edit_ratio,
            length_drift_ratio=selected.length_drift_ratio,
            invariants_passed=selected.fidelity_passed,
            evidence=(selected.score.to_evidence(),) if selected.score else (),
            accepted=True,
        )


RESIDUAL_RISK_BY_STATE: dict[str, str] = {
    "verified_below_threshold": (
        "The named detector's score fell below its configured threshold after the rewrite."
    ),
    "verified_reduction_only": (
        "The named detector measured the requested score reduction after the rewrite."
    ),
    "surrogate_reduced": (
        "The surrogate score decreased. The rewrite also changed token, n-gram, "
        "sentence, and model-probability patterns."
    ),
    "rewritten_unverified": (
        "The rewrite changed token, n-gram, sentence, and model-probability patterns, "
        "reducing the chance that the original statistical signature persists. "
        "Broader rewrites generally disrupt more of that signature."
    ),
    "abstained": ("No fidelity-valid rewrite met the request inside the configured budgets."),
    "unsupported": "A capability required by the requested evidence state was unavailable.",
    "sanitized": "Not applicable to rewrite strategies.",
}
