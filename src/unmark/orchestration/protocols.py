"""Extension points for later phases.

These protocols exist so Phase 2+ components plug in without reshaping the
application layer. They are intentionally small: each has a concrete consumer in
this slice or an immediate one in the next phase. Nothing here imports a model,
detector, or vendor SDK.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from unmark.core.budgets import BudgetAccount
from unmark.core.document import Document
from unmark.core.operations import Operation
from unmark.core.policies import FidelityPolicy
from unmark.core.results import CandidateResult
from unmark.core.targets import ScoreEvidence, StrategyDescriptor


@runtime_checkable
class Detector(Protocol):
    """Scores text for a provenance signal. Implemented in Phase 2."""

    id: str
    version: str

    def score(self, text: str) -> ScoreEvidence:
        """Return evidence for ``text``. Must not mutate it."""
        ...


@runtime_checkable
class RewriteModel(Protocol):
    """Produces rewrite candidates. Implemented in Phase 4."""

    id: str

    def rewrite(self, text: str, *, instruction: str, seed: int | None = None) -> str: ...


@runtime_checkable
class FidelityGate(Protocol):
    """A single stage of the staged rejection pipeline.

    Gates run cheapest-first and a failure is terminal for that candidate: a
    detector score can never override a hard fidelity failure.
    """

    id: str

    def check(
        self,
        document: Document,
        candidate_text: str,
        operations: tuple[Operation, ...],
        policy: FidelityPolicy,
    ) -> GateOutcome: ...


class GateOutcome:
    """Result of one fidelity gate."""

    __slots__ = ("gate_id", "passed", "reason")

    def __init__(self, gate_id: str, passed: bool, reason: str = "") -> None:
        self.gate_id = gate_id
        self.passed = passed
        self.reason = reason

    def __repr__(self) -> str:
        status = "pass" if self.passed else f"fail: {self.reason}"
        return f"<GateOutcome {self.gate_id} {status}>"


@runtime_checkable
class Strategy(Protocol):
    """An editing strategy. The sanitation strategy in this slice implements it."""

    descriptor: StrategyDescriptor

    def run(
        self,
        document: Document,
        budget: BudgetAccount,
        policy: FidelityPolicy,
    ) -> CandidateResult: ...
