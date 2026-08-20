"""Strategy-level dependency injection contracts."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from unmark.core.budgets import BudgetAccount, BudgetLease
from unmark.core.document import Document
from unmark.core.operations import Operation
from unmark.core.policies import FidelityPolicy
from unmark.core.spans import Span
from unmark.core.targets import ReductionTarget, StrategyDescriptor
from unmark.detectors.localization import TextRegion
from unmark.detectors.protocols import DetectorScore

if TYPE_CHECKING:
    from unmark.strategies.targeted.propose import ProposedReplacement


@runtime_checkable
class RegionLocalizer(Protocol):
    def localize(
        self, document: Document, baseline: DetectorScore | None
    ) -> tuple[TextRegion, ...]: ...


@runtime_checkable
class ProposalProvider(Protocol):
    id: str
    uses_model: bool
    estimated_cost_usd: Decimal

    def propose(
        self,
        *,
        source_text: str,
        candidate_text: str,
        region: TextRegion,
        protected_spans: tuple[Span, ...],
        seed: int,
        limit: int,
    ) -> tuple[ProposedReplacement, ...]: ...


@runtime_checkable
class FidelityEvaluator(Protocol):
    def evaluate(
        self,
        document: Document,
        operations: tuple[Operation, ...],
        policy: FidelityPolicy,
        budget: BudgetAccount,
        editable_regions: tuple[TextRegion, ...],
    ) -> object: ...


@runtime_checkable
class CandidateScorer(Protocol):
    def score(self, text: str, budget: BudgetAccount) -> DetectorScore: ...


@runtime_checkable
class AttackStrategy(Protocol):
    descriptor: StrategyDescriptor

    def run(
        self,
        document: Document,
        target: ReductionTarget,
        budget: BudgetAccount,
        policy: FidelityPolicy,
    ) -> object: ...


@runtime_checkable
class CheckpointStore(Protocol):
    def save(self, checkpoint: dict[str, object]) -> None: ...

    def load(self) -> dict[str, object] | None: ...


__all__ = [
    "AttackStrategy",
    "BudgetLease",
    "CandidateScorer",
    "CheckpointStore",
    "FidelityEvaluator",
    "ProposalProvider",
    "RegionLocalizer",
]
