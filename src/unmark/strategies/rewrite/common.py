"""Operations shared by the built-in rewrite strategies."""

from __future__ import annotations

from decimal import Decimal

from unmark.core.budgets import BudgetAccount
from unmark.core.document import Document
from unmark.core.errors import BudgetExhaustedError
from unmark.core.results import ResultState
from unmark.core.targets import ReductionTarget
from unmark.detectors.protocols import Detector, DetectorScore
from unmark.fidelity.protocols import BasicFidelityEvaluator
from unmark.models.protocols import ModelAdapter
from unmark.strategies.objectives import successful_state
from unmark.strategies.rewrite.candidates import RewriteCandidate
from unmark.strategies.rewrite.config import RewriteConfig
from unmark.strategies.rewrite.engine import RewriteEngine


class RewriteStrategyBase:
    """Shared dependencies for one-shot and recursive rewriting."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        config: RewriteConfig,
        fidelity: BasicFidelityEvaluator | None = None,
        detector: Detector | None = None,
        estimated_cost_usd: Decimal = Decimal("0"),
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.detector = detector
        self.engine = RewriteEngine(
            adapter=adapter,
            config=config,
            fidelity=fidelity,
            detector=detector,
            estimated_cost_usd=estimated_cost_usd,
        )


def detector_baseline(
    detector: Detector | None,
    document: Document,
    budget: BudgetAccount,
) -> DetectorScore | None:
    """Score the source once, returning no baseline when unavailable by budget."""
    if detector is None:
        return None
    try:
        with budget.reserve("detector_queries") as lease:
            score = detector.score(document.source_text)
            lease.settle(1)
    except BudgetExhaustedError:
        return None
    return score


def result_state(target: ReductionTarget, selected: RewriteCandidate) -> ResultState:
    """Map a selected candidate to the strongest evidence-backed result state."""
    if selected.score is not None and selected.target_met:
        return successful_state(target, selected.score)
    return "rewritten_unverified"
