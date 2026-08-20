"""One-shot rewrite: a single prompt, N candidates, pick the smallest safe edit.

This is the simplest baseline. It renders one rewrite prompt, asks the model for
``candidate_count`` completions, validates each, and selects using Unmarked's
minimum-edit order (see :mod:`unmark.strategies.rewrite.candidates`). With the
``print-prompt`` backend it emits the prompt and stops.
"""

from __future__ import annotations

import time

from unmark.core.budgets import BudgetAccount
from unmark.core.document import Document
from unmark.core.errors import BudgetExhaustedError, DependencyUnavailableError
from unmark.core.policies import FidelityPolicy
from unmark.core.results import ResultState
from unmark.core.targets import ReductionTarget, StrategyDescriptor
from unmark.detectors.protocols import DetectorScore
from unmark.strategies.objectives import target_met
from unmark.strategies.rewrite.candidates import RewriteCandidate, select_candidate, selection_key
from unmark.strategies.rewrite.common import RewriteStrategyBase, detector_baseline, result_state
from unmark.strategies.rewrite.result import (
    RESIDUAL_RISK_BY_STATE,
    RewriteResult,
    RewriteTrace,
)
from unmark.strategies.runtime import settle_runtime


class OneShotRewriteStrategy(RewriteStrategyBase):
    """A single-prompt, multi-candidate rewrite baseline."""

    descriptor = StrategyDescriptor(
        id="rewrite-one-shot",
        version="1.0.0",
        stability="experimental",
        capabilities=frozenset({"prompt-rewrite", "print-prompt"}),
        invasiveness="medium",
    )

    def run(
        self,
        document: Document,
        target: ReductionTarget,
        budget: BudgetAccount,
        policy: FidelityPolicy,
    ) -> RewriteResult:
        started = time.monotonic()
        baseline = detector_baseline(self.detector, document, budget)

        try:
            step = self.engine.step(
                document=document,
                base_text=document.source_text,
                style=self.config.style,
                parent_id=None,
                budget=budget,
                policy=policy,
                baseline=baseline,
                count=self.config.candidate_count,
            )
        except DependencyUnavailableError as error:
            return self._terminal(
                state="unsupported",
                baseline=baseline,
                selected=None,
                alternatives=(),
                rejected=(),
                budget=budget,
                started=started,
                prompt_only=None,
                stopping_reason=str(error),
            )
        except BudgetExhaustedError as error:
            return self._terminal(
                state="abstained",
                baseline=baseline,
                selected=None,
                alternatives=(),
                rejected=(),
                budget=budget,
                started=started,
                prompt_only=None,
                stopping_reason=str(error),
            )

        if step.prompt_only is not None:
            return self._terminal(
                state="unsupported",
                baseline=baseline,
                selected=None,
                alternatives=(),
                rejected=(),
                budget=budget,
                started=started,
                prompt_only=step.prompt_only,
                stopping_reason="print-prompt backend emitted the prompt without rewriting",
            )

        scored = tuple(
            candidate.model_copy(
                update={
                    "target_met": target_met(target, baseline, candidate.score),
                }
            )
            for candidate in step.candidates
        )
        selected = select_candidate(scored)
        if selected is None:
            return self._terminal(
                state="abstained",
                baseline=baseline,
                selected=None,
                alternatives=(),
                rejected=step.rejected,
                budget=budget,
                started=started,
                prompt_only=None,
                stopping_reason="no fidelity-valid candidate was produced",
            )

        state = result_state(target, selected)
        alternatives = tuple(
            candidate
            for candidate in sorted(scored, key=selection_key)
            if candidate.candidate_id != selected.candidate_id
        )
        return self._terminal(
            state=state,
            baseline=baseline,
            selected=selected,
            alternatives=alternatives,
            rejected=step.rejected,
            budget=budget,
            started=started,
            prompt_only=None,
            stopping_reason="selected the minimum-edit fidelity-valid candidate",
        )

    def _terminal(
        self,
        *,
        state: ResultState,
        baseline: DetectorScore | None,
        selected: RewriteCandidate | None,
        alternatives: tuple[RewriteCandidate, ...],
        rejected: tuple[RewriteCandidate, ...],
        budget: BudgetAccount,
        started: float,
        prompt_only: str | None,
        stopping_reason: str,
    ) -> RewriteResult:
        settle_runtime(budget, started)
        return RewriteResult(
            state=state,
            baseline=baseline,
            selected=selected,
            alternatives=alternatives,
            rejected=rejected,
            usage=budget.usage(
                char_edit_ratio=selected.char_edit_ratio if selected else 0.0,
                token_edit_ratio=selected.token_edit_ratio if selected else 0.0,
                length_drift_ratio=selected.length_drift_ratio if selected else 0.0,
            ),
            trace=RewriteTrace(
                strategy_id=self.descriptor.id,
                backend_id=self.adapter.id,
                style=self.config.style,
                prompt_only=prompt_only,
                candidates_generated=len(alternatives) + (1 if selected else 0),
                candidates_rejected=len(rejected),
                rejection_reasons=tuple(
                    candidate.rejection_reason
                    for candidate in rejected
                    if candidate.rejection_reason
                ),
                stopping_reason=stopping_reason,
                selection_reason=(
                    "hard fidelity, then target status, then minimum edit cost, length "
                    "drift, residual risk, resource use, and a deterministic hash"
                )
                if selected
                else "",
            ),
            residual_risk=RESIDUAL_RISK_BY_STATE[state],
        )
