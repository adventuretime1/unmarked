"""Recursive rewrite with early stopping and rollback.

Each hop rewrites the *previous hop's text*, but every intermediate is still
materialized as a single whole-document replace against the immutable source, so
fidelity, protected-span, and budget gating apply identically at every hop.

Discipline required of this strategy:

* run 1-5 hops on a configurable style schedule;
* validate every hop and score valid intermediates when a detector exists;
* retain every valid intermediate as a frontier candidate;
* stop early when the requested target is met;
* stop on budget exhaustion or on stagnation (``early_stop_patience`` hops with
  no residual-risk improvement);
* roll back: the base text for the next hop is always the best candidate seen so
  far, so a hop that regresses cannot drag the chain downhill;
* never let the final hop win automatically — selection runs over the whole
  retained frontier with the minimum-edit order, exactly like one-shot.
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
from unmark.strategies.rewrite.candidates import (
    RewriteCandidate,
    select_candidate,
    selection_key,
)
from unmark.strategies.rewrite.common import RewriteStrategyBase, detector_baseline, result_state
from unmark.strategies.rewrite.result import (
    RESIDUAL_RISK_BY_STATE,
    RewriteHopRecord,
    RewriteResult,
    RewriteTrace,
)
from unmark.strategies.runtime import settle_runtime


class RecursiveRewriteStrategy(RewriteStrategyBase):
    """A hop-by-hop rewrite baseline with early stopping and rollback."""

    descriptor = StrategyDescriptor(
        id="rewrite-recursive",
        version="1.0.0",
        stability="experimental",
        capabilities=frozenset({"prompt-rewrite", "print-prompt", "recursive"}),
        invasiveness="high",
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

        # The source itself is the trivial zero-edit frontier member. It never
        # "wins" unless nothing better is valid, because it has target_met=False
        # for any real reduction request and a higher (or equal) residual risk.
        source_candidate = self._source_candidate(document, baseline, target)
        frontier: list[RewriteCandidate] = [source_candidate]
        rejected: list[RewriteCandidate] = []
        hops: list[RewriteHopRecord] = []

        # The best candidate so far is what the next hop rewrites from. This is
        # the rollback mechanism: a regressing hop never becomes the new base.
        best = source_candidate
        best_risk = source_candidate.residual_risk
        stagnant = 0
        stopping_reason = "hop schedule exhausted"
        prompt_only: str | None = None

        for round_index in range(self.config.rounds):
            style = self.config.style_for_round(round_index)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if elapsed_ms >= budget.budget.max_runtime_ms:
                stopping_reason = "runtime budget exhausted"
                break
            try:
                step = self.engine.step(
                    document=document,
                    base_text=best.text,
                    style=style,
                    parent_id=best.candidate_id,
                    budget=budget,
                    policy=policy,
                    baseline=baseline,
                    count=self.config.candidate_count,
                )
            except DependencyUnavailableError as error:
                stopping_reason = str(error)
                if not _has_rewrite(frontier):
                    return self._terminal(
                        state="unsupported",
                        baseline=baseline,
                        selected=None,
                        frontier=(),
                        rejected=tuple(rejected),
                        hops=tuple(hops),
                        budget=budget,
                        started=started,
                        prompt_only=None,
                        stopping_reason=stopping_reason,
                        target=target,
                    )
                break
            except BudgetExhaustedError as error:
                stopping_reason = str(error)
                break

            if step.prompt_only is not None:
                prompt_only = step.prompt_only
                stopping_reason = "print-prompt backend emitted the prompt without rewriting"
                break

            scored = tuple(
                candidate.model_copy(
                    update={"target_met": target_met(target, baseline, candidate.score)}
                )
                for candidate in step.candidates
            )
            rejected.extend(step.rejected)
            frontier.extend(scored)

            hop_best = min(scored, key=selection_key) if scored else None
            hops.append(
                RewriteHopRecord(
                    round_index=round_index,
                    style=style,
                    base_content_hash=best.content_hash,
                    best_candidate_id=hop_best.candidate_id if hop_best else None,
                    best_residual_risk=(
                        hop_best.residual_risk if hop_best and hop_best.score is not None else None
                    ),
                    accepted_count=len(scored),
                    rejected_count=len(step.rejected),
                    note="no valid candidate this hop" if hop_best is None else "",
                )
            )

            # Early stop: the requested target is met by some valid candidate.
            if any(candidate.target_met for candidate in scored):
                stopping_reason = "requested target met; stopping early"
                break

            # Rollback + stagnation: advance the base only if this hop strictly
            # improved on the best residual risk seen so far. A worse or equal hop
            # leaves ``best`` untouched, so the next hop rewrites the earlier,
            # better candidate rather than the regression.
            if hop_best is not None and hop_best.residual_risk < best_risk:
                best = hop_best
                best_risk = hop_best.residual_risk
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= self.config.early_stop_patience:
                    stopping_reason = "stagnation limit reached; stopping early"
                    break

        selected = select_candidate(tuple(frontier))
        # The source-only candidate winning means no rewrite improved on doing
        # nothing; report that honestly as an abstain rather than a rewrite.
        if selected is None or (selected.candidate_id == source_candidate.candidate_id):
            selected_rewrite = select_candidate(
                tuple(c for c in frontier if c.candidate_id != source_candidate.candidate_id)
            )
            if selected_rewrite is None:
                return self._terminal(
                    state="abstained",
                    baseline=baseline,
                    selected=None,
                    frontier=tuple(frontier),
                    rejected=tuple(rejected),
                    hops=tuple(hops),
                    budget=budget,
                    started=started,
                    prompt_only=prompt_only,
                    stopping_reason=(
                        "no fidelity-valid rewrite improved on the source"
                        if prompt_only is None
                        else stopping_reason
                    ),
                    target=target,
                )
            selected = selected_rewrite

        state = result_state(target, selected)
        alternatives = tuple(
            candidate
            for candidate in sorted(frontier, key=selection_key)
            if candidate.candidate_id != selected.candidate_id
        )
        return self._terminal(
            state=state,
            baseline=baseline,
            selected=selected,
            frontier=alternatives,
            rejected=tuple(rejected),
            hops=tuple(hops),
            budget=budget,
            started=started,
            prompt_only=prompt_only,
            stopping_reason=stopping_reason,
            target=target,
        )

    @staticmethod
    def _source_candidate(
        document: Document, baseline: DetectorScore | None, target: ReductionTarget
    ) -> RewriteCandidate:
        from unmark.core.operations import normalized_content_hash

        content_hash = normalized_content_hash(document.source_text)
        return RewriteCandidate(
            candidate_id=f"source:{content_hash[:16]}",
            parent_id=None,
            origin="source",
            text=document.source_text,
            operations=(),
            content_hash=content_hash,
            fidelity_passed=True,
            char_edit_ratio=0.0,
            token_edit_ratio=0.0,
            length_drift_ratio=0.0,
            score=baseline,
            target_met=target_met(target, baseline, baseline),
        )

    def _terminal(
        self,
        *,
        state: ResultState,
        baseline: DetectorScore | None,
        selected: RewriteCandidate | None,
        frontier: tuple[RewriteCandidate, ...],
        rejected: tuple[RewriteCandidate, ...],
        hops: tuple[RewriteHopRecord, ...],
        budget: BudgetAccount,
        started: float,
        prompt_only: str | None,
        stopping_reason: str,
        target: ReductionTarget,
    ) -> RewriteResult:
        settle_runtime(budget, started)
        return RewriteResult(
            state=state,
            baseline=baseline,
            selected=selected,
            alternatives=frontier,
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
                candidates_generated=len(frontier) + (1 if selected else 0),
                candidates_rejected=len(rejected),
                rejection_reasons=tuple(
                    candidate.rejection_reason
                    for candidate in rejected
                    if candidate.rejection_reason
                ),
                hops=hops,
                stopping_reason=stopping_reason,
                selection_reason=(
                    "best retained frontier candidate by hard fidelity, target status, "
                    "minimum edit cost, length drift, residual risk, resource use, and hash"
                )
                if selected
                else "",
            ),
            residual_risk=RESIDUAL_RISK_BY_STATE[state],
        )


def _has_rewrite(frontier: list[RewriteCandidate]) -> bool:
    return any(candidate.origin != "source" for candidate in frontier)
