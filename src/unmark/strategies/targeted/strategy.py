"""Production-shaped targeted constrained-search strategy."""

from __future__ import annotations

import random
import time
from contextlib import ExitStack
from decimal import Decimal
from typing import Literal

from pydantic import Field

from unmark.core.budgets import BudgetAccount, BudgetUsage, CounterName
from unmark.core.document import Document
from unmark.core.errors import BudgetExhaustedError, DependencyUnavailableError
from unmark.core.operations import Operation, normalized_content_hash
from unmark.core.policies import FidelityPolicy
from unmark.core.results import CandidateResult, ResultState
from unmark.core.spans import StrictModel
from unmark.core.targets import ReductionTarget, StrategyDescriptor
from unmark.detectors.protocols import Detector, DetectorScore
from unmark.detectors.surrogate.pll import CachedPllScorer
from unmark.fidelity.protocols import BasicFidelityEvaluator, FidelityReport
from unmark.strategies.objectives import (
    successful_state,
    target_met,
    unsupported_reason,
)
from unmark.strategies.protocols import CheckpointStore, ProposalProvider
from unmark.strategies.runtime import settle_runtime
from unmark.strategies.targeted.beam import (
    CandidateMetrics,
    RejectedCandidate,
    SearchCandidate,
    TargetedCheckpoint,
    candidate_id,
    diverse_beam,
    lexicographic_key,
    pareto_frontier,
)
from unmark.strategies.targeted.config import TargetedSearchConfig
from unmark.strategies.targeted.localize import LocalizationResult, TargetedLocalizer
from unmark.strategies.targeted.propose import ProposedReplacement


class TargetedSearchTrace(StrictModel):
    localizer_mode: str = ""
    localizer_reason: str = ""
    localized_regions: tuple[tuple[int, int, float], ...] = ()
    proposals_generated: int = Field(default=0, ge=0)
    candidates_deduplicated: int = Field(default=0, ge=0)
    candidates_scored: int = Field(default=0, ge=0)
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    stopping_reason: str
    selection_reason: str = ""
    warnings: tuple[str, ...] = ()


class TargetedSearchResult(StrictModel):
    schema_version: Literal["1"] = "1"
    state: ResultState
    baseline: DetectorScore | None = None
    selected: SearchCandidate | None = None
    frontier: tuple[SearchCandidate, ...] = ()
    successes: tuple[SearchCandidate, ...] = ()
    rejected: tuple[RejectedCandidate, ...] = ()
    usage: BudgetUsage
    trace: TargetedSearchTrace
    residual_risk: str
    checkpoint: TargetedCheckpoint | None = None

    def selected_candidate_result(self) -> CandidateResult | None:
        if self.selected is None:
            return None
        return CandidateResult(
            candidate_id=self.selected.candidate_id,
            parent_id=self.selected.parent_id,
            strategy_id="targeted-search",
            operations=self.selected.operations,
            content_sha256=self.selected.content_hash,
            char_edit_ratio=self.selected.metrics.char_edit_ratio,
            token_edit_ratio=self.selected.metrics.token_edit_ratio,
            length_drift_ratio=self.selected.metrics.length_drift_ratio,
            invariants_passed=self.selected.fidelity_passed,
            evidence=(self.selected.score.to_evidence(),) if self.selected.score else (),
            accepted=True,
        )


def _listify(value: object) -> object:
    if isinstance(value, tuple):
        return [_listify(item) for item in value]
    if isinstance(value, list):
        return [_listify(item) for item in value]
    return value


def _tuplify(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    return value


class TargetedSearchStrategy:
    """Bounded beam search over small, fidelity-gated source-anchored edits."""

    descriptor = StrategyDescriptor(
        id="targeted-search",
        version="1.0.0",
        stability="experimental",
        capabilities=frozenset({"localized-proposals", "detector-or-surrogate"}),
        invasiveness="low",
    )

    def __init__(
        self,
        *,
        config: TargetedSearchConfig,
        proposal_provider: ProposalProvider,
        detector: Detector | None = None,
        pll: CachedPllScorer | None = None,
        fidelity: BasicFidelityEvaluator | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.config = config
        self.proposal_provider = proposal_provider
        self.detector = detector
        self.pll = pll
        self.fidelity = fidelity or BasicFidelityEvaluator()
        self.checkpoint_store = checkpoint_store

    def _score(
        self, text: str, budget: BudgetAccount, deadline: float | None = None
    ) -> DetectorScore | None:
        if self.detector is None:
            return None
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExhaustedError("runtime budget exhausted before detector query")
        with budget.reserve("detector_queries") as query_lease:
            score = self.detector.score(text)
            query_lease.settle(1)
        if deadline is not None and time.monotonic() > deadline:
            raise BudgetExhaustedError("runtime budget exhausted during detector query")
        return score

    def _propose(
        self,
        *,
        document: Document,
        candidate: SearchCandidate,
        region_index: int,
        localization: LocalizationResult,
        budget: BudgetAccount,
        rng: random.Random,
        deadline: float,
    ) -> tuple[ProposedReplacement, ...]:
        region = localization.regions[region_index]
        if time.monotonic() >= deadline:
            raise BudgetExhaustedError("runtime budget exhausted before proposal call")
        with ExitStack() as stack:
            model_lease = None
            cost_lease = None
            if self.proposal_provider.uses_model:
                model_lease = stack.enter_context(budget.reserve("model_calls"))
                cost = self.proposal_provider.estimated_cost_usd
                cost_lease = stack.enter_context(budget.reserve("cost_usd", cost))
            proposals = self.proposal_provider.propose(
                source_text=document.source_text,
                candidate_text=candidate.text,
                region=region,
                protected_spans=document.protected_spans,
                seed=rng.randrange(0, 2**31),
                limit=self.config.candidates_per_node,
            )
            if model_lease is not None:
                model_lease.settle(1)
            if cost_lease is not None:
                cost_lease.settle(self.proposal_provider.estimated_cost_usd)
        if time.monotonic() > deadline:
            raise BudgetExhaustedError("runtime budget exhausted during proposal call")
        return proposals

    @staticmethod
    def _eligible_region(
        candidate: SearchCandidate, localization: LocalizationResult
    ) -> int | None:
        for index, region in enumerate(localization.regions):
            if all(
                operation.end <= region.start or operation.start >= region.end
                for operation in candidate.operations
            ):
                return index
        return None

    def _candidate_from_proposal(
        self,
        *,
        document: Document,
        parent: SearchCandidate,
        proposal: ProposedReplacement,
        region_index: int,
        localization: LocalizationResult,
        budget: BudgetAccount,
        policy: FidelityPolicy,
    ) -> tuple[SearchCandidate | None, FidelityReport, str]:
        region = localization.regions[region_index]
        operation = Operation(
            start=region.start,
            end=region.end,
            text=proposal.replacement,
            original=document.source_text[region.start : region.end],
            operator=proposal.operator,
            reason=proposal.reason,
        )
        operations = tuple(sorted((*parent.operations, operation), key=lambda item: item.start))
        report = self.fidelity.evaluate(document, operations, policy, budget, localization.regions)
        content_hash = normalized_content_hash(report.candidate_text)
        if not report.passed:
            return None, report, content_hash
        candidate = SearchCandidate(
            candidate_id=candidate_id(content_hash, operations),
            parent_id=parent.candidate_id,
            depth=parent.depth + 1,
            text=report.candidate_text,
            operations=operations,
            content_hash=content_hash,
            metrics=CandidateMetrics(
                detector_risk=parent.metrics.detector_risk,
                char_edit_ratio=report.char_edit_ratio,
                token_edit_ratio=report.token_edit_ratio,
                length_drift_ratio=report.length_drift_ratio,
                model_calls=parent.metrics.model_calls + int(self.proposal_provider.uses_model),
                detector_queries=parent.metrics.detector_queries,
                monetary_cost=parent.metrics.monetary_cost
                + float(self.proposal_provider.estimated_cost_usd),
            ),
            fidelity_unavailable=tuple(
                gate.gate_id for gate in report.gates if gate.status == "unavailable"
            ),
        )
        return candidate, report, content_hash

    def _score_candidate(
        self,
        candidate: SearchCandidate,
        *,
        target: ReductionTarget,
        baseline: DetectorScore | None,
        budget: BudgetAccount,
        deadline: float,
    ) -> SearchCandidate:
        score = self._score(candidate.text, budget, deadline)
        if (
            score is not None
            and target.threshold is not None
            and score.threshold != target.threshold
        ):
            score = score.model_copy(
                update={"threshold": target.threshold, "threshold_is_calibrated": False}
            )
        return candidate.model_copy(
            update={
                "score": score,
                "metrics": candidate.metrics.model_copy(
                    update={
                        "detector_risk": score.risk if score else 0.0,
                        "detector_queries": candidate.metrics.detector_queries
                        + int(score is not None),
                    }
                ),
                "target_met": target_met(
                    target,
                    baseline,
                    score,
                    default_minimum_improvement=self.config.minimum_score_improvement,
                ),
            }
        )

    def run(
        self,
        document: Document,
        target: ReductionTarget,
        budget: BudgetAccount,
        policy: FidelityPolicy,
        *,
        resume: bool = False,
    ) -> TargetedSearchResult:
        started = time.monotonic()
        rng = random.Random(self.config.seed)
        warnings: list[str] = []
        rejected: list[RejectedCandidate] = []
        successes: list[SearchCandidate] = []
        archive: list[SearchCandidate] = []
        proposals_generated = 0
        deduplicated = 0
        candidates_scored = 0
        rejection_counts: dict[str, int] = {}
        stopping_reason = "search depth exhausted"
        localization = LocalizationResult((), "", "")
        checkpoint: TargetedCheckpoint | None = None

        missing_fidelity = self.fidelity.missing_required_capabilities(policy)
        if missing_fidelity:
            return self._terminal(
                state="unsupported",
                baseline=None,
                selected=None,
                archive=(),
                successes=(),
                rejected=(),
                budget=budget,
                started=started,
                localization=localization,
                proposals_generated=0,
                deduplicated=0,
                candidates_scored=0,
                rejection_counts={},
                stopping_reason=(
                    "missing required fidelity capability: " + ", ".join(missing_fidelity)
                ),
                warnings=(),
                checkpoint=None,
            )

        if resume:
            if self.checkpoint_store is None:
                raise ValueError("resume requires a checkpoint store")
            saved = self.checkpoint_store.load()
            if saved is None:
                raise ValueError("no targeted-search checkpoint is available")
            checkpoint = TargetedCheckpoint.model_validate(saved)
            expected_config = TargetedCheckpoint.hash_config(self.config.model_dump_json())
            if checkpoint.source_hash != document.source_sha256:
                raise ValueError("checkpoint source hash does not match document")
            if checkpoint.config_hash != expected_config:
                raise ValueError("checkpoint config hash does not match configuration")
            self._restore_usage(budget, checkpoint.budget_usage)
            rng.setstate(_tuplify(checkpoint.rng_state))  # type: ignore[arg-type]
            archive.extend(checkpoint.pareto_archive)
            successes.extend(checkpoint.successful_archive)
            rejected.extend(checkpoint.rejected_candidates)
            localization = LocalizationResult(
                regions=checkpoint.localized_regions,
                selected_mode=checkpoint.localizer_mode,
                reason=checkpoint.localizer_reason,
                warnings=checkpoint.localizer_warnings,
            )

        remaining_runtime_ms = int(max(Decimal(0), budget.remaining("runtime_ms")))
        deadline = time.monotonic() + remaining_runtime_ms / 1000

        if checkpoint is not None:
            baseline = checkpoint.baseline
        else:
            try:
                baseline = self._score(document.source_text, budget, deadline)
            except BudgetExhaustedError as error:
                return self._terminal(
                    state="abstained",
                    baseline=None,
                    selected=None,
                    archive=(),
                    successes=(),
                    rejected=(),
                    budget=budget,
                    started=started,
                    localization=localization,
                    proposals_generated=0,
                    deduplicated=0,
                    candidates_scored=0,
                    rejection_counts={},
                    stopping_reason=str(error),
                    warnings=(),
                    checkpoint=None,
                )

        unsupported = unsupported_reason(target, baseline)
        if unsupported is not None:
            return self._terminal(
                state="unsupported",
                baseline=baseline,
                selected=None,
                archive=(),
                successes=(),
                rejected=(),
                budget=budget,
                started=started,
                localization=localization,
                proposals_generated=0,
                deduplicated=0,
                candidates_scored=0,
                rejection_counts={},
                stopping_reason=unsupported,
                warnings=(),
                checkpoint=None,
            )

        source = SearchCandidate.source(document.source_text, baseline)
        source = source.model_copy(
            update={
                "target_met": target_met(
                    target,
                    baseline,
                    baseline,
                    default_minimum_improvement=self.config.minimum_score_improvement,
                )
            }
        )
        if source.target_met:
            return self._terminal(
                state=successful_state(target, baseline),
                baseline=baseline,
                selected=source,
                archive=(source,),
                successes=(source,),
                rejected=(),
                budget=budget,
                started=started,
                localization=localization,
                proposals_generated=0,
                deduplicated=0,
                candidates_scored=0,
                rejection_counts={},
                stopping_reason="source already met the requested target",
                warnings=(),
                checkpoint=None,
            )

        if checkpoint is None:
            localizer = TargetedLocalizer(
                config=self.config,
                budget=budget,
                detector=self.detector,
                pll=self.pll,
                deadline=deadline,
            )
            try:
                localization = localizer.localize(document, baseline)
            except (DependencyUnavailableError, BudgetExhaustedError) as error:
                return self._terminal(
                    state="unsupported"
                    if isinstance(error, DependencyUnavailableError)
                    else "abstained",
                    baseline=baseline,
                    selected=None,
                    archive=(),
                    successes=(),
                    rejected=(),
                    budget=budget,
                    started=started,
                    localization=localization,
                    proposals_generated=0,
                    deduplicated=0,
                    candidates_scored=0,
                    rejection_counts={},
                    stopping_reason=str(error),
                    warnings=(),
                    checkpoint=None,
                )
        warnings.extend(localization.warnings)
        if not localization.regions:
            return self._terminal(
                state="abstained",
                baseline=baseline,
                selected=None,
                archive=(),
                successes=(),
                rejected=(),
                budget=budget,
                started=started,
                localization=localization,
                proposals_generated=0,
                deduplicated=0,
                candidates_scored=0,
                rejection_counts={},
                stopping_reason="localization found no editable regions",
                warnings=tuple(warnings),
                checkpoint=None,
            )

        beam: tuple[SearchCandidate, ...] = (source,)
        start_depth = 1
        if checkpoint is not None:
            beam = checkpoint.beam
            start_depth = checkpoint.current_depth + 1

        seen_hashes = {source.content_hash, *(candidate.content_hash for candidate in archive)}
        best_risk = baseline.risk if baseline is not None else float("inf")
        stagnant = 0

        for depth in range(start_depth, self.config.max_search_depth + 1):
            if time.monotonic() >= deadline:
                stopping_reason = "runtime budget exhausted"
                break
            depth_candidates: list[SearchCandidate] = []
            budget_stopped = False
            try:
                with budget.reserve("rounds") as round_lease:
                    for parent in beam:
                        region_index = self._eligible_region(parent, localization)
                        if region_index is None:
                            continue
                        try:
                            if budget.remaining("candidates") < 1:
                                raise BudgetExhaustedError(
                                    "budget exhausted for candidates before proposal generation"
                                )
                            proposals = self._propose(
                                document=document,
                                candidate=parent,
                                region_index=region_index,
                                localization=localization,
                                budget=budget,
                                rng=rng,
                                deadline=deadline,
                            )
                        except BudgetExhaustedError as error:
                            stopping_reason = str(error)
                            budget_stopped = True
                            break
                        proposals_generated += len(proposals)
                        for proposal in proposals:
                            try:
                                with budget.reserve("candidates") as candidate_lease:
                                    candidate, report, content_hash = self._candidate_from_proposal(
                                        document=document,
                                        parent=parent,
                                        proposal=proposal,
                                        region_index=region_index,
                                        localization=localization,
                                        budget=budget,
                                        policy=policy,
                                    )
                                    if content_hash in seen_hashes:
                                        deduplicated += 1
                                        continue
                                    seen_hashes.add(content_hash)
                                    if candidate is None:
                                        reason = (
                                            report.rejection_reason or "hard fidelity gate failed"
                                        )
                                        category = reason.split(":", 1)[0]
                                        rejection_counts[category] = (
                                            rejection_counts.get(category, 0) + 1
                                        )
                                        rejected.append(
                                            RejectedCandidate(
                                                parent_id=parent.candidate_id,
                                                content_hash=content_hash,
                                                operator=proposal.operator,
                                                reason=reason,
                                            )
                                        )
                                        candidate_lease.settle(1)
                                        continue
                                    candidate = self._score_candidate(
                                        candidate,
                                        target=target,
                                        baseline=baseline,
                                        budget=budget,
                                        deadline=deadline,
                                    )
                                    candidate_lease.settle(1)
                            except BudgetExhaustedError as error:
                                stopping_reason = str(error)
                                budget_stopped = True
                                break
                            candidates_scored += int(candidate.score is not None)
                            depth_candidates.append(candidate)
                            archive.append(candidate)
                            if candidate.target_met:
                                successes.append(candidate)
                        if budget_stopped:
                            break
                    round_lease.settle(1)
            except BudgetExhaustedError as error:
                stopping_reason = str(error)
                budget_stopped = True

            current_frontier = pareto_frontier(archive)
            checkpoint = self._checkpoint(
                document=document,
                depth=depth,
                beam=tuple(depth_candidates),
                archive=current_frontier,
                successes=tuple(successes),
                rejected=tuple(rejected),
                budget=budget,
                rng=rng,
                baseline=baseline,
                localization=localization,
            )
            if self.checkpoint_store is not None:
                self.checkpoint_store.save(checkpoint.model_dump(mode="json"))

            if successes:
                selected = min(successes, key=lexicographic_key)
                return self._terminal(
                    state=successful_state(target, selected.score),
                    baseline=baseline,
                    selected=selected,
                    archive=current_frontier,
                    successes=tuple(sorted(successes, key=lexicographic_key)),
                    rejected=tuple(rejected),
                    budget=budget,
                    started=started,
                    localization=localization,
                    proposals_generated=proposals_generated,
                    deduplicated=deduplicated,
                    candidates_scored=candidates_scored,
                    rejection_counts=rejection_counts,
                    stopping_reason="minimum-edit success found at the shallowest successful depth",
                    warnings=tuple(warnings),
                    checkpoint=checkpoint,
                )

            if budget_stopped:
                break
            if not depth_candidates:
                stopping_reason = "no fidelity-valid candidates remained"
                break

            round_best_risk = min(candidate.metrics.detector_risk for candidate in depth_candidates)
            if best_risk - round_best_risk < self.config.minimum_score_improvement:
                stagnant += 1
            else:
                stagnant = 0
                best_risk = round_best_risk
            if stagnant >= self.config.stagnation_rounds:
                stopping_reason = "stagnation limit reached"
                break
            beam = diverse_beam(
                pareto_frontier(depth_candidates),
                width=self.config.beam_width,
                similarity_ceiling=self.config.diversity_similarity_ceiling,
            )

        frontier = pareto_frontier(archive)
        selected_candidate = min(frontier, key=lexicographic_key) if frontier else None
        state: ResultState = "abstained"
        if selected_candidate is not None and selected_candidate.score is None:
            state = "rewritten_unverified"
        return self._terminal(
            state=state,
            baseline=baseline,
            selected=selected_candidate,
            archive=frontier,
            successes=tuple(successes),
            rejected=tuple(rejected),
            budget=budget,
            started=started,
            localization=localization,
            proposals_generated=proposals_generated,
            deduplicated=deduplicated,
            candidates_scored=candidates_scored,
            rejection_counts=rejection_counts,
            stopping_reason=stopping_reason,
            warnings=tuple(warnings),
            checkpoint=checkpoint,
        )

    def _checkpoint(
        self,
        *,
        document: Document,
        depth: int,
        beam: tuple[SearchCandidate, ...],
        archive: tuple[SearchCandidate, ...],
        successes: tuple[SearchCandidate, ...],
        rejected: tuple[RejectedCandidate, ...],
        budget: BudgetAccount,
        rng: random.Random,
        baseline: DetectorScore | None,
        localization: LocalizationResult,
    ) -> TargetedCheckpoint:
        return TargetedCheckpoint(
            source_hash=document.source_sha256,
            config_hash=TargetedCheckpoint.hash_config(self.config.model_dump_json()),
            current_depth=depth,
            beam=beam,
            pareto_archive=archive,
            successful_archive=successes,
            rejected_candidates=rejected,
            budget_usage=budget.usage(),
            rng_state=_listify(rng.getstate()),  # type: ignore[arg-type]
            baseline=baseline,
            localized_regions=localization.regions,
            localizer_mode=localization.selected_mode,
            localizer_reason=localization.reason,
            localizer_warnings=localization.warnings,
            detector_id=self.detector.id if self.detector else None,
            detector_version=self.detector.version if self.detector else None,
        )

    @staticmethod
    def _restore_usage(budget: BudgetAccount, usage: BudgetUsage) -> None:
        counters: tuple[tuple[CounterName, Decimal | int], ...] = (
            ("runtime_ms", usage.runtime_ms),
            ("model_calls", usage.model_calls),
            ("detector_queries", usage.detector_queries),
            ("cost_usd", usage.cost_usd),
            ("candidates", usage.candidates),
            ("rounds", usage.rounds),
        )
        for counter, amount in counters:
            if amount:
                with budget.reserve(counter, amount) as lease:
                    lease.settle(amount)

    def _terminal(
        self,
        *,
        state: ResultState,
        baseline: DetectorScore | None,
        selected: SearchCandidate | None,
        archive: tuple[SearchCandidate, ...],
        successes: tuple[SearchCandidate, ...],
        rejected: tuple[RejectedCandidate, ...],
        budget: BudgetAccount,
        started: float,
        localization: LocalizationResult,
        proposals_generated: int,
        deduplicated: int,
        candidates_scored: int,
        rejection_counts: dict[str, int],
        stopping_reason: str,
        warnings: tuple[str, ...],
        checkpoint: TargetedCheckpoint | None,
    ) -> TargetedSearchResult:
        settle_runtime(budget, started)
        if checkpoint is not None:
            checkpoint = checkpoint.model_copy(update={"budget_usage": budget.usage()})
            if self.checkpoint_store is not None:
                self.checkpoint_store.save(checkpoint.model_dump(mode="json"))
        selection_reason = ""
        if selected is not None:
            selection_reason = (
                "selected by hard feasibility, target status, edit ratios, residual risk, "
                "resource use, then deterministic content hash"
            )
        residual = {
            "verified_below_threshold": (
                "The named detector's score fell below its configured threshold after the rewrite."
            ),
            "verified_reduction_only": (
                "The named detector measured the requested score reduction after the rewrite."
            ),
            "surrogate_reduced": (
                "The surrogate score decreased while the rewrite changed token, n-gram, "
                "sentence, and model-probability patterns."
            ),
            "rewritten_unverified": (
                "The rewrite changed token, n-gram, sentence, and model-probability "
                "patterns without a detector measurement."
            ),
            "abstained": (
                "No candidate reconciled the requested reduction and fidelity constraints "
                "inside the budgets."
            ),
            "unsupported": "A capability required by the requested evidence state was unavailable.",
            "sanitized": "Not applicable to targeted constrained search.",
        }[state]
        return TargetedSearchResult(
            state=state,
            baseline=baseline,
            selected=selected,
            frontier=archive,
            successes=successes,
            rejected=rejected,
            usage=budget.usage(
                char_edit_ratio=selected.metrics.char_edit_ratio if selected else 0,
                token_edit_ratio=selected.metrics.token_edit_ratio if selected else 0,
                length_drift_ratio=selected.metrics.length_drift_ratio if selected else 0,
            ),
            trace=TargetedSearchTrace(
                localizer_mode=localization.selected_mode,
                localizer_reason=localization.reason,
                localized_regions=tuple(
                    (region.start, region.end, region.risk) for region in localization.regions
                ),
                proposals_generated=proposals_generated,
                candidates_deduplicated=deduplicated,
                candidates_scored=candidates_scored,
                rejection_counts=rejection_counts,
                stopping_reason=stopping_reason,
                selection_reason=selection_reason,
                warnings=warnings,
            ),
            residual_risk=residual,
            checkpoint=checkpoint,
        )
