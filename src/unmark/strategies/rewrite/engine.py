"""The shared rewrite engine: prompt a model, validate every reply.

Both baselines call :class:`RewriteEngine` to turn one rewrite step into a set of
validated :class:`RewriteCandidate`. The engine owns the untrusted-output
discipline:

* it reserves model-call and cost budget around every ``generate``;
* it materializes each reply as a single source-anchored replace operation over
  the *step's* base text and validates it with the fidelity evaluator;
* it never trusts the model's text — a reply that changes a protected number,
  quotation, URL, code span, or locked span, or that busts an edit/length budget,
  is recorded as rejected and can never be selected.

The engine is detector-optional. When a detector is supplied it scores valid
candidates (reserving detector-query budget); otherwise candidates carry no score
and the run can only reach ``rewritten_unverified``.
"""

from __future__ import annotations

from decimal import Decimal
from difflib import SequenceMatcher

from unmark.core.budgets import BudgetAccount
from unmark.core.document import Document
from unmark.core.errors import BudgetExhaustedError, DependencyUnavailableError
from unmark.core.operations import Operation, normalized_content_hash
from unmark.core.policies import FidelityPolicy
from unmark.detectors.localization import TextRegion
from unmark.detectors.protocols import Detector, DetectorScore
from unmark.fidelity.protocols import BasicFidelityEvaluator
from unmark.models.protocols import ModelAdapter, ModelCompletion, ModelRequest
from unmark.strategies.rewrite.candidates import RewriteCandidate
from unmark.strategies.rewrite.config import RewriteConfig
from unmark.strategies.rewrite.prompts import RewriteStyle, build_rewrite_prompt


def diff_operations(source: str, candidate: str, *, operator: str) -> tuple[Operation, ...]:
    """Minimal source-anchored operations turning ``source`` into ``candidate``.

    A rewrite is validated as the *edits it actually makes*, not as one
    whole-document replace. Diffing keeps unchanged regions — including protected
    spans the model left alone — untouched, so the protected-span gate only fires
    when a change genuinely lands on protected content. Adjacent changed regions
    are emitted as separate replace/insert/delete operations over the exact source
    offsets that differ.
    """
    matcher = SequenceMatcher(a=source, b=candidate, autojunk=False)
    operations: list[Operation] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        replacement = candidate[j1:j2]
        if tag == "insert":
            # A pure insertion at offset i1 (i1 == i2).
            operations.append(
                Operation(
                    start=i1,
                    end=i1,
                    text=replacement,
                    original="",
                    operator=operator,
                    reason="rewrite insertion",
                )
            )
        else:  # "replace" or "delete"
            operations.append(
                Operation(
                    start=i1,
                    end=i2,
                    text=replacement,
                    original=source[i1:i2],
                    operator=operator,
                    reason="rewrite edit",
                )
            )
    return tuple(operations)


def _editable_regions(
    operations: tuple[Operation, ...], source_length: int
) -> tuple[TextRegion, ...]:
    """One non-empty editable region per changed operation.

    Each region must contain its operation (operation-locality) and cover at least
    one source character. A pure insertion is a zero-width edit, so its region is
    widened by one character toward whichever side has room. Protected-span
    checking works on the operations themselves, not these regions, so a region
    that borders protected text is harmless.
    """
    regions: list[TextRegion] = []
    for op in operations:
        start, end = op.start, op.end
        if start == end:
            if end < source_length:
                end += 1
            elif start > 0:
                start -= 1
            else:
                # An insertion into empty source: nothing to cover, but the region
                # must be non-empty; the source has no characters, so skip it and
                # rely on the operation-locality gate treating it as local below.
                continue
        regions.append(TextRegion(start=start, end=end, risk=0.0, mode="rewrite-edit"))
    return tuple(regions)


class RewriteStepResult:
    """Outcome of one rewrite step: the candidates and any rejections."""

    def __init__(
        self,
        *,
        candidates: tuple[RewriteCandidate, ...],
        rejected: tuple[RewriteCandidate, ...],
        prompt_only: str | None,
    ) -> None:
        self.candidates = candidates
        self.rejected = rejected
        #: Set only for the ``print-prompt`` backend: the exact rendered prompt.
        self.prompt_only = prompt_only


class RewriteEngine:
    """Prompts a model and validates the replies for one rewrite step."""

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
        self.fidelity = fidelity or BasicFidelityEvaluator()
        self.detector = detector
        self.estimated_cost_usd = estimated_cost_usd

    def _locked_spans(self, document: Document) -> tuple[str, ...]:
        return tuple(span.value for span in document.protected_spans if span.kind == "user_lock")

    def build_request(
        self, document: Document, base_text: str, *, style: RewriteStyle
    ) -> ModelRequest:
        prompt = build_rewrite_prompt(
            base_text,
            style=style,
            strength=self.config.strength,
            locked_spans=self._locked_spans(document),
            target_length_ratio=self.config.target_length_ratio,
            voice=self.config.voice,
        )
        return ModelRequest(
            prompt=prompt.user,
            system=prompt.system,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            seed=self.config.seed,
        )

    def _score(self, text: str, budget: BudgetAccount) -> DetectorScore | None:
        if self.detector is None:
            return None
        with budget.reserve("detector_queries") as lease:
            score = self.detector.score(text)
            lease.settle(1)
        return score

    def _validate(
        self,
        *,
        document: Document,
        text: str,
        origin: str,
        parent_id: str | None,
        budget: BudgetAccount,
        policy: FidelityPolicy,
        baseline: DetectorScore | None,
    ) -> RewriteCandidate:
        """Materialize ``text`` as minimal source-anchored edits and validate it.

        The reply is diffed against the source so only the regions that actually
        changed become operations; each changed region is its own editable
        :class:`TextRegion`. Unchanged text — including any protected span the
        model preserved — is never touched, so the protected-span gate fires only
        when an edit genuinely lands on protected content.
        """
        source = document.source_text
        content_hash = normalized_content_hash(text)
        if text == source:
            # A no-op reply: nothing changed, so there is no operation to build.
            return RewriteCandidate(
                candidate_id=f"noop:{content_hash[:16]}",
                parent_id=parent_id,
                origin=origin,
                text=text,
                operations=(),
                content_hash=content_hash,
                fidelity_passed=True,
                rejection_reason=None,
                char_edit_ratio=0.0,
                token_edit_ratio=0.0,
                length_drift_ratio=0.0,
                score=baseline,
            )
        operations = diff_operations(source, text, operator=origin)
        editable = _editable_regions(operations, len(source))
        report = self.fidelity.evaluate(document, operations, policy, budget, editable)
        candidate_id = f"{origin}:{content_hash[:16]}"
        if not report.passed:
            return RewriteCandidate(
                candidate_id=candidate_id,
                parent_id=parent_id,
                origin=origin,
                text=report.candidate_text,
                operations=operations,
                content_hash=content_hash,
                fidelity_passed=False,
                rejection_reason=report.rejection_reason,
                char_edit_ratio=report.char_edit_ratio,
                token_edit_ratio=report.token_edit_ratio,
                length_drift_ratio=report.length_drift_ratio,
            )
        score = self._score(report.candidate_text, budget)
        return RewriteCandidate(
            candidate_id=candidate_id,
            parent_id=parent_id,
            origin=origin,
            text=report.candidate_text,
            operations=operations,
            content_hash=content_hash,
            fidelity_passed=True,
            rejection_reason=None,
            char_edit_ratio=report.char_edit_ratio,
            token_edit_ratio=report.token_edit_ratio,
            length_drift_ratio=report.length_drift_ratio,
            score=score,
        )

    def step(
        self,
        *,
        document: Document,
        base_text: str,
        style: RewriteStyle,
        parent_id: str | None,
        budget: BudgetAccount,
        policy: FidelityPolicy,
        baseline: DetectorScore | None,
        count: int,
    ) -> RewriteStepResult:
        """Run one rewrite step and validate every reply.

        For the ``print-prompt`` backend, no model is called: the rendered prompt
        is returned on ``prompt_only`` and no candidates are produced.
        """
        request = self.build_request(document, base_text, style=style)
        origin = f"{self.config_origin(style)}"

        if not self.adapter.uses_network and self.adapter.id == "print-prompt":
            return RewriteStepResult(
                candidates=(),
                rejected=(),
                prompt_only=self.adapter.render(request),
            )

        completions = self._generate(request, count=count, budget=budget)
        seen: set[str] = set()
        candidates: list[RewriteCandidate] = []
        rejected: list[RewriteCandidate] = []
        for completion in completions:
            text = completion.text
            candidate = self._validate(
                document=document,
                text=text,
                origin=origin,
                parent_id=parent_id,
                budget=budget,
                policy=policy,
                baseline=baseline,
            )
            if candidate.content_hash in seen:
                continue
            seen.add(candidate.content_hash)
            if candidate.fidelity_passed:
                candidates.append(candidate)
            else:
                rejected.append(candidate)
        return RewriteStepResult(
            candidates=tuple(candidates),
            rejected=tuple(rejected),
            prompt_only=None,
        )

    def config_origin(self, style: RewriteStyle) -> str:
        return f"{self.adapter.id}:{style}"

    def _generate(
        self, request: ModelRequest, *, count: int, budget: BudgetAccount
    ) -> tuple[ModelCompletion, ...]:
        """Call the model once, reserving model-call and cost budget."""
        # A networked adapter must be paid for; refuse if the budget forbids it.
        if self.adapter.uses_network and budget.budget.max_model_calls <= 0:
            msg = "networked model calls are disabled by the run budget"
            raise DependencyUnavailableError(msg)
        try:
            with budget.reserve("model_calls", count) as call_lease:
                cost = self.estimated_cost_usd * count
                if cost > 0:
                    with budget.reserve("cost_usd", cost) as cost_lease:
                        completions = self.adapter.generate(request, count=count)
                        cost_lease.settle(cost)
                else:
                    completions = self.adapter.generate(request, count=count)
                call_lease.settle(count)
        except BudgetExhaustedError:
            raise
        return completions
