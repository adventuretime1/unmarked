"""The deterministic sanitation strategy.

Tier 0-1 only: no model calls, no detector queries, no network. Every change is a
source-anchored operation carrying the reason it was made.
"""

from __future__ import annotations

from unmark.core.budgets import BudgetAccount
from unmark.core.document import Document
from unmark.core.operations import (
    Operation,
    apply_operations,
    char_edit_ratio,
    length_drift_ratio,
    normalized_content_hash,
    validate_operations,
)
from unmark.core.policies import FidelityPolicy, UnicodePolicy
from unmark.core.results import CandidateResult
from unmark.core.targets import StrategyDescriptor
from unmark.inspect.unicode_scan import (
    UnicodeFinding,
    blocked_by_protection,
    inspect_text,
    sanitation_operations,
)

DESCRIPTOR = StrategyDescriptor(
    id="sanitation",
    version="1",
    stability="production",
    capabilities=frozenset(),
    invasiveness="none",
)


class SanitationOutcome:
    """Everything the report needs from one sanitation pass."""

    __slots__ = (
        "blocked",
        "candidate",
        "candidate_text",
        "findings",
        "operations",
    )

    def __init__(
        self,
        findings: tuple[UnicodeFinding, ...],
        operations: tuple[Operation, ...],
        blocked: tuple[UnicodeFinding, ...],
        candidate_text: str,
        candidate: CandidateResult,
    ) -> None:
        self.findings = findings
        self.operations = operations
        self.blocked = blocked
        self.candidate_text = candidate_text
        self.candidate = candidate


class SanitationStrategy:
    """Conservative Unicode sanitation.

    Satisfies :class:`~unmark.orchestration.protocols.Strategy`.
    """

    descriptor = DESCRIPTOR

    def __init__(self, unicode_policy: UnicodePolicy) -> None:
        self.unicode_policy = unicode_policy

    def run(
        self,
        document: Document,
        budget: BudgetAccount,
        policy: FidelityPolicy,
    ) -> CandidateResult:
        return self.execute(document, budget, policy).candidate

    def execute(
        self,
        document: Document,
        budget: BudgetAccount,
        policy: FidelityPolicy,
    ) -> SanitationOutcome:
        """Run sanitation and return findings, operations, and the candidate."""
        source = document.source_text
        findings = inspect_text(source, self.unicode_policy)

        operations = sanitation_operations(
            source, findings, self.unicode_policy, document.protected_spans
        )
        blocked = blocked_by_protection(findings, document.protected_spans)

        # Validate before applying so a malformed operation set fails loudly
        # rather than producing a silently wrong candidate.
        operations = validate_operations(operations, source)
        candidate_text = apply_operations(source, operations)

        char_ratio = char_edit_ratio(source, candidate_text, operations)
        drift = length_drift_ratio(source, candidate_text)

        # Rewrite ratios bound semantic damage. Safe/typographic canonicalization
        # is deterministic and can legitimately touch every encoded space in a
        # patterned carrier, so proportional rewrite caps do not apply here.

        with budget.reserve("candidates", 1) as lease:
            lease.settle(1)

        candidate = CandidateResult(
            candidate_id="c1",
            strategy_id=self.descriptor.id,
            operations=operations,
            content_sha256=normalized_content_hash(candidate_text),
            char_edit_ratio=char_ratio,
            token_edit_ratio=0.0,
            length_drift_ratio=drift,
            invariants_passed=True,
            accepted=True,
        )
        return SanitationOutcome(findings, operations, blocked, candidate_text, candidate)
