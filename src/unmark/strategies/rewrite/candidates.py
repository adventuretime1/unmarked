"""Rewrite candidates and the safe candidate-selection order.

A :class:`RewriteCandidate` is a whole-document rewrite materialized as a single
source-anchored replace operation, already run through fidelity validation.

Selection order
---------------
Unmarked prefers the smallest safe edit, in this strict lexicographic order:

1. hard fidelity gates pass (invalid candidates are never selectable);
2. the requested detector/surrogate target is met, if one was requested;
3. minimum character/token edit cost;
4. minimum length drift;
5. best (lowest) residual detector/surrogate risk;
6. lowest call/cost/runtime usage;
7. deterministic content-hash tie-breaker.

``bigram_jaccard_divergence`` is retained purely as a diagnostic; it never drives
normal selection.
"""

from __future__ import annotations

import re
from itertools import pairwise

from pydantic import Field

from unmark.core.operations import Operation
from unmark.core.spans import StrictModel
from unmark.detectors.protocols import DetectorScore


class RewriteCandidate(StrictModel):
    """A validated whole-document rewrite."""

    candidate_id: str
    parent_id: str | None = None
    origin: str = Field(description="How this candidate was produced, e.g. 'one-shot:lexical'.")
    text: str
    operations: tuple[Operation, ...]
    content_hash: str
    fidelity_passed: bool
    rejection_reason: str | None = None
    char_edit_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    token_edit_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    length_drift_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    score: DetectorScore | None = None
    target_met: bool = False
    model_calls: int = Field(default=0, ge=0)
    detector_queries: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @property
    def residual_risk(self) -> float:
        """Lower is better; ``+inf`` when no score is available."""
        return self.score.risk if self.score is not None else float("inf")


def selection_key(candidate: RewriteCandidate) -> tuple[object, ...]:
    """The lexicographic sort key implementing Unmarked's selection order.

    Only fidelity-valid candidates should be passed here; the leading term keeps a
    stray invalid candidate from ever sorting first regardless.
    """
    edit_cost = max(candidate.char_edit_ratio, candidate.token_edit_ratio)
    return (
        0 if candidate.fidelity_passed else 1,
        0 if candidate.target_met else 1,
        round(edit_cost, 6),
        round(candidate.length_drift_ratio, 6),
        round(candidate.residual_risk, 6),
        candidate.model_calls,
        candidate.detector_queries,
        round(candidate.cost_usd, 6),
        candidate.content_hash,
    )


def select_candidate(candidates: tuple[RewriteCandidate, ...]) -> RewriteCandidate | None:
    """Return the best fidelity-valid candidate, or ``None`` if none is valid."""
    valid = [candidate for candidate in candidates if candidate.fidelity_passed]
    if not valid:
        return None
    return min(valid, key=selection_key)


def _bigrams(text: str) -> set[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return {f"{a} {b}" for a, b in pairwise(tokens)}


def bigram_jaccard_divergence(source: str, candidate: str) -> float:
    """Word-bigram Jaccard divergence, in ``[0, 1]``.

    This metric is diagnostic only; Unmarked never selects on it.
    """
    source_bigrams = _bigrams(source)
    candidate_bigrams = _bigrams(candidate)
    if not source_bigrams and not candidate_bigrams:
        return 0.0
    union = source_bigrams | candidate_bigrams
    if not union:
        return 0.0
    intersection = source_bigrams & candidate_bigrams
    return 1.0 - len(intersection) / len(union)
