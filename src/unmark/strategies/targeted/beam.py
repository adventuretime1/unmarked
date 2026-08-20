"""Candidate state, Pareto archiving, diversity, and checkpoints."""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import Field

from unmark.core.budgets import BudgetUsage
from unmark.core.operations import Operation, normalized_content_hash
from unmark.core.spans import StrictModel
from unmark.detectors.localization import TextRegion
from unmark.detectors.protocols import DetectorScore


class CandidateMetrics(StrictModel):
    detector_risk: float
    char_edit_ratio: float = Field(ge=0.0, le=1.0)
    token_edit_ratio: float = Field(ge=0.0, le=1.0)
    length_drift_ratio: float = Field(ge=0.0, le=1.0)
    model_calls: int = Field(default=0, ge=0)
    detector_queries: int = Field(default=0, ge=0)
    monetary_cost: float = Field(default=0.0, ge=0.0)
    semantic_distance: float = Field(default=0.0, ge=0.0)
    style_distance: float = Field(default=0.0, ge=0.0)

    @property
    def weighted_edit_cost(self) -> float:
        """Stable surface/length cost; effectiveness is never mixed into it."""
        return (
            0.55 * self.char_edit_ratio
            + 0.35 * self.token_edit_ratio
            + 0.10 * self.length_drift_ratio
        )


class SearchCandidate(StrictModel):
    candidate_id: str
    parent_id: str | None = None
    depth: int = Field(default=0, ge=0)
    text: str
    operations: tuple[Operation, ...] = ()
    content_hash: str
    score: DetectorScore | None = None
    metrics: CandidateMetrics
    target_met: bool = False
    fidelity_passed: bool = True
    fidelity_unavailable: tuple[str, ...] = ()

    @classmethod
    def source(cls, text: str, score: DetectorScore | None = None) -> SearchCandidate:
        content_hash = normalized_content_hash(text)
        return cls(
            candidate_id=f"candidate-{content_hash[:12]}",
            text=text,
            content_hash=content_hash,
            score=score,
            metrics=CandidateMetrics(
                detector_risk=score.risk if score else 0.0,
                char_edit_ratio=0.0,
                token_edit_ratio=0.0,
                length_drift_ratio=0.0,
            ),
        )


class RejectedCandidate(StrictModel):
    parent_id: str
    content_hash: str
    operator: str
    reason: str


def candidate_id(content_hash: str, operations: Sequence[Operation]) -> str:
    payload = (
        content_hash
        + "|"
        + "|".join(f"{op.start}:{op.end}:{op.operator}:{op.text}" for op in operations)
    )
    return f"candidate-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def lexicographic_key(candidate: SearchCandidate) -> tuple[object, ...]:
    """Constraint-aware ordering; intentionally not a weighted sum."""
    return (
        not candidate.fidelity_passed,
        not candidate.target_met,
        candidate.metrics.weighted_edit_cost,
        candidate.metrics.char_edit_ratio,
        candidate.metrics.token_edit_ratio,
        len(candidate.operations),
        candidate.metrics.detector_risk,
        candidate.metrics.semantic_distance,
        candidate.metrics.style_distance,
        candidate.metrics.model_calls,
        candidate.metrics.detector_queries,
        candidate.metrics.monetary_cost,
        candidate.content_hash,
    )


def dominates(left: SearchCandidate, right: SearchCandidate) -> bool:
    """Pareto dominance across effectiveness, fidelity, and resource dimensions."""
    left_values = (
        left.metrics.detector_risk,
        left.metrics.char_edit_ratio,
        left.metrics.token_edit_ratio,
        left.metrics.length_drift_ratio,
        left.metrics.model_calls,
        left.metrics.detector_queries,
        left.metrics.monetary_cost,
    )
    right_values = (
        right.metrics.detector_risk,
        right.metrics.char_edit_ratio,
        right.metrics.token_edit_ratio,
        right.metrics.length_drift_ratio,
        right.metrics.model_calls,
        right.metrics.detector_queries,
        right.metrics.monetary_cost,
    )
    return all(
        left_value <= right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ) and any(
        left_value < right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )


def pareto_frontier(candidates: Iterable[SearchCandidate]) -> tuple[SearchCandidate, ...]:
    unique = {candidate.content_hash: candidate for candidate in candidates}
    values = tuple(unique.values())
    frontier = [
        candidate
        for candidate in values
        if candidate.fidelity_passed
        and not any(
            other.content_hash != candidate.content_hash and dominates(other, candidate)
            for other in values
            if other.fidelity_passed
        )
    ]
    return tuple(sorted(frontier, key=lexicographic_key))


def text_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def diverse_beam(
    candidates: Iterable[SearchCandidate], *, width: int, similarity_ceiling: float
) -> tuple[SearchCandidate, ...]:
    selected: list[SearchCandidate] = []
    for candidate in sorted(candidates, key=lexicographic_key):
        if all(
            text_similarity(candidate.text, existing.text) < similarity_ceiling
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) == width:
            break
    if len(selected) < width:
        selected_hashes = {item.content_hash for item in selected}
        for candidate in sorted(candidates, key=lexicographic_key):
            if candidate.content_hash not in selected_hashes:
                selected.append(candidate)
                selected_hashes.add(candidate.content_hash)
            if len(selected) == width:
                break
    return tuple(selected)


class TargetedCheckpoint(StrictModel):
    schema_version: Literal["1"] = "1"
    source_hash: str
    config_hash: str
    current_depth: int = Field(ge=0)
    beam: tuple[SearchCandidate, ...]
    pareto_archive: tuple[SearchCandidate, ...]
    successful_archive: tuple[SearchCandidate, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    budget_usage: BudgetUsage
    rng_state: list[object]
    baseline: DetectorScore | None = None
    localized_regions: tuple[TextRegion, ...] = ()
    localizer_mode: str = ""
    localizer_reason: str = ""
    localizer_warnings: tuple[str, ...] = ()
    detector_id: str | None = None
    detector_version: str | None = None
    cache_key_versions: tuple[str, ...] = ("normalized-content-sha256-nfc-v1",)

    @staticmethod
    def hash_config(config_json: str) -> str:
        canonical = json.dumps(json.loads(config_json), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class MemoryCheckpointStore:
    def __init__(self) -> None:
        self.value: dict[str, object] | None = None

    def save(self, checkpoint: dict[str, object]) -> None:
        self.value = json.loads(json.dumps(checkpoint))

    def load(self) -> dict[str, object] | None:
        return None if self.value is None else json.loads(json.dumps(self.value))
