"""Reduction targets and score evidence.

Only ``sanitize_only`` is executable in this iteration. The remaining modes are
part of the frozen contract so that detector work in Phase 2 does not reshape the
request or report schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from unmark.core.spans import StrictModel

TargetMode = Literal[
    "verify_below_threshold",
    "relative_reduction",
    "minimize_surrogate",
    "sanitize_only",
    "stress_test",
]


class ReductionTarget(StrictModel):
    """What the user asked the engine to achieve."""

    schema_version: Literal["1"] = "1"
    mode: TargetMode = "sanitize_only"
    detector_ids: tuple[str, ...] = ()
    threshold: float | None = None
    min_score_reduction: float | None = Field(default=None, ge=0.0, le=1.0)
    aggregation: Literal["all", "any", "max_risk", "weighted"] = "all"


class ScoreEvidence(StrictModel):
    """A single detector or surrogate measurement.

    ``kind`` distinguishes a provider or scheme-specific detector measurement
    from research and surrogate scoring; renderers must keep those visually
    distinct. It does not verify the rewritten text. ``threshold_provenance``
    records ``user_supplied_uncalibrated`` when a user overrides a calibrated threshold.
    """

    schema_version: Literal["1"] = "1"
    scorer_id: str
    scorer_version: str
    kind: Literal["official", "research", "surrogate", "heuristic"]
    score: float
    threshold: float | None = None
    threshold_provenance: Literal["calibrated", "user_supplied_uncalibrated"] | None = None
    calibrated_fpr: float | None = Field(default=None, ge=0.0, le=1.0)
    token_count: int = Field(default=0, ge=0)


class StrategyDescriptor(StrictModel):
    """Static declaration used for eligibility before a strategy runs."""

    schema_version: Literal["1"] = "1"
    id: str
    version: str
    stability: Literal["production", "experimental", "research_only"]
    capabilities: frozenset[str] = frozenset()
    supported_languages: frozenset[str] = frozenset({"*"})
    supported_media_types: frozenset[str] = frozenset({"text/plain", "text/markdown"})
    invasiveness: Literal["none", "low", "medium", "high"]
