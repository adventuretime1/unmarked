"""Normalized detector contracts used by search strategies.

Adapters translate vendor- or research-specific responses into these types.  Core
search code never imports a detector implementation or model library directly.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from unmark.core.spans import StrictModel
from unmark.core.targets import ScoreEvidence

EvidenceKind = Literal["official", "research", "surrogate"]
RiskDirection = Literal["higher", "lower"]


class ScoreContribution(StrictModel):
    """A detector contribution aligned to exact source character offsets."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    score: float
    token: str | None = None


class DetectorScore(StrictModel):
    """Complete, normalized evidence from one detector or surrogate."""

    detector_id: str
    detector_version: str
    evidence_kind: EvidenceKind
    score: float
    threshold: float | None = None
    calibrated_fpr: float | None = Field(default=None, ge=0.0, le=1.0)
    token_count: int = Field(default=0, ge=0)
    contributions: tuple[ScoreContribution, ...] = ()
    applicability_warnings: tuple[str, ...] = ()
    risk_direction: RiskDirection = "higher"
    threshold_is_calibrated: bool = True

    @property
    def risk(self) -> float:
        """Risk normalized so a lower number is always better."""
        return self.score if self.risk_direction == "higher" else -self.score

    def to_evidence(self) -> ScoreEvidence:
        return ScoreEvidence(
            scorer_id=self.detector_id,
            scorer_version=self.detector_version,
            kind=self.evidence_kind,
            score=self.score,
            threshold=self.threshold,
            threshold_provenance=(
                "calibrated"
                if self.threshold is not None and self.threshold_is_calibrated
                else "user_supplied_uncalibrated"
                if self.threshold is not None
                else None
            ),
            calibrated_fpr=self.calibrated_fpr,
            token_count=self.token_count,
        )


@runtime_checkable
class Detector(Protocol):
    """A side-effect-free detector adapter.

    The caller, not the adapter, owns query and cost reservation.  Adapters may
    expose token contributions, but must use exact character offsets when they do.
    """

    id: str
    version: str

    def score(self, text: str) -> DetectorScore: ...
