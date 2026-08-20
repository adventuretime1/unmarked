"""Target semantics and constraint-aware candidate ordering."""

from __future__ import annotations

from unmark.core.results import ResultState
from unmark.core.targets import ReductionTarget
from unmark.detectors.protocols import DetectorScore


def threshold_met(score: DetectorScore, threshold: float) -> bool:
    if score.risk_direction == "higher":
        return score.score <= threshold
    return score.score >= threshold


def reduction_amount(baseline: DetectorScore, candidate: DetectorScore) -> float:
    return baseline.risk - candidate.risk


def target_met(
    target: ReductionTarget,
    baseline: DetectorScore | None,
    candidate: DetectorScore | None,
    *,
    default_minimum_improvement: float = 0.0,
) -> bool:
    if candidate is None:
        return False
    if target.mode == "verify_below_threshold":
        if candidate.evidence_kind == "surrogate":
            return False
        threshold = target.threshold if target.threshold is not None else candidate.threshold
        return threshold is not None and threshold_met(candidate, threshold)
    if target.mode == "relative_reduction":
        if baseline is None:
            return False
        required = target.min_score_reduction
        if required is None:
            required = default_minimum_improvement
        reduction = reduction_amount(baseline, candidate)
        return reduction > 0.0 and reduction >= required
    if target.mode == "minimize_surrogate":
        if baseline is None or candidate.evidence_kind != "surrogate":
            return False
        required = target.min_score_reduction
        if required is None:
            required = default_minimum_improvement
        return reduction_amount(baseline, candidate) > required
    if target.mode == "stress_test":
        return False
    return False


def successful_state(target: ReductionTarget, score: DetectorScore | None) -> ResultState:
    if score is None:
        return "rewritten_unverified"
    if score.evidence_kind == "surrogate":
        return "surrogate_reduced"
    if target.mode == "verify_below_threshold":
        return "verified_below_threshold"
    return "verified_reduction_only"


def unsupported_reason(target: ReductionTarget, baseline: DetectorScore | None) -> str | None:
    if target.mode == "sanitize_only":
        return "targeted-search does not implement deterministic sanitation"
    if target.mode == "verify_below_threshold":
        if baseline is None:
            return "below-threshold measurement requires a configured versioned detector"
        if baseline.evidence_kind == "surrogate":
            return "a surrogate cannot satisfy a named-detector threshold request"
    if target.mode == "minimize_surrogate" and (
        baseline is None or baseline.evidence_kind != "surrogate"
    ):
        return "surrogate reduction requires a configured surrogate scorer"
    if len(target.detector_ids) > 1:
        return (
            "this targeted-search instance supports one detector; use orchestration for ensembles"
        )
    if target.detector_ids and (
        baseline is None or baseline.detector_id not in target.detector_ids
    ):
        return "the configured scorer does not match the requested detector ID"
    return None
