"""Stateless attachment application service."""

from __future__ import annotations

from unmark.attachments.c2pa import inspect_c2pa
from unmark.attachments.common import BinaryAttachment, make_attachment, summary_for
from unmark.attachments.formats import clean_format, inspect_format
from unmark.attachments.models import (
    AttachmentCleanOutcome,
    AttachmentEvidence,
    AttachmentLimits,
    AttachmentReport,
    AttachmentResultState,
)

_BASE_RESIDUAL_RISK = (
    "Removing embedded metadata does not prove human authorship and does not affect "
    "provider-side logs, retrieval matches, or unrelated classifiers.",
    "Pixel-watermark detection is not implemented in this release; absence of embedded "
    "evidence is not evidence that no pixel-domain signal exists.",
)
_STATE_PRIORITY: tuple[AttachmentResultState, ...] = (
    "c2pa_invalid",
    "c2pa_verified",
    "soft_binding_declared",
    "unsigned_ai_metadata",
    "pixel_watermark",
    "unknown",
    "unsupported",
    "embedded_metadata",
    "not_detected",
)


def _state_for(evidence: tuple[AttachmentEvidence, ...]) -> AttachmentResultState:
    kinds = {item.kind for item in evidence}
    for state in _STATE_PRIORITY:
        if state in kinds:
            return state
    return "not_detected"


def _deduplicate(evidence: tuple[AttachmentEvidence, ...]) -> tuple[AttachmentEvidence, ...]:
    result: list[AttachmentEvidence] = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in evidence:
        key = (item.kind, item.source, item.metadata_key)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)


def _inspect_parts(
    data: bytes, limits: AttachmentLimits
) -> tuple[BinaryAttachment, tuple[AttachmentEvidence, ...], str]:
    attachment = make_attachment(data, limits)
    format_result = inspect_format(data, attachment.media_type, limits)
    c2pa_result = inspect_c2pa(
        data,
        attachment.media_type,
        embedded_hint=format_result.c2pa_hint,
        source=format_result.c2pa_source,
    )
    evidence = _deduplicate(format_result.evidence + c2pa_result.evidence)
    if not evidence:
        evidence = (
            AttachmentEvidence(
                kind="not_detected",
                source="attachment",
                description="No targeted embedded provenance metadata was detected.",
                confidence="container",
                removable=False,
            ),
        )
    return attachment, evidence, c2pa_result.verifier


def inspect_attachment(data: bytes, *, limits: AttachmentLimits | None = None) -> AttachmentReport:
    """Inspect one SVG/PNG/JPEG attachment without writing or making network calls."""
    effective_limits = limits or AttachmentLimits()
    attachment, evidence, verifier = _inspect_parts(data, effective_limits)
    notes: list[str] = []
    if "unavailable" in verifier:
        notes.append(
            "Install the 'attachments' extra to distinguish valid, invalid, and trusted "
            "C2PA signatures."
        )
    return AttachmentReport(
        kind="attachment_inspect",
        state=_state_for(evidence),
        source=attachment.summary,
        evidence=evidence,
        c2pa_verifier=verifier,
        residual_risk=_BASE_RESIDUAL_RISK,
        notes=tuple(notes),
    )


def clean_attachment(
    data: bytes, *, limits: AttachmentLimits | None = None
) -> AttachmentCleanOutcome:
    """Inspect, transform, re-inspect, and return bytes only after verification."""
    effective_limits = limits or AttachmentLimits()
    attachment, before_evidence, before_verifier = _inspect_parts(data, effective_limits)
    transformed = clean_format(data, attachment.media_type, effective_limits)
    invariants_passed = all(item.passed for item in transformed.invariants)

    _output_attachment, after_evidence, after_verifier = _inspect_parts(
        transformed.data, effective_limits
    )
    surviving = tuple(item for item in after_evidence if item.kind != "not_detected")
    success = invariants_passed and not surviving
    if not success:
        reasons: list[str] = []
        if not invariants_passed:
            reasons.append("a rendered-content fidelity invariant failed")
        if surviving:
            reasons.append(
                "targeted evidence survived re-inspection: "
                + ", ".join(sorted({item.kind for item in surviving}))
            )
        report = AttachmentReport(
            kind="attachment_clean",
            state="failed",
            source=attachment.summary,
            evidence=before_evidence,
            actions=transformed.actions,
            invariants=transformed.invariants,
            c2pa_verifier=after_verifier,
            residual_risk=_BASE_RESIDUAL_RISK,
            notes=tuple(reasons),
        )
        return AttachmentCleanOutcome(report=report)

    state: AttachmentResultState = "removed_verified" if transformed.actions else "not_detected"
    notes: tuple[str, ...] = ()
    if "unavailable" in before_verifier and any(
        item.kind in {"unknown", "c2pa_invalid", "c2pa_verified"} for item in before_evidence
    ):
        notes = (
            "The C2PA container's removal was structurally verified, but its original "
            "signature/signer could not be classified because the official verifier "
            "was unavailable.",
        )
    report = AttachmentReport(
        kind="attachment_clean",
        state=state,
        source=attachment.summary,
        output=summary_for(transformed.data, attachment.media_type),
        evidence=before_evidence,
        actions=transformed.actions,
        invariants=transformed.invariants,
        c2pa_verifier=after_verifier,
        residual_risk=_BASE_RESIDUAL_RISK,
        notes=notes,
    )
    return AttachmentCleanOutcome(report=report, output_bytes=transformed.data)
