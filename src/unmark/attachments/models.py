"""Versioned contracts for binary attachment inspection and cleaning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from unmark.core.spans import StrictModel

AttachmentMediaType = Literal[
    "image/png",
    "image/jpeg",
    "image/svg+xml",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.oasis.opendocument.text",
    "text/html",
    "text/markdown",
]
EvidenceKind = Literal[
    "c2pa_verified",
    "c2pa_invalid",
    "unsigned_ai_metadata",
    "soft_binding_declared",
    "pixel_watermark",
    "not_detected",
    "unknown",
    "unsupported",
    "embedded_metadata",
]
AttachmentResultState = Literal[
    "c2pa_verified",
    "c2pa_invalid",
    "unsigned_ai_metadata",
    "soft_binding_declared",
    "pixel_watermark",
    "not_detected",
    "unknown",
    "unsupported",
    "embedded_metadata",
    "removed_verified",
    "removed_unverified",
    "failed",
]
EvidenceConfidence = Literal["cryptographic", "container", "unsigned", "declaration", "unknown"]
VendorAttribution = Literal["anthropic_verified", "vendor_unverified", "not_applicable"]


class AttachmentLimits(StrictModel):
    """Security and resource bounds applied before or during parsing."""

    max_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    max_pixels: int = Field(default=100_000_000, ge=1)
    max_chunks: int = Field(default=10_000, ge=1)
    max_metadata_bytes: int = Field(default=4 * 1024 * 1024, ge=1)
    max_decompressed_metadata_bytes: int = Field(default=1024 * 1024, ge=1)
    max_xml_nodes: int = Field(default=100_000, ge=1)
    max_xml_depth: int = Field(default=128, ge=1)


class AttachmentSummary(StrictModel):
    """Stable identity and byte-sniffed type for one attachment."""

    schema_version: Literal["1"] = "1"
    media_type: AttachmentMediaType
    byte_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AttachmentEvidence(StrictModel):
    """One independently explainable provenance observation."""

    schema_version: Literal["1"] = "1"
    kind: EvidenceKind
    source: str
    description: str
    confidence: EvidenceConfidence
    removable: bool = False
    vendor: str | None = None
    vendor_attribution: VendorAttribution = "not_applicable"
    metadata_key: str | None = None
    validation_state: str | None = None
    validation_codes: tuple[str, ...] = ()


class AttachmentAction(StrictModel):
    """One targeted container mutation."""

    schema_version: Literal["1"] = "1"
    kind: Literal["removed_container", "rewrote_metadata", "redacted_metadata"]
    source: str
    description: str
    bytes_before: int = Field(ge=0)
    bytes_after: int = Field(ge=0)


class FidelityInvariant(StrictModel):
    """A machine-checked property required before output publication."""

    schema_version: Literal["1"] = "1"
    name: str
    passed: bool
    description: str


class AttachmentReport(StrictModel):
    """Inspection or clean report shared by all product surfaces."""

    schema_version: Literal["1"] = "1"
    kind: Literal["attachment_inspect", "attachment_clean"]
    state: AttachmentResultState
    source: AttachmentSummary
    output: AttachmentSummary | None = None
    evidence: tuple[AttachmentEvidence, ...] = ()
    actions: tuple[AttachmentAction, ...] = ()
    invariants: tuple[FidelityInvariant, ...] = ()
    c2pa_verifier: str
    residual_risk: tuple[str, ...]
    notes: tuple[str, ...] = ()


class AttachmentCleanOutcome(StrictModel):
    """A clean report plus unpublished output bytes.

    ``output_bytes`` is present only when every required verification passes.
    Keeping publication outside this model lets the filesystem CLI use atomic
    writes while HTTP and extension adapters can stream the exact same bytes.
    """

    report: AttachmentReport
    output_bytes: bytes | None = Field(default=None, repr=False)


@dataclass(frozen=True)
class FormatInspection:
    """Internal format adapter inspection result."""

    evidence: tuple[AttachmentEvidence, ...]
    c2pa_hint: bool
    c2pa_source: str | None


@dataclass(frozen=True)
class FormatClean:
    """Internal format adapter cleanup result."""

    data: bytes
    actions: tuple[AttachmentAction, ...]
    invariants: tuple[FidelityInvariant, ...]
