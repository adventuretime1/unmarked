"""Shared attachment detection and metadata marker helpers."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass

from unmark.attachments.models import AttachmentLimits, AttachmentMediaType, AttachmentSummary
from unmark.core.errors import UnsupportedError, ValidationError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"
C2PA_UUID = bytes.fromhex("6332706100110010800000aa00389b71")

AI_VALUE_MARKERS: tuple[bytes, ...] = (
    b"anthropic",
    b"claude",
    b"openai",
    b"chatgpt",
    b"dall-e",
    b"dalle",
    b"midjourney",
    b"stable diffusion",
    b"comfyui",
    b"automatic1111",
    b"invokeai",
    b"generative ai",
    b"ai-generated",
    b"ai generated",
    b"made with ai",
    b"synthid",
)
PROVENANCE_MARKERS: tuple[bytes, ...] = (
    b"trainedalgorithmicmedia",
    b"compositewithtrainedalgorithmicmedia",
    b"compositesynthetic",
    b"aisystemused",
    b"aisystemversionused",
    b"aipromptinformation",
    b"aipromptwritername",
    b"tc260:aigc",
    b"tc260.org.cn/ns/aigc",
)
SOFT_BINDING_MARKERS: tuple[bytes, ...] = (
    b"softbinding",
    b"soft-binding",
    b"soft_binding",
    b"trustmark",
    b"c2pa.watermarked",
)
AI_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "parameters",
        "prompt",
        "negative_prompt",
        "workflow",
        "comfyui",
        "sd-metadata",
        "invokeai_metadata",
        "generation_data",
        "ai_metadata",
        "aigc",
        "c2pa",
        "c2pa_chunk",
        "hf-job-id",
    }
)
_SVG_ROOT = re.compile(rb"<(?:[A-Za-z_][\w.-]*:)?svg(?:\s|>)", re.IGNORECASE)


@dataclass(frozen=True)
class BinaryAttachment:
    """Immutable source bytes paired with their validated, sniffed type."""

    data: bytes
    media_type: AttachmentMediaType

    @property
    def summary(self) -> AttachmentSummary:
        return summary_for(self.data, self.media_type)


def summary_for(data: bytes, media_type: AttachmentMediaType) -> AttachmentSummary:
    return AttachmentSummary(
        media_type=media_type,
        byte_count=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def make_attachment(data: bytes, limits: AttachmentLimits) -> BinaryAttachment:
    """Identify an attachment from bytes, never from a filename or claimed MIME."""
    if len(data) > limits.max_bytes:
        raise ValidationError(f"attachment is {len(data)} bytes; limit is {limits.max_bytes} bytes")
    if data.startswith(PNG_SIGNATURE):
        return BinaryAttachment(data=data, media_type="image/png")
    if data.startswith(JPEG_SIGNATURE):
        return BinaryAttachment(data=data, media_type="image/jpeg")
    if data.startswith(b"%PDF-"):
        return BinaryAttachment(data=data, media_type="application/pdf")
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile as exc:
            raise ValidationError("invalid ZIP attachment") from exc
        if "word/document.xml" in names:
            return BinaryAttachment(
                data=data,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        if {"content.xml", "meta.xml"}.issubset(names):
            return BinaryAttachment(data=data, media_type="application/vnd.oasis.opendocument.text")
    head = data[:4096].lstrip(b"\xef\xbb\xbf\x00\t\r\n ")
    if _SVG_ROOT.search(head) or b"<svg" in head.lower():
        return BinaryAttachment(data=data, media_type="image/svg+xml")
    lowered = head.lower()
    if b"<html" in lowered or b"<!doctype html" in lowered:
        return BinaryAttachment(data=data, media_type="text/html")
    if head.startswith(b"---\n") or head.startswith(b"---\r\n"):
        return BinaryAttachment(data=data, media_type="text/markdown")
    raise UnsupportedError(
        "unsupported attachment bytes; expected PNG, JPEG, SVG, PDF, DOCX, ODT, HTML, or Markdown"
    )


def is_ai_key(key: str) -> bool:
    normalized = key.strip().casefold().replace(" ", "_")
    return normalized in AI_METADATA_KEYS or "c2pa" in normalized or "aigc" in normalized


def ai_vendor_in(data: bytes) -> str | None:
    lowered = data.lower()
    if b"anthropic" in lowered or b"claude" in lowered:
        return "Anthropic/Claude"
    if b"openai" in lowered or b"chatgpt" in lowered or b"dall-e" in lowered:
        return "OpenAI"
    for marker in AI_VALUE_MARKERS:
        if marker in lowered:
            return marker.decode("ascii", "replace")
    return None


def has_ai_value(data: bytes) -> bool:
    lowered = data.lower()
    return any(marker in lowered for marker in AI_VALUE_MARKERS + PROVENANCE_MARKERS)


def has_soft_binding(data: bytes) -> bool:
    lowered = data.lower()
    return any(marker in lowered for marker in SOFT_BINDING_MARKERS)


def looks_like_c2pa(data: bytes) -> bool:
    """Require a structural marker, avoiding random ``c2pa`` pixel collisions."""
    lowered = data.lower()
    return C2PA_UUID in data or (b"jumb" in lowered and b"c2pa" in lowered)
