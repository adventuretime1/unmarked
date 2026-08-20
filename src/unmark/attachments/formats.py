"""Bounded SVG/PNG/JPEG adapters with targeted, render-preserving mutations."""

from __future__ import annotations

import binascii
import copy
import hashlib
import re
import struct
import zlib
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from unmark.attachments.common import (
    AI_VALUE_MARKERS,
    PNG_SIGNATURE,
    PROVENANCE_MARKERS,
    ai_vendor_in,
    has_ai_value,
    has_soft_binding,
    is_ai_key,
    looks_like_c2pa,
)
from unmark.attachments.documents import clean_document, inspect_document
from unmark.attachments.exif import (
    find_ai_exif,
    find_private_exif,
    redact_exif_findings,
)
from unmark.attachments.models import (
    AttachmentAction,
    AttachmentEvidence,
    AttachmentLimits,
    AttachmentMediaType,
    FidelityInvariant,
    FormatClean,
    FormatInspection,
)
from unmark.attachments.xml_tools import (
    inspect_xml_metadata,
    local_name,
    parse_xml,
    redact_xml_metadata,
    serialize_xml,
    svg_render_fingerprint,
)
from unmark.core.errors import ValidationError


@dataclass(frozen=True)
class _PngChunk:
    index: int
    kind: bytes
    payload: bytes
    raw: bytes


@dataclass(frozen=True)
class _PngText:
    key: str
    value: bytes
    prefix: bytes
    compressed: bool
    chunk_kind: bytes


@dataclass(frozen=True)
class _MetadataDecision:
    targeted: bool = False
    embedded_metadata: bool = False
    removable: bool = False
    description: str = ""
    key: str | None = None
    replacement: bytes | None = None
    remove_container: bool = False
    vendor: str | None = None
    soft_binding: bool = False


@dataclass(frozen=True)
class _JpegToken:
    index: int
    marker: int | None
    raw: bytes
    payload: bytes = b""
    entropy: bool = False
    trailer: bool = False


def inspect_format(
    data: bytes, media_type: AttachmentMediaType, limits: AttachmentLimits
) -> FormatInspection:
    if media_type == "image/png":
        return _inspect_png(data, limits)
    if media_type == "image/jpeg":
        return _inspect_jpeg(data, limits)
    if media_type == "image/svg+xml":
        return _inspect_svg(data, limits)
    return inspect_document(data, media_type, limits)


def clean_format(
    data: bytes, media_type: AttachmentMediaType, limits: AttachmentLimits
) -> FormatClean:
    if media_type == "image/png":
        return _clean_png(data, limits)
    if media_type == "image/jpeg":
        return _clean_jpeg(data, limits)
    if media_type == "image/svg+xml":
        return _clean_svg(data, limits)
    return clean_document(data, media_type, limits)


def _evidence(source: str, decision: _MetadataDecision) -> tuple[AttachmentEvidence, ...]:
    items: list[AttachmentEvidence] = []
    if decision.targeted:
        items.append(
            AttachmentEvidence(
                kind="unsigned_ai_metadata",
                source=source,
                description=decision.description,
                confidence="unsigned",
                removable=decision.removable,
                vendor=decision.vendor,
                vendor_attribution=(
                    "vendor_unverified" if decision.vendor is not None else "not_applicable"
                ),
                metadata_key=decision.key,
            )
        )
    if decision.embedded_metadata:
        items.append(
            AttachmentEvidence(
                kind="embedded_metadata",
                source=source,
                description=decision.description,
                confidence="container",
                removable=decision.removable,
                metadata_key=decision.key,
            )
        )
    if decision.soft_binding:
        items.append(
            AttachmentEvidence(
                kind="soft_binding_declared",
                source=source,
                description="Embedded metadata declares a durable or external binding.",
                confidence="declaration",
                removable=decision.removable,
                vendor=decision.vendor,
                vendor_attribution="not_applicable",
                metadata_key=decision.key,
            )
        )
    return tuple(items)


# PNG -----------------------------------------------------------------------


def _parse_png(data: bytes, limits: AttachmentLimits) -> tuple[_PngChunk, ...]:
    if not data.startswith(PNG_SIGNATURE):
        raise ValidationError("invalid PNG signature")
    chunks: list[_PngChunk] = []
    position = len(PNG_SIGNATURE)
    saw_iend = False
    while position < len(data):
        if len(chunks) >= limits.max_chunks:
            raise ValidationError(f"PNG exceeds the {limits.max_chunks}-chunk limit")
        if position + 12 > len(data):
            raise ValidationError("truncated PNG chunk header")
        length = int.from_bytes(data[position : position + 4], "big")
        end = position + 12 + length
        if end > len(data):
            raise ValidationError("PNG chunk length exceeds remaining bytes")
        kind = data[position + 4 : position + 8]
        if len(kind) != 4 or not all(65 <= value <= 90 or 97 <= value <= 122 for value in kind):
            raise ValidationError("invalid PNG chunk type")
        payload = data[position + 8 : position + 8 + length]
        expected_crc = int.from_bytes(data[position + 8 + length : end], "big")
        actual_crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            name = kind.decode("ascii", "replace")
            raise ValidationError(f"PNG chunk {name} has an invalid CRC")
        chunks.append(
            _PngChunk(
                index=len(chunks),
                kind=kind,
                payload=payload,
                raw=data[position:end],
            )
        )
        position = end
        if kind == b"IEND":
            saw_iend = True
            break
    if not chunks or chunks[0].kind != b"IHDR" or len(chunks[0].payload) != 13:
        raise ValidationError("PNG must begin with a 13-byte IHDR chunk")
    if not saw_iend or chunks[-1].kind != b"IEND" or position != len(data):
        raise ValidationError("PNG must end exactly at IEND")
    width, height = struct.unpack(">II", chunks[0].payload[:8])
    if width == 0 or height == 0 or width * height > limits.max_pixels:
        raise ValidationError(f"PNG dimensions exceed the {limits.max_pixels}-pixel limit")
    if not any(chunk.kind == b"IDAT" for chunk in chunks):
        raise ValidationError("PNG has no IDAT image data")
    return tuple(chunks)


def _bounded_inflate(data: bytes, limit: int) -> bytes:
    inflater = zlib.decompressobj()
    try:
        result = inflater.decompress(data, limit + 1)
        if len(result) <= limit:
            result += inflater.flush(limit + 1 - len(result))
    except zlib.error as exc:
        raise ValidationError("invalid compressed PNG metadata") from exc
    if len(result) > limit or inflater.unconsumed_tail or not inflater.eof:
        raise ValidationError(f"decompressed PNG metadata exceeds the {limit}-byte limit")
    return result


def _split_null(data: bytes, start: int = 0) -> tuple[bytes, int]:
    end = data.find(b"\x00", start)
    if end < 0:
        raise ValidationError("malformed PNG text metadata")
    return data[start:end], end + 1


def _png_text(chunk: _PngChunk, limits: AttachmentLimits) -> _PngText:
    if len(chunk.payload) > limits.max_metadata_bytes:
        raise ValidationError("PNG metadata chunk exceeds the configured limit")
    key_bytes, cursor = _split_null(chunk.payload)
    try:
        key = key_bytes.decode("latin-1")
    except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 is total
        raise ValidationError("invalid PNG text keyword") from exc
    if not key or len(key_bytes) > 79:
        raise ValidationError("PNG text keyword must contain 1-79 bytes")
    if chunk.kind == b"tEXt":
        return _PngText(key, chunk.payload[cursor:], chunk.payload[:cursor], False, chunk.kind)
    if chunk.kind == b"zTXt":
        if cursor >= len(chunk.payload) or chunk.payload[cursor] != 0:
            raise ValidationError("unsupported PNG zTXt compression method")
        value = _bounded_inflate(
            chunk.payload[cursor + 1 :], limits.max_decompressed_metadata_bytes
        )
        return _PngText(key, value, chunk.payload[: cursor + 1], True, chunk.kind)
    if cursor + 2 > len(chunk.payload):
        raise ValidationError("truncated PNG iTXt metadata")
    compressed = chunk.payload[cursor] == 1
    if chunk.payload[cursor] not in {0, 1} or chunk.payload[cursor + 1] != 0:
        raise ValidationError("unsupported PNG iTXt compression flags")
    _, after_language = _split_null(chunk.payload, cursor + 2)
    _, after_translated = _split_null(chunk.payload, after_language)
    encoded = chunk.payload[after_translated:]
    value = (
        _bounded_inflate(encoded, limits.max_decompressed_metadata_bytes) if compressed else encoded
    )
    if len(value) > limits.max_decompressed_metadata_bytes:
        raise ValidationError("PNG iTXt metadata exceeds the configured limit")
    return _PngText(key, value, chunk.payload[:after_translated], compressed, chunk.kind)


def _rebuild_png_text(record: _PngText, value: bytes) -> bytes:
    encoded = zlib.compress(value) if record.compressed else value
    return record.prefix + encoded


def _xmp_decision(value: bytes, limits: AttachmentLimits, key: str) -> _MetadataDecision:
    root = parse_xml(value, limits)
    scan = inspect_xml_metadata(root)
    if (
        scan.target_count == 0
        and scan.privacy_count == 0
        and scan.soft_binding_count == 0
        and scan.ambiguous_count == 0
    ):
        return _MetadataDecision()
    cleaned = copy.deepcopy(root)
    changed = redact_xml_metadata(cleaned)
    removable = changed.ambiguous_count == 0
    return _MetadataDecision(
        targeted=changed.target_count > 0 or changed.ambiguous_count > 0,
        embedded_metadata=changed.privacy_count > 0,
        removable=removable,
        description=(
            (
                "AI/C2PA and ordinary identifying XMP metadata fields are separable."
                if changed.target_count and changed.privacy_count
                else (
                    "Targeted AI/C2PA XMP fields are separable from neighboring metadata."
                    if changed.target_count
                    else "Ordinary identifying XMP metadata fields are separable."
                )
            )
            if removable
            else "AI/C2PA text shares an inseparable XML metadata value; preserving the block."
        ),
        key=key,
        replacement=serialize_xml(cleaned) if removable else None,
        vendor=ai_vendor_in(value),
        soft_binding=changed.soft_binding_count > 0,
    )


def _png_decision(chunk: _PngChunk, limits: AttachmentLimits) -> _MetadataDecision:
    if chunk.kind == b"caBX":
        return _MetadataDecision(
            targeted=True,
            removable=True,
            description="PNG caBX C2PA manifest container.",
            key="caBX",
            remove_container=True,
            vendor=ai_vendor_in(chunk.payload),
            soft_binding=has_soft_binding(chunk.payload),
        )
    if chunk.kind == b"eXIf":
        ai_findings = find_ai_exif(chunk.payload)
        privacy_findings = find_private_exif(chunk.payload)
        findings = tuple(
            {(item.offset, item.length): item for item in ai_findings + privacy_findings}.values()
        )
        if not findings:
            return _MetadataDecision()
        replacement = redact_exif_findings(chunk.payload, findings)
        vendors = sorted({item.vendor for item in ai_findings if item.vendor})
        return _MetadataDecision(
            targeted=bool(ai_findings),
            embedded_metadata=bool(privacy_findings),
            removable=True,
            description=(
                "AI-related and ordinary identifying EXIF fields can be redacted in place."
                if ai_findings and privacy_findings
                else (
                    "AI-related EXIF text fields can be redacted in place."
                    if ai_findings
                    else "Ordinary identifying EXIF and GPS fields can be redacted in place."
                )
            ),
            key="eXIf",
            replacement=replacement,
            vendor=", ".join(vendors) if vendors else None,
        )
    if chunk.kind not in {b"tEXt", b"zTXt", b"iTXt"}:
        return _MetadataDecision()
    record = _png_text(chunk, limits)
    key_lower = record.key.casefold()
    if "xmp" in key_lower or record.value.lstrip().startswith((b"<x:", b"<rdf:", b"<?xml")):
        xmp = _xmp_decision(record.value, limits, record.key)
        if xmp.replacement is not None:
            return _MetadataDecision(
                **{
                    **xmp.__dict__,
                    "replacement": _rebuild_png_text(record, xmp.replacement),
                }
            )
        return xmp
    soft = has_soft_binding(record.value)
    if is_ai_key(record.key):
        return _MetadataDecision(
            targeted=True,
            removable=True,
            description="Dedicated AI-generation metadata field.",
            key=record.key,
            remove_container=True,
            vendor=ai_vendor_in(record.value),
            soft_binding=soft,
        )
    if has_ai_value(record.value):
        protected = {"author", "copyright", "license", "rights", "title", "description"}
        removable = key_lower not in protected
        return _MetadataDecision(
            targeted=True,
            removable=removable,
            description=(
                "AI generator marker in a separable metadata field."
                if removable
                else "AI marker shares a legal/descriptive metadata field; preserving it."
            ),
            key=record.key,
            remove_container=removable,
            vendor=ai_vendor_in(record.value),
            soft_binding=soft,
        )
    if key_lower in {
        "author",
        "artist",
        "creator",
        "creation time",
        "creation_time",
        "date",
        "last-modified",
        "last_modified",
        "modified",
        "owner",
    }:
        return _MetadataDecision(
            embedded_metadata=True,
            removable=True,
            description="Ordinary identifying PNG text metadata field.",
            key=record.key,
            remove_container=True,
        )
    if soft:
        return _MetadataDecision(
            removable=False,
            description="External binding declaration in otherwise unrelated metadata.",
            key=record.key,
            soft_binding=True,
        )
    return _MetadataDecision()


def _serialize_png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + kind + payload + crc.to_bytes(4, "big")


def _png_render_digest(chunks: tuple[_PngChunk, ...]) -> str:
    # Every non-metadata chunk is retained byte-for-byte. This covers IHDR,
    # palette/transparency, color, animation, and compressed pixel data.
    projection: list[bytes] = []
    for chunk in chunks:
        if chunk.kind in {b"caBX", b"tEXt", b"zTXt", b"iTXt"}:
            continue
        if chunk.kind == b"eXIf":
            findings = find_ai_exif(chunk.payload) + find_private_exif(chunk.payload)
            normalized = redact_exif_findings(chunk.payload, findings)
            projection.append(_serialize_png_chunk(chunk.kind, normalized))
        else:
            projection.append(chunk.raw)
    rendered = b"".join(projection)
    return hashlib.sha256(rendered).hexdigest()


def _inspect_png(data: bytes, limits: AttachmentLimits) -> FormatInspection:
    chunks = _parse_png(data, limits)
    evidence: list[AttachmentEvidence] = []
    c2pa = False
    c2pa_source: str | None = None
    for chunk in chunks:
        decision = _png_decision(chunk, limits)
        source = f"png:{chunk.kind.decode('ascii')}[{chunk.index}]"
        if chunk.kind == b"caBX":
            c2pa = True
            c2pa_source = source
            # C2PA validity comes only from the official verifier.
            decision = _MetadataDecision(
                soft_binding=decision.soft_binding,
                removable=True,
                key="caBX",
            )
        evidence.extend(_evidence(source, decision))
    return FormatInspection(tuple(evidence), c2pa, c2pa_source)


def _clean_png(data: bytes, limits: AttachmentLimits) -> FormatClean:
    chunks = _parse_png(data, limits)
    before_digest = _png_render_digest(chunks)
    output = bytearray(PNG_SIGNATURE)
    actions: list[AttachmentAction] = []
    for chunk in chunks:
        decision = _png_decision(chunk, limits)
        source = f"png:{chunk.kind.decode('ascii')}[{chunk.index}]"
        if decision.targeted and not decision.removable:
            output += chunk.raw
            continue
        if decision.remove_container:
            actions.append(
                AttachmentAction(
                    kind="removed_container",
                    source=source,
                    description=decision.description,
                    bytes_before=len(chunk.raw),
                    bytes_after=0,
                )
            )
            continue
        if decision.replacement is not None and decision.replacement != chunk.payload:
            replacement = _serialize_png_chunk(chunk.kind, decision.replacement)
            output += replacement
            actions.append(
                AttachmentAction(
                    kind="redacted_metadata" if chunk.kind == b"eXIf" else "rewrote_metadata",
                    source=source,
                    description=decision.description,
                    bytes_before=len(chunk.raw),
                    bytes_after=len(replacement),
                )
            )
            continue
        output += chunk.raw
    cleaned = bytes(output)
    after_chunks = _parse_png(cleaned, limits)
    invariant = FidelityInvariant(
        name="png_render_chunks_unchanged",
        passed=before_digest == _png_render_digest(after_chunks),
        description=(
            "All render-bearing, color, animation, and compressed pixel chunks are byte-identical."
        ),
    )
    return FormatClean(cleaned, tuple(actions), (invariant,))


# JPEG ----------------------------------------------------------------------


def _parse_jpeg(data: bytes, limits: AttachmentLimits) -> tuple[_JpegToken, ...]:
    if not data.startswith(b"\xff\xd8"):
        raise ValidationError("invalid JPEG signature")
    tokens: list[_JpegToken] = [_JpegToken(0, 0xD8, data[:2])]
    position = 2
    saw_sos = False
    saw_eoi = False
    while position < len(data):
        if len(tokens) >= limits.max_chunks:
            raise ValidationError(f"JPEG exceeds the {limits.max_chunks}-segment limit")
        if data[position] != 0xFF:
            raise ValidationError("invalid JPEG marker boundary")
        marker_start = position
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            raise ValidationError("truncated JPEG marker")
        marker = data[position]
        position += 1
        if marker == 0x00:
            raise ValidationError("stuffed JPEG byte outside an entropy-coded scan")
        prefix = data[marker_start:position]
        if marker == 0xD9:
            tokens.append(_JpegToken(len(tokens), marker, prefix))
            saw_eoi = True
            if position < len(data):
                tokens.append(_JpegToken(len(tokens), None, data[position:], trailer=True))
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            tokens.append(_JpegToken(len(tokens), marker, prefix))
            continue
        if position + 2 > len(data):
            raise ValidationError("truncated JPEG segment length")
        length = int.from_bytes(data[position : position + 2], "big")
        if length < 2 or position + length > len(data):
            raise ValidationError("JPEG segment length exceeds remaining bytes")
        raw = data[marker_start : position + length]
        payload = data[position + 2 : position + length]
        tokens.append(_JpegToken(len(tokens), marker, raw, payload))
        position += length
        if marker != 0xDA:
            continue
        saw_sos = True
        scan_start = position
        cursor = position
        while cursor < len(data):
            next_ff = data.find(b"\xff", cursor)
            if next_ff < 0 or next_ff + 1 >= len(data):
                raise ValidationError("JPEG entropy-coded scan has no terminating marker")
            after = next_ff + 1
            while after < len(data) and data[after] == 0xFF:
                after += 1
            if after >= len(data):
                raise ValidationError("truncated JPEG entropy-coded scan")
            code = data[after]
            if code == 0x00 or 0xD0 <= code <= 0xD7:
                cursor = after + 1
                continue
            break
        if next_ff > scan_start:
            tokens.append(_JpegToken(len(tokens), None, data[scan_start:next_ff], entropy=True))
        position = next_ff
    if not saw_sos or not saw_eoi:
        raise ValidationError("JPEG must contain SOS image data and EOI")
    for token in tokens:
        if (
            token.marker
            in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }
            and len(token.payload) >= 5
        ):
            height = int.from_bytes(token.payload[1:3], "big")
            width = int.from_bytes(token.payload[3:5], "big")
            if width == 0 or height == 0 or width * height > limits.max_pixels:
                raise ValidationError(f"JPEG dimensions exceed the {limits.max_pixels}-pixel limit")
    return tuple(tokens)


def _jpeg_c2pa_indices(tokens: tuple[_JpegToken, ...]) -> set[int]:
    result: set[int] = set()
    position = 0
    while position < len(tokens):
        if tokens[position].marker != 0xEB:
            position += 1
            continue
        end = position
        while end < len(tokens) and tokens[end].marker == 0xEB:
            end += 1
        group = tokens[position:end]
        combined = b"".join(token.payload for token in group)
        if looks_like_c2pa(combined) or (
            b"c2pa" in combined.lower()
            and any(b"jumb" in token.payload[:256].lower() for token in group)
        ):
            result.update(token.index for token in group)
        position = end
    return result


def _jpeg_xmp(token: _JpegToken, limits: AttachmentLimits) -> tuple[bytes, bytes] | None:
    prefixes = (
        b"http://ns.adobe.com/xap/1.0/\x00",
        b"http://ns.adobe.com/xmp/extension/\x00",
    )
    for prefix in prefixes:
        if token.payload.startswith(prefix):
            body = token.payload[len(prefix) :]
            if len(body) > limits.max_metadata_bytes:
                raise ValidationError("JPEG XMP segment exceeds the configured limit")
            return prefix, body
    return None


def _replace_jpeg_payload(token: _JpegToken, payload: bytes) -> bytes:
    if token.marker is None:
        raise AssertionError("cannot replace payload on a non-marker token")
    length = len(payload) + 2
    if length > 0xFFFF:
        raise ValidationError("rewritten JPEG metadata exceeds one marker segment")
    marker_prefix_length = len(token.raw) - len(token.payload) - 2
    prefix = token.raw[:marker_prefix_length]
    return prefix + length.to_bytes(2, "big") + payload


def _redact_known_tokens(data: bytes) -> bytes:
    output = data
    for token in AI_VALUE_MARKERS + PROVENANCE_MARKERS:
        output = re.sub(
            re.escape(token), lambda match: b"\x00" * len(match.group()), output, flags=re.I
        )
    return output


def _jpeg_decision(
    token: _JpegToken, limits: AttachmentLimits, c2pa_indices: set[int]
) -> _MetadataDecision:
    if token.index in c2pa_indices:
        return _MetadataDecision(
            targeted=True,
            removable=True,
            description="JPEG APP11 C2PA/JUMBF manifest segment.",
            key="APP11",
            remove_container=True,
            vendor=ai_vendor_in(token.payload),
            soft_binding=has_soft_binding(token.payload),
        )
    if token.marker == 0xE1 and token.payload.startswith(b"Exif\x00\x00"):
        tiff = token.payload[6:]
        ai_findings = find_ai_exif(tiff)
        privacy_findings = find_private_exif(tiff)
        findings = tuple(
            {(item.offset, item.length): item for item in ai_findings + privacy_findings}.values()
        )
        if not findings:
            return _MetadataDecision()
        replacement = redact_exif_findings(tiff, findings)
        vendors = sorted({item.vendor for item in ai_findings if item.vendor})
        return _MetadataDecision(
            targeted=bool(ai_findings),
            embedded_metadata=bool(privacy_findings),
            removable=True,
            description=(
                "AI-related and ordinary identifying EXIF fields can be redacted "
                "without changing scan data."
                if ai_findings and privacy_findings
                else (
                    "AI-related EXIF text fields can be redacted without changing scan data."
                    if ai_findings
                    else (
                        "Ordinary identifying EXIF and GPS fields can be redacted "
                        "without changing scan data."
                    )
                )
            ),
            key="EXIF",
            replacement=b"Exif\x00\x00" + replacement,
            vendor=", ".join(vendors) if vendors else None,
        )
    xmp = _jpeg_xmp(token, limits) if token.marker == 0xE1 else None
    if xmp is not None:
        prefix, body = xmp
        if b"xmp/extension" in prefix:
            if has_ai_value(body) or looks_like_c2pa(body) or has_soft_binding(body):
                return _MetadataDecision(
                    targeted=True,
                    removable=False,
                    description=(
                        "Fragmented extended XMP contains provenance that cannot be "
                        "separated safely."
                    ),
                    key="extended XMP",
                    vendor=ai_vendor_in(body),
                    soft_binding=has_soft_binding(body),
                )
            return _MetadataDecision()
        decision = _xmp_decision(body, limits, "XMP")
        if decision.replacement is not None:
            return _MetadataDecision(
                **{**decision.__dict__, "replacement": prefix + decision.replacement}
            )
        return decision
    if token.marker == 0xED and (has_ai_value(token.payload) or has_soft_binding(token.payload)):
        replacement = _redact_known_tokens(token.payload)
        removable = replacement != token.payload and not has_ai_value(replacement)
        return _MetadataDecision(
            targeted=has_ai_value(token.payload),
            removable=removable,
            description=(
                "AI IPTC fields can be blanked in place while retaining neighboring records."
                if removable
                else "IPTC provenance marker cannot be isolated safely."
            ),
            key="APP13/IPTC",
            replacement=replacement if removable else None,
            vendor=ai_vendor_in(token.payload),
            soft_binding=has_soft_binding(token.payload),
        )
    if token.marker == 0xFE and has_ai_value(token.payload):
        return _MetadataDecision(
            targeted=True,
            removable=True,
            description="AI generator marker in a standalone JPEG comment.",
            key="COM",
            remove_container=True,
            vendor=ai_vendor_in(token.payload),
            soft_binding=has_soft_binding(token.payload),
        )
    if token.trailer and has_ai_value(token.raw):
        return _MetadataDecision(
            targeted=True,
            removable=False,
            description="Post-EOI trailer contains mixed AI metadata and is preserved by default.",
            key="trailer",
            vendor=ai_vendor_in(token.raw),
            soft_binding=has_soft_binding(token.raw),
        )
    return _MetadataDecision()


def _jpeg_scan_digest(tokens: tuple[_JpegToken, ...]) -> str:
    # Includes SOS headers, entropy bytes, and restart markers for every scan.
    content = bytearray()
    for token in tokens:
        if (
            token.marker == 0xDA
            or token.entropy
            or (token.marker is not None and 0xD0 <= token.marker <= 0xD7)
        ):
            content += token.raw
    return hashlib.sha256(content).hexdigest()


def _inspect_jpeg(data: bytes, limits: AttachmentLimits) -> FormatInspection:
    tokens = _parse_jpeg(data, limits)
    c2pa_indices = _jpeg_c2pa_indices(tokens)
    evidence: list[AttachmentEvidence] = []
    for token in tokens:
        decision = _jpeg_decision(token, limits, c2pa_indices)
        if token.index in c2pa_indices:
            decision = _MetadataDecision(
                soft_binding=decision.soft_binding,
                removable=True,
                key="APP11",
            )
        label = (
            f"APP{token.marker - 0xE0}"
            if token.marker is not None and 0xE0 <= token.marker <= 0xEF
            else ("COM" if token.marker == 0xFE else "trailer")
        )
        evidence.extend(_evidence(f"jpeg:{label}[{token.index}]", decision))
    source = f"jpeg:APP11[{min(c2pa_indices)}]" if c2pa_indices else None
    return FormatInspection(tuple(evidence), bool(c2pa_indices), source)


def _clean_jpeg(data: bytes, limits: AttachmentLimits) -> FormatClean:
    tokens = _parse_jpeg(data, limits)
    before_scan = _jpeg_scan_digest(tokens)
    c2pa_indices = _jpeg_c2pa_indices(tokens)
    output = bytearray()
    actions: list[AttachmentAction] = []
    for token in tokens:
        decision = _jpeg_decision(token, limits, c2pa_indices)
        label = (
            f"APP{token.marker - 0xE0}"
            if token.marker is not None and 0xE0 <= token.marker <= 0xEF
            else ("COM" if token.marker == 0xFE else "trailer")
        )
        source = f"jpeg:{label}[{token.index}]"
        if decision.targeted and not decision.removable:
            output += token.raw
            continue
        if decision.remove_container:
            actions.append(
                AttachmentAction(
                    kind="removed_container",
                    source=source,
                    description=decision.description,
                    bytes_before=len(token.raw),
                    bytes_after=0,
                )
            )
            continue
        if decision.replacement is not None and decision.replacement != token.payload:
            replacement = _replace_jpeg_payload(token, decision.replacement)
            output += replacement
            actions.append(
                AttachmentAction(
                    kind="redacted_metadata" if decision.key != "XMP" else "rewrote_metadata",
                    source=source,
                    description=decision.description,
                    bytes_before=len(token.raw),
                    bytes_after=len(replacement),
                )
            )
            continue
        output += token.raw
    cleaned = bytes(output)
    after_tokens = _parse_jpeg(cleaned, limits)
    invariant = FidelityInvariant(
        name="jpeg_entropy_scans_unchanged",
        passed=before_scan == _jpeg_scan_digest(after_tokens),
        description=(
            "SOS headers and entropy-coded scan bytes are byte-identical; no JPEG "
            "recompression occurred."
        ),
    )
    return FormatClean(cleaned, tuple(actions), (invariant,))


# SVG -----------------------------------------------------------------------


def _parse_svg(data: bytes, limits: AttachmentLimits) -> ET.Element:
    root = parse_xml(data, limits)
    if local_name(root.tag) != "svg":
        raise ValidationError("XML root element is not SVG")
    return root


def _svg_c2pa_hint(data: bytes) -> bool:
    lowered = data.lower()
    return looks_like_c2pa(data) or (
        b"c2pa" in lowered and (b"manifest" in lowered or b"contentcredentials" in lowered)
    )


def _inspect_svg(data: bytes, limits: AttachmentLimits) -> FormatInspection:
    root = _parse_svg(data, limits)
    result = inspect_xml_metadata(root)
    evidence: list[AttachmentEvidence] = []
    if result.target_count or result.ambiguous_count:
        evidence.append(
            AttachmentEvidence(
                kind="unsigned_ai_metadata",
                source="svg:metadata",
                description=(
                    "Targeted SVG provenance fields are separable from render and "
                    "licensing metadata."
                    if result.ambiguous_count == 0
                    else (
                        "SVG provenance shares an inseparable metadata value; the cleaner "
                        "will abstain."
                    )
                ),
                confidence="unsigned",
                removable=result.ambiguous_count == 0,
                vendor=ai_vendor_in(data),
                vendor_attribution=(
                    "vendor_unverified" if ai_vendor_in(data) else "not_applicable"
                ),
                metadata_key="metadata",
            )
        )
    if result.privacy_count:
        evidence.append(
            AttachmentEvidence(
                kind="embedded_metadata",
                source="svg:metadata",
                description="Ordinary identifying SVG metadata fields are removable.",
                confidence="container",
                removable=True,
                metadata_key="metadata",
            )
        )
    if result.soft_binding_count:
        evidence.append(
            AttachmentEvidence(
                kind="soft_binding_declared",
                source="svg:metadata",
                description="SVG metadata declares a durable or external binding.",
                confidence="declaration",
                removable=result.ambiguous_count == 0,
            )
        )
    c2pa = _svg_c2pa_hint(data)
    return FormatInspection(tuple(evidence), c2pa, "svg:metadata" if c2pa else None)


def _clean_svg(data: bytes, limits: AttachmentLimits) -> FormatClean:
    root = _parse_svg(data, limits)
    before = svg_render_fingerprint(root)
    result = redact_xml_metadata(root)
    if result.target_count == 0 and result.privacy_count == 0:
        return FormatClean(
            data,
            (),
            (
                FidelityInvariant(
                    name="svg_render_projection_unchanged",
                    passed=True,
                    description=(
                        "No removable SVG metadata was present; original bytes were retained."
                    ),
                ),
            ),
        )
    cleaned = serialize_xml(root)
    reparsed = _parse_svg(cleaned, limits)
    after = svg_render_fingerprint(reparsed)
    actions: tuple[AttachmentAction, ...] = ()
    if result.target_count or result.privacy_count:
        actions = (
            AttachmentAction(
                kind="rewrote_metadata",
                source="svg:metadata",
                description=(
                    "Removed separable AI/C2PA and ordinary identifying SVG metadata fields."
                    if result.target_count and result.privacy_count
                    else (
                        "Removed separable AI/C2PA SVG metadata fields."
                        if result.target_count
                        else "Removed ordinary identifying SVG metadata fields."
                    )
                ),
                bytes_before=len(data),
                bytes_after=len(cleaned),
            ),
        )
    invariant = FidelityInvariant(
        name="svg_render_projection_unchanged",
        passed=before == after,
        description=(
            "The SVG element/attribute/text projection outside metadata and comments is unchanged."
        ),
    )
    return FormatClean(cleaned, actions, (invariant,))
