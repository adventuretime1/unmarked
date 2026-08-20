"""Targeted provenance metadata adapters for document and web containers.

The module intentionally operates only on well-defined metadata surfaces.  It
does not scan or alter visible document bodies merely because they mention an AI
vendor; that would be both a false positive and an unacceptable content edit.
"""

from __future__ import annotations

import copy
import io
import re
import zipfile

from unmark.attachments.common import ai_vendor_in, has_ai_value, has_soft_binding, looks_like_c2pa
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
    parse_xml,
    redact_xml_metadata,
    serialize_xml,
)
from unmark.core.errors import ValidationError

_AI_KEY = re.compile(
    r"(?:ai[-_ ]?generated|aigc|c2pa|content.?credential|provenance|digital.?source|"
    r"claude|anthropic|openai|chatgpt|gemini|synthid|copilot|generative.?ai)",
    re.IGNORECASE,
)
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_META_TAG = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_HTML_PRIVACY_META = re.compile(
    r"\b(?:name|property)\s*=\s*(['\"])(?:author|creator|generator|date|created|"
    r"creation[-_ ]?date|modified|modification[-_ ]?date|last[-_ ]?modified|"
    r"lastmodifiedby|owner)\1",
    re.IGNORECASE,
)
_JSON_LD = re.compile(
    r"<script\b[^>]*type\s*=\s*['\"]application/ld\+json['\"][^>]*>.*?</script>",
    re.IGNORECASE | re.DOTALL,
)
_DATA_AI = re.compile(r"\sdata-ai[\w-]*\s*=\s*(['\"]).*?\1", re.IGNORECASE)
_PDF_INFO = re.compile(
    rb"/(?P<key>Creator|Producer|Author|CreationDate|ModDate|ModificationDate|CreatorTool|"
    rb"AIGC|AIGenerated|DigitalSourceType)\s*\((?P<value>(?:[^\\()]|\\.)*)\)",
    re.IGNORECASE,
)
_FRONTMATTER_PRIVACY_KEYS = frozenset(
    {
        "author",
        "creator",
        "generator",
        "date",
        "created",
        "creation_date",
        "modified",
        "modification_date",
        "last_modified",
        "lastmodifiedby",
        "owner",
    }
)
_PDF_PRIVACY_KEYS = frozenset(
    {b"creator", b"producer", b"author", b"creationdate", b"moddate", b"modificationdate"}
)


def _evidence(
    source: str, description: str, key: str, value: bytes
) -> tuple[AttachmentEvidence, ...]:
    vendor = ai_vendor_in(value)
    items = [
        AttachmentEvidence(
            kind="unsigned_ai_metadata",
            source=source,
            description=description,
            confidence="unsigned",
            removable=True,
            metadata_key=key,
            vendor=vendor,
            vendor_attribution="vendor_unverified" if vendor else "not_applicable",
        )
    ]
    if has_soft_binding(value):
        items.append(
            AttachmentEvidence(
                kind="soft_binding_declared",
                source=source,
                description="Metadata declares a durable or external binding.",
                confidence="declaration",
                removable=True,
                metadata_key=key,
            )
        )
    return tuple(items)


def _action(source: str, description: str, before: int, after: int) -> AttachmentAction:
    return AttachmentAction(
        kind="rewrote_metadata",
        source=source,
        description=description,
        bytes_before=before,
        bytes_after=after,
    )


def _invariant(name: str, passed: bool, description: str) -> FidelityInvariant:
    return FidelityInvariant(name=name, passed=passed, description=description)


def _decode_text(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{label} must be UTF-8") from exc


def _markdown_decisions(text: str) -> list[str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return []
    return [line for line in match.group(1).splitlines() if _AI_KEY.search(line)]


def _metadata_key(line: str) -> str:
    return line.split(":", 1)[0].strip().casefold().replace("-", "_").replace(" ", "_")


def _markdown_privacy_decisions(text: str) -> list[str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return []
    return [
        line
        for line in match.group(1).splitlines()
        if _metadata_key(line) in _FRONTMATTER_PRIVACY_KEYS
    ]


def _inspect_markdown(data: bytes) -> FormatInspection:
    text = _decode_text(data, "Markdown")
    lines = _markdown_decisions(text)
    evidence: list[AttachmentEvidence] = []
    for line in lines:
        evidence.extend(
            _evidence(
                "markdown:frontmatter",
                "AI provenance frontmatter field.",
                line.split(":", 1)[0],
                line.encode(),
            )
        )
    for line in _markdown_privacy_decisions(text):
        evidence.append(
            AttachmentEvidence(
                kind="embedded_metadata",
                source="markdown:frontmatter",
                description="Ordinary identifying Markdown frontmatter field.",
                confidence="container",
                removable=True,
                metadata_key=_metadata_key(line),
            )
        )
    return FormatInspection(tuple(evidence), False, None)


def _clean_markdown(data: bytes) -> FormatClean:
    text = _decode_text(data, "Markdown")
    match = _FRONTMATTER.match(text)
    if not match:
        return FormatClean(
            data, (), (_invariant("markdown_body_unchanged", True, "No frontmatter was present."),)
        )
    ai_lines = _markdown_decisions(text)
    privacy_lines = _markdown_privacy_decisions(text)
    removable = {*ai_lines, *privacy_lines}
    if not removable:
        return FormatClean(
            data,
            (),
            (_invariant("markdown_body_unchanged", True, "No removable metadata was present."),),
        )
    kept = [line for line in match.group(1).splitlines() if line not in removable]
    block = "\n".join(kept).strip("\n")
    output = (f"---\n{block}\n---\n" if block else "") + text[match.end() :]
    encoded = output.encode("utf-8")
    actions = (
        ()
        if encoded == data
        else (
            _action(
                "markdown:frontmatter",
                (
                    "Removed AI provenance and ordinary identifying frontmatter fields."
                    if ai_lines and privacy_lines
                    else (
                        "Removed AI provenance frontmatter fields."
                        if ai_lines
                        else "Removed ordinary identifying frontmatter fields."
                    )
                ),
                len(data),
                len(encoded),
            ),
        )
    )
    return FormatClean(
        encoded,
        actions,
        (_invariant("markdown_body_unchanged", True, "Only YAML frontmatter was changed."),),
    )


def _html_matches(text: str) -> list[str]:
    matches = [tag for tag in _META_TAG.findall(text) if _AI_KEY.search(tag)]
    matches.extend(
        item.group(0) for item in _JSON_LD.finditer(text) if _AI_KEY.search(item.group(0))
    )
    matches.extend(item.group(0) for item in _DATA_AI.finditer(text))
    return matches


def _html_privacy_matches(text: str) -> list[str]:
    return [tag for tag in _META_TAG.findall(text) if _HTML_PRIVACY_META.search(tag)]


def _inspect_html(data: bytes) -> FormatInspection:
    text = _decode_text(data, "HTML")
    evidence: list[AttachmentEvidence] = []
    for value in _html_matches(text):
        evidence.extend(
            _evidence(
                "html:head", "AI provenance HTML metadata.", "meta/json-ld/data-ai", value.encode()
            )
        )
    for _value in _html_privacy_matches(text):
        evidence.append(
            AttachmentEvidence(
                kind="embedded_metadata",
                source="html:head",
                description="Ordinary identifying HTML meta field.",
                confidence="container",
                removable=True,
                metadata_key="meta",
            )
        )
    return FormatInspection(tuple(evidence), False, None)


def _clean_html(data: bytes) -> FormatClean:
    text = _decode_text(data, "HTML")
    ai_matches = _html_matches(text)
    privacy_matches = _html_privacy_matches(text)
    if not ai_matches and not privacy_matches:
        return FormatClean(
            data,
            (),
            (
                _invariant(
                    "html_visible_text_unchanged",
                    True,
                    "No removable metadata was present; original bytes were retained.",
                ),
            ),
        )
    output = _META_TAG.sub(
        lambda item: (
            ""
            if _AI_KEY.search(item.group(0)) or _HTML_PRIVACY_META.search(item.group(0))
            else item.group(0)
        ),
        text,
    )
    output = _JSON_LD.sub(
        lambda item: "" if _AI_KEY.search(item.group(0)) else item.group(0), output
    )
    output = _DATA_AI.sub("", output)
    encoded = output.encode("utf-8")
    actions = (
        ()
        if encoded == data
        else (
            _action(
                "html:metadata",
                (
                    "Removed AI provenance and ordinary identifying HTML metadata fields."
                    if ai_matches and privacy_matches
                    else (
                        "Removed AI provenance HTML metadata fields."
                        if ai_matches
                        else "Removed ordinary identifying HTML metadata fields."
                    )
                ),
                len(data),
                len(encoded),
            ),
        )
    )
    return FormatClean(
        encoded,
        actions,
        (
            _invariant(
                "html_visible_text_unchanged",
                True,
                "Only metadata tags and attributes were changed.",
            ),
        ),
    )


_DOCX_METADATA_PREFIXES = ("docProps/", "customXml/")


def _zip_parts(data: bytes, limits: AttachmentLimits) -> tuple[zipfile.ZipInfo, ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = tuple(archive.infolist())
    except zipfile.BadZipFile as exc:
        raise ValidationError("invalid Office ZIP attachment") from exc
    if len(infos) > limits.max_chunks:
        raise ValidationError(f"Office ZIP exceeds the {limits.max_chunks}-part limit")
    if sum(info.file_size for info in infos) > limits.max_bytes:
        raise ValidationError("Office ZIP decompressed size exceeds the attachment limit")
    return infos


def _office_metadata_evidence(
    source: str, raw: bytes, limits: AttachmentLimits
) -> tuple[AttachmentEvidence, ...]:
    result = inspect_xml_metadata(parse_xml(raw, limits))
    evidence: list[AttachmentEvidence] = []
    if result.target_count or result.ambiguous_count or _has_ai_metadata(raw):
        evidence.extend(
            _evidence(
                source,
                "AI provenance in a document metadata part.",
                source.rsplit(":", 1)[-1],
                raw,
            )
        )
    if result.privacy_count:
        evidence.append(
            AttachmentEvidence(
                kind="embedded_metadata",
                source=source,
                description=(
                    "Ordinary identifying document metadata (such as author, company, "
                    "or timestamps) is removable."
                ),
                confidence="container",
                removable=True,
                metadata_key=source.rsplit(":", 1)[-1],
            )
        )
    return tuple(evidence)


def _has_ai_metadata(raw: bytes) -> bool:
    raw_text = raw.decode("utf-8", "ignore")
    return has_ai_value(raw) or looks_like_c2pa(raw) or bool(_AI_KEY.search(raw_text))


def _inspect_office(
    data: bytes, media_type: AttachmentMediaType, limits: AttachmentLimits
) -> FormatInspection:
    infos = _zip_parts(data, limits)
    prefix = (
        _DOCX_METADATA_PREFIXES
        if media_type.endswith("wordprocessingml.document")
        else ("meta.xml",)
    )
    evidence: list[AttachmentEvidence] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in infos:
            if info.is_dir():
                continue
            if not any(info.filename.startswith(item) for item in prefix):
                continue
            raw = archive.read(info.filename)
            evidence.extend(_office_metadata_evidence(f"office:{info.filename}", raw, limits))
    return FormatInspection(tuple(evidence), False, None)


def _clean_office(
    data: bytes, media_type: AttachmentMediaType, limits: AttachmentLimits
) -> FormatClean:
    infos = _zip_parts(data, limits)
    prefix = (
        _DOCX_METADATA_PREFIXES
        if media_type.endswith("wordprocessingml.document")
        else ("meta.xml",)
    )
    replacements: dict[str, bytes] = {}
    changed = 0
    ai_fields = 0
    privacy_fields = 0
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        for info in infos:
            if info.is_dir():
                continue
            if not any(info.filename.startswith(item) for item in prefix):
                continue
            raw = source.read(info.filename)
            root = parse_xml(raw, limits)
            redaction = redact_xml_metadata(root)
            count = redaction.target_count + redaction.privacy_count
            if count:
                replacements[info.filename] = serialize_xml(root)
                changed += count
                ai_fields += int(_has_ai_metadata(raw))
                privacy_fields += redaction.privacy_count
    if not replacements:
        name = (
            "docx_visible_parts_unchanged"
            if media_type.endswith("wordprocessingml.document")
            else "odt_visible_parts_unchanged"
        )
        return FormatClean(
            data,
            (),
            (
                _invariant(
                    name,
                    True,
                    "No removable metadata was present; original bytes were retained.",
                ),
            ),
        )

    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data)) as source,
        zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as destination,
    ):
        for info in infos:
            raw = source.read(info.filename)
            raw = replacements.get(info.filename, raw)
            destination.writestr(copy.copy(info), raw)
    encoded = output.getvalue()
    actions = (
        ()
        if not changed
        else (
            _action(
                "office:metadata",
                (
                    "Redacted "
                    + ("AI provenance" if ai_fields else "")
                    + (" and " if ai_fields and privacy_fields else "")
                    + ("ordinary identifying metadata" if privacy_fields else "")
                    + " from document metadata parts."
                ),
                len(data),
                len(encoded),
            ),
        )
    )
    name = (
        "docx_visible_parts_unchanged"
        if media_type.endswith("wordprocessingml.document")
        else "odt_visible_parts_unchanged"
    )
    return FormatClean(
        encoded,
        actions,
        (_invariant(name, True, "Only metadata parts were eligible for changes."),),
    )


def _pdf_fields(data: bytes) -> list[re.Match[bytes]]:
    return [item for item in _PDF_INFO.finditer(data) if item.group("value").strip()]


def _inspect_pdf(data: bytes) -> FormatInspection:
    evidence: list[AttachmentEvidence] = []
    for field in _pdf_fields(data):
        value = field.group(0)
        key = field.group("key").decode("latin-1")
        if _AI_KEY.search(value.decode("latin-1")):
            evidence.extend(
                _evidence("pdf:info", "AI provenance PDF Info dictionary field.", key, value)
            )
        if field.group("key").lower() in _PDF_PRIVACY_KEYS:
            evidence.append(
                AttachmentEvidence(
                    kind="embedded_metadata",
                    source="pdf:info",
                    description="Ordinary identifying PDF Info dictionary field.",
                    confidence="container",
                    removable=True,
                    metadata_key=key,
                )
            )
    if looks_like_c2pa(data):
        evidence.append(
            AttachmentEvidence(
                kind="unknown",
                source="pdf:embedded-manifest",
                description=(
                    "Potential PDF C2PA/JUMBF data; safe removal requires a dedicated PDF/C2PA "
                    "tool."
                ),
                confidence="unknown",
                removable=False,
            )
        )
    return FormatInspection(
        tuple(evidence),
        looks_like_c2pa(data),
        "pdf:embedded-manifest" if looks_like_c2pa(data) else None,
    )


def _clean_pdf(data: bytes) -> FormatClean:
    def replace(match: re.Match[bytes]) -> bytes:
        value = match.group(0)
        return re.sub(
            rb"\((?:[^\\()]|\\.)*\)",
            lambda item: b"(" + b" " * (len(item.group(0)) - 2) + b")",
            value,
        )

    fields = _pdf_fields(data)
    ai_fields = [item for item in fields if _AI_KEY.search(item.group(0).decode("latin-1"))]
    privacy_fields = [item for item in fields if item.group("key").lower() in _PDF_PRIVACY_KEYS]
    removable = {item.span() for item in ai_fields + privacy_fields}
    output = _PDF_INFO.sub(
        lambda item: replace(item) if item.span() in removable else item.group(0), data
    )
    actions = (
        ()
        if output == data
        else (
            _action(
                "pdf:info",
                (
                    (
                        "Blanked AI provenance and ordinary identifying PDF Info values"
                        if ai_fields and privacy_fields
                        else (
                            "Blanked AI provenance PDF Info values"
                            if ai_fields
                            else "Blanked ordinary identifying PDF Info values"
                        )
                    )
                    + " without changing byte offsets."
                ),
                len(data),
                len(output),
            ),
        )
    )
    return FormatClean(
        output,
        actions,
        (
            _invariant(
                "pdf_byte_offsets_unchanged",
                len(data) == len(output),
                "Only equal-length PDF Info values were blanked.",
            ),
        ),
    )


def inspect_document(
    data: bytes, media_type: AttachmentMediaType, limits: AttachmentLimits
) -> FormatInspection:
    if media_type == "text/markdown":
        return _inspect_markdown(data)
    if media_type == "text/html":
        return _inspect_html(data)
    if media_type == "application/pdf":
        return _inspect_pdf(data)
    return _inspect_office(data, media_type, limits)


def clean_document(
    data: bytes, media_type: AttachmentMediaType, limits: AttachmentLimits
) -> FormatClean:
    if media_type == "text/markdown":
        return _clean_markdown(data)
    if media_type == "text/html":
        return _clean_html(data)
    if media_type == "application/pdf":
        return _clean_pdf(data)
    return _clean_office(data, media_type, limits)
