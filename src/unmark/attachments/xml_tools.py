"""Network-free, bounded XML helpers for SVG and XMP packets."""

from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import cast

from unmark.attachments.common import has_ai_value, has_soft_binding, looks_like_c2pa
from unmark.attachments.models import AttachmentLimits
from unmark.core.errors import ValidationError

_TARGET_NAMES = frozenset(
    {
        "aigc",
        "aisystemused",
        "aisystemversionused",
        "aipromptinformation",
        "aipromptwritername",
        "claimgenerator",
        "claim_generator",
        "creatortool",
        "digitalsourcetype",
        "generator",
        "provenance",
        "software",
    }
)
_METADATA_CONTAINERS = frozenset({"metadata", "rdf", "description", "xmpmeta"})
_PRIVACY_NAMES = frozenset(
    {
        "author",
        "creator",
        "company",
        "lastmodifiedby",
        "last_modified_by",
        "created",
        "creationdate",
        "date",
        "modified",
        "modifieddate",
        "lastprinted",
        "manager",
        "owner",
        "person",
        "cameraownername",
        "bodyserialnumber",
        "imageuniqueid",
    }
)
_METADATA_CONTAINERS = _METADATA_CONTAINERS | frozenset(
    {"core", "coreproperties", "properties", "documentproperties", "meta"}
)


@dataclass(frozen=True)
class XmlRedaction:
    target_count: int
    privacy_count: int
    soft_binding_count: int
    ambiguous_count: int


def decode_xml(data: bytes) -> str:
    """Decode the Unicode encodings used by browser-produced SVG/XMP."""
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            return data.decode("utf-16")
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError("SVG/XMP must use UTF-8 or BOM-marked UTF-16") from exc


def parse_xml(data: bytes, limits: AttachmentLimits) -> ET.Element:
    """Parse XML without DTDs/entities and enforce node/depth bounds."""
    text = decode_xml(data)
    lowered = text.casefold()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise ValidationError("XML DTDs and entity declarations are not allowed")
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
        root = ET.fromstring(text, parser=parser)
    except ET.ParseError as exc:
        raise ValidationError(f"malformed XML attachment metadata: {exc}") from exc

    count = 0
    stack: list[tuple[ET.Element, int]] = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        count += 1
        if count > limits.max_xml_nodes:
            raise ValidationError(f"XML exceeds the {limits.max_xml_nodes}-node limit")
        if depth > limits.max_xml_depth:
            raise ValidationError(f"XML exceeds the {limits.max_xml_depth}-level depth limit")
        stack.extend((child, depth + 1) for child in list(element))
    return root


def local_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    return name.rsplit("}", 1)[-1].split(":")[-1].casefold()


def _is_comment(element: ET.Element) -> bool:
    return cast("object", element.tag) is ET.Comment


def _bytes(value: str | None) -> bytes:
    return (value or "").encode("utf-8", "replace")


def _target_value(value: str | None) -> bool:
    raw = _bytes(value)
    return has_ai_value(raw) or looks_like_c2pa(raw)


def _soft_value(value: str | None) -> bool:
    return has_soft_binding(_bytes(value))


def inspect_xml_metadata(root: ET.Element) -> XmlRedaction:
    clone = copy.deepcopy(root)
    return redact_xml_metadata(clone)


def redact_xml_metadata(root: ET.Element) -> XmlRedaction:
    """Remove separable provenance and identifying fields while preserving metadata.

    Unstructured mixed text is reported as ambiguous and left in place. The
    caller must then abstain instead of deleting a whole licensing/accessibility
    block.
    """
    targets = 0
    privacy = 0
    soft_bindings = 0
    ambiguous = 0

    def walk(element: ET.Element, *, in_metadata: bool) -> None:
        nonlocal targets, privacy, soft_bindings, ambiguous
        name = local_name(element.tag)
        in_metadata = in_metadata or name in _METADATA_CONTAINERS

        for attribute, value in list(element.attrib.items()):
            attr_name = local_name(attribute)
            if _soft_value(value):
                soft_bindings += 1
            if in_metadata and attr_name in _PRIVACY_NAMES:
                del element.attrib[attribute]
                privacy += 1
            elif in_metadata and (
                attr_name in _TARGET_NAMES
                or "c2pa" in attr_name
                or "aigc" in attr_name
                or _target_value(value)
            ):
                del element.attrib[attribute]
                targets += 1

        for child in list(element):
            if _is_comment(child):
                if _soft_value(child.text):
                    soft_bindings += 1
                if _target_value(child.text):
                    element.remove(child)
                    targets += 1
                continue
            child_name = local_name(child.tag)
            child_namespace = str(child.tag).casefold()
            child_blob = ET.tostring(child, encoding="utf-8")
            if has_soft_binding(child_blob):
                soft_bindings += 1
            explicitly_targeted = (
                child_name in _TARGET_NAMES
                or "c2pa" in child_name
                or "aigc" in child_name
                or "c2pa" in child_namespace
                or "tc260" in child_namespace
            )
            if in_metadata and child_name in _PRIVACY_NAMES:
                element.remove(child)
                privacy += 1
                continue
            if in_metadata and (explicitly_targeted or _target_value(child.text)):
                element.remove(child)
                targets += 1
                continue
            walk(child, in_metadata=in_metadata)

        if in_metadata and element.text and _target_value(element.text):
            # Text attached directly to a generic metadata/RDF container cannot be
            # separated from a neighboring license or accessibility statement.
            ambiguous += 1

    walk(root, in_metadata=False)
    return XmlRedaction(
        target_count=targets,
        privacy_count=privacy,
        soft_binding_count=soft_bindings,
        ambiguous_count=ambiguous,
    )


def serialize_xml(root: ET.Element) -> bytes:
    return cast(
        "bytes",
        ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True),
    )


def svg_render_fingerprint(root: ET.Element) -> str:
    """Hash the render-bearing XML projection, excluding comments/metadata."""

    def project(element: ET.Element) -> object:
        children: list[object] = []
        for child in list(element):
            if _is_comment(child) or local_name(child.tag) == "metadata":
                continue
            children.append(project(child))
        attributes = sorted(
            (key, value)
            for key, value in element.attrib.items()
            if local_name(key) not in _TARGET_NAMES
            and "c2pa" not in local_name(key)
            and "aigc" not in local_name(key)
        )
        return (str(element.tag), attributes, element.text, element.tail, children)

    encoded = json.dumps(project(root), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
