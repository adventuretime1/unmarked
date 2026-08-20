"""Bounded TIFF/EXIF text inspection and same-size targeted redaction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from unmark.attachments.common import ai_vendor_in, has_ai_value

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 13: 4}
_IFD_POINTER_TAGS = frozenset({0x014A, 0x8769, 0x8825, 0xA005})
_TEXT_UNDEFINED_TAGS = frozenset({0x9286})  # UserComment
_PRIVATE_TEXT_TAGS = frozenset(
    {
        0x0132,  # DateTime
        0x013B,  # Artist
        0x013C,  # HostComputer
        0x9003,  # DateTimeOriginal
        0x9004,  # DateTimeDigitized
        0xA420,  # ImageUniqueID
        0xA430,  # CameraOwnerName
        0xA431,  # BodySerialNumber
        0xA435,  # LensSerialNumber
    }
)
_MAX_IFDS = 32
_MAX_ENTRIES = 4096


@dataclass(frozen=True)
class ExifFinding:
    tag: int
    vendor: str | None
    offset: int
    length: int


def _find_exif(
    tiff: bytes,
    include: Callable[[int, bool, bool, bytes], bool],
) -> tuple[ExifFinding, ...]:
    """Find bounded, in-place-redactable EXIF values without decoding pixels."""
    if len(tiff) < 8 or tiff[:2] not in {b"II", b"MM"}:
        return ()
    byteorder: Literal["little", "big"] = "little" if tiff[:2] == b"II" else "big"
    if _number(tiff, 2, 2, byteorder) != 42:
        return ()
    first = _number(tiff, 4, 4, byteorder)
    if first is None:
        return ()

    findings: list[ExifFinding] = []
    pending = [(first, False)]
    visited: set[int] = set()
    while pending and len(visited) < _MAX_IFDS:
        ifd, is_gps_ifd = pending.pop()
        if ifd in visited or ifd + 2 > len(tiff):
            continue
        visited.add(ifd)
        count = _number(tiff, ifd, 2, byteorder)
        if count is None or count > _MAX_ENTRIES:
            continue
        for index in range(count):
            entry = ifd + 2 + index * 12
            if entry + 12 > len(tiff):
                break
            tag = _number(tiff, entry, 2, byteorder)
            field_type = _number(tiff, entry + 2, 2, byteorder)
            item_count = _number(tiff, entry + 4, 4, byteorder)
            if tag is None or field_type is None or item_count is None:
                continue
            unit = _TYPE_SIZES.get(field_type)
            if unit is None or item_count > len(tiff):
                continue
            size = unit * item_count
            value_offset = entry + 8
            if size > 4:
                pointed = _number(tiff, entry + 8, 4, byteorder)
                if pointed is None:
                    continue
                value_offset = pointed
            if value_offset + size > len(tiff):
                continue
            if tag in _IFD_POINTER_TAGS and field_type in {4, 13}:
                for pointer_index in range(min(item_count, _MAX_IFDS)):
                    pointer = _number(tiff, value_offset + pointer_index * 4, 4, byteorder)
                    if pointer is not None:
                        pending.append((pointer, tag == 0x8825))
            textual = field_type == 2 or (field_type == 7 and tag in _TEXT_UNDEFINED_TAGS)
            value = tiff[value_offset : value_offset + size]
            if include(tag, is_gps_ifd, textual, value):
                findings.append(
                    ExifFinding(
                        tag=tag,
                        vendor=ai_vendor_in(value),
                        offset=value_offset,
                        length=size,
                    )
                )
        next_offset_at = ifd + 2 + count * 12
        following = _number(tiff, next_offset_at, 4, byteorder)
        if following:
            pending.append((following, is_gps_ifd))
    return tuple(findings)


def _number(
    data: bytes | bytearray,
    offset: int,
    length: int,
    byteorder: Literal["little", "big"],
) -> int | None:
    end = offset + length
    if offset < 0 or end > len(data):
        return None
    return int.from_bytes(data[offset:end], byteorder)


def find_ai_exif(tiff: bytes) -> tuple[ExifFinding, ...]:
    """Find AI-bearing ASCII/UserComment fields without decoding image pixels."""
    return _find_exif(tiff, lambda _tag, _gps, textual, value: textual and has_ai_value(value))


def find_private_exif(tiff: bytes) -> tuple[ExifFinding, ...]:
    """Find personal/date fields and GPS IFD values safe to blank in place."""
    return _find_exif(
        tiff,
        lambda tag, is_gps_ifd, textual, value: bool(value.strip(b"\x00 \t\r\n"))
        and (is_gps_ifd or (textual and tag in _PRIVATE_TEXT_TAGS)),
    )


def redact_exif_findings(tiff: bytes, findings: tuple[ExifFinding, ...]) -> bytes:
    if not findings:
        return tiff
    output = bytearray(tiff)
    for finding in findings:
        output[finding.offset : finding.offset + finding.length] = b"\x00" * finding.length
    return bytes(output)


def redact_ai_exif(tiff: bytes) -> tuple[bytes, tuple[ExifFinding, ...]]:
    findings = find_ai_exif(tiff)
    return redact_exif_findings(tiff, findings), findings
