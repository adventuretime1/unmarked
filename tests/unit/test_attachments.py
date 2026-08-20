"""Attachment contracts, format safety, and verify-before-publish behavior."""

from __future__ import annotations

import binascii
import importlib
import json
import sys
import types
import zlib

import pytest

from unmark.attachments import clean_attachment, inspect_attachment
from unmark.attachments.common import make_attachment
from unmark.attachments.models import AttachmentLimits
from unmark.core.errors import UnsupportedError, ValidationError


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + kind + payload + crc.to_bytes(4, "big")


def make_png(*metadata: tuple[bytes, bytes]) -> bytes:
    ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    pixels = zlib.compress(b"\x00\x10\x20\x30")
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + b"".join(png_chunk(kind, payload) for kind, payload in metadata)
        + png_chunk(b"IDAT", pixels)
        + png_chunk(b"IEND", b"")
    )


def jpeg_segment(marker: int, payload: bytes) -> bytes:
    return b"\xff" + bytes([marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def make_jpeg(*segments: tuple[int, bytes]) -> bytes:
    sof = jpeg_segment(0xC0, b"\x08\x00\x01\x00\x01\x01\x01\x11\x00")
    sos = jpeg_segment(0xDA, b"\x01\x01\x00\x00\x3f\x00")
    entropy = b"\x11\xff\x00\x22\xff\xd0\x33"
    return (
        b"\xff\xd8"
        + b"".join(jpeg_segment(marker, payload) for marker, payload in segments)
        + sof
        + sos
        + entropy
        + b"\xff\xd9"
    )


def make_private_exif() -> bytes:
    artist = b"Ada\x00"
    timestamp = b"2026:01:01 12:00:00\x00"
    value_offset = 8 + 2 + 2 * 12 + 4
    return (
        b"II*\x00\x08\x00\x00\x00"
        + b"\x02\x00"
        + b"\x3b\x01\x02\x00\x04\x00\x00\x00"
        + artist
        + b"\x32\x01\x02\x00\x14\x00\x00\x00"
        + value_offset.to_bytes(4, "little")
        + b"\x00\x00\x00\x00"
        + timestamp
    )


def test_byte_sniffing_does_not_need_a_filename() -> None:
    attachment = make_attachment(make_png(), AttachmentLimits())
    assert attachment.media_type == "image/png"


def test_unsupported_bytes_are_rejected() -> None:
    with pytest.raises(UnsupportedError):
        inspect_attachment(b"not an image")


def test_png_targeted_text_removal_preserves_pixels_and_license() -> None:
    source = make_png(
        (b"tEXt", b"Software\x00Claude"),
        (b"tEXt", b"Copyright\x00Copyright 2026 Example"),
    )
    inspected = inspect_attachment(source)
    assert inspected.state == "unsigned_ai_metadata"
    assert inspected.evidence[0].vendor_attribution == "vendor_unverified"

    outcome = clean_attachment(source)
    assert outcome.report.state == "removed_verified"
    assert outcome.output_bytes is not None
    assert b"Claude" not in outcome.output_bytes
    assert b"Copyright 2026 Example" in outcome.output_bytes
    assert outcome.report.invariants[0].passed is True
    assert inspect_attachment(outcome.output_bytes).state == "not_detected"


def test_png_author_text_metadata_is_removed_but_license_is_retained() -> None:
    source = make_png(
        (b"tEXt", b"Author\x00Ada Example"),
        (b"tEXt", b"Copyright\x00Copyright 2026 Example"),
    )
    inspected = inspect_attachment(source)
    assert inspected.state == "embedded_metadata"
    outcome = clean_attachment(source)
    assert outcome.report.state == "removed_verified"
    assert outcome.output_bytes is not None
    assert b"Ada Example" not in outcome.output_bytes
    assert b"Copyright 2026 Example" in outcome.output_bytes
    assert outcome.report.actions[0].source == "png:tEXt[1]"


def test_png_c2pa_locator_never_claims_signature_validity_without_reader(monkeypatch) -> None:
    def unavailable(_name: str):
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", unavailable)
    source = make_png((b"caBX", b"\x00\x00jumb manifest c2pa Anthropic"))
    inspected = inspect_attachment(source)
    assert inspected.state == "unknown"
    assert inspected.evidence[0].confidence == "unknown"

    outcome = clean_attachment(source)
    assert outcome.report.state == "removed_verified"
    assert outcome.output_bytes is not None
    assert b"caBX" not in outcome.output_bytes


def test_png_mixed_legal_metadata_causes_fail_before_publish() -> None:
    source = make_png((b"tEXt", b"Copyright\x00Licensed work made with Claude"))
    inspected = inspect_attachment(source)
    assert inspected.evidence[0].removable is False
    outcome = clean_attachment(source)
    assert outcome.report.state == "failed"
    assert outcome.output_bytes is None
    assert "survived" in outcome.report.notes[0]


def test_png_invalid_crc_is_rejected() -> None:
    source = bytearray(make_png())
    source[-1] ^= 0x01
    with pytest.raises(ValidationError, match="CRC"):
        inspect_attachment(bytes(source))


def test_jpeg_comment_removal_preserves_entropy_scan() -> None:
    source = make_jpeg((0xFE, b"Generated by Claude"), (0xE2, b"unrelated-profile"))
    outcome = clean_attachment(source)
    assert outcome.report.state == "removed_verified"
    assert outcome.output_bytes is not None
    assert b"Generated by Claude" not in outcome.output_bytes
    assert b"unrelated-profile" in outcome.output_bytes
    assert outcome.report.invariants[0].name == "jpeg_entropy_scans_unchanged"
    assert outcome.report.invariants[0].passed is True


def test_jpeg_private_exif_is_blanked_without_recompressing_pixels() -> None:
    source = make_jpeg((0xE1, b"Exif\x00\x00" + make_private_exif()))
    outcome = clean_attachment(source)
    assert outcome.report.state == "removed_verified"
    assert outcome.output_bytes is not None
    assert len(outcome.output_bytes) == len(source)
    assert b"Ada" not in outcome.output_bytes
    assert b"2026:01:01" not in outcome.output_bytes
    assert outcome.report.invariants[0].name == "jpeg_entropy_scans_unchanged"
    assert outcome.report.invariants[0].passed is True
    assert outcome.report.evidence[0].kind == "embedded_metadata"


def test_jpeg_c2pa_app11_group_is_removed_but_other_app11_is_preserved() -> None:
    source = make_jpeg(
        (0xEB, b"JP\x00\x01\x00\x00\x00\x01\x00\x00\x00\x20jumb"),
        (0xEB, b"c2pa manifest payload"),
        (0xE2, b"ICC_PROFILE\x00keep"),
    )
    outcome = clean_attachment(source)
    assert outcome.output_bytes is not None
    assert b"c2pa manifest payload" not in outcome.output_bytes
    assert b"ICC_PROFILE\x00keep" in outcome.output_bytes


def test_svg_redaction_preserves_render_script_and_license() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg">
      <metadata><generator>Claude</generator><license>MIT</license></metadata>
      <script>window.example = true</script><rect width="10" height="10"/>
    </svg>"""
    outcome = clean_attachment(source)
    assert outcome.report.state == "removed_verified"
    assert outcome.output_bytes is not None
    assert b"Claude" not in outcome.output_bytes
    assert b"MIT" in outcome.output_bytes
    assert b"window.example = true" in outcome.output_bytes
    assert outcome.report.invariants[0].passed is True


def test_svg_noop_keeps_original_bytes_exactly() -> None:
    source = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">\n'
        b'  <metadata><license>MIT</license></metadata><rect width="10" height="10"/>\n'
        b"</svg>"
    )
    outcome = clean_attachment(source)
    assert outcome.report.state == "not_detected"
    assert outcome.report.actions == ()
    assert outcome.output_bytes == source


def test_svg_creator_metadata_is_removed_without_touching_license_or_rendering() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg">
      <metadata><creator>Ada Example</creator><license>MIT</license></metadata>
      <rect width="10" height="10"/>
    </svg>"""
    inspected = inspect_attachment(source)
    assert inspected.state == "embedded_metadata"
    assert inspected.evidence[0].kind == "embedded_metadata"
    outcome = clean_attachment(source)
    assert outcome.report.state == "removed_verified"
    assert outcome.output_bytes is not None
    assert b"Ada Example" not in outcome.output_bytes
    assert b"MIT" in outcome.output_bytes
    assert outcome.report.actions[0].description == (
        "Removed ordinary identifying SVG metadata fields."
    )


def test_svg_external_entities_are_rejected() -> None:
    source = b'<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg>&xxe;</svg>'
    with pytest.raises(ValidationError, match="DTD"):
        inspect_attachment(source)


def test_official_reader_validates_but_does_not_name_match_anthropic(monkeypatch) -> None:
    store = {
        "active_manifest": "claim",
        "manifests": {
            "claim": {
                "claim_generator": "Claude/fixture",
                "signature_info": {
                    "issuer": "Unmapped Anthropic Test Issuer",
                    "cert_serial_number": "123",
                },
            }
        },
    }

    class FakeContext:
        @classmethod
        def from_dict(cls, _settings):
            return cls()

        def close(self):
            return None

    class FakeReader:
        def __init__(self, _media_type, _stream, *, context=None):
            self.context = context

        def json(self):
            return json.dumps(store)

        def get_validation_state(self):
            return "Trusted"

        def get_validation_results(self):
            return {}

        def close(self):
            return None

    fake = types.ModuleType("c2pa")
    fake.__dict__["Context"] = FakeContext
    fake.__dict__["Reader"] = FakeReader
    monkeypatch.setitem(sys.modules, "c2pa", fake)

    inspected = inspect_attachment(make_png((b"caBX", b"jumb c2pa manifest")))
    assert inspected.state == "c2pa_verified"
    evidence = inspected.evidence[0]
    assert evidence.vendor == "Anthropic/Claude"
    assert evidence.vendor_attribution == "vendor_unverified"
