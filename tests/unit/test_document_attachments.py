"""Targeted cleanup of non-raster provenance containers."""

from __future__ import annotations

import io
import zipfile

from unmark.attachments import clean_attachment, inspect_attachment


def _zip(parts: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return output.getvalue()


def test_markdown_clean_removes_only_ai_frontmatter() -> None:
    source = (
        b"---\ntitle: Notes\ngenerator: Claude\nlicense: CC-BY\n---\n\nA human-readable body.\n"
    )
    outcome = clean_attachment(source)
    assert outcome.output_bytes is not None
    assert b"generator: Claude" not in outcome.output_bytes
    assert b"license: CC-BY" in outcome.output_bytes
    assert b"A human-readable body." in outcome.output_bytes
    assert inspect_attachment(outcome.output_bytes).state == "not_detected"


def test_html_clean_removes_privacy_and_ai_metadata_but_preserves_visible_text() -> None:
    source = b"""<!doctype html><html><head>
    <meta name="generator" content="WordPress 6">
    <meta name="ai-generated" content="Claude">
    <script type="application/ld+json">{"digitalSourceType":"trainedAlgorithmicMedia"}</script>
    </head><body>Hello <strong>world</strong>.</body></html>"""
    outcome = clean_attachment(source)
    assert outcome.output_bytes is not None
    assert b"WordPress 6" not in outcome.output_bytes
    assert b"ai-generated" not in outcome.output_bytes
    assert b"trainedAlgorithmicMedia" not in outcome.output_bytes
    assert b"Hello <strong>world</strong>." in outcome.output_bytes
    assert any(item.kind == "embedded_metadata" for item in outcome.report.evidence)


def test_html_author_metadata_is_removed_and_noop_is_byte_identical() -> None:
    source = (
        b"<html><head><meta name=\"author\" content=\"Ada Example\">"
        b"<meta name=\"creation-date\" content=\"2026-01-01\">"
        b"<meta name=\"viewport\" content=\"width=device-width\">"
        b"</head><body>Visible <b>content</b>.</body></html>"
    )
    outcome = clean_attachment(source)
    assert outcome.report.state == "removed_verified"
    assert outcome.output_bytes is not None
    assert b"Ada Example" not in outcome.output_bytes
    assert b"2026-01-01" not in outcome.output_bytes
    assert b"viewport" in outcome.output_bytes
    assert b"Visible <b>content</b>." in outcome.output_bytes
    noop = b"<html><head><meta name=\"viewport\" content=\"width=device-width\"></head></html>"
    no_op_outcome = clean_attachment(noop)
    assert no_op_outcome.output_bytes == noop
    assert no_op_outcome.report.actions == ()


def test_markdown_author_frontmatter_is_removed_and_noop_is_byte_identical() -> None:
    source = (
        b"---\ntitle: Notes\nauthor: Ada Example\ndate: 2026-01-01\n"
        b"license: CC-BY\n---\nBody.\n"
    )
    outcome = clean_attachment(source)
    assert outcome.output_bytes is not None
    assert b"Ada Example" not in outcome.output_bytes
    assert b"2026-01-01" not in outcome.output_bytes
    assert b"title: Notes" in outcome.output_bytes
    assert b"license: CC-BY" in outcome.output_bytes
    assert b"Body." in outcome.output_bytes
    assert any(item.kind == "embedded_metadata" for item in outcome.report.evidence)
    noop = b"---\ntitle: Notes\nlicense: CC-BY\n---\nBody.\n"
    no_op_outcome = clean_attachment(noop)
    assert no_op_outcome.output_bytes == noop
    assert no_op_outcome.report.actions == ()


def test_docx_clean_does_not_scan_or_change_visible_body() -> None:
    source = _zip(
        {
            "[Content_Types].xml": b"<Types/>",
            "word/document.xml": (
                b"<document><p>Claude is a character in this story.</p></document>"
            ),
            "docProps/core.xml": (
                b"<core><creator>Ada Example</creator><company>Example Co</company>"
                b"<created>2026-01-01</created><lastModifiedBy>ChatGPT</lastModifiedBy>"
                b"<title>Keep me</title></core>"
            ),
        }
    )
    outcome = clean_attachment(source)
    assert outcome.output_bytes is not None
    with zipfile.ZipFile(io.BytesIO(outcome.output_bytes)) as archive:
        assert archive.read("word/document.xml") == _zip_part(source, "word/document.xml")
        assert b"<title>Keep me</title>" in archive.read("docProps/core.xml")
        assert b"Ada Example" not in archive.read("docProps/core.xml")
        assert b"Example Co" not in archive.read("docProps/core.xml")
        assert b"2026-01-01" not in archive.read("docProps/core.xml")
        assert b"ChatGPT" not in archive.read("docProps/core.xml")
        assert {item.kind for item in outcome.report.evidence} == {
            "embedded_metadata",
            "unsigned_ai_metadata",
        }
        assert outcome.report.actions[0].description == (
            "Redacted AI provenance and ordinary identifying metadata from document "
            "metadata parts."
        )


def test_docx_noop_retains_zip_bytes_exactly() -> None:
    source = _zip(
        {
            "[Content_Types].xml": b"<Types/>",
            "docProps/": b"",
            "word/": b"",
            "word/document.xml": b"<document><p>Visible body.</p></document>",
            "docProps/core.xml": b"<core><title>Keep me</title></core>",
        }
    )
    outcome = clean_attachment(source)
    assert outcome.report.state == "not_detected"
    assert outcome.report.actions == ()
    assert outcome.output_bytes == source


def _zip_part(data: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return archive.read(name)


def test_odt_clean_redacts_meta_xml_only() -> None:
    source = _zip(
        {
            "mimetype": b"application/vnd.oasis.opendocument.text",
            "content.xml": b"<document><p>Claude is a valid name.</p></document>",
            "meta.xml": b"<meta><generator>OpenAI</generator><title>Keep</title></meta>",
        }
    )
    outcome = clean_attachment(source)
    assert outcome.output_bytes is not None
    with zipfile.ZipFile(io.BytesIO(outcome.output_bytes)) as archive:
        assert archive.read("content.xml") == _zip_part(source, "content.xml")
        assert b"<title>Keep</title>" in archive.read("meta.xml")
        assert b"OpenAI" not in archive.read("meta.xml")


def test_pdf_info_values_are_redacted_without_changing_offsets() -> None:
    source = (
        b"%PDF-1.4\n1 0 obj << /Creator (Claude) /Author (Ada Example) "
        b"/CreationDate (D:20260101) /ModDate (D:20260102) /Title (Keep) >> endobj\n%%EOF\n"
    )
    outcome = clean_attachment(source)
    assert outcome.output_bytes is not None
    assert len(outcome.output_bytes) == len(source)
    assert b"Claude" not in outcome.output_bytes
    assert b"Ada Example" not in outcome.output_bytes
    assert b"D:20260101" not in outcome.output_bytes
    assert b"D:20260102" not in outcome.output_bytes
    assert b"/Title (Keep)" in outcome.output_bytes
    assert any(item.kind == "embedded_metadata" for item in outcome.report.evidence)
