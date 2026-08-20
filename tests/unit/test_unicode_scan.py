"""Script-aware Unicode inspection and sanitation.

These tests encode the central safety promise: conservative by default, and never
destroying characters that carry linguistic or visual meaning.
"""

from __future__ import annotations

import pytest

from unmark.core.operations import apply_operations
from unmark.core.policies import UnicodePolicy
from unmark.core.spans import Span
from unmark.inspect.unicode_scan import (
    blocked_by_protection,
    inspect_text,
    sanitation_operations,
)

SAFE = UnicodePolicy(name="safe")
REPORT = UnicodePolicy(name="report")
TYPOGRAPHIC = UnicodePolicy(name="typographic")
AGGRESSIVE = UnicodePolicy(name="aggressive")


def sanitize(text: str, policy: UnicodePolicy = SAFE, spans=()) -> str:
    findings = inspect_text(text, policy)
    ops = sanitation_operations(text, findings, policy, spans)
    return apply_operations(text, ops)


def kinds(text: str, policy: UnicodePolicy = SAFE) -> set[str]:
    return {f.kind for f in inspect_text(text, policy)}


class TestZeroWidth:
    def test_zero_width_space_removed(self):
        assert sanitize("hello​world") == "helloworld"

    def test_bom_removed(self):
        assert sanitize("﻿hello") == "hello"

    def test_word_joiner_removed(self):
        assert sanitize("a⁠b") == "ab"

    def test_finding_reports_exact_codepoint_and_offset(self):
        findings = inspect_text("ab​cd")
        assert len(findings) == 1
        finding = findings[0]
        assert finding.offset == 2
        assert finding.codepoint == 0x200B
        assert finding.label == "U+200B"
        assert finding.name == "ZERO WIDTH SPACE"
        assert finding.category == "Cf"
        assert finding.reason


class TestEmojiPreservation:
    def test_zwj_in_family_emoji_preserved(self):
        text = "\U0001f468‍\U0001f469‍\U0001f467"
        assert sanitize(text) == text

    def test_variation_selector_preserved(self):
        text = "heart ❤️ here"
        assert sanitize(text) == text

    def test_orphan_variation_selector_removed(self):
        assert sanitize("plain️ text") == "plain text"

    def test_han_variation_selector_preserved(self):
        text = "漢󠄀字"
        assert sanitize(text) == text

    def test_keycap_sequence_preserved(self):
        text = "1️⃣"
        assert sanitize(text) == text

    def test_flag_tag_sequence_preserved(self):
        # Scottish flag: base emoji + tag characters + cancel tag.
        text = "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f"
        assert sanitize(text) == text

    def test_zwj_findings_are_marked_preserved(self):
        findings = inspect_text("\U0001f468‍\U0001f469")
        zwj = [f for f in findings if f.codepoint == 0x200D]
        assert zwj and all(not f.removable for f in zwj)
        assert all(f.protected_by == "preserve_emoji_sequences" for f in zwj)


class TestLanguageControls:
    def test_persian_zwnj_preserved(self):
        text = "می‌روم"
        assert sanitize(text) == text

    def test_zwnj_finding_cites_the_script(self):
        findings = inspect_text("می‌ر")
        zwnj = [f for f in findings if f.codepoint == 0x200C]
        assert zwnj and not zwnj[0].removable
        assert zwnj[0].protected_by == "preserve_language_controls"

    def test_devanagari_zwnj_preserved(self):
        text = "क्‌ष"
        assert sanitize(text) == text

    def test_bidi_mark_preserved_next_to_arabic(self):
        text = "مرحبا ‏hello"
        assert sanitize(text) == text

    def test_bidi_mark_removed_with_no_rtl_text(self):
        # With no right-to-left text anywhere near it, the control cannot be
        # doing its job and is a likely carrier.
        assert sanitize("plain ‏text") == "plain text"

    def test_bidi_override_needs_research_mode(self):
        text = "file‮txt.exe"
        assert sanitize(text, SAFE) == text
        findings = inspect_text(text, SAFE)
        override = [f for f in findings if f.codepoint == 0x202E]
        assert override and override[0].severity == "suspicious"
        assert not override[0].removable

    def test_bidi_override_removed_only_when_aggressive(self):
        assert sanitize("file‮txt", AGGRESSIVE) == "filetxt"


class TestSpaces:
    def test_nbsp_preserved_under_safe(self):
        assert sanitize("Fig. 1") == "Fig. 1"

    def test_nbsp_preserved_under_typographic(self):
        # NBSP is deliberate typography, not a carrier.
        assert sanitize("Fig. 1", TYPOGRAPHIC) == "Fig. 1"

    def test_thin_space_normalized_under_safe(self):
        assert sanitize("a b", SAFE) == "a b"

    def test_three_per_em_and_ideographic_spaces_normalized_under_safe(self):
        assert sanitize("a b　c", SAFE) == "a b c"

    def test_thin_space_normalized_under_typographic(self):
        assert sanitize("a b", TYPOGRAPHIC) == "a b"

    def test_em_space_normalized_under_typographic(self):
        assert sanitize("a b", TYPOGRAPHIC) == "a b"

    def test_ideographic_space_normalized_under_typographic(self):
        assert sanitize("a　b", TYPOGRAPHIC) == "a b"

    def test_nfc_composes_unprotected_text(self):
        assert sanitize("Café", SAFE) == "Café"

    def test_nfc_does_not_change_a_locked_sequence(self):
        text = "Café"
        span = Span(start=3, end=5, kind="user_lock", value="é")
        assert sanitize(text, SAFE, [span]) == text

    def test_soft_hyphen_kept_under_safe_removed_under_typographic(self):
        assert sanitize("sun­shine", SAFE) == "sun­shine"
        assert sanitize("sun­shine", TYPOGRAPHIC) == "sunshine"


class TestTagCharactersAndControls:
    def test_bare_tag_characters_removed(self):
        assert sanitize("hi\U000e0001\U000e0022 there") == "hi there"

    def test_control_character_removed(self):
        assert sanitize("a\x07b") == "ab"

    def test_newlines_and_tabs_preserved(self):
        text = "line\n\tindented\r\n"
        assert sanitize(text) == text

    def test_private_use_reported_not_removed(self):
        text = "logo  here"
        assert sanitize(text) == text
        assert "private_use" in kinds(text)

    def test_replacement_character_reported_never_removed(self):
        text = "broken � text"
        assert sanitize(text) == text
        findings = [f for f in inspect_text(text) if f.kind == "replacement_character"]
        assert findings and not findings[0].removable


class TestConfusables:
    def test_cyrillic_in_latin_word_reported(self):
        findings = [f for f in inspect_text("The аpple") if f.kind == "confusable"]
        assert findings
        assert findings[0].codepoint == 0x0430

    def test_confusables_are_never_removed(self):
        text = "The аpple"
        assert sanitize(text) == text
        assert sanitize(text, AGGRESSIVE) == text

    def test_pure_cyrillic_word_not_flagged(self):
        assert not [f for f in inspect_text("мир") if f.kind == "confusable"]


class TestPolicies:
    def test_report_policy_never_mutates(self):
        text = "a​b­c\x07d"
        findings = inspect_text(text, REPORT)
        assert findings
        assert sanitation_operations(text, findings, REPORT) == ()
        assert sanitize(text, REPORT) == text

    def test_clean_text_yields_no_operations(self):
        text = "Perfectly ordinary text.\n"
        assert inspect_text(text) == ()
        assert sanitize(text) == text

    def test_policy_flags(self):
        assert not REPORT.mutates
        assert SAFE.mutates
        assert AGGRESSIVE.is_research_only
        assert not SAFE.is_research_only

    def test_disabling_emoji_preservation_allows_zwj_removal(self):
        policy = UnicodePolicy(name="safe", preserve_emoji_sequences=False)
        text = "\U0001f468‍\U0001f469"
        assert sanitize(text, policy) == "\U0001f468\U0001f469"


class TestProtectedSpans:
    def test_finding_inside_protected_span_is_skipped(self):
        text = "code ​here"
        span = Span(start=5, end=6, kind="code", value="​")
        assert sanitize(text, SAFE, [span]) == text

    def test_blocked_findings_are_reported(self):
        text = "code ​here"
        span = Span(start=5, end=6, kind="code", value="​")
        findings = inspect_text(text, SAFE)
        blocked = blocked_by_protection(findings, [span])
        assert len(blocked) == 1
        assert blocked[0].codepoint == 0x200B

    def test_finding_outside_protected_span_still_removed(self):
        text = "​code here"
        span = Span(start=5, end=9, kind="code", value="e he")
        assert sanitize(text, SAFE, [span]) == "code here"


class TestOperationLog:
    def test_every_change_has_a_reason_and_operator(self):
        text = "a​b\x07c"
        findings = inspect_text(text)
        ops = sanitation_operations(text, findings, SAFE)
        assert len(ops) == 2
        for operation in ops:
            assert operation.reason
            assert operation.operator.startswith("unicode:")
            assert operation.original is not None

    def test_operations_record_the_original_character(self):
        findings = inspect_text("a​b")
        ops = sanitation_operations("a​b", findings, SAFE)
        assert ops[0].original == "​"
        assert ops[0].text == ""


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain ascii",
        "Ünïcödé áccents",
        "日本語のテキスト",
        "العربية نص",
        "🎉 party 🎊",
        "mixed 日本 and English",
    ],
)
def test_clean_text_round_trips(text):
    assert sanitize(text) == text
