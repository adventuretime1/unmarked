"""Markdown/plain structure parsing and protected-span discovery."""

from __future__ import annotations

from itertools import pairwise

import pytest

from unmark.core.document import MediaType
from unmark.core.errors import UsageError
from unmark.core.policies import FidelityPolicy
from unmark.inspect.protected import compile_locks, discover_spans, resolve_overlaps
from unmark.inspect.structure import parse_blocks


def blocks_of(text: str, media_type: MediaType = "text/markdown"):
    return [(b.kind, text[b.start : b.end]) for b in parse_blocks(text, media_type)]


def spans_of(text: str, media_type: MediaType = "text/markdown", **kwargs):
    blocks = parse_blocks(text, media_type)
    spans = discover_spans(text, blocks, media_type=media_type, **kwargs)
    return [(s.kind, s.value) for s in spans]


class TestMarkdownStructure:
    def test_heading(self):
        assert blocks_of("# Title\n") == [("heading", "# Title")]

    def test_heading_levels(self):
        parsed = parse_blocks("# One\n\n### Three\n", "text/markdown")
        assert [b.level for b in parsed] == [1, 3]

    def test_paragraph(self):
        assert blocks_of("Some text here.\n") == [("paragraph", "Some text here.")]

    def test_multi_line_paragraph_is_one_block(self):
        assert blocks_of("Line one\nline two.\n") == [("paragraph", "Line one\nline two.")]

    def test_blank_line_separates_paragraphs(self):
        assert blocks_of("First.\n\nSecond.\n") == [
            ("paragraph", "First."),
            ("paragraph", "Second."),
        ]

    def test_list_items(self):
        assert blocks_of("- one\n- two\n") == [("list_item", "- one"), ("list_item", "- two")]

    def test_ordered_list_items(self):
        assert blocks_of("1. one\n2. two\n") == [
            ("list_item", "1. one"),
            ("list_item", "2. two"),
        ]

    def test_block_quote_is_one_block(self):
        assert blocks_of("> a\n> b\n") == [("quote", "> a\n> b")]

    def test_fenced_code(self):
        assert blocks_of("```py\nx=1\n```\n") == [("code", "```py\nx=1\n```")]

    def test_tilde_fenced_code(self):
        assert blocks_of("~~~\nx=1\n~~~\n") == [("code", "~~~\nx=1\n~~~")]

    def test_unterminated_fence_stays_code(self):
        # Safer to treat the remainder as protected code than as prose.
        assert blocks_of("```\nx=1\n") == [("code", "```\nx=1\n")]

    def test_headings_inside_code_are_not_headings(self):
        parsed = blocks_of("```\n# not a heading\n```\n")
        assert parsed == [("code", "```\n# not a heading\n```")]

    def test_table(self):
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        assert blocks_of(text) == [("table", "| A | B |\n| --- | --- |\n| 1 | 2 |")]

    def test_offsets_are_exact(self):
        text = "# Title\n\nBody text.\n"
        for block in parse_blocks(text, "text/markdown"):
            assert text[block.start : block.end]
            assert block.end <= len(text)

    def test_plain_text_paragraphs(self):
        assert blocks_of("One.\n\nTwo.\n", "text/plain") == [
            ("paragraph", "One."),
            ("paragraph", "Two."),
        ]

    def test_markdown_syntax_ignored_in_plain_text(self):
        assert blocks_of("# Not a heading\n", "text/plain") == [("paragraph", "# Not a heading")]

    def test_empty_document(self):
        assert parse_blocks("", "text/markdown") == ()


class TestProtectedSpans:
    def test_url(self):
        assert ("url", "https://example.com/a?b=1") in spans_of("See https://example.com/a?b=1 now")

    def test_url_trailing_punctuation_excluded(self):
        assert ("url", "https://example.com") in spans_of("Visit https://example.com.")

    def test_email(self):
        assert ("url", "a.b@example.com") in spans_of("Mail a.b@example.com today")

    def test_markdown_link_target(self):
        assert ("url", "https://example.com") in spans_of("[text](https://example.com)")

    def test_inline_code(self):
        assert ("code", "`f()`") in spans_of("Call `f()` now")

    def test_fenced_code_block(self):
        assert ("code", "```\nx=1\n```") in spans_of("```\nx=1\n```\n")

    def test_date_iso(self):
        assert ("date", "2026-08-13") in spans_of("On 2026-08-13 we shipped")

    def test_date_written(self):
        assert ("date", "August 13, 2026") in spans_of("On August 13, 2026 we shipped")

    def test_number(self):
        assert ("number", "42") in spans_of("The answer is 42 today")

    def test_unit_percentage(self):
        assert ("unit", "42%") in spans_of("Grew 42% last year")

    def test_unit_currency(self):
        assert ("unit", "$1.5 million") in spans_of("Raised $1.5 million total")

    def test_unit_beats_bare_number(self):
        kinds = [kind for kind, _ in spans_of("Grew 42% now")]
        assert "unit" in kinds

    def test_citation_author_year(self):
        assert ("citation", "(Smith, 2024)") in spans_of("As shown (Smith, 2024) here")

    def test_citation_bracketed(self):
        assert ("citation", "[12]") in spans_of("As shown [12] here")

    def test_typographic_quote(self):
        assert ("quote", "“exact words”") in spans_of("He said “exact words” loudly")

    def test_balanced_straight_quote(self):
        assert ("quote", '"exact words"') in spans_of('He said "exact words" loudly')

    def test_unbalanced_straight_quote_not_locked(self):
        # A lone quote is more likely an inch mark than a quotation.
        assert not any(k == "quote" for k, _ in spans_of('A 6" pipe here'))

    def test_identifier_snake_case(self):
        assert ("identifier", "run_test") in spans_of("Call run_test soon")

    def test_identifier_semver(self):
        assert ("identifier", "1.2.3") in spans_of("Version 1.2.3 shipped")

    def test_spans_never_overlap(self):
        text = "See https://example.com/2026-08-13 and `code_here()` with 42% growth."
        blocks = parse_blocks(text, "text/markdown")
        spans = discover_spans(text, blocks, media_type="text/markdown")
        for first, second in pairwise(spans):
            assert first.end <= second.start

    def test_span_values_match_source(self):
        text = "Visit https://example.com on 2026-08-13 for 42% off."
        blocks = parse_blocks(text, "text/markdown")
        for span in discover_spans(text, blocks, media_type="text/markdown"):
            assert text[span.start : span.end] == span.value

    def test_policy_can_disable_a_lock(self):
        policy = FidelityPolicy(lock_numbers=False)
        kinds = [k for k, _ in spans_of("The answer is 42", policy=policy)]
        assert "number" not in kinds


class TestUserLocks:
    def test_user_lock_matches(self):
        assert ("user_lock", "ACME-123") in spans_of("Ticket ACME-123 filed", locks=(r"ACME-\d+",))

    def test_user_lock_wins_over_other_kinds(self):
        # A user lock outranks the number heuristic that would otherwise claim it.
        spans = spans_of("Value 42 here", locks=(r"42",))
        assert ("user_lock", "42") in spans
        assert ("number", "42") not in spans

    def test_multiple_locks(self):
        spans = spans_of("A-1 and B-2", locks=(r"A-\d", r"B-\d"))
        assert ("user_lock", "A-1") in spans
        assert ("user_lock", "B-2") in spans

    def test_invalid_regex_is_a_usage_error(self):
        with pytest.raises(UsageError, match="invalid --lock regex"):
            compile_locks(["("])

    def test_zero_width_lock_match_ignored(self):
        # A pattern that can only match empty must not create zero-length spans.
        assert not any(k == "user_lock" for k, _ in spans_of("text", locks=(r"q*",)))


class TestOverlapResolution:
    def test_higher_priority_wins(self):
        from unmark.core.spans import Span

        url = Span(start=0, end=20, kind="url", value="x" * 20)
        number = Span(start=5, end=7, kind="number", value="xx")
        assert [s.kind for s in resolve_overlaps([number, url])] == ["url"]

    def test_longer_wins_at_equal_priority(self):
        from unmark.core.spans import Span

        short = Span(start=0, end=3, kind="number", value="123")
        long = Span(start=0, end=6, kind="number", value="123456")
        assert [s.value for s in resolve_overlaps([short, long])] == ["123456"]

    def test_result_is_sorted_by_offset(self):
        from unmark.core.spans import Span

        a = Span(start=10, end=12, kind="number", value="ab")
        b = Span(start=0, end=2, kind="number", value="cd")
        assert [s.start for s in resolve_overlaps([a, b])] == [0, 10]
