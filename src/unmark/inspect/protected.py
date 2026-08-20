"""Protected-span discovery.

These are heuristics, and the report says so. They are tuned to be
*under*-inclusive on ambiguous constructs and over-inclusive on unambiguous ones
(URLs, code, digits): a missed lock is a fidelity risk, but a spurious lock only
costs edit freedom.

Every span records its discovering heuristic in ``Span.detector`` so a reviewer can
tell why a region was locked. Overlapping spans are resolved by a fixed priority
order, keeping the higher-priority and then longer span.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from unmark.core.errors import UsageError
from unmark.core.policies import FidelityPolicy
from unmark.core.spans import Block, Span, SpanKind

# Priority: lower number wins when two spans overlap.
_PRIORITY: dict[SpanKind, int] = {
    "user_lock": 0,
    "code": 1,
    "url": 2,
    "citation": 3,
    "quote": 4,
    "date": 5,
    "identifier": 6,
    # Units outrank bare numbers: "42%" must lock as a quantity-with-unit rather
    # than losing its unit to a shorter "42" number span.
    "unit": 7,
    "number": 8,
    "formula": 9,
    "entity": 10,
}

_URL = re.compile(r"(?:https?|ftp|mailto):[^\s<>()\[\]{}\"'`]+[^\s<>()\[\]{}\"'`.,;:!?]")
_BARE_DOMAIN = re.compile(r"\bwww\.[^\s<>()\[\]{}\"'`]+[^\s<>()\[\]{}\"'`.,;:!?]")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)
_MD_LINK_TARGET = re.compile(r"\]\(\s*(<[^>]*>|[^\s)]+)")
_MD_REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S+", re.MULTILINE)

# Quotation marks: straight and typographic, plus guillemets and CJK brackets.
_QUOTED = re.compile(
    r"[“„]([^“”„\n]{2,400})[”“]"
    r"|«([^«»\n]{2,400})»"
    r"|「([^「」\n]{2,400})」"
)
_STRAIGHT_QUOTED = re.compile(r'"([^"\n]{2,400})"')

_CITATION = re.compile(
    r"\((?:(?:see|cf\.|e\.g\.|i\.e\.)\s+)?[A-Z][\w.'’-]+"
    r"(?:\s+(?:et\s+al\.?|and|&|,)\s*[A-Z]?[\w.'’-]*)*,?\s+\d{4}[a-z]?\)"
    r"|\[\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\]"
    r"|\bdoi:\s*10\.\d{4,9}/\S+"
    r"|\barXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?"
)

_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b"
    rf"|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b"
    rf"|\b(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}}\b"
    rf"|\b\d{{1,2}}\s+(?:{_MONTHS})\.?\s+\d{{4}}\b"
    rf"|\b(?:{_MONTHS})\s+\d{{4}}\b",
    re.IGNORECASE,
)

# Word-character units need a trailing \b so "5 min" does not match inside
# "5 minutes"; symbol units like "%" and "°" cannot use \b at all, because a word
# boundary never occurs between two non-word characters.
_UNIT_WORD_SUFFIX = (
    r"percent|kg|km|cm|mm|nm|ms|GHz|MHz|kHz|Hz|GB|MB|KB|TB|kW|MW|mL|L|mg|g|lb|oz"
    r"|ft|in|mi|yd|s|min|h|hr|hrs|day|days|year|years|USD|EUR|GBP|bps|dpi|px"
)
_UNIT_SYMBOL_SUFFIX = r"%|°[CF]?|″|′"
_UNIT = re.compile(
    rf"\b\d+(?:[.,]\d+)*\s*(?:{_UNIT_WORD_SUFFIX})\b"
    rf"|\b\d+(?:[.,]\d+)*\s*(?:{_UNIT_SYMBOL_SUFFIX})"
    rf"|[$£€¥]\s?\d+(?:[.,]\d+)*(?:\s?(?:million|billion|trillion|k|M|B))?",
)

_NUMBER = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?(?![\w])")

# Identifiers: things that must survive verbatim but are not prose words.
_IDENTIFIER = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:[._-][A-Za-z0-9_]+)*\(\)"  # function()
    r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"  # CamelCase
    r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"  # snake_case
    r"|\b[A-Z]{2,}(?:_[A-Z0-9]+)+\b"  # CONSTANT_CASE
    r"|\b[0-9a-fA-F]{7,40}\b(?=\s|$|[.,;:)])"  # hashes / SHAs
    r"|\bv?\d+\.\d+\.\d+(?:-[\w.]+)?\b"  # semantic versions
)


def _add(spans: list[Span], text: str, start: int, end: int, kind: SpanKind, detector: str) -> None:
    if end <= start:
        return
    spans.append(Span(start=start, end=end, kind=kind, value=text[start:end], detector=detector))


def _scan(
    spans: list[Span], text: str, pattern: re.Pattern[str], kind: SpanKind, detector: str
) -> None:
    for match in pattern.finditer(text):
        _add(spans, text, match.start(), match.end(), kind, detector)


def compile_locks(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """Compile user-supplied lock regexes, raising a usage error on a bad pattern."""
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            msg = f"invalid --lock regex {pattern!r}: {exc}"
            raise UsageError(msg) from exc
    return tuple(compiled)


def discover_spans(
    text: str,
    blocks: Iterable[Block],
    *,
    policy: FidelityPolicy | None = None,
    locks: Sequence[str] = (),
    media_type: str = "text/plain",
) -> tuple[Span, ...]:
    """Find protected spans in ``text``.

    ``locks`` are user-supplied regexes; every match becomes a ``user_lock`` span
    with the highest priority.
    """
    policy = policy or FidelityPolicy()
    spans: list[Span] = []

    for pattern in compile_locks(locks):
        for match in pattern.finditer(text):
            if match.end() > match.start():
                _add(spans, text, match.start(), match.end(), "user_lock", "user_lock")

    if policy.lock_code:
        for block in blocks:
            if block.kind == "code":
                _add(spans, text, block.start, block.end, "code", "block:code")
        if media_type == "text/markdown":
            for match in _INLINE_CODE.finditer(text):
                _add(spans, text, match.start(), match.end(), "code", "markdown:inline_code")

    if policy.lock_urls:
        _scan(spans, text, _URL, "url", "url:scheme")
        _scan(spans, text, _BARE_DOMAIN, "url", "url:www")
        _scan(spans, text, _EMAIL, "url", "url:email")
        if media_type == "text/markdown":
            for match in _MD_LINK_TARGET.finditer(text):
                _add(spans, text, match.start(1), match.end(1), "url", "markdown:link_target")
            for match in _MD_REFERENCE.finditer(text):
                _add(spans, text, match.start(), match.end(), "url", "markdown:link_reference")

    if policy.lock_citations:
        _scan(spans, text, _CITATION, "citation", "citation")

    if policy.lock_quotes:
        _scan(spans, text, _QUOTED, "quote", "quote:typographic")
        # Straight quotes are ambiguous (they are also used for emphasis and
        # inches), so only lock them when they are balanced on the line.
        for match in _STRAIGHT_QUOTED.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            line = text[line_start : line_end if line_end != -1 else len(text)]
            if line.count('"') % 2 == 0:
                _add(spans, text, match.start(), match.end(), "quote", "quote:straight")

    if policy.lock_dates:
        _scan(spans, text, _DATE, "date", "date")
    if policy.lock_units:
        _scan(spans, text, _UNIT, "unit", "unit")
    if policy.lock_identifiers:
        _scan(spans, text, _IDENTIFIER, "identifier", "identifier")
    if policy.lock_numbers:
        _scan(spans, text, _NUMBER, "number", "number")

    return resolve_overlaps(spans)


def resolve_overlaps(spans: Iterable[Span]) -> tuple[Span, ...]:
    """Drop spans contained in or overlapping a higher-priority span.

    Ordering is by priority, then by descending length, then by offset, so the
    result is deterministic regardless of discovery order.
    """
    ordered = sorted(
        spans,
        key=lambda s: (_PRIORITY.get(s.kind, 99), -(s.end - s.start), s.start, s.kind),
    )
    kept: list[Span] = []
    for span in ordered:
        if any(existing.overlaps(span.start, span.end) for existing in kept):
            continue
        kept.append(span)
    return tuple(sorted(kept, key=lambda s: (s.start, s.end)))


def is_protected(spans: Sequence[Span], start: int, end: int) -> Span | None:
    """Return the first span protecting ``[start, end)``, if any."""
    for span in spans:
        if span.overlaps(start, end):
            return span
    return None
