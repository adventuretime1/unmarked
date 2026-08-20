"""Script-aware Unicode inspection and conservative sanitation.

Design stance
-------------
The reference implementations surveyed in the research document strip every
``Cf`` (format) character by default. That is unsafe: ZWJ and variation selectors
drive emoji rendering, ZWNJ is orthographically required in Persian and several
Indic scripts, bidi controls matter in Arabic and Hebrew, and NBSP/thin spaces are
deliberate typography.

Unmarked therefore classifies each finding and decides removal from *context*:

* A finding is only removed when the active policy allows its class **and** the
  character is not plausibly meaningful where it appears.
* Emoji sequences and language controls are preserved by default.
* Every removal or replacement becomes an :class:`~unmark.core.operations.Operation`
  with a human-readable reason, so the report can explain each change.
* ``aggressive`` ignores context and is research-only.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from unmark.core.operations import Operation, apply_operations
from unmark.core.policies import UnicodePolicy
from unmark.core.spans import Span, StrictModel
from unmark.inspect.scripts import JOINING_SCRIPTS, RTL_SCRIPTS, neighbouring_scripts

FindingKind = Literal[
    "zero_width",
    "bidi_control",
    "variation_selector",
    "tag_character",
    "unusual_space",
    "non_breaking_space",
    "control",
    "format",
    "deprecated",
    "confusable",
    "private_use",
    "replacement_character",
]

Severity = Literal["info", "notice", "suspicious"]

ZWSP = "​"
ZWNJ = "‌"
ZWJ = "‍"
LRM = "‎"
RLM = "‏"
WORD_JOINER = "⁠"
BOM = "﻿"
NBSP = " "
NARROW_NBSP = " "
SOFT_HYPHEN = "­"
COMBINING_GRAPHEME_JOINER = "͏"
KHMER_INHERENT_AQ = "឴"
KHMER_INHERENT_AA = "឵"
OGHAM_SPACE_MARK = " "
VARIATION_SELECTOR_15 = "︎"
VARIATION_SELECTOR_16 = "️"
COMBINING_ENCLOSING_KEYCAP = "⃣"

#: Zero-width and invisible characters with no legitimate role in plain prose.
_ALWAYS_UNSAFE_INVISIBLE: frozenset[str] = frozenset(
    {
        ZWSP,
        BOM,
        WORD_JOINER,
        "᠎",  # Mongolian vowel separator (deprecated as a space)
        "⁡",  # function application
        "⁢",  # invisible times
        "⁣",  # invisible separator
        "⁤",  # invisible plus
        "ᅟ",  # Hangul choseong filler
        "ᅠ",  # Hangul jungseong filler
        "ㅤ",  # Hangul filler
        "ﾠ",  # halfwidth Hangul filler
    }
)

#: Bidi formatting and isolate controls.
_BIDI_CONTROLS: frozenset[str] = frozenset(
    {
        LRM,
        RLM,
        "؜",  # Arabic letter mark
        "‪",  # LRE
        "‫",  # RLE
        "‬",  # PDF
        "‭",  # LRO
        "‮",  # RLO
        "⁦",  # LRI
        "⁧",  # RLI
        "⁨",  # FSI
        "⁩",  # PDI
    }
)

#: Overriding controls are dangerous even in RTL text: they can reorder rendering
#: to disguise content. They are reported as suspicious regardless of script.
_BIDI_OVERRIDES: frozenset[str] = frozenset({"‭", "‮"})

#: Spaces that are not U+0020 but are legitimate typography.
_SPACE_REPLACEMENTS: dict[str, str] = {
    " ": " ",  # en quad
    " ": " ",  # em quad
    " ": " ",  # en space
    " ": " ",  # em space
    " ": " ",  # three-per-em
    " ": " ",  # four-per-em
    " ": " ",  # six-per-em
    " ": " ",  # punctuation space
    " ": " ",  # thin space
    " ": " ",  # hair space
    " ": " ",  # medium mathematical space
    "　": " ",  # ideographic space
}

#: Spaces that carry meaning and are preserved unless policy is aggressive.
_MEANINGFUL_SPACES: frozenset[str] = frozenset(
    {
        NBSP,
        NARROW_NBSP,
        " ",  # figure space
        "‑",  # non-breaking hyphen
    }
)

#: Cyrillic/Greek characters commonly used as Latin homoglyphs.
_CONFUSABLES: dict[str, str] = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
    "у": "y",
    "А": "A",
    "В": "B",
    "Е": "E",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "Х": "X",
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Χ": "X",
    "ο": "o",
    "ԁ": "d",
    "ԛ": "q",
}

_WORD = re.compile(r"\w+", re.UNICODE)


class UnicodeFinding(StrictModel):
    """One reported code point."""

    schema_version: Literal["1"] = "1"
    offset: int = Field(ge=0)
    codepoint: int = Field(ge=0)
    char: str
    name: str
    category: str
    kind: FindingKind
    severity: Severity
    reason: str
    removable: bool = Field(
        description="Whether the active policy may remove or replace this finding."
    )
    replacement: str | None = None
    protected_by: str | None = Field(
        default=None,
        description="Why the finding was preserved, when it was not removable.",
    )

    @property
    def label(self) -> str:
        return f"U+{self.codepoint:04X}"


def _char_name(char: str) -> str:
    try:
        return unicodedata.name(char)
    except ValueError:
        return f"<unnamed U+{ord(char):04X}>"


def _is_emoji_context(text: str, offset: int) -> bool:
    """Whether the character at ``offset`` sits inside an emoji sequence."""
    for neighbour in (offset - 1, offset + 1):
        if 0 <= neighbour < len(text):
            char = text[neighbour]
            code = ord(char)
            if (
                0x1F000 <= code <= 0x1FAFF
                or 0x2600 <= code <= 0x27BF
                or 0x2190 <= code <= 0x21FF
                or 0xFE00 <= code <= 0xFE0F
                or code in {0x00A9, 0x00AE, 0x2122, 0x20E3}
                or 0x1F1E6 <= code <= 0x1F1FF
                or 0xE0020 <= code <= 0xE007F
                or char.isdigit()
                or char in {"#", "*"}
            ):
                return True
    return False


def _in_emoji_tag_sequence(text: str, offset: int) -> bool:
    """Whether the tag character at ``offset`` belongs to a real emoji tag sequence.

    A valid sequence is a base emoji (the black flag, U+1F3F4) followed by tag
    characters and terminated by the cancel tag U+E007F. Walk back over the run of
    tag characters and require that base; a bare run of tag characters in prose is
    a hidden payload, not a flag.
    """
    index = offset
    while index > 0 and 0xE0000 <= ord(text[index - 1]) <= 0xE007F:
        index -= 1
    return index > 0 and ord(text[index - 1]) == 0x1F3F4


def _classify(text: str, offset: int, char: str, policy: UnicodePolicy) -> UnicodeFinding | None:
    """Classify one character, or return ``None`` when it is unremarkable."""
    code = ord(char)
    category = unicodedata.category(char)
    aggressive = policy.name == "aggressive"

    def finding(
        kind: FindingKind,
        severity: Severity,
        reason: str,
        *,
        removable: bool,
        replacement: str | None = None,
        protected_by: str | None = None,
    ) -> UnicodeFinding:
        return UnicodeFinding(
            offset=offset,
            codepoint=code,
            char=char,
            name=_char_name(char),
            category=category,
            kind=kind,
            severity=severity,
            reason=reason,
            removable=removable,
            replacement=replacement,
            protected_by=protected_by,
        )

    if char in _ALWAYS_UNSAFE_INVISIBLE:
        return finding(
            "zero_width",
            "suspicious",
            "invisible character with no linguistic role in running text",
            removable=True,
            replacement="",
        )

    if char in (ZWNJ, ZWJ):
        scripts = neighbouring_scripts(text, offset)
        emoji = _is_emoji_context(text, offset)
        if char == ZWJ and emoji and policy.preserve_emoji_sequences and not aggressive:
            return finding(
                "zero_width",
                "info",
                "zero-width joiner inside an emoji sequence",
                removable=False,
                protected_by="preserve_emoji_sequences",
            )
        if scripts & JOINING_SCRIPTS and policy.preserve_language_controls and not aggressive:
            joined = ", ".join(sorted(scripts & JOINING_SCRIPTS))
            return finding(
                "zero_width",
                "info",
                f"joiner is orthographically meaningful in {joined}",
                removable=False,
                protected_by="preserve_language_controls",
            )
        return finding(
            "zero_width",
            "suspicious",
            "zero-width joiner outside any script or emoji context that needs it",
            removable=True,
            replacement="",
        )

    if char in _BIDI_CONTROLS:
        scripts = neighbouring_scripts(text, offset, window=40)
        if char in _BIDI_OVERRIDES:
            return finding(
                "bidi_control",
                "suspicious",
                "bidirectional override can disguise the rendered order of text",
                removable=aggressive,
                replacement="" if aggressive else None,
                protected_by=None if aggressive else "requires_research_mode",
            )
        if scripts & RTL_SCRIPTS and policy.preserve_language_controls and not aggressive:
            joined = ", ".join(sorted(scripts & RTL_SCRIPTS))
            return finding(
                "bidi_control",
                "info",
                f"bidi control is meaningful alongside {joined}",
                removable=False,
                protected_by="preserve_language_controls",
            )
        if not scripts & RTL_SCRIPTS:
            # No right-to-left text anywhere near this control, so it cannot be
            # doing the job bidi controls exist for. Removing it cannot change the
            # rendered order of purely left-to-right text.
            return finding(
                "bidi_control",
                "suspicious",
                "bidi control with no adjacent right-to-left text",
                removable=True,
                replacement="",
            )
        return finding(
            "bidi_control",
            "notice",
            "bidi control in mixed-direction text",
            removable=aggressive,
            replacement="" if aggressive else None,
            protected_by=None if aggressive else "preserve_language_controls",
        )

    if 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF:
        if policy.preserve_emoji_sequences and _is_emoji_context(text, offset) and not aggressive:
            return finding(
                "variation_selector",
                "info",
                "variation selector controls emoji presentation",
                removable=False,
                protected_by="preserve_emoji_sequences",
            )
        scripts = neighbouring_scripts(text, offset)
        if scripts & {"Han", "Hiragana", "Katakana", "Mongolian"} and not aggressive:
            return finding(
                "variation_selector",
                "info",
                "variation selector controls a script-specific glyph variant",
                removable=False,
                protected_by="preserve_language_controls",
            )
        return finding(
            "variation_selector",
            "suspicious",
            "variation selector has no valid emoji or script presentation context",
            removable=True,
            replacement="",
        )

    if 0xE0000 <= code <= 0xE007F:
        # Tag characters are only legitimate in an emoji tag sequence, which must
        # begin with a base emoji (U+1F3F4 in practice). Another tag character
        # nearby is not evidence of legitimacy -- that is exactly what a hidden
        # payload looks like -- so scan back to the start of the tag run.
        if _in_emoji_tag_sequence(text, offset) and policy.preserve_emoji_sequences:
            return finding(
                "tag_character",
                "notice",
                "tag character inside an emoji tag sequence",
                removable=False,
                protected_by="preserve_emoji_sequences",
            )
        return finding(
            "tag_character",
            "suspicious",
            "tag character can encode hidden data invisibly",
            removable=True,
            replacement="",
        )

    if char == COMBINING_GRAPHEME_JOINER:
        # U+034F is an invisible combining mark with almost no legitimate role in
        # running prose; it is a known steganographic carrier. It *can* matter in a
        # handful of scripts (e.g. to block a canonical reordering), so preserve it
        # when language controls are respected and a joining/RTL script is adjacent.
        scripts = neighbouring_scripts(text, offset)
        if scripts & (JOINING_SCRIPTS | RTL_SCRIPTS) and policy.preserve_language_controls:
            joined = ", ".join(sorted(scripts & (JOINING_SCRIPTS | RTL_SCRIPTS)))
            return finding(
                "zero_width",
                "notice",
                f"combining grapheme joiner can affect ordering in {joined}",
                removable=False,
                protected_by="preserve_language_controls",
            )
        return finding(
            "zero_width",
            "suspicious",
            "combining grapheme joiner is invisible and has no role in this context",
            removable=True,
            replacement="",
        )

    if 0x180B <= code <= 0x180D:
        # Mongolian free variation selectors behave like variation selectors: they
        # pick a glyph form and are meaningful in Mongolian text.
        scripts = neighbouring_scripts(text, offset)
        if "Mongolian" in scripts and policy.preserve_language_controls and not aggressive:
            return finding(
                "variation_selector",
                "info",
                "Mongolian free variation selector controls glyph form",
                removable=False,
                protected_by="preserve_language_controls",
            )
        return finding(
            "variation_selector",
            "suspicious",
            "Mongolian free variation selector outside any Mongolian context",
            removable=True,
            replacement="",
        )

    if char in (KHMER_INHERENT_AQ, KHMER_INHERENT_AA):
        # Deprecated Khmer inherent vowels: invisible, and the Unicode standard
        # recommends against them. Preserve inside Khmer text under language
        # controls; otherwise they are a hidden carrier.
        scripts = neighbouring_scripts(text, offset)
        if "Khmer" in scripts and policy.preserve_language_controls and not aggressive:
            return finding(
                "zero_width",
                "notice",
                "deprecated Khmer inherent vowel kept inside Khmer text",
                removable=False,
                protected_by="preserve_language_controls",
            )
        return finding(
            "zero_width",
            "suspicious",
            "deprecated Khmer inherent vowel outside any Khmer context",
            removable=True,
            replacement="",
        )

    if char == OGHAM_SPACE_MARK:
        # Ogham space mark renders as a visible mark in Ogham but as blank space in
        # most fonts; treat it as an unusual space, normalized only under the
        # typographic policy and only outside Ogham text.
        allowed = policy.name in {"typographic", "aggressive"} and policy.normalize_spaces
        return finding(
            "unusual_space",
            "notice",
            "Ogham space mark; normalizes to a plain space outside Ogham text",
            removable=allowed,
            replacement=" " if allowed else None,
            protected_by=None if allowed else "requires_typographic_policy",
        )

    if char == SOFT_HYPHEN:
        return finding(
            "format",
            "notice",
            "soft hyphen is invisible unless the line wraps",
            removable=policy.name in {"typographic", "aggressive"},
            replacement="" if policy.name in {"typographic", "aggressive"} else None,
            protected_by=None if policy.name in {"typographic", "aggressive"} else "safe_policy",
        )

    if char in _MEANINGFUL_SPACES:
        return finding(
            "non_breaking_space",
            "info",
            "non-breaking or fixed-width space is often deliberate typography",
            removable=aggressive,
            replacement=" " if aggressive else None,
            protected_by=None if aggressive else "preserve_typography",
        )

    if char in _SPACE_REPLACEMENTS:
        # Breakable spaces are safe to canonicalize on the normal path. Spaces
        # with layout semantics (NBSP, narrow NBSP, figure space) are handled by
        # _MEANINGFUL_SPACES above and remain protected.
        allowed = policy.name in {"safe", "typographic", "aggressive"} and policy.normalize_spaces
        return finding(
            "unusual_space",
            "notice",
            "unusual space character; normalizes to a plain space",
            removable=allowed,
            replacement=" " if allowed else None,
            protected_by=None if allowed else "space_normalization_disabled",
        )

    if char == "�":
        return finding(
            "replacement_character",
            "suspicious",
            "replacement character indicates prior decoding damage",
            removable=False,
            protected_by="lossy_removal_would_hide_damage",
        )

    if category == "Co":
        return finding(
            "private_use",
            "suspicious",
            "private use character has no standard meaning",
            removable=aggressive,
            replacement="" if aggressive else None,
            protected_by=None if aggressive else "requires_research_mode",
        )

    if category in {"Cc"} and char not in {"\n", "\t", "\r"}:
        return finding(
            "control",
            "suspicious",
            "control character is not printable and is rarely intentional",
            removable=True,
            replacement="",
        )

    if category == "Cf":
        return finding(
            "format",
            "notice",
            "format character with no recognized role here",
            removable=policy.name in {"typographic", "aggressive"},
            replacement="" if policy.name in {"typographic", "aggressive"} else None,
            protected_by=None if policy.name in {"typographic", "aggressive"} else "safe_policy",
        )

    return None


def _confusable_findings(text: str) -> list[UnicodeFinding]:
    """Report Cyrillic/Greek homoglyphs inside otherwise-Latin words.

    Only reported, never removed: substituting a letter changes the text's meaning
    for a reader of that script, and a false positive would corrupt legitimate
    multilingual prose.
    """
    findings: list[UnicodeFinding] = []
    for match in _WORD.finditer(text):
        word = match.group()
        if len(word) < 2:
            continue
        has_latin = any("a" <= c.lower() <= "z" for c in word)
        if not has_latin:
            continue
        for index, char in enumerate(word):
            if char in _CONFUSABLES:
                offset = match.start() + index
                findings.append(
                    UnicodeFinding(
                        offset=offset,
                        codepoint=ord(char),
                        char=char,
                        name=_char_name(char),
                        category=unicodedata.category(char),
                        kind="confusable",
                        severity="suspicious",
                        reason=(
                            f"character resembles Latin {_CONFUSABLES[char]!r} "
                            f"inside the mixed-script word {word!r}"
                        ),
                        removable=False,
                        protected_by="substitution_would_change_meaning",
                    )
                )
    return findings


def inspect_text(text: str, policy: UnicodePolicy | None = None) -> tuple[UnicodeFinding, ...]:
    """Report every notable code point in ``text``."""
    policy = policy or UnicodePolicy()
    findings: list[UnicodeFinding] = []
    for offset, char in enumerate(text):
        finding = _classify(text, offset, char, policy)
        if finding is not None:
            findings.append(finding)
    findings.extend(_confusable_findings(text))
    return tuple(sorted(findings, key=lambda f: (f.offset, f.codepoint)))


def sanitation_operations(
    text: str,
    findings: Sequence[UnicodeFinding],
    policy: UnicodePolicy,
    protected_spans: Sequence[Span] = (),
) -> tuple[Operation, ...]:
    """Turn removable findings into operations.

    ``report`` produces nothing. Findings inside a protected span are skipped and
    the reason is recorded on the finding by the caller.
    """
    if not policy.mutates:
        return ()

    operations: list[Operation] = []
    for finding in findings:
        if not finding.removable or finding.replacement is None:
            continue
        if any(span.overlaps(finding.offset, finding.offset + 1) for span in protected_spans):
            continue
        operations.append(
            Operation(
                start=finding.offset,
                end=finding.offset + 1,
                text=finding.replacement,
                original=finding.char,
                operator=f"unicode:{finding.kind}",
                reason=finding.reason,
            )
        )

    normalization = _normalization_operations(text, policy, protected_spans)
    if not normalization:
        return tuple(operations)
    normalization_form = policy.normalization_form
    assert normalization_form in {"NFC", "NFKC"}

    # A variation selector or another removable combining mark can share a
    # normalization cluster. Replace that cluster once so operations remain
    # non-overlapping and source-anchored.
    consumed: set[int] = set()
    combined: list[Operation] = []
    for norm in normalization:
        overlapping = [
            (index, operation)
            for index, operation in enumerate(operations)
            if operation.start < norm.end and norm.start < operation.end
        ]
        if not overlapping:
            combined.append(norm)
            continue
        local = tuple(
            operation.model_copy(
                update={
                    "start": operation.start - norm.start,
                    "end": operation.end - norm.start,
                }
            )
            for _, operation in overlapping
        )
        original = text[norm.start : norm.end]
        sanitized = apply_operations(original, local)
        replacement = unicodedata.normalize(normalization_form, sanitized)
        consumed.update(index for index, _ in overlapping)
        if replacement != original:
            combined.append(
                Operation(
                    start=norm.start,
                    end=norm.end,
                    text=replacement,
                    original=original,
                    operator="unicode:canonicalize",
                    reason="carrier cleanup and Unicode canonical composition",
                )
            )
    combined.extend(
        operation for index, operation in enumerate(operations) if index not in consumed
    )
    return tuple(sorted(combined, key=lambda operation: (operation.start, operation.end)))


def _normalization_operations(
    text: str,
    policy: UnicodePolicy,
    protected_spans: Sequence[Span],
) -> tuple[Operation, ...]:
    """Return source-anchored NFC/NFKC operations outside protected spans.

    Work a combining cluster at a time instead of normalizing the whole string.
    This preserves exact locks and keeps the operation log precise. It covers the
    canonical decomposed sequences used as practical carrier variants without a
    blanket compatibility rewrite across protected code, quotes, URLs, or IDs.
    """
    form = policy.normalization_form
    if form == "none":
        return ()
    operations: list[Operation] = []
    offset = 0
    while offset < len(text):
        end = offset + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        if any(span.overlaps(offset, end) for span in protected_spans):
            offset = end
            continue
        original = text[offset:end]
        normalized = unicodedata.normalize(form, original)
        if normalized != original:
            operations.append(
                Operation(
                    start=offset,
                    end=end,
                    text=normalized,
                    original=original,
                    operator=f"unicode:{form.lower()}",
                    reason=f"canonical Unicode composition ({form})",
                )
            )
        offset = end
    return tuple(operations)


def blocked_by_protection(
    findings: Sequence[UnicodeFinding], protected_spans: Sequence[Span]
) -> tuple[UnicodeFinding, ...]:
    """Removable findings that were skipped because they sit in a protected span."""
    blocked = []
    for finding in findings:
        if not finding.removable or finding.replacement is None:
            continue
        if any(span.overlaps(finding.offset, finding.offset + 1) for span in protected_spans):
            blocked.append(finding)
    return tuple(blocked)
