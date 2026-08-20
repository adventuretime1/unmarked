"""Minimal script detection.

Used to decide whether a format character is plausibly meaningful in context.
This is a coarse block lookup, not a full UAX #24 implementation; it only needs to
answer "which writing systems surround this offset".
"""

from __future__ import annotations

import unicodedata

# (inclusive_start, inclusive_end, script)
_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x024F, "Latin"),
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0750, 0x077F, "Arabic"),
    (0x0780, 0x07BF, "Thaana"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"),
    (0x1200, 0x137F, "Ethiopic"),
    (0x1780, 0x17FF, "Khmer"),
    (0x1800, 0x18AF, "Mongolian"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x3400, 0x4DBF, "Han"),
    (0x4E00, 0x9FFF, "Han"),
    (0xA000, 0xA4CF, "Yi"),
    (0xAC00, 0xD7AF, "Hangul"),
    (0xFB1D, 0xFB4F, "Hebrew"),
    (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"),
    (0x10E60, 0x10E7F, "Arabic"),
    (0x1E900, 0x1E95F, "Adlam"),
    (0x20000, 0x2A6DF, "Han"),
)

#: Scripts in which ZWNJ and ZWJ carry orthographic meaning.
JOINING_SCRIPTS: frozenset[str] = frozenset(
    {
        "Arabic",
        "Syriac",
        "Thaana",
        "Adlam",
        "Devanagari",
        "Bengali",
        "Gurmukhi",
        "Gujarati",
        "Oriya",
        "Tamil",
        "Telugu",
        "Kannada",
        "Malayalam",
        "Sinhala",
        "Myanmar",
        "Khmer",
        "Tibetan",
        "Mongolian",
    }
)

#: Scripts written right-to-left, where bidi controls are routinely meaningful.
RTL_SCRIPTS: frozenset[str] = frozenset({"Arabic", "Hebrew", "Syriac", "Thaana", "Adlam"})


def script_of(char: str) -> str:
    """Coarse script name for a single character."""
    code = ord(char)
    for start, end, name in _RANGES:
        if start <= code <= end:
            return name
    category = unicodedata.category(char)
    if category.startswith(("L", "M")):
        return "Unknown"
    return "Common"


def neighbouring_scripts(text: str, offset: int, window: int = 12) -> frozenset[str]:
    """Scripts of the letters surrounding ``offset``, ignoring Common/Unknown.

    The character *at* ``offset`` is excluded so a script-neutral carrier is judged
    by its surroundings, not by the block it happens to be encoded in. A Khmer
    inherent vowel dropped into Latin prose, for instance, must read as "no Khmer
    context" rather than counting itself as Khmer.
    """
    start = max(0, offset - window)
    end = min(len(text), offset + window + 1)
    found = {
        script_of(char)
        for index, char in enumerate(text[start:end], start=start)
        if index != offset and unicodedata.category(char).startswith(("L", "M"))
    }
    return frozenset(found - {"Common", "Unknown"})


def document_scripts(text: str) -> frozenset[str]:
    """All scripts appearing in the document."""
    found = {script_of(char) for char in text if unicodedata.category(char).startswith(("L", "M"))}
    return frozenset(found - {"Common", "Unknown"})
