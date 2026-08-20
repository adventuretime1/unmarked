"""Conservative plain-text and Markdown block structure.

This is a deliberately small, offset-exact block scanner rather than a full
CommonMark parser. Unmarked only needs enough structure to (a) protect code and
quotes, and (b) report where edits land. A full parser would give a richer tree
but would also normalize text, and we cannot afford to lose byte-exact offsets
into the source string.

Known limitations, stated rather than hidden:

* Setext headings (``===`` underlines) are not recognized.
* Nested list structure is flattened; each list item is its own block.
* Tables are recognized only by a pipe-delimited header/delimiter pair.
* Fenced code takes precedence over every other construct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from unmark.core.document import MediaType
from unmark.core.spans import Block

_ATX_HEADING = re.compile(r"^(#{1,6})\s+\S")
_LIST_ITEM = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,9}[.)])\s+")
_BLOCK_QUOTE = re.compile(r"^\s{0,3}>")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
_TABLE_DELIMITER = re.compile(r"^\s{0,3}\|?[\s:-]*-[\s:|-]*\|?\s*$")


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    text: str


def _split_lines(text: str) -> list[_Line]:
    """Split into lines, keeping exact source offsets. Line text excludes the newline."""
    lines: list[_Line] = []
    start = 0
    for index, char in enumerate(text):
        if char == "\n":
            lines.append(_Line(start=start, end=index, text=text[start:index]))
            start = index + 1
    if start < len(text):
        lines.append(_Line(start=start, end=len(text), text=text[start:]))
    return lines


def parse_blocks(text: str, media_type: MediaType) -> tuple[Block, ...]:
    """Identify block structure for ``text``."""
    if media_type == "text/plain":
        return _parse_plain(text)
    return _parse_markdown(text)


def _parse_plain(text: str) -> tuple[Block, ...]:
    """Plain text: blank-line-separated paragraphs."""
    blocks: list[Block] = []
    counter = 0
    for match in re.finditer(r"[^\n](?:.|\n(?!\s*\n))*", text):
        start, end = match.start(), match.end()
        while end > start and text[end - 1] in "\n\r\t ":
            end -= 1
        if end <= start:
            continue
        counter += 1
        blocks.append(Block(id=f"b{counter}", kind="paragraph", start=start, end=end, script=None))
    return tuple(blocks)


def _finish(
    blocks: list[Block],
    kind: str,
    start: int,
    end: int,
    counter: int,
    level: int | None = None,
) -> int:
    if end <= start:
        return counter
    counter += 1
    blocks.append(
        Block(
            id=f"b{counter}",
            kind=kind,  # type: ignore[arg-type]
            start=start,
            end=end,
            level=level,
        )
    )
    return counter


def _parse_markdown(text: str) -> tuple[Block, ...]:
    lines = _split_lines(text)
    blocks: list[Block] = []
    counter = 0

    paragraph_start: int | None = None
    paragraph_end = 0
    fence: str | None = None
    fence_start = 0

    def flush_paragraph() -> None:
        nonlocal counter, paragraph_start
        if paragraph_start is not None:
            counter = _finish(blocks, "paragraph", paragraph_start, paragraph_end, counter)
            paragraph_start = None

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.text.strip()

        if fence is not None:
            closing = _FENCE.match(line.text)
            # A closing fence must use the same character and be at least as long.
            if (
                closing is not None
                and closing.group(1)[0] == fence[0]
                and len(closing.group(1)) >= len(fence)
            ):
                counter = _finish(blocks, "code", fence_start, line.end, counter)
                fence = None
            index += 1
            continue

        fence_match = _FENCE.match(line.text)
        if fence_match is not None:
            flush_paragraph()
            fence = fence_match.group(1)
            fence_start = line.start
            index += 1
            continue

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        heading = _ATX_HEADING.match(line.text)
        if heading is not None:
            flush_paragraph()
            counter = _finish(
                blocks, "heading", line.start, line.end, counter, level=len(heading.group(1))
            )
            index += 1
            continue

        if _BLOCK_QUOTE.match(line.text):
            flush_paragraph()
            start = line.start
            end = line.end
            while index + 1 < len(lines) and _BLOCK_QUOTE.match(lines[index + 1].text):
                index += 1
                end = lines[index].end
            counter = _finish(blocks, "quote", start, end, counter)
            index += 1
            continue

        if _LIST_ITEM.match(line.text):
            flush_paragraph()
            start = line.start
            end = line.end
            # Continuation lines of the same item are indented and not new items.
            while (
                index + 1 < len(lines)
                and lines[index + 1].text.strip()
                and not _LIST_ITEM.match(lines[index + 1].text)
                and lines[index + 1].text.startswith((" ", "\t"))
            ):
                index += 1
                end = lines[index].end
            counter = _finish(blocks, "list_item", start, end, counter)
            index += 1
            continue

        if (
            "|" in line.text
            and index + 1 < len(lines)
            and _TABLE_DELIMITER.match(lines[index + 1].text)
            and "|" in lines[index + 1].text
        ):
            flush_paragraph()
            start = line.start
            end = lines[index + 1].end
            index += 2
            while index < len(lines) and "|" in lines[index].text and lines[index].text.strip():
                end = lines[index].end
                index += 1
            counter = _finish(blocks, "table", start, end, counter)
            continue

        if paragraph_start is None:
            paragraph_start = line.start
        paragraph_end = line.end
        index += 1

    if fence is not None:
        # Unterminated fence: treat the remainder as code so it stays protected.
        counter = _finish(blocks, "code", fence_start, len(text), counter)
    else:
        flush_paragraph()

    return tuple(sorted(blocks, key=lambda b: b.start))
