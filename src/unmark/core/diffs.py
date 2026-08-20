"""Unified and operation-level diffs."""

from __future__ import annotations

import difflib
from collections.abc import Sequence

from unmark.core.operations import Operation


def unified_diff(
    source: str,
    candidate: str,
    *,
    source_label: str = "a/source",
    candidate_label: str = "b/candidate",
    context: int = 3,
) -> str:
    """A standard unified diff between source and candidate text."""
    source_lines = source.splitlines(keepends=True)
    candidate_lines = candidate.splitlines(keepends=True)
    diff = difflib.unified_diff(
        source_lines,
        candidate_lines,
        fromfile=source_label,
        tofile=candidate_label,
        n=context,
    )
    return "".join(line if line.endswith("\n") else line + "\n" for line in diff)


def _describe(text: str) -> str:
    """Render a fragment so invisible characters are visible in a diff."""
    if not text:
        return "''"
    parts = []
    for char in text:
        if char.isprintable() and char not in {" "}:
            parts.append(char)
        else:
            parts.append(f"U+{ord(char):04X}")
    return "".join(f"<{p}>" if p.startswith("U+") else p for p in parts)


def line_column(text: str, offset: int) -> tuple[int, int]:
    """1-based line and column for a code point offset."""
    prefix = text[:offset]
    line = prefix.count("\n") + 1
    last_newline = prefix.rfind("\n")
    column = offset - last_newline
    return line, column


def operation_diff(source: str, operations: Sequence[Operation]) -> str:
    """A human-readable, operation-level diff.

    Unlike a unified diff this shows exactly which operator made each change and
    why, which is what the report needs for attribution.
    """
    if not operations:
        return "no operations\n"
    lines = []
    for index, op in enumerate(sorted(operations, key=lambda o: (o.start, o.end)), start=1):
        line, column = line_column(source, op.start)
        original = source[op.start : op.end]
        lines.append(
            f"{index}. {op.kind} at {op.start}-{op.end} (line {line}, col {column}) [{op.operator}]"
        )
        lines.append(f"   - {_describe(original)}")
        lines.append(f"   + {_describe(op.text)}")
        lines.append(f"   reason: {op.reason}")
    return "\n".join(lines) + "\n"
