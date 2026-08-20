"""Operations: the only way candidate text is derived from an immutable source.

A candidate is always ``source_text`` plus an ordered tuple of :class:`Operation`.
Applying operations is deterministic, validated, and exactly invertible, so diffs,
offsets, and rollback stay precise for the lifetime of a run.

All operation offsets are code point indices into the **source** text (see
:mod:`unmark.core.spans` for the offset convention). Because every operation is
anchored to the source rather than to the partially-edited text, an operation list
is order-independent in meaning; Unmarked still stores it sorted for determinism.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from unmark.core.errors import OperationError
from unmark.core.spans import StrictModel

OperationKind = Literal["delete", "insert", "replace"]


class Operation(StrictModel):
    """A single source-anchored text edit.

    ``start``/``end`` delimit the replaced source region; ``text`` is the
    replacement. A pure insertion has ``start == end``; a pure deletion has an
    empty ``text``.
    """

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str = ""
    reason: str = Field(description="Why this edit was made; surfaced in reports.")
    operator: str = Field(description="Identifier of the operator that produced it.")
    original: str | None = Field(
        default=None,
        description="Source text being replaced; validated against the document on apply.",
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.end < self.start:
            msg = f"operation end {self.end} precedes start {self.start}"
            raise ValueError(msg)
        if self.start == self.end and not self.text:
            msg = "operation is a no-op: empty range and empty replacement"
            raise ValueError(msg)
        return self

    @property
    def kind(self) -> OperationKind:
        if self.start == self.end:
            return "insert"
        if not self.text:
            return "delete"
        return "replace"

    @property
    def length_delta(self) -> int:
        return len(self.text) - (self.end - self.start)


def sort_operations(operations: Iterable[Operation]) -> tuple[Operation, ...]:
    """Return operations in deterministic application order."""
    return tuple(sorted(operations, key=lambda op: (op.start, op.end, op.text)))


def validate_operations(operations: Sequence[Operation], source: str) -> tuple[Operation, ...]:
    """Validate bounds, overlap, and recorded originals; return sorted operations.

    Raises :class:`OperationError` for any out-of-bounds, overlapping, or
    mismatched operation. Two insertions at the same offset are rejected because
    their relative order would be ambiguous.
    """
    ordered = sort_operations(operations)
    length = len(source)
    previous: Operation | None = None

    for op in ordered:
        if op.end > length:
            msg = f"operation [{op.start}, {op.end}) exceeds source length {length}"
            raise OperationError(msg)
        if op.original is not None and source[op.start : op.end] != op.original:
            msg = (
                f"operation at [{op.start}, {op.end}) expected original "
                f"{op.original!r} but source has {source[op.start : op.end]!r}"
            )
            raise OperationError(msg)
        if previous is not None:
            if op.start < previous.end:
                msg = (
                    f"operations overlap: [{previous.start}, {previous.end}) and "
                    f"[{op.start}, {op.end})"
                )
                raise OperationError(msg)
            if op.start == previous.start and op.end == previous.end and op.start == op.end:
                msg = f"two insertions share offset {op.start}; order is ambiguous"
                raise OperationError(msg)
        previous = op

    return ordered


def apply_operations(source: str, operations: Sequence[Operation]) -> str:
    """Apply validated operations to ``source`` and return the candidate text."""
    ordered = validate_operations(operations, source)
    parts: list[str] = []
    cursor = 0
    for op in ordered:
        parts.append(source[cursor : op.start])
        parts.append(op.text)
        cursor = op.end
    parts.append(source[cursor:])
    return "".join(parts)


def invert_operations(source: str, operations: Sequence[Operation]) -> tuple[Operation, ...]:
    """Build operations that map the candidate text back to ``source``.

    The returned operations are anchored to the *candidate* text produced by
    applying ``operations`` to ``source``, so ``apply_operations(candidate,
    invert_operations(source, operations)) == source``.
    """
    ordered = validate_operations(operations, source)
    inverse: list[Operation] = []
    drift = 0
    for op in ordered:
        start = op.start + drift
        end = start + len(op.text)
        original = source[op.start : op.end]
        if start != end or original:
            # Two deletions that were adjacent in the source invert to two
            # insertions at the same candidate offset, which is ambiguous. Merge
            # them: source order fixes the correct concatenation order.
            if inverse and start == end and inverse[-1].start == inverse[-1].end == start:
                previous = inverse[-1]
                inverse[-1] = previous.model_copy(update={"text": previous.text + original})
            else:
                inverse.append(
                    Operation(
                        start=start,
                        end=end,
                        text=original,
                        reason=f"rollback: {op.reason}",
                        operator=f"invert:{op.operator}",
                        original=op.text or None,
                    )
                )
        drift += op.length_delta
    return tuple(inverse)


def rollback(source: str, operations: Sequence[Operation]) -> str:
    """Apply then invert, returning the recovered source. Used to prove exactness."""
    candidate = apply_operations(source, operations)
    return apply_operations(candidate, invert_operations(source, operations))


def normalized_content_hash(text: str) -> str:
    """Stable content hash over NFC-normalized text.

    Normalization means two candidates that differ only in Unicode composition
    form hash identically, which is what candidate deduplication needs. The raw
    source hash (:func:`sha256_text`) is deliberately *not* normalized.
    """
    normalized = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    """Exact SHA-256 of the UTF-8 encoding of ``text``, without normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def char_edit_ratio(source: str, candidate: str, operations: Sequence[Operation]) -> float:
    """Fraction of source characters touched by ``operations``.

    Measured from the operation log rather than from a string distance so the
    number matches exactly what the engine did. Clamped to ``[0, 1]``.
    """
    if not source:
        return 0.0 if not candidate else 1.0
    touched = sum(max(op.end - op.start, len(op.text)) for op in operations)
    return min(touched / len(source), 1.0)


def length_drift_ratio(source: str, candidate: str) -> float:
    """Absolute length change relative to the source, clamped to ``[0, 1]``."""
    if not source:
        return 0.0 if not candidate else 1.0
    return min(abs(len(candidate) - len(source)) / len(source), 1.0)
