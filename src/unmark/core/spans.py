"""Spans and blocks.

Offset convention
-----------------
Every ``start``/``end`` offset in Unmarked is an index into the original Python
``str`` of the source document, i.e. a **code point index**, not a byte index and
not a UTF-16 code unit index. Ranges are half-open: ``[start, end)``. Offsets
always refer to ``Document.source_text``, never to a candidate or intermediate
text. This makes offsets stable for the lifetime of a run, because the source is
immutable and candidates are represented as source plus an ordered operation log.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SpanKind = Literal[
    "quote",
    "citation",
    "url",
    "code",
    "formula",
    "entity",
    "number",
    "date",
    "unit",
    "identifier",
    "user_lock",
]

SpanPolicy = Literal["exact", "normalized_equal", "semantic"]

BlockKind = Literal["heading", "paragraph", "list_item", "quote", "code", "table"]


class StrictModel(BaseModel):
    """Base for every Unmarked contract model: strict, frozen, and closed."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class Span(StrictModel):
    """A protected region of the source document."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    kind: SpanKind
    value: str
    policy: SpanPolicy = "exact"
    detector: str | None = Field(
        default=None,
        description="Identifier of the heuristic that discovered this span.",
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.end < self.start:
            msg = f"span end {self.end} precedes start {self.start}"
            raise ValueError(msg)
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, start: int, end: int) -> bool:
        """Whether ``[start, end)`` intersects this span.

        Zero-length ranges touching a boundary do not count as overlapping, but a
        zero-length insertion strictly inside a protected span does.
        """
        if start == end:
            return self.start < start < self.end
        return start < self.end and self.start < end


class Block(StrictModel):
    """A structural block of the document."""

    id: str
    kind: BlockKind
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    language: str | None = None
    script: str | None = None
    level: int | None = Field(default=None, ge=1, le=6)

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.end < self.start:
            msg = f"block end {self.end} precedes start {self.start}"
            raise ValueError(msg)
        return self


class AtomicClaim(StrictModel):
    """A minimal, forward-compatible atomic-claim contract.

    Phase 3 will populate these from a claim extractor. The contract exists now so
    that document and report schemas do not change shape when it arrives.
    """

    id: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    block_id: str | None = None
    extractor: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.end < self.start:
            msg = f"claim end {self.end} precedes start {self.start}"
            raise ValueError(msg)
        return self
