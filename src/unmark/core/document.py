"""The immutable document intermediate representation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from unmark.core.operations import apply_operations, sha256_text
from unmark.core.spans import AtomicClaim, Block, Span, StrictModel

MediaType = Literal["text/plain", "text/markdown"]


class Document(StrictModel):
    """Parsed source document.

    ``source_text`` is immutable for the lifetime of a run; every candidate is
    this text plus an ordered operation log. All block, span, and claim offsets are
    code point indices into ``source_text``.
    """

    schema_version: Literal["1"] = "1"
    source_sha256: str
    source_text: str
    media_type: MediaType
    origin: str = Field(default="-", description="Path the text came from, or '-' for stdin.")
    blocks: tuple[Block, ...] = ()
    protected_spans: tuple[Span, ...] = ()
    claims: tuple[AtomicClaim, ...] = ()

    @model_validator(mode="after")
    def _check_offsets_and_hash(self) -> Self:
        expected = sha256_text(self.source_text)
        if self.source_sha256 != expected:
            msg = f"source_sha256 {self.source_sha256} does not match source text"
            raise ValueError(msg)
        length = len(self.source_text)
        for block in self.blocks:
            if block.end > length:
                msg = f"block {block.id} ends at {block.end}, past source length {length}"
                raise ValueError(msg)
        for span in self.protected_spans:
            if span.end > length:
                msg = f"span [{span.start}, {span.end}) is past source length {length}"
                raise ValueError(msg)
            if self.source_text[span.start : span.end] != span.value:
                msg = f"span [{span.start}, {span.end}) value does not match source text"
                raise ValueError(msg)
        for claim in self.claims:
            if claim.end > length:
                msg = f"claim {claim.id} ends at {claim.end}, past source length {length}"
                raise ValueError(msg)
        return self

    @classmethod
    def build(
        cls,
        source_text: str,
        media_type: MediaType,
        *,
        origin: str = "-",
        blocks: tuple[Block, ...] = (),
        protected_spans: tuple[Span, ...] = (),
        claims: tuple[AtomicClaim, ...] = (),
    ) -> Document:
        return cls(
            source_sha256=sha256_text(source_text),
            source_text=source_text,
            media_type=media_type,
            origin=origin,
            blocks=blocks,
            protected_spans=protected_spans,
            claims=claims,
        )

    def render(self, operations: tuple[object, ...] = ()) -> str:
        """Materialize a candidate; with no operations this returns the source."""
        if not operations:
            return self.source_text
        from unmark.core.operations import Operation  # local import keeps the type narrow

        typed = tuple(op for op in operations if isinstance(op, Operation))
        return apply_operations(self.source_text, typed)

    def spans_overlapping(self, start: int, end: int) -> tuple[Span, ...]:
        """Protected spans intersecting ``[start, end)``."""
        return tuple(span for span in self.protected_spans if span.overlaps(start, end))

    def block_at(self, offset: int) -> Block | None:
        for block in self.blocks:
            if block.start <= offset < block.end:
                return block
        return None
