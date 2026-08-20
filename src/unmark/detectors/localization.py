"""Character-aligned regions and shared localization mechanics."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import Field, model_validator

from unmark.core.spans import Span, StrictModel

_TOKEN_RE = re.compile(r"\S+")


class TokenAlignment(StrictModel):
    """One tokenizer token mapped back to exact text character offsets."""

    index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _valid_range(self) -> TokenAlignment:
        if self.end <= self.start:
            raise ValueError("token alignment must cover at least one character")
        return self


class TextRegion(StrictModel):
    """A ranked, editable source region."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    risk: float
    mode: str
    token_start: int | None = Field(default=None, ge=0)
    token_end: int | None = Field(default=None, ge=0)
    rationale: str = ""

    @model_validator(mode="after")
    def _valid_range(self) -> TextRegion:
        if self.end <= self.start:
            raise ValueError("region must cover at least one character")
        return self

    def overlaps(self, other: TextRegion) -> bool:
        return self.start < other.end and other.start < self.end


def align_regex_tokens(text: str) -> tuple[TokenAlignment, ...]:
    """Deterministic fallback alignment for window boundaries.

    This is deliberately named as a regex alignment rather than a model-token
    alignment.  ML adapters must supply their own offsets and never assume that
    whitespace tokens correspond to model tokens.
    """
    return tuple(
        TokenAlignment(index=index, start=match.start(), end=match.end(), text=match.group())
        for index, match in enumerate(_TOKEN_RE.finditer(text))
    )


def validate_alignments(text: str, tokens: Sequence[TokenAlignment]) -> None:
    previous_end = 0
    for token in tokens:
        if token.end > len(text) or token.start < previous_end:
            raise ValueError("token alignments are overlapping, unsorted, or out of bounds")
        if text[token.start : token.end] != token.text:
            raise ValueError("token alignment text does not match its character offsets")
        previous_end = token.end


def token_windows(
    tokens: Sequence[TokenAlignment], window_tokens: int, stride: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Return ``(char_start, char_end, token_start, token_end)`` windows."""
    if window_tokens <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if not tokens:
        return ()
    starts = list(range(0, len(tokens), stride))
    windows: list[tuple[int, int, int, int]] = []
    for token_start in starts:
        token_end = min(token_start + window_tokens, len(tokens))
        windows.append(
            (tokens[token_start].start, tokens[token_end - 1].end, token_start, token_end)
        )
        if token_end == len(tokens):
            break
    return tuple(windows)


def clip_region_around_protected(
    region: TextRegion, protected_spans: Sequence[Span]
) -> tuple[TextRegion, ...]:
    """Subtract protected ranges instead of letting an edit cut through them."""
    intervals = [(region.start, region.end)]
    for span in sorted(protected_spans, key=lambda item: (item.start, item.end)):
        next_intervals: list[tuple[int, int]] = []
        for start, end in intervals:
            if span.end <= start or span.start >= end:
                next_intervals.append((start, end))
                continue
            if start < span.start:
                next_intervals.append((start, span.start))
            if span.end < end:
                next_intervals.append((span.end, end))
        intervals = next_intervals
    return tuple(
        region.model_copy(
            update={
                "start": start,
                "end": end,
                "token_start": None,
                "token_end": None,
                "rationale": f"{region.rationale}; clipped around protected content".strip("; "),
            }
        )
        for start, end in intervals
        if end > start
    )


def merge_overlapping_regions(
    regions: Iterable[TextRegion], *, overlap_fraction: float = 0.5
) -> tuple[TextRegion, ...]:
    """Merge regions whose intersection covers a substantial smaller fraction."""
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1]")
    merged: list[TextRegion] = []
    for region in sorted(regions, key=lambda item: (item.start, item.end)):
        if not merged:
            merged.append(region)
            continue
        previous = merged[-1]
        intersection = max(0, min(previous.end, region.end) - max(previous.start, region.start))
        smaller = min(previous.end - previous.start, region.end - region.start)
        if smaller and intersection / smaller >= overlap_fraction:
            merged[-1] = TextRegion(
                start=min(previous.start, region.start),
                end=max(previous.end, region.end),
                risk=max(previous.risk, region.risk),
                mode=previous.mode,
                token_start=(
                    min(previous.token_start, region.token_start)
                    if previous.token_start is not None and region.token_start is not None
                    else None
                ),
                token_end=(
                    max(previous.token_end, region.token_end)
                    if previous.token_end is not None and region.token_end is not None
                    else None
                ),
                rationale=f"merged overlapping {previous.mode} regions",
            )
        else:
            merged.append(region)
    return tuple(sorted(merged, key=lambda item: (-item.risk, item.start, item.end)))
