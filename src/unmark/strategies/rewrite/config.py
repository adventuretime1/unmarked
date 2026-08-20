"""Configuration for the simple rewrite strategies."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from unmark.core.spans import StrictModel
from unmark.strategies.rewrite.prompts import (
    REWRITE_STRENGTHS,
    REWRITE_STYLES,
    RewriteStrength,
    RewriteStyle,
)


class RewriteConfig(StrictModel):
    """Knobs shared by the one-shot and recursive rewrite strategies.

    Budgets (model calls, cost, runtime, edit ratios, length drift) live on the
    :class:`~unmark.core.budgets.RunBudget` and are enforced there; this holds only
    the rewrite-specific choices.
    """

    style: RewriteStyle = "faithful"
    strength: RewriteStrength = "medium"
    candidate_count: int = Field(default=1, ge=1, le=16)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    target_length_ratio: float | None = Field(default=None, gt=0.0, le=4.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    seed: int = 0

    #: Description from a voice profile, rendered into the prompt. Empty means
    #: no voice was requested and the prompt is unchanged.
    voice: str = ""

    # Recursive-only.
    rounds: int = Field(default=1, ge=1, le=5)
    style_schedule: tuple[RewriteStyle, ...] = ()
    early_stop_patience: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_schedule(self) -> Self:
        for style in self.style_schedule:
            if style not in REWRITE_STYLES:
                msg = f"unknown rewrite style in schedule: {style!r}"
                raise ValueError(msg)
        if self.strength not in REWRITE_STRENGTHS:
            msg = f"unknown rewrite strength: {self.strength!r}"
            raise ValueError(msg)
        return self

    def style_for_round(self, round_index: int) -> RewriteStyle:
        """Style for a zero-based hop, falling back to the default style."""
        if self.style_schedule:
            return self.style_schedule[round_index % len(self.style_schedule)]
        return self.style
