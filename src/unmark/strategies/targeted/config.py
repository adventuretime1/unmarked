"""Strict configuration for targeted constrained search."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from unmark.core.budgets import RunBudget
from unmark.core.spans import StrictModel

LocalizerMode = Literal["auto", "detector_contributions", "window_score", "window_delta", "pll"]
OperatorName = Literal[
    "exact_phrase",
    "short_span_rewrite",
    "sentence_rewrite",
    "sentence_split",
    "sentence_merge",
    "clause_reorder",
    "masked_span_infill",
    "punctuation",
    "connective",
]

DEFAULT_OPERATORS: tuple[OperatorName, ...] = (
    "exact_phrase",
    "short_span_rewrite",
    "sentence_rewrite",
    "sentence_split",
    "sentence_merge",
    "clause_reorder",
    "masked_span_infill",
    "punctuation",
    "connective",
)


class TargetedSearchConfig(StrictModel):
    """Safe production configuration; unknown keys are rejected by ``StrictModel``."""

    schema_version: Literal["1"] = "1"
    localizer: LocalizerMode = "auto"
    top_k_regions: int = Field(default=5, ge=1, le=20)
    top_k_tokens: int = Field(default=20, ge=1, le=100)
    window_tokens: int = Field(default=64, ge=24, le=256)
    window_stride: int = Field(default=16, ge=4, le=128)
    beam_width: int = Field(default=8, ge=1, le=32)
    candidates_per_node: int = Field(default=4, ge=1, le=16)
    max_search_depth: int = Field(default=3, ge=1, le=8)
    minimum_score_improvement: float = Field(default=0.0, ge=0.0)
    stagnation_rounds: int = Field(default=2, ge=1, le=5)
    operator_allowlist: tuple[OperatorName, ...] = DEFAULT_OPERATORS
    proposal_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 2025
    diversity_similarity_ceiling: float = Field(default=0.92, ge=0.0, le=1.0)
    research_mode: bool = False
    run_budget: RunBudget = RunBudget(
        max_runtime_ms=30_000,
        max_model_calls=12,
        max_detector_queries=32,
        max_candidates=32,
        max_rounds=3,
    )

    @model_validator(mode="after")
    def _safe_combinations(self) -> Self:
        if self.window_stride > self.window_tokens:
            raise ValueError("window_stride must not exceed window_tokens")
        if self.max_search_depth > self.run_budget.max_rounds:
            raise ValueError("max_search_depth must not exceed run_budget.max_rounds")
        if self.proposal_temperature > 1.0 and not self.research_mode:
            raise ValueError("proposal_temperature above 1 requires research_mode")
        if len(set(self.operator_allowlist)) != len(self.operator_allowlist):
            raise ValueError("operator_allowlist must not contain duplicates")
        return self
