"""Strict config, PLL cache keys, and checkpoint serialization."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from unmark.core.budgets import BudgetUsage, RunBudget
from unmark.detectors.localization import TokenAlignment
from unmark.detectors.surrogate.pll import CachedPllScorer, PllTokenScore
from unmark.strategies.targeted.beam import SearchCandidate, TargetedCheckpoint
from unmark.strategies.targeted.config import TargetedSearchConfig


class FakePllBackend:
    model_id = "fake-mlm"
    model_revision = "model-r1"
    tokenizer_revision = "tokenizer-r1"
    config_id = "fake-config"

    def __init__(self) -> None:
        self.calls = 0

    def score_tokens(self, text: str) -> tuple[PllTokenScore, ...]:
        self.calls += 1
        return (
            PllTokenScore(
                token=TokenAlignment(index=0, start=0, end=len(text), text=text),
                log_likelihood=-2,
            ),
        )


def test_unknown_config_key_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TargetedSearchConfig.model_validate({"beam_width": 2, "mystery": True})


def test_research_temperature_requires_explicit_mode() -> None:
    with pytest.raises(ValidationError, match="research_mode"):
        TargetedSearchConfig(proposal_temperature=1.5)


def test_search_depth_cannot_exceed_shared_round_budget() -> None:
    with pytest.raises(ValidationError, match="max_search_depth"):
        TargetedSearchConfig(
            max_search_depth=3,
            run_budget=RunBudget(max_rounds=2),
        )


def test_pll_cache_uses_versions_and_avoids_hidden_second_call() -> None:
    backend = FakePllBackend()
    scorer = CachedPllScorer(backend)
    first = scorer.score_tokens("token")
    second = scorer.score_tokens("token")
    assert first == second
    assert backend.calls == 1
    original_key = scorer.cache_key("token")
    backend.model_revision = "model-r2"
    assert scorer.cache_key("token") != original_key


def test_checkpoint_json_round_trip_preserves_rng_and_versions() -> None:
    source = SearchCandidate.source("source text")
    checkpoint = TargetedCheckpoint(
        source_hash="abc",
        config_hash="def",
        current_depth=1,
        beam=(source,),
        pareto_archive=(source,),
        successful_archive=(),
        rejected_candidates=(),
        budget_usage=BudgetUsage(detector_queries=2, candidates=1),
        rng_state=[3, [1, 2, 3], None],
        detector_id="synthetic",
        detector_version="v1",
    )
    payload = json.loads(checkpoint.model_dump_json())
    restored = TargetedCheckpoint.model_validate(payload)
    assert restored == checkpoint
    assert restored.detector_version == "v1"
