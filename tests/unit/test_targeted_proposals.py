"""Deterministic and model-backed proposal validation."""

from __future__ import annotations

import json

from unmark.detectors.localization import TextRegion
from unmark.models.local import FakeModelAdapter
from unmark.strategies.targeted.propose import ModelAdapterProposalProvider


def test_model_provider_accepts_only_structured_local_replacement() -> None:
    adapter = FakeModelAdapter(
        responses=(
            json.dumps({"replacement": "plain wording", "operator": "short_span_rewrite"}),
            "Here is the whole rewritten document...",
            json.dumps(
                {
                    "replacement": "bad",
                    "operator": "short_span_rewrite",
                    "explanation": "extra field",
                }
            ),
        )
    )
    provider = ModelAdapterProposalProvider(adapter)
    text = "prefix marked wording suffix"
    start = text.index("marked")
    result = provider.propose(
        source_text=text,
        candidate_text=text,
        region=TextRegion(
            start=start,
            end=start + len("marked wording"),
            risk=1,
            mode="test",
        ),
        protected_spans=(),
        seed=7,
        limit=3,
    )
    assert [proposal.replacement for proposal in result] == ["plain wording"]


def test_model_provider_rejects_disallowed_operator() -> None:
    adapter = FakeModelAdapter(
        responses=(json.dumps({"replacement": "morked", "operator": "misspelling"}),)
    )
    provider = ModelAdapterProposalProvider(adapter)
    result = provider.propose(
        source_text="marked",
        candidate_text="marked",
        region=TextRegion(start=0, end=6, risk=1, mode="test"),
        protected_spans=(),
        seed=1,
        limit=1,
    )
    assert result == ()
