"""Character alignment, protected clipping, merging, and query accounting."""

from __future__ import annotations

from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.spans import Span
from unmark.detectors.localization import (
    TextRegion,
    align_regex_tokens,
    clip_region_around_protected,
    merge_overlapping_regions,
    token_windows,
    validate_alignments,
)
from unmark.detectors.protocols import DetectorScore
from unmark.inspect.ingest import build_document
from unmark.strategies.targeted.config import TargetedSearchConfig
from unmark.strategies.targeted.localize import window_score_regions


class CountingDetector:
    id = "counting"
    version = "1"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, text: str) -> DetectorScore:
        self.calls += 1
        return DetectorScore(
            detector_id=self.id,
            detector_version=self.version,
            evidence_kind="research",
            score=float(len(text.split())),
            token_count=len(text.split()),
        )


def test_regex_alignment_uses_exact_character_offsets() -> None:
    text = "naïve  東京\nemoji👩‍💻"
    tokens = align_regex_tokens(text)
    validate_alignments(text, tokens)
    assert [text[token.start : token.end] for token in tokens] == [
        "naïve",
        "東京",
        "emoji👩‍💻",
    ]


def test_token_windows_end_on_aligned_characters() -> None:
    text = "one  two three four five"
    tokens = align_regex_tokens(text)
    windows = token_windows(tokens, window_tokens=3, stride=2)
    assert [text[start:end] for start, end, _, _ in windows] == [
        "one  two three",
        "three four five",
    ]


def test_protected_span_is_subtracted_from_region() -> None:
    text = "alpha LOCK omega"
    protected = Span(start=6, end=10, kind="user_lock", value="LOCK")
    region = TextRegion(start=0, end=len(text), risk=1, mode="test")
    clipped = clip_region_around_protected(region, (protected,))
    assert [(item.start, item.end) for item in clipped] == [(0, 6), (10, 16)]
    assert all(not protected.overlaps(item.start, item.end) for item in clipped)


def test_overlapping_windows_merge_deterministically() -> None:
    merged = merge_overlapping_regions(
        (
            TextRegion(start=0, end=10, risk=1, mode="window"),
            TextRegion(start=5, end=15, risk=3, mode="window"),
            TextRegion(start=30, end=40, risk=2, mode="window"),
        )
    )
    assert [(item.start, item.end, item.risk) for item in merged] == [
        (0, 15, 3),
        (30, 40, 2),
    ]


def test_window_scan_accounts_for_each_detector_query() -> None:
    text = " ".join(f"token{index}" for index in range(72))
    document = build_document(text, "text/plain")
    detector = CountingDetector()
    budget = BudgetAccount(RunBudget(max_detector_queries=4))
    config = TargetedSearchConfig(
        window_tokens=24,
        window_stride=12,
        top_k_regions=5,
        run_budget=RunBudget(
            max_detector_queries=4,
            max_model_calls=0,
            max_candidates=8,
            max_rounds=3,
        ),
    )
    result = window_score_regions(document, detector, budget, config)
    assert detector.calls == 4
    assert budget.usage().detector_queries == 4
    assert "budget" in result.warnings[-1]
