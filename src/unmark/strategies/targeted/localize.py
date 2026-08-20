"""Contribution, window, delta, PLL, and automatic localization."""

from __future__ import annotations

import time
from dataclasses import dataclass

from unmark.core.budgets import BudgetAccount
from unmark.core.document import Document
from unmark.core.errors import BudgetExhaustedError, DependencyUnavailableError
from unmark.detectors.localization import (
    TextRegion,
    align_regex_tokens,
    clip_region_around_protected,
    merge_overlapping_regions,
    token_windows,
)
from unmark.detectors.protocols import Detector, DetectorScore
from unmark.detectors.surrogate.pll import CachedPllScorer
from unmark.strategies.targeted.config import LocalizerMode, TargetedSearchConfig


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    regions: tuple[TextRegion, ...]
    selected_mode: str
    reason: str
    warnings: tuple[str, ...] = ()


def _score(
    detector: Detector,
    text: str,
    budget: BudgetAccount,
    deadline: float | None = None,
) -> DetectorScore:
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhaustedError("runtime budget exhausted before detector query")
    with budget.reserve("detector_queries") as lease:
        result = detector.score(text)
        lease.settle(1)
    if deadline is not None and time.monotonic() > deadline:
        raise BudgetExhaustedError("runtime budget exhausted during detector query")
    return result


def _editable_ranked(
    document: Document, regions: tuple[TextRegion, ...], limit: int
) -> tuple[TextRegion, ...]:
    clipped: list[TextRegion] = []
    for region in regions:
        clipped.extend(clip_region_around_protected(region, document.protected_spans))
    useful = [
        region for region in clipped if document.source_text[region.start : region.end].strip()
    ]
    return tuple(sorted(useful, key=lambda item: (-item.risk, item.start, item.end))[:limit])


def contribution_regions(
    document: Document, baseline: DetectorScore, config: TargetedSearchConfig
) -> LocalizationResult:
    ranked = sorted(
        baseline.contributions,
        key=lambda contribution: (-contribution.score, contribution.start, contribution.end),
    )[: config.top_k_tokens]
    regions = tuple(
        TextRegion(
            start=item.start,
            end=item.end,
            risk=item.score,
            mode="detector_contributions",
            rationale=f"high {baseline.detector_id} token contribution",
        )
        for item in ranked
        if item.end <= len(document.source_text) and item.end > item.start
    )
    return LocalizationResult(
        regions=_editable_ranked(document, regions, config.top_k_regions),
        selected_mode="detector_contributions",
        reason="the detector supplied exact character-aligned contributions",
    )


def window_score_regions(
    document: Document,
    detector: Detector,
    budget: BudgetAccount,
    config: TargetedSearchConfig,
    deadline: float | None = None,
) -> LocalizationResult:
    tokens = align_regex_tokens(document.source_text)
    regions: list[TextRegion] = []
    warnings: list[str] = []
    for start, end, token_start, token_end in token_windows(
        tokens, config.window_tokens, config.window_stride
    ):
        try:
            score = _score(detector, document.source_text[start:end], budget, deadline)
        except BudgetExhaustedError:
            warnings.append("detector-query budget stopped the window scan")
            break
        token_count = score.token_count or token_end - token_start
        risk = score.risk / max(token_count, 1)
        regions.append(
            TextRegion(
                start=start,
                end=end,
                risk=risk,
                mode="window_score",
                token_start=token_start,
                token_end=token_end,
                rationale="length-normalized detector window risk",
            )
        )
    merged = merge_overlapping_regions(regions)
    return LocalizationResult(
        regions=_editable_ranked(document, merged, config.top_k_regions),
        selected_mode="window_score",
        reason="the detector was available for bounded overlapping-window scoring",
        warnings=tuple(warnings),
    )


def window_delta_regions(
    document: Document,
    detector: Detector,
    baseline: DetectorScore,
    budget: BudgetAccount,
    config: TargetedSearchConfig,
    deadline: float | None = None,
) -> LocalizationResult:
    """Estimate contribution using deliberately expensive black-box masking probes."""
    tokens = align_regex_tokens(document.source_text)
    regions: list[TextRegion] = []
    warnings: list[str] = ["window_delta uses one detector query per masking probe"]
    for start, end, token_start, token_end in token_windows(
        tokens, config.window_tokens, config.window_stride
    ):
        probe = document.source_text[:start] + " " + document.source_text[end:]
        try:
            score = _score(detector, probe, budget, deadline)
        except BudgetExhaustedError:
            warnings.append("detector-query budget stopped the delta scan")
            break
        regions.append(
            TextRegion(
                start=start,
                end=end,
                risk=max(0.0, baseline.risk - score.risk),
                mode="window_delta",
                token_start=token_start,
                token_end=token_end,
                rationale="baseline risk minus bounded masking-probe risk",
            )
        )
    merged = merge_overlapping_regions(regions)
    return LocalizationResult(
        regions=_editable_ranked(document, merged, config.top_k_regions),
        selected_mode="window_delta",
        reason="explicit black-box score deltas were requested",
        warnings=tuple(warnings),
    )


def pll_regions(
    document: Document,
    pll: CachedPllScorer,
    budget: BudgetAccount,
    config: TargetedSearchConfig,
    deadline: float | None = None,
) -> LocalizationResult:
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhaustedError("runtime budget exhausted before PLL model call")
    with budget.reserve("model_calls") as lease:
        scores = pll.score_tokens(document.source_text)
        lease.settle(1)
    if deadline is not None and time.monotonic() > deadline:
        raise BudgetExhaustedError("runtime budget exhausted during PLL model call")
    ranked = sorted(scores, key=lambda item: (-item.suspicion, item.token.start))[
        : config.top_k_tokens
    ]
    regions = tuple(
        TextRegion(
            start=item.token.start,
            end=item.token.end,
            risk=item.suspicion,
            mode="pll",
            token_start=item.token.index,
            token_end=item.token.index + 1,
            rationale="low pseudo-log-likelihood; surrogate localization only",
        )
        for item in ranked
    )
    return LocalizationResult(
        regions=_editable_ranked(document, regions, config.top_k_regions),
        selected_mode="pll",
        reason="no stronger detector localization signal was available",
        warnings=("PLL is a surrogate localizer, not watermark verification",),
    )


class TargetedLocalizer:
    """Capability-aware localizer with an explainable automatic mode."""

    def __init__(
        self,
        *,
        config: TargetedSearchConfig,
        budget: BudgetAccount,
        detector: Detector | None = None,
        pll: CachedPllScorer | None = None,
        deadline: float | None = None,
    ) -> None:
        self.config = config
        self.budget = budget
        self.detector = detector
        self.pll = pll
        self.deadline = deadline

    def localize(self, document: Document, baseline: DetectorScore | None) -> LocalizationResult:
        mode: LocalizerMode = self.config.localizer
        if mode in {"auto", "detector_contributions"} and baseline is not None:
            if baseline.contributions:
                return contribution_regions(document, baseline, self.config)
            if mode == "detector_contributions":
                raise DependencyUnavailableError("detector supplied no token contributions")
        if mode in {"auto", "window_score"} and self.detector is not None:
            return window_score_regions(
                document, self.detector, self.budget, self.config, self.deadline
            )
        if mode == "window_delta":
            if self.detector is None or baseline is None:
                raise DependencyUnavailableError("window_delta requires a detector baseline")
            return window_delta_regions(
                document,
                self.detector,
                baseline,
                self.budget,
                self.config,
                self.deadline,
            )
        if mode in {"auto", "pll"} and self.pll is not None:
            return pll_regions(document, self.pll, self.budget, self.config, self.deadline)
        raise DependencyUnavailableError(f"localizer {mode!r} has no available adapter")
