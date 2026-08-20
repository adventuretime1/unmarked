"""Offline end-to-end targeted constrained-search demonstrations."""

from __future__ import annotations

from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.policies import FidelityPolicy
from unmark.core.targets import ReductionTarget
from unmark.detectors.localization import TokenAlignment
from unmark.detectors.protocols import DetectorScore, ScoreContribution
from unmark.detectors.surrogate.pll import CachedPllScorer, PllTokenScore
from unmark.inspect.ingest import build_document
from unmark.strategies.targeted.beam import MemoryCheckpointStore
from unmark.strategies.targeted.config import TargetedSearchConfig
from unmark.strategies.targeted.propose import (
    DeterministicProposalProvider,
    ProposedReplacement,
    StructuredModelProposalProvider,
)
from unmark.strategies.targeted.strategy import TargetedSearchStrategy


class PhraseDetector:
    id = "synthetic:phrase"
    version = "1"

    def __init__(self, *, kind: str = "research") -> None:
        self.kind = kind
        self.calls: list[str] = []

    def score(self, text: str) -> DetectorScore:
        self.calls.append(text)
        phrase = "marked phrase"
        start = text.find(phrase)
        score = 10.0 if start >= 0 else 1.0
        contributions = (
            (ScoreContribution(start=start, end=start + len(phrase), score=10, token=phrase),)
            if start >= 0
            else ()
        )
        return DetectorScore(
            detector_id=self.id,
            detector_version=self.version,
            evidence_kind=self.kind,  # type: ignore[arg-type]
            score=score,
            threshold=2,
            token_count=len(text.split()),
            contributions=contributions,
        )


def config() -> TargetedSearchConfig:
    return TargetedSearchConfig(
        beam_width=4,
        candidates_per_node=4,
        max_search_depth=2,
        minimum_score_improvement=0.1,
        run_budget=RunBudget(
            max_runtime_ms=10_000,
            max_model_calls=0,
            max_detector_queries=20,
            max_candidates=12,
            max_rounds=2,
            max_char_edit_ratio=0.8,
            max_token_edit_ratio=0.8,
            max_length_drift_ratio=0.8,
        ),
    )


def test_selects_smallest_success_and_scores_only_after_fidelity() -> None:
    text = "This marked phrase appears in a deliberately long sentence for stable ratios."
    document = build_document(text, "text/plain")
    detector = PhraseDetector()
    provider = DeterministicProposalProvider(
        {
            "marked phrase": (
                "plain phrase",
                "an extensively rewritten and needlessly expanded phrase",
            )
        },
        allowed_operators=("exact_phrase",),
    )
    strategy = TargetedSearchStrategy(
        config=config(), proposal_provider=provider, detector=detector
    )
    result = strategy.run(
        document,
        ReductionTarget(mode="verify_below_threshold", threshold=2),
        BudgetAccount(config().run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    assert result.state == "verified_below_threshold"
    assert result.selected is not None
    assert "plain phrase" in result.selected.text
    assert "needlessly" not in result.selected.text
    assert result.trace.candidates_scored == 2
    assert result.trace.stopping_reason.startswith("minimum-edit success")


def test_surrogate_run_reports_surrogate_reduced() -> None:
    text = "A marked phrase is localized."
    document = build_document(text, "text/plain")
    detector = PhraseDetector(kind="surrogate")
    strategy = TargetedSearchStrategy(
        config=config(),
        proposal_provider=DeterministicProposalProvider(
            {"marked phrase": ("plain phrase",)},
            allowed_operators=("exact_phrase",),
        ),
        detector=detector,
    )
    result = strategy.run(
        document,
        ReductionTarget(mode="minimize_surrogate", min_score_reduction=1),
        BudgetAccount(config().run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    assert result.state == "surrogate_reduced"
    assert "surrogate score decreased" in result.residual_risk.lower()


def test_verified_request_with_only_surrogate_is_unsupported_without_edit() -> None:
    text = "A marked phrase is localized."
    document = build_document(text, "text/plain")
    detector = PhraseDetector(kind="surrogate")
    strategy = TargetedSearchStrategy(
        config=config(),
        proposal_provider=DeterministicProposalProvider(
            {"marked phrase": ("plain phrase",)},
            allowed_operators=("exact_phrase",),
        ),
        detector=detector,
    )
    result = strategy.run(
        document,
        ReductionTarget(mode="verify_below_threshold", threshold=2),
        BudgetAccount(config().run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    assert result.state == "unsupported"
    assert result.selected is None
    assert detector.calls == [text]


def test_checkpoint_is_saved_with_fixed_seed_and_round_trips() -> None:
    text = "A marked phrase is localized."
    document = build_document(text, "text/plain")
    store = MemoryCheckpointStore()
    strategy = TargetedSearchStrategy(
        config=config(),
        proposal_provider=DeterministicProposalProvider(
            {"marked phrase": ("plain phrase",)},
            allowed_operators=("exact_phrase",),
        ),
        detector=PhraseDetector(),
        checkpoint_store=store,
    )
    result = strategy.run(
        document,
        ReductionTarget(mode="relative_reduction", min_score_reduction=1),
        BudgetAccount(config().run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    assert result.state == "verified_reduction_only"
    assert store.load() is not None
    assert result.checkpoint is not None
    assert result.checkpoint.current_depth == 1


def test_duplicate_candidates_are_removed_before_detector_scoring() -> None:
    text = "A marked phrase is localized."
    document = build_document(text, "text/plain")
    detector = PhraseDetector()
    provider = DeterministicProposalProvider(
        {
            "marked phrase": (
                ProposedReplacement(
                    replacement="plain phrase",
                    operator="exact_phrase",
                    reason="first route",
                    provider_id="test",
                ),
                ProposedReplacement(
                    replacement="plain phrase",
                    operator="short_span_rewrite",
                    reason="second route",
                    provider_id="test",
                ),
            )
        },
        allowed_operators=("exact_phrase", "short_span_rewrite"),
    )
    strategy = TargetedSearchStrategy(
        config=config(), proposal_provider=provider, detector=detector
    )
    result = strategy.run(
        document,
        ReductionTarget(mode="verify_below_threshold", threshold=2),
        BudgetAccount(config().run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    assert result.trace.proposals_generated == 2
    assert result.trace.candidates_deduplicated == 1
    assert result.trace.candidates_scored == 1
    assert len(detector.calls) == 2  # baseline plus one unique survivor


class TwoPhraseDetector:
    id = "synthetic:two-phrase"
    version = "1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def score(self, text: str) -> DetectorScore:
        self.calls.append(text)
        contributions = []
        score = 0.0
        for phrase, risk in (("marked alpha", 6.0), ("marked beta", 4.0)):
            start = text.find(phrase)
            if start >= 0:
                score += risk
                contributions.append(
                    ScoreContribution(
                        start=start,
                        end=start + len(phrase),
                        score=risk,
                        token=phrase,
                    )
                )
        return DetectorScore(
            detector_id=self.id,
            detector_version=self.version,
            evidence_kind="research",
            score=score,
            threshold=1,
            token_count=len(text.split()),
            contributions=tuple(contributions),
        )


class InterruptOnceStore(MemoryCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self.interrupt = True

    def save(self, checkpoint: dict[str, object]) -> None:
        super().save(checkpoint)
        if self.interrupt:
            self.interrupt = False
            raise RuntimeError("simulated interruption")


def test_resume_uses_saved_baseline_localization_budget_and_rng() -> None:
    text = "The marked alpha and marked beta signals remain."
    document = build_document(text, "text/plain")
    store = InterruptOnceStore()
    provider = DeterministicProposalProvider(
        {
            "marked alpha": ("plain alpha",),
            "marked beta": ("plain beta",),
        },
        allowed_operators=("exact_phrase",),
    )
    first_detector = TwoPhraseDetector()
    strategy = TargetedSearchStrategy(
        config=config(),
        proposal_provider=provider,
        detector=first_detector,
        checkpoint_store=store,
    )
    try:
        strategy.run(
            document,
            ReductionTarget(mode="verify_below_threshold", threshold=1),
            BudgetAccount(config().run_budget),
            FidelityPolicy(require_bidirectional_entailment=False),
        )
    except RuntimeError as error:
        assert str(error) == "simulated interruption"
    else:
        raise AssertionError("the simulated interruption did not fire")

    resumed_detector = TwoPhraseDetector()
    resumed = TargetedSearchStrategy(
        config=config(),
        proposal_provider=provider,
        detector=resumed_detector,
        checkpoint_store=store,
    ).run(
        document,
        ReductionTarget(mode="verify_below_threshold", threshold=1),
        BudgetAccount(config().run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
        resume=True,
    )
    assert resumed.state == "verified_below_threshold"
    assert resumed.selected is not None
    assert "plain alpha" in resumed.selected.text
    assert "plain beta" in resumed.selected.text
    assert len(resumed_detector.calls) == 1  # no repeated baseline/localization query
    assert resumed.usage.detector_queries == 3


def test_zero_runtime_budget_makes_no_external_call() -> None:
    text = "A marked phrase is localized."
    document = build_document(text, "text/plain")
    detector = PhraseDetector()
    zero_runtime_config = TargetedSearchConfig(
        run_budget=RunBudget(
            max_runtime_ms=0,
            max_detector_queries=10,
            max_candidates=4,
            max_rounds=3,
        )
    )
    result = TargetedSearchStrategy(
        config=zero_runtime_config,
        proposal_provider=DeterministicProposalProvider(
            {"marked phrase": ("plain phrase",)},
            allowed_operators=("exact_phrase",),
        ),
        detector=detector,
    ).run(
        document,
        ReductionTarget(mode="verify_below_threshold", threshold=2),
        BudgetAccount(zero_runtime_config.run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    assert result.state == "abstained"
    assert detector.calls == []


def test_missing_required_nli_is_unsupported_before_detector_call() -> None:
    text = "A marked phrase is localized."
    document = build_document(text, "text/plain")
    detector = PhraseDetector()
    result = TargetedSearchStrategy(
        config=config(),
        proposal_provider=DeterministicProposalProvider(
            {"marked phrase": ("plain phrase",)},
            allowed_operators=("exact_phrase",),
        ),
        detector=detector,
    ).run(
        document,
        ReductionTarget(mode="verify_below_threshold", threshold=2),
        BudgetAccount(config().run_budget),
        FidelityPolicy(),
    )
    assert result.state == "unsupported"
    assert "fidelity capability" in result.trace.stopping_reason
    assert detector.calls == []


class OneTokenPll:
    model_id = "synthetic-pll"
    model_revision = "1"
    tokenizer_revision = "1"
    config_id = "one-token"

    def score_tokens(self, text: str) -> tuple[PllTokenScore, ...]:
        start = text.index("marked")
        return (
            PllTokenScore(
                token=TokenAlignment(
                    index=0,
                    start=start,
                    end=start + len("marked"),
                    text="marked",
                ),
                log_likelihood=-8,
            ),
        )


def test_detector_blind_rewrite_is_labeled_unverified() -> None:
    text = "A marked token is localized by a surrogate heuristic."
    document = build_document(text, "text/plain")
    blind_config = TargetedSearchConfig(
        max_search_depth=1,
        run_budget=RunBudget(
            max_runtime_ms=10_000,
            max_model_calls=1,
            max_detector_queries=0,
            max_candidates=4,
            max_rounds=1,
            max_char_edit_ratio=0.5,
            max_token_edit_ratio=0.5,
            max_length_drift_ratio=0.5,
        ),
    )
    result = TargetedSearchStrategy(
        config=blind_config,
        proposal_provider=DeterministicProposalProvider(
            {"marked": ("plain",)}, allowed_operators=("exact_phrase",)
        ),
        pll=CachedPllScorer(OneTokenPll()),
    ).run(
        document,
        ReductionTarget(mode="stress_test"),
        BudgetAccount(blind_config.run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    assert result.state == "rewritten_unverified"
    assert result.selected is not None
    assert result.selected.score is None
    assert result.usage.model_calls == 1
    assert "without a detector measurement" in result.residual_risk


def test_model_call_budget_stops_before_provider_execution() -> None:
    text = "A marked phrase is localized."
    document = build_document(text, "text/plain")
    calls = 0

    def callback(*args: object) -> tuple[ProposedReplacement, ...]:
        nonlocal calls
        calls += 1
        return ()

    no_model_config = TargetedSearchConfig(
        run_budget=RunBudget(
            max_runtime_ms=10_000,
            max_model_calls=0,
            max_detector_queries=4,
            max_candidates=4,
            max_rounds=3,
        )
    )
    result = TargetedSearchStrategy(
        config=no_model_config,
        proposal_provider=StructuredModelProposalProvider(
            "blocked-model",
            callback,
            allowed_operators=("short_span_rewrite",),
        ),
        detector=PhraseDetector(),
    ).run(
        document,
        ReductionTarget(mode="verify_below_threshold", threshold=2),
        BudgetAccount(no_model_config.run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    assert result.state == "abstained"
    assert calls == 0
    assert "model_calls" in result.trace.stopping_reason


def test_detector_query_budget_preserves_frontier_and_stops_cleanly() -> None:
    text = "A marked phrase is localized."
    document = build_document(text, "text/plain")
    one_query_config = TargetedSearchConfig(
        run_budget=RunBudget(
            max_runtime_ms=10_000,
            max_model_calls=0,
            max_detector_queries=1,
            max_candidates=4,
            max_rounds=3,
            max_char_edit_ratio=0.5,
            max_token_edit_ratio=0.5,
            max_length_drift_ratio=0.5,
        )
    )
    result = TargetedSearchStrategy(
        config=one_query_config,
        proposal_provider=DeterministicProposalProvider(
            {"marked phrase": ("plain phrase",)},
            allowed_operators=("exact_phrase",),
        ),
        detector=PhraseDetector(),
    ).run(
        document,
        ReductionTarget(mode="verify_below_threshold", threshold=2),
        BudgetAccount(one_query_config.run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    assert result.state == "abstained"
    assert result.usage.detector_queries == 1
    assert result.trace.proposals_generated == 1
    assert "detector_queries" in result.trace.stopping_reason
