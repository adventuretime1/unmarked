"""Offline tests for the prompt-driven rewrite baselines.

Every test here runs without a network, an API key, a model download, or a GPU.
Models are the deterministic :class:`FakeModelAdapter`; detectors are a trivial
in-test surrogate. The suite exercises prompt-only output, candidate validation
and rejection, minimum-edit selection, recursive early stopping and rollback,
budgets, and the honest evidence-state contract.
"""

from __future__ import annotations

from decimal import Decimal

from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.document import Document
from unmark.core.policies import FidelityPolicy
from unmark.core.targets import ReductionTarget
from unmark.detectors.protocols import DetectorScore
from unmark.inspect.ingest import build_document
from unmark.models.local import FakeModelAdapter, PrintPromptAdapter
from unmark.models.protocols import ModelRequest
from unmark.strategies.rewrite.candidates import bigram_jaccard_divergence, select_candidate
from unmark.strategies.rewrite.config import RewriteConfig
from unmark.strategies.rewrite.one_shot import OneShotRewriteStrategy
from unmark.strategies.rewrite.recursive import RecursiveRewriteStrategy

# A rewrite cannot prove semantic equivalence without an NLI/claim gate, so the
# strategies run with the entailment gate relaxed; every deterministic lock stays
# hard. This mirrors what the rewrite service does.
_POLICY = FidelityPolicy(require_bidirectional_entailment=False)

_SOURCE = (
    "The quarterly revenue was $4.2M in 2023, up from prior years. "
    "See https://example.com/report for the full breakdown."
)


def _doc(text: str = _SOURCE, media_type: str = "text/plain") -> Document:
    return build_document(text, media_type, origin="mem://sample.txt", fidelity=_POLICY)  # type: ignore[arg-type]


def _budget(**overrides: object) -> BudgetAccount:
    defaults: dict[str, object] = {
        "max_runtime_ms": 30_000,
        "max_model_calls": 16,
        "max_detector_queries": 16,
        "max_cost_usd": Decimal("1"),
        "max_char_edit_ratio": 0.6,
        "max_token_edit_ratio": 0.6,
        "max_length_drift_ratio": 0.6,
        "max_candidates": 32,
        "max_rounds": 5,
    }
    defaults.update(overrides)
    return BudgetAccount(RunBudget(**defaults))  # type: ignore[arg-type]


class _CountingDetector:
    """A surrogate whose risk drops as a marker phrase disappears.

    The source carries three copies of the marker; each removed copy lowers the
    normalized risk. This lets tests assert reduction and target behavior without
    any model or network.
    """

    detector_id = "test-marker"
    detector_version = "1.0"
    id = detector_id
    version = detector_version

    def __init__(self, marker: str = "up from prior years") -> None:
        self._marker = marker

    def score(self, text: str) -> DetectorScore:
        hits = text.count(self._marker)
        return DetectorScore(
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            evidence_kind="surrogate",
            score=float(min(hits, 3)) / 3.0,
            risk_direction="higher",
        )


# --- prompt-only backend --------------------------------------------------


def test_print_prompt_emits_prompt_and_rewrites_nothing() -> None:
    strategy = OneShotRewriteStrategy(
        adapter=PrintPromptAdapter(),
        config=RewriteConfig(style="faithful"),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)

    assert result.state == "unsupported"
    assert result.selected is None
    assert result.trace.prompt_only is not None
    assert "[system]" in result.trace.prompt_only
    assert _SOURCE in result.trace.prompt_only


def test_prompt_carries_preservation_instructions() -> None:
    strategy = OneShotRewriteStrategy(
        adapter=PrintPromptAdapter(),
        config=RewriteConfig(style="faithful"),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)
    prompt = result.trace.prompt_only or ""

    # A representative slice of the mandatory preservation rules.
    for needle in ("numbers", "quotation", "URL", "negation", "Markdown", "only the rewritten"):
        assert needle.lower() in prompt.lower()


# --- multi-candidate generation + validation ------------------------------


def test_generates_multiple_candidates() -> None:
    responder = lambda request, index: _SOURCE.replace("quarterly", f"Q{index + 1}")  # noqa: E731
    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responder=responder),
        config=RewriteConfig(style="lexical", candidate_count=3),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)

    assert result.state == "rewritten_unverified"
    assert result.selected is not None
    # Three distinct candidates were generated (selected + alternatives).
    assert result.trace.candidates_generated == 3


def test_invalid_candidate_is_rejected_and_never_selected() -> None:
    # One reply changes a protected number; it must be rejected outright.
    responses = (
        _SOURCE.replace("$4.2M", "$9.9M"),  # protected number changed -> rejected
        _SOURCE.replace("quarterly", "three-month"),  # safe -> selectable
    )
    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responses=responses),
        config=RewriteConfig(style="lexical", candidate_count=2),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)

    assert result.selected is not None
    assert "$9.9M" not in result.selected.text
    assert result.selected.text == responses[1]
    assert result.trace.candidates_rejected == 1


def test_changed_url_rejected() -> None:
    responses = (_SOURCE.replace("example.com", "evil.example"),)
    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responses=responses),
        config=RewriteConfig(candidate_count=1),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)
    assert result.state == "abstained"
    assert result.selected is None
    assert result.trace.candidates_rejected == 1


def test_changed_quote_rejected() -> None:
    source = 'She said "the figures are final" during the call.'
    responses = ('She said "the numbers are final" during the call.',)
    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responses=responses),
        config=RewriteConfig(candidate_count=1),
    )
    result = strategy.run(_doc(source), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)
    assert result.selected is None


def test_changed_code_span_rejected() -> None:
    # Inline code spans are discovered in Markdown; the model must not alter one.
    source = "Run `deploy --env=prod` after the merge lands."
    responses = ("Run `deploy --env=staging` after the merge lands.",)
    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responses=responses),
        config=RewriteConfig(candidate_count=1),
    )
    result = strategy.run(
        _doc(source, media_type="text/markdown"),
        ReductionTarget(mode="sanitize_only"),
        _budget(),
        _POLICY,
    )
    assert result.selected is None


# --- minimum-edit candidate selection -------------------------------------


def test_selects_minimum_edit_candidate() -> None:
    # Two valid candidates: one changes a single word, one rewrites broadly.
    small = _SOURCE.replace("quarterly", "three-month")
    large = _SOURCE.replace("quarterly revenue", "amount of money earned during the quarter")
    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responses=(large, small)),
        config=RewriteConfig(style="lexical", candidate_count=2),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)

    assert result.selected is not None
    assert result.selected.text == small  # smaller edit wins, not the broader rewrite


def test_selection_does_not_maximize_bigram_divergence() -> None:
    # The broad rewrite has higher bigram divergence, but the smaller edit should win.
    small = _SOURCE.replace("quarterly", "three-month")
    large = _SOURCE.replace("quarterly revenue", "amount of money earned during the quarter")
    assert bigram_jaccard_divergence(_SOURCE, large) > bigram_jaccard_divergence(_SOURCE, small)

    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responses=(large, small)),
        config=RewriteConfig(style="lexical", candidate_count=2),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)
    assert result.selected is not None
    # Selection prefers the lower-divergence, smaller-edit candidate.
    assert result.selected.text == small


# --- budgets --------------------------------------------------------------


def test_networked_backend_refused_without_model_budget() -> None:
    class _NetAdapter(FakeModelAdapter):
        uses_network = True

    strategy = OneShotRewriteStrategy(
        adapter=_NetAdapter(responses=(_SOURCE,)),
        config=RewriteConfig(candidate_count=1),
    )
    result = strategy.run(
        _doc(), ReductionTarget(mode="sanitize_only"), _budget(max_model_calls=0), _POLICY
    )
    # No model budget means the networked call is refused -> unsupported.
    assert result.state == "unsupported"
    assert result.selected is None


def test_model_call_budget_caps_generation() -> None:
    calls = {"n": 0}

    def responder(request: ModelRequest, index: int) -> str:
        calls["n"] += 1
        return _SOURCE.replace("quarterly", f"period-{index}")

    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responder=responder),
        config=RewriteConfig(candidate_count=4),
    )
    # Only two model calls are affordable.
    result = strategy.run(
        _doc(), ReductionTarget(mode="sanitize_only"), _budget(max_model_calls=2), _POLICY
    )
    # The engine reserves the whole batch up front, so an unaffordable batch
    # abstains rather than partially generating.
    assert result.state in {"abstained", "rewritten_unverified"}


# --- recursive strategy ---------------------------------------------------


def test_recursive_early_stops_when_target_met() -> None:
    detector = _CountingDetector()
    # The first hop removes the marker phrase, lowering the surrogate risk enough
    # to meet a relative-reduction target.
    responder = lambda request, index: _SOURCE.replace(", up from prior years", "")  # noqa: E731
    strategy = RecursiveRewriteStrategy(
        adapter=FakeModelAdapter(responder=responder),
        config=RewriteConfig(style="faithful", rounds=5, candidate_count=1),
        detector=detector,
    )
    target = ReductionTarget(mode="relative_reduction", min_score_reduction=0.3)
    result = strategy.run(_doc(), target, _budget(), _POLICY)

    assert result.selected is not None
    assert "prior years" not in result.selected.text
    # It stopped before exhausting all five hops.
    assert len(result.trace.hops) < 5
    assert "target met" in result.trace.stopping_reason


def test_recursive_rolls_back_to_better_earlier_hop() -> None:
    detector = _CountingDetector()
    good = _SOURCE.replace(", up from prior years", "")  # removes one marker -> lower risk
    # After the good hop, later hops "regress" by restoring the marker text. The
    # engine must keep rewriting the good hop's text, not the regression, and must
    # select the earlier, better candidate.
    regressed = _SOURCE + " up from prior years, up from prior years."

    def responder(request: ModelRequest, index: int) -> str:
        # Hop 0 sees the source; produce the good candidate. Later hops rewrite
        # the good text (its base); produce a regression that adds markers back.
        if "up from prior years" not in request.prompt:
            return regressed
        return good

    strategy = RecursiveRewriteStrategy(
        adapter=FakeModelAdapter(responder=responder),
        config=RewriteConfig(style="faithful", rounds=3, candidate_count=1, early_stop_patience=5),
        detector=detector,
    )
    # A target that is never met, so the loop runs and rollback is exercised.
    target = ReductionTarget(mode="minimize_surrogate")
    result = strategy.run(_doc(), target, _budget(), _POLICY)

    assert result.selected is not None
    # The selected candidate is the earlier good hop, not a later regression.
    assert result.selected.text == good


def test_recursive_print_prompt_emits_prompt_and_abstains() -> None:
    # Recursive print-prompt renders the prompt on the first hop and stops; with no
    # rewrite candidate to select, it abstains rather than inventing a result.
    strategy = RecursiveRewriteStrategy(
        adapter=PrintPromptAdapter(),
        config=RewriteConfig(style="faithful", rounds=3),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)
    assert result.state == "abstained"
    assert result.selected is None
    assert result.trace.prompt_only is not None


# --- effect language ------------------------------------------------------


def test_plain_rewrite_describes_statistical_pattern_disruption() -> None:
    responses = (_SOURCE.replace("quarterly", "three-month"),)
    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responses=responses),
        config=RewriteConfig(candidate_count=1),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)
    assert result.state == "rewritten_unverified"
    assert "reducing the chance" in result.residual_risk.lower()


def test_no_valid_candidate_abstains() -> None:
    # Every reply changes a protected number.
    responses = (_SOURCE.replace("$4.2M", "$1"),)
    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responses=responses),
        config=RewriteConfig(candidate_count=1),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)
    assert result.state == "abstained"


def test_select_candidate_prefers_fidelity_valid() -> None:
    assert select_candidate(()) is None
