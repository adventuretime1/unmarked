"""End-to-end rewrite demonstrations with a deterministic fake model.

These run offline and encode the two scenarios the product must get right:

1. One-shot: the first candidate changes a protected fact (rejected), the second
   is valid but unnecessarily large, and the third is a smaller valid rewrite that
   is the one selected.
2. Recursive: an early hop improves the surrogate risk, a later hop regresses, and
   the engine restores the earlier, better result rather than letting the last hop
   win.
"""

from __future__ import annotations

from decimal import Decimal

from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.document import Document
from unmark.core.policies import FidelityPolicy
from unmark.core.targets import ReductionTarget
from unmark.detectors.protocols import DetectorScore
from unmark.inspect.ingest import build_document
from unmark.models.local import FakeModelAdapter
from unmark.models.protocols import ModelRequest
from unmark.strategies.rewrite.config import RewriteConfig
from unmark.strategies.rewrite.one_shot import OneShotRewriteStrategy
from unmark.strategies.rewrite.recursive import RecursiveRewriteStrategy

_POLICY = FidelityPolicy(require_bidirectional_entailment=False)
_SOURCE = (
    "The pilot cut latency by 37% in the March 2024 rollout. "
    "It should be noted that these results were significant. "
    "It should be noted that adoption also grew. "
    "It should be noted that costs stayed flat."
)


def _doc() -> Document:
    return build_document(_SOURCE, "text/plain", origin="mem://demo.txt", fidelity=_POLICY)


def _budget() -> BudgetAccount:
    return BudgetAccount(
        RunBudget(
            max_runtime_ms=30_000,
            max_model_calls=16,
            max_detector_queries=16,
            max_cost_usd=Decimal("1"),
            max_char_edit_ratio=0.6,
            max_token_edit_ratio=0.6,
            max_length_drift_ratio=0.6,
        )
    )


def test_one_shot_rejects_bad_fact_and_picks_smallest_valid() -> None:
    # Candidate order as returned by the fake model:
    #   0: changes the protected percentage 37% -> 40%  (rejected: protected fact)
    #   1: valid, but a broad, unnecessarily large rewrite
    #   2: valid and minimal — the smallest safe edit
    changed_fact = _SOURCE.replace("37%", "40%")
    large_valid = _SOURCE.replace(
        "The pilot cut latency by 37%",
        "Across the trial, the initiative reduced response latency by fully 37%",
    )
    small_valid = _SOURCE.replace("The pilot cut", "The pilot trimmed")

    strategy = OneShotRewriteStrategy(
        adapter=FakeModelAdapter(responses=(changed_fact, large_valid, small_valid)),
        config=RewriteConfig(style="lexical", candidate_count=3),
    )
    result = strategy.run(_doc(), ReductionTarget(mode="sanitize_only"), _budget(), _POLICY)

    assert result.state == "rewritten_unverified"
    assert result.selected is not None
    # The bad-fact candidate was rejected and can never be selected.
    assert "40%" not in result.selected.text
    assert result.trace.candidates_rejected == 1
    # The smallest valid rewrite is chosen over the larger valid one.
    assert result.selected.text == small_valid
    # And it really is the smaller edit of the two valid candidates.
    assert result.selected.char_edit_ratio < _char_edit(large_valid)


def _char_edit(candidate: str) -> float:
    from unmark.core.operations import char_edit_ratio
    from unmark.strategies.rewrite.engine import diff_operations

    ops = diff_operations(_SOURCE, candidate, operator="test")
    return char_edit_ratio(_SOURCE, candidate, ops)


class _MarkerDetector:
    """Risk falls as a filler marker phrase disappears."""

    detector_id = "demo-marker"
    detector_version = "1.0"
    id = detector_id
    version = detector_version

    def score(self, text: str) -> DetectorScore:
        hits = text.count("It should be noted")
        return DetectorScore(
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            evidence_kind="surrogate",
            score=float(hits) / 3.0,
            risk_direction="higher",
        )


def test_recursive_restores_earlier_result_when_later_hop_regresses() -> None:
    detector = _MarkerDetector()
    # The improving hop drops one filler marker (risk 3 -> 2) with a small edit.
    improved = _SOURCE.replace(
        " It should be noted that these results were significant.",
        " These results were significant.",
        1,
    )
    # A regressing later hop re-adds a marker (risk climbs back to 3) *and* makes a
    # broad, larger edit. It is worse on residual risk and on edit cost, so it can
    # never win selection — the engine keeps the earlier improved candidate.
    regressed = improved.replace(
        " These results were significant.",
        " It should further be noted, importantly, that these particular results were "
        "quite significant indeed.",
        1,
    )

    def responder(request: ModelRequest, index: int) -> str:
        # Hop 0 rewrites the source (which still has all three markers) -> improve.
        # Later hops rewrite the improved base (one marker already dropped) -> regress.
        if request.prompt.count("It should be noted") >= 3:
            return improved
        return regressed

    strategy = RecursiveRewriteStrategy(
        adapter=FakeModelAdapter(responder=responder),
        config=RewriteConfig(style="faithful", rounds=3, candidate_count=1, early_stop_patience=5),
        detector=detector,
    )
    # A large required reduction (0.9) is never met by dropping a single marker
    # (worth ~0.33), so the loop runs all its hops and the rollback path — keep the
    # improved base, discard the regression — is exercised.
    target = ReductionTarget(mode="relative_reduction", min_score_reduction=0.9)
    result = strategy.run(_doc(), target, _budget(), _POLICY)

    assert result.selected is not None
    # The engine selected the earlier improved result, not a later regression.
    assert result.selected.text == improved
    assert result.selected.text.count("It should be noted") == 2  # one marker dropped, not re-added
    # More than one hop ran (so a later hop did regress and was discarded).
    assert len(result.trace.hops) >= 2
