"""Fidelity ordering, Pareto behavior, and evidence-state semantics."""

from __future__ import annotations

import pytest

from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.operations import Operation, normalized_content_hash
from unmark.core.policies import FidelityPolicy
from unmark.core.targets import ReductionTarget
from unmark.detectors.localization import TextRegion
from unmark.detectors.protocols import DetectorScore
from unmark.fidelity.protocols import BasicFidelityEvaluator
from unmark.inspect.ingest import build_document
from unmark.strategies.objectives import successful_state, target_met
from unmark.strategies.targeted.beam import (
    CandidateMetrics,
    SearchCandidate,
    diverse_beam,
    lexicographic_key,
    pareto_frontier,
)


def candidate(
    text: str,
    *,
    risk: float,
    char_ratio: float,
    token_ratio: float | None = None,
    target: bool = False,
) -> SearchCandidate:
    content_hash = normalized_content_hash(text)
    return SearchCandidate(
        candidate_id=content_hash[:12],
        text=text,
        content_hash=content_hash,
        metrics=CandidateMetrics(
            detector_risk=risk,
            char_edit_ratio=char_ratio,
            token_edit_ratio=token_ratio if token_ratio is not None else char_ratio,
            length_drift_ratio=0,
        ),
        target_met=target,
    )


def test_protected_number_rejected_before_any_scorer_is_needed() -> None:
    text = "The measured value is 42 units."
    document = build_document(text, "text/plain")
    number = next(span for span in document.protected_spans if span.value == "42")
    operation = Operation(
        start=number.start,
        end=number.end,
        text="41",
        original="42",
        operator="unsafe-test",
        reason="prove rejection",
    )
    region = TextRegion(start=0, end=len(text), risk=1, mode="test")
    report = BasicFidelityEvaluator().evaluate(
        document,
        (operation,),
        FidelityPolicy(),
        BudgetAccount(
            RunBudget(
                max_char_edit_ratio=1,
                max_token_edit_ratio=1,
                max_length_drift_ratio=1,
            )
        ),
        (region,),
    )
    assert not report.passed
    assert report.rejection_reason is not None
    assert report.rejection_reason.startswith("protected_spans:")


def test_operation_outside_editable_region_rejected() -> None:
    text = "alpha beta gamma"
    document = build_document(text, "text/plain")
    operation = Operation(
        start=0,
        end=5,
        text="omega",
        original="alpha",
        operator="test",
        reason="outside",
    )
    report = BasicFidelityEvaluator().evaluate(
        document,
        (operation,),
        FidelityPolicy(),
        BudgetAccount(RunBudget(max_char_edit_ratio=1, max_token_edit_ratio=1)),
        (TextRegion(start=6, end=10, risk=1, mode="test"),),
    )
    assert not report.passed
    assert report.rejection_reason is not None
    assert report.rejection_reason.startswith("operation_locality:")


def test_block_structure_change_is_a_hard_failure() -> None:
    text = "First paragraph. Second sentence."
    document = build_document(text, "text/plain")
    split = text.index(" Second")
    operation = Operation(
        start=split,
        end=split + 1,
        text="\n\n",
        original=" ",
        operator="unsafe-structure",
        reason="create another block",
    )
    report = BasicFidelityEvaluator().evaluate(
        document,
        (operation,),
        FidelityPolicy(require_bidirectional_entailment=False),
        BudgetAccount(
            RunBudget(
                max_char_edit_ratio=1,
                max_token_edit_ratio=1,
                max_length_drift_ratio=1,
            )
        ),
        (TextRegion(start=0, end=len(text), risk=1, mode="test"),),
    )
    assert not report.passed
    assert report.rejection_reason is not None
    assert report.rejection_reason.startswith("document_structure:")


@pytest.mark.parametrize(
    "operation,budget,expected",
    [
        (
            Operation(
                start=0,
                end=3,
                text="two",
                original="one",
                operator="test",
                reason="character budget",
            ),
            RunBudget(
                max_char_edit_ratio=0.01,
                max_token_edit_ratio=1,
                max_length_drift_ratio=1,
            ),
            "character edit ratio",
        ),
        (
            Operation(
                start=0,
                end=3,
                text="alpha",
                original="one",
                operator="test",
                reason="token budget",
            ),
            RunBudget(
                max_char_edit_ratio=1,
                max_token_edit_ratio=0.01,
                max_length_drift_ratio=1,
            ),
            "token edit ratio",
        ),
        (
            Operation(
                start=49,
                end=49,
                text=" with substantial extra wording",
                original="",
                operator="test",
                reason="length budget",
            ),
            RunBudget(
                max_char_edit_ratio=1,
                max_token_edit_ratio=1,
                max_length_drift_ratio=0.01,
            ),
            "length drift",
        ),
    ],
)
def test_each_edit_budget_is_a_hard_gate(
    operation: Operation, budget: RunBudget, expected: str
) -> None:
    text = "one two three four five six seven eight nine ten."
    assert len(text) == 49
    document = build_document(text, "text/plain")
    report = BasicFidelityEvaluator().evaluate(
        document,
        (operation,),
        FidelityPolicy(require_bidirectional_entailment=False),
        BudgetAccount(budget),
        (TextRegion(start=0, end=len(text), risk=1, mode="test"),),
    )
    budget_gate = next(gate for gate in report.gates if gate.gate_id == "edit_budgets")
    assert budget_gate.status == "failed"
    assert expected in budget_gate.reason


def test_lexicographic_selection_prefers_smaller_target_meeting_edit() -> None:
    small = candidate("small", risk=2, char_ratio=0.05, target=True)
    large = candidate("large", risk=0, char_ratio=0.25, target=True)
    assert min((large, small), key=lexicographic_key) is small


def test_pareto_keeps_tradeoffs_and_removes_dominated_candidate() -> None:
    low_risk = candidate("low risk", risk=1, char_ratio=0.3)
    low_edit = candidate("low edit", risk=4, char_ratio=0.05)
    dominated = candidate("dominated", risk=5, char_ratio=0.4)
    assert set(pareto_frontier((low_risk, low_edit, dominated))) == {low_risk, low_edit}


def test_diversity_prevents_near_duplicate_from_filling_beam() -> None:
    base = candidate("The quick brown fox jumps.", risk=1, char_ratio=0.1)
    near = candidate("The quick brown fox jumps!", risk=1.1, char_ratio=0.11)
    different = candidate("A fox quickly clears the fence.", risk=2, char_ratio=0.2)
    selected = diverse_beam((base, near, different), width=2, similarity_ceiling=0.9)
    assert selected == (base, different)


def test_surrogate_cannot_meet_verified_target() -> None:
    score = DetectorScore(
        detector_id="pll",
        detector_version="1",
        evidence_kind="surrogate",
        score=0.1,
        threshold=0.5,
    )
    target = ReductionTarget(mode="verify_below_threshold", threshold=0.5)
    assert not target_met(target, score, score)


def test_pll_success_state_is_labeled_as_a_surrogate_measurement() -> None:
    score = DetectorScore(
        detector_id="pll",
        detector_version="1",
        evidence_kind="surrogate",
        score=0.1,
    )
    assert successful_state(ReductionTarget(mode="minimize_surrogate"), score) == (
        "surrogate_reduced"
    )
