"""Deterministic end-to-end demonstration for targeted constrained search.

Run with ``python -m unmark.evaluation.targeted_demo``.  It uses no network,
model download, API key, GPU, or undisclosed watermark claim.
"""

from __future__ import annotations

import json

from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.policies import FidelityPolicy
from unmark.core.targets import ReductionTarget
from unmark.detectors.protocols import DetectorScore, ScoreContribution
from unmark.inspect.ingest import build_document
from unmark.strategies.targeted.config import TargetedSearchConfig
from unmark.strategies.targeted.propose import DeterministicProposalProvider
from unmark.strategies.targeted.strategy import TargetedSearchStrategy


class SyntheticContributionDetector:
    """Known-region research detector used only for the offline demonstration."""

    id = "synthetic:known-region"
    version = "1"
    phrase = "signal-bearing wording"

    def score(self, text: str) -> DetectorScore:
        start = text.find(self.phrase)
        detected = start >= 0
        return DetectorScore(
            detector_id=self.id,
            detector_version=self.version,
            evidence_kind="research",
            score=8.0 if detected else 1.0,
            threshold=2.0,
            calibrated_fpr=0.01,
            token_count=len(text.split()),
            contributions=(
                ScoreContribution(
                    start=start,
                    end=start + len(self.phrase),
                    score=8.0,
                    token=self.phrase,
                ),
            )
            if detected
            else (),
        )


def run_demo() -> dict[str, object]:
    source = (
        "This signal-bearing wording appears in an intentionally longer sentence "
        "with protected value 42 and enough context for a meaningful edit ratio."
    )
    document = build_document(source, "text/plain")
    run_budget = RunBudget(
        max_runtime_ms=10_000,
        max_detector_queries=8,
        max_candidates=8,
        max_rounds=2,
        max_char_edit_ratio=0.5,
        max_token_edit_ratio=0.5,
        max_length_drift_ratio=0.5,
    )
    config = TargetedSearchConfig(
        beam_width=4,
        candidates_per_node=4,
        max_search_depth=2,
        minimum_score_improvement=0.1,
        run_budget=run_budget,
    )
    strategy = TargetedSearchStrategy(
        config=config,
        detector=SyntheticContributionDetector(),
        proposal_provider=DeterministicProposalProvider(
            {
                SyntheticContributionDetector.phrase: (
                    "neutral wording",
                    "a much longer and needlessly expansive alternative wording",
                )
            },
            allowed_operators=("exact_phrase",),
        ),
    )
    result = strategy.run(
        document,
        ReductionTarget(mode="verify_below_threshold", threshold=2),
        BudgetAccount(run_budget),
        FidelityPolicy(require_bidirectional_entailment=False),
    )
    selected = result.selected
    return {
        "state": result.state,
        "baseline_score": result.baseline.score if result.baseline else None,
        "final_score": selected.score.score if selected and selected.score else None,
        "selected_text": selected.text if selected else None,
        "selected_char_edit_ratio": (selected.metrics.char_edit_ratio if selected else None),
        "protected_42_preserved": bool(selected and "42" in selected.text),
        "detector_queries": result.usage.detector_queries,
        "model_calls": result.usage.model_calls,
        "stopping_reason": result.trace.stopping_reason,
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
