"""Application-level composition point for targeted constrained search.

The foundation CLI can call this function without importing any detector or model
implementation.  CLI rendering and strategy logic remain separate.
"""

from __future__ import annotations

from unmark.core.budgets import BudgetAccount
from unmark.core.document import Document
from unmark.core.policies import FidelityPolicy
from unmark.core.targets import ReductionTarget
from unmark.strategies.targeted.strategy import TargetedSearchResult, TargetedSearchStrategy


def run_targeted_search(
    *,
    strategy: TargetedSearchStrategy,
    document: Document,
    target: ReductionTarget,
    policy: FidelityPolicy,
) -> TargetedSearchResult:
    budget = BudgetAccount(strategy.config.run_budget)
    return strategy.run(document, target, budget, policy)
