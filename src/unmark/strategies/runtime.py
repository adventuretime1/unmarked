"""Shared strategy budget accounting."""

from __future__ import annotations

import time
from decimal import Decimal

from unmark.core.budgets import BudgetAccount


def settle_runtime(budget: BudgetAccount, started: float) -> None:
    """Charge elapsed wall time without exceeding the remaining budget."""
    elapsed = int((time.monotonic() - started) * 1000)
    remaining = int(max(Decimal(0), budget.remaining("runtime_ms")))
    amount = min(elapsed, remaining)
    if amount:
        with budget.reserve("runtime_ms", amount) as lease:
            lease.settle(amount)
