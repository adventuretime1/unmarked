"""Budget accounting, reservations, and leases."""

from __future__ import annotations

import threading
from decimal import Decimal

import pytest

from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.errors import BudgetExhaustedError


class TestRunBudget:
    def test_ratios_must_be_in_unit_interval(self):
        with pytest.raises(ValueError):
            RunBudget(max_char_edit_ratio=1.5)
        with pytest.raises(ValueError):
            RunBudget(max_length_drift_ratio=-0.1)

    def test_negative_counters_rejected(self):
        with pytest.raises(ValueError):
            RunBudget(max_model_calls=-1)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError):
            RunBudget(not_a_field=1)  # type: ignore[call-arg]  # negative test


class TestReservations:
    def test_reserve_and_settle(self):
        account = BudgetAccount(RunBudget(max_model_calls=5))
        with account.reserve("model_calls", 2) as lease:
            lease.settle(2)
        assert account.spent("model_calls") == Decimal(2)
        assert account.remaining("model_calls") == Decimal(3)

    def test_unsettled_lease_spends_nothing(self):
        account = BudgetAccount(RunBudget(max_model_calls=5))
        with account.reserve("model_calls", 3):
            pass
        assert account.spent("model_calls") == Decimal(0)
        assert account.remaining("model_calls") == Decimal(5)

    def test_settling_less_than_reserved_releases_remainder(self):
        account = BudgetAccount(RunBudget(max_model_calls=10))
        with account.reserve("model_calls", 5) as lease:
            lease.settle(2)
        assert account.spent("model_calls") == Decimal(2)
        assert account.remaining("model_calls") == Decimal(8)

    def test_cannot_settle_more_than_reserved(self):
        account = BudgetAccount(RunBudget(max_model_calls=10))
        with pytest.raises(BudgetExhaustedError, match="exceeds reservation"):
            with account.reserve("model_calls", 1) as lease:
                lease.settle(5)

    def test_over_reservation_refused(self):
        account = BudgetAccount(RunBudget(max_model_calls=3))
        with pytest.raises(BudgetExhaustedError, match="budget exhausted"):
            account.reserve("model_calls", 4)

    def test_open_reservation_blocks_a_second(self):
        # This is the property that makes concurrent strategies safe: an open
        # lease holds capacity even before it is settled.
        account = BudgetAccount(RunBudget(max_model_calls=4))
        with account.reserve("model_calls", 3):
            assert account.remaining("model_calls") == Decimal(1)
            with pytest.raises(BudgetExhaustedError):
                account.reserve("model_calls", 2)

    def test_exception_inside_lease_releases_reservation(self):
        account = BudgetAccount(RunBudget(max_model_calls=5))
        with pytest.raises(RuntimeError), account.reserve("model_calls", 5):
            raise RuntimeError("boom")
        assert account.remaining("model_calls") == Decimal(5)

    def test_zero_budget_refuses_any_call(self):
        account = BudgetAccount(RunBudget(max_model_calls=0))
        with pytest.raises(BudgetExhaustedError):
            account.reserve("model_calls", 1)

    def test_decimal_cost_accounting(self):
        account = BudgetAccount(RunBudget(max_cost_usd=Decimal("1.00")))
        with account.reserve("cost_usd", Decimal("0.30")) as lease:
            lease.settle(Decimal("0.25"))
        assert account.spent("cost_usd") == Decimal("0.25")


class TestConcurrency:
    def test_concurrent_reservations_cannot_overspend(self):
        limit = 50
        account = BudgetAccount(RunBudget(max_model_calls=limit))
        granted = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(20):
                try:
                    lease = account.reserve("model_calls", 1)
                except BudgetExhaustedError:
                    return
                lease.settle(1)
                lease.close()
                with lock:
                    granted.append(1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(granted) == limit
        assert account.spent("model_calls") == Decimal(limit)
        assert account.remaining("model_calls") == Decimal(0)


class TestRatioChecks:
    def test_ratio_within_limit_passes(self):
        account = BudgetAccount(RunBudget(max_char_edit_ratio=0.1))
        account.check_ratio("char_edit_ratio", 0.05, 0.1)

    def test_ratio_over_limit_raises(self):
        account = BudgetAccount(RunBudget(max_char_edit_ratio=0.1))
        with pytest.raises(BudgetExhaustedError, match="exceeds budget"):
            account.check_ratio("char_edit_ratio", 0.5, 0.1)

    def test_usage_snapshot(self):
        account = BudgetAccount(RunBudget(max_candidates=5))
        with account.reserve("candidates", 1) as lease:
            lease.settle(1)
        usage = account.usage(char_edit_ratio=0.01, length_drift_ratio=0.02)
        assert usage.candidates == 1
        assert usage.char_edit_ratio == pytest.approx(0.01)
