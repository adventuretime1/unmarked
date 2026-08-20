"""Budget accounting with reservations.

Strategies never mutate a global counter. They ask a :class:`BudgetAccount` for a
:class:`BudgetLease`, which *reserves* capacity up front and settles to the actual
usage afterwards. Reserving before the work happens is what makes concurrent
proposal generation unable to overspend by race: two strategies cannot both see
the same remaining capacity and both spend it.
"""

from __future__ import annotations

import threading
from decimal import Decimal
from types import TracebackType
from typing import Literal, Self

from pydantic import Field

from unmark.core.errors import BudgetExhaustedError
from unmark.core.spans import StrictModel

CounterName = Literal[
    "runtime_ms",
    "model_calls",
    "detector_queries",
    "cost_usd",
    "candidates",
    "rounds",
]


class RunBudget(StrictModel):
    """Hard limits for one run.

    Three different meanings of "how long" are kept separate: ``max_input_chars``
    bounds the input, ``max_length_drift_ratio`` bounds the permitted output-length
    change, and ``max_runtime_ms`` bounds processing time.
    """

    schema_version: Literal["1"] = "1"
    max_runtime_ms: int = Field(default=30_000, ge=0)
    max_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    max_model_calls: int = Field(default=0, ge=0)
    max_detector_queries: int = Field(default=0, ge=0)
    max_char_edit_ratio: float = Field(default=0.08, ge=0.0, le=1.0)
    max_token_edit_ratio: float = Field(default=0.12, ge=0.0, le=1.0)
    max_length_drift_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    max_rounds: int = Field(default=3, ge=0)
    max_candidates: int = Field(default=32, ge=0)
    max_input_chars: int = Field(default=1_000_000, ge=0)

    def limit_for(self, counter: CounterName) -> Decimal:
        limits: dict[CounterName, Decimal] = {
            "runtime_ms": Decimal(self.max_runtime_ms),
            "model_calls": Decimal(self.max_model_calls),
            "detector_queries": Decimal(self.max_detector_queries),
            "cost_usd": self.max_cost_usd,
            "candidates": Decimal(self.max_candidates),
            "rounds": Decimal(self.max_rounds),
        }
        return limits[counter]


class BudgetUsage(StrictModel):
    """Settled usage for a run; reported verbatim in the run report."""

    schema_version: Literal["1"] = "1"
    runtime_ms: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    detector_queries: int = Field(default=0, ge=0)
    cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    candidates: int = Field(default=0, ge=0)
    rounds: int = Field(default=0, ge=0)
    char_edit_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    token_edit_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    length_drift_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class BudgetLease:
    """A reservation against a :class:`BudgetAccount`.

    Use as a context manager. On exit the lease settles: the reserved amount is
    released and the actual amount recorded. An unsettled lease releases its full
    reservation, so an exception never silently consumes budget.
    """

    def __init__(self, account: BudgetAccount, counter: CounterName, amount: Decimal) -> None:
        self._account = account
        self._counter = counter
        self._reserved = amount
        self._actual: Decimal | None = None
        self._closed = False

    @property
    def counter(self) -> CounterName:
        return self._counter

    @property
    def reserved(self) -> Decimal:
        return self._reserved

    def settle(self, actual: Decimal | int | float) -> None:
        """Record the amount actually used. Must not exceed the reservation."""
        value = Decimal(str(actual))
        if value < 0:
            msg = "settled amount cannot be negative"
            raise ValueError(msg)
        if value > self._reserved:
            msg = (
                f"settled {value} exceeds reservation {self._reserved} for counter {self._counter}"
            )
            raise BudgetExhaustedError(msg)
        self._actual = value

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        actual = self._actual if self._actual is not None else Decimal(0)
        self._account._release(self._counter, self._reserved, actual)


class BudgetAccount:
    """Thread-safe central enforcement of a :class:`RunBudget`.

    Tracks, per counter, the amount already spent and the amount currently
    reserved by open leases. A reservation is refused when
    ``spent + reserved + requested > limit``.
    """

    def __init__(self, budget: RunBudget) -> None:
        self._budget = budget
        self._lock = threading.Lock()
        self._spent: dict[CounterName, Decimal] = {}
        self._reserved: dict[CounterName, Decimal] = {}

    @property
    def budget(self) -> RunBudget:
        return self._budget

    def spent(self, counter: CounterName) -> Decimal:
        with self._lock:
            return self._spent.get(counter, Decimal(0))

    def remaining(self, counter: CounterName) -> Decimal:
        with self._lock:
            return self._remaining_locked(counter)

    def _remaining_locked(self, counter: CounterName) -> Decimal:
        used = self._spent.get(counter, Decimal(0)) + self._reserved.get(counter, Decimal(0))
        return self._budget.limit_for(counter) - used

    def reserve(self, counter: CounterName, amount: Decimal | int | float = 1) -> BudgetLease:
        """Reserve capacity, or raise :class:`BudgetExhaustedError`."""
        value = Decimal(str(amount))
        if value < 0:
            msg = "reservation cannot be negative"
            raise ValueError(msg)
        with self._lock:
            if value > self._remaining_locked(counter):
                limit = self._budget.limit_for(counter)
                spent = self._spent.get(counter, Decimal(0))
                reserved = self._reserved.get(counter, Decimal(0))
                msg = (
                    f"budget exhausted for {counter}: requested {value}, "
                    f"limit {limit}, spent {spent}, reserved {reserved}"
                )
                raise BudgetExhaustedError(msg)
            self._reserved[counter] = self._reserved.get(counter, Decimal(0)) + value
        return BudgetLease(self, counter, value)

    def _release(self, counter: CounterName, reserved: Decimal, actual: Decimal) -> None:
        with self._lock:
            self._reserved[counter] = self._reserved.get(counter, Decimal(0)) - reserved
            if self._reserved[counter] <= 0:
                self._reserved.pop(counter)
            if actual:
                self._spent[counter] = self._spent.get(counter, Decimal(0)) + actual

    def check_ratio(self, name: str, value: float, limit: float) -> None:
        """Enforce a ratio limit that is measured rather than reserved."""
        if value > limit:
            msg = f"{name} {value:.4f} exceeds budget {limit:.4f}"
            raise BudgetExhaustedError(msg)

    def usage(
        self,
        *,
        char_edit_ratio: float = 0.0,
        token_edit_ratio: float = 0.0,
        length_drift_ratio: float = 0.0,
    ) -> BudgetUsage:
        """Snapshot settled counters, combined with measured ratios.

        Ratios are measured after the fact rather than reserved, so they are
        passed in explicitly instead of being tracked as counters.
        """
        with self._lock:
            spent = dict(self._spent)
        return BudgetUsage(
            runtime_ms=int(spent.get("runtime_ms", Decimal(0))),
            model_calls=int(spent.get("model_calls", Decimal(0))),
            detector_queries=int(spent.get("detector_queries", Decimal(0))),
            cost_usd=spent.get("cost_usd", Decimal(0)),
            candidates=int(spent.get("candidates", Decimal(0))),
            rounds=int(spent.get("rounds", Decimal(0))),
            char_edit_ratio=char_edit_ratio,
            token_edit_ratio=token_edit_ratio,
            length_drift_ratio=length_drift_ratio,
        )
