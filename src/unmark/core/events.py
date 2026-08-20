"""Lifecycle events.

Every event carries a monotonically increasing sequence number, a UTC timestamp,
and the run ID. Events are appended to ``events.jsonl`` in the run directory.
"""

from __future__ import annotations

import itertools
import threading
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field

from unmark.core.spans import StrictModel

JobState = Literal[
    "created",
    "inspecting",
    "baseline_scored",
    "running",
    "validating",
    "selecting",
    "checkpointed",
    "completed",
    "abstained",
    "failed",
]

EventKind = Literal["state_change", "progress", "warning", "operation", "usage"]


class Event(StrictModel):
    """One record in the run event log."""

    schema_version: Literal["1"] = "1"
    sequence: int = Field(ge=0)
    run_id: str
    timestamp: datetime
    kind: EventKind
    state: JobState | None = None
    message: str = ""
    candidate_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class EventRecorder:
    """Thread-safe monotonic event sequencer.

    Holds events in memory and hands them to a sink callable; the run store
    supplies a sink that appends to ``events.jsonl``.
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._events: list[Event] = []

    @property
    def events(self) -> tuple[Event, ...]:
        with self._lock:
            return tuple(self._events)

    def record(
        self,
        kind: EventKind,
        message: str = "",
        *,
        state: JobState | None = None,
        candidate_id: str | None = None,
        **data: Any,
    ) -> Event:
        with self._lock:
            event = Event(
                sequence=next(self._counter),
                run_id=self._run_id,
                timestamp=datetime.now(UTC),
                kind=kind,
                state=state,
                message=message,
                candidate_id=candidate_id,
                data=data,
            )
            self._events.append(event)
            return event

    def state(self, state: JobState, message: str = "") -> Event:
        return self.record("state_change", message or state, state=state)
