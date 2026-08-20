"""Local run store under ``.unmark/runs/<run-id>/``.

Layout for this slice:

```text
.unmark/runs/<run-id>/
  request.json
  effective-config.json
  source.sha256
  events.jsonl
  output.txt      (when retained)
  diff.patch      (when requested)
  report.json     (terminal state, written atomically last)
```

``report.json`` is written atomically and last, so the presence of a valid report
means the run reached a terminal state.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from unmark.core.errors import UsageError
from unmark.core.events import Event
from unmark.storage.atomic import atomic_write_text

RUNS_DIRNAME = ".unmark"


def new_run_id(now: datetime | None = None, entropy: str | None = None) -> str:
    """A sortable, filesystem-safe run identifier."""
    moment = now or datetime.now(UTC)
    suffix = entropy or os.urandom(4).hex()
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{suffix}"


def _dump(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


class RunStore:
    """Filesystem-backed store for one workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root / RUNS_DIRNAME / "runs"

    def directory(self, run_id: str) -> Path:
        return self.root / run_id

    def create(self, run_id: str) -> Path:
        directory = self.directory(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_json(self, run_id: str, name: str, value: Any) -> Path:
        directory = self.create(run_id)
        path = directory / name
        atomic_write_text(path, _dump(value) + "\n", force=True)
        return path

    def write_text(self, run_id: str, name: str, text: str) -> Path:
        directory = self.create(run_id)
        path = directory / name
        atomic_write_text(path, text, force=True)
        return path

    def append_events(self, run_id: str, events: tuple[Event, ...]) -> Path:
        """Append events as JSONL, preserving sequence order."""
        directory = self.create(run_id)
        path = directory / "events.jsonl"
        lines = "".join(event.model_dump_json() + "\n" for event in events)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(lines)
            stream.flush()
            os.fsync(stream.fileno())
        return path

    def exists(self, run_id: str) -> bool:
        return self.directory(run_id).is_dir()

    def load_json(self, run_id: str, name: str) -> Any:
        path = self.directory(run_id) / name
        if not path.exists():
            msg = f"run {run_id} has no {name}"
            raise UsageError(msg)
        return json.loads(path.read_text(encoding="utf-8"))

    def read_events(self, run_id: str) -> tuple[dict[str, Any], ...]:
        path = self.directory(run_id) / "events.jsonl"
        if not path.exists():
            return ()
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return tuple(json.loads(line) for line in lines)

    def list_runs(self) -> tuple[str, ...]:
        if not self.root.is_dir():
            return ()
        return tuple(sorted(p.name for p in self.root.iterdir() if p.is_dir()))
