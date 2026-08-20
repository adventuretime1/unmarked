"""Human-facing rendering.

Rich is used only for stderr diagnostics. Machine-readable output (JSON, edited
text, diffs) is written with plain ``print`` so it is never decorated, wrapped, or
colorized, which would corrupt a pipeline.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from rich.console import Console


def make_console(*, no_color: bool = False, quiet: bool = False) -> Console:
    """A stderr console. Respects ``--no-color`` and the ``NO_COLOR`` convention."""
    disabled = no_color or bool(os.environ.get("NO_COLOR"))
    return Console(
        stderr=True,
        no_color=disabled,
        force_terminal=False if disabled else None,
        quiet=quiet,
        highlight=False,
        soft_wrap=True,
    )


def emit_stdout(text: str) -> None:
    """Write machine-consumable text to stdout exactly as given."""
    sys.stdout.write(text)
    sys.stdout.flush()


def emit_diagnostic(console: Console, text: str) -> None:
    """Write human diagnostics to stderr."""
    console.print(text, markup=False, highlight=False)


def emit_error(console: Console, message: str) -> None:
    console.print(f"error: {message}", markup=False, highlight=False, style="bold red")


def emit_warning(console: Console, message: str) -> None:
    console.print(f"warning: {message}", markup=False, highlight=False, style="yellow")


def to_json(value: Any) -> str:
    """Serialize a report or mapping to indented JSON."""
    import json

    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)
