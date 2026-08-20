"""Request objects.

The CLI maps arguments onto these; a future HTTP or queue worker will construct
the same objects. Services take a request and return a result, so no editing
decision lives in a command function.
"""

from __future__ import annotations

from io import BufferedIOBase
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from unmark.core.document import MediaType
from unmark.core.spans import StrictModel

DiffMode = Literal["none", "unified", "operations"]
OutputFormat = Literal["text", "json"]


class InspectRequest(StrictModel):
    """A read-only inspection request."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    input: str
    media_type: MediaType | None = None
    unicode_policy: Literal["report", "safe", "typographic", "aggressive"] = "report"
    locks: tuple[str, ...] = ()
    config_path: Path | None = None
    report_path: Path | None = None
    output_format: OutputFormat = "text"
    research_mode: bool = False
    stdin: BufferedIOBase | None = None


class EditRequest(StrictModel):
    """An editing request.

    ``output`` is ``None`` for the default sibling path, ``"-"`` for stdout, or an
    explicit path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    input: str
    output: str | None = None
    preset: str = "sanitize"
    #: Whether this run is a prompt-driven rewrite rather than sanitation. When
    #: true the edit command routes to the rewrite service.
    rewrite: bool = False
    #: Overrides for the ``[rewrite]`` config section, merged as a CLI layer. Only
    #: non-secret scalar settings; an API key is never carried here.
    rewrite_overrides: dict[str, object] = Field(default_factory=dict)
    #: A stored voice-profile name, or a path to a profile file. Resolved by the
    #: rewrite service; only meaningful when ``rewrite`` is true.
    voice: str | None = None
    media_type: MediaType | None = None
    unicode_policy: Literal["report", "safe", "typographic", "aggressive"] | None = None
    locks: tuple[str, ...] = ()
    config_path: Path | None = None
    report_path: Path | None = None
    diff: DiffMode = "none"
    output_format: OutputFormat = "text"
    dry_run: bool = False
    force: bool = False
    assume_yes: bool = False
    research_mode: bool = False
    quiet: bool = False
    retain_run: bool = True
    workspace: Path | None = Field(
        default=None, description="Root for the .unmark run store; defaults to the CWD."
    )
    stdin: BufferedIOBase | None = None
