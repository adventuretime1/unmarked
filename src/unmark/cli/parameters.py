"""Shared validation for CLI parameter values."""

from __future__ import annotations

from typing import cast

import typer

from unmark.application.requests import OutputFormat
from unmark.core.document import MediaType


def parse_media_type(value: str | None) -> MediaType | None:
    if value is None:
        return None
    if value not in {"text/plain", "text/markdown"}:
        raise typer.BadParameter("must be text/plain or text/markdown", param_hint="--media-type")
    return cast(MediaType, value)


def parse_output_format(value: str) -> OutputFormat:
    if value not in {"text", "json"}:
        raise typer.BadParameter("must be text or json", param_hint="--format")
    return cast(OutputFormat, value)
