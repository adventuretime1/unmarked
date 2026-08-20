"""``unmark inspect`` -- read-only document inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from unmark.application.inspect_service import inspect_document
from unmark.application.requests import InspectRequest
from unmark.cli.parameters import parse_media_type, parse_output_format
from unmark.cli.render import emit_diagnostic, emit_stdout, make_console, to_json
from unmark.core.policies import UnicodePolicyName
from unmark.reporting.render_text import render_inspect
from unmark.storage.atomic import atomic_write_text


def inspect_command(
    input_path: Annotated[
        str, typer.Argument(metavar="INPUT", help="File path, or '-' for stdin.")
    ],
    media_type: Annotated[
        str | None,
        typer.Option("--media-type", help="text/plain or text/markdown."),
    ] = None,
    unicode_policy: Annotated[
        str,
        typer.Option(
            "--unicode-policy",
            help="report|safe|typographic|aggressive. Inspection never mutates.",
        ),
    ] = "report",
    lock: Annotated[
        list[str] | None,
        typer.Option("--lock", help="Regex that must survive verbatim. Repeatable."),
    ] = None,
    output_format: Annotated[str, typer.Option("--format", help="text or json.")] = "text",
    report: Annotated[
        Path | None, typer.Option("--report", help="Write a JSON report to PATH.")
    ] = None,
    research_mode: Annotated[
        bool,
        typer.Option("--research-mode", help="Acknowledge research-only Unicode policies."),
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress diagnostics.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Report structure, protected spans, Unicode findings, and the source hash."""
    console = make_console(no_color=no_color, quiet=quiet)

    request = InspectRequest(
        input=input_path,
        media_type=parse_media_type(media_type),
        unicode_policy=_unicode_policy(unicode_policy),
        locks=tuple(lock or ()),
        report_path=report,
        output_format=parse_output_format(output_format),
        research_mode=research_mode,
    )
    result = inspect_document(request)

    if request.output_format == "json":
        emit_stdout(to_json(result) + "\n")
    elif not quiet:
        emit_diagnostic(console, render_inspect(result))

    if report is not None:
        atomic_write_text(report, to_json(result) + "\n", force=True)


def _unicode_policy(value: str) -> UnicodePolicyName:
    allowed = {"report", "safe", "typographic", "aggressive"}
    if value not in allowed:
        raise typer.BadParameter(
            f"must be one of {', '.join(sorted(allowed))}", param_hint="--unicode-policy"
        )
    return cast(UnicodePolicyName, value)
