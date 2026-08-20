"""``unmark attachment`` commands for supported image and document files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from unmark.attachments import clean_attachment, inspect_attachment
from unmark.cli.render import emit_diagnostic, emit_stdout, make_console, to_json
from unmark.core.errors import UsageError, ValidationError
from unmark.storage.atomic import atomic_write_bytes, atomic_write_text, default_output_path

attachment_app = typer.Typer(
    name="attachment",
    no_args_is_help=True,
    help=(
        "Inspect or clean embedded metadata in PNG, JPEG, SVG, PDF, DOCX, ODT, HTML, "
        "and frontmatter-bearing Markdown attachments."
    ),
)


def _read_binary(input_path: str) -> tuple[bytes, Path | None]:
    if input_path == "-":
        return sys.stdin.buffer.read(), None
    path = Path(input_path)
    if not path.exists():
        raise UsageError(f"input does not exist: {path}")
    if not path.is_file():
        raise UsageError(f"input is not a file: {path}")
    try:
        return path.read_bytes(), path
    except OSError as exc:
        raise UsageError(f"could not read {path}: {exc}") from exc


def _write_report(report_path: Path | None, value: object) -> None:
    if report_path is not None:
        atomic_write_text(report_path, to_json(value) + "\n", force=True)


@attachment_app.command("inspect")
def attachment_inspect_command(
    input_path: Annotated[
        str, typer.Argument(metavar="INPUT", help="Attachment path, or '-' for binary stdin.")
    ],
    output_format: Annotated[str, typer.Option("--format", help="text or json.")] = "text",
    report: Annotated[
        Path | None, typer.Option("--report", help="Write the JSON evidence report to PATH.")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress text diagnostics.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Inspect attachment type and embedded metadata without modifying bytes."""
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("must be text or json", param_hint="--format")
    data, _ = _read_binary(input_path)
    result = inspect_attachment(data)
    console = make_console(no_color=no_color, quiet=quiet)
    if output_format == "json":
        emit_stdout(to_json(result) + "\n")
    elif not quiet:
        evidence = ", ".join(item.kind for item in result.evidence)
        emit_diagnostic(
            console,
            f"{result.source.media_type}  {result.source.byte_count} bytes  state={result.state}\n"
            f"evidence: {evidence}\nC2PA verifier: {result.c2pa_verifier}",
        )
    _write_report(report, result)


@attachment_app.command("clean")
def attachment_clean_command(
    input_path: Annotated[
        str, typer.Argument(metavar="INPUT", help="Attachment path, or '-' for binary stdin.")
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Destination path, or '-' for binary stdout. Defaults to a sibling file.",
        ),
    ] = None,
    output_format: Annotated[
        str, typer.Option("--format", help="text or json evidence report.")
    ] = "text",
    report: Annotated[
        Path | None, typer.Option("--report", help="Write the JSON evidence report to PATH.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Transform and verify in memory; publish nothing.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Replace an existing output file.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress text diagnostics.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Remove targeted embedded metadata and publish only after re-inspection."""
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("must be text or json", param_hint="--format")
    data, source = _read_binary(input_path)
    if source is None and output is None and not dry_run:
        output = "-"
    destination = (
        default_output_path(source)
        if output is None and source is not None
        else (Path(output) if output not in {None, "-"} else None)
    )
    outcome = clean_attachment(data)
    console = make_console(no_color=no_color, quiet=quiet)
    _write_report(report, outcome.report)

    if outcome.output_bytes is None:
        if output_format == "json":
            emit_stdout(to_json(outcome.report) + "\n")
        elif not quiet:
            emit_diagnostic(console, "attachment clean failed verification; no output published")
            for note in outcome.report.notes:
                emit_diagnostic(console, f"- {note}")
        raise ValidationError("attachment clean failed verification; see the evidence report")

    if not dry_run:
        if output == "-":
            sys.stdout.buffer.write(outcome.output_bytes)
            sys.stdout.buffer.flush()
        elif destination is not None:
            atomic_write_bytes(destination, outcome.output_bytes, source=source, force=force)

    if output_format == "json":
        serialized = to_json(outcome.report)
        # Never mix a JSON report into binary stdout.
        if output == "-" and not dry_run:
            emit_diagnostic(console, serialized)
        else:
            emit_stdout(serialized + "\n")
    elif not quiet:
        target = "dry-run" if dry_run else (str(destination) if destination else "stdout")
        emit_diagnostic(
            console,
            f"state={outcome.report.state} actions={len(outcome.report.actions)} output={target}",
        )
