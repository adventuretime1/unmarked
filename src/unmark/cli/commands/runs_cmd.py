"""``unmark runs`` -- inspect the local run store."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from unmark.cli.render import emit_diagnostic, emit_stdout, make_console, to_json
from unmark.core.errors import UsageError
from unmark.storage.run_store import RunStore

runs_app = typer.Typer(no_args_is_help=True, help="Inspect stored runs.")


@runs_app.command("list")
def runs_list(
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace root (default: CWD).")
    ] = None,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """List stored run identifiers."""
    console = make_console(no_color=no_color)
    store = RunStore(workspace or Path.cwd())
    run_ids = store.list_runs()
    if not run_ids:
        emit_diagnostic(console, "no runs recorded")
        return
    emit_diagnostic(console, "\n".join(run_ids))


@runs_app.command("show")
def runs_show(
    run_id: Annotated[str, typer.Argument(metavar="RUN_ID", help="Run identifier.")],
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace root (default: CWD).")
    ] = None,
    output_format: Annotated[str, typer.Option("--format", help="text or json.")] = "text",
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Show the report for a stored run."""
    console = make_console(no_color=no_color)
    store = RunStore(workspace or Path.cwd())
    if not store.exists(run_id):
        msg = f"unknown run: {run_id}"
        raise UsageError(msg)

    report = store.load_json(run_id, "report.json")
    if output_format == "json":
        emit_stdout(to_json(report) + "\n")
        return

    events = store.read_events(run_id)
    lines = [
        f"Run {run_id}",
        f"  state:      {report.get('state')}",
        f"  preset:     {report.get('preset')}",
        f"  source:     {report.get('document', {}).get('origin')}",
        f"  sha256:     {report.get('document', {}).get('source_sha256')}",
        f"  operations: {report.get('operation_count')}",
        f"  output:     {report.get('output_path') or 'not retained'}",
        f"  directory:  {store.directory(run_id)}",
        "",
        f"Events: {len(events)}",
    ]
    for event in events:
        lines.append(
            f"  {event.get('sequence'):>3}  {event.get('kind'):<13} {event.get('message')}"
        )
    lines.append("")
    lines.append(f"Note: {report.get('residual_risk', '')}")
    emit_diagnostic(console, "\n".join(lines))
