"""``unmark voice`` -- manage stored writing-voice profiles."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from unmark.cli.render import emit_diagnostic, emit_stdout, make_console, to_json
from unmark.core.errors import UsageError
from unmark.storage.voice_store import VoiceStore, new_profile, resolve_voice

voice_app = typer.Typer(no_args_is_help=True, help="Manage writing-voice profiles.")


@voice_app.command("path")
def voice_path(
    name: Annotated[
        str | None,
        typer.Argument(metavar="NAME", help="Print the path this profile would occupy."),
    ] = None,
) -> None:
    """Print where profiles are stored.

    Written to stdout so the voice-analysis skill can ask Unmarked for the location
    instead of hard-coding it. For a name that already exists, the real path is
    printed; otherwise the path ``voice save`` would create.
    """
    store = VoiceStore()
    if not name:
        emit_stdout(f"{store.root}\n")
        return
    existing = store.find(name)
    emit_stdout(f"{existing or store.path_for(name, '.json')}\n")


@voice_app.command("list")
def voice_list(
    output_format: Annotated[str, typer.Option("--format", help="text or json.")] = "text",
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """List stored voice profiles."""
    console = make_console(no_color=no_color)
    store = VoiceStore()
    names = store.list_names()

    if output_format == "json":
        emit_stdout(to_json({"directory": str(store.root), "voices": list(names)}) + "\n")
        return

    if not names:
        emit_diagnostic(
            console,
            f"no voice profiles in {store.root}\n"
            "Run the voice-analysis skill in your coding agent to create one.",
        )
        return
    emit_diagnostic(console, "\n".join(names))


@voice_app.command("show")
def voice_show(
    name: Annotated[str, typer.Argument(metavar="NAME", help="Profile name or path.")],
    output_format: Annotated[str, typer.Option("--format", help="text or json.")] = "text",
) -> None:
    """Print a stored profile."""
    profile = resolve_voice(name)
    if output_format == "json":
        emit_stdout(to_json(profile) + "\n")
        return
    emit_stdout(profile.description.rstrip("\n") + "\n")


@voice_app.command("save")
def voice_save(
    name: Annotated[str, typer.Argument(metavar="NAME", help="Name to store it under.")],
    from_file: Annotated[
        Path | None,
        typer.Option("--from", help="Read the description from PATH, or '-' for stdin."),
    ] = None,
    generated_by: Annotated[
        str, typer.Option("--generated-by", help="Tool that produced the description.")
    ] = "",
    force: Annotated[bool, typer.Option("--force", help="Replace an existing profile.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Store a voice profile from a file or stdin.

    The description is prose written by a coding agent that analyzed the user's
    writing samples; Unmarked stores it verbatim.
    """
    console = make_console(no_color=no_color)

    if from_file is None or str(from_file) == "-":
        if sys.stdin.isatty():
            msg = "no input: pass --from PATH, or pipe the description on stdin"
            raise UsageError(msg)
        description = sys.stdin.read()
    else:
        if not from_file.is_file():
            msg = f"input not found: {from_file}"
            raise UsageError(msg)
        description = from_file.read_text(encoding="utf-8")

    store = VoiceStore()
    profile = new_profile(name, description, generated_by=generated_by)
    path = store.save(profile, force=force)
    emit_diagnostic(console, f"saved voice profile {name!r} to {path}")


@voice_app.command("delete")
def voice_delete(
    name: Annotated[str, typer.Argument(metavar="NAME", help="Profile to remove.")],
    assume_yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Do not prompt for confirmation.")
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Delete a stored voice profile."""
    console = make_console(no_color=no_color)
    store = VoiceStore()
    path = store.find(name)
    if path is None:
        msg = f"no voice profile named {name!r} in {store.root}"
        raise UsageError(msg)

    # Profiles are hand-authored and cheap to lose; never delete unprompted.
    if not assume_yes:
        if not sys.stdin.isatty():
            msg = f"refusing to delete {name!r} without confirmation; pass --yes"
            raise UsageError(msg)
        if not typer.confirm(f"delete voice profile {name!r} ({path})?"):
            emit_diagnostic(console, "cancelled")
            return

    store.delete(name)
    emit_diagnostic(console, f"deleted voice profile {name!r}")
