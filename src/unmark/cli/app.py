"""CLI entry point.

Every :class:`~unmark.core.errors.UnmarkedError` is mapped to its documented exit code
in one place, so no command function decides an exit status.
"""

from __future__ import annotations

import sys
from typing import Annotated, NoReturn

import typer

from unmark import __version__
from unmark.cli.commands.attachment_cmd import attachment_app
from unmark.cli.commands.config_cmd import config_app
from unmark.cli.commands.edit_cmd import edit_command
from unmark.cli.commands.inspect_cmd import inspect_command
from unmark.cli.commands.runs_cmd import runs_app
from unmark.cli.commands.skills_cmd import skills_app
from unmark.cli.commands.voice_cmd import voice_app
from unmark.cli.render import emit_error, make_console
from unmark.core.errors import ExitCode, UnmarkedError

app = typer.Typer(
    name="unmark",
    no_args_is_help=True,
    add_completion=True,
    help=(
        "Inspect and prepare text or attachment metadata with evidence reports.\n\n"
        "Unicode cleaning removes recognized hidden carriers. Rewriting paraphrases text "
        "through fidelity gates and still requires review. Attachment cleaning targets "
        "embedded metadata without changing pixels. Networked rewriting and remote "
        "endpoints require explicit selection and opt-in."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command("inspect")(inspect_command)
app.command("edit")(edit_command)
app.add_typer(attachment_app, name="attachment")
app.add_typer(config_app, name="config")
app.add_typer(runs_app, name="runs")
app.add_typer(skills_app, name="skills")
app.add_typer(voice_app, name="voice")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"unmark {__version__}")
        raise typer.Exit(ExitCode.SUCCESS)


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Unmarked command line interface."""


def main() -> NoReturn:
    """Console-script entry point with centralized error handling."""
    try:
        app(standalone_mode=False)
    except UnmarkedError as exc:
        emit_error(make_console(), exc.message)
        sys.exit(int(exc.exit_code))
    except typer.BadParameter as exc:
        emit_error(make_console(), str(exc))
        sys.exit(int(ExitCode.USAGE))
    except typer.Exit as exc:
        sys.exit(exc.exit_code)
    except click_exceptions() as exc:  # pragma: no cover - delegated to click
        code = getattr(exc, "exit_code", int(ExitCode.USAGE))
        message = getattr(exc, "message", str(exc))
        emit_error(make_console(), message)
        sys.exit(code)
    except KeyboardInterrupt:
        emit_error(make_console(), "interrupted")
        sys.exit(int(ExitCode.INTERRUPTED_WITH_CHECKPOINT))
    except BrokenPipeError:  # pragma: no cover - depends on the consumer
        sys.exit(int(ExitCode.SUCCESS))
    sys.exit(int(ExitCode.SUCCESS))


def click_exceptions() -> tuple[type[BaseException], ...]:
    """Click's usage errors, resolved lazily so click stays an implementation detail.

    Returns an empty tuple when click is absent -- recent Typer vendors its own
    CLI layer, so click is not guaranteed to be importable. An empty tuple in an
    ``except`` clause simply matches nothing, which lets the handler below run.
    Importing unconditionally here would raise ``ModuleNotFoundError`` while
    handling an unrelated error and mask the real failure.
    """
    try:
        import click
    except ModuleNotFoundError:
        return ()
    return (click.ClickException, click.exceptions.UsageError)


if __name__ == "__main__":  # pragma: no cover
    main()
