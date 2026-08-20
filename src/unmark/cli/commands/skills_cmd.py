"""Install the agent skills bundled with Unmarked."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated

import typer

from unmark.cli.render import emit_diagnostic, make_console
from unmark.core.errors import UsageError

skills_app = typer.Typer(
    name="skills",
    no_args_is_help=True,
    help="Install Unmarked's bundled skills into an agent skills directory.",
)


def _bundled_skills() -> Traversable:
    bundled = files("unmark").joinpath("bundled_skills")
    if bundled.is_dir():
        return bundled
    # Editable source checkout: build-time force-include has not populated the
    # installed package tree yet, so use the canonical repository directory.
    checkout_skills = Path(__file__).parents[4] / "skills"
    if checkout_skills.is_dir():
        return checkout_skills
    raise UsageError("this Unmarked installation contains no bundled skills")


def _copy_resource(source: Traversable, destination: Path) -> None:
    if destination.is_symlink():
        raise UsageError(f"refusing symlinked skill file: {destination}")
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_resource(child, destination / child.name)
        return
    destination.write_bytes(source.read_bytes())


@skills_app.command("install")
def install_skills_command(
    target: Annotated[
        Path,
        typer.Option(
            "--target",
            help=(
                "Project skills directory. Defaults to .agents/skills; "
                "Claude Code commonly uses .claude/skills."
            ),
        ),
    ] = Path(".agents/skills"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace files in already-installed Unmarked skill folders."),
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Copy the bundled Unmarked skills into TARGET."""
    bundled = _bundled_skills()
    skill_dirs = sorted((item for item in bundled.iterdir() if item.is_dir()), key=lambda p: p.name)

    for skill in skill_dirs:
        destination = target / skill.name
        if destination.is_symlink():
            raise UsageError(f"refusing symlinked skill destination: {destination}")
        if destination.exists() and not destination.is_dir():
            raise UsageError(f"skill destination is not a directory: {destination}")
        if destination.exists() and not force:
            raise UsageError(f"skill already exists: {destination}; pass --force to update it")

    for skill in skill_dirs:
        _copy_resource(skill, target / skill.name)

    emit_diagnostic(
        make_console(no_color=no_color),
        f"installed {len(skill_dirs)} Unmarked skills in {target}",
    )
