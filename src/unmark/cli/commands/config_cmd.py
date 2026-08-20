"""``unmark config`` -- init, validate, and schema."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from unmark.application.rewrite_service import check_rewrite_configuration
from unmark.cli.render import emit_diagnostic, emit_stdout, make_console, to_json
from unmark.orchestration.config import (
    CONFIG_FILENAME,
    DEFAULT_CONFIG_TEMPLATE,
    UnmarkedConfig,
    resolve_config,
)
from unmark.storage.atomic import atomic_write_text

config_app = typer.Typer(no_args_is_help=True, help="Inspect and manage configuration.")


@config_app.command("init")
def config_init(
    path: Annotated[
        Path | None, typer.Option("--path", help=f"Where to write (default ./{CONFIG_FILENAME}).")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing file.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Write a commented starter configuration file."""
    console = make_console(no_color=no_color)
    destination = path or Path.cwd() / CONFIG_FILENAME
    atomic_write_text(destination, DEFAULT_CONFIG_TEMPLATE, force=force)
    emit_diagnostic(console, f"wrote {destination}")


@config_app.command("validate")
def config_validate(
    config: Annotated[
        Path | None, typer.Option("--config", help="Validate this file specifically.")
    ] = None,
    explain: Annotated[
        bool, typer.Option("--explain", help="Show each effective value and its source.")
    ] = False,
    output_format: Annotated[str, typer.Option("--format", help="text or json.")] = "text",
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Validate configuration and report the effective values."""
    console = make_console(no_color=no_color)
    resolved = resolve_config(explicit_config=config)

    if output_format == "json":
        payload = {
            "valid": True,
            "config": resolved.config.model_dump(mode="json"),
            "sources": resolved.sources,
            "layers": [{"name": layer.name, "origin": layer.origin} for layer in resolved.layers],
        }
        emit_stdout(to_json(payload) + "\n")
        return

    lines = ["configuration is valid", ""]
    lines.append("layers, lowest precedence first:")
    for layer in resolved.layers:
        lines.append(f"  {layer.name} ({layer.origin})")
    if explain:
        lines.append("")
        lines.append("effective values:")
        width = max((len(key) for key, _, _ in resolved.explain()), default=0)
        for key, value, source in resolved.explain():
            lines.append(f"  {key:<{width}}  {value!r:<24}  from {source}")
    emit_diagnostic(console, "\n".join(lines))


@config_app.command("rewrite-check")
def config_rewrite_check(
    config: Annotated[
        Path | None, typer.Option("--config", help="Check this configuration specifically.")
    ] = None,
    source_provider: Annotated[
        str | None,
        typer.Option(
            "--source-provider",
            help="Provider that generated the input, or human/unknown.",
        ),
    ] = None,
    output_format: Annotated[str, typer.Option("--format", help="text or json.")] = "text",
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
) -> None:
    """Check rewrite provider separation and credential readiness without a model call."""
    console = make_console(no_color=no_color)
    resolved = resolve_config(explicit_config=config)
    section = resolved.config.rewrite
    if source_provider is not None:
        section = section.model_copy(update={"source_provider": source_provider})
    status = check_rewrite_configuration(section)

    if output_format == "json":
        emit_stdout(to_json(status) + "\n")
        return
    lines = [
        "rewrite configuration is ready",
        f"  backend: {status['backend']}",
        f"  endpoint: {status['endpoint']}",
        f"  model: {status['model']}",
        f"  source provider: {status['source_provider']}",
        f"  rewrite provider: {status['rewrite_provider']}",
    ]
    if status["credential_required"]:
        lines.append(f"  credential: environment variable {status['key_env']} is set")
    else:
        lines.append("  credential: not required for this endpoint")
    emit_diagnostic(console, "\n".join(lines))


@config_app.command("schema")
def config_schema() -> None:
    """Print the JSON Schema for the configuration file."""
    emit_stdout(to_json(UnmarkedConfig.model_json_schema()) + "\n")
