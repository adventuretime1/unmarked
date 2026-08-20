"""``unmark edit`` -- deterministic sanitation for this iteration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from unmark.application.edit_service import edit_document
from unmark.application.requests import DiffMode, EditRequest
from unmark.application.rewrite_service import rewrite_document
from unmark.cli.parameters import parse_media_type, parse_output_format
from unmark.cli.render import emit_diagnostic, emit_stdout, make_console, to_json
from unmark.core.errors import AbstainedError, UnsupportedError
from unmark.core.policies import UnicodePolicyName
from unmark.reporting.render_text import render_edit, render_rewrite
from unmark.storage.atomic import atomic_write_text
from unmark.strategies.rewrite.intensity import REWRITE_INTENSITIES, intensity_profile

_REWRITE_STYLES = {"faithful", "syntax", "lexical", "polish", "simplify", "academic"}
_REWRITE_STRENGTHS = {"light", "medium", "strong"}
_REWRITE_BACKENDS = {"print-prompt", "ollama", "openai-compatible"}
_REWRITE_STRATEGIES = {"one-shot", "recursive"}
_REWRITE_INTENSITIES = set(REWRITE_INTENSITIES)


def edit_command(
    input_path: Annotated[
        str, typer.Argument(metavar="INPUT", help="File path, or '-' for stdin.")
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Destination path, or '-' for stdout. Defaults to a sibling file.",
        ),
    ] = None,
    preset: Annotated[
        str, typer.Option("--preset", help="Only 'sanitize' is available in this build.")
    ] = "sanitize",
    media_type: Annotated[
        str | None, typer.Option("--media-type", help="text/plain or text/markdown.")
    ] = None,
    unicode_policy: Annotated[
        str | None,
        typer.Option("--unicode-policy", help="report|safe|typographic|aggressive."),
    ] = None,
    lock: Annotated[
        list[str] | None,
        typer.Option("--lock", help="Regex that must survive verbatim. Repeatable."),
    ] = None,
    config: Annotated[Path | None, typer.Option("--config", help="Explicit config file.")] = None,
    report: Annotated[
        Path | None, typer.Option("--report", help="Write a JSON report to PATH.")
    ] = None,
    diff: Annotated[str, typer.Option("--diff", help="none|unified|operations.")] = "none",
    output_format: Annotated[str, typer.Option("--format", help="text or json.")] = "text",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan only; write nothing.")] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Replace an existing output file.")
    ] = False,
    assume_yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Acknowledge warnings. Does not enable research mode."),
    ] = False,
    research_mode: Annotated[
        bool,
        typer.Option("--research-mode", help="Acknowledge research-only Unicode policies."),
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress diagnostics.")] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable color.")] = False,
    rewrite: Annotated[
        bool,
        typer.Option(
            "--rewrite",
            help="Run a prompt-driven rewrite instead of Unicode sanitation.",
        ),
    ] = False,
    intensity: Annotated[
        str | None,
        typer.Option(
            "--intensity",
            help=(
                "Rewrite strength: low modestly disrupts the original statistical "
                "patterns; medium changes them substantially; high uses the broadest "
                "repeated rewrite."
            ),
        ),
    ] = None,
    strategy: Annotated[
        str | None, typer.Option("--strategy", help="Rewrite (advanced): one-shot|recursive.")
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Rewrite: print-prompt|ollama|openai-compatible."),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Rewrite: model name for the backend.")
    ] = None,
    fallback_model: Annotated[
        list[str] | None,
        typer.Option("--fallback-model", help="Rewrite: ordered fallback model; repeatable."),
    ] = None,
    provider_only: Annotated[
        list[str] | None,
        typer.Option("--provider-only", help="Rewrite: allowed provider slug; repeatable."),
    ] = None,
    endpoint: Annotated[
        str | None, typer.Option("--endpoint", help="Rewrite: model endpoint URL.")
    ] = None,
    source_provider: Annotated[
        str | None,
        typer.Option(
            "--source-provider",
            help="Rewrite: provider the input came from (or human/unknown).",
        ),
    ] = None,
    rewrite_provider: Annotated[
        str | None,
        typer.Option(
            "--rewrite-provider",
            help="Rewrite: model provider; required only when it cannot be derived.",
        ),
    ] = None,
    rewrite_style: Annotated[
        str | None,
        typer.Option(
            "--rewrite-style",
            help="Rewrite: faithful|syntax|lexical|polish|simplify|academic.",
        ),
    ] = None,
    rewrite_strength: Annotated[
        str | None, typer.Option("--rewrite-strength", help="Rewrite: light|medium|strong.")
    ] = None,
    candidates: Annotated[
        int | None, typer.Option("--candidates", help="Rewrite: candidates per hop (1-16).")
    ] = None,
    temperature: Annotated[
        float | None, typer.Option("--temperature", help="Rewrite: sampling temperature.")
    ] = None,
    target_length_ratio: Annotated[
        float | None,
        typer.Option("--target-length-ratio", help="Rewrite: desired output/source length."),
    ] = None,
    max_output_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-output-tokens",
            help="Rewrite: maximum generated tokens per candidate.",
        ),
    ] = None,
    rounds: Annotated[
        int | None, typer.Option("--rounds", help="Recursive rewrite: number of hops (1-5).")
    ] = None,
    early_stop_patience: Annotated[
        int | None,
        typer.Option("--early-stop-patience", help="Recursive: hops without gain before stopping."),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Rewrite: deterministic seed.")] = None,
    voice: Annotated[
        str | None,
        typer.Option(
            "--voice",
            help="Rewrite: stored voice-profile name, or a path to a profile file.",
        ),
    ] = None,
    allow_remote: Annotated[
        bool,
        typer.Option(
            "--allow-remote",
            help="Rewrite: permit non-loopback model endpoints. Off by default.",
        ),
    ] = False,
    key_env: Annotated[
        str | None,
        typer.Option(
            "--key-env",
            help="Rewrite: name of the env var holding the API key. Never the key itself.",
        ),
    ] = None,
    no_retain_run: Annotated[
        bool,
        typer.Option(
            "--no-retain-run",
            help="Do not persist this run under .unmark/runs.",
        ),
    ] = False,
) -> None:
    """Apply deterministic Unicode sanitation, or a prompt-driven rewrite.

    Sanitation (the default) removes recognized hidden Unicode carriers and their
    signature. A rewrite (``--rewrite``, or any rewrite-specific option) changes
    visible token, n-gram, sentence, and model-probability patterns.
    """
    console = make_console(no_color=no_color, quiet=quiet)

    rewrite_overrides = _rewrite_overrides(
        intensity=intensity,
        strategy=strategy,
        backend=backend,
        model=model,
        fallback_model=fallback_model,
        provider_only=provider_only,
        endpoint=endpoint,
        source_provider=source_provider,
        rewrite_provider=rewrite_provider,
        style=rewrite_style,
        strength=rewrite_strength,
        candidates=candidates,
        temperature=temperature,
        target_length_ratio=target_length_ratio,
        max_output_tokens=max_output_tokens,
        rounds=rounds,
        early_stop_patience=early_stop_patience,
        seed=seed,
        allow_remote=allow_remote,
        key_env=key_env,
    )
    is_rewrite = rewrite or bool(rewrite_overrides)

    request = EditRequest(
        input=input_path,
        output=output,
        preset=preset,
        rewrite=is_rewrite,
        rewrite_overrides=rewrite_overrides,
        voice=voice,
        media_type=parse_media_type(media_type),
        unicode_policy=_unicode_policy(unicode_policy),
        locks=tuple(lock or ()),
        config_path=config,
        report_path=report,
        diff=_diff(diff),
        output_format=parse_output_format(output_format),
        dry_run=dry_run,
        force=force,
        assume_yes=assume_yes,
        research_mode=research_mode,
        quiet=quiet,
        retain_run=not no_retain_run,
    )

    if is_rewrite:
        rewrite_outcome = rewrite_document(request)
        _emit(
            console,
            report_obj=rewrite_outcome.report,
            rendered=render_rewrite(rewrite_outcome.report),
            stdout_text=rewrite_outcome.stdout_text,
            output_format=request.output_format,
            quiet=quiet,
            report_path=report,
        )
        if rewrite_outcome.report.state == "abstained":
            raise AbstainedError("No fidelity-valid rewrite met the configured limits.")
        if (
            rewrite_outcome.report.state == "unsupported"
            and rewrite_outcome.report.backend != "print-prompt"
        ):
            # print-prompt deliberately uses ``unsupported`` to mean that it
            # rendered a prompt without producing a rewrite.  A model-backed
            # unsupported result, however, is a failed rewrite and must be
            # visible to callers through the documented nonzero exit status.
            reason = rewrite_outcome.report.stopping_reason
            raise UnsupportedError(reason or "Model-backed rewrite was unsupported.")
        return

    outcome = edit_document(request)
    _emit(
        console,
        report_obj=outcome.report,
        rendered=render_edit(outcome.report),
        stdout_text=outcome.stdout_text,
        output_format=request.output_format,
        quiet=quiet,
        report_path=report,
        diff_text=outcome.report.diff,
    )


def _emit(
    console: object,
    *,
    report_obj: object,
    rendered: str,
    stdout_text: str | None,
    output_format: str,
    quiet: bool,
    report_path: Path | None,
    diff_text: str | None = None,
) -> None:
    """Emit a report the same way for both edit and rewrite runs."""
    from rich.console import Console

    assert isinstance(console, Console)
    # Machine-consumable text goes to stdout only when the user asked for it.
    if stdout_text is not None:
        emit_stdout(stdout_text)

    if output_format == "json":
        # With text already on stdout, the JSON report goes to stderr so the
        # pipeline stays usable; --report writes a clean file either way.
        if stdout_text is not None:
            emit_diagnostic(console, to_json(report_obj))
        else:
            emit_stdout(to_json(report_obj) + "\n")
    elif not quiet:
        emit_diagnostic(console, rendered)
        if diff_text:
            emit_diagnostic(console, diff_text)

    if report_path is not None:
        atomic_write_text(report_path, to_json(report_obj) + "\n", force=True)


def _rewrite_overrides(
    *,
    intensity: str | None,
    strategy: str | None,
    backend: str | None,
    model: str | None,
    fallback_model: list[str] | None,
    provider_only: list[str] | None,
    endpoint: str | None,
    source_provider: str | None,
    rewrite_provider: str | None,
    style: str | None,
    strength: str | None,
    candidates: int | None,
    temperature: float | None,
    target_length_ratio: float | None,
    max_output_tokens: int | None,
    rounds: int | None,
    early_stop_patience: int | None,
    seed: int | None,
    allow_remote: bool,
    key_env: str | None,
) -> dict[str, object]:
    """Validate and collect rewrite CLI options into a config-override mapping.

    An empty mapping means no rewrite option was supplied. ``--intensity`` is the
    single primary dial: it expands to a baseline set of rewrite settings that any
    explicit advanced flag still overrides. ``key_env`` names an environment
    variable; the API key itself is never accepted on the command line.
    """
    overrides: dict[str, object] = {}
    if strategy is not None:
        if strategy not in _REWRITE_STRATEGIES:
            raise typer.BadParameter("must be one-shot or recursive", param_hint="--strategy")
        overrides["strategy"] = strategy
    if backend is not None:
        if backend not in _REWRITE_BACKENDS:
            raise typer.BadParameter(
                "must be print-prompt, ollama, or openai-compatible", param_hint="--backend"
            )
        overrides["backend"] = backend
    if model is not None:
        overrides["model"] = model
    if fallback_model:
        overrides["fallback_models"] = tuple(fallback_model)
    if provider_only:
        overrides["provider_only"] = tuple(provider_only)
    if endpoint is not None:
        overrides["endpoint"] = endpoint
    if source_provider is not None:
        overrides["source_provider"] = source_provider
    if rewrite_provider is not None:
        overrides["rewrite_provider"] = rewrite_provider
    if style is not None:
        if style not in _REWRITE_STYLES:
            raise typer.BadParameter(
                f"must be one of {', '.join(sorted(_REWRITE_STYLES))}", param_hint="--rewrite-style"
            )
        overrides["style"] = style
    if strength is not None:
        if strength not in _REWRITE_STRENGTHS:
            raise typer.BadParameter(
                "must be light, medium, or strong", param_hint="--rewrite-strength"
            )
        overrides["strength"] = strength
    if candidates is not None:
        overrides["candidate_count"] = candidates
    if temperature is not None:
        overrides["temperature"] = temperature
    if target_length_ratio is not None:
        overrides["target_length_ratio"] = target_length_ratio
    if max_output_tokens is not None:
        overrides["max_output_tokens"] = max_output_tokens
    if rounds is not None:
        overrides["rounds"] = rounds
    if early_stop_patience is not None:
        overrides["early_stop_patience"] = early_stop_patience
    if seed is not None:
        overrides["seed"] = seed
    if allow_remote:
        overrides["allow_remote"] = True
    if key_env is not None:
        overrides["key_env"] = key_env

    if intensity is not None:
        if intensity not in _REWRITE_INTENSITIES:
            raise typer.BadParameter(
                "must be low, medium, or high",
                param_hint="--intensity",
            )
        # The dial sets a baseline; any explicit advanced flag above wins, so the
        # profile fills only the keys the user did not pin themselves.
        profile = intensity_profile(intensity)
        overrides = {**profile, **overrides}
    return overrides


def _unicode_policy(value: str | None) -> UnicodePolicyName | None:
    if value is None:
        return None
    allowed = {"report", "safe", "typographic", "aggressive"}
    if value not in allowed:
        raise typer.BadParameter(
            f"must be one of {', '.join(sorted(allowed))}", param_hint="--unicode-policy"
        )
    return cast(UnicodePolicyName, value)


def _diff(value: str) -> DiffMode:
    allowed = {"none", "unified", "operations"}
    if value not in allowed:
        raise typer.BadParameter(
            f"must be one of {', '.join(sorted(allowed))}", param_hint="--diff"
        )
    return cast(DiffMode, value)
