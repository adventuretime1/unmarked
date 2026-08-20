"""Read-only inspection service."""

from __future__ import annotations

from pathlib import Path

from unmark.application.requests import InspectRequest
from unmark.core.errors import ResearchModeRequiredError
from unmark.core.policies import FidelityPolicy, UnicodePolicy
from unmark.inspect.ingest import load_document
from unmark.orchestration.config import ResolvedConfig, resolve_config
from unmark.reporting.build import build_inspect_report
from unmark.reporting.schema import InspectReport


def _fidelity_from_config(resolved: ResolvedConfig) -> FidelityPolicy:
    fidelity = resolved.config.fidelity
    return FidelityPolicy(
        level=fidelity.level,
        lock_numbers=fidelity.lock_numbers,
        lock_citations=fidelity.lock_citations,
        lock_quotes=fidelity.lock_quotes,
        lock_urls=fidelity.lock_urls,
        lock_code=fidelity.lock_code,
        lock_dates=fidelity.lock_dates,
        lock_units=fidelity.lock_units,
        lock_identifiers=fidelity.lock_identifiers,
    )


def unicode_policy_from_config(resolved: ResolvedConfig, research_mode: bool) -> UnicodePolicy:
    """Build a Unicode policy, refusing ``aggressive`` without research mode."""
    unicode_config = resolved.config.unicode
    if unicode_config.policy == "aggressive" and not research_mode:
        msg = (
            "the 'aggressive' Unicode policy is research-only: it removes format "
            "characters regardless of script context and can corrupt legitimate "
            "text. Pass --research-mode to acknowledge this. Note that --yes does "
            "not enable it."
        )
        raise ResearchModeRequiredError(msg)
    return UnicodePolicy(
        name=unicode_config.policy,
        preserve_emoji_sequences=unicode_config.preserve_emoji_sequences,
        preserve_language_controls=unicode_config.preserve_language_controls,
        normalize_spaces=unicode_config.normalize_spaces,
        normalization_form=unicode_config.normalization_form,
    )


def inspect_document(request: InspectRequest) -> InspectReport:
    """Inspect a document. Never mutates anything on disk."""
    start = Path(request.input).parent if request.input != "-" else Path.cwd()
    resolved = resolve_config(
        cli_overrides={"unicode": {"policy": request.unicode_policy}},
        explicit_config=request.config_path,
        start_dir=start,
    )
    policy = unicode_policy_from_config(resolved, request.research_mode)
    fidelity = _fidelity_from_config(resolved)

    locks = tuple(resolved.config.fidelity.locks) + request.locks
    document = load_document(
        request.input,
        media_type=request.media_type,
        fidelity=fidelity,
        locks=locks,
        stdin=request.stdin,
        max_chars=resolved.config.budget.max_input_chars,
    )

    from unmark.inspect.unicode_scan import inspect_text

    findings = inspect_text(document.source_text, policy)
    return build_inspect_report(
        document,
        policy.name,
        findings,
        resolved.config.model_dump(mode="json"),
    )
