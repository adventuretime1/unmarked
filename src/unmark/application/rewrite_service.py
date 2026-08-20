"""Rewrite service: prompt-driven text rewriting for ``unmark edit``.

This is a sibling of :mod:`unmark.application.edit_service`; it reuses the same
config resolution, ingestion, atomic-write, and run-store machinery but runs a
prompt-driven rewrite strategy instead of deterministic sanitation.

Security posture is inherited from the model adapters and enforced here at the
boundary:

* the backend defaults to ``print-prompt`` (no network);
* a networked backend requires an explicit remote opt-in *and* a non-zero model
  budget; both are refused by default;
* an API key is read only from a named environment variable, never from a CLI
  argument or config value;
* the rendered prompt, report, and run store never carry a credential.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from unmark.application.edit_service import STDOUT_SENTINEL
from unmark.application.inspect_service import _fidelity_from_config, unicode_policy_from_config
from unmark.application.requests import EditRequest
from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.diffs import operation_diff, unified_diff
from unmark.core.document import Document
from unmark.core.errors import UnsupportedError, UsageError
from unmark.core.events import EventRecorder
from unmark.core.operations import char_edit_ratio, length_drift_ratio, normalized_content_hash
from unmark.core.policies import FidelityPolicy
from unmark.core.targets import ReductionTarget
from unmark.inspect.ingest import STDIN_SENTINEL, build_document, load_document
from unmark.models.local import PrintPromptAdapter
from unmark.models.protocols import ModelAdapter
from unmark.models.remote import (
    LOOPBACK_HOSTS,
    KeyResolver,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    env_key_resolver,
)
from unmark.orchestration.config import ResolvedConfig, RewriteConfigSection, resolve_config
from unmark.orchestration.sanitation import SanitationOutcome, SanitationStrategy
from unmark.reporting.build import build_rewrite_report, build_unicode_summary
from unmark.reporting.schema import CanonicalizationStageSummary, RewriteReport
from unmark.storage.atomic import atomic_write_text, check_destination, default_output_path
from unmark.storage.run_store import RunStore, new_run_id
from unmark.storage.voice_store import resolve_voice
from unmark.strategies.rewrite.config import RewriteConfig
from unmark.strategies.rewrite.engine import diff_operations
from unmark.strategies.rewrite.one_shot import OneShotRewriteStrategy
from unmark.strategies.rewrite.recursive import RecursiveRewriteStrategy
from unmark.strategies.rewrite.result import RewriteResult

_REWRITE_EDIT_BUDGETS: dict[str, tuple[float, float, float]] = {
    "light": (0.20, 0.25, 0.10),
    "medium": (0.50, 0.55, 0.15),
    "strong": (0.70, 0.75, 0.25),
}

_PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "chatgpt": "openai",
    "codex": "openai",
    "google": "google",
    "gemini": "google",
    "local": "local",
    "ollama": "local",
    "meta": "meta-llama",
    "llama": "meta-llama",
    "meta-llama": "meta-llama",
}

_ENDPOINT_PROVIDERS = {
    "api.anthropic.com": "anthropic",
    "api.openai.com": "openai",
    "generativelanguage.googleapis.com": "google",
}


def normalize_provider(value: str) -> str:
    """Return a stable provider-family name for comparison and reports."""
    normalized = value.strip().lower().replace("_", "-")
    if "/" in normalized:
        normalized = normalized.split("/", maxsplit=1)[0]
    return _PROVIDER_ALIASES.get(normalized, normalized)


def _derived_rewrite_provider(section: RewriteConfigSection) -> str:
    if section.backend == "print-prompt":
        return "none"
    if section.backend == "ollama":
        return "local"
    if "/" in section.model:
        return normalize_provider(section.model)
    hostname = urlsplit(section.endpoint).hostname or ""
    return _ENDPOINT_PROVIDERS.get(hostname.lower(), "")


def _resolve_provider_separation(section: RewriteConfigSection) -> tuple[str, str]:
    """Resolve provenance and refuse same-provider model calls.

    This is an execution boundary, not watermark detection. A caller may use
    ``human`` or ``unknown`` when no model provider is attributable, but must
    still state that provenance explicitly for every model-backed rewrite.
    """
    source_provider = normalize_provider(section.source_provider)
    derived_provider = _derived_rewrite_provider(section)
    configured_provider = normalize_provider(section.rewrite_provider)

    if section.backend == "print-prompt":
        return source_provider, configured_provider or derived_provider
    if not source_provider:
        msg = (
            "model-backed rewriting requires rewrite.source_provider; set "
            "--source-provider to where the input came from (or human/unknown)"
        )
        raise UsageError(msg)
    if configured_provider and derived_provider and configured_provider != derived_provider:
        msg = (
            f"rewrite provider {configured_provider!r} conflicts with provider "
            f"{derived_provider!r} derived from backend/model"
        )
        raise UsageError(msg)
    rewrite_provider = configured_provider or derived_provider
    if not rewrite_provider:
        msg = (
            "could not determine rewrite provider from this model endpoint; "
            "set --rewrite-provider explicitly"
        )
        raise UsageError(msg)
    if source_provider == rewrite_provider:
        msg = (
            f"source and rewrite providers both resolve to {source_provider!r}; "
            "choose a model from a different provider"
        )
        raise UsageError(msg)
    if source_provider not in {"human", "unknown"}:
        overlapping_fallbacks = tuple(
            model
            for model in section.fallback_models
            if "/" in model and normalize_provider(model) == source_provider
        )
        if overlapping_fallbacks:
            joined = ", ".join(overlapping_fallbacks)
            raise UsageError(
                f"fallback model provider overlaps source provider {source_provider!r}: {joined}; "
                "remove same-provider fallbacks"
            )
    return source_provider, rewrite_provider


@dataclass
class RewriteOutcome:
    report: RewriteReport
    candidate_text: str
    stdout_text: str | None
    run_id: str


def _resolve_destination(request: EditRequest, document: Document) -> Path | None:
    if request.output == STDOUT_SENTINEL:
        return None
    if request.output is not None:
        return Path(request.output)
    if document.origin == STDIN_SENTINEL:
        return None
    return default_output_path(Path(document.origin))


def _rewrite_budget(resolved: ResolvedConfig) -> RunBudget:
    budget = resolved.config.budget
    section = resolved.config.rewrite

    # Sanitation's 2% character cap and the package's 12% token cap make normal
    # paraphrases abstain by construction. Replace only built-in/preset values
    # with strength-appropriate rewrite defaults. A value supplied by user,
    # project, explicit config, or CLI remains a hard limit.
    default_char, default_token, default_length = _REWRITE_EDIT_BUDGETS[section.strength]

    def effective(name: str, current: float, rewrite_default: float) -> float:
        source = resolved.sources.get(f"budget.{name}", "package defaults")
        if source == "package defaults" or source.startswith("preset "):
            return rewrite_default
        return current

    max_char_edit_ratio = effective("max_char_edit_ratio", budget.max_char_edit_ratio, default_char)
    max_token_edit_ratio = effective(
        "max_token_edit_ratio", budget.max_token_edit_ratio, default_token
    )
    max_length_drift_ratio = effective(
        "max_length_drift_ratio", budget.max_length_drift_ratio, default_length
    )

    # A networked backend needs a non-zero model-call budget to run at all. When
    # the operator has not set one explicitly, derive it from the requested
    # rounds and candidate count so a normal rewrite works without hand-tuning
    # budgets; a print-prompt run stays at zero and makes no calls. The
    # remote-network gate is still the adapter's allow_remote check, not this.
    max_model_calls = budget.max_model_calls
    if section.backend != "print-prompt" and max_model_calls == 0:
        hops = section.rounds if section.strategy == "recursive" else 1
        max_model_calls = hops * section.candidate_count

    return RunBudget(
        max_runtime_ms=budget.max_runtime_ms,
        max_cost_usd=budget.max_cost_usd,
        max_model_calls=max_model_calls,
        max_detector_queries=budget.max_detector_queries,
        max_char_edit_ratio=max_char_edit_ratio,
        max_token_edit_ratio=max_token_edit_ratio,
        max_length_drift_ratio=max_length_drift_ratio,
        max_candidates=budget.max_candidates,
        max_rounds=budget.max_rounds,
        max_input_chars=budget.max_input_chars,
    )


def _build_adapter(section: RewriteConfigSection) -> ModelAdapter:
    """Construct the configured backend, enforcing the remote opt-in.

    ``print-prompt`` needs nothing. A networked backend is refused unless
    ``allow_remote`` is set (loopback Ollama excepted, which the adapter permits
    on its own). API keys are resolved from ``key_env`` at call time only.
    """
    if section.backend == "print-prompt":
        return PrintPromptAdapter()
    if section.backend == "ollama":
        base_url = section.endpoint or "http://127.0.0.1:11434"
        if not section.model:
            msg = "the ollama backend requires rewrite.model"
            raise UsageError(msg)
        return OllamaAdapter(
            model=section.model,
            base_url=base_url,
            allow_remote=section.allow_remote,
        )
    # The only remaining Literal value is openai-compatible.
    if not section.endpoint:
        msg = "the openai-compatible backend requires rewrite.endpoint"
        raise UsageError(msg)
    if not section.model:
        msg = "the openai-compatible backend requires rewrite.model"
        raise UsageError(msg)
    resolver = _openai_key_resolver(section)
    return OpenAICompatibleAdapter(
        base_url=section.endpoint,
        model=section.model,
        key_resolver=resolver,
        allow_remote=section.allow_remote,
        fallback_models=section.fallback_models,
        provider_only=section.provider_only,
    )


def _openai_key_resolver(section: RewriteConfigSection) -> KeyResolver | None:
    """Return a configured key resolver without retaining the credential.

    Remote OpenAI-compatible endpoints must name a credential environment
    variable. The value is checked now for an actionable preflight error and read
    again by the adapter at request time; it is never copied into configuration,
    reports, rendered prompts, or exception text.
    """
    hostname = (urlsplit(section.endpoint).hostname or "").lower()
    is_remote = hostname not in LOOPBACK_HOSTS
    if is_remote and not section.key_env:
        raise UsageError(
            "a remote openai-compatible endpoint requires rewrite.key_env; set "
            "--key-env to the name of an environment variable such as OPENROUTER_API_KEY"
        )
    if not section.key_env:
        return None
    resolver = env_key_resolver(section.key_env)
    if resolver() is None:
        raise UsageError(
            f"credential environment variable {section.key_env!r} is not set or is empty; "
            "set it outside the command and do not pass the API key itself"
        )
    return resolver


def check_rewrite_configuration(section: RewriteConfigSection) -> dict[str, object]:
    """Validate model-backed rewrite readiness without sending text or a request."""
    if section.backend == "print-prompt":
        raise UsageError(
            "rewrite backend is print-prompt; configure ollama or openai-compatible "
            "before requesting a generated rewrite"
        )
    source_provider, rewrite_provider = _resolve_provider_separation(section)
    adapter = _build_adapter(section)
    hostname = (urlsplit(section.endpoint).hostname or "").lower()
    credential_required = section.backend == "openai-compatible" and hostname not in LOOPBACK_HOSTS
    return {
        "ready": True,
        "backend": section.backend,
        "adapter": adapter.id,
        "endpoint": section.endpoint or "http://127.0.0.1:11434",
        "model": section.model,
        "source_provider": source_provider,
        "rewrite_provider": rewrite_provider,
        "fallback_models": list(section.fallback_models),
        "key_env": section.key_env or None,
        "credential_required": credential_required,
        "credential_configured": bool(section.key_env),
        "remote_enabled": section.allow_remote,
    }


def _rewrite_config(section: RewriteConfigSection, voice: str = "") -> RewriteConfig:
    return RewriteConfig(
        voice=voice,
        style=section.style,
        strength=section.strength,
        candidate_count=section.candidate_count,
        temperature=section.temperature,
        target_length_ratio=section.target_length_ratio,
        max_output_tokens=section.max_output_tokens,
        seed=section.seed,
        rounds=section.rounds,
        style_schedule=section.style_schedule,
        early_stop_patience=section.early_stop_patience,
    )


def _run_strategy(
    section: RewriteConfigSection,
    adapter: ModelAdapter,
    config: RewriteConfig,
    document: Document,
    budget: BudgetAccount,
    policy: FidelityPolicy,
) -> RewriteResult:
    target = ReductionTarget(mode="sanitize_only")
    if section.strategy == "recursive":
        strategy = RecursiveRewriteStrategy(adapter=adapter, config=config)
        return strategy.run(document, target, budget, policy)
    one_shot = OneShotRewriteStrategy(adapter=adapter, config=config)
    return one_shot.run(document, target, budget, policy)


def _canonicalize(
    document: Document,
    resolved: ResolvedConfig,
    policy: FidelityPolicy,
    *,
    research_mode: bool,
) -> SanitationOutcome:
    """Run deterministic carrier cleanup without consuming rewrite call budgets."""
    unicode_policy = unicode_policy_from_config(resolved, research_mode=research_mode)
    budget = BudgetAccount(
        RunBudget(
            max_candidates=1,
            max_input_chars=max(1, len(document.source_text)),
        )
    )
    return SanitationStrategy(unicode_policy).execute(document, budget, policy)


def _canonicalization_stage(
    stage: Literal["input", "output"],
    resolved: ResolvedConfig,
    outcome: SanitationOutcome,
) -> CanonicalizationStageSummary:
    return CanonicalizationStageSummary(
        stage=stage,
        unicode=build_unicode_summary(
            resolved.config.unicode.policy,
            outcome.findings,
            outcome.blocked,
        ),
        operations=outcome.operations,
        operation_count=len(outcome.operations),
    )


def rewrite_document(request: EditRequest) -> RewriteOutcome:
    """Run a prompt-driven rewrite and return the outcome."""
    started = datetime.now(UTC)
    clock = time.monotonic()

    overrides: dict[str, object] = {
        "preset": request.preset,
        # These options govern the actual run, so record their command-line
        # values in the effective configuration emitted with the evidence.
        "research_mode": request.research_mode,
        "output": {"format": request.output_format},
    }
    if request.unicode_policy is not None:
        overrides["unicode"] = {"policy": request.unicode_policy}
    if request.diff != "none":
        output = overrides["output"]
        assert isinstance(output, dict)
        output["diff"] = request.diff
    if request.rewrite_overrides:
        # Scalar rewrite settings from the CLI; a key-like field would be refused
        # by config validation, and no API key is ever routed through here.
        overrides["rewrite"] = dict(request.rewrite_overrides)

    start_dir = Path(request.input).parent if request.input != STDIN_SENTINEL else Path.cwd()
    resolved = resolve_config(
        cli_overrides=overrides,
        explicit_config=request.config_path,
        start_dir=start_dir,
        preset_name=request.preset,
    )
    section = resolved.config.rewrite
    source_provider, rewrite_provider = _resolve_provider_separation(section)
    section = section.model_copy(
        update={
            "source_provider": source_provider,
            "rewrite_provider": rewrite_provider,
        }
    )
    resolved = replace(
        resolved,
        config=resolved.config.model_copy(update={"rewrite": section}),
    )

    # A rewrite cannot prove semantic equivalence without a model gate, so the
    # entailment gate is not treated as hard here; every deterministic lock
    # (numbers, URLs, quotes, code, protected/locked spans, structure) stays hard.
    fidelity = _fidelity_from_config(resolved).model_copy(
        update={"require_bidirectional_entailment": False}
    )
    locks = tuple(resolved.config.fidelity.locks) + request.locks

    document = load_document(
        request.input,
        media_type=request.media_type,
        fidelity=fidelity,
        locks=locks,
        stdin=request.stdin,
        max_chars=resolved.config.budget.max_input_chars,
    )
    if document.media_type not in {"text/plain", "text/markdown"}:
        msg = f"unsupported media type: {document.media_type}"
        raise UnsupportedError(msg)

    destination = _resolve_destination(request, document)
    source_path = Path(document.origin) if document.origin != STDIN_SENTINEL else None
    if destination is not None and not request.dry_run:
        check_destination(destination, source=source_path, force=request.force)

    run_id = new_run_id(started)
    recorder = EventRecorder(run_id)
    recorder.state("created")
    recorder.state("inspecting", f"parsed {len(document.blocks)} blocks")

    input_cleanup = _canonicalize(
        document,
        resolved,
        fidelity,
        research_mode=request.research_mode,
    )
    working_document = document
    if input_cleanup.candidate_text != document.source_text:
        working_document = build_document(
            input_cleanup.candidate_text,
            document.media_type,
            origin=document.origin,
            fidelity=fidelity,
            locks=locks,
        )
        recorder.record(
            "progress",
            f"canonicalized {len(input_cleanup.operations)} input carrier operation(s)",
        )

    adapter = _build_adapter(section)
    voice = resolve_voice(request.voice).description if request.voice else ""
    config = _rewrite_config(section, voice)
    budget = BudgetAccount(_rewrite_budget(resolved))

    recorder.state("running", f"strategy rewrite-{section.strategy} backend {adapter.id}")
    result = _run_strategy(section, adapter, config, working_document, budget, fidelity)
    recorder.state("validating", result.state)

    raw_candidate = result.selected.text if result.selected else working_document.source_text
    output_document = build_document(
        raw_candidate,
        document.media_type,
        origin=document.origin,
        fidelity=fidelity,
        locks=locks,
    )
    output_cleanup = _canonicalize(
        output_document,
        resolved,
        fidelity,
        research_mode=request.research_mode,
    )
    candidate_text = output_cleanup.candidate_text

    if result.selected is not None:
        final_operations = diff_operations(
            document.source_text,
            candidate_text,
            operator=f"rewrite-{section.strategy}:canonicalized",
        )
        selected = result.selected.model_copy(
            update={
                "text": candidate_text,
                "operations": final_operations,
                "content_hash": normalized_content_hash(candidate_text),
                "char_edit_ratio": char_edit_ratio(
                    document.source_text,
                    candidate_text,
                    final_operations,
                ),
                "length_drift_ratio": length_drift_ratio(
                    document.source_text,
                    candidate_text,
                ),
            }
        )
        result = result.model_copy(
            update={
                "selected": selected,
                "usage": result.usage.model_copy(
                    update={
                        "char_edit_ratio": selected.char_edit_ratio,
                        "length_drift_ratio": selected.length_drift_ratio,
                    }
                ),
            }
        )

    diff_mode = resolved.config.output.diff if request.diff == "none" else request.diff
    diff_text: str | None = None
    if candidate_text != document.source_text and diff_mode == "unified":
        diff_text = unified_diff(
            document.source_text,
            candidate_text,
            source_label=f"a/{Path(document.origin).name}",
            candidate_label=f"b/{Path(document.origin).name}",
        )
    elif candidate_text != document.source_text and diff_mode == "operations":
        operations = (
            result.selected.operations if result.selected is not None else input_cleanup.operations
        )
        diff_text = operation_diff(document.source_text, operations)

    elapsed_ms = int((time.monotonic() - clock) * 1000)
    remaining = int(max(Decimal(0), budget.remaining("runtime_ms")))
    settle_ms = min(elapsed_ms, remaining)
    if settle_ms:
        with budget.reserve("runtime_ms", settle_ms) as lease:
            lease.settle(settle_ms)

    notes_list = list(_notes(request, result))
    if input_cleanup.operations:
        notes_list.append(
            f"canonicalized {len(input_cleanup.operations)} input Unicode operation(s)"
        )
    if output_cleanup.operations:
        notes_list.append(
            f"canonicalized {len(output_cleanup.operations)} output Unicode operation(s)"
        )
    notes = tuple(notes_list)

    # Only commit output when a rewrite was actually selected and it is not a dry
    # run. print-prompt and abstained runs write nothing.
    output_path_str: str | None = None
    stdout_text: str | None = None
    canonicalized_only = (
        result.selected is None
        and result.trace.prompt_only is None
        and candidate_text != document.source_text
    )
    committed = (result.selected is not None or canonicalized_only) and not request.dry_run
    if committed and destination is None:
        stdout_text = candidate_text
        output_path_str = STDOUT_SENTINEL
    elif committed and destination is not None:
        atomic_write_text(destination, candidate_text, source=source_path, force=request.force)
        output_path_str = str(destination)
        recorder.record("progress", f"wrote {destination}")

    report = build_rewrite_report(
        run_id=run_id,
        result=result,
        strategy=f"rewrite-{section.strategy}",
        backend=adapter.id,
        dry_run=request.dry_run,
        document=document,
        candidate_text=candidate_text,
        output_path=output_path_str,
        diff=diff_text,
        effective_config=resolved.config.model_dump(mode="json"),
        canonicalization=(
            _canonicalization_stage("input", resolved, input_cleanup),
            _canonicalization_stage("output", resolved, output_cleanup),
        ),
        notes=notes,
    )

    if request.retain_run and not request.dry_run:
        _persist_run(
            request, resolved, document, candidate_text, report, recorder, run_id, diff_text
        )

    if request.report_path is not None:
        atomic_write_text(
            request.report_path,
            report.model_dump_json(indent=2) + "\n",
            source=source_path,
            force=True,
        )

    recorder.state("completed", result.state)
    return RewriteOutcome(
        report=report,
        candidate_text=candidate_text,
        stdout_text=stdout_text,
        run_id=run_id,
    )


def _persist_run(
    request: EditRequest,
    resolved: ResolvedConfig,
    document: Document,
    candidate_text: str,
    report: RewriteReport,
    recorder: EventRecorder,
    run_id: str,
    diff_text: str | None,
) -> None:
    """Write the run directory. The terminal report is written last.

    Only committed rewrite output is retained; a print-prompt or abstained run
    selects nothing, so no ``output.txt`` is written. No credential is persisted:
    the request excludes stdin and the config never carries a key.
    """
    workspace = request.workspace or Path.cwd()
    store = RunStore(workspace)
    store.create(run_id)
    store.write_json(run_id, "request.json", request.model_dump(mode="json", exclude={"stdin"}))
    store.write_json(run_id, "effective-config.json", resolved.config.model_dump(mode="json"))
    store.write_text(run_id, "source.sha256", document.source_sha256 + "\n")
    store.append_events(run_id, recorder.events)
    if resolved.config.output.retain_output and report.output_sha256 is not None:
        store.write_text(run_id, "output.txt", candidate_text)
    if diff_text:
        store.write_text(run_id, "diff.patch", diff_text)
    store.write_json(run_id, "report.json", report)


def _notes(request: EditRequest, result: RewriteResult) -> tuple[str, ...]:
    notes: list[str] = []
    if result.trace.prompt_only is not None:
        notes.append("print-prompt backend: the rendered prompt is shown; nothing was rewritten.")
    if result.state == "abstained":
        notes.append("No fidelity-valid rewrite was selected; the source is unchanged.")
    if request.dry_run:
        notes.append("Dry run: nothing was written.")
    return tuple(notes)
