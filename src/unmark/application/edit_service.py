"""Editing service.

This owns the whole edit lifecycle: config resolution, ingestion, sanitation,
budget enforcement, output commitment, and the run record. The CLI only maps
arguments in and renders the result out.

Ordering matters for safety. The destination is validated *before* any work, the
run directory is written as the run progresses, and the terminal report is written
last, so a valid ``report.json`` means the run really finished.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from unmark.application.inspect_service import (
    _fidelity_from_config,
    unicode_policy_from_config,
)
from unmark.application.requests import EditRequest
from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.diffs import operation_diff, unified_diff
from unmark.core.document import Document
from unmark.core.errors import UnsupportedError, UsageError
from unmark.core.events import EventRecorder
from unmark.core.results import ResultState
from unmark.inspect.ingest import STDIN_SENTINEL, load_document
from unmark.orchestration.config import ResolvedConfig, resolve_config
from unmark.orchestration.presets import get_preset
from unmark.orchestration.sanitation import SanitationStrategy
from unmark.reporting.build import build_edit_report
from unmark.reporting.schema import EditReport
from unmark.storage.atomic import atomic_write_text, check_destination, default_output_path
from unmark.storage.run_store import RunStore, new_run_id

STDOUT_SENTINEL = "-"


@dataclass
class EditOutcome:
    """What the CLI needs to render and to decide what to print to stdout."""

    report: EditReport
    candidate_text: str
    stdout_text: str | None
    run_id: str


def _resolve_destination(request: EditRequest, document: Document) -> Path | None:
    """Decide where output goes. ``None`` means stdout."""
    if request.output == STDOUT_SENTINEL:
        return None
    if request.output is not None:
        return Path(request.output)
    if document.origin == STDIN_SENTINEL:
        # stdin has no sibling to derive a name from, so stdout is the only
        # sensible default; the user can still pass --output explicitly.
        return None
    return default_output_path(Path(document.origin))


def _budget_from_config(resolved: ResolvedConfig) -> RunBudget:
    budget = resolved.config.budget
    return RunBudget(
        max_runtime_ms=budget.max_runtime_ms,
        max_char_edit_ratio=budget.max_char_edit_ratio,
        max_token_edit_ratio=budget.max_token_edit_ratio,
        max_length_drift_ratio=budget.max_length_drift_ratio,
        max_candidates=budget.max_candidates,
        max_rounds=budget.max_rounds,
        max_input_chars=budget.max_input_chars,
        # This slice makes no model or detector calls, so those budgets are zero
        # and any attempt to reserve against them fails loudly.
        max_model_calls=0,
        max_detector_queries=0,
    )


def edit_document(request: EditRequest) -> EditOutcome:
    """Run an edit and return the outcome."""
    started = datetime.now(UTC)
    clock = time.monotonic()

    preset = get_preset(request.preset)

    overrides: dict[str, object] = {
        "preset": preset.name,
        # These command-line switches control the run even though they do not
        # alter sanitation itself.  Include them in the resolved configuration
        # so an evidence report describes what actually happened.
        "research_mode": request.research_mode,
        "output": {"format": request.output_format},
    }
    if request.unicode_policy is not None:
        overrides["unicode"] = {"policy": request.unicode_policy}
    if request.diff != "none":
        output = overrides["output"]
        assert isinstance(output, dict)
        output["diff"] = request.diff

    start_dir = Path(request.input).parent if request.input != STDIN_SENTINEL else Path.cwd()
    resolved = resolve_config(
        cli_overrides=overrides,
        explicit_config=request.config_path,
        start_dir=start_dir,
        preset_name=preset.name,
    )

    unicode_policy = unicode_policy_from_config(resolved, request.research_mode)
    if unicode_policy.is_research_only and preset.name != "sanitize":
        msg = "the aggressive Unicode policy is not reachable from a normal preset"
        raise UsageError(msg)

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

    if document.media_type not in {"text/plain", "text/markdown"}:
        msg = f"unsupported media type: {document.media_type}"
        raise UnsupportedError(msg)

    destination = _resolve_destination(request, document)
    source_path = Path(document.origin) if document.origin != STDIN_SENTINEL else None

    # Validate the destination before doing any work, so a refused write costs
    # nothing and cannot leave a half-finished run behind.
    if destination is not None and not request.dry_run:
        check_destination(destination, source=source_path, force=request.force)

    run_id = new_run_id(started)
    recorder = EventRecorder(run_id)
    recorder.state("created")
    recorder.state("inspecting", f"parsed {len(document.blocks)} blocks")

    budget = BudgetAccount(_budget_from_config(resolved))
    strategy = SanitationStrategy(unicode_policy)

    recorder.state("running", f"strategy {strategy.descriptor.id}")
    outcome = strategy.execute(document, budget, fidelity)
    recorder.state("validating", f"{len(outcome.operations)} operations")

    for operation in outcome.operations:
        recorder.record(
            "operation",
            operation.reason,
            offset=operation.start,
            operator=operation.operator,
        )

    diff_mode = resolved.config.output.diff if request.diff == "none" else request.diff
    diff_text: str | None = None
    if diff_mode == "unified":
        diff_text = unified_diff(
            document.source_text,
            outcome.candidate_text,
            source_label=f"a/{Path(document.origin).name}",
            candidate_label=f"b/{Path(document.origin).name}",
        )
    elif diff_mode == "operations":
        diff_text = operation_diff(document.source_text, outcome.operations)

    elapsed_ms = int((time.monotonic() - clock) * 1000)
    with budget.reserve("runtime_ms", elapsed_ms) as lease:
        lease.settle(elapsed_ms)

    usage = budget.usage(
        char_edit_ratio=outcome.candidate.char_edit_ratio,
        token_edit_ratio=outcome.candidate.token_edit_ratio,
        length_drift_ratio=outcome.candidate.length_drift_ratio,
    )

    state: ResultState = "sanitized"
    notes: list[str] = []
    if not outcome.operations:
        notes.append("No actionable Unicode findings; the document is unchanged.")
    if outcome.blocked:
        notes.append(
            f"{len(outcome.blocked)} finding(s) were left in place because they fall "
            "inside a protected span."
        )
    if request.dry_run:
        notes.append("Dry run: nothing was written.")

    # Commit output before the terminal report, so the report can record the
    # committed path and hash truthfully.
    output_path_str: str | None = None
    stdout_text: str | None = None
    if request.dry_run:
        stdout_text = None
    elif destination is None:
        stdout_text = outcome.candidate_text
        output_path_str = STDOUT_SENTINEL
    else:
        atomic_write_text(
            destination,
            outcome.candidate_text,
            source=source_path,
            force=request.force,
        )
        output_path_str = str(destination)
        recorder.record("progress", f"wrote {destination}")

    report = build_edit_report(
        run_id=run_id,
        state=state,
        preset=preset.name,
        dry_run=request.dry_run,
        document=document,
        policy_name=unicode_policy.name,
        findings=outcome.findings,
        blocked=outcome.blocked,
        operations=outcome.operations,
        candidate_text=outcome.candidate_text,
        output_path=output_path_str,
        diff=diff_text,
        usage=usage,
        effective_config=resolved.config.model_dump(mode="json"),
        notes=tuple(notes),
    )

    if request.retain_run and not request.dry_run:
        _persist_run(
            request,
            resolved,
            document,
            outcome.candidate_text,
            report,
            recorder,
            run_id,
            diff_text,
        )

    if request.report_path is not None:
        atomic_write_text(
            request.report_path,
            report.model_dump_json(indent=2) + "\n",
            source=source_path,
            force=True,
        )

    recorder.state("completed", state)
    return EditOutcome(
        report=report,
        candidate_text=outcome.candidate_text,
        stdout_text=stdout_text,
        run_id=run_id,
    )


def _persist_run(
    request: EditRequest,
    resolved: ResolvedConfig,
    document: Document,
    candidate_text: str,
    report: EditReport,
    recorder: EventRecorder,
    run_id: str,
    diff_text: str | None,
) -> None:
    """Write the run directory. The terminal report is written last."""
    workspace = request.workspace or Path.cwd()
    store = RunStore(workspace)
    store.create(run_id)
    store.write_json(
        run_id,
        "request.json",
        request.model_dump(mode="json", exclude={"stdin"}),
    )
    store.write_json(run_id, "effective-config.json", resolved.config.model_dump(mode="json"))
    store.write_text(run_id, "source.sha256", document.source_sha256 + "\n")
    store.append_events(run_id, recorder.events)
    if resolved.config.output.retain_output:
        store.write_text(run_id, "output.txt", candidate_text)
    if diff_text:
        store.write_text(run_id, "diff.patch", diff_text)
    store.write_json(run_id, "report.json", report)
