"""Report construction."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from unmark.core.budgets import BudgetUsage
from unmark.core.document import Document
from unmark.core.operations import Operation, sha256_text
from unmark.core.results import ResultState
from unmark.inspect.scripts import document_scripts
from unmark.inspect.unicode_scan import UnicodeFinding
from unmark.reporting.schema import (
    CanonicalizationStageSummary,
    DocumentSummary,
    EditReport,
    InspectReport,
    RewriteHopSummary,
    RewriteReport,
    UnicodeSummary,
)
from unmark.strategies.rewrite.result import RewriteResult


def _version() -> str:
    from unmark import __version__

    return __version__


def build_document_summary(document: Document) -> DocumentSummary:
    return DocumentSummary(
        origin=document.origin,
        media_type=document.media_type,
        source_sha256=document.source_sha256,
        characters=len(document.source_text),
        lines=document.source_text.count("\n") + (1 if document.source_text else 0),
        scripts=tuple(sorted(document_scripts(document.source_text))),
        blocks=document.blocks,
        block_counts=dict(Counter(block.kind for block in document.blocks)),
        protected_spans=document.protected_spans,
        span_counts=dict(Counter(span.kind for span in document.protected_spans)),
    )


def build_unicode_summary(
    policy_name: str,
    findings: tuple[UnicodeFinding, ...],
    blocked: tuple[UnicodeFinding, ...] = (),
) -> UnicodeSummary:
    removable = sum(1 for f in findings if f.removable and f.replacement is not None)
    return UnicodeSummary(
        policy=policy_name,
        findings=findings,
        counts_by_kind=dict(Counter(f.kind for f in findings)),
        removable=removable,
        preserved=len(findings) - removable,
        blocked_by_protection=blocked,
    )


def build_inspect_report(
    document: Document,
    policy_name: str,
    findings: tuple[UnicodeFinding, ...],
    effective_config: dict[str, Any],
) -> InspectReport:
    return InspectReport(
        generated_at=datetime.now(UTC),
        tool_version=_version(),
        document=build_document_summary(document),
        unicode=build_unicode_summary(policy_name, findings),
        effective_config=effective_config,
    )


def build_edit_report(
    *,
    run_id: str,
    state: ResultState,
    preset: str,
    dry_run: bool,
    document: Document,
    policy_name: str,
    findings: tuple[UnicodeFinding, ...],
    blocked: tuple[UnicodeFinding, ...],
    operations: tuple[Operation, ...],
    candidate_text: str,
    output_path: str | None,
    diff: str | None,
    usage: BudgetUsage,
    effective_config: dict[str, Any],
    notes: tuple[str, ...] = (),
) -> EditReport:
    return EditReport(
        run_id=run_id,
        generated_at=datetime.now(UTC),
        tool_version=_version(),
        state=state,
        preset=preset,
        dry_run=dry_run,
        document=build_document_summary(document),
        unicode=build_unicode_summary(policy_name, findings, blocked),
        operations=operations,
        operation_count=len(operations),
        output_sha256=sha256_text(candidate_text),
        output_path=output_path,
        diff=diff,
        usage=usage,
        effective_config=effective_config,
        notes=notes,
    )


def build_rewrite_report(
    *,
    run_id: str,
    result: RewriteResult,
    strategy: str,
    backend: str,
    dry_run: bool,
    document: Document,
    candidate_text: str,
    output_path: str | None,
    diff: str | None,
    effective_config: dict[str, Any],
    canonicalization: tuple[CanonicalizationStageSummary, ...] = (),
    notes: tuple[str, ...] = (),
) -> RewriteReport:
    """Build a rewrite report from a :class:`RewriteResult`.

    The prompt preview is copied verbatim from the print-prompt trace; it never
    contains a credential (the rendered prompt is text only). Output hash and path
    are set only when a rewrite was actually committed.
    """
    trace = result.trace
    operations = result.selected.operations if result.selected else ()
    return RewriteReport(
        run_id=run_id,
        generated_at=datetime.now(UTC),
        tool_version=_version(),
        state=result.state,
        strategy=strategy,
        backend=backend,
        style=trace.style,
        dry_run=dry_run,
        document=build_document_summary(document),
        prompt_preview=trace.prompt_only,
        canonicalization=canonicalization,
        operations=operations,
        operation_count=len(operations),
        candidates_generated=trace.candidates_generated,
        candidates_rejected=trace.candidates_rejected,
        rejection_reasons=trace.rejection_reasons,
        hops=tuple(
            RewriteHopSummary(
                round_index=hop.round_index,
                style=hop.style,
                best_residual_risk=hop.best_residual_risk,
                accepted_count=hop.accepted_count,
                rejected_count=hop.rejected_count,
                note=hop.note,
            )
            for hop in trace.hops
        ),
        stopping_reason=trace.stopping_reason,
        selection_reason=trace.selection_reason,
        output_sha256=(
            sha256_text(candidate_text)
            if result.selected or candidate_text != document.source_text
            else None
        ),
        output_path=output_path,
        diff=diff,
        usage=result.usage,
        effective_config=effective_config,
        residual_risk=result.residual_risk,
        notes=notes,
    )
