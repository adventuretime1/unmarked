"""Report schema.

Reports are the product's evidence record, so they must be explicit about what a
run does and does not establish. ``residual_risk`` is required on every report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from unmark.core.budgets import BudgetUsage
from unmark.core.operations import Operation
from unmark.core.results import ResultState
from unmark.core.spans import Block, Span, StrictModel
from unmark.inspect.unicode_scan import UnicodeFinding

REPORT_SCHEMA_VERSION: Literal["1"] = "1"

#: Stated on every sanitize-only run. This describes the exact signal family
#: affected rather than treating all text watermarks as one mechanism.
SANITIZE_RESIDUAL_RISK = (
    "Removed recognized hidden Unicode carriers and their embedded signature. "
    "Statistical token, n-gram, and model-probability patterns are a separate signal "
    "family addressed by rewriting. Provider logs and retrieval records are external "
    "to the text."
)

INSPECT_RESIDUAL_RISK = (
    "Read-only inspection changed nothing. Findings cover recognized Unicode "
    "signals only; statistical patterns, unknown signals, provider logs, and "
    "retrieval records were not verified."
)


class DocumentSummary(StrictModel):
    """Structure and protected content, as discovered."""

    schema_version: Literal["1"] = "1"
    origin: str
    media_type: str
    source_sha256: str
    characters: int = Field(ge=0)
    lines: int = Field(ge=0)
    scripts: tuple[str, ...] = ()
    blocks: tuple[Block, ...] = ()
    block_counts: dict[str, int] = Field(default_factory=dict)
    protected_spans: tuple[Span, ...] = ()
    span_counts: dict[str, int] = Field(default_factory=dict)
    heuristic_caveat: str = (
        "Protected spans are found by heuristics and are neither complete nor "
        "guaranteed correct. Use --lock for regions that must survive verbatim."
    )


class UnicodeSummary(StrictModel):
    """Unicode findings and what the policy did with them."""

    schema_version: Literal["1"] = "1"
    policy: str
    findings: tuple[UnicodeFinding, ...] = ()
    counts_by_kind: dict[str, int] = Field(default_factory=dict)
    removable: int = Field(default=0, ge=0)
    preserved: int = Field(default=0, ge=0)
    blocked_by_protection: tuple[UnicodeFinding, ...] = ()


class InspectReport(StrictModel):
    """Output of ``unmark inspect``. Read-only by construction."""

    schema_version: Literal["1"] = REPORT_SCHEMA_VERSION
    kind: Literal["inspect"] = "inspect"
    generated_at: datetime
    tool_version: str
    document: DocumentSummary
    unicode: UnicodeSummary
    effective_config: dict[str, Any] = Field(default_factory=dict)
    residual_risk: str = INSPECT_RESIDUAL_RISK


class EditReport(StrictModel):
    """Output of ``unmark edit``."""

    schema_version: Literal["1"] = REPORT_SCHEMA_VERSION
    kind: Literal["edit"] = "edit"
    run_id: str
    generated_at: datetime
    tool_version: str
    state: ResultState
    preset: str
    dry_run: bool
    document: DocumentSummary
    unicode: UnicodeSummary
    operations: tuple[Operation, ...] = ()
    operation_count: int = Field(default=0, ge=0)
    output_sha256: str | None = None
    output_path: str | None = None
    diff: str | None = None
    usage: BudgetUsage = BudgetUsage()
    effective_config: dict[str, Any] = Field(default_factory=dict)
    residual_risk: str = SANITIZE_RESIDUAL_RISK
    notes: tuple[str, ...] = ()


class RewriteHopSummary(StrictModel):
    """One recursive hop, for the report's audit trail."""

    schema_version: Literal["1"] = "1"
    round_index: int = Field(ge=0)
    style: str
    best_residual_risk: float | None = None
    accepted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    note: str = ""


class CanonicalizationStageSummary(StrictModel):
    """One deterministic carrier-cleanup stage around a rewrite."""

    schema_version: Literal["1"] = "1"
    stage: Literal["input", "output"]
    unicode: UnicodeSummary
    operations: tuple[Operation, ...] = ()
    operation_count: int = Field(default=0, ge=0)


class RewriteReport(StrictModel):
    """Output of a ``unmark edit`` rewrite run.

    Deliberately distinct from :class:`EditReport`: a rewrite changes visible
    token, n-gram, sentence, and model-probability patterns, while Unicode
    sanitation removes literal hidden carriers. No secret, endpoint credential,
    or raw prompt containing one ever appears here.
    """

    schema_version: Literal["1"] = REPORT_SCHEMA_VERSION
    kind: Literal["rewrite"] = "rewrite"
    run_id: str
    generated_at: datetime
    tool_version: str
    state: ResultState
    strategy: str
    backend: str
    style: str
    dry_run: bool
    document: DocumentSummary
    prompt_preview: str | None = Field(
        default=None,
        description="Set only for the print-prompt backend: the exact rendered prompt.",
    )
    canonicalization: tuple[CanonicalizationStageSummary, ...] = ()
    operations: tuple[Operation, ...] = ()
    operation_count: int = Field(default=0, ge=0)
    candidates_generated: int = Field(default=0, ge=0)
    candidates_rejected: int = Field(default=0, ge=0)
    rejection_reasons: tuple[str, ...] = ()
    hops: tuple[RewriteHopSummary, ...] = ()
    stopping_reason: str = ""
    selection_reason: str = ""
    output_sha256: str | None = None
    output_path: str | None = None
    diff: str | None = None
    usage: BudgetUsage = BudgetUsage()
    effective_config: dict[str, Any] = Field(default_factory=dict)
    residual_risk: str
    notes: tuple[str, ...] = ()
