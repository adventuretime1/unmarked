"""Plain-text rendering of reports.

Pure string building with no Rich dependency, so it is testable and usable from
any front end. The CLI decides where the text goes and whether to colorize.
"""

from __future__ import annotations

from unmark.reporting.schema import EditReport, InspectReport, RewriteReport, UnicodeSummary

_MAX_LISTED = 40


def _counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{kind} {count}" for kind, count in sorted(counts.items()))


def _unicode_lines(unicode: UnicodeSummary) -> list[str]:
    lines = [
        f"Unicode findings ({unicode.policy} policy): {len(unicode.findings)}",
        f"  by kind:    {_counts(unicode.counts_by_kind)}",
        f"  actionable: {unicode.removable}",
        f"  preserved:  {unicode.preserved}",
    ]
    if not unicode.findings:
        return lines

    lines.append("")
    shown = unicode.findings[:_MAX_LISTED]
    for finding in shown:
        action = (
            "remove"
            if finding.removable and finding.replacement == ""
            else (f"-> {finding.replacement!r}" if finding.removable else "keep")
        )
        lines.append(
            f"  {finding.label:<9} {finding.category:<3} offset {finding.offset:<7} "
            f"{finding.kind:<20} {action}"
        )
        lines.append(f"      {finding.name}")
        lines.append(f"      {finding.reason}")
        if finding.protected_by:
            lines.append(f"      preserved because: {finding.protected_by}")
    if len(unicode.findings) > _MAX_LISTED:
        lines.append(f"  ... and {len(unicode.findings) - _MAX_LISTED} more (use --format json)")

    if unicode.blocked_by_protection:
        lines.append("")
        lines.append(
            f"  {len(unicode.blocked_by_protection)} actionable finding(s) left in place "
            "because they sit inside a protected span."
        )
    return lines


def render_inspect(report: InspectReport) -> str:
    doc = report.document
    lines = [
        f"Document: {doc.origin}",
        f"  media type: {doc.media_type}",
        f"  sha256:     {doc.source_sha256}",
        f"  size:       {doc.characters} characters, {doc.lines} lines",
        f"  scripts:    {', '.join(doc.scripts) if doc.scripts else 'none detected'}",
        "",
        f"Structure: {len(doc.blocks)} blocks",
        f"  {_counts(doc.block_counts)}",
        "",
        f"Protected spans: {len(doc.protected_spans)}",
        f"  {_counts(doc.span_counts)}",
    ]
    if doc.protected_spans:
        lines.append("")
        for span in doc.protected_spans[:_MAX_LISTED]:
            value = span.value if len(span.value) <= 60 else span.value[:57] + "..."
            lines.append(
                f"  {span.kind:<11} [{span.start}, {span.end})  {value!r}"
                + (f"  via {span.detector}" if span.detector else "")
            )
        if len(doc.protected_spans) > _MAX_LISTED:
            lines.append(f"  ... and {len(doc.protected_spans) - _MAX_LISTED} more")
    lines.append(f"  {doc.heuristic_caveat}")
    lines.append("")
    lines.extend(_unicode_lines(report.unicode))
    lines.append("")
    lines.append(f"Note: {report.residual_risk}")
    return "\n".join(lines) + "\n"


def render_rewrite(report: RewriteReport) -> str:
    doc = report.document
    header = "Planned rewrite (dry run)" if report.dry_run else "Rewrite"
    lines = [
        f"Run {report.run_id}",
        f"  state:    {report.state}",
        f"  strategy: {report.strategy}",
        f"  backend:  {report.backend}",
        f"  style:    {report.style}",
        f"  source:   {doc.origin} ({doc.source_sha256[:16]}...)",
    ]
    if report.output_path:
        lines.append(f"  output:   {report.output_path}")
    lines.append("")

    # print-prompt backend: show the exact prompt and nothing else was done.
    if report.prompt_preview is not None:
        lines.append("Rendered prompt (no model was called):")
        lines.append("")
        lines.append(report.prompt_preview)
        lines.append("")
        lines.append(f"Note: {report.residual_risk}")
        for note in report.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines) + "\n"

    lines.append(f"{header}: {report.operation_count} operation(s)")
    lines.append(f"  candidates generated: {report.candidates_generated}")
    lines.append(f"  candidates rejected:  {report.candidates_rejected}")
    if report.rejection_reasons:
        lines.append(f"  rejection reasons:    {_counts(_tally(report.rejection_reasons))}")

    if report.hops:
        lines.append("")
        lines.append("Hops:")
        for hop in report.hops:
            risk = "n/a" if hop.best_residual_risk is None else f"{hop.best_residual_risk:.4f}"
            note = f"  {hop.note}" if hop.note else ""
            lines.append(
                f"  round {hop.round_index} [{hop.style}] "
                f"accepted {hop.accepted_count} rejected {hop.rejected_count} "
                f"best risk {risk}{note}"
            )

    if report.stopping_reason:
        lines.append(f"  stopping reason: {report.stopping_reason}")
    if report.selection_reason:
        lines.append(f"  selection:       {report.selection_reason}")

    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"Note: {note}")
    lines.append("")
    lines.append(f"Note: {report.residual_risk}")
    return "\n".join(lines) + "\n"


def _tally(items: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def render_edit(report: EditReport) -> str:
    doc = report.document
    header = "Planned changes (dry run)" if report.dry_run else "Applied changes"
    lines = [
        f"Run {report.run_id}",
        f"  state:  {report.state}",
        f"  preset: {report.preset}",
        f"  source: {doc.origin} ({doc.source_sha256[:16]}...)",
    ]
    if report.output_path:
        lines.append(f"  output: {report.output_path}")
    lines.append("")
    lines.extend(_unicode_lines(report.unicode))
    lines.append("")
    lines.append(f"{header}: {report.operation_count} operation(s)")
    if report.operation_count:
        lines.append(f"  characters changed: {report.usage.char_edit_ratio:.4%}")
        lines.append(f"  length drift:       {report.usage.length_drift_ratio:.4%}")
    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"Note: {note}")
    lines.append("")
    lines.append(f"Note: {report.residual_risk}")
    return "\n".join(lines) + "\n"
