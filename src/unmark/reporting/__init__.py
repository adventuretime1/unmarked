"""Report schemas, construction, and rendering."""

from unmark.reporting.build import build_edit_report, build_inspect_report
from unmark.reporting.render_text import render_edit, render_inspect
from unmark.reporting.schema import (
    REPORT_SCHEMA_VERSION,
    DocumentSummary,
    EditReport,
    InspectReport,
    UnicodeSummary,
)

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "DocumentSummary",
    "EditReport",
    "InspectReport",
    "UnicodeSummary",
    "build_edit_report",
    "build_inspect_report",
    "render_edit",
    "render_inspect",
]
