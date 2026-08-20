"""Read-only document inspection: structure, protected spans, and Unicode."""

from unmark.inspect.ingest import build_document, infer_media_type, load_document, read_source
from unmark.inspect.protected import discover_spans, is_protected
from unmark.inspect.structure import parse_blocks
from unmark.inspect.unicode_scan import (
    UnicodeFinding,
    inspect_text,
    sanitation_operations,
)

__all__ = [
    "UnicodeFinding",
    "build_document",
    "discover_spans",
    "infer_media_type",
    "inspect_text",
    "is_protected",
    "load_document",
    "parse_blocks",
    "read_source",
    "sanitation_operations",
]
