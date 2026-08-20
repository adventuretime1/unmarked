"""Application services.

The CLI and any future HTTP or queue worker call into this layer. Editing
decisions live here, never in a command function.
"""

from unmark.application.edit_service import EditOutcome, edit_document
from unmark.application.inspect_service import inspect_document
from unmark.application.requests import EditRequest, InspectRequest

__all__ = [
    "EditOutcome",
    "EditRequest",
    "InspectRequest",
    "edit_document",
    "inspect_document",
]
