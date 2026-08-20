"""Attachment provenance inspection and targeted metadata redaction.

The public API is deliberately bytes-in/bytes-out so the CLI, a website worker,
and a Chrome-extension bridge can share one contract without granting the core
filesystem or network access.
"""

from unmark.attachments.models import (
    AttachmentCleanOutcome,
    AttachmentEvidence,
    AttachmentLimits,
    AttachmentReport,
)
from unmark.attachments.service import clean_attachment, inspect_attachment

__all__ = [
    "AttachmentCleanOutcome",
    "AttachmentEvidence",
    "AttachmentLimits",
    "AttachmentReport",
    "clean_attachment",
    "inspect_attachment",
]
