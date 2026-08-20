"""Environment-backed configuration for the HTTP sidecar."""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
DEFAULT_REWRITE_TIMEOUT_S = 600.0


def max_file_bytes(environ: Mapping[str, str] | None = None) -> int:
    """Return the upload limit, allowing an explicit environment override."""
    source = os.environ if environ is None else environ
    return int(source.get("UNMARK_MAX_FILE_BYTES", str(DEFAULT_MAX_FILE_BYTES)))


def rewrite_timeout_s(environ: Mapping[str, str] | None = None) -> float:
    """Return the rewrite subprocess timeout in seconds."""
    source = os.environ if environ is None else environ
    return float(source.get("UNMARK_REWRITE_TIMEOUT_S", str(DEFAULT_REWRITE_TIMEOUT_S)))
