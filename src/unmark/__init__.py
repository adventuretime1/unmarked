"""Unmarked: quality-first, evidence-backed text transformation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("unmark")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
