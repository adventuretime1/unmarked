"""HTTP delivery adapter for the Unmarked engine.

A thin FastAPI service that exposes the core's versioned service contracts
(text sanitation, attachment inspect/clean, and the gated prompt-driven rewrite)
over JSON, so the Unmarked website and other surfaces consume one contract
instead of re-implementing any transformation. It adds no editing logic of its
own: every endpoint delegates to :mod:`unmark.application` or
:mod:`unmark.attachments`.

This package is optional. Install it with the ``api`` extra::

    uv pip install -e ".[api]"

and run it with::

    unmark-api            # or: uvicorn unmark.api.app:app
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["create_app"]


def create_app() -> FastAPI:
    """Build and return the FastAPI application (imported lazily)."""
    from unmark.api.app import create_app as _create_app

    return _create_app()
