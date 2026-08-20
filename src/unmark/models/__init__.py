"""Model adapters for rewrite strategies.

A model adapter turns a fully-rendered prompt into one or more completions. The
rewrite strategies own budgeting, fidelity validation, and candidate selection;
an adapter is a thin, testable transport with three responsibilities:

* build the exact request/prompt,
* enforce transport-level safety (scheme, endpoint, redirects, sizes, timeouts),
* normalize provider-specific responses into :class:`ModelCompletion`.

No adapter here imports a vendor SDK. Remote adapters use the standard library so
the default test suite needs no network, keys, or downloads.
"""

from __future__ import annotations

from unmark.models.local import FakeModelAdapter, PrintPromptAdapter
from unmark.models.protocols import (
    ModelAdapter,
    ModelCompletion,
    ModelRequest,
    ModelUsage,
)
from unmark.models.remote import (
    OllamaAdapter,
    OpenAICompatibleAdapter,
    check_endpoint,
    env_key_resolver,
)

__all__ = [
    "FakeModelAdapter",
    "ModelAdapter",
    "ModelCompletion",
    "ModelRequest",
    "ModelUsage",
    "OllamaAdapter",
    "OpenAICompatibleAdapter",
    "PrintPromptAdapter",
    "check_endpoint",
    "env_key_resolver",
]
