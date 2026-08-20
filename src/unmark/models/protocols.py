"""Model-adapter contracts.

These types are deliberately transport-agnostic. ``ModelRequest`` is the rendered
prompt plus sampling parameters; ``ModelCompletion`` is a single normalized reply
with optional usage/provenance. Strategies reserve budget around
:meth:`ModelAdapter.generate`, so an adapter never touches the budget itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from unmark.core.spans import StrictModel


class ModelRequest(StrictModel):
    """A single rendered request to a model.

    ``prompt`` is the exact text that would be sent; ``system`` is an optional
    system instruction. No secret ever appears here: API keys are supplied to a
    remote adapter out of band (environment/secret provider), never on the request.
    """

    prompt: str
    system: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None


class ModelUsage(StrictModel):
    """Best-effort usage and provenance from one call.

    Every field is optional because not all providers report them; recorded
    verbatim in the run report. Never carries a key or endpoint credential.
    """

    provider: str = ""
    model: str = ""
    model_version: str = ""
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ModelCompletion(StrictModel):
    """One normalized completion."""

    text: str
    usage: ModelUsage = ModelUsage()


@runtime_checkable
class ModelAdapter(Protocol):
    """Turns a request into completions.

    ``id`` names the backend for reports. ``uses_network`` lets a strategy refuse
    to run a networked adapter when the run budget forbids model calls. ``render``
    returns the exact wire prompt without calling anything, which powers the
    ``print-prompt`` mode and prompt-preservation tests.
    """

    id: str
    uses_network: bool

    def render(self, request: ModelRequest) -> str:
        """Return the exact prompt/request text without contacting a model."""
        ...

    def generate(self, request: ModelRequest, *, count: int) -> tuple[ModelCompletion, ...]:
        """Return ``count`` completions for ``request``."""
        ...
