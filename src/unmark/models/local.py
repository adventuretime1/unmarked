"""Non-networked adapters: ``print-prompt`` and a deterministic fake.

Both are safe to use in the default test suite: neither opens a socket, reads a
key, or downloads a model.
"""

from __future__ import annotations

from collections.abc import Callable

from unmark.models.protocols import ModelCompletion, ModelRequest, ModelUsage


def _render_prompt(request: ModelRequest) -> str:
    if request.system:
        return f"[system]\n{request.system}\n\n[user]\n{request.prompt}"
    return request.prompt


class PrintPromptAdapter:
    """Emits the exact prompt instead of a completion.

    ``generate`` returns the rendered prompt as its single "completion" so the
    pipeline can surface it without special-casing; strategies detect this backend
    by ``uses_network is False`` and ``id == "print-prompt"`` and never treat the
    prompt as a rewrite candidate.
    """

    id = "print-prompt"
    uses_network = False

    def render(self, request: ModelRequest) -> str:
        return _render_prompt(request)

    def generate(self, request: ModelRequest, *, count: int) -> tuple[ModelCompletion, ...]:
        rendered = self.render(request)
        usage = ModelUsage(provider="print-prompt", model="none")
        return (ModelCompletion(text=rendered, usage=usage),)


#: A completion generator: ``(request, index) -> completion text``.
FakeResponder = Callable[[ModelRequest, int], str]


class FakeModelAdapter:
    """A deterministic, offline adapter for tests and demonstrations.

    Supply a ``responder`` mapping ``(request, candidate_index)`` to text, or a
    fixed list of ``responses`` cycled by index. Produces stable output for a given
    input so tests never depend on sampling.
    """

    uses_network = False

    def __init__(
        self,
        *,
        responder: FakeResponder | None = None,
        responses: tuple[str, ...] | None = None,
        model_id: str = "fake",
    ) -> None:
        if responder is None and responses is None:
            msg = "FakeModelAdapter needs either a responder or a list of responses"
            raise ValueError(msg)
        self._responder = responder
        self._responses = responses
        self.id = f"fake:{model_id}"
        self._model_id = model_id

    def render(self, request: ModelRequest) -> str:
        return _render_prompt(request)

    def _one(self, request: ModelRequest, index: int) -> str:
        if self._responder is not None:
            return self._responder(request, index)
        assert self._responses is not None
        return self._responses[index % len(self._responses)]

    def generate(self, request: ModelRequest, *, count: int) -> tuple[ModelCompletion, ...]:
        usage = ModelUsage(provider="fake", model=self._model_id, model_version="test")
        return tuple(
            ModelCompletion(text=self._one(request, index), usage=usage) for index in range(count)
        )
