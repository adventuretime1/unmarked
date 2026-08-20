"""Networked adapters: OpenAI-compatible and Ollama.

Security posture (default-deny):

* only ``http``/``https`` endpoints are accepted; other schemes are refused;
* non-loopback endpoints are denied unless remote access is explicitly enabled;
* redirects are refused outright, so an ``Authorization`` header can never be
  re-sent to an unvalidated host;
* API keys come only from the environment or a secret provider, never from CLI
  arguments, and are never serialized into requests' logs, reports, or errors;
* requests enforce a timeout, a response-size cap, and an input-size cap.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlparse

from unmark.core.errors import DependencyUnavailableError, UsageError
from unmark.models.protocols import ModelCompletion, ModelRequest, ModelUsage

#: Hosts treated as this machine; anything else is "remote".
LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

#: Hard cap on a single response body, guarding against a hostile/huge reply.
DEFAULT_MAX_RESPONSE_BYTES = 8 << 20  # 8 MiB

#: Hard cap on the rendered prompt sent to a model.
DEFAULT_MAX_INPUT_CHARS = 200_000

#: A resolver for an API key. Injected so keys never pass through CLI arguments
#: and a secret-provider backend can replace the default environment lookup.
KeyResolver = Callable[[], str | None]


def env_key_resolver(var_name: str) -> KeyResolver:
    """A key resolver that reads ``var_name`` from the environment at call time."""

    def resolve() -> str | None:
        import os

        value = os.environ.get(var_name)
        return value or None

    return resolve


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects.

    urllib re-sends request headers on 3xx, which would forward the API key to a
    redirect target behind the loopback allowlist. Surfacing any 3xx as an error
    keeps a key from leaving the validated host.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)  # type: ignore[arg-type]


def check_endpoint(base_url: str, *, allow_remote: bool) -> None:
    """Enforce the endpoint allowlist. Raises :class:`UsageError` on refusal."""
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        msg = f"model endpoint must be http(s); got scheme {parsed.scheme!r}: {base_url}"
        raise UsageError(msg)
    host = parsed.hostname or ""
    if host in LOOPBACK_HOSTS:
        return
    if not allow_remote:
        msg = (
            f"model endpoint host {host!r} is not loopback; refusing to send content "
            "off-machine. Enable remote access explicitly to override."
        )
        raise UsageError(msg)


def _post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    timeout: float,
    max_response_bytes: int,
) -> dict[str, object]:
    if urlparse(url).scheme not in ("http", "https"):
        msg = f"refusing non-http(s) model endpoint: {url}"
        raise UsageError(msg)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(max_response_bytes + 1)
    except urllib.error.URLError as error:
        # str(error) can only contain the URL and reason, never the key (which
        # lives in a header we never stringify).
        raise DependencyUnavailableError(f"model endpoint call failed: {error}") from error
    if len(raw) > max_response_bytes:
        msg = f"model response exceeded {max_response_bytes} bytes"
        raise DependencyUnavailableError(msg)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise DependencyUnavailableError(f"model response was not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise DependencyUnavailableError("model response was not a JSON object")
    return parsed


def _openai_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if urlparse(base).path.rstrip("/").endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class OpenAICompatibleAdapter:
    """POSTs to an OpenAI-style ``/v1/chat/completions`` endpoint."""

    uses_network = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        key_resolver: KeyResolver | None = None,
        allow_remote: bool = False,
        timeout: float = 60.0,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        fallback_models: tuple[str, ...] = (),
        provider_only: tuple[str, ...] = (),
    ) -> None:
        check_endpoint(base_url, allow_remote=allow_remote)
        self.id = f"openai-compatible:{model}"
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._key_resolver = key_resolver
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._max_input_chars = max_input_chars
        self._fallback_models = tuple(fallback_models)
        self._provider_only = tuple(provider_only)

    def render(self, request: ModelRequest) -> str:
        return _render_chat_request(self._model, request)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._key_resolver is not None:
            key = self._key_resolver()
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def generate(self, request: ModelRequest, *, count: int) -> tuple[ModelCompletion, ...]:
        if len(request.prompt) > self._max_input_chars:
            msg = f"prompt is {len(request.prompt)} chars, over the {self._max_input_chars} limit"
            raise UsageError(msg)
        url = _openai_chat_url(self._base_url)
        completions: list[ModelCompletion] = []
        for _ in range(max(1, count)):
            payload: dict[str, object] = {
                "model": self._model,
                "messages": _chat_messages(request),
                "temperature": request.temperature,
            }
            if self._fallback_models:
                payload.pop("model", None)
                payload["models"] = [self._model, *self._fallback_models]
            if self._provider_only:
                payload["provider"] = {
                    "only": list(self._provider_only),
                    "allow_fallbacks": True,
                }
            if request.max_output_tokens is not None:
                payload["max_tokens"] = request.max_output_tokens
            if request.seed is not None:
                payload["seed"] = request.seed
            data = _post_json(
                url,
                payload,
                self._headers(),
                timeout=self._timeout,
                max_response_bytes=self._max_response_bytes,
            )
            completions.append(self._parse(data))
        return tuple(completions)

    def _parse(self, data: dict[str, object]) -> ModelCompletion:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DependencyUnavailableError("model response had no choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content:
            raise DependencyUnavailableError("model response had empty content")
        raw_usage = data.get("usage")
        usage_block: dict[str, object] = raw_usage if isinstance(raw_usage, dict) else {}
        usage = ModelUsage(
            provider="openai-compatible",
            model=str(data.get("model") or self._model),
            input_tokens=_int_or_none(usage_block.get("prompt_tokens")),
            output_tokens=_int_or_none(usage_block.get("completion_tokens")),
        )
        return ModelCompletion(text=content.strip(), usage=usage)


class OllamaAdapter:
    """POSTs to an Ollama ``/api/chat`` endpoint. Defaults to loopback."""

    uses_network = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        allow_remote: bool = False,
        timeout: float = 120.0,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ) -> None:
        check_endpoint(base_url, allow_remote=allow_remote)
        self.id = f"ollama:{model}"
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._max_input_chars = max_input_chars

    def render(self, request: ModelRequest) -> str:
        return _render_chat_request(self._model, request)

    def generate(self, request: ModelRequest, *, count: int) -> tuple[ModelCompletion, ...]:
        if len(request.prompt) > self._max_input_chars:
            msg = f"prompt is {len(request.prompt)} chars, over the {self._max_input_chars} limit"
            raise UsageError(msg)
        url = f"{self._base_url}/api/chat"
        completions: list[ModelCompletion] = []
        for _ in range(max(1, count)):
            payload: dict[str, object] = {
                "model": self._model,
                "stream": False,
                "messages": _chat_messages(request),
                "options": {"temperature": request.temperature},
            }
            if request.seed is not None:
                options = payload["options"]
                assert isinstance(options, dict)
                options["seed"] = request.seed
            data = _post_json(
                url,
                payload,
                {},
                timeout=self._timeout,
                max_response_bytes=self._max_response_bytes,
            )
            completions.append(self._parse(data))
        return tuple(completions)

    def _parse(self, data: dict[str, object]) -> ModelCompletion:
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content:
            raise DependencyUnavailableError("model response had empty content")
        usage = ModelUsage(
            provider="ollama",
            model=str(data.get("model") or self._model),
            input_tokens=_int_or_none(data.get("prompt_eval_count")),
            output_tokens=_int_or_none(data.get("eval_count")),
        )
        return ModelCompletion(text=content.strip(), usage=usage)


def _chat_messages(request: ModelRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})
    return messages


def _render_chat_request(model: str, request: ModelRequest) -> str:
    return json.dumps({"model": model, "messages": _chat_messages(request)}, indent=2)


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
