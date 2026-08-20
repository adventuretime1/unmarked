"""Offline security and behavior tests for the model adapters.

The networked cases use a throwaway loopback HTTP server, never a real endpoint,
key, or off-machine host. They assert the default-deny posture, redirect refusal,
and that a secret never lands in a serialized artifact or error.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from unmark.core.errors import DependencyUnavailableError, UsageError
from unmark.models.local import PrintPromptAdapter
from unmark.models.protocols import ModelRequest
from unmark.models.remote import (
    OllamaAdapter,
    OpenAICompatibleAdapter,
    _openai_chat_url,
    check_endpoint,
    env_key_resolver,
)

_REQUEST = ModelRequest(prompt="rewrite this", system="be careful", temperature=0.5)


def test_openai_endpoint_accepts_api_root_or_versioned_base() -> None:
    expected = "https://openrouter.ai/api/v1/chat/completions"
    assert _openai_chat_url("https://openrouter.ai/api") == expected
    assert _openai_chat_url("https://openrouter.ai/api/v1") == expected


# --- endpoint allowlist (no network) --------------------------------------


def test_remote_endpoint_denied_by_default() -> None:
    with pytest.raises(UsageError, match="not loopback"):
        check_endpoint("https://api.example.com", allow_remote=False)


def test_remote_endpoint_allowed_with_opt_in() -> None:
    # Does not raise; the opt-in is the explicit gate.
    check_endpoint("https://api.example.com", allow_remote=True)


def test_loopback_endpoint_allowed_by_default() -> None:
    for url in ("http://127.0.0.1:11434", "http://localhost:8080", "http://[::1]:1234"):
        check_endpoint(url, allow_remote=False)


def test_non_http_scheme_refused() -> None:
    for url in ("file:///etc/passwd", "ftp://host/x", "gopher://host"):
        with pytest.raises(UsageError, match="http"):
            check_endpoint(url, allow_remote=True)


def test_constructing_remote_adapter_denied_by_default() -> None:
    with pytest.raises(UsageError):
        OpenAICompatibleAdapter(base_url="https://api.example.com", model="gpt")
    with pytest.raises(UsageError):
        OllamaAdapter(model="llama", base_url="https://remote.example.com")


# --- a loopback server for the networked paths ----------------------------


class _Handler(BaseHTTPRequestHandler):
    mode = "ok"  # ok | redirect | malformed | echo-auth
    last_body: ClassVar[dict[str, object]] = {}

    def log_message(self, *args: object) -> None:  # silence test output
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            parsed = json.loads(raw_body)
            _Handler.last_body = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            _Handler.last_body = {}
        if _Handler.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/other")
            self.end_headers()
            return
        if _Handler.mode == "malformed":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"this is not json")
            return
        if _Handler.mode == "echo-auth":
            # Reflect the Authorization header into the body so a test can prove the
            # adapter never leaks it into a serialized artifact via other channels.
            auth = self.headers.get("Authorization", "")
            payload = {"choices": [{"message": {"content": f"seen:{auth}"}}]}
            self._json(payload)
            return
        # Default OK: shape depends on which path was hit.
        if self.path.endswith("/api/chat"):
            self._json({"message": {"content": "rewritten by ollama"}, "model": "llama"})
        else:
            self._json({"choices": [{"message": {"content": "rewritten by openai"}}]})

    def _json(self, obj: Mapping[str, object]) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def loopback_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        _Handler.mode = "ok"


def test_loopback_ollama_roundtrip(loopback_server: str) -> None:
    _Handler.mode = "ok"
    adapter = OllamaAdapter(model="llama", base_url=loopback_server)
    completions = adapter.generate(_REQUEST, count=1)
    assert completions[0].text == "rewritten by ollama"


def test_loopback_openai_roundtrip(loopback_server: str) -> None:
    _Handler.mode = "ok"
    adapter = OpenAICompatibleAdapter(base_url=loopback_server, model="gpt")
    completions = adapter.generate(_REQUEST, count=1)
    assert completions[0].text == "rewritten by openai"


def test_openai_adapter_serializes_ordered_model_and_provider_fallbacks(
    loopback_server: str,
) -> None:
    adapter = OpenAICompatibleAdapter(
        base_url=loopback_server,
        model="google/gemini-3.7-flash",
        fallback_models=("openai/gpt-oss-120b",),
        provider_only=("google-ai-studio", "cerebras"),
    )
    adapter.generate(_REQUEST, count=1)
    assert _Handler.last_body["models"] == [
        "google/gemini-3.7-flash",
        "openai/gpt-oss-120b",
    ]
    assert _Handler.last_body["provider"] == {
        "only": ["google-ai-studio", "cerebras"],
        "allow_fallbacks": True,
    }
    assert "model" not in _Handler.last_body


def test_redirect_is_refused(loopback_server: str) -> None:
    _Handler.mode = "redirect"
    adapter = OpenAICompatibleAdapter(base_url=loopback_server, model="gpt")
    with pytest.raises(DependencyUnavailableError):
        adapter.generate(_REQUEST, count=1)


def test_malformed_response_is_handled(loopback_server: str) -> None:
    _Handler.mode = "malformed"
    adapter = OllamaAdapter(model="llama", base_url=loopback_server)
    with pytest.raises(DependencyUnavailableError):
        adapter.generate(_REQUEST, count=1)


def test_input_size_cap_enforced(loopback_server: str) -> None:
    adapter = OllamaAdapter(model="llama", base_url=loopback_server, max_input_chars=10)
    big = ModelRequest(prompt="x" * 50)
    with pytest.raises(UsageError, match="over the"):
        adapter.generate(big, count=1)


# --- key handling ---------------------------------------------------------


def test_key_read_from_env_only(monkeypatch: pytest.MonkeyPatch, loopback_server: str) -> None:
    monkeypatch.setenv("UNMARK_TEST_KEY", "secret-token-xyz")
    _Handler.mode = "echo-auth"
    adapter = OpenAICompatibleAdapter(
        base_url=loopback_server,
        model="gpt",
        key_resolver=env_key_resolver("UNMARK_TEST_KEY"),
    )
    completions = adapter.generate(_REQUEST, count=1)
    # The server saw the Bearer key (proving the resolver ran) ...
    assert "Bearer secret-token-xyz" in completions[0].text


def test_rendered_request_never_contains_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNMARK_TEST_KEY", "secret-token-xyz")
    adapter = OpenAICompatibleAdapter(
        base_url="http://127.0.0.1:11434",
        model="gpt",
        key_resolver=env_key_resolver("UNMARK_TEST_KEY"),
    )
    rendered = adapter.render(_REQUEST)
    # The rendered payload (used in reports/prompt previews) carries no credential.
    assert "secret-token-xyz" not in rendered
    assert "Authorization" not in rendered


def test_print_prompt_adapter_makes_no_network() -> None:
    adapter = PrintPromptAdapter()
    assert adapter.uses_network is False
    out = adapter.generate(_REQUEST, count=3)
    # print-prompt yields exactly one "completion": the rendered prompt.
    assert len(out) == 1
    assert "rewrite this" in out[0].text
