"""HTTP-delivery contract coverage for web-only options."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from unmark.api import app as api_app
from unmark.api.models import (
    InspectTextRequest,
    RewriteRequest,
    SanitizeRequest,
    VoiceAnalyzeRequest,
)


def test_http_locks_are_treated_as_literal_phrases() -> None:
    assert api_app._literal_locks([".*", "ACME-123"]) == (r"\.\*", r"ACME\-123")


def test_text_requests_accept_markdown_locks_and_diffs() -> None:
    sanitize = SanitizeRequest(
        text="# Draft",
        media_type="text/markdown",
        locks=["ACME-123"],
        diff="operations",
    )
    inspect = InspectTextRequest(
        text="# Draft",
        media_type="text/markdown",
        locks=["v1.2.3"],
    )

    assert sanitize.locks == ["ACME-123"]
    assert sanitize.diff == "operations"
    assert inspect.media_type == "text/markdown"


def test_rewrite_forwards_advanced_options_and_disables_retention(monkeypatch) -> None:
    captured: list[str] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        captured.extend(cmd)
        report_path = Path(cmd[cmd.index("--report") + 1])
        report_path.write_text(json.dumps({"state": "rewritten_unverified"}))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"Rewritten text\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    response = api_app._run_rewrite(
        RewriteRequest(
            text="Original text",
            openrouter_key=SecretStr("sk-or-test"),
            model="provider/model",
            source_provider="unknown",
            style="academic",
            strength="strong",
            candidates=4,
            temperature=0.8,
            target_length_ratio=0.9,
            max_output_tokens=500,
            early_stop_patience=2,
            seed=42,
            locks=["ACME-123"],
            voice_description="Short sentences. Direct claims.",
            diff="operations",
        )
    )

    assert response.rewritten is True
    assert response.notes == ["result: rewritten_unverified"]
    assert "--no-retain-run" in captured
    assert captured[captured.index("--source-provider") + 1] == "unknown"
    assert captured[captured.index("--rewrite-style") + 1] == "academic"
    assert captured[captured.index("--max-output-tokens") + 1] == "500"
    assert captured[captured.index("--lock") + 1] == r"ACME\-123"
    voice_path = Path(captured[captured.index("--voice") + 1])
    assert voice_path.name == "voice.md"


def test_rewrite_can_use_service_environment_key_without_persisting_it(monkeypatch) -> None:
    captured_env: dict[str, str] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        captured_env.update(environment)
        report_path = Path(cmd[cmd.index("--report") + 1])
        report_path.write_text(json.dumps({"state": "rewritten_unverified"}))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"Rewritten text\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.delenv("UNMARK_OPENROUTER_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "service-secret-never-reported")
    response = api_app._run_rewrite(
        RewriteRequest(
            text="Original text",
            model="google/gemini-3.7-flash",
            source_provider="anthropic",
        )
    )

    assert response.rewritten is True
    assert captured_env["UNMARK_OPENROUTER_KEY"] == "service-secret-never-reported"
    assert "service-secret-never-reported" not in repr(response)


def test_rewrite_refuses_missing_service_and_request_key(monkeypatch) -> None:
    monkeypatch.delenv("UNMARK_OPENROUTER_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    request = RewriteRequest(
        text="Original text",
        model="google/gemini-3.7-flash",
        source_provider="anthropic",
    )
    with pytest.raises(HTTPException) as error:
        api_app._run_rewrite(request)
    assert error.value.status_code == 422
    assert "credential is not configured" in str(error.value.detail)


def test_voice_analysis_request_bounds_samples() -> None:
    request = VoiceAnalyzeRequest(
        name="work",
        samples=["A sufficiently long sample is validated by the analyzer endpoint itself."],
        analysis_completion='{"prose":"direct"}',
    )
    assert request.retention_mode == "derive_discard"


def test_default_auto_rewrite_is_pinned_to_cerebras(monkeypatch) -> None:
    captured: list[str] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        captured.extend(cmd)
        report_path = Path(cmd[cmd.index("--report") + 1])
        report_path.write_text(json.dumps({"state": "rewritten_unverified"}))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"Rewritten text\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    api_app._run_rewrite(
        RewriteRequest(
            text="Original text",
            openrouter_key=SecretStr("sk-or-test"),
            model="openai/gpt-oss-120b",
            source_provider="unknown",
        )
    )
    assert "--fallback-model" not in captured
    providers = [
        captured[index + 1] for index, value in enumerate(captured) if value == "--provider-only"
    ]
    assert providers == ["cerebras"]


def test_deep_rewrite_is_pinned_to_modal(monkeypatch) -> None:
    captured: list[str] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        captured.extend(cmd)
        report_path = Path(cmd[cmd.index("--report") + 1])
        report_path.write_text(json.dumps({"state": "rewritten_unverified"}))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"Rewritten text\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    api_app._run_rewrite(
        RewriteRequest(
            text="Original text",
            openrouter_key=SecretStr("sk-or-test"),
            model="qwen/qwen3.8-2.4t-a95b",
            source_provider="unknown",
        )
    )
    assert "--fallback-model" not in captured
    providers = [
        captured[index + 1] for index, value in enumerate(captured) if value == "--provider-only"
    ]
    assert providers == ["modal"]
