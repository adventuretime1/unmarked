"""Source/rewrite provider separation for model-backed rewrites."""

from __future__ import annotations

import io

import pytest

import unmark.application.rewrite_service as rewrite_service
from unmark.application.requests import EditRequest
from unmark.core.errors import UsageError
from unmark.orchestration.config import RewriteConfigSection


def section(**updates: object) -> RewriteConfigSection:
    values: dict[str, object] = {
        "backend": "openai-compatible",
        "endpoint": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4.1-mini",
        "source_provider": "anthropic",
        "allow_remote": True,
    }
    values.update(updates)
    return RewriteConfigSection.model_validate(values)


def test_openrouter_slug_derives_cross_provider_family() -> None:
    assert rewrite_service._resolve_provider_separation(section()) == ("anthropic", "openai")


def test_common_aliases_are_normalized() -> None:
    with pytest.raises(UsageError, match="both resolve to 'openai'"):
        rewrite_service._resolve_provider_separation(section(source_provider="chatgpt"))


def test_ambiguous_custom_model_requires_explicit_provider() -> None:
    ambiguous = section(endpoint="https://models.example.test/v1", model="rewrite-model")
    with pytest.raises(UsageError, match="set --rewrite-provider"):
        rewrite_service._resolve_provider_separation(ambiguous)

    explicit = ambiguous.model_copy(update={"rewrite_provider": "mistral"})
    assert rewrite_service._resolve_provider_separation(explicit) == ("anthropic", "mistral")


def test_model_backend_requires_stated_source() -> None:
    with pytest.raises(UsageError, match="--source-provider"):
        rewrite_service._resolve_provider_separation(section(source_provider=""))


def test_same_provider_fallback_is_refused() -> None:
    configured = section(
        model="google/gemini-3.7-flash",
        source_provider="openai",
        fallback_models=("openai/gpt-oss-120b",),
    )
    with pytest.raises(UsageError, match="fallback model provider overlaps"):
        rewrite_service._resolve_provider_separation(configured)


def test_remote_rewrite_requires_named_configured_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(UsageError, match=r"requires rewrite\.key_env"):
        rewrite_service.check_rewrite_configuration(section())

    configured = section(key_env="UNMARK_TEST_OPENROUTER_KEY")
    monkeypatch.delenv("UNMARK_TEST_OPENROUTER_KEY", raising=False)
    with pytest.raises(UsageError, match="is not set or is empty"):
        rewrite_service.check_rewrite_configuration(configured)

    monkeypatch.setenv("UNMARK_TEST_OPENROUTER_KEY", "secret-never-reported")
    status = rewrite_service.check_rewrite_configuration(configured)
    assert status["ready"] is True
    assert status["key_env"] == "UNMARK_TEST_OPENROUTER_KEY"
    assert "secret-never-reported" not in repr(status)


def test_overlap_refused_before_adapter_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter_built = False

    def unexpected_adapter(_section: RewriteConfigSection) -> None:
        nonlocal adapter_built
        adapter_built = True

    monkeypatch.setattr(rewrite_service, "_build_adapter", unexpected_adapter)
    request = EditRequest(
        input="-",
        output="-",
        rewrite=True,
        rewrite_overrides={
            "backend": "openai-compatible",
            "endpoint": "https://openrouter.ai/api/v1",
            "model": "anthropic/claude-sonnet-4",
            "source_provider": "claude",
            "allow_remote": True,
        },
        stdin=io.BytesIO(b"Text that must not be sent to a model."),
        retain_run=False,
    )

    with pytest.raises(UsageError, match="choose a model from a different provider"):
        rewrite_service.rewrite_document(request)
    assert not adapter_built


def test_normalized_providers_are_recorded_in_effective_config() -> None:
    request = EditRequest(
        input="-",
        output="-",
        rewrite=True,
        rewrite_overrides={
            "backend": "print-prompt",
            "source_provider": "claude",
            "rewrite_provider": "codex",
        },
        stdin=io.BytesIO(b"A short source paragraph."),
        dry_run=True,
        output_format="json",
        research_mode=True,
        retain_run=False,
    )

    outcome = rewrite_service.rewrite_document(request)
    effective = outcome.report.effective_config["rewrite"]
    assert effective["source_provider"] == "anthropic"
    assert effective["rewrite_provider"] == "openai"
    assert outcome.report.effective_config["research_mode"] is True
    assert outcome.report.effective_config["output"]["format"] == "json"
