"""Request/response DTOs for the HTTP adapter.

These mirror the CLI's request surface but carry raw text instead of file paths,
because an HTTP client has no filesystem the service should touch. Response
bodies embed the core report objects verbatim (as already-serialized JSON) so
the versioned reporting contract is the single source of truth.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

MediaType = Literal["text/plain", "text/markdown"]
DiffMode = Literal["none", "unified", "operations"]
RewriteStyle = Literal["faithful", "syntax", "lexical", "polish", "simplify", "academic"]
RewriteStrength = Literal["light", "medium", "strong"]


class SanitizeRequest(BaseModel):
    """Deterministic Unicode sanitation of a piece of text (free tier)."""

    text: str = Field(..., description="The text to sanitize.")
    media_type: MediaType = "text/plain"
    #: ``safe`` removes unambiguous carriers and canonicalizes breakable spaces;
    #: ``typographic`` additionally removes soft hyphens and generic format
    #: marks. Homoglyphs are reported, not substituted, by construction.
    policy: Literal["safe", "typographic"] = "safe"
    locks: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Exact text phrases to preserve verbatim (not regular expressions).",
    )
    diff: DiffMode = "none"


class InspectTextRequest(BaseModel):
    """Read-only scan of a piece of text (free tier)."""

    text: str
    media_type: MediaType = "text/plain"
    policy: Literal["report", "safe", "typographic"] = "report"
    locks: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Exact text phrases to preserve verbatim (not regular expressions).",
    )


class RewriteRequest(BaseModel):
    """Prompt-driven, nondeterministic rewrite (gated tier).

    The authenticated caller may supply an end-user OpenRouter key, or the
    service may use ``UNMARK_OPENROUTER_KEY``/``OPENROUTER_API_KEY`` from its
    environment. Either value is used only in the rewrite subprocess environment
    and is never persisted or included in reports.
    """

    text: str
    media_type: MediaType = "text/plain"
    #: The end user's OpenRouter API key (``sk-or-...``), forwarded from the
    #: authenticated web server. Never stored here.
    openrouter_key: SecretStr | None = Field(default=None, min_length=8)
    model: str = Field(..., description="OpenRouter model id, e.g. 'openai/gpt-4o-mini'.")
    source_provider: str = Field(
        ...,
        min_length=1,
        description="Provider the input came from, or 'human'/'unknown'.",
    )
    intensity: Literal["low", "medium", "high"] = "medium"
    strategy: Literal["one-shot", "recursive"] = "one-shot"
    rounds: int = Field(default=1, ge=1, le=5)
    style: RewriteStyle | None = None
    strength: RewriteStrength | None = None
    candidates: int | None = Field(default=None, ge=1, le=16)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    target_length_ratio: float | None = Field(default=None, gt=0.0, le=4.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    early_stop_patience: int | None = Field(default=None, ge=1, le=5)
    seed: int | None = None
    voice_description: str | None = Field(default=None, max_length=20_000)
    locks: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Exact text phrases to preserve verbatim (not regular expressions).",
    )
    diff: DiffMode = "none"
    retain_run: bool = False
    endpoint: str = "https://openrouter.ai/api/v1"


class SanitizeResponse(BaseModel):
    clean_text: str
    report: dict[str, Any]


class InspectResponse(BaseModel):
    report: dict[str, Any]


class AttachmentInspectResponse(BaseModel):
    report: dict[str, Any]


class AttachmentCleanResponse(BaseModel):
    #: True when a rewritten, verified copy was produced.
    cleaned: bool
    filename: str
    media_type: str
    #: Base64 of the cleaned bytes, present only when ``cleaned`` is true.
    output_base64: str | None
    report: dict[str, Any]


class RewriteResponse(BaseModel):
    #: The selected rewrite, or the original text when the run abstained.
    text: str
    #: True when a rewrite candidate was actually selected and committed.
    rewritten: bool
    report: dict[str, Any]
    notes: list[str] = Field(default_factory=list)


class VoiceAnalyzeRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    samples: list[str] = Field(..., min_length=1, max_length=20)
    retention_mode: Literal["derive_discard", "encrypted"] = "derive_discard"
    analysis_completion: str = Field(..., min_length=2, max_length=20_000)


class VoicePromptRequest(BaseModel):
    samples: list[str] = Field(..., min_length=1, max_length=20)
    correction: str = Field(default="", max_length=500)


class VoicePromptResponse(BaseModel):
    system: str
    prompt: str
    temperature: float
    max_output_tokens: int
    models: list[str]
    provider_only: list[str]


class VoiceAnalyzeResponse(BaseModel):
    profile: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    tool_version: str
    c2pa_verifier: str
