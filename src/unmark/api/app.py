"""FastAPI application exposing the Unmarked engine over HTTP.

Endpoints delegate to the core services; no transformation logic lives here.

- ``POST /v1/text/sanitize``    deterministic Unicode sanitation (free)
- ``POST /v1/text/inspect``     read-only scan (free)
- ``POST /v1/attachment/inspect`` provenance inspection of an image (free)
- ``POST /v1/attachment/clean``   targeted metadata removal of an image (free)
- ``POST /v1/voice/prompt``       canonical qualitative analysis prompt
- ``POST /v1/voice/analyze``      validated qualitative voice profile
- ``POST /v1/text/rewrite``     prompt-driven nondeterministic rewrite (gated)

The gated rewrite is the only endpoint that touches a model. It runs the
``unmark`` CLI in a subprocess so an end-user or service-configured OpenRouter key
lives only in that child process's environment for a single call — honoring the
core rule that an API key is read from the environment and never persisted,
logged, or copied into reports.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from unmark.api.config import max_file_bytes, rewrite_timeout_s
from unmark.api.models import (
    AttachmentCleanResponse,
    AttachmentInspectResponse,
    HealthResponse,
    InspectResponse,
    InspectTextRequest,
    RewriteRequest,
    RewriteResponse,
    SanitizeRequest,
    SanitizeResponse,
    VoiceAnalyzeRequest,
    VoiceAnalyzeResponse,
    VoicePromptRequest,
    VoicePromptResponse,
)
from unmark.application.edit_service import edit_document
from unmark.application.inspect_service import inspect_document
from unmark.application.requests import EditRequest, InspectRequest
from unmark.attachments import clean_attachment, inspect_attachment
from unmark.core.errors import UnsupportedError, UsageError, ValidationError
from unmark.core.model_defaults import (
    OPENROUTER_MODELS,
    OPENROUTER_PROVIDER_ONLY,
    OPENROUTER_RECOMMENDED_MODES,
    openrouter_route,
)
from unmark.core.voice_model import (
    analyze_voice_samples_from_completion,
    build_voice_analysis_request,
)

# ── configuration (all optional, read from the environment) ──────────────────
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("UNMARK_API_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
_API_TOKEN = os.environ.get("UNMARK_API_TOKEN") or None
_MAX_TEXT_CHARS = int(os.environ.get("UNMARK_MAX_TEXT_CHARS", "200000"))
_MAX_FILE_BYTES = max_file_bytes()
_REWRITE_TIMEOUT_S = rewrite_timeout_s()
_KEY_ENV_NAME = "UNMARK_OPENROUTER_KEY"
_SERVICE_KEY_ENV_NAMES = (_KEY_ENV_NAME, "OPENROUTER_API_KEY")


def _default_unmark_bin() -> str:
    """Locate the ``unmark`` CLI, preferring the one beside this interpreter.

    Falls back to bare ``unmark`` (PATH lookup) when a sibling binary is absent,
    e.g. a system-wide install. ``UNMARK_BIN`` overrides everything.
    """
    override = os.environ.get("UNMARK_BIN")
    if override:
        return override
    sibling = Path(sys.executable).with_name("unmark")
    return str(sibling) if sibling.exists() else "unmark"


_UNMARK_BIN = _default_unmark_bin()


def _rewrite_key(req: RewriteRequest) -> str:
    """Resolve a per-request or service key without retaining or reporting it."""
    if req.openrouter_key is not None:
        return req.openrouter_key.get_secret_value()
    for name in _SERVICE_KEY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    names = " or ".join(_SERVICE_KEY_ENV_NAMES)
    raise HTTPException(
        status_code=422,
        detail=(
            f"rewrite credential is not configured; set {names} in the service "
            "environment or supply an authenticated per-request key"
        ),
    )


def _tool_version() -> str:
    try:
        return version("unmark")
    except PackageNotFoundError:  # pragma: no cover - only when run from a raw checkout
        return "0.0.0"


def _require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    """Optional shared-secret gate for the whole API.

    When ``UNMARK_API_TOKEN`` is set, every request must present it as a bearer
    token. This keeps the sidecar private to the trusted web server even if its
    port is reachable. When unset (local dev) the gate is open.
    """
    if _API_TOKEN is None:
        return
    expected = f"Bearer {_API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="missing or invalid API token")


def _http_error(exc: Exception) -> HTTPException:
    """Map core errors to HTTP status codes."""
    if isinstance(exc, UnsupportedError):
        return HTTPException(status_code=415, detail=str(exc))
    if isinstance(exc, (UsageError, ValidationError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail=f"internal error: {exc}")


def _guard_text(text: str) -> None:
    if len(text) > _MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"text is {len(text)} chars, over the {_MAX_TEXT_CHARS} limit",
        )


def _suffix_for(media_type: str) -> str:
    return ".md" if media_type == "text/markdown" else ".txt"


def _literal_locks(locks: list[str]) -> tuple[str, ...]:
    """Turn untrusted HTTP lock phrases into safe literal regexes.

    The local CLI intentionally accepts trusted regular expressions. The public
    HTTP adapter accepts exact phrases instead so anonymous requests cannot run
    pathological backtracking expressions inside the long-lived API process.
    """
    cleaned: list[str] = []
    for lock in locks:
        value = lock.strip()
        if not value:
            continue
        if len(value) > 500:
            raise HTTPException(status_code=422, detail="a protected phrase is over 500 characters")
        cleaned.append(re.escape(value))
    return tuple(cleaned)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Unmarked API",
        version=_tool_version(),
        summary="Evidence-backed redaction for text and AI-generated attachments.",
        dependencies=[Depends(_require_token)],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, dependencies=[])
    def health() -> HealthResponse:
        try:
            import c2pa  # type: ignore[import-untyped]  # noqa: F401

            verifier = "official"
        except Exception:
            verifier = "unavailable"
        return HealthResponse(status="ok", tool_version=_tool_version(), c2pa_verifier=verifier)

    @app.post("/v1/text/sanitize", response_model=SanitizeResponse)
    def text_sanitize(req: SanitizeRequest) -> SanitizeResponse:
        _guard_text(req.text)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f"input{_suffix_for(req.media_type)}"
            src.write_text(req.text, encoding="utf-8")
            try:
                outcome = edit_document(
                    EditRequest(
                        input=str(src),
                        output="-",
                        preset="sanitize",
                        media_type=req.media_type,
                        unicode_policy=req.policy,
                        locks=_literal_locks(req.locks),
                        diff=req.diff,
                        retain_run=False,
                        output_format="json",
                    )
                )
            except Exception as exc:
                raise _http_error(exc) from exc
        return SanitizeResponse(
            clean_text=outcome.candidate_text,
            report=outcome.report.model_dump(mode="json"),
        )

    @app.post("/v1/text/inspect", response_model=InspectResponse)
    def text_inspect(req: InspectTextRequest) -> InspectResponse:
        _guard_text(req.text)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f"input{_suffix_for(req.media_type)}"
            src.write_text(req.text, encoding="utf-8")
            try:
                report = inspect_document(
                    InspectRequest(
                        input=str(src),
                        media_type=req.media_type,
                        unicode_policy=req.policy,
                        locks=_literal_locks(req.locks),
                        output_format="json",
                    )
                )
            except Exception as exc:
                raise _http_error(exc) from exc
        return InspectResponse(report=report.model_dump(mode="json"))

    @app.post("/v1/attachment/inspect", response_model=AttachmentInspectResponse)
    def attachment_inspect(file: Annotated[UploadFile, File()]) -> AttachmentInspectResponse:
        data = _read_upload(file)
        try:
            report = inspect_attachment(data)
        except Exception as exc:
            raise _http_error(exc) from exc
        return AttachmentInspectResponse(report=report.model_dump(mode="json"))

    @app.post("/v1/attachment/clean", response_model=AttachmentCleanResponse)
    def attachment_clean(file: Annotated[UploadFile, File()]) -> AttachmentCleanResponse:
        data = _read_upload(file)
        try:
            outcome = clean_attachment(data)
        except Exception as exc:
            raise _http_error(exc) from exc
        name = file.filename or "attachment"
        output_bytes = outcome.output_bytes
        cleaned = output_bytes is not None
        return AttachmentCleanResponse(
            cleaned=cleaned,
            filename=_cleaned_name(name),
            media_type=file.content_type or "application/octet-stream",
            output_base64=(
                base64.b64encode(output_bytes).decode("ascii") if output_bytes is not None else None
            ),
            report=outcome.report.model_dump(mode="json"),
        )

    @app.post("/v1/text/rewrite", response_model=RewriteResponse)
    def text_rewrite(req: RewriteRequest) -> RewriteResponse:
        _guard_text(req.text)
        return _run_rewrite(req)

    @app.post("/v1/voice/analyze", response_model=VoiceAnalyzeResponse)
    def voice_analyze(req: VoiceAnalyzeRequest) -> VoiceAnalyzeResponse:
        try:
            profile = analyze_voice_samples_from_completion(
                req.name,
                req.samples,
                req.analysis_completion,
                retention_mode=req.retention_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return VoiceAnalyzeResponse(profile=profile.model_dump(mode="json"))

    @app.post("/v1/voice/prompt", response_model=VoicePromptResponse)
    def voice_prompt(req: VoicePromptRequest) -> VoicePromptResponse:
        request = build_voice_analysis_request(req.samples, correction=req.correction)
        return VoicePromptResponse(
            system=request.system or "",
            prompt=request.prompt,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens or 1200,
            models=list(OPENROUTER_MODELS),
            provider_only=list(OPENROUTER_PROVIDER_ONLY),
        )

    return app


def _read_upload(file: UploadFile) -> bytes:
    data = file.file.read()
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file is {len(data)} bytes, over the {_MAX_FILE_BYTES} limit",
        )
    if not data:
        raise HTTPException(status_code=422, detail="empty upload")
    return data


def _cleaned_name(name: str) -> str:
    stem, dot, ext = name.rpartition(".")
    return f"{stem}.unmark.{ext}" if dot else f"{name}.unmark"


def _run_rewrite(req: RewriteRequest) -> RewriteResponse:
    """Invoke the CLI rewrite in an isolated subprocess with the user's key."""
    import json

    env = dict(os.environ)
    env[_KEY_ENV_NAME] = _rewrite_key(req)
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        cmd = [
            _UNMARK_BIN,
            "edit",
            "-",
            "--media-type",
            req.media_type,
            "--rewrite",
            "--backend",
            "openai-compatible",
            "--endpoint",
            req.endpoint,
            "--model",
            req.model,
            "--source-provider",
            req.source_provider,
            "--key-env",
            _KEY_ENV_NAME,
            "--allow-remote",
            "--intensity",
            req.intensity,
            "--strategy",
            req.strategy,
            "--rounds",
            str(req.rounds),
            "--diff",
            req.diff,
            "--output",
            "-",
            "--report",
            str(report_path),
            "--quiet",
        ]
        optional_args: tuple[tuple[str, object | None], ...] = (
            ("--rewrite-style", req.style),
            ("--rewrite-strength", req.strength),
            ("--candidates", req.candidates),
            ("--temperature", req.temperature),
            ("--target-length-ratio", req.target_length_ratio),
            ("--max-output-tokens", req.max_output_tokens),
            ("--early-stop-patience", req.early_stop_patience),
            ("--seed", req.seed),
        )
        for flag, value in optional_args:
            if value is not None:
                cmd.extend((flag, str(value)))
        recommended_mode = next(
            (
                mode
                for mode, route in OPENROUTER_RECOMMENDED_MODES.items()
                if route["model"] == req.model
            ),
            None,
        )
        if recommended_mode is not None:
            route_models, route_providers = openrouter_route(recommended_mode)
            for fallback_model in route_models[1:]:
                cmd.extend(("--fallback-model", fallback_model))
            for provider in route_providers:
                cmd.extend(("--provider-only", provider))
        for lock in _literal_locks(req.locks):
            cmd.extend(("--lock", lock))
        if req.voice_description:
            voice_path = Path(tmp) / "voice.md"
            voice_path.write_text(req.voice_description, encoding="utf-8")
            cmd.extend(("--voice", str(voice_path)))
        if not req.retain_run:
            cmd.append("--no-retain-run")
        try:
            proc = subprocess.run(
                cmd,
                input=req.text.encode("utf-8"),
                env=env,
                capture_output=True,
                timeout=_REWRITE_TIMEOUT_S,
                check=False,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"unmark binary not found (set UNMARK_BIN): {exc}",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="rewrite timed out") from exc

        report: dict[str, object] = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = {}

    stdout_text = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0 and not report:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        # Exit code 3 is a deliberate "abstained", not a failure.
        if proc.returncode != 3:
            raise HTTPException(status_code=502, detail=stderr or "rewrite failed")

    rewritten = bool(stdout_text.strip())
    notes: list[str] = []
    state = report.get("state")
    result_section = report.get("result")
    if state is None and isinstance(result_section, dict):
        state = result_section.get("state")
    if isinstance(state, str):
        notes.append(f"result: {state}")
    return RewriteResponse(
        text=stdout_text if rewritten else req.text,
        rewritten=rewritten,
        report=report,
        notes=notes,
    )


# Module-level app for `uvicorn unmark.api.app:app`.
app = create_app()


def main() -> None:
    """Console-script entry point: ``unmark-api``."""
    import uvicorn

    host = os.environ.get("UNMARK_API_HOST", "127.0.0.1")
    port = int(os.environ.get("UNMARK_API_PORT", "8787"))
    uvicorn.run("unmark.api.app:app", host=host, port=port, reload=False)
