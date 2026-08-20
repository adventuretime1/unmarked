# Unmarked CLI

The Unmarked CLI inspects and prepares text and supported attachments for sharing.
It separates deterministic Unicode and metadata cleaning from optional
model-backed rewriting.

Unmarked reports what it found and changed. It does not certify human authorship or
prove that an unknown watermark is absent.

## Install

From GitHub:

```bash
uv tool install "unmark @ git+https://github.com/adventuretime1/unmarked.git@main"
export PATH="$(uv tool dir --bin):$PATH" # needed only for this shell when `unmark` is not found
unmark --version
```

For a permanent shell configuration, run `uv tool update-shell`, restart the
shell, then run `unmark --version`.

From a checkout:

```bash
cd unmark
uv sync --all-extras
uv run unmark --help
```

The default install includes cryptographic C2PA verification plus attachment
metadata inspection and removal. Attachment cleaning removes format-supported
embedded privacy and provenance metadata; it is not a universal anonymizer and
does not remove image pixels, every possible metadata field, or provider-side
records.

## Commands

```bash
# Inspect text or Markdown without changing it
unmark inspect draft.md --format json --report inspect.json

# Remove recognized hidden Unicode; source remains unchanged
unmark edit draft.md --unicode-policy safe --report clean.json

# Inspect or clean supported attachment metadata
unmark attachment inspect report.docx --format json
unmark attachment clean report.docx --report attachment-clean.json

# Review effective configuration and saved runs
unmark config validate --explain
unmark runs show RUN_ID

# Manage reusable writing-voice profiles
unmark voice --help

# Install bundled agent skills
unmark skills install --target .agents/skills
```

Supported attachments are PNG, JPEG, SVG, PDF, DOCX, ODT, HTML, and Markdown.
Attachment cleaning targets safely separable EXIF, XMP, C2PA/JUMBF, document
properties, PDF Info, and explicit AI-related metadata. It does not change image
pixels.

## Unicode policies

- `report`: inspect only.
- `safe`: remove unambiguous hidden carriers while preserving meaningful script
  and emoji characters. This is the default.
- `typographic`: `safe` plus conservative spacing normalization.
- `aggressive`: research-only and requires `--research-mode`.

Unmarked reports confusable characters but does not replace them automatically.

## Rewriting

Rewriting is opt-in. The default `print-prompt` backend makes no model call.
Ollama and OpenAI-compatible endpoints are supported.

```bash
# Local Ollama
unmark edit draft.md --rewrite --backend ollama --model llama3.1 \
  --source-provider anthropic --intensity low --report rewrite.json

# OpenAI-compatible remote endpoint
export OPENROUTER_API_KEY=...  # keep this outside Git
unmark edit draft.md --rewrite --backend openai-compatible \
  --endpoint https://openrouter.ai/api/v1 --model openai/gpt-4.1-mini \
  --source-provider anthropic --key-env OPENROUTER_API_KEY --allow-remote \
  --intensity low --report rewrite.json

# Verify routing, provider separation, and credential presence without a model call
unmark config rewrite-check --source-provider anthropic --format json
```

Use the actual source provider when known. Unmarked normalizes common provider
aliases and refuses the model call when the primary or a known fallback model
matches the source provider. Use `unknown` only when provenance is genuinely
unknown. Store endpoint/model routing and the environment-variable name in
`.unmark.toml`; keep the credential value in the shell or a secret manager.

Intensity controls the tradeoff:

- `low`: one conservative lexical pass that still requires meaningful wording
  or flow changes where safe.
- `medium`: a broader one-pass lexical rewrite.
- `high`: three staged rounds—sentence structure, wording, then polish—with
  more candidates, latency, and model usage.

Candidates must preserve protected spans, locks, structure, and configured
fidelity limits. Accepted output remains a draft for review. A rewrite report is
not detector verification unless a named detector was actually measured.

## Files and safety

- Source files are never overwritten by default.
- Outputs use a `.unmark` sibling name unless `--output` is supplied.
- Existing outputs require `--force`; symlink destinations are refused.
- Writes are validated and atomically published.
- Remote endpoints require `--allow-remote`.
- API keys are read only from the environment variable named by `--key-env`.
- `unmark config rewrite-check` reports readiness without printing the key or
  sending source text.
- Local run records live under `.unmark/runs/`; use `--no-retain-run` to disable
  rewrite retention.

## Agent skills

```bash
unmark skills install --target .agents/skills
```

Use `.claude/skills` for Claude Code or the project skills directory configured
by another agent. The bundled skills cover deterministic cleaning, guarded
rewriting, and optional voice-profile creation.

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run mypy
uv run pytest
uv run python -m build
```

Evaluation fixtures are intentionally small and tracked under `evals/`.
Generated corpora, model weights, detector outputs, and credentials stay outside
Git.
