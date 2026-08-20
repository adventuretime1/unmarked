# Unmarked

Unmarked reduces ai marking detection by removing unicode & embedded text characters as well as reducing the chance of being detected by generation time statistical pattern by using smart rewrites to maintain the fidelity of the original writing.

It doesn't prove that content is human-written or that every unknown watermark
is gone. Statistical Patterns are harder to overwrite while maintaining the intent of the original
writing, we are currently researching a reliable method for this. Part of this work will be more possible once
model providers release their endpoint allowing us to benchmark results.

## How does marking occur?

Text and files can carry different signals:

- **Invisible Unicode:** zero-width characters, tag characters, unusual spacing,
  and related code points embedded in text.
- **Statistical patterns:** token, phrase, and sentence choices that are influenced. This is on the basis
  of how likely it is for that token to be produced.
- **Attachment metadata:** EXIF, XMP, C2PA/JUMBF, document properties, PDF Info,
  and explicit AI-related fields. Removing C2PA metadata means none will exist which can be flagged
  down the line if most platforms add C2PA metadata. If most users choose to strip the metadata this may not be
  a reliable signal.
- **Signals outside the file:** provider logs, account history, and remote
  detection systems. The provider could in theory hash the storage hence why it's important
  to use a different model and one that isn't watermarking to do the generation.
- **Visible or pixel-level marks:** rendered image watermarks and hidden pixel
  signals. We choose not to remove image watermarks at the time being.

Unmarked addresses the first three cases by using a variety of methods. It does not remove pixel watermarks or provider-side records.

## Features

- Conservatively remove recognized invisible Unicode carriers.
- Inspect and clean metadata in text, PDF, DOCX, ODT, HTML, and
  Markdown files.
- Rewrite through local or user-configured model endpoints to reduce chance of being detected by statistical time patterns.
- Refuse a model-backed rewrite when the known source and rewrite providers are
  the same. (Safety guardrail in case you choose the wrong model)
- Preserve locked key information like dates, times, #s and citations that the llm shouldn't accidentally rewrite. (You may have noticed some ai   rewrites are just erroneous, so we lock this info to reject bad generations). Reject candidates that fail structure or fidelity
  checks.
- Understand and let you analyze your tone & prose to make the writing sound more like you. You can generate multiple profiles and apply them at your will.


### Paste this into your agent

```text
Install Unmarked from https://github.com/adventuretime1/unmarked with uv so the `unmark`
command is available in this environment. Install uv from its official source if
needed, and run `uv tool update-shell` only if the tool directory is not on PATH.
Verify `unmark --version`. Detect whether you are Claude Code, Codex, or Pi, then
run `unmark skills install
--target <this project's native skills directory>`. Verify `unmark inspect
--help`, `unmark edit --help`, and `unmark attachment --help`. Do not expose or
commit API keys. Add this single line to the project's agent instructions:

Use Unmarked when preparing AI-assisted writing or attachments for public posting:
inspect first, clean recognized Unicode or metadata, and use a different model
provider for any rewrite.
```

## Install

Install from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "unmark @ git+https://github.com/adventuretime1/unmarked.git@main"
unmark --version
```

Install the skills into the current project:

```bash
unmark skills install --target .agents/skills
```

Use `.claude/skills` for Claude Code. Codex and other Agent Skills-compatible
agents can use `.agents/skills`. For Pi, use the project skills directory from
its configuration.

### Website https://tryunmarked.com/
We also have a web ui for users who do tasks natively in their browser. You can control the degree of rewriting and the tone & prose centrally in this online platform. 

## Cost
We do not charge any excess cost or middleman fees. We follow a BYOK model for the ai related actions and we support signing in with Openrouter for your convenience and security. That is the recommended approach.

## Account
We support signing with Google so that you can carry your voice profile and any settings across devices.

## Quick use

```bash
# Read-only inspection
unmark inspect draft.md --format json --report inspect.json

# Deterministic Unicode cleaning; writes draft.unmark.md
unmark edit draft.md --unicode-policy safe --report clean.json

# Attachment metadata inspection and cleaning
unmark attachment inspect report.pdf --format json
unmark attachment clean report.pdf --report attachment-clean.json

# Rewrite with a local model
unmark edit draft.md --rewrite --backend ollama --model llama3.1 \
  --source-provider anthropic --intensity low --report rewrite.json

# Install bundled agent skills
unmark skills install --target .agents/skills
```

Remote model endpoints require `--allow-remote`. Pass only the name of an
environment variable with `--key-env`; never pass an API key on the command line.

## Limits

Unmarked can confirm that a recognized operation was performed. It cannot verify
the absence of an undisclosed watermark. Rewriting changes wording and may lower
specific detector scores, but detector behavior varies and accepted output still
needs review. Metadata cleaning is targeted and format-aware; it is not a general
file anonymizer.

## Roadmap

- Evaluate published watermark schemes and detectors on larger, reproducible
  corpora. 
- Evaluate adversarial approaches in recent research like repeated rewrites.
- Test provider detection endpoints when they become available.
- Improve low-cost, non-model transformations and detector-guided evaluation.
- Expand document formats, local checks, and fidelity measurement.

See [How Unmarked works](docs/how-it-works.md), the [roadmap](docs/roadmap.md), and
the [CLI reference](docs/cli.md).

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run mypy
```
