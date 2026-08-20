---
name: unmark-clean
description: Inspect and deterministically clean recognized Unicode carriers or embedded metadata with Unmarked while preserving visible content. Use for pasted text, Markdown, HTML, DOCX, ODT, PDF, SVG, PNG, or JPEG files when the user wants metadata inspection, Unicode sanitation, or a non-rewrite first pass.
---

# Unmarked Clean

Use Unmarked's deterministic tools. Do not rewrite text merely because it may be
AI-assisted.

## Workflow

1. Identify whether the input is plain text, Markdown, or a supported attachment.
   Record provenance as unknown when it is not established; do not infer it from
   writing style.
2. Inspect before changing anything:

   ```bash
   unmark inspect INPUT --format json --report inspect.json
   unmark attachment inspect INPUT --format json --report attachment-inspect.json
   ```

   Use `unmark inspect` for text or Markdown. Use `unmark attachment inspect` for a
   supported file.
3. Apply only the relevant deterministic operation:

   ```bash
   unmark edit INPUT --unicode-policy safe --output OUTPUT --report clean.json
   unmark attachment clean INPUT --output OUTPUT --report attachment-clean.json
   ```

   Prefer `safe`. Use `typographic` only when conservative spacing normalization
   is wanted. Treat `aggressive` as research-only because it can damage
   legitimate multilingual text.
4. Re-inspect the output and retain the JSON reports. Attachment cleaning already
   re-inspects before publishing its output.
5. Report exactly what Unmarked recognized and changed. Do not infer authorship and
   do not claim that an unknown watermark is absent.

If deterministic cleaning satisfies the request, stop there. Use
`unmark-rewrite` only when the user also wants paraphrasing and accepts wording
changes.

## Tools and limits

- `unmark inspect`: structure, protected spans, hashes, and Unicode findings.
- `unmark edit`: `report`, `safe`, `typographic`, and research-only `aggressive`
  Unicode policies.
- `unmark attachment inspect`: PNG, JPEG, SVG, PDF, DOCX, ODT, HTML, and Markdown.
- `unmark attachment clean`: targeted EXIF, XMP, C2PA/JUMBF, document properties,
  PDF Info, and explicit AI-related metadata, followed by reinspection.
- `unmark config` and `unmark runs`: effective settings and retained reports.

Pixel watermark removal and provider-side records are outside this workflow.
