# How Unmarked works

Unmarked treats watermarking and fingerprinting as separate signal families. That
keeps each operation narrow and makes the result easier to explain.

## Deterministic cleaning

`unmark inspect` reports recognized Unicode carriers, document structure,
protected spans, and hashes without changing the input. `unmark edit` can then
remove safe carriers while preserving meaningful multilingual characters,
formatting, code, URLs, and locked text.

`unmark attachment inspect` reads supported files by their bytes rather than
trusting the filename. `unmark attachment clean` targets safely separable
metadata such as EXIF, XMP, C2PA/JUMBF, document properties, PDF Info, and
explicit AI-related fields. It re-inspects the result before publishing it.

These steps can verify that recognized fields or characters were removed. They
cannot rule out an unknown signal.

## Statistical rewriting

Some watermark schemes influence which tokens or phrases a model selects. A
meaning-preserving paraphrase changes those choices. Unmarked can request several
candidates, preserve exact locks, reject structural or fidelity failures, and
retain the smallest acceptable change.

The source provider matters. If content is known to come from one provider,
Unmarked requires a different provider family for the rewrite. This guard runs
before the model call. Unknown provenance stays explicitly unknown.

Broader and repeated rewriting usually changes more of the original statistical
pattern, but it also creates more semantic and stylistic risk. No rewrite is
proof against an undisclosed detector.

## Evidence

Unmarked keeps the source file unchanged, writes a new output, and can save a JSON
report containing findings, operations, hashes, result state, and rewrite
rejections. A report describes only what Unmarked measured; it does not infer
authorship.

## Out of scope

- Visible and invisible pixel-level image marks
- Provider logs or account-side provenance
- Universal AI authorship detection
- Claims that an unknown watermark is absent
- Automatic publication without human review
