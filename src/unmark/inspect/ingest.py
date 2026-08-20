"""Document ingestion from files and stdin."""

from __future__ import annotations

import sys
from io import BufferedIOBase
from pathlib import Path

from unmark.core.document import Document, MediaType
from unmark.core.errors import UnsupportedError, UsageError, ValidationError
from unmark.core.policies import FidelityPolicy
from unmark.inspect.protected import discover_spans
from unmark.inspect.structure import parse_blocks

STDIN_SENTINEL = "-"

_EXTENSIONS: dict[str, MediaType] = {
    ".txt": "text/plain",
    ".text": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".mdown": "text/markdown",
}


def infer_media_type(path: Path) -> MediaType:
    """Infer a media type from a file suffix.

    Unknown suffixes are an explicit ``unsupported`` result rather than a silent
    guess: parsing an unknown format as Markdown could mis-protect code.
    """
    suffix = path.suffix.lower()
    try:
        return _EXTENSIONS[suffix]
    except KeyError:
        known = ", ".join(sorted(_EXTENSIONS))
        msg = (
            f"cannot infer media type for {path.name!r}; pass --media-type "
            f"explicitly. Known suffixes: {known}"
        )
        raise UnsupportedError(msg) from None


def read_source(
    source: str, *, stdin: BufferedIOBase | None = None, max_chars: int | None = None
) -> tuple[str, str]:
    """Read UTF-8 text from a path or stdin.

    Returns ``(text, origin)`` where origin is the path string or ``"-"``.
    """
    if source == STDIN_SENTINEL:
        stream = stdin if stdin is not None else sys.stdin.buffer
        raw = stream.read()
        origin = STDIN_SENTINEL
    else:
        path = Path(source)
        if not path.exists():
            msg = f"input not found: {source}"
            raise UsageError(msg)
        if path.is_dir():
            msg = f"input is a directory: {source}"
            raise UsageError(msg)
        raw = path.read_bytes()
        origin = str(path)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"input is not valid UTF-8 at byte {exc.start}: {exc.reason}"
        raise UnsupportedError(msg) from exc

    if max_chars is not None and len(text) > max_chars:
        msg = f"input is {len(text)} characters, over the {max_chars} character limit"
        raise ValidationError(msg)

    return text, origin


def build_document(
    text: str,
    media_type: MediaType,
    *,
    origin: str = STDIN_SENTINEL,
    fidelity: FidelityPolicy | None = None,
    locks: tuple[str, ...] = (),
) -> Document:
    """Parse structure and protected spans into an immutable :class:`Document`."""
    blocks = parse_blocks(text, media_type)
    spans = discover_spans(
        text,
        blocks,
        policy=fidelity or FidelityPolicy(),
        locks=locks,
        media_type=media_type,
    )
    return Document.build(
        text,
        media_type,
        origin=origin,
        blocks=blocks,
        protected_spans=spans,
    )


def load_document(
    source: str,
    *,
    media_type: MediaType | None = None,
    fidelity: FidelityPolicy | None = None,
    locks: tuple[str, ...] = (),
    stdin: BufferedIOBase | None = None,
    max_chars: int | None = None,
) -> Document:
    """Read and parse a document in one step."""
    text, origin = read_source(source, stdin=stdin, max_chars=max_chars)
    if media_type is None:
        media_type = "text/plain" if origin == STDIN_SENTINEL else infer_media_type(Path(origin))
    return build_document(
        text,
        media_type,
        origin=origin,
        fidelity=fidelity,
        locks=locks,
    )
