"""Atomic filesystem writes.

Guarantees, in the order the CLI specification requires them:

* never overwrite the source document;
* never overwrite any existing destination without ``--force``;
* refuse symlink destinations;
* write through a same-directory temporary file, ``fsync``, then ``os.replace``;
* leave no partial file behind on interruption or validation failure.

The temporary file must live in the destination directory because ``os.replace``
is only atomic within a single filesystem.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from unmark.core.errors import AtomicWriteError, UsageError


def check_destination(
    destination: Path, *, source: Path | None = None, force: bool = False
) -> None:
    """Validate a destination path before any work is done.

    Raises :class:`UsageError` so the CLI exits 2 for a refused destination, which
    is a usage problem, rather than 7, which means a write actually failed.
    """
    if destination.is_symlink():
        msg = (
            f"refusing to write through a symlink: {destination}. "
            "Resolve the link and pass the real path."
        )
        raise UsageError(msg)

    if source is not None:
        try:
            same = destination.exists() and source.exists() and destination.samefile(source)
        except OSError:
            same = False
        if same or destination.resolve() == source.resolve():
            msg = f"refusing to overwrite the source document: {destination}"
            raise UsageError(msg)

    if destination.exists():
        if destination.is_dir():
            msg = f"destination is a directory: {destination}"
            raise UsageError(msg)
        if not force:
            msg = f"destination already exists: {destination}. Pass --force to replace it."
            raise UsageError(msg)

    parent = destination.parent
    if not parent.exists():
        msg = f"destination directory does not exist: {parent}"
        raise UsageError(msg)


@contextmanager
def atomic_path(destination: Path) -> Iterator[Path]:
    """Yield a temporary path that is atomically moved onto ``destination``.

    If the body raises, the temporary file is removed and the destination is left
    untouched.
    """
    parent = destination.parent
    handle, temp_name = tempfile.mkstemp(dir=parent, prefix=f".{destination.name}.", suffix=".tmp")
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        yield temp_path
        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_path, destination)  # noqa: PTH105 - atomic rename
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def atomic_write_text(
    destination: Path,
    text: str,
    *,
    source: Path | None = None,
    force: bool = False,
    newline: str = "",
    validate: bool = True,
) -> None:
    """Write ``text`` to ``destination`` atomically.

    When ``validate`` is set the written bytes are read back and compared before
    the rename is committed, so a truncated or corrupted write never replaces a
    good file.
    """
    check_destination(destination, source=source, force=force)
    encoded = text.encode("utf-8")
    try:
        with atomic_path(destination) as temp_path:
            with temp_path.open("w", encoding="utf-8", newline=newline) as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            if validate and temp_path.read_bytes() != encoded:
                msg = f"validation failed: {destination} did not round-trip after write"
                raise AtomicWriteError(msg)
    except OSError as exc:
        msg = f"could not write {destination}: {exc}"
        raise AtomicWriteError(msg) from exc


def atomic_write_bytes(
    destination: Path,
    data: bytes,
    *,
    source: Path | None = None,
    force: bool = False,
    validate: bool = True,
) -> None:
    """Write binary output through the same fail-before-publish path as text.

    The temporary file is read back before ``os.replace`` so an interrupted or
    truncated attachment write never replaces an existing destination.
    """
    check_destination(destination, source=source, force=force)
    try:
        with atomic_path(destination) as temp_path:
            with temp_path.open("wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if validate and temp_path.read_bytes() != data:
                msg = f"validation failed: {destination} did not round-trip after write"
                raise AtomicWriteError(msg)
    except OSError as exc:
        msg = f"could not write {destination}: {exc}"
        raise AtomicWriteError(msg) from exc


def default_output_path(source: Path) -> Path:
    """Sibling destination for a source file, e.g. ``draft.md`` -> ``draft.unmark.md``."""
    suffix = source.suffix
    stem = source.name[: -len(suffix)] if suffix else source.name
    return source.with_name(f"{stem}.unmark{suffix}")
