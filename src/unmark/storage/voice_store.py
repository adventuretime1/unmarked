"""Local voice-profile store.

Profiles live in one place -- the user's own config directory::

    ~/.config/unmark/voices/<name>.md      (or .json)

They are personal, not per-project: a writing voice belongs to the person, not
the repository, so there is a single location and no precedence chain to
explain. A project-specific or shared profile is still reachable by passing an
explicit path to ``--voice``.

Nothing here touches the network. ``.md`` files hold the description verbatim,
which is what a coding agent writes; ``.json`` files hold a serialized
:class:`~unmark.core.voice.VoiceProfile`. Both load to the same model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from unmark.core.errors import UsageError
from unmark.core.voice import VoiceProfile, is_valid_voice_name
from unmark.orchestration.config import user_config_path
from unmark.storage.atomic import atomic_write_text

VOICES_DIRNAME = "voices"

#: Extensions searched for a bare name, in preference order.
VOICE_SUFFIXES: tuple[str, ...] = (".md", ".json")

#: Refuse to read an implausibly large file rather than paging it into memory.
MAX_VOICE_FILE_BYTES = 1 << 20  # 1 MiB


def voices_dir() -> Path:
    """The directory holding stored profiles. Not created as a side effect."""
    return user_config_path().parent / VOICES_DIRNAME


class VoiceStore:
    """Reads and writes profiles under :func:`voices_dir`."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else voices_dir()

    def path_for(self, name: str, suffix: str = ".md") -> Path:
        """The path a profile named ``name`` would occupy."""
        _require_valid_name(name)
        return self.root / f"{name}{suffix}"

    def list_names(self) -> tuple[str, ...]:
        """Stored profile names, sorted and de-duplicated across suffixes."""
        if not self.root.is_dir():
            return ()
        names = {
            entry.stem
            for entry in self.root.iterdir()
            if entry.is_file() and entry.suffix in VOICE_SUFFIXES
        }
        return tuple(sorted(names))

    def find(self, name: str) -> Path | None:
        """The file backing ``name``, preferring ``.md``."""
        _require_valid_name(name)
        for suffix in VOICE_SUFFIXES:
            candidate = self.root / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def load(self, name: str) -> VoiceProfile:
        """Load a stored profile by name."""
        path = self.find(name)
        if path is None:
            known = ", ".join(self.list_names()) or "none"
            msg = f"no voice profile named {name!r} in {self.root}. Stored profiles: {known}."
            raise UsageError(msg)
        return load_profile(path)

    def save(self, profile: VoiceProfile, *, force: bool = False) -> Path:
        """Store ``profile``, returning the path written.

        Written as JSON when there is metadata worth keeping, and as plain
        Markdown otherwise: a description with no provenance is more useful as a
        file the user can open and edit than as a JSON blob.
        """
        if not profile.name:
            msg = "a voice profile must be named before it can be stored"
            raise UsageError(msg)
        self.root.mkdir(parents=True, exist_ok=True)

        has_metadata = bool(profile.generated_by or profile.generated_at or profile.notes)
        suffix = ".json" if has_metadata else ".md"
        destination = self.path_for(profile.name, suffix)

        # A profile may already exist under the other suffix; refuse that too,
        # rather than silently leaving a stale duplicate that shadows this one.
        existing = self.find(profile.name)
        if existing is not None and existing != destination:
            if not force:
                msg = (
                    f"voice profile {profile.name!r} already exists at {existing}. "
                    "Pass --force to replace it."
                )
                raise UsageError(msg)
            existing.unlink()

        body = (
            profile.model_dump_json(indent=2) + "\n" if has_metadata else profile.description + "\n"
        )
        atomic_write_text(destination, body, force=force)
        return destination

    def delete(self, name: str) -> Path:
        """Remove a stored profile, returning the path removed."""
        path = self.find(name)
        if path is None:
            msg = f"no voice profile named {name!r} in {self.root}"
            raise UsageError(msg)
        path.unlink()
        return path


def _require_valid_name(name: str) -> None:
    if not is_valid_voice_name(name):
        msg = (
            f"invalid voice name {name!r}: use letters, digits, hyphen, and "
            "underscore only. To load a file directly, pass its path to --voice."
        )
        raise UsageError(msg)


def _read_text(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        msg = f"could not read voice profile {path}: {exc}"
        raise UsageError(msg) from exc
    if size > MAX_VOICE_FILE_BYTES:
        msg = f"voice profile {path} is too large ({size} bytes)"
        raise UsageError(msg)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        msg = f"voice profile {path} is not valid UTF-8"
        raise UsageError(msg) from exc
    except OSError as exc:
        msg = f"could not read voice profile {path}: {exc}"
        raise UsageError(msg) from exc


def load_profile(path: Path) -> VoiceProfile:
    """Load a profile from an explicit ``.md`` or ``.json`` path."""
    if not path.is_file():
        msg = f"voice profile not found: {path}"
        raise UsageError(msg)
    text = _read_text(path)

    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"invalid JSON in voice profile {path}: {exc}"
            raise UsageError(msg) from exc
        if not isinstance(data, dict):
            msg = f"voice profile {path} must contain a JSON object"
            raise UsageError(msg)
        data.setdefault("name", path.stem)
        try:
            return VoiceProfile.model_validate(data)
        except PydanticValidationError as exc:
            msg = f"invalid voice profile {path}: {exc}"
            raise UsageError(msg) from exc

    # Any other suffix is treated as prose: the file *is* the description.
    try:
        return VoiceProfile(name=path.stem, description=text)
    except PydanticValidationError as exc:
        msg = f"invalid voice profile {path}: {exc}"
        raise UsageError(msg) from exc


def resolve_voice(reference: str, *, store: VoiceStore | None = None) -> VoiceProfile:
    """Resolve a ``--voice`` argument.

    A bare name (``formal``) is looked up in the store. Anything containing a
    path separator or a file suffix is treated as a literal path, which is how a
    one-off or project-specific profile is used without storing it.
    """
    active = store if store is not None else VoiceStore()
    looks_like_path = "/" in reference or "\\" in reference or Path(reference).suffix != ""
    if looks_like_path:
        return load_profile(Path(reference).expanduser())
    return active.load(reference)


def new_profile(name: str, description: str, *, generated_by: str = "") -> VoiceProfile:
    """Build a profile stamped with the current time."""
    return VoiceProfile(
        name=name,
        description=description,
        generated_by=generated_by,
        generated_at=datetime.now(UTC),
    )
