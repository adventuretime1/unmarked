"""Editable voice profiles derived from a person's writing samples.

The structured fields hold aggregate, non-content traits measured by the local
analyzer. ``description`` remains the only part rendered into a rewrite prompt,
so a user can inspect and correct exactly what influences generation. Raw
samples are deliberately outside this schema and are never required at rewrite
time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from unmark.core.spans import StrictModel

#: Guards against a runaway file being pasted into a prompt.
MAX_DESCRIPTION_CHARS = 20_000


class VoiceStyle(StrictModel):
    """Measured aggregate traits derived from writing samples."""

    schema_version: Literal["1"] = "1"
    sentence_count: int = 0
    mean_sentence_words: float = 0.0
    sentence_word_stdev: float = 0.0
    short_sentence_ratio: float = 0.0
    paragraph_count: int = 0
    mean_paragraph_sentences: float = 0.0
    type_token_ratio: float = 0.0
    mean_word_characters: float = 0.0
    contraction_rate_per_1000_words: float = 0.0
    first_person_rate_per_1000_words: float = 0.0
    second_person_rate_per_1000_words: float = 0.0
    punctuation_per_1000_words: dict[str, float] = Field(default_factory=dict)


class VoiceProfile(StrictModel):
    """One named description of a writing voice."""

    schema_version: str = "2"
    name: str = ""
    description: str
    generated_by: str = ""
    generated_at: datetime | None = None
    notes: str = ""
    structured_style: VoiceStyle | None = None
    sample_count: int = 0
    sample_word_count: int = 0
    sample_hashes: tuple[str, ...] = ()
    retention_mode: Literal["derive_discard", "encrypted"] = "derive_discard"
    analyzer_id: str = ""
    analyzer_version: str = ""

    @field_validator("description")
    @classmethod
    def _check_description(cls, value: str) -> str:
        text = value.strip()
        if not text:
            msg = "voice profile description is empty"
            raise ValueError(msg)
        if len(text) > MAX_DESCRIPTION_CHARS:
            msg = (
                f"voice profile description is {len(text)} characters; "
                f"the maximum is {MAX_DESCRIPTION_CHARS}"
            )
            raise ValueError(msg)
        return text

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if value and not is_valid_voice_name(value):
            msg = f"invalid voice name {value!r}: use letters, digits, hyphen, and underscore only"
            raise ValueError(msg)
        return value

    def prompt_section(self) -> str:
        """The block injected into a rewrite prompt."""
        return self.description


def is_valid_voice_name(name: str) -> bool:
    """Whether ``name`` is safe to use as a store filename.

    Rejects path separators, ``..``, and leading dots so a name can never escape
    the voices directory or create a hidden file.
    """
    if not name or name.startswith("."):
        return False
    return all(char.isalnum() or char in "-_" for char in name)
