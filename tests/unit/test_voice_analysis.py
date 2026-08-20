"""Deterministic writing-sample voice analysis."""

from __future__ import annotations

import pytest

from unmark.core.voice_analysis import ANALYZER_ID, analyze_voice_samples
from unmark.core.voice_model import (
    MODEL_VOICE_ANALYZER_ID,
    VoiceAnalysisError,
    analyze_voice_samples_from_completion,
    build_voice_analysis_request,
)

SAMPLE = (
    "I write short notes after each release. They are direct, practical, and tied to the "
    "work we just finished. I don't add a long introduction. When something failed, I name "
    "the command and the result.\n\nThe next paragraph usually says what I will try tomorrow. "
    "It may be longer because I include the reason, the constraint, and one concrete example."
)


def test_analysis_returns_measured_profile_without_raw_samples() -> None:
    profile = analyze_voice_samples("work", [SAMPLE])

    assert profile.generated_by == ANALYZER_ID
    assert profile.sample_count == 1
    assert profile.sample_word_count >= 25
    assert profile.structured_style is not None
    assert profile.structured_style.sentence_count >= 4
    assert len(profile.sample_hashes) == 1
    assert SAMPLE not in profile.model_dump_json()
    assert not any(character.isdigit() for character in profile.description)
    assert "forcing an exact length" in profile.description


def test_analysis_is_deterministic_except_timestamp() -> None:
    first = analyze_voice_samples("work", [SAMPLE])
    second = analyze_voice_samples("work", [SAMPLE])
    assert first.model_copy(update={"generated_at": None}) == second.model_copy(
        update={"generated_at": None}
    )


def test_analysis_rejects_too_little_evidence() -> None:
    with pytest.raises(ValueError, match="at least 25 words"):
        analyze_voice_samples("work", ["A tiny sample."])


def test_analysis_records_encrypted_retention_choice() -> None:
    profile = analyze_voice_samples("work", [SAMPLE], retention_mode="encrypted")
    assert profile.retention_mode == "encrypted"


QUALITATIVE = """{
  "prose": "Uses developed sentences with varied rhythm and plain transitions",
  "speaking_style": "Sounds direct, practical, and calmly conversational",
  "casual_register": "Moves quickly to the point and uses contractions naturally",
  "professional_register": "Explains constraints before proposing the next action",
  "disagreement_style": "States the concern directly while preserving uncertainty",
  "tells": ["pairs a claim with a concrete reason", "uses restrained emphasis"],
  "avoid": ["ornate introductions", "sales language"]
}"""


def test_model_profile_is_qualitative_and_keeps_samples_out() -> None:
    profile = analyze_voice_samples_from_completion("work", [SAMPLE], QUALITATIVE)
    assert profile.generated_by == MODEL_VOICE_ANALYZER_ID
    assert "Speaking style:" in profile.description
    assert not any(character.isdigit() for character in profile.description)
    assert SAMPLE not in profile.model_dump_json()


def test_voice_prompt_calibrates_across_contexts_without_numeric_targets() -> None:
    request = build_voice_analysis_request([SAMPLE])
    assert "casual update" in request.prompt
    assert "respectful disagreement" in request.prompt
    assert "Do not include digits" in request.prompt
    assert request.temperature == 0.2


def test_model_profile_rejects_numeric_guidance() -> None:
    with pytest.raises(VoiceAnalysisError, match="numeric guidance"):
        analyze_voice_samples_from_completion(
            "work",
            [SAMPLE],
            QUALITATIVE.replace("developed sentences", "sentences averaging 23 words"),
        )
