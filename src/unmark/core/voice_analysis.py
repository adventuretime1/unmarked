"""Deterministic, local writing-sample analysis for voice profiles."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import UTC, datetime

from unmark.core.operations import sha256_text
from unmark.core.voice import VoiceProfile, VoiceStyle

ANALYZER_ID = "unmark-statistical-voice"
ANALYZER_VERSION = "1"
MAX_SAMPLES = 20
MAX_SAMPLE_CHARS = 200_000

_WORDS = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+|\n+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_CONTRACTION = re.compile(
    r"\b(?:\w+(?:n't|'(?:d|ll|m|re|s|ve))|(?:can't|won't|shan't))\b",
    re.IGNORECASE,
)
_FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
_SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
_PUNCTUATION = {
    "em_dash": "—",
    "semicolon": ";",
    "colon": ":",
    "question": "?",
    "exclamation": "!",
    "parentheses": "(",
    "ellipsis": "…",
}


def _rate(count: int, words: int) -> float:
    return round((count / words) * 1000, 2) if words else 0.0


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BREAK.split(text) if _WORDS.search(part)]


def _description(style: VoiceStyle) -> str:
    mean = style.mean_sentence_words
    sentence_shape = (
        "mostly short" if mean < 10 else "mostly medium-length" if mean < 18 else "mostly long"
    )
    rhythm = (
        "fairly even"
        if style.sentence_word_stdev < 4
        else "moderately varied"
        if style.sentence_word_stdev < 8
        else "highly varied"
    )
    contractions = (
        "rare"
        if style.contraction_rate_per_1000_words < 2
        else "occasional"
        if style.contraction_rate_per_1000_words < 10
        else "frequent"
    )
    perspective: list[str] = []
    if style.first_person_rate_per_1000_words >= 5:
        perspective.append("uses first person naturally")
    if style.second_person_rate_per_1000_words >= 5:
        perspective.append("addresses the reader directly")
    perspective_text = (
        ", and ".join(perspective) if perspective else "avoids heavy personal address"
    )
    punctuation = sorted(
        style.punctuation_per_1000_words.items(),
        key=lambda item: (-item[1], item[0]),
    )
    visible = [name.replace("_", " ") for name, rate in punctuation if rate >= 1.0][:3]
    punctuation_text = (
        ", ".join(visible) if visible else "light punctuation beyond periods and commas"
    )
    return (
        f"Use {sentence_shape} sentences with a {rhythm} rhythm. Contractions are "
        f"{contractions}; the writer {perspective_text}. Typical punctuation includes "
        f"{punctuation_text}. Preserve the source's facts and register. Reproduce these "
        "aggregate habits without copying phrases from the samples or forcing an exact length."
    )


def analyze_voice_samples(
    name: str,
    samples: list[str] | tuple[str, ...],
    *,
    retention_mode: str = "derive_discard",
) -> VoiceProfile:
    """Derive an editable profile without returning or retaining raw samples."""
    cleaned = [sample.strip() for sample in samples if sample.strip()]
    if not cleaned:
        raise ValueError("at least one non-empty writing sample is required")
    if len(cleaned) > MAX_SAMPLES:
        raise ValueError(f"at most {MAX_SAMPLES} writing samples are allowed")
    if any(len(sample) > MAX_SAMPLE_CHARS for sample in cleaned):
        raise ValueError(f"each writing sample must be at most {MAX_SAMPLE_CHARS} characters")

    words = [word for sample in cleaned for word in _WORDS.findall(sample)]
    if len(words) < 25:
        raise ValueError("writing samples must contain at least 25 words in total")
    lowered = [word.lower().replace("’", "'") for word in words]
    sentences = [sentence for sample in cleaned for sentence in _sentences(sample)]
    sentence_lengths = [len(_WORDS.findall(sentence)) for sentence in sentences]
    paragraphs = [
        paragraph.strip()
        for sample in cleaned
        for paragraph in _PARAGRAPH_BREAK.split(sample)
        if _WORDS.search(paragraph)
    ]
    paragraph_sentence_counts = [max(1, len(_sentences(paragraph))) for paragraph in paragraphs]
    total_words = len(words)
    punctuation_counts = Counter(
        {key: sum(sample.count(mark) for sample in cleaned) for key, mark in _PUNCTUATION.items()}
    )
    style = VoiceStyle(
        sentence_count=len(sentence_lengths),
        mean_sentence_words=round(statistics.fmean(sentence_lengths), 2),
        sentence_word_stdev=round(statistics.pstdev(sentence_lengths), 2),
        short_sentence_ratio=round(
            sum(length <= 6 for length in sentence_lengths) / len(sentence_lengths), 4
        ),
        paragraph_count=len(paragraphs),
        mean_paragraph_sentences=round(statistics.fmean(paragraph_sentence_counts), 2),
        type_token_ratio=round(len(set(lowered)) / total_words, 4),
        mean_word_characters=round(statistics.fmean(len(word) for word in words), 2),
        contraction_rate_per_1000_words=_rate(
            sum(len(_CONTRACTION.findall(sample.replace("’", "'"))) for sample in cleaned),
            total_words,
        ),
        first_person_rate_per_1000_words=_rate(
            sum(word in _FIRST_PERSON for word in lowered), total_words
        ),
        second_person_rate_per_1000_words=_rate(
            sum(word in _SECOND_PERSON for word in lowered), total_words
        ),
        punctuation_per_1000_words={
            key: _rate(value, total_words) for key, value in punctuation_counts.items()
        },
    )
    return VoiceProfile(
        name=name,
        description=_description(style),
        generated_by=ANALYZER_ID,
        generated_at=datetime.now(UTC),
        structured_style=style,
        sample_count=len(cleaned),
        sample_word_count=total_words,
        sample_hashes=tuple(sha256_text(sample) for sample in cleaned),
        retention_mode="encrypted" if retention_mode == "encrypted" else "derive_discard",
        analyzer_id=ANALYZER_ID,
        analyzer_version=ANALYZER_VERSION,
    )
