"""Prompt construction for the simple rewrite baselines.

A rewrite prompt is fully deterministic given (style, strength, text, locks): the
same inputs render the same bytes, which is what the ``print-prompt`` backend and
the prompt-preservation tests rely on.

Every prompt carries an explicit, non-negotiable preservation contract. The model
is told to keep facts and atomic claims, numbers/dates/units/identifiers/named
entities, quotations and citations, URLs/code/user-locked spans, negation,
modality, caveats, register, approximate length, and Markdown structure. Those
instructions are advisory only — Unmarked still validates the reply — but stating
them narrows the model toward edits that will actually survive validation.
"""

from __future__ import annotations

from typing import Literal

from unmark.core.spans import StrictModel

RewriteStyle = Literal[
    "faithful",
    "syntax",
    "lexical",
    "polish",
    "simplify",
    "academic",
]
"""How the rewrite should read.

* ``faithful``  – minimal touch; only smooth what is necessary.
* ``syntax``    – reorder and restructure sentences, keep the wording.
* ``lexical``   – substitute wording/synonyms, keep the sentence shapes.
* ``polish``    – light copy-edit for flow and clarity.
* ``simplify``  – plainer language, shorter sentences, same content.
* ``academic``  – measured, formal register.
"""

RewriteStrength = Literal["light", "medium", "strong"]
"""How far to push the chosen style, from a gentle touch to a firm rewrite."""

REWRITE_STYLES: tuple[RewriteStyle, ...] = (
    "faithful",
    "syntax",
    "lexical",
    "polish",
    "simplify",
    "academic",
)

REWRITE_STRENGTHS: tuple[RewriteStrength, ...] = ("light", "medium", "strong")

_STYLE_INSTRUCTIONS: dict[RewriteStyle, str] = {
    "faithful": (
        "Rewrite as little as possible. Fix only what is awkward or unclear and "
        "leave everything else exactly as written."
    ),
    "syntax": (
        "Vary sentence structure: reorder clauses, split or merge sentences, and "
        "change the flow. Keep the vocabulary and phrasing close to the original."
    ),
    "lexical": (
        "Vary word choice: replace words and short phrases with natural synonyms. "
        "Keep sentence structure and length close to the original."
    ),
    "polish": (
        "Copy-edit for clarity and flow. Improve readability without changing the "
        "content, structure, or level of detail."
    ),
    "simplify": (
        "Use plainer language and shorter sentences while conveying exactly the "
        "same information. Do not omit or add content."
    ),
    "academic": (
        "Adopt a measured, formal, academic register while preserving the meaning, "
        "content, and structure of the original."
    ),
}

_STRENGTH_INSTRUCTIONS: dict[RewriteStrength, str] = {
    "light": (
        "Make a noticeable but conservative pass. Change wording or flow wherever "
        "a natural alternative exists; do not return the source unchanged unless no "
        "safe edit is possible."
    ),
    "medium": "Apply this moderately across the text.",
    "strong": (
        "Apply the chosen style throughout the text and substantially reshape its "
        "expression, but never at the cost of the rules below."
    ),
}

#: The preservation contract, stated to the model verbatim on every request. The
#: engine enforces the machine-checkable parts (numbers, URLs, quotations,
#: locked spans, length) independently; the rest guide the model.
PRESERVATION_RULES: tuple[str, ...] = (
    "Preserve every fact and atomic claim. Do not add, drop, or alter information.",
    "Reproduce all numbers, dates, times, quantities, units, currencies, "
    "identifiers, and named entities exactly.",
    "Reproduce all quotations and citations verbatim, including punctuation.",
    "Reproduce all URLs, file paths, code, and inline/fenced code spans exactly.",
    "Keep any text the user has locked exactly as given.",
    "Preserve negation, modality, hedging, caveats, and qualifiers. Do not turn "
    '"may" into "will" or drop a "not".',
    "Preserve the tone and register of the original.",
    "Keep the length close to the original; do not summarize or expand.",
    "Preserve Markdown structure: headings, lists, tables, links, emphasis, and code fences.",
    "Return only the rewritten text, with no preamble, explanation, or commentary.",
)

_SYSTEM_PROMPT = (
    "You are a careful copy editor. Rewrite at the requested intensity so the "
    "source's token choices, repeated n-grams, sentence patterns, and model-probability "
    "patterns are less likely to persist. Preserve the meaning exactly: never invent, "
    "omit, or change factual content. Output only the rewritten text."
)


class RewritePrompt(StrictModel):
    """A fully rendered rewrite prompt plus the parameters that produced it."""

    style: RewriteStyle
    strength: RewriteStrength
    system: str
    user: str


def _format_voice(voice: str) -> str:
    """Render a voice description into the prompt.

    Placed after the preservation rules and before the text: the description
    guides *how* the rewrite reads, and must never license changing *what* it
    says. Empty when no profile was supplied, so prompts stay byte-identical to
    the no-voice case.
    """
    description = voice.strip()
    if not description:
        return ""
    return (
        "\n\nWrite in the following voice. This governs tone, rhythm, and word "
        "choice only; it never overrides the rules above, and you must not add, "
        "drop, or alter any content to fit it:\n"
        f"{description}"
    )


def _format_locks(locked_spans: tuple[str, ...]) -> str:
    if not locked_spans:
        return ""
    lines = "\n".join(f"- {span}" for span in locked_spans)
    return f"\n\nThe following spans are locked and must appear unchanged in your output:\n{lines}"


def build_rewrite_prompt(
    text: str,
    *,
    style: RewriteStyle = "faithful",
    strength: RewriteStrength = "medium",
    locked_spans: tuple[str, ...] = (),
    target_length_ratio: float | None = None,
    voice: str = "",
) -> RewritePrompt:
    """Render a deterministic rewrite prompt.

    ``locked_spans`` are surfaced to the model verbatim (the engine also enforces
    them as protected spans). ``target_length_ratio`` is expressed as a soft
    instruction; the hard length-drift limit lives in the budget. ``voice`` is a
    description of how the author writes, from a stored voice profile.
    """
    rules = "\n".join(f"{index}. {rule}" for index, rule in enumerate(PRESERVATION_RULES, start=1))
    length_note = ""
    if target_length_ratio is not None:
        pct = round(target_length_ratio * 100)
        length_note = f"\n\nAim for roughly {pct}% of the original length."
    user = (
        f"{_STYLE_INSTRUCTIONS[style]} {_STRENGTH_INSTRUCTIONS[strength]}\n\n"
        f"Rules you must follow:\n{rules}"
        f"{length_note}"
        f"{_format_voice(voice)}"
        f"{_format_locks(locked_spans)}\n\n"
        f"Text to rewrite:\n{text}"
    )
    return RewritePrompt(style=style, strength=strength, system=_SYSTEM_PROMPT, user=user)
