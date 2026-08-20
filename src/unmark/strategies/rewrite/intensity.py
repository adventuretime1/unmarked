"""The single user-facing rewrite dial: ``low | medium | high``.

Users should not have to reason about strategy, strength, candidate count, hops,
and temperature separately. Those are one axis in practice — "stay close to the
original" versus "allow a broader paraphrase" — so we expose exactly one control
and expand it here into concrete rewrite settings.

``intensity_profile`` returns a *config-override mapping* in the same shape the
CLI already produces, so it composes with the existing override layer: a profile
sets the baseline, and any setting the user specified explicitly still wins. The
mapping is pure and deterministic, and the effective config it produces is
recorded in the run report, so the expansion is always auditable.

Honesty: ``high`` means "explore broader changes" — never "guaranteed
undetectable" and never a licence to break fidelity. The hard fidelity gates
apply identically at every level; a higher intensity only widens how much
*valid* change is explored.
"""

from __future__ import annotations

from typing import Literal, TypedDict, cast, get_args

from unmark.strategies.rewrite.prompts import RewriteStrength, RewriteStyle

RewriteIntensity = Literal["low", "medium", "high"]
"""How much the rewrite may change the text, as one dial.

* ``low``    – stay close to the original; small, conservative edits.
* ``medium`` – a balanced rewrite (the default when the dial is used).
* ``high``   – explore broader changes across multiple passes while keeping the
  same fidelity gates.
"""

REWRITE_INTENSITIES: tuple[RewriteIntensity, ...] = get_args(RewriteIntensity)


class RewriteIntensityProfile(TypedDict, total=False):
    """Typed subset of rewrite settings controlled by the intensity dial."""

    strategy: Literal["one-shot", "recursive"]
    style: RewriteStyle
    strength: RewriteStrength
    candidate_count: int
    temperature: float
    rounds: int
    style_schedule: tuple[RewriteStyle, ...]


#: Each level expands to concrete rewrite settings. These are *baseline*
#: overrides: an explicit CLI flag for any of these keys takes precedence. Only
#: keys a profile actually sets appear, so unrelated config is left untouched.
_PROFILES: dict[RewriteIntensity, RewriteIntensityProfile] = {
    "low": {
        "strategy": "one-shot",
        "style": "lexical",
        "strength": "light",
        "candidate_count": 1,
        "temperature": 0.4,
    },
    "medium": {
        "strategy": "one-shot",
        "style": "lexical",
        "strength": "medium",
        "candidate_count": 3,
        "temperature": 0.7,
    },
    "high": {
        "strategy": "recursive",
        "style": "lexical",
        "strength": "strong",
        "candidate_count": 3,
        "temperature": 0.9,
        "rounds": 3,
        "style_schedule": ("syntax", "lexical", "polish"),
    },
}


def intensity_profile(level: RewriteIntensity) -> RewriteIntensityProfile:
    """Expand an intensity level into baseline rewrite config overrides.

    Returns a fresh mapping keyed by :class:`RewriteConfigSection` field names.
    Callers merge it *under* any explicit per-field overrides so a user who sets
    the dial and also pins one knob keeps their pinned knob.
    """
    if level not in _PROFILES:
        allowed = ", ".join(REWRITE_INTENSITIES)
        msg = f"unknown rewrite intensity {level!r}; expected one of {allowed}"
        raise ValueError(msg)
    return cast(RewriteIntensityProfile, dict(_PROFILES[level]))
