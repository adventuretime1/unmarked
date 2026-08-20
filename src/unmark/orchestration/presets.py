"""Versioned presets.

A preset expands to an immutable configuration recorded in the report. Changing a
preset's behavior requires a **new version** so an old run stays reproducible;
never mutate an existing entry.

Only ``sanitize`` is executable in this slice. The others are declared so the CLI
can report honestly that they are not yet available, rather than silently doing
something weaker than the name implies.
"""

from __future__ import annotations

from typing import Literal

from unmark.core.policies import FidelityPolicy, UnicodePolicy
from unmark.core.spans import StrictModel
from unmark.core.targets import ReductionTarget

PresetName = Literal["sanitize", "light", "balanced", "strong", "offline-max"]

AVAILABLE_PRESETS: frozenset[str] = frozenset({"sanitize"})


class Preset(StrictModel):
    """An immutable, versioned bundle of policy defaults."""

    schema_version: Literal["1"] = "1"
    name: str
    version: str
    description: str
    available: bool
    unicode: UnicodePolicy
    fidelity: FidelityPolicy
    target: ReductionTarget
    max_char_edit_ratio: float = 0.0
    requires: tuple[str, ...] = ()


_SANITIZE = Preset(
    name="sanitize",
    version="1",
    description=(
        "Deterministic, script-aware Unicode hygiene. No model calls, no intended "
        "wording change, and no claim of statistical-watermark reduction."
    ),
    available=True,
    unicode=UnicodePolicy(name="safe"),
    fidelity=FidelityPolicy(level="strict"),
    target=ReductionTarget(mode="sanitize_only"),
    max_char_edit_ratio=0.02,
)


def _deferred(name: str, description: str, requires: tuple[str, ...]) -> Preset:
    return Preset(
        name=name,
        version="0",
        description=description,
        available=False,
        unicode=UnicodePolicy(name="safe"),
        fidelity=FidelityPolicy(level="strict"),
        target=ReductionTarget(mode="sanitize_only"),
        requires=requires,
    )


PRESETS: dict[str, Preset] = {
    "sanitize": _SANITIZE,
    "light": _deferred(
        "light",
        "Sanitation plus bounded surface edits and small targeted search.",
        ("rewrite_model",),
    ),
    "balanced": _deferred(
        "balanced",
        "Light plus constrained one-shot rewrite and targeted beam search.",
        ("rewrite_model", "detector_or_surrogate"),
    ),
    "strong": _deferred(
        "strong",
        "Balanced plus recursive hops and a bounded random walk.",
        ("rewrite_model", "detector_or_surrogate"),
    ),
    "offline-max": _deferred(
        "offline-max",
        "All production strategies plus evolutionary Pareto search.",
        ("rewrite_model", "detector_or_surrogate", "offline_runtime"),
    ),
}


def get_preset(name: str) -> Preset:
    """Look up a preset by name, or raise :class:`UnsupportedError`."""
    from unmark.core.errors import UnsupportedError, UsageError

    try:
        preset = PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        msg = f"unknown preset {name!r}. Known presets: {known}"
        raise UsageError(msg) from None

    if not preset.available:
        needs = ", ".join(preset.requires)
        msg = (
            f"preset {name!r} is not available in this build: it requires {needs}, "
            "which is deferred to a later phase. Use --preset sanitize."
        )
        raise UnsupportedError(msg)
    return preset
