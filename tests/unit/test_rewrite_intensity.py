"""The single low|medium|high rewrite dial and its expansion.

The dial is the primary user-facing control (CLI ``--intensity`` and the Chrome
extension). These tests pin the mapping's shape and the guarantees the UI relies
on: every level stays fidelity-gated (never breaks the text), higher intensity
does not do *less* work, and the expansion composes as a baseline that explicit
settings override.
"""

from __future__ import annotations

import pytest

from unmark.application.rewrite_service import _rewrite_budget
from unmark.orchestration.config import RewriteConfigSection, resolve_config
from unmark.strategies.rewrite.intensity import (
    REWRITE_INTENSITIES,
    RewriteIntensity,
    intensity_profile,
)


def test_levels_are_exactly_low_medium_high() -> None:
    assert REWRITE_INTENSITIES == ("low", "medium", "high")


@pytest.mark.parametrize("level", REWRITE_INTENSITIES)
def test_profile_expands_to_valid_config(level: RewriteIntensity) -> None:
    # Every level must produce a mapping that the config section accepts, so the
    # dial can never yield an invalid rewrite configuration.
    profile = intensity_profile(level)
    section = RewriteConfigSection(**profile)  # would raise on any bad field
    assert section.strategy in {"one-shot", "recursive"}


def test_low_is_conservative_high_is_stronger() -> None:
    low = intensity_profile("low")
    high = intensity_profile("high")
    # Low stays close to the source; high pushes harder.
    assert low["strength"] == "light"
    assert low["style"] == "lexical"
    assert low["candidate_count"] == 1
    assert high["strength"] == "strong"
    # High explores at least as many candidates as low and adds hops.
    assert high["candidate_count"] >= low["candidate_count"]
    assert high["strategy"] == "recursive"
    assert "rounds" in high
    assert high["style_schedule"] == ("syntax", "lexical", "polish")
    # Low is a single pass — no multi-hop budget implied.
    assert low["strategy"] == "one-shot"


def test_profile_is_a_fresh_mapping() -> None:
    # Callers merge and mutate the result; it must not alias shared state.
    first = intensity_profile("medium")
    first["candidate_count"] = 7
    assert intensity_profile("medium")["candidate_count"] == 3


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown rewrite intensity"):
        intensity_profile("extreme")  # type: ignore[arg-type]


def test_explicit_override_wins_over_profile() -> None:
    # This mirrors the CLI merge: the profile is the baseline, an explicit value
    # for the same key takes precedence.
    profile = intensity_profile("high")  # strength=strong
    explicit = {"strength": "light"}
    merged = {**profile, **explicit}
    assert merged["strength"] == "light"
    # Untouched keys still come from the profile.
    assert merged["strategy"] == "recursive"


def test_medium_rewrite_uses_rewrite_edit_budgets(tmp_path) -> None:
    resolved = resolve_config(
        cli_overrides={"preset": "sanitize", "rewrite": {"strength": "medium"}},
        start_dir=tmp_path,
        include_user_config=False,
    )
    budget = _rewrite_budget(resolved)
    assert budget.max_char_edit_ratio == 0.50
    assert budget.max_token_edit_ratio == 0.55
    assert budget.max_length_drift_ratio == 0.15


def test_explicit_edit_budget_remains_a_hard_rewrite_limit(tmp_path) -> None:
    config = tmp_path / "limits.toml"
    config.write_text(
        "[budget]\n"
        "max_char_edit_ratio = 0.11\n"
        "max_token_edit_ratio = 0.12\n"
        "max_length_drift_ratio = 0.03\n",
        encoding="utf-8",
    )
    resolved = resolve_config(
        cli_overrides={"preset": "sanitize", "rewrite": {"strength": "strong"}},
        explicit_config=config,
        start_dir=tmp_path,
        include_user_config=False,
    )
    budget = _rewrite_budget(resolved)
    assert budget.max_char_edit_ratio == 0.11
    assert budget.max_token_edit_ratio == 0.12
    assert budget.max_length_drift_ratio == 0.03
