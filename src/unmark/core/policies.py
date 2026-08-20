"""Fidelity and Unicode policy contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from unmark.core.spans import StrictModel

FidelityLevel = Literal["strict", "standard", "creative"]

UnicodePolicyName = Literal["report", "safe", "typographic", "aggressive"]
"""Unicode handling policies.

``report``
    Find and report only; never mutate.
``safe``
    Remove unambiguous invisible carriers that have no linguistic role in the
    surrounding script (e.g. zero-width space, tag characters, most C0/C1
    controls). Script-aware: ZWJ/ZWNJ, bidi controls, and variation selectors are
    preserved wherever they can be meaningful.
``typographic``
    ``safe`` plus conservative whitespace normalization: unusual spaces become a
    normal space where doing so cannot change meaning. NBSP is preserved.
``aggressive``
    Research-only. Removes format characters regardless of script context and can
    corrupt legitimate text. Requires an explicit research-mode acknowledgement
    and is never reachable from a normal preset.
"""


class FidelityPolicy(StrictModel):
    """Hard preservation requirements.

    Only the deterministic locks are enforced in this iteration; the semantic
    fields are part of the frozen contract and are consumed from Phase 3 onwards.
    """

    schema_version: Literal["1"] = "1"
    level: FidelityLevel = "strict"
    lock_numbers: bool = True
    lock_citations: bool = True
    lock_quotes: bool = True
    lock_urls: bool = True
    lock_code: bool = True
    lock_dates: bool = True
    lock_units: bool = True
    lock_identifiers: bool = True
    require_bidirectional_entailment: bool = True
    min_claim_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    max_new_claims: int = Field(default=0, ge=0)
    min_style_similarity: float = Field(default=0.80, ge=0.0, le=1.0)
    domain_validator: str | None = None


class UnicodePolicy(StrictModel):
    """Configuration for the Unicode inspector and sanitizer."""

    schema_version: Literal["1"] = "1"
    name: UnicodePolicyName = "safe"
    preserve_emoji_sequences: bool = True
    preserve_language_controls: bool = True
    normalize_spaces: bool = True
    normalization_form: Literal["none", "NFC", "NFKC"] = "NFC"

    @property
    def mutates(self) -> bool:
        return self.name != "report"

    @property
    def is_research_only(self) -> bool:
        return self.name == "aggressive"
