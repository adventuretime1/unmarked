"""Simple text-editing baselines: prompt-driven whole-document rewriting.

These strategies ask a model to rewrite the document, then treat the reply as
*untrusted* input: every candidate is materialized as a single source-anchored
replace operation and pushed through the same fidelity, budget, and
protected-span validation the targeted search uses. No candidate is accepted on
the model's say-so.

Prompt styles and strengths control how candidates are generated. Candidate
selection remains conservative: it minimizes validated edits instead of
maximizing lexical divergence.
"""

from __future__ import annotations

from unmark.strategies.rewrite.candidates import (
    RewriteCandidate,
    bigram_jaccard_divergence,
    select_candidate,
    selection_key,
)
from unmark.strategies.rewrite.config import RewriteConfig
from unmark.strategies.rewrite.engine import RewriteEngine, RewriteStepResult
from unmark.strategies.rewrite.one_shot import OneShotRewriteStrategy
from unmark.strategies.rewrite.prompts import (
    REWRITE_STRENGTHS,
    REWRITE_STYLES,
    RewritePrompt,
    RewriteStrength,
    RewriteStyle,
    build_rewrite_prompt,
)
from unmark.strategies.rewrite.recursive import RecursiveRewriteStrategy
from unmark.strategies.rewrite.result import RewriteResult, RewriteTrace

__all__ = [
    "REWRITE_STRENGTHS",
    "REWRITE_STYLES",
    "OneShotRewriteStrategy",
    "RecursiveRewriteStrategy",
    "RewriteCandidate",
    "RewriteConfig",
    "RewriteEngine",
    "RewritePrompt",
    "RewriteResult",
    "RewriteStepResult",
    "RewriteStrength",
    "RewriteStyle",
    "RewriteTrace",
    "bigram_jaccard_divergence",
    "build_rewrite_prompt",
    "select_candidate",
    "selection_key",
]
