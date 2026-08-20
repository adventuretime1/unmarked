"""JSON oracle used by the browser engine's cross-language parity test.

This is deliberately outside the production API. It exposes only deterministic
core values, omitting timestamps, run ids, and wall-clock usage.
"""

from __future__ import annotations

import json
import re
import sys

from unmark import __version__
from unmark.core.budgets import BudgetAccount, RunBudget
from unmark.core.diffs import operation_diff
from unmark.core.operations import sha256_text
from unmark.core.policies import FidelityPolicy, UnicodePolicy
from unmark.inspect.ingest import build_document
from unmark.orchestration.config import UnmarkedConfig
from unmark.orchestration.sanitation import SanitationStrategy
from unmark.reporting.build import build_document_summary, build_unicode_summary
from unmark.reporting.schema import SANITIZE_RESIDUAL_RISK


def evaluate(case: dict[str, object]) -> dict[str, object]:
    text = str(case["text"])
    media_type = str(case.get("media_type", "text/plain"))
    policy_name = str(case.get("policy", "safe"))
    raw_locks = case.get("locks", [])
    if not isinstance(raw_locks, list):
        raise ValueError("locks must be a JSON array")
    literal_locks = tuple(
        re.escape(value.strip()) for raw in raw_locks if (value := str(raw).strip())
    )
    fidelity = FidelityPolicy()
    document = build_document(
        text,
        media_type,  # type: ignore[arg-type]
        origin="-",
        fidelity=fidelity,
        locks=literal_locks,
    )
    policy = UnicodePolicy(name=policy_name)  # type: ignore[arg-type]
    budget = BudgetAccount(RunBudget(max_char_edit_ratio=0.02, max_length_drift_ratio=0.05))
    outcome = SanitationStrategy(policy).execute(document, budget, fidelity)
    notes: list[str] = []
    if not outcome.operations:
        notes.append("No actionable Unicode findings; the document is unchanged.")
    if outcome.blocked:
        notes.append(
            f"{len(outcome.blocked)} finding(s) were left in place because they fall "
            "inside a protected span."
        )
    config = UnmarkedConfig.model_validate(
        {
            "preset": "sanitize",
            "output": {"diff": "operations"},
            "budget": {"max_char_edit_ratio": 0.02},
            "unicode": {"policy": policy_name},
        }
    )
    return {
        "schema_version": "1",
        "kind": "edit",
        "tool_version": __version__,
        "state": "sanitized",
        "preset": "sanitize",
        "dry_run": False,
        "clean_text": outcome.candidate_text,
        "document": build_document_summary(document).model_dump(mode="json"),
        "unicode": build_unicode_summary(policy.name, outcome.findings, outcome.blocked).model_dump(
            mode="json"
        ),
        "operations": [op.model_dump(mode="json") for op in outcome.operations],
        "operation_count": len(outcome.operations),
        "output_sha256": sha256_text(outcome.candidate_text),
        "output_path": "-",
        "diff": operation_diff(text, outcome.operations),
        "usage": {
            "schema_version": "1",
            "model_calls": 0,
            "detector_queries": 0,
            "cost_usd": "0",
            "candidates": 1,
            "rounds": 0,
            "char_edit_ratio": outcome.candidate.char_edit_ratio,
            "token_edit_ratio": 0.0,
            "length_drift_ratio": outcome.candidate.length_drift_ratio,
        },
        "effective_config": config.model_dump(mode="json"),
        "residual_risk": SANITIZE_RESIDUAL_RISK,
        "notes": notes,
    }


def main() -> None:
    payload = json.load(sys.stdin)
    json.dump(
        [evaluate(case) for case in payload],
        sys.stdout,
        ensure_ascii=False,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
