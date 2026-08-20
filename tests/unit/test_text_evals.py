"""Regression runner for the intentionally small text-signal evaluation set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unmark.core.policies import UnicodePolicy
from unmark.inspect.unicode_scan import inspect_text, sanitation_operations


def test_tiny_text_signal_evaluation_set() -> None:
    root = Path(__file__).parents[2] / "evals" / "text"
    if not root.is_dir():
        pytest.skip("evaluation corpus is intentionally not included in this repository")
    cases = json.loads((root / "cases.json").read_text())
    policy = UnicodePolicy(name="safe")
    for case in cases:
        text = (root / case["file"]).read_text()
        findings = inspect_text(text, policy)
        operations = sanitation_operations(text, findings, policy)
        assert len(operations) == case["actionable"], case["file"]
