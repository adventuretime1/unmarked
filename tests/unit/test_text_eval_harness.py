"""Contract tests for the small, dependency-free evaluation harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2] / "evals" / "text"


def _load(name: str):
    if not ROOT.is_dir():
        pytest.skip("evaluation corpus is intentionally not included in this repository")
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hc3_manifest_is_small_pinned_and_paired() -> None:
    fetch_hc3 = _load("fetch_hc3")
    spec_path = Path(__file__).parents[2] / "evals" / "text" / "hc3_subset.json"
    spec = fetch_hc3.json.loads(spec_path.read_text())
    assert spec["revision"] == "4d0ff18143b5a7e1b1e79beb540c04549d1e59d3"
    assert spec["rows"] == list(range(15))
    assert spec["license"] == "CC-BY-SA-4.0"
    assert f"revision={spec['revision']}" in fetch_hc3._row_url(spec, 0)


def test_safe_unicode_variant_removes_a_known_carrier() -> None:
    harness = _load("run")
    text, operations = harness.safe_unicode("a\u200bb")
    assert text == "ab"
    assert operations == 1


def test_transform_safe_unicode_records_quality() -> None:
    transform = _load("transform")
    text, quality = transform._safe_unicode("a\u200bb")
    assert text == "ab"
    assert quality == {"state": "sanitized", "accepted": True, "operations": 1}


def test_summary_reports_detection_and_error_rates() -> None:
    harness = _load("run")
    records = [
        {"label": "ai", "prediction": "ai"},
        {"label": "ai", "prediction": "human"},
        {"label": "human", "prediction": "human"},
        {"label": "human", "prediction": "ai"},
    ]
    summary = harness.summarize(records)
    assert summary == {
        "records": 4,
        "accuracy": 0.5,
        "ai_samples": 2,
        "ai_detected": 1,
        "ai_detection_rate": 0.5,
        "ai_not_detected": 1,
        "human_samples": 2,
        "human_flagged": 1,
        "false_positive_rate": 0.5,
    }
