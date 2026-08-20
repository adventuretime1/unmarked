"""Dependency-free contract tests for the release smoke evaluation harness."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from unmark.application.requests import EditRequest

ROOT = Path(__file__).parents[2] / "evals"


def _require_evals() -> None:
    if not ROOT.is_dir():
        pytest.skip("evaluation corpus is intentionally not included in this repository")


def test_edit_request_accepts_in_memory_binary_stdin() -> None:
    request = EditRequest(input="-", stdin=io.BytesIO(b"sample"))
    assert request.stdin is not None
    assert request.stdin.read() == b"sample"


def _load(path: Path):
    _require_evals()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_watermark_matrix_is_exactly_twenty_small_cases() -> None:
    generator = _load(ROOT / "text" / "generate_watermarked.py")
    prompts = json.loads((ROOT / "text" / "prompts.json").read_text())
    assert len(prompts) == 10
    assert set(generator.SCHEMES) == {"kgw-lefthash", "kgw-selfhash"}
    assert len(prompts) * len(generator.SCHEMES) == 20
    assert generator.MODEL_REVISION == "7ae557604adf67be50417f59c2c2f167def9a775"


def test_organized_runner_refuses_repository_output_and_logs_stages(tmp_path: Path) -> None:
    runner = _load(ROOT / "run_smoke.py")
    with pytest.raises(ValueError, match="outside the repository"):
        runner._outside_repository(ROOT / "runs" / "bad")

    run = runner.layout(tmp_path / "run")
    run.logs.mkdir(parents=True)
    result = runner._stage(
        "fixture",
        [sys.executable, "-c", 'import json; print(json.dumps({"passed": 1}))'],
        run,
        dict(os.environ),
    )
    assert result == {"passed": 1}
    log = json.loads((run.logs / "fixture.json").read_text())
    assert log["returncode"] == 0
    assert log["stage"] == "fixture"


def test_organized_runner_requires_detectable_watermarked_sources() -> None:
    runner = _load(ROOT / "run_smoke.py")
    runner._require_detectable_sources({"watermarked_records": 20, "detection_rate": 0.95}, 0.8)
    with pytest.raises(RuntimeError, match="qualification failed"):
        runner._require_detectable_sources({"watermarked_records": 20, "detection_rate": 0.75}, 0.8)


def test_organized_runner_reads_only_named_env_value(tmp_path: Path) -> None:
    runner = _load(ROOT / "run_smoke.py")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "UNRELATED=ignore-me\nexport OPENROUTER_API_KEY='test-key'\n",
        encoding="utf-8",
    )
    assert runner._read_named_env_value(env_file, "OPENROUTER_API_KEY") == "test-key"
    assert runner._read_named_env_value(env_file, "MISSING") is None


def test_organized_runner_rejects_duplicate_named_env_values(tmp_path: Path) -> None:
    runner = _load(ROOT / "run_smoke.py")
    env_file = tmp_path / ".env"
    env_file.write_text("TOKEN=first\nTOKEN=second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate values"):
        runner._read_named_env_value(env_file, "TOKEN")


def test_organized_runner_imports_validated_source_corpus(tmp_path: Path) -> None:
    runner = _load(ROOT / "run_smoke.py")
    source = tmp_path / "source.jsonl"
    records = [
        {
            "id": f"sample-{index}",
            "watermark_scheme": "kgw-lefthash" if index < 20 else None,
            "watermark_device": "mps",
        }
        for index in range(30)
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    run = runner.layout(tmp_path / "run")
    run.logs.mkdir(parents=True)
    destination = run.corpora / "original.jsonl"

    result = runner._import_source_corpus(source, destination, run)

    assert result["records_written"] == 30
    assert result["watermarked"] == 20
    assert result["controls"] == 10
    assert result["watermark_device"] == "mps"
    assert destination.read_bytes() == source.read_bytes()
    assert (run.logs / "import-source.json").is_file()


def test_comparison_reports_detection_flips_and_quality(tmp_path: Path) -> None:
    comparator = _load(ROOT / "text" / "compare.py")
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    before.write_text(
        json.dumps(
            {
                "id": "one",
                "text_sha256": "before",
                "watermark_scheme": "kgw-lefthash",
                "watermark_detection": {"detected": True, "z_score": 6.0},
            }
        )
        + "\n"
    )
    after.write_text(
        json.dumps(
            {
                "id": "one",
                "text_sha256": "after",
                "watermark_scheme": "kgw-lefthash",
                "watermark_detection": {"detected": False, "z_score": 1.0},
                "quality": {"accepted": True},
            }
        )
        + "\n"
    )
    result = comparator.compare(before, after)
    assert result["observed_detection_flip_with_quality"] is True
    assert result["overall"]["unmark_rate"] == 1.0
    assert result["overall"]["mean_z_reduction"] == 5.0


def test_safe_unicode_transform_writes_a_paired_variant(tmp_path: Path) -> None:
    transformer = _load(ROOT / "text" / "transform.py")
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text(json.dumps({"id": "carrier", "text": "a\u200bb"}) + "\n")
    result = transformer.transform(
        source,
        output,
        Namespace(method="safe-unicode"),
        tmp_path / "workspace",
    )
    record = json.loads(output.read_text())
    assert result == {"records": 1, "accepted": 1, "changed": 1}
    assert record["text"] == "ab"
    assert record["variant"] == "safe-unicode"
    assert record["quality"]["operations"] == 1


def test_comparison_reports_unwatermarked_control_false_positives(tmp_path: Path) -> None:
    comparator = _load(ROOT / "text" / "compare.py")
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    control = {
        "id": "control",
        "text_sha256": "same",
        "watermark_scheme": None,
        "watermark_detection": None,
        "watermark_control_detections": {
            "kgw-lefthash": {"detected": False, "z_score": 0.1},
            "kgw-selfhash": {"detected": True, "z_score": 3.2},
        },
    }
    before.write_text(json.dumps(control) + "\n")
    after.write_text(json.dumps({**control, "quality": {"accepted": True}}) + "\n")
    result = comparator.compare(before, after)
    assert result["controls"] == {
        "tests": 2,
        "false_positives": 1,
        "false_positive_rate": 0.5,
    }


def test_ten_document_fixtures_clean_and_preserve_content(tmp_path: Path) -> None:
    _require_evals()
    documents = _load(ROOT / "documents" / "run.py")
    result = documents.evaluate(
        ROOT / "documents" / "cases.json", tmp_path / "document-results.jsonl"
    )
    assert result == {"documents": 10, "passed": 10, "pass_rate": 1.0}
