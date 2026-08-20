"""End-to-end CLI behavior: streams, files, exit codes, and reports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from unmark.core.errors import ExitCode

FIXTURES = Path(__file__).parent.parent / "fixtures"

ZWSP = "​"


def run_unmark(
    *args: str, cwd: Path, stdin: str | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Invoke the CLI as a subprocess, which is how users actually run it."""
    return subprocess.run(
        [sys.executable, "-m", "unmark.cli.app", *args],
        cwd=cwd,
        input=stdin.encode("utf-8") if stdin is not None else b"",
        env={**os.environ, **(env or {})},
        capture_output=True,
        check=False,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("")
    return tmp_path


@pytest.fixture
def draft(workspace: Path) -> Path:
    path = workspace / "draft.md"
    path.write_text(
        f"# Report{ZWSP}\n\nValue 42% on 2026-08-13 via https://example.com/a.\n",
        encoding="utf-8",
    )
    return path


class TestVersionAndHelp:
    def test_version(self, workspace):
        result = run_unmark("--version", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        assert result.stdout.decode().startswith("unmark ")

    def test_help(self, workspace):
        result = run_unmark("--help", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS

    def test_help_explains_scope_without_unverified_detector_claims(self, workspace):
        result = run_unmark("--help", cwd=workspace)
        # Rich wraps the help across lines, so normalize whitespace before matching.
        text = " ".join(result.stdout.decode().lower().split())
        assert "removes recognized hidden carriers" in text
        assert "requires review" in text
        assert "model-probability patterns" not in text

    def test_attachment_help_lists_supported_formats(self, workspace):
        result = run_unmark("attachment", "--help", cwd=workspace)
        text = " ".join(result.stdout.decode().split())
        assert result.returncode == ExitCode.SUCCESS
        assert "PDF" in text
        assert "DOCX" in text
        assert "frontmatter-bearing Markdown" in text


class TestInspect:
    def test_reports_structure_and_findings(self, draft, workspace):
        result = run_unmark("inspect", str(draft), cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        stderr = result.stderr.decode()
        assert "U+200B" in stderr
        assert "text/markdown" in stderr

    def test_json_output_goes_to_stdout(self, draft, workspace):
        result = run_unmark("inspect", str(draft), "--format", "json", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        payload = json.loads(result.stdout.decode())
        assert payload["kind"] == "inspect"
        assert payload["document"]["media_type"] == "text/markdown"
        assert payload["unicode"]["findings"][0]["codepoint"] == 0x200B
        assert "read-only inspection changed nothing" in payload["residual_risk"].lower()

    def test_is_read_only(self, draft, workspace):
        original = draft.read_bytes()
        before = {p.name for p in workspace.iterdir()}
        run_unmark("inspect", str(draft), cwd=workspace)
        assert draft.read_bytes() == original
        assert {p.name for p in workspace.iterdir()} == before

    def test_stdin_input(self, workspace):
        result = run_unmark("inspect", "-", cwd=workspace, stdin=f"text{ZWSP}here\n")
        assert result.returncode == ExitCode.SUCCESS
        assert "U+200B" in result.stderr.decode()

    def test_unknown_extension_is_unsupported(self, workspace):
        path = workspace / "file.rst"
        path.write_text("content")
        result = run_unmark("inspect", str(path), cwd=workspace)
        assert result.returncode == ExitCode.UNSUPPORTED

    def test_missing_input_is_usage_error(self, workspace):
        result = run_unmark("inspect", "absent.md", cwd=workspace)
        assert result.returncode == ExitCode.USAGE

    def test_invalid_utf8_is_unsupported(self, workspace):
        path = workspace / "bad.txt"
        path.write_bytes(b"\xff\xfe invalid")
        result = run_unmark("inspect", str(path), cwd=workspace)
        assert result.returncode == ExitCode.UNSUPPORTED

    def test_report_file_written(self, draft, workspace):
        report = workspace / "report.json"
        result = run_unmark("inspect", str(draft), "--report", str(report), cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        assert json.loads(report.read_text())["kind"] == "inspect"


class TestSkills:
    def test_installs_all_bundled_skills(self, workspace):
        target = workspace / "agent-skills"
        result = run_unmark("skills", "install", "--target", str(target), cwd=workspace)

        assert result.returncode == ExitCode.SUCCESS
        assert {path.name for path in target.iterdir()} == {
            "unmark-clean",
            "unmark-rewrite",
            "voice-analysis",
        }
        assert all((path / "SKILL.md").is_file() for path in target.iterdir())

    def test_refuses_to_replace_installed_skill_without_force(self, workspace):
        target = workspace / "agent-skills"
        first = run_unmark("skills", "install", "--target", str(target), cwd=workspace)
        second = run_unmark("skills", "install", "--target", str(target), cwd=workspace)

        assert first.returncode == ExitCode.SUCCESS
        assert second.returncode == ExitCode.USAGE
        assert "--force" in second.stderr.decode()


class TestEditOutput:
    def test_default_sibling_output(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--preset", "sanitize", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        output = workspace / "draft.unmark.md"
        assert output.exists()
        assert ZWSP not in output.read_text(encoding="utf-8")

    def test_source_is_never_modified(self, draft, workspace):
        original = draft.read_bytes()
        run_unmark("edit", str(draft), "--preset", "sanitize", cwd=workspace)
        assert draft.read_bytes() == original

    def test_output_to_stdout(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--output", "-", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        stdout = result.stdout.decode()
        assert "# Report" in stdout
        assert ZWSP not in stdout

    def test_text_never_goes_to_stdout_by_default(self, draft, workspace):
        result = run_unmark("edit", str(draft), cwd=workspace)
        assert "# Report" not in result.stdout.decode()

    def test_diagnostics_go_to_stderr(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--output", "-", cwd=workspace)
        assert "sanitized" in result.stderr.decode()

    def test_explicit_output_path(self, draft, workspace):
        target = workspace / "clean.md"
        result = run_unmark("edit", str(draft), "--output", str(target), cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        assert target.exists()

    def test_stdin_to_stdout(self, workspace):
        result = run_unmark("edit", "-", cwd=workspace, stdin=f"hello{ZWSP}world\n")
        assert result.returncode == ExitCode.SUCCESS
        assert result.stdout.decode() == "helloworld\n"


class TestEditRefusals:
    def test_existing_output_refused(self, draft, workspace):
        existing = workspace / "draft.unmark.md"
        existing.write_text("do not clobber")
        result = run_unmark("edit", str(draft), cwd=workspace)
        assert result.returncode == ExitCode.USAGE
        assert existing.read_text() == "do not clobber"

    def test_force_replaces_existing_output(self, draft, workspace):
        existing = workspace / "draft.unmark.md"
        existing.write_text("old")
        result = run_unmark("edit", str(draft), "--force", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        assert existing.read_text() != "old"

    def test_overwriting_source_refused(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--output", str(draft), "--force", cwd=workspace)
        assert result.returncode == ExitCode.USAGE
        assert "source" in result.stderr.decode().lower()

    def test_symlink_destination_refused(self, draft, workspace):
        real = workspace / "real.md"
        real.write_text("content")
        link = workspace / "link.md"
        link.symlink_to(real)
        result = run_unmark("edit", str(draft), "--output", str(link), "--force", cwd=workspace)
        assert result.returncode == ExitCode.USAGE
        assert "symlink" in result.stderr.decode().lower()
        assert real.read_text() == "content"

    def test_deferred_preset_is_unsupported(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--preset", "balanced", cwd=workspace)
        assert result.returncode == ExitCode.UNSUPPORTED

    def test_unknown_preset_is_usage_error(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--preset", "nonsense", cwd=workspace)
        assert result.returncode == ExitCode.USAGE

    def test_aggressive_policy_requires_research_mode(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--unicode-policy", "aggressive", cwd=workspace)
        assert result.returncode == ExitCode.USAGE
        assert "research" in result.stderr.decode().lower()

    def test_yes_does_not_enable_research_mode(self, draft, workspace):
        result = run_unmark(
            "edit", str(draft), "--unicode-policy", "aggressive", "--yes", cwd=workspace
        )
        assert result.returncode == ExitCode.USAGE

    def test_research_mode_enables_aggressive(self, draft, workspace):
        result = run_unmark(
            "edit",
            str(draft),
            "--unicode-policy",
            "aggressive",
            "--research-mode",
            "--output",
            "-",
            cwd=workspace,
        )
        assert result.returncode == ExitCode.SUCCESS

    def test_model_backed_unsupported_rewrite_exits_nonzero(self, workspace):
        source = workspace / "source.txt"
        source.write_text("Plain source text.\n", encoding="utf-8")
        result = run_unmark(
            "edit",
            str(source),
            "--rewrite",
            "--backend",
            "ollama",
            "--model",
            "definitely-not-installed",
            "--endpoint",
            "http://127.0.0.1:1",
            "--source-provider",
            "human",
            "--format",
            "json",
            cwd=workspace,
        )

        assert result.returncode == ExitCode.UNSUPPORTED
        payload = json.loads(result.stdout.decode())
        assert payload["state"] == "unsupported"
        assert payload["output_path"] is None
        assert not (workspace / "source.unmark.txt").exists()


class TestDryRun:
    def test_writes_no_output(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--dry-run", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        assert not (workspace / "draft.unmark.md").exists()

    def test_leaves_source_untouched(self, draft, workspace):
        original = draft.read_bytes()
        run_unmark("edit", str(draft), "--dry-run", cwd=workspace)
        assert draft.read_bytes() == original

    def test_creates_no_run_directory(self, draft, workspace):
        run_unmark("edit", str(draft), "--dry-run", cwd=workspace)
        assert not (workspace / ".unmark").exists()

    def test_still_reports_planned_operations(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--dry-run", "--format", "json", cwd=workspace)
        payload = json.loads(result.stdout.decode())
        assert payload["dry_run"] is True
        assert payload["operation_count"] == 1


class TestDiffs:
    def test_no_diff_by_default(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--format", "json", cwd=workspace)
        assert json.loads(result.stdout.decode())["diff"] is None

    def test_unified_diff(self, draft, workspace):
        result = run_unmark(
            "edit", str(draft), "--diff", "unified", "--format", "json", cwd=workspace
        )
        diff = json.loads(result.stdout.decode())["diff"]
        assert diff.startswith("---")

    def test_operations_diff_names_the_operator(self, draft, workspace):
        result = run_unmark(
            "edit", str(draft), "--diff", "operations", "--format", "json", cwd=workspace
        )
        diff = json.loads(result.stdout.decode())["diff"]
        assert "unicode:zero_width" in diff
        assert "U+200B" in diff


class TestLocks:
    def test_user_lock_prevents_sanitation(self, workspace):
        path = workspace / "locked.txt"
        path.write_text(f"KEEP{ZWSP}ME and other text\n", encoding="utf-8")
        result = run_unmark(
            "edit", str(path), "--lock", f"KEEP{ZWSP}ME", "--output", "-", cwd=workspace
        )
        assert result.returncode == ExitCode.SUCCESS
        assert ZWSP in result.stdout.decode()

    def test_invalid_lock_regex_is_usage_error(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--lock", "(unclosed", cwd=workspace)
        assert result.returncode == ExitCode.USAGE


class TestReportsAndRuns:
    def test_report_states_sanitized(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--format", "json", cwd=workspace)
        payload = json.loads(result.stdout.decode())
        assert payload["state"] == "sanitized"

    def test_report_records_cli_research_mode_and_output_format(self, draft, workspace):
        result = run_unmark(
            "edit",
            str(draft),
            "--unicode-policy",
            "aggressive",
            "--research-mode",
            "--format",
            "json",
            cwd=workspace,
        )

        assert result.returncode == ExitCode.SUCCESS
        effective = json.loads(result.stdout.decode())["effective_config"]
        assert effective["research_mode"] is True
        assert effective["output"]["format"] == "json"

    def test_report_describes_the_signal_family_changed(self, draft, workspace):
        result = run_unmark("edit", str(draft), "--format", "json", cwd=workspace)
        payload = json.loads(result.stdout.decode())
        assert payload["state"] != "verified_below_threshold"
        assert "removed recognized hidden unicode carriers" in payload["residual_risk"].lower()

    def test_run_directory_layout(self, draft, workspace):
        run_unmark("edit", str(draft), "--diff", "unified", cwd=workspace)
        runs = list((workspace / ".unmark" / "runs").iterdir())
        assert len(runs) == 1
        names = {p.name for p in runs[0].iterdir()}
        assert {
            "request.json",
            "effective-config.json",
            "source.sha256",
            "events.jsonl",
            "output.txt",
            "diff.patch",
            "report.json",
        } <= names

    def test_events_have_monotonic_sequence(self, draft, workspace):
        run_unmark("edit", str(draft), cwd=workspace)
        runs = list((workspace / ".unmark" / "runs").iterdir())
        lines = (runs[0] / "events.jsonl").read_text().strip().splitlines()
        sequences = [json.loads(line)["sequence"] for line in lines]
        assert sequences == sorted(sequences)
        assert sequences == list(range(len(sequences)))

    def test_runs_show(self, draft, workspace):
        edit = run_unmark("edit", str(draft), "--format", "json", cwd=workspace)
        run_id = json.loads(edit.stdout.decode())["run_id"]
        result = run_unmark("runs", "show", run_id, cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        assert run_id in result.stderr.decode()

    def test_runs_show_unknown_id(self, workspace):
        result = run_unmark("runs", "show", "nonexistent", cwd=workspace)
        assert result.returncode == ExitCode.USAGE

    def test_source_hash_recorded(self, draft, workspace):
        import hashlib

        run_unmark("edit", str(draft), cwd=workspace)
        runs = list((workspace / ".unmark" / "runs").iterdir())
        recorded = (runs[0] / "source.sha256").read_text().strip()
        assert recorded == hashlib.sha256(draft.read_bytes()).hexdigest()


class TestConfigCommands:
    def test_init_creates_config(self, workspace):
        result = run_unmark("config", "init", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        assert (workspace / ".unmark.toml").exists()

    def test_init_refuses_existing_without_force(self, workspace):
        (workspace / ".unmark.toml").write_text("existing")
        result = run_unmark("config", "init", cwd=workspace)
        assert result.returncode == ExitCode.USAGE
        assert (workspace / ".unmark.toml").read_text() == "existing"

    def test_generated_config_is_valid(self, workspace):
        run_unmark("config", "init", cwd=workspace)
        result = run_unmark("config", "validate", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS

    def test_validate_explain_shows_sources(self, workspace):
        (workspace / ".unmark.toml").write_text('[unicode]\npolicy = "typographic"\n')
        result = run_unmark("config", "validate", "--explain", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        assert "unicode.policy" in result.stderr.decode()

    def test_rewrite_check_reports_readiness_without_exposing_key(self, workspace):
        (workspace / ".unmark.toml").write_text(
            "[rewrite]\n"
            'backend = "openai-compatible"\n'
            'endpoint = "https://openrouter.ai/api/v1"\n'
            'model = "google/gemini-3.7-flash"\n'
            'rewrite_provider = "google"\n'
            "allow_remote = true\n"
            'key_env = "UNMARK_TEST_OPENROUTER_KEY"\n'
        )
        result = run_unmark(
            "config",
            "rewrite-check",
            "--source-provider",
            "anthropic",
            "--format",
            "json",
            cwd=workspace,
            env={"UNMARK_TEST_OPENROUTER_KEY": "secret-never-printed"},
        )
        assert result.returncode == ExitCode.SUCCESS
        payload = json.loads(result.stdout.decode())
        assert payload["ready"] is True
        assert payload["rewrite_provider"] == "google"
        assert payload["key_env"] == "UNMARK_TEST_OPENROUTER_KEY"
        assert "secret-never-printed" not in result.stdout.decode()

    def test_rewrite_check_fails_before_model_call_when_key_is_missing(self, workspace):
        (workspace / ".unmark.toml").write_text(
            "[rewrite]\n"
            'backend = "openai-compatible"\n'
            'endpoint = "https://openrouter.ai/api/v1"\n'
            'model = "google/gemini-3.7-flash"\n'
            "allow_remote = true\n"
            'key_env = "UNMARK_DEFINITELY_MISSING_KEY"\n'
        )
        result = run_unmark(
            "config",
            "rewrite-check",
            "--source-provider",
            "anthropic",
            cwd=workspace,
        )
        assert result.returncode == ExitCode.USAGE
        assert "UNMARK_DEFINITELY_MISSING_KEY" in result.stderr.decode()

    def test_unknown_key_rejected(self, workspace):
        (workspace / ".unmark.toml").write_text("bogus_key = 1\n")
        result = run_unmark("config", "validate", cwd=workspace)
        assert result.returncode == ExitCode.USAGE
        assert "unknown key" in result.stderr.decode()

    def test_schema_is_valid_json(self, workspace):
        result = run_unmark("config", "schema", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        schema = json.loads(result.stdout.decode())
        assert "properties" in schema

    def test_project_config_applies_to_edit(self, draft, workspace):
        (workspace / ".unmark.toml").write_text('[unicode]\npolicy = "report"\n')
        result = run_unmark("edit", str(draft), "--format", "json", cwd=workspace)
        payload = json.loads(result.stdout.decode())
        # report policy never mutates, so nothing should be planned.
        assert payload["operation_count"] == 0


class TestDeterminism:
    def test_identical_input_yields_identical_output(self, draft, workspace):
        first = run_unmark("edit", str(draft), "--output", "-", cwd=workspace)
        second = run_unmark("edit", str(draft), "--output", "-", cwd=workspace)
        assert first.stdout == second.stdout

    def test_identical_input_yields_identical_hash(self, draft, workspace):
        # --force on the repeat run: the second invocation would otherwise refuse
        # to replace the sibling output written by the first.
        results = []
        for extra in ([], ["--force"]):
            result = run_unmark("edit", str(draft), "--format", "json", *extra, cwd=workspace)
            assert result.returncode == ExitCode.SUCCESS
            results.append(json.loads(result.stdout.decode())["output_sha256"])
        assert results[0] == results[1]


class TestFixtures:
    def test_structured_markdown(self, workspace):
        source = FIXTURES / "structured.md"
        result = run_unmark("inspect", str(source), "--format", "json", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        payload = json.loads(result.stdout.decode())
        kinds = payload["document"]["block_counts"]
        assert {"heading", "paragraph", "list_item", "quote", "code", "table"} <= set(kinds)
        span_kinds = set(payload["document"]["span_counts"])
        assert {"url", "code", "citation"} <= span_kinds

    def test_unicode_carriers_fixture_sanitizes_conservatively(self, workspace):
        source = FIXTURES / "unicode_carriers.txt"
        result = run_unmark("edit", str(source), "--output", "-", cwd=workspace)
        assert result.returncode == ExitCode.SUCCESS
        output = result.stdout.decode()
        assert "​" not in output  # zero-width space removed
        assert "‍" in output  # emoji ZWJ preserved
        assert "‌" in output  # Persian ZWNJ preserved
        assert "️" in output  # variation selector preserved
        assert " " in output  # NBSP preserved
        assert "\U000e0001" not in output  # hidden tag payload removed

    def test_plain_text_fixture(self, workspace):
        source = FIXTURES / "simple.txt"
        result = run_unmark("inspect", str(source), "--format", "json", cwd=workspace)
        payload = json.loads(result.stdout.decode())
        assert payload["document"]["media_type"] == "text/plain"
        assert payload["unicode"]["findings"] == []
