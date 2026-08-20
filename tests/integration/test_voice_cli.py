"""``unmark voice`` end to end.

The store lives in the user's config directory, so every test redirects
``XDG_CONFIG_HOME`` into a tmp path. Nothing here touches the real one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from unmark.core.errors import ExitCode


def run_unmark(
    *args: str, cwd: Path, config_home: Path, stdin: str | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Invoke the CLI with the voice store redirected into a tmp directory."""
    import os

    env = dict(os.environ)
    env["XDG_CONFIG_HOME"] = str(config_home)
    env["NO_COLOR"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "unmark.cli.app", *args],
        cwd=cwd,
        input=stdin.encode("utf-8") if stdin is not None else b"",
        capture_output=True,
        check=False,
        env=env,
    )


@pytest.fixture
def config_home(tmp_path: Path) -> Path:
    home = tmp_path / "config"
    home.mkdir()
    return home


SAMPLE = "Short declarative sentences. No hedging, no corporate filler."


class TestVoicePath:
    def test_path_prints_the_directory(self, tmp_path, config_home):
        result = run_unmark("voice", "path", cwd=tmp_path, config_home=config_home)
        assert result.returncode == ExitCode.SUCCESS
        assert "voices" in result.stdout.decode()

    def test_path_is_under_the_config_home(self, tmp_path, config_home):
        result = run_unmark("voice", "path", cwd=tmp_path, config_home=config_home)
        assert str(config_home) in result.stdout.decode()

    def test_named_path_includes_the_name(self, tmp_path, config_home):
        result = run_unmark("voice", "path", "work", cwd=tmp_path, config_home=config_home)
        assert "work" in result.stdout.decode()

    def test_named_path_reports_where_a_saved_profile_actually_is(self, tmp_path, config_home):
        run_unmark("voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin=SAMPLE)
        result = run_unmark("voice", "path", "work", cwd=tmp_path, config_home=config_home)
        assert Path(result.stdout.decode().strip()).is_file()

    def test_unsafe_name_refused(self, tmp_path, config_home):
        result = run_unmark("voice", "path", "../etc", cwd=tmp_path, config_home=config_home)
        assert result.returncode == ExitCode.USAGE


class TestVoiceSaveAndShow:
    def test_save_from_stdin_then_show(self, tmp_path, config_home):
        saved = run_unmark(
            "voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin=SAMPLE
        )
        assert saved.returncode == ExitCode.SUCCESS

        shown = run_unmark("voice", "show", "work", cwd=tmp_path, config_home=config_home)
        assert shown.returncode == ExitCode.SUCCESS
        assert SAMPLE in shown.stdout.decode()

    def test_save_from_file(self, tmp_path, config_home):
        source = tmp_path / "description.md"
        source.write_text(SAMPLE, encoding="utf-8")
        result = run_unmark(
            "voice", "save", "work", "--from", str(source), cwd=tmp_path, config_home=config_home
        )
        assert result.returncode == ExitCode.SUCCESS

    def test_save_writes_into_the_store(self, tmp_path, config_home):
        run_unmark("voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin=SAMPLE)
        assert (config_home / "unmark" / "voices" / "work.json").is_file()

    def test_missing_source_file_is_a_usage_error(self, tmp_path, config_home):
        result = run_unmark(
            "voice", "save", "work", "--from", "absent.md", cwd=tmp_path, config_home=config_home
        )
        assert result.returncode == ExitCode.USAGE

    def test_empty_description_refused(self, tmp_path, config_home):
        result = run_unmark(
            "voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin="   \n"
        )
        assert result.returncode != ExitCode.SUCCESS

    def test_overwrite_refused_without_force(self, tmp_path, config_home):
        run_unmark("voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin=SAMPLE)
        again = run_unmark(
            "voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin="different"
        )
        assert again.returncode == ExitCode.USAGE
        assert "--force" in again.stderr.decode()

    def test_force_replaces(self, tmp_path, config_home):
        run_unmark("voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin=SAMPLE)
        run_unmark(
            "voice",
            "save",
            "work",
            "--force",
            cwd=tmp_path,
            config_home=config_home,
            stdin="replaced",
        )
        shown = run_unmark("voice", "show", "work", cwd=tmp_path, config_home=config_home)
        assert "replaced" in shown.stdout.decode()

    def test_show_unknown_profile_is_a_usage_error(self, tmp_path, config_home):
        result = run_unmark("voice", "show", "absent", cwd=tmp_path, config_home=config_home)
        assert result.returncode == ExitCode.USAGE

    def test_show_json_carries_metadata(self, tmp_path, config_home):
        run_unmark(
            "voice",
            "save",
            "work",
            "--generated-by",
            "claude-code",
            cwd=tmp_path,
            config_home=config_home,
            stdin=SAMPLE,
        )
        result = run_unmark(
            "voice", "show", "work", "--format", "json", cwd=tmp_path, config_home=config_home
        )
        payload = json.loads(result.stdout.decode())
        assert payload["description"] == SAMPLE
        assert payload["generated_by"] == "claude-code"


class TestVoiceList:
    def test_empty_store_is_not_an_error(self, tmp_path, config_home):
        result = run_unmark("voice", "list", cwd=tmp_path, config_home=config_home)
        assert result.returncode == ExitCode.SUCCESS
        assert "no voice profiles" in result.stderr.decode()

    def test_lists_saved_profiles(self, tmp_path, config_home):
        for name in ("work", "casual"):
            run_unmark("voice", "save", name, cwd=tmp_path, config_home=config_home, stdin=SAMPLE)
        result = run_unmark("voice", "list", cwd=tmp_path, config_home=config_home)
        listed = result.stderr.decode()
        assert "work" in listed
        assert "casual" in listed

    def test_json_listing_is_machine_readable(self, tmp_path, config_home):
        run_unmark("voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin=SAMPLE)
        result = run_unmark(
            "voice", "list", "--format", "json", cwd=tmp_path, config_home=config_home
        )
        assert json.loads(result.stdout.decode())["voices"] == ["work"]


class TestVoiceDelete:
    def test_delete_requires_confirmation_when_not_a_tty(self, tmp_path, config_home):
        run_unmark("voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin=SAMPLE)
        result = run_unmark("voice", "delete", "work", cwd=tmp_path, config_home=config_home)
        assert result.returncode == ExitCode.USAGE
        assert (config_home / "unmark" / "voices" / "work.json").is_file()

    def test_delete_with_yes_removes_it(self, tmp_path, config_home):
        run_unmark("voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin=SAMPLE)
        result = run_unmark(
            "voice", "delete", "work", "--yes", cwd=tmp_path, config_home=config_home
        )
        assert result.returncode == ExitCode.SUCCESS
        assert not (config_home / "unmark" / "voices" / "work.json").exists()

    def test_delete_unknown_profile_is_a_usage_error(self, tmp_path, config_home):
        result = run_unmark(
            "voice", "delete", "absent", "--yes", cwd=tmp_path, config_home=config_home
        )
        assert result.returncode == ExitCode.USAGE


class TestVoiceInRewrite:
    """The voice must reach the prompt, which print-prompt makes observable."""

    def test_stored_voice_reaches_the_prompt(self, tmp_path, config_home):
        run_unmark("voice", "save", "work", cwd=tmp_path, config_home=config_home, stdin=SAMPLE)
        draft = tmp_path / "draft.md"
        draft.write_text("Some text to rewrite.\n", encoding="utf-8")

        result = run_unmark(
            "edit",
            str(draft),
            "--rewrite",
            "--backend",
            "print-prompt",
            "--voice",
            "work",
            "--dry-run",
            "--format",
            "json",
            cwd=tmp_path,
            config_home=config_home,
        )
        assert result.returncode == ExitCode.SUCCESS
        assert SAMPLE in result.stdout.decode()

    def test_voice_by_path_reaches_the_prompt(self, tmp_path, config_home):
        profile = tmp_path / "loose.md"
        profile.write_text("Ad hoc voice description.", encoding="utf-8")
        draft = tmp_path / "draft.md"
        draft.write_text("Some text to rewrite.\n", encoding="utf-8")

        result = run_unmark(
            "edit",
            str(draft),
            "--rewrite",
            "--backend",
            "print-prompt",
            "--voice",
            str(profile),
            "--dry-run",
            "--format",
            "json",
            cwd=tmp_path,
            config_home=config_home,
        )
        assert "Ad hoc voice description." in result.stdout.decode()

    def test_unknown_voice_fails_before_any_model_call(self, tmp_path, config_home):
        draft = tmp_path / "draft.md"
        draft.write_text("Some text to rewrite.\n", encoding="utf-8")
        result = run_unmark(
            "edit",
            str(draft),
            "--rewrite",
            "--backend",
            "print-prompt",
            "--voice",
            "absent",
            "--dry-run",
            cwd=tmp_path,
            config_home=config_home,
        )
        assert result.returncode == ExitCode.USAGE

    def test_sanitation_ignores_voice(self, tmp_path, config_home):
        """A voice is meaningless without a rewrite; it must not break sanitation."""
        draft = tmp_path / "draft.md"
        draft.write_text("Hello​world.\n", encoding="utf-8")
        result = run_unmark("edit", str(draft), "--dry-run", cwd=tmp_path, config_home=config_home)
        assert result.returncode == ExitCode.SUCCESS
