"""Atomic writes, destination refusal, and the run store."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from unmark.core.errors import AtomicWriteError, UsageError
from unmark.storage.atomic import (
    atomic_path,
    atomic_write_text,
    check_destination,
    default_output_path,
)
from unmark.storage.run_store import RunStore, new_run_id


class TestDestinationChecks:
    def test_new_destination_allowed(self, tmp_path):
        check_destination(tmp_path / "new.txt")

    def test_existing_destination_refused_without_force(self, tmp_path):
        target = tmp_path / "existing.txt"
        target.write_text("old")
        with pytest.raises(UsageError, match="already exists"):
            check_destination(target)

    def test_existing_destination_allowed_with_force(self, tmp_path):
        target = tmp_path / "existing.txt"
        target.write_text("old")
        check_destination(target, force=True)

    def test_source_overwrite_refused(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("content")
        with pytest.raises(UsageError, match="refusing to overwrite the source"):
            check_destination(source, source=source, force=True)

    def test_symlink_destination_refused(self, tmp_path):
        real = tmp_path / "real.txt"
        real.write_text("content")
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        with pytest.raises(UsageError, match="symlink"):
            check_destination(link, force=True)

    def test_directory_destination_refused(self, tmp_path):
        with pytest.raises(UsageError, match="is a directory"):
            check_destination(tmp_path, force=True)

    def test_missing_parent_refused(self, tmp_path):
        with pytest.raises(UsageError, match="directory does not exist"):
            check_destination(tmp_path / "nope" / "file.txt")


class TestAtomicWrite:
    def test_writes_content(self, tmp_path):
        target = tmp_path / "out.txt"
        atomic_write_text(target, "hello")
        assert target.read_text() == "hello"

    def test_unicode_round_trips(self, tmp_path):
        target = tmp_path / "out.txt"
        text = "emoji 🎉 arabic مرحبا cjk 日本語\n"
        atomic_write_text(target, text)
        assert target.read_text(encoding="utf-8") == text

    def test_replaces_with_force(self, tmp_path):
        target = tmp_path / "out.txt"
        target.write_text("old")
        atomic_write_text(target, "new", force=True)
        assert target.read_text() == "new"

    def test_no_temp_files_left_behind(self, tmp_path):
        atomic_write_text(tmp_path / "out.txt", "content")
        assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]

    def test_failure_leaves_no_partial_file(self, tmp_path):
        target = tmp_path / "out.txt"
        with patch("os.replace", side_effect=OSError("simulated failure")):
            with pytest.raises(AtomicWriteError):
                atomic_write_text(target, "content")
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_failure_preserves_existing_file(self, tmp_path):
        target = tmp_path / "out.txt"
        target.write_text("original")
        with patch("os.replace", side_effect=OSError("simulated failure")):
            with pytest.raises(AtomicWriteError):
                atomic_write_text(target, "replacement", force=True)
        assert target.read_text() == "original"
        assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]

    def test_atomic_path_cleans_up_on_exception(self, tmp_path):
        target = tmp_path / "out.txt"
        with pytest.raises(RuntimeError), atomic_path(target) as temp:
            temp.write_text("partial")
            raise RuntimeError("boom")
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_temp_file_is_in_destination_directory(self, tmp_path):
        # Required for os.replace to be atomic: same filesystem.
        target = tmp_path / "out.txt"
        seen = []
        with atomic_path(target) as temp:
            seen.append(temp.parent)
            temp.write_text("x")
        assert seen == [tmp_path]


class TestDefaultOutputPath:
    @pytest.mark.parametrize(
        "source,expected",
        [
            ("draft.md", "draft.unmark.md"),
            ("notes.txt", "notes.unmark.txt"),
            ("README", "README.unmark"),
            ("a.b.md", "a.b.unmark.md"),
        ],
    )
    def test_sibling_name(self, source, expected):
        assert default_output_path(Path("/tmp") / source).name == expected

    def test_stays_in_same_directory(self):
        result = default_output_path(Path("/some/dir/draft.md"))
        assert result.parent == Path("/some/dir")


class TestRunStore:
    def test_run_ids_are_unique_and_sortable(self):
        ids = {new_run_id() for _ in range(50)}
        assert len(ids) == 50

    def test_write_and_read_json(self, tmp_path):
        store = RunStore(tmp_path)
        run_id = "run-1"
        store.write_json(run_id, "report.json", {"state": "sanitized"})
        assert store.load_json(run_id, "report.json") == {"state": "sanitized"}

    def test_events_are_appended_in_order(self, tmp_path):
        from unmark.core.events import EventRecorder

        store = RunStore(tmp_path)
        recorder = EventRecorder("run-1")
        recorder.state("created")
        recorder.state("running")
        recorder.state("completed")
        store.append_events("run-1", recorder.events)

        events = store.read_events("run-1")
        assert [e["sequence"] for e in events] == [0, 1, 2]
        assert [e["state"] for e in events] == ["created", "running", "completed"]

    def test_missing_run_raises(self, tmp_path):
        store = RunStore(tmp_path)
        assert not store.exists("nope")
        with pytest.raises(UsageError, match="has no"):
            store.load_json("nope", "report.json")

    def test_list_runs(self, tmp_path):
        store = RunStore(tmp_path)
        store.create("run-b")
        store.create("run-a")
        assert store.list_runs() == ("run-a", "run-b")

    def test_list_runs_empty_workspace(self, tmp_path):
        assert RunStore(tmp_path).list_runs() == ()
