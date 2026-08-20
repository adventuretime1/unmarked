"""Voice profiles: model validation, the local store, and prompt injection."""

from __future__ import annotations

import json

import pytest

from unmark.core.errors import UsageError
from unmark.core.voice import MAX_DESCRIPTION_CHARS, VoiceProfile, is_valid_voice_name
from unmark.storage.voice_store import (
    VoiceStore,
    load_profile,
    new_profile,
    resolve_voice,
)
from unmark.strategies.rewrite.prompts import build_rewrite_prompt

SAMPLE = "Short declarative sentences. Avoids hedging and corporate filler."


class TestVoiceProfile:
    def test_description_is_required(self):
        with pytest.raises(ValueError):
            VoiceProfile()  # type: ignore[call-arg]  # negative test

    def test_empty_description_rejected(self):
        with pytest.raises(ValueError):
            VoiceProfile(description="   \n  ")

    def test_description_is_stripped(self):
        assert VoiceProfile(description="  hello  ").description == "hello"

    def test_oversized_description_rejected(self):
        with pytest.raises(ValueError, match="maximum"):
            VoiceProfile(description="x" * (MAX_DESCRIPTION_CHARS + 1))

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError):
            VoiceProfile(description=SAMPLE, tone="snarky")  # type: ignore[call-arg]

    def test_bad_name_rejected(self):
        with pytest.raises(ValueError, match="invalid voice name"):
            VoiceProfile(name="../escape", description=SAMPLE)

    def test_round_trips_through_json(self):
        profile = new_profile("work", SAMPLE, generated_by="claude-code")
        restored = VoiceProfile.model_validate_json(profile.model_dump_json())
        assert restored == profile


class TestVoiceNames:
    @pytest.mark.parametrize("name", ["work", "casual-2", "my_voice", "a1"])
    def test_accepts_plain_names(self, name):
        assert is_valid_voice_name(name)

    @pytest.mark.parametrize(
        "name",
        ["", ".hidden", "../escape", "a/b", "a\\b", "with space", "sla/sh"],
    )
    def test_rejects_unsafe_names(self, name):
        assert not is_valid_voice_name(name)


class TestVoiceStore:
    def test_empty_store_lists_nothing(self, tmp_path):
        assert VoiceStore(tmp_path).list_names() == ()

    def test_missing_directory_lists_nothing(self, tmp_path):
        assert VoiceStore(tmp_path / "absent").list_names() == ()

    def test_save_then_load(self, tmp_path):
        store = VoiceStore(tmp_path)
        store.save(new_profile("work", SAMPLE))
        assert store.list_names() == ("work",)
        assert store.load("work").description == SAMPLE

    def test_save_refuses_to_clobber_without_force(self, tmp_path):
        store = VoiceStore(tmp_path)
        store.save(new_profile("work", SAMPLE))
        with pytest.raises(UsageError, match="already exists"):
            store.save(new_profile("work", "different"))

    def test_force_replaces(self, tmp_path):
        store = VoiceStore(tmp_path)
        store.save(new_profile("work", SAMPLE))
        store.save(new_profile("work", "replaced"), force=True)
        assert store.load("work").description == "replaced"

    def test_unnamed_profile_cannot_be_saved(self, tmp_path):
        with pytest.raises(UsageError, match="must be named"):
            VoiceStore(tmp_path).save(VoiceProfile(description=SAMPLE))

    def test_missing_profile_names_the_alternatives(self, tmp_path):
        store = VoiceStore(tmp_path)
        store.save(new_profile("work", SAMPLE))
        with pytest.raises(UsageError, match="work"):
            store.load("absent")

    def test_unsafe_name_refused_at_lookup(self, tmp_path):
        with pytest.raises(UsageError, match="invalid voice name"):
            VoiceStore(tmp_path).load("../../etc/passwd")

    def test_delete_removes_the_file(self, tmp_path):
        store = VoiceStore(tmp_path)
        store.save(new_profile("work", SAMPLE))
        store.delete("work")
        assert store.list_names() == ()

    def test_delete_unknown_profile_fails(self, tmp_path):
        with pytest.raises(UsageError, match="no voice profile"):
            VoiceStore(tmp_path).delete("absent")

    def test_markdown_preferred_over_json(self, tmp_path):
        (tmp_path / "work.md").write_text("from markdown", encoding="utf-8")
        (tmp_path / "work.json").write_text(
            json.dumps({"description": "from json"}), encoding="utf-8"
        )
        assert VoiceStore(tmp_path).load("work").description == "from markdown"


class TestLoadProfile:
    def test_markdown_body_becomes_the_description(self, tmp_path):
        path = tmp_path / "work.md"
        path.write_text(SAMPLE, encoding="utf-8")
        profile = load_profile(path)
        assert profile.description == SAMPLE
        assert profile.name == "work"

    def test_json_is_parsed_as_a_profile(self, tmp_path):
        path = tmp_path / "work.json"
        path.write_text(
            json.dumps({"description": SAMPLE, "generated_by": "codex"}), encoding="utf-8"
        )
        profile = load_profile(path)
        assert profile.generated_by == "codex"
        assert profile.name == "work"

    def test_invalid_json_reports_the_path(self, tmp_path):
        path = tmp_path / "work.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(UsageError, match="invalid JSON"):
            load_profile(path)

    def test_json_array_rejected(self, tmp_path):
        path = tmp_path / "work.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(UsageError, match="JSON object"):
            load_profile(path)

    def test_empty_markdown_rejected(self, tmp_path):
        path = tmp_path / "work.md"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(UsageError, match="invalid voice profile"):
            load_profile(path)

    def test_missing_file_reports_the_path(self, tmp_path):
        with pytest.raises(UsageError, match="not found"):
            load_profile(tmp_path / "absent.md")

    def test_non_utf8_rejected(self, tmp_path):
        path = tmp_path / "work.md"
        path.write_bytes(b"\xff\xfe invalid")
        with pytest.raises(UsageError, match="UTF-8"):
            load_profile(path)


class TestResolveVoice:
    def test_bare_name_uses_the_store(self, tmp_path):
        store = VoiceStore(tmp_path)
        store.save(new_profile("work", SAMPLE))
        assert resolve_voice("work", store=store).description == SAMPLE

    def test_path_bypasses_the_store(self, tmp_path):
        path = tmp_path / "loose.md"
        path.write_text("ad hoc voice", encoding="utf-8")
        store = VoiceStore(tmp_path / "empty")
        assert resolve_voice(str(path), store=store).description == "ad hoc voice"


class TestPromptInjection:
    def test_absent_voice_leaves_the_prompt_unchanged(self):
        without = build_rewrite_prompt("Some text.")
        with_empty = build_rewrite_prompt("Some text.", voice="")
        assert without.user == with_empty.user

    def test_voice_appears_in_the_prompt(self):
        prompt = build_rewrite_prompt("Some text.", voice=SAMPLE)
        assert SAMPLE in prompt.user

    def test_voice_is_subordinate_to_the_rules(self):
        prompt = build_rewrite_prompt("Some text.", voice=SAMPLE)
        assert "never overrides the rules above" in prompt.user

    def test_voice_precedes_the_text(self):
        prompt = build_rewrite_prompt("UNIQUETEXT", voice=SAMPLE)
        assert prompt.user.index(SAMPLE) < prompt.user.index("UNIQUETEXT")

    def test_whitespace_only_voice_is_ignored(self):
        baseline = build_rewrite_prompt("Some text.")
        assert build_rewrite_prompt("Some text.", voice="   \n ").user == baseline.user

    def test_rendering_is_deterministic(self):
        first = build_rewrite_prompt("Some text.", voice=SAMPLE)
        second = build_rewrite_prompt("Some text.", voice=SAMPLE)
        assert first.user == second.user
