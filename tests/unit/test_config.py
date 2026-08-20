"""Configuration loading, precedence, discovery, and validation."""

from __future__ import annotations

import pytest

from unmark.core.errors import ConfigError, UnsupportedError, UsageError
from unmark.orchestration.config import (
    UnmarkedConfig,
    discover_project_config,
    find_workspace_root,
    load_toml,
    resolve_config,
)
from unmark.orchestration.presets import get_preset


def write(path, text: str):
    path.write_text(text, encoding="utf-8")
    return path


class TestValidation:
    def test_defaults_are_valid(self):
        config = UnmarkedConfig()
        assert config.preset == "sanitize"
        assert config.unicode.policy == "safe"
        assert config.fidelity.level == "strict"

    def test_unknown_top_level_key_rejected(self, tmp_path):
        write(tmp_path / ".unmark.toml", 'nonsense = "value"\n')
        with pytest.raises(ConfigError, match="unknown key"):
            resolve_config(explicit_config=tmp_path / ".unmark.toml", include_user_config=False)

    def test_unknown_nested_key_rejected(self, tmp_path):
        write(tmp_path / ".unmark.toml", "[unicode]\nbogus = true\n")
        with pytest.raises(ConfigError, match="unknown key"):
            resolve_config(explicit_config=tmp_path / ".unmark.toml", include_user_config=False)

    def test_invalid_value_rejected(self, tmp_path):
        write(tmp_path / ".unmark.toml", '[unicode]\npolicy = "nonsense"\n')
        with pytest.raises(ConfigError):
            resolve_config(explicit_config=tmp_path / ".unmark.toml", include_user_config=False)

    def test_out_of_range_ratio_rejected(self, tmp_path):
        write(tmp_path / ".unmark.toml", "[budget]\nmax_char_edit_ratio = 1.5\n")
        with pytest.raises(ConfigError):
            resolve_config(explicit_config=tmp_path / ".unmark.toml", include_user_config=False)

    def test_malformed_toml_rejected(self, tmp_path):
        write(tmp_path / ".unmark.toml", "this is not = = toml\n")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_toml(tmp_path / ".unmark.toml")

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_toml(tmp_path / "absent.toml")

    def test_error_message_names_the_source(self, tmp_path):
        config = write(tmp_path / "custom.toml", "bogus_key = 1\n")
        with pytest.raises(ConfigError, match="--config"):
            resolve_config(explicit_config=config, include_user_config=False)


class TestSecrets:
    @pytest.mark.parametrize("key", ["api_key", "secret", "token", "password", "credential"])
    def test_secret_like_keys_refused(self, tmp_path, key):
        write(tmp_path / "c.toml", f'{key} = "value"\n')
        with pytest.raises(ConfigError, match="looks like a secret"):
            load_toml(tmp_path / "c.toml")

    def test_nested_secret_refused(self, tmp_path):
        write(tmp_path / "c.toml", '[unicode]\napi_key = "x"\n')
        with pytest.raises(ConfigError, match="looks like a secret"):
            load_toml(tmp_path / "c.toml")

    def test_token_edit_budget_is_not_mistaken_for_a_secret(self, tmp_path):
        path = write(tmp_path / "c.toml", "[budget]\nmax_token_edit_ratio = 0.25\n")
        assert load_toml(path)["budget"]["max_token_edit_ratio"] == 0.25

    def test_key_env_accepts_only_an_environment_variable_name(self, tmp_path):
        config = write(
            tmp_path / "c.toml",
            '[rewrite]\nkey_env = "sk-or-secret-value"\n',
        )
        with pytest.raises(ConfigError, match="environment-variable name"):
            resolve_config(explicit_config=config, include_user_config=False)


class TestPrecedence:
    def test_cli_beats_project_config(self, tmp_path):
        write(tmp_path / ".unmark.toml", '[unicode]\npolicy = "report"\n')
        resolved = resolve_config(
            cli_overrides={"unicode": {"policy": "typographic"}},
            start_dir=tmp_path,
            include_user_config=False,
        )
        assert resolved.config.unicode.policy == "typographic"
        assert resolved.sources["unicode.policy"] == "CLI option"

    def test_explicit_config_beats_project_config(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        write(project / ".unmark.toml", '[unicode]\npolicy = "report"\n')
        explicit = write(tmp_path / "explicit.toml", '[unicode]\npolicy = "typographic"\n')
        resolved = resolve_config(
            explicit_config=explicit, start_dir=project, include_user_config=False
        )
        assert resolved.config.unicode.policy == "typographic"

    def test_project_config_beats_preset(self, tmp_path):
        write(tmp_path / ".unmark.toml", '[unicode]\npolicy = "typographic"\n')
        resolved = resolve_config(start_dir=tmp_path, include_user_config=False)
        assert resolved.config.unicode.policy == "typographic"
        assert resolved.sources["unicode.policy"] == "project config"

    def test_defaults_used_when_nothing_set(self, tmp_path):
        resolved = resolve_config(start_dir=tmp_path, include_user_config=False)
        assert resolved.config.unicode.policy == "safe"

    def test_explain_lists_values_and_sources(self, tmp_path):
        write(tmp_path / ".unmark.toml", '[unicode]\npolicy = "typographic"\n')
        resolved = resolve_config(start_dir=tmp_path, include_user_config=False)
        rows = {key: source for key, _, source in resolved.explain()}
        assert rows["unicode.policy"] == "project config"
        assert "budget.max_runtime_ms" in rows


class TestDiscovery:
    def test_finds_config_in_same_directory(self, tmp_path):
        write(tmp_path / "pyproject.toml", "")
        config = write(tmp_path / ".unmark.toml", "")
        assert discover_project_config(tmp_path) == config

    def test_finds_config_in_parent_within_workspace(self, tmp_path):
        write(tmp_path / "pyproject.toml", "")
        config = write(tmp_path / ".unmark.toml", "")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert discover_project_config(nested) == config

    def test_stops_at_workspace_boundary(self, tmp_path):
        # A config above the workspace marker must not be picked up.
        write(tmp_path / ".unmark.toml", '[unicode]\npolicy = "aggressive"\n')
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        write(workspace / ".git", "")
        assert discover_project_config(workspace) is None

    def test_workspace_root_detection(self, tmp_path):
        workspace = tmp_path / "repo"
        workspace.mkdir()
        (workspace / ".git").mkdir()
        nested = workspace / "src" / "deep"
        nested.mkdir(parents=True)
        assert find_workspace_root(nested) == workspace.resolve()

    def test_no_config_returns_none(self, tmp_path):
        write(tmp_path / "pyproject.toml", "")
        assert discover_project_config(tmp_path) is None


class TestPresets:
    def test_sanitize_is_available(self):
        preset = get_preset("sanitize")
        assert preset.available
        assert preset.unicode.name == "safe"
        assert preset.target.mode == "sanitize_only"

    def test_unknown_preset_is_usage_error(self):
        with pytest.raises(UsageError, match="unknown preset"):
            get_preset("nonexistent")

    @pytest.mark.parametrize("name", ["light", "balanced", "strong", "offline-max"])
    def test_deferred_presets_report_unsupported(self, name):
        with pytest.raises(UnsupportedError, match="not available in this build"):
            get_preset(name)

    def test_no_preset_selects_aggressive_unicode(self):
        from unmark.orchestration.presets import PRESETS

        for preset in PRESETS.values():
            assert preset.unicode.name != "aggressive"

    def test_presets_are_versioned(self):
        assert get_preset("sanitize").version == "1"
