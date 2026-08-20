"""Strict TOML configuration.

Precedence, highest first:

1. CLI options
2. explicitly supplied ``--config PATH``
3. project ``.unmark.toml``
4. user config
5. versioned preset
6. package defaults

Project discovery walks upward from the input file but **stops at the workspace
boundary** -- a directory containing a VCS marker, a project marker, or the user's
home directory -- so Unmarked never picks up a stray config from an unrelated parent
directory.

Unknown keys are rejected. Secrets are not supported: a config key that looks like
a credential is refused outright.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic import ValidationError as PydanticValidationError

from unmark.core.errors import ConfigError
from unmark.core.policies import UnicodePolicyName
from unmark.core.spans import StrictModel

#: Directory names that end upward project-config discovery.
_WORKSPACE_MARKERS: frozenset[str] = frozenset(
    {".git", ".hg", ".svn", ".jj", "pyproject.toml", "package.json", "go.mod", "Cargo.toml"}
)

_SECRET_HINTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
)

CONFIG_FILENAME = ".unmark.toml"


class OutputConfig(StrictModel):
    diff: Literal["none", "unified", "operations"] = "none"
    format: Literal["text", "json"] = "text"
    retain_output: bool = True


class BudgetConfig(StrictModel):
    max_runtime_ms: int = Field(default=30_000, ge=0)
    max_char_edit_ratio: float = Field(default=0.08, ge=0.0, le=1.0)
    max_token_edit_ratio: float = Field(default=0.12, ge=0.0, le=1.0)
    max_length_drift_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    max_candidates: int = Field(default=32, ge=0)
    max_rounds: int = Field(default=3, ge=0)
    max_input_chars: int = Field(default=1_000_000, ge=0)
    # Zero by default: sanitation makes no model or detector calls, and rewrite
    # runs must raise these explicitly, which is also the remote-opt-in gate.
    max_model_calls: int = Field(default=0, ge=0)
    max_detector_queries: int = Field(default=0, ge=0)
    max_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)


class UnicodeConfig(StrictModel):
    policy: UnicodePolicyName = "safe"
    preserve_emoji_sequences: bool = True
    preserve_language_controls: bool = True
    normalize_spaces: bool = True
    normalization_form: Literal["none", "NFC", "NFKC"] = "NFC"


class FidelityConfig(StrictModel):
    level: Literal["strict", "standard", "creative"] = "strict"
    lock_numbers: bool = True
    lock_citations: bool = True
    lock_quotes: bool = True
    lock_urls: bool = True
    lock_code: bool = True
    lock_dates: bool = True
    lock_units: bool = True
    lock_identifiers: bool = True
    locks: tuple[str, ...] = ()


class RewriteConfigSection(StrictModel):
    """Settings for the simple prompt-driven rewrite baselines.

    No credential ever lives here: ``_reject_secrets`` refuses key-like config
    fields, and the backend reads any API key from the environment out of band.
    Remote model calls are off unless ``allow_remote`` is explicitly set true.
    """

    backend: Literal["print-prompt", "ollama", "openai-compatible"] = "print-prompt"
    strategy: Literal["one-shot", "recursive"] = "one-shot"
    model: str = ""
    fallback_models: tuple[str, ...] = ()
    provider_only: tuple[str, ...] = ()
    endpoint: str = ""
    source_provider: str = ""
    rewrite_provider: str = ""
    allow_remote: bool = False
    style: Literal["faithful", "syntax", "lexical", "polish", "simplify", "academic"] = "faithful"
    strength: Literal["light", "medium", "strong"] = "medium"
    candidate_count: int = Field(default=1, ge=1, le=16)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    target_length_ratio: float | None = Field(default=None, gt=0.0, le=4.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    rounds: int = Field(default=3, ge=1, le=5)
    style_schedule: tuple[
        Literal["faithful", "syntax", "lexical", "polish", "simplify", "academic"], ...
    ] = ()
    early_stop_patience: int = Field(default=1, ge=1)
    seed: int = 0
    key_env: str = Field(
        default="",
        description="Name of the environment variable holding the API key, if any.",
    )

    @field_validator("key_env")
    @classmethod
    def _validate_key_env(cls, value: str) -> str:
        name = value.strip()
        if name and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(
                "must be an environment-variable name, not an API key or credential value"
            )
        return name


class UnmarkedConfig(StrictModel):
    """The complete effective configuration for a run."""

    schema_version: Literal["1"] = "1"
    preset: str = "sanitize"
    research_mode: bool = False
    output: OutputConfig = OutputConfig()
    budget: BudgetConfig = BudgetConfig()
    unicode: UnicodeConfig = UnicodeConfig()
    fidelity: FidelityConfig = FidelityConfig()
    rewrite: RewriteConfigSection = RewriteConfigSection()


@dataclass
class ConfigLayer:
    """One contributing source of configuration values."""

    name: str
    origin: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedConfig:
    """The effective config plus per-field provenance for ``--explain``."""

    config: UnmarkedConfig
    sources: dict[str, str]
    layers: tuple[ConfigLayer, ...]

    def explain(self) -> list[tuple[str, Any, str]]:
        """``(dotted_key, value, source)`` for every effective field, sorted."""
        rows = []
        for key, value in sorted(_flatten(self.config.model_dump(mode="json")).items()):
            rows.append((key, value, self.sources.get(key, "package default")))
        return rows


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _reject_secrets(values: dict[str, Any], origin: str) -> None:
    for key in _flatten(values):
        leaf = key.rsplit(".", maxsplit=1)[-1].lower().replace("-", "_")
        credential_suffixes = tuple(f"_{hint}" for hint in _SECRET_HINTS)
        if leaf in _SECRET_HINTS or leaf.endswith(credential_suffixes):
            msg = (
                f"{origin}: key {key!r} looks like a secret. Unmarked does not read "
                "credentials from configuration files; use the environment instead."
            )
            raise ConfigError(msg)


def load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML config file, raising :class:`ConfigError` on any problem."""
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except FileNotFoundError as exc:
        msg = f"config file not found: {path}"
        raise ConfigError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"invalid TOML in {path}: {exc}"
        raise ConfigError(msg) from exc
    _reject_secrets(data, str(path))
    return data


def find_workspace_root(start: Path) -> Path:
    """The workspace boundary at or above ``start``."""
    current = start if start.is_dir() else start.parent
    current = current.resolve()
    home = Path.home().resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _WORKSPACE_MARKERS):
            return candidate
        if candidate == home:
            return candidate
    return current


def discover_project_config(start: Path) -> Path | None:
    """Find ``.unmark.toml`` upward from ``start``, stopping at the workspace root."""
    current = (start if start.is_dir() else start.parent).resolve()
    root = find_workspace_root(current)
    for candidate in (current, *current.parents):
        config = candidate / CONFIG_FILENAME
        if config.is_file():
            return config
        if candidate == root:
            break
    return None


def user_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "unmark" / "config.toml"


def resolve_config(
    *,
    cli_overrides: dict[str, Any] | None = None,
    explicit_config: Path | None = None,
    start_dir: Path | None = None,
    preset_name: str | None = None,
    include_user_config: bool = True,
) -> ResolvedConfig:
    """Merge every configuration layer into an effective :class:`UnmarkedConfig`."""
    from unmark.orchestration.presets import PRESETS

    layers: list[ConfigLayer] = []

    layers.append(ConfigLayer("package defaults", "built-in", UnmarkedConfig().model_dump()))

    name = preset_name or (cli_overrides or {}).get("preset") or "sanitize"
    preset = PRESETS.get(str(name))
    if preset is not None and preset.available:
        layers.append(
            ConfigLayer(
                f"preset {preset.name} v{preset.version}",
                f"preset:{preset.name}@{preset.version}",
                {
                    "preset": preset.name,
                    "unicode": {"policy": preset.unicode.name},
                    "fidelity": {"level": preset.fidelity.level},
                    "budget": {"max_char_edit_ratio": preset.max_char_edit_ratio},
                },
            )
        )

    if include_user_config:
        user_path = user_config_path()
        if user_path.is_file():
            layers.append(ConfigLayer("user config", str(user_path), load_toml(user_path)))

    project_path = discover_project_config(start_dir or Path.cwd())
    if project_path is not None:
        layers.append(ConfigLayer("project config", str(project_path), load_toml(project_path)))

    if explicit_config is not None:
        layers.append(ConfigLayer("--config", str(explicit_config), load_toml(explicit_config)))

    if cli_overrides:
        layers.append(ConfigLayer("CLI option", "command line", cli_overrides))

    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for layer in layers:
        merged = _merge(merged, layer.values)
        for key in _flatten(layer.values):
            sources[key] = layer.name

    try:
        config = UnmarkedConfig.model_validate(merged)
    except PydanticValidationError as exc:
        raise ConfigError(_format_validation_error(exc, layers)) from exc

    return ResolvedConfig(config=config, sources=sources, layers=tuple(layers))


def _format_validation_error(
    exc: PydanticValidationError, layers: tuple[ConfigLayer, ...] | list[ConfigLayer]
) -> str:
    """Turn a Pydantic error into a message that names the offending layer."""
    lines = ["configuration is invalid:"]
    for error in exc.errors():
        dotted = ".".join(str(part) for part in error["loc"])
        origin = "unknown source"
        for layer in reversed(list(layers)):
            if dotted in _flatten(layer.values) or dotted in layer.values:
                origin = f"{layer.name} ({layer.origin})"
                break
        if error["type"] == "extra_forbidden":
            lines.append(f"  unknown key {dotted!r} from {origin}")
        else:
            lines.append(f"  {dotted}: {error['msg']} (from {origin})")
    return "\n".join(lines)


DEFAULT_CONFIG_TEMPLATE = """\
# Unmarked project configuration.
# Every key below is optional; the values shown are the package defaults.
# Unknown keys are rejected. Secrets are never read from this file.

preset = "sanitize"

[output]
diff = "none"       # none | unified | operations
format = "text"     # text | json

[unicode]
policy = "safe"     # report | safe | typographic | aggressive (research-only)
preserve_emoji_sequences = true
preserve_language_controls = true
normalize_spaces = true
normalization_form = "NFC" # none | NFC | NFKC

[fidelity]
level = "strict"    # strict | standard | creative
lock_numbers = true
lock_citations = true
lock_quotes = true
lock_urls = true
lock_code = true
# Regexes that must survive byte-identical:
# locks = ["ACME-\\\\d+"]

[budget]
max_runtime_ms = 30000
max_char_edit_ratio = 0.08
max_length_drift_ratio = 0.05
# Model/detector budgets are zero by default. A rewrite run derives a model-call
# budget from its rounds and candidate count; raise these to cap a networked run.
# max_model_calls = 0
# max_detector_queries = 0
# max_cost_usd = "0"

[rewrite]
# Prompt-driven rewriting for `unmark edit --rewrite`. Unlike Unicode sanitation,
# it disrupts visible token, n-gram, sentence, and model-probability patterns.
backend = "print-prompt"    # print-prompt | ollama | openai-compatible
strategy = "one-shot"       # one-shot | recursive
# model = "llama3.2"
# endpoint = "http://127.0.0.1:11434"   # loopback ollama needs no allow_remote
# source_provider = "anthropic" # where the input text came from
# rewrite_provider = "openai"   # optional when derived from model
allow_remote = false        # required to reach a non-loopback endpoint
style = "faithful"          # faithful | syntax | lexical | polish | simplify | academic
strength = "medium"         # light | medium | strong
candidate_count = 1         # candidates generated per hop (1-16)
temperature = 0.7
rounds = 3                  # recursive strategy: number of hops (1-5)
early_stop_patience = 1     # recursive: hops without gain before stopping
seed = 0
# The API key is read only from this environment variable, never from config:
# key_env = "OPENAI_API_KEY"
#
# Example OpenRouter setup (the secret itself stays in the environment):
# backend = "openai-compatible"
# endpoint = "https://openrouter.ai/api/v1"
# model = "openai/gpt-oss-120b"
# rewrite_provider = "google"
# allow_remote = true
# key_env = "OPENROUTER_API_KEY"
"""
