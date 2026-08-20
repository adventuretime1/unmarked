"""Shared open-weight hosted-model routing defaults for Unmarked applications."""

from __future__ import annotations

from typing import Literal, TypedDict

OpenRouterMode = Literal["low", "medium", "high"]


class OpenRouterModelMode(TypedDict):
    label: str
    model: str
    model_label: str
    provider: str
    provider_label: str


OPENROUTER_RECOMMENDED_MODES: dict[OpenRouterMode, OpenRouterModelMode] = {
    "low": {
        "label": "Fast",
        "model": "thinkingmachines/inkling-small",
        "model_label": "Inkling Small",
        "provider": "deepinfra",
        "provider_label": "DeepInfra",
    },
    "medium": {
        "label": "Auto",
        "model": "openai/gpt-oss-120b",
        "model_label": "GPT-OSS 120B",
        "provider": "cerebras",
        "provider_label": "Cerebras",
    },
    "high": {
        "label": "Deep",
        "model": "qwen/qwen3.8-2.4t-a95b",
        "model_label": "Qwen 3.8 2.4T",
        "provider": "modal",
        "provider_label": "Modal",
    },
}

OPENROUTER_MODELS: tuple[str, ...] = (
    OPENROUTER_RECOMMENDED_MODES["medium"]["model"],
)
OPENROUTER_PROVIDER_ONLY: tuple[str, ...] = (
    OPENROUTER_RECOMMENDED_MODES["medium"]["provider"],
)


def openrouter_route(mode: OpenRouterMode) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return one pinned model/provider pair without cross-provider fallback."""
    selected = OPENROUTER_RECOMMENDED_MODES[mode]
    return (selected["model"],), (selected["provider"],)
