"""Shared recommended OpenRouter model routes."""

from unmark.core.model_defaults import openrouter_route


def test_openrouter_modes_pin_one_requested_provider_each() -> None:
    assert openrouter_route("low") == (
        ("thinkingmachines/inkling-small",),
        ("deepinfra",),
    )
    assert openrouter_route("medium") == (
        ("openai/gpt-oss-120b",),
        ("cerebras",),
    )
    assert openrouter_route("high") == (
        ("qwen/qwen3.8-2.4t-a95b",),
        ("modal",),
    )
