"""LLM provider adapters for the research engine.

Three providers per D-019: ``anthropic`` (Opus 4.7 + extended thinking,
D-030), ``deepseek`` (V4-Pro / V4-Flash text-only), and
``openai_compatible`` (DeepSeek alt endpoint + future GPT / Gemini).

Slice 0 exercises only the ``test_auth()`` surface — the LLM call paths
land in Slice 1. ``get_provider(name)`` returns a fresh adapter per
call; adapters are tiny and stateless (the heavy SDK clients are lazily
constructed inside ``run()``).
"""

from __future__ import annotations

from .base import (
    AuthStatus,
    Provider,
    ProviderTestResult,
)


def list_providers() -> list[str]:
    """Names of the registered providers (sorted for determinism)."""
    return sorted(_REGISTRY.keys())


def get_provider(name: str) -> Provider:
    """Factory — returns a fresh provider adapter for ``name``."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown provider {name!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


# Lazy registry — adapters are imported on demand so a missing optional SDK
# (anthropic, openai) doesn't break ``import rainier.research``.
def _build_anthropic() -> Provider:
    from .anthropic import AnthropicProvider

    return AnthropicProvider()


def _build_deepseek() -> Provider:
    from .deepseek import DeepseekProvider

    return DeepseekProvider()


def _build_openai_compatible() -> Provider:
    from .openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider()


_REGISTRY: dict[str, callable] = {
    "anthropic": _build_anthropic,
    "deepseek": _build_deepseek,
    "openai_compatible": _build_openai_compatible,
}

__all__ = [
    "AuthStatus",
    "Provider",
    "ProviderTestResult",
    "get_provider",
    "list_providers",
]
