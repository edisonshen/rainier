"""Provider adapter tests.

The Slice 0 surface is `providers test` — sanity-check auth + endpoint
reachability + schema-compliance of a no-op call. Providers are stubbed
here (no network); the adapter shape is what we lock in.
"""

from __future__ import annotations

import pytest

from rainier.research.providers import (
    AuthStatus,
    ProviderTestResult,
    get_provider,
    list_providers,
)


def test_list_providers_returns_three():
    """v0/v1 ships anthropic + deepseek + openai_compatible per D-019."""
    names = set(list_providers())
    assert names == {"anthropic", "deepseek", "openai_compatible"}


def test_get_provider_anthropic_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = get_provider("anthropic")
    result = provider.test_auth()
    assert isinstance(result, ProviderTestResult)
    assert result.auth is AuthStatus.MISSING
    assert "ANTHROPIC_API_KEY" in (result.next_step or "")


def test_get_provider_anthropic_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    provider = get_provider("anthropic")
    result = provider.test_auth()
    # We don't hit the network in test mode; "ok" means the key was loaded.
    assert result.auth is AuthStatus.OK
    assert result.provider == "anthropic"


def test_get_provider_deepseek_missing_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = get_provider("deepseek")
    result = provider.test_auth()
    assert result.auth is AuthStatus.MISSING
    assert "DEEPSEEK_API_KEY" in (result.next_step or "")


def test_get_provider_openai_compatible_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = get_provider("openai_compatible")
    result = provider.test_auth()
    assert result.auth is AuthStatus.MISSING


def test_get_provider_unknown_name_raises():
    with pytest.raises(ValueError) as exc:
        get_provider("not-a-real-provider")
    assert "not-a-real-provider" in str(exc.value)


def test_provider_result_to_dict_shape():
    """`ProviderTestResult.as_dict()` is the JSON payload the CLI emits."""
    r = ProviderTestResult(
        provider="anthropic",
        auth=AuthStatus.OK,
        endpoint_reachable=True,
        schema_compliant=True,
        latency_ms=42,
        next_step=None,
    )
    d = r.as_dict()
    assert d["provider"] == "anthropic"
    assert d["auth"] == "ok"
    assert d["endpoint_reachable"] is True
    assert d["schema_compliant"] is True
    assert d["latency_ms"] == 42
    assert "next_step" in d
