"""Anthropic provider adapter (Opus 4.7 + extended thinking per D-030)."""

from __future__ import annotations

import os

from .base import AuthStatus, ProviderTestResult


class AnthropicProvider:
    name = "anthropic"
    env_var = "ANTHROPIC_API_KEY"

    def test_auth(self) -> ProviderTestResult:
        key = os.environ.get(self.env_var)
        if not key:
            return ProviderTestResult(
                provider=self.name,
                auth=AuthStatus.MISSING,
                endpoint_reachable=False,
                next_step=(
                    f"export {self.env_var}=... "
                    "(create at https://console.anthropic.com/settings/keys)"
                ),
            )
        # Slice 0: we don't poke the network from the auth-test path so the
        # CLI works offline. Endpoint-reachability + schema-compliance flip to
        # `False` here; Slice 1's `run()` exercises the real round-trip.
        return ProviderTestResult(
            provider=self.name,
            auth=AuthStatus.OK,
            endpoint_reachable=False,
            schema_compliant=False,
            latency_ms=0,
            next_step=None,
        )
