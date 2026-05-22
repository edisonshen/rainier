"""DeepSeek provider adapter (V4-Pro reasoning + V4-Flash text-only per D-030)."""

from __future__ import annotations

import os

from .base import AuthStatus, ProviderTestResult


class DeepseekProvider:
    name = "deepseek"
    env_var = "DEEPSEEK_API_KEY"

    def test_auth(self) -> ProviderTestResult:
        key = os.environ.get(self.env_var)
        if not key:
            return ProviderTestResult(
                provider=self.name,
                auth=AuthStatus.MISSING,
                endpoint_reachable=False,
                next_step=(
                    f"export {self.env_var}=... "
                    "(create at https://platform.deepseek.com/api_keys)"
                ),
            )
        return ProviderTestResult(
            provider=self.name,
            auth=AuthStatus.OK,
            endpoint_reachable=False,
            schema_compliant=False,
            latency_ms=0,
            next_step=None,
        )
