"""OpenAI-compatible provider adapter (DeepSeek alt + future GPT / Gemini).

Accepts either ``OPENAI_COMPATIBLE_API_KEY`` (explicit) or
``DEEPSEEK_API_KEY`` (the most common backing implementation). The two
fallbacks let the operator point this adapter at any OpenAI-compatible
endpoint without renaming env vars.
"""

from __future__ import annotations

import os

from .base import AuthStatus, ProviderTestResult


class OpenAICompatibleProvider:
    name = "openai_compatible"
    primary_env_var = "OPENAI_COMPATIBLE_API_KEY"
    fallback_env_var = "DEEPSEEK_API_KEY"

    def test_auth(self) -> ProviderTestResult:
        key = (
            os.environ.get(self.primary_env_var)
            or os.environ.get(self.fallback_env_var)
        )
        if not key:
            return ProviderTestResult(
                provider=self.name,
                auth=AuthStatus.MISSING,
                endpoint_reachable=False,
                next_step=(
                    f"export {self.primary_env_var}=... "
                    f"or {self.fallback_env_var}=... "
                    "and set OPENAI_COMPATIBLE_BASE_URL to your endpoint"
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
