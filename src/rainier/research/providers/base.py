"""Provider protocol + result types.

Per CLAUDE.md (project) module-layout discipline: ``research/`` imports
only from ``core/`` + stdlib. We define a lightweight Protocol here
rather than reusing ``core/protocols.py`` because the research engine's
provider surface is research-specific and should not bleed into the
production signals/backtest path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class AuthStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    FAILED = "failed"


@dataclass(frozen=True)
class ProviderTestResult:
    """One row of `providers test --json` output.

    `next_step` is the operator-actionable remediation (per the
    "surface-don't-silo" memory rule). Example: when the API key is
    missing we emit ``"export ANTHROPIC_API_KEY=..."``.
    """

    provider: str
    auth: AuthStatus
    endpoint_reachable: bool = False
    schema_compliant: bool = False
    latency_ms: int = 0
    next_step: str | None = None

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "auth": self.auth.value,
            "endpoint_reachable": self.endpoint_reachable,
            "schema_compliant": self.schema_compliant,
            "latency_ms": self.latency_ms,
            "next_step": self.next_step,
        }


@runtime_checkable
class Provider(Protocol):
    """Stateless adapter surface.

    Slice 0 ships only ``test_auth()``; ``run()`` (the actual LLM call)
    arrives in Slice 1. Keep the surface minimal — adding methods now
    bakes premature abstractions for code that doesn't exist yet.
    """

    name: str

    def test_auth(self) -> ProviderTestResult: ...
