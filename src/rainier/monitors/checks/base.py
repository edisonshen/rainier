"""MonitorCheck protocol and CheckResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rainier.monitors.sources.base import MonitorReading


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of evaluating a check against a reading."""

    triggered: bool
    severity: str  # "info", "warning", "critical"
    message: str
    field_name: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MonitorCheck(Protocol):
    """Evaluates a condition against a monitor reading."""

    def evaluate(
        self,
        reading: MonitorReading,
        history: list[MonitorReading],
    ) -> CheckResult:
        """Evaluate the check condition.

        Args:
            reading: The current reading for a specific field.
            history: Previous readings for the same monitor+field,
                     ordered most-recent-first.

        Returns:
            CheckResult indicating whether the condition triggered.
        """
        ...
