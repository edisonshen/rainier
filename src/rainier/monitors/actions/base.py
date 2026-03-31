"""MonitorAction protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rainier.monitors.checks.base import CheckResult
from rainier.monitors.sources.base import MonitorReading


@runtime_checkable
class MonitorAction(Protocol):
    """Fires an alert or side-effect when a check triggers."""

    async def execute(
        self,
        monitor_name: str,
        reading: MonitorReading,
        result: CheckResult,
    ) -> None:
        """Execute the action.

        Args:
            monitor_name: Name of the monitor that triggered.
            reading: The reading that caused the trigger.
            result: The check result with severity and message.
        """
        ...
