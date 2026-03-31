"""MonitorSource protocol and MonitorReading dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class MonitorReading:
    """A single data point fetched by a monitor source.

    For multi-selector sources, each selector produces its own reading.
    The ``metadata`` dict carries source-specific context (URL, selector used, etc.).
    """

    monitor_name: str
    field_name: str  # which selector/field produced this value
    timestamp: datetime
    raw_value: str
    numeric_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MonitorSource(Protocol):
    """Fetches data from an external source (web page, API, etc.)."""

    async def fetch(self, params: dict[str, Any]) -> list[MonitorReading]:
        """Fetch one or more readings from the source.

        Args:
            params: Source configuration merged with runtime defaults
                    (url, selectors, timeout_ms, monitor_name, etc.).

        Returns:
            List of readings — one per extracted field/selector.
        """
        ...
