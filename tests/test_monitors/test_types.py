"""Tests for monitor protocols, dataclasses, and ORM models."""

from datetime import datetime, timezone

from rainier.monitors.actions.base import MonitorAction
from rainier.monitors.checks.base import CheckResult, MonitorCheck
from rainier.monitors.sources.base import MonitorReading, MonitorSource

# ---------------------------------------------------------------------------
# Dataclass creation
# ---------------------------------------------------------------------------


class TestMonitorReading:
    def test_create_with_numeric(self):
        r = MonitorReading(
            monitor_name="test",
            field_name="price",
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
            raw_value="$1,234.56",
            numeric_value=1234.56,
        )
        assert r.monitor_name == "test"
        assert r.field_name == "price"
        assert r.numeric_value == 1234.56
        assert r.raw_value == "$1,234.56"

    def test_create_without_numeric(self):
        r = MonitorReading(
            monitor_name="test",
            field_name="status",
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
            raw_value="CLOSED",
        )
        assert r.numeric_value is None

    def test_frozen(self):
        r = MonitorReading(
            monitor_name="test",
            field_name="v",
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
            raw_value="42",
        )
        try:
            r.raw_value = "99"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_metadata_defaults_to_empty_dict(self):
        r = MonitorReading(
            monitor_name="test",
            field_name="v",
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
            raw_value="42",
        )
        assert r.metadata == {}


class TestCheckResult:
    def test_triggered(self):
        cr = CheckResult(
            triggered=True,
            severity="critical",
            message="Value exceeded threshold",
            field_name="ship_transits",
        )
        assert cr.triggered is True
        assert cr.severity == "critical"
        assert cr.field_name == "ship_transits"

    def test_not_triggered(self):
        cr = CheckResult(triggered=False, severity="info", message="")
        assert cr.triggered is False

    def test_frozen(self):
        cr = CheckResult(triggered=True, severity="warning", message="test")
        try:
            cr.triggered = False  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Protocol structural subtyping
# ---------------------------------------------------------------------------


class _FakeSource:
    async def fetch(self, params: dict) -> list[MonitorReading]:
        return []


class _FakeCheck:
    def evaluate(
        self, reading: MonitorReading, history: list[MonitorReading]
    ) -> CheckResult:
        return CheckResult(triggered=False, severity="info", message="")


class _FakeAction:
    async def execute(
        self, monitor_name: str, reading: MonitorReading, result: CheckResult
    ) -> None:
        pass


class TestProtocolSubtyping:
    def test_source_protocol(self):
        assert isinstance(_FakeSource(), MonitorSource)

    def test_check_protocol(self):
        assert isinstance(_FakeCheck(), MonitorCheck)

    def test_action_protocol(self):
        assert isinstance(_FakeAction(), MonitorAction)

    def test_non_conforming_rejected(self):
        class _Bad:
            pass

        assert not isinstance(_Bad(), MonitorSource)
        assert not isinstance(_Bad(), MonitorCheck)
        assert not isinstance(_Bad(), MonitorAction)


# ---------------------------------------------------------------------------
# ORM model import (verifies models are properly defined)
# ---------------------------------------------------------------------------


class TestORMModels:
    def test_monitor_reading_record_importable(self):
        from rainier.core.models import MonitorReadingRecord

        assert MonitorReadingRecord.__tablename__ == "monitor_readings"

    def test_monitor_alert_record_importable(self):
        from rainier.core.models import MonitorAlertRecord

        assert MonitorAlertRecord.__tablename__ == "monitor_alerts"

    def test_hypertable_registered(self):
        from rainier.core.models import HYPERTABLES

        assert "monitor_readings" in HYPERTABLES
        assert HYPERTABLES["monitor_readings"] == "recorded_at"
