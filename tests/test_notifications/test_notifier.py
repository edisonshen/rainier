"""Tests for the notification module."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from rainier.core.config import AppConfig, NotifyConfig, ScrapingConfig, Settings
from rainier.scrapers.base import ScrapeResult


def _make_settings(notify_urls: str = "", enabled: bool = True, **overrides) -> Settings:
    """Create a Settings instance with test defaults."""
    defaults = {
        "database_url": "postgresql://test:test@localhost/test",
        "notify_urls": notify_urls,
        "app": AppConfig(),
        "notify": NotifyConfig(enabled=enabled),
        "scraping": ScrapingConfig(),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_result(records: int = 100, errors: list[str] | None = None) -> ScrapeResult:
    """Create a ScrapeResult for testing."""
    now = datetime.now(timezone.utc)
    r = ScrapeResult(scraper_name="qu", started_at=now)
    r.records_created = records
    r.errors = errors or []
    r.finished_at = now
    return r


class TestGetApprise:
    """Test _get_apprise() configuration logic."""

    def test_returns_none_when_disabled(self):
        settings = _make_settings(notify_urls="mailto://x:y@gmail.com", enabled=False)
        with patch("rainier.notifications.notifier.get_settings", return_value=settings):
            from rainier.notifications.notifier import _get_apprise
            assert _get_apprise() is None

    def test_returns_none_when_no_urls(self):
        settings = _make_settings(notify_urls="")
        with patch("rainier.notifications.notifier.get_settings", return_value=settings):
            from rainier.notifications.notifier import _get_apprise
            assert _get_apprise() is None

    def test_returns_apprise_when_configured(self):
        settings = _make_settings(notify_urls="json://localhost")
        with patch("rainier.notifications.notifier.get_settings", return_value=settings):
            from rainier.notifications.notifier import _get_apprise
            ap = _get_apprise()
            assert ap is not None
            assert len(ap) == 1

    def test_handles_multiple_urls(self):
        settings = _make_settings(notify_urls="json://localhost, json://other")
        with patch("rainier.notifications.notifier.get_settings", return_value=settings):
            from rainier.notifications.notifier import _get_apprise
            ap = _get_apprise()
            assert ap is not None
            assert len(ap) == 2


class TestNotifyScrapeResult:
    """Test notify_scrape_result() message formatting."""

    def test_skips_when_disabled(self):
        settings = _make_settings(enabled=False)
        with patch("rainier.notifications.notifier.get_settings", return_value=settings):
            from rainier.notifications.notifier import notify_scrape_result
            # Should not raise
            notify_scrape_result("morning", _make_result())

    def test_sends_success_notification(self):
        settings = _make_settings(notify_urls="json://localhost")
        result = _make_result(records=42)
        mock_ap = MagicMock()
        mock_ap.notify.return_value = True
        mock_ap.__len__ = lambda self: 1

        with (
            patch("rainier.notifications.notifier.get_settings", return_value=settings),
            patch("rainier.notifications.notifier._get_apprise", return_value=mock_ap),
        ):
            from rainier.notifications.notifier import notify_scrape_result
            notify_scrape_result("morning", result)

        mock_ap.notify.assert_called_once()
        call_kwargs = mock_ap.notify.call_args
        assert "[Rainier] Scrape OK: morning" in call_kwargs.kwargs["title"]
        assert "42" in call_kwargs.kwargs["body"]

    def test_sends_warning_notification_with_errors(self):
        settings = _make_settings(notify_urls="json://localhost")
        result = _make_result(records=80, errors=["timeout on page 5"])
        mock_ap = MagicMock()
        mock_ap.notify.return_value = True
        mock_ap.__len__ = lambda self: 1

        with (
            patch("rainier.notifications.notifier.get_settings", return_value=settings),
            patch("rainier.notifications.notifier._get_apprise", return_value=mock_ap),
        ):
            from rainier.notifications.notifier import notify_scrape_result
            notify_scrape_result("midday", result)

        mock_ap.notify.assert_called_once()
        call_kwargs = mock_ap.notify.call_args
        assert "partial" in call_kwargs.kwargs["title"].lower()
        assert "timeout on page 5" in call_kwargs.kwargs["body"]


class TestNotifyScrapeFailure:
    """Test notify_scrape_failure() message formatting."""

    def test_skips_when_disabled(self):
        settings = _make_settings(enabled=False)
        with patch("rainier.notifications.notifier.get_settings", return_value=settings):
            from rainier.notifications.notifier import notify_scrape_failure
            notify_scrape_failure("morning", "browser crashed")

    def test_sends_failure_notification(self):
        settings = _make_settings(notify_urls="json://localhost")
        mock_ap = MagicMock()
        mock_ap.notify.return_value = True
        mock_ap.__len__ = lambda self: 1

        with (
            patch("rainier.notifications.notifier.get_settings", return_value=settings),
            patch("rainier.notifications.notifier._get_apprise", return_value=mock_ap),
        ):
            from rainier.notifications.notifier import notify_scrape_failure
            notify_scrape_failure("close", "Playwright timeout")

        mock_ap.notify.assert_called_once()
        call_kwargs = mock_ap.notify.call_args
        assert "FAILED" in call_kwargs.kwargs["title"]
        assert "Playwright timeout" in call_kwargs.kwargs["body"]


class TestSendErrorHandling:
    """Test that _send never propagates exceptions."""

    def test_catches_apprise_exception(self):
        settings = _make_settings(notify_urls="json://localhost")
        mock_ap = MagicMock()
        mock_ap.notify.side_effect = RuntimeError("SMTP connection refused")
        mock_ap.__len__ = lambda self: 1

        with (
            patch("rainier.notifications.notifier.get_settings", return_value=settings),
            patch("rainier.notifications.notifier._get_apprise", return_value=mock_ap),
        ):
            from rainier.notifications.notifier import notify_scrape_failure
            # Should NOT raise
            notify_scrape_failure("morning", "test error")


class TestNotifyConfig:
    """Test the NotifyConfig model."""

    def test_defaults(self):
        cfg = NotifyConfig()
        assert cfg.enabled is True
        assert cfg.subject_prefix == "[Rainier]"

    def test_custom_prefix(self):
        cfg = NotifyConfig(subject_prefix="[Test]")
        assert cfg.subject_prefix == "[Test]"

    def test_settings_has_notify(self):
        s = _make_settings()
        assert s.notify.enabled is True
