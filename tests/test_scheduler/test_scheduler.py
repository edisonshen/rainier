"""Tests for the scheduler module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rainier.core.config import AppConfig, ScrapingConfig, ScrapingSchedule, Settings


def _make_settings(**overrides) -> Settings:
    """Create a Settings instance with test defaults."""
    defaults = {
        "database_url": "postgresql://test:test@localhost/test",
        "app": AppConfig(timezone="America/Los_Angeles"),
        "scraping": ScrapingConfig(
            schedule=ScrapingSchedule(
                morning="08:35",
                midday="10:35",
                afternoon="12:35",
                close="14:35",
            )
        ),
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestBuildScheduler:
    """Test that build_scheduler creates the correct jobs."""

    def test_creates_four_jobs(self):
        settings = _make_settings()
        with patch("rainier.scheduler.service.get_settings", return_value=settings):
            from rainier.scheduler.service import build_scheduler
            scheduler = build_scheduler()

        jobs = scheduler.get_jobs()
        assert len(jobs) == 4

    def test_job_ids(self):
        settings = _make_settings()
        with patch("rainier.scheduler.service.get_settings", return_value=settings):
            from rainier.scheduler.service import build_scheduler
            scheduler = build_scheduler()

        job_ids = {j.id for j in scheduler.get_jobs()}
        assert job_ids == {
            "qu_scrape_morning",
            "qu_scrape_midday",
            "qu_scrape_afternoon",
            "qu_scrape_close",
        }

    def test_job_names(self):
        settings = _make_settings()
        with patch("rainier.scheduler.service.get_settings", return_value=settings):
            from rainier.scheduler.service import build_scheduler
            scheduler = build_scheduler()

        names = {j.name for j in scheduler.get_jobs()}
        assert "QU100 scrape (morning)" in names
        assert "QU100 scrape (close)" in names

    def test_cron_triggers_weekdays_only(self):
        settings = _make_settings()
        with patch("rainier.scheduler.service.get_settings", return_value=settings):
            from rainier.scheduler.service import build_scheduler
            scheduler = build_scheduler()

        for job in scheduler.get_jobs():
            trigger = job.trigger
            # CronTrigger fields: day_of_week should be mon-fri
            dow_field = trigger.fields[4]  # day_of_week is index 4
            assert str(dow_field) == "mon-fri"

    def test_morning_job_time(self):
        settings = _make_settings()
        with patch("rainier.scheduler.service.get_settings", return_value=settings):
            from rainier.scheduler.service import build_scheduler
            scheduler = build_scheduler()

        morning_job = scheduler.get_job("qu_scrape_morning")
        trigger = morning_job.trigger
        hour_field = trigger.fields[5]   # hour is index 5
        minute_field = trigger.fields[6]  # minute is index 6
        assert str(hour_field) == "8"
        assert str(minute_field) == "35"

    def test_close_job_time(self):
        settings = _make_settings()
        with patch("rainier.scheduler.service.get_settings", return_value=settings):
            from rainier.scheduler.service import build_scheduler
            scheduler = build_scheduler()

        close_job = scheduler.get_job("qu_scrape_close")
        trigger = close_job.trigger
        hour_field = trigger.fields[5]
        minute_field = trigger.fields[6]
        assert str(hour_field) == "14"
        assert str(minute_field) == "35"

    def test_custom_schedule_times(self):
        settings = _make_settings(
            scraping=ScrapingConfig(
                schedule=ScrapingSchedule(
                    morning="09:00",
                    midday="11:00",
                    afternoon="13:00",
                    close="15:00",
                )
            )
        )
        with patch("rainier.scheduler.service.get_settings", return_value=settings):
            from rainier.scheduler.service import build_scheduler
            scheduler = build_scheduler()

        morning_job = scheduler.get_job("qu_scrape_morning")
        trigger = morning_job.trigger
        assert str(trigger.fields[5]) == "9"
        assert str(trigger.fields[6]) == "0"

    def test_custom_timezone(self):
        settings = _make_settings(
            app=AppConfig(timezone="US/Eastern"),
        )
        with patch("rainier.scheduler.service.get_settings", return_value=settings):
            from rainier.scheduler.service import build_scheduler
            scheduler = build_scheduler()

        assert str(scheduler.timezone) == "US/Eastern"

    def test_misfire_grace_time(self):
        settings = _make_settings()
        with patch("rainier.scheduler.service.get_settings", return_value=settings):
            from rainier.scheduler.service import build_scheduler
            scheduler = build_scheduler()

        for job in scheduler.get_jobs():
            assert job.misfire_grace_time == 300


class TestRunQuScrape:
    """Test the run_qu_scrape job function."""

    @pytest.mark.asyncio
    async def test_calls_scraper_execute(self):
        """Verify run_qu_scrape launches browser and calls scraper.execute()."""
        from unittest.mock import AsyncMock, MagicMock

        mock_result = MagicMock()
        mock_result.records_created = 42
        mock_result.errors = []
        mock_result.duration_seconds = 5.0

        mock_scraper = AsyncMock()
        mock_scraper.execute = AsyncMock(return_value=mock_result)

        with (
            patch("rainier.scrapers.browser.BrowserManager") as MockBM,
            patch("rainier.scrapers.get_scraper", return_value=mock_scraper) as mock_get,
        ):
            mock_bm_instance = AsyncMock()
            MockBM.return_value.__aenter__ = AsyncMock(return_value=mock_bm_instance)
            MockBM.return_value.__aexit__ = AsyncMock(return_value=False)

            from rainier.scheduler.service import run_qu_scrape
            await run_qu_scrape("morning")

        MockBM.assert_called_once_with(headless=True)
        mock_get.assert_called_once_with("qu", mock_bm_instance)
        mock_scraper.execute.assert_called_once_with(session="morning")

    @pytest.mark.asyncio
    async def test_handles_scraper_exception(self):
        """Verify run_qu_scrape catches exceptions without propagating."""
        from unittest.mock import AsyncMock

        with (
            patch("rainier.scrapers.browser.BrowserManager") as MockBM,
            patch("rainier.scrapers.get_scraper", side_effect=RuntimeError("browser died")),
        ):
            MockBM.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            MockBM.return_value.__aexit__ = AsyncMock(return_value=False)

            from rainier.scheduler.service import run_qu_scrape
            # Should NOT raise — errors are logged, not propagated
            await run_qu_scrape("morning")


class TestAppConfig:
    """Test the new AppConfig model."""

    def test_default_timezone(self):
        cfg = AppConfig()
        assert cfg.timezone == "America/Los_Angeles"

    def test_custom_timezone(self):
        cfg = AppConfig(timezone="US/Eastern")
        assert cfg.timezone == "US/Eastern"

    def test_settings_has_app(self):
        s = _make_settings()
        assert s.app.timezone == "America/Los_Angeles"
