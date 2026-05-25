"""Unit tests for BrowserManager — filesystem logic, no real browser."""

from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rainier.scrapers.browser import BrowserManager


class TestIsSessionValid:
    def test_no_file(self, tmp_path):
        assert BrowserManager.is_session_valid(tmp_path / "nope.json", ttl_hours=12) is False

    def test_fresh_file(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        assert BrowserManager.is_session_valid(p, ttl_hours=12) is True

    def test_expired_file(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        # Backdate file modification time by 13 hours
        old_time = time.time() - (13 * 3600)
        os.utime(p, (old_time, old_time))
        assert BrowserManager.is_session_valid(p, ttl_hours=12) is False

    def test_exactly_at_boundary(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        # Set to exactly 12 hours ago — should be invalid (not strictly less than)
        boundary_time = time.time() - (12 * 3600)
        os.utime(p, (boundary_time, boundary_time))
        assert BrowserManager.is_session_valid(p, ttl_hours=12) is False

    def test_just_under_boundary(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        # Set to 11 hours 59 minutes ago — should be valid
        under_time = time.time() - (11 * 3600 + 59 * 60)
        os.utime(p, (under_time, under_time))
        assert BrowserManager.is_session_valid(p, ttl_hours=12) is True

    def test_zero_ttl(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        assert BrowserManager.is_session_valid(p, ttl_hours=0) is False


def _mock_playwright(connect_side_effect):
    """Build a patched async_playwright() whose chromium.connect_over_cdp uses the given side effect."""
    mock_chromium = MagicMock()
    mock_chromium.connect_over_cdp = AsyncMock(side_effect=connect_side_effect)
    mock_pw = MagicMock()
    mock_pw.chromium = mock_chromium
    mock_pw_factory = MagicMock()
    mock_pw_factory.start = AsyncMock(return_value=mock_pw)
    return mock_pw_factory, mock_chromium


class TestCDPConnectRetry:
    """start() force-restarts Chrome and retries once if connect_over_cdp fails."""

    async def test_retries_after_force_restart(self):
        mock_browser = MagicMock()
        factory, chromium = _mock_playwright(
            [RuntimeError("Browser.setDownloadBehavior: not supported"), mock_browser]
        )
        with (
            patch("rainier.scrapers.browser.async_playwright", return_value=factory),
            patch.object(BrowserManager, "_force_restart_chrome", new=AsyncMock()) as restart,
        ):
            bm = BrowserManager(cdp_url="http://localhost:9222")
            await bm.start()

        assert bm._browser is mock_browser
        assert bm._is_cdp is True
        assert chromium.connect_over_cdp.await_count == 2
        restart.assert_awaited_once()

    async def test_second_failure_propagates(self):
        factory, chromium = _mock_playwright(RuntimeError("still broken"))
        # Mock the kill helper so this test cannot SIGTERM a real debug
        # Chrome that happens to be listening on :9222 on the dev's box.
        # The retry path now reaps the Chrome it spawned when the second
        # connect fails (so __aexit__ never runs); without this mock we
        # would call real lsof/kill from a unit test.
        with (
            patch("rainier.scrapers.browser.async_playwright", return_value=factory),
            patch.object(BrowserManager, "_force_restart_chrome", new=AsyncMock()) as restart,
            patch.object(BrowserManager, "_kill_chrome_we_spawned", new=AsyncMock()) as kill,
        ):
            bm = BrowserManager(cdp_url="http://localhost:9222")
            with pytest.raises(RuntimeError, match="still broken"):
                await bm.start()

        assert chromium.connect_over_cdp.await_count == 2
        restart.assert_awaited_once()
        # And the cleanup ran exactly once for the spawned Chrome.
        kill.assert_awaited_once()

    async def test_happy_path_no_restart(self):
        mock_browser = MagicMock()
        factory, chromium = _mock_playwright([mock_browser])
        with (
            patch("rainier.scrapers.browser.async_playwright", return_value=factory),
            patch.object(BrowserManager, "_force_restart_chrome", new=AsyncMock()) as restart,
        ):
            bm = BrowserManager(cdp_url="http://localhost:9222")
            await bm.start()

        assert bm._browser is mock_browser
        assert chromium.connect_over_cdp.await_count == 1
        restart.assert_not_awaited()


class TestFreshTabInExistingContext:
    """fresh_tab_in_existing_context (DESIGN D-10) in CDP mode."""

    async def test_fresh_tab_no_contexts_raises_clearly(self):
        """When the CDP browser exposes an empty ``contexts`` list, the helper
        must raise ``RuntimeError`` with a clear message rather than silently
        creating a new (auth-less) context or letting ``IndexError`` bubble.

        Rationale: contexts[] is empty only when Chrome has no operator window
        open — in that state every downstream cookie-bound fetch will fail
        anyway. Raising immediately with a self-explanatory message is much
        less confusing than the IndexError that previously surfaced from
        ``contexts[0]``.
        """
        bm = BrowserManager(cdp_url="http://localhost:9222")
        # Simulate a started CDP browser with no contexts.
        mock_browser = MagicMock()
        mock_browser.contexts = []
        bm._browser = mock_browser
        bm._is_cdp = True

        with pytest.raises(RuntimeError, match="CDP browser has no contexts"):
            async with bm.fresh_tab_in_existing_context():
                pass  # pragma: no cover — we expect to raise before this

        # And we did NOT silently materialize a new context as a workaround.
        mock_browser.new_context.assert_not_called()
