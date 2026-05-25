"""Tests for Chrome-ownership cleanup on BrowserManager.stop().

Background: when ``BrowserManager`` falls into the CDP-connect retry path
(``connect_over_cdp`` fails → ``_force_restart_chrome`` → retry), the
fresh Chrome process is owned by US, not the operator. Closing only the
scrape tab on teardown leaves an empty Chrome window listening on :9222
that accumulates after every cron run.

The fix: track ``_we_spawned_chrome`` (flipped to True in the recovery
path) and on ``stop()`` kill the Chrome process if we own it. If we
attached to the operator's pre-existing Chrome (no force-restart this
session), preserve the D-10 contract and never touch the Chrome process.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from rainier.scrapers.browser import BrowserManager


def _mock_playwright(connect_side_effect):
    """Build a patched async_playwright() whose chromium.connect_over_cdp uses the given side effect."""
    mock_chromium = MagicMock()
    mock_chromium.connect_over_cdp = AsyncMock(side_effect=connect_side_effect)
    mock_pw = MagicMock()
    mock_pw.chromium = mock_chromium
    mock_pw_factory = MagicMock()
    mock_pw_factory.start = AsyncMock(return_value=mock_pw)
    return mock_pw_factory, mock_chromium


class TestWeSpawnedFlag:
    """The _we_spawned_chrome flag defaults to False and flips to True only
    when this BrowserManager triggered a Chrome force-restart during start()."""

    def test_defaults_to_false(self):
        bm = BrowserManager(cdp_url="http://localhost:9222")
        assert bm._we_spawned_chrome is False

    async def test_force_restart_sets_we_spawned_flag(self):
        """If the first connect_over_cdp raises and we force-restart Chrome,
        the manager records that it now owns the Chrome process."""
        mock_browser = MagicMock()
        factory, _ = _mock_playwright(
            [RuntimeError("Browser context management is not supported"), mock_browser]
        )
        with (
            patch("rainier.scrapers.browser.async_playwright", return_value=factory),
            patch.object(BrowserManager, "_force_restart_chrome", new=AsyncMock()),
        ):
            bm = BrowserManager(cdp_url="http://localhost:9222")
            await bm.start()

        assert bm._we_spawned_chrome is True

    async def test_happy_cdp_path_does_not_set_we_spawned(self):
        """When connect_over_cdp succeeds on the first try, we attached to
        the operator's pre-existing Chrome — flag must stay False."""
        mock_browser = MagicMock()
        factory, _ = _mock_playwright([mock_browser])
        with (
            patch("rainier.scrapers.browser.async_playwright", return_value=factory),
            patch.object(BrowserManager, "_force_restart_chrome", new=AsyncMock()),
        ):
            bm = BrowserManager(cdp_url="http://localhost:9222")
            await bm.start()

        assert bm._we_spawned_chrome is False

    async def test_launch_mode_does_not_set_we_spawned(self):
        """Launch mode never touches the CDP recovery path — flag stays False."""
        mock_browser = AsyncMock()
        mock_chromium = MagicMock()
        mock_chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw = MagicMock()
        mock_pw.chromium = mock_chromium
        factory = MagicMock()
        factory.start = AsyncMock(return_value=mock_pw)

        with patch("rainier.scrapers.browser.async_playwright", return_value=factory):
            bm = BrowserManager()  # no cdp_url → launch mode
            await bm.start()

        assert bm._we_spawned_chrome is False


class TestStopKillsChromeWhenWeSpawned:
    """When stop() runs with _is_cdp=True AND _we_spawned_chrome=True, the
    Chrome process we spawned is killed (not just the tab)."""

    async def test_stop_kills_chrome_when_we_spawned(self):
        bm = BrowserManager(cdp_url="http://localhost:9222")
        bm._browser = MagicMock()
        bm._is_cdp = True
        bm._we_spawned_chrome = True
        bm._playwright = AsyncMock()

        with patch.object(
            BrowserManager, "_kill_chrome_we_spawned", new=AsyncMock()
        ) as kill:
            await bm.stop()

        kill.assert_awaited_once()

    async def test_stop_preserves_chrome_when_attached(self):
        """CDP mode but no force-restart this session — don't kill
        operator's Chrome (D-10 contract)."""
        bm = BrowserManager(cdp_url="http://localhost:9222")
        bm._browser = MagicMock()
        bm._is_cdp = True
        bm._we_spawned_chrome = False
        bm._playwright = AsyncMock()

        with patch.object(
            BrowserManager, "_kill_chrome_we_spawned", new=AsyncMock()
        ) as kill:
            await bm.stop()

        kill.assert_not_awaited()

    async def test_stop_closes_browser_in_launch_mode(self):
        """Launch mode unchanged: browser.close() called; kill helper NOT
        invoked even if we_spawned_chrome were somehow True (defensive)."""
        bm = BrowserManager()  # launch mode
        mock_browser = AsyncMock()
        bm._browser = mock_browser
        bm._is_cdp = False
        bm._we_spawned_chrome = False
        bm._playwright = AsyncMock()

        with patch.object(
            BrowserManager, "_kill_chrome_we_spawned", new=AsyncMock()
        ) as kill:
            await bm.stop()

        mock_browser.close.assert_awaited_once()
        kill.assert_not_awaited()


class TestKillChromeImplementation:
    """_kill_chrome_we_spawned uses lsof + kill to terminate the Chrome
    process bound to :9222. Failure to find or kill the process is logged
    but does not raise (teardown must be best-effort)."""

    async def test_kill_chrome_runs_lsof_and_kills_pids(self):
        """Happy path: lsof returns pids; each is killed via subprocess."""
        # asyncio.create_subprocess_exec → process with stdout/stderr we can drive.
        async def fake_exec(*args, **kwargs):
            proc = AsyncMock()
            if "lsof" in args:
                proc.communicate = AsyncMock(return_value=(b"12345\n67890\n", b""))
                proc.returncode = 0
            else:  # kill
                proc.communicate = AsyncMock(return_value=(b"", b""))
                proc.returncode = 0
            return proc

        with patch(
            "rainier.scrapers.browser.asyncio.create_subprocess_exec",
            side_effect=fake_exec,
        ) as exec_mock:
            await BrowserManager._kill_chrome_we_spawned()

        # At minimum: one lsof call + two kill calls (one per pid).
        call_strs = [tuple(c.args) for c in exec_mock.call_args_list]
        lsof_calls = [c for c in call_strs if "lsof" in c]
        kill_calls = [c for c in call_strs if c and c[0] == "kill"]
        assert len(lsof_calls) == 1, f"expected 1 lsof call, got {call_strs}"
        assert len(kill_calls) == 2, f"expected 2 kill calls, got {call_strs}"

    async def test_kill_chrome_no_pids_is_noop(self):
        """If lsof finds no pids (Chrome already gone), do not raise."""
        async def fake_exec(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 1  # lsof returns non-zero when nothing matches
            return proc

        with patch(
            "rainier.scrapers.browser.asyncio.create_subprocess_exec",
            side_effect=fake_exec,
        ):
            # Must not raise even when lsof exits non-zero / empty.
            await BrowserManager._kill_chrome_we_spawned()

    async def test_kill_chrome_swallows_unexpected_errors(self):
        """An unexpected exception inside the kill path must be caught:
        teardown is best-effort and must not break the calling code."""
        with patch(
            "rainier.scrapers.browser.asyncio.create_subprocess_exec",
            side_effect=RuntimeError("subprocess unavailable"),
        ):
            # Should not raise.
            await BrowserManager._kill_chrome_we_spawned()

    async def test_kill_chrome_lsof_restricts_to_listen_socket(self):
        """Regression for codex iter-1 P2: lsof MUST filter to LISTEN state.

        Without ``-sTCP:LISTEN``, lsof on :9222 returns every client of
        the port (including Playwright's CDP websocket, which lives in
        this same Python process). The kill loop would then ``kill`` our
        own PID mid-teardown. Assert the LISTEN filter is present in the
        argv we hand to lsof.
        """
        captured_args: list[tuple] = []

        async def fake_exec(*args, **kwargs):
            captured_args.append(args)
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            return proc

        with patch(
            "rainier.scrapers.browser.asyncio.create_subprocess_exec",
            side_effect=fake_exec,
        ):
            await BrowserManager._kill_chrome_we_spawned()

        lsof_invocations = [a for a in captured_args if a and a[0] == "lsof"]
        assert len(lsof_invocations) == 1, captured_args
        lsof_argv = lsof_invocations[0]
        assert "-sTCP:LISTEN" in lsof_argv, (
            f"lsof must restrict to LISTEN state to avoid killing CDP "
            f"clients (including our own process); got argv={lsof_argv}"
        )


class TestWeSpawnedFlagResetOnReuse:
    """Regression for codex iter-1 P2: reusing a BrowserManager across
    multiple start()/stop() cycles must reset ``_we_spawned_chrome`` so
    a recovery-set True from cycle N never leaks into cycle N+1 and
    causes stop() to kill the operator's Chrome.
    """

    async def test_start_resets_we_spawned_flag(self):
        """Pre-condition: flag is True (simulating stale state from a
        previous cycle). After a clean start() against a healthy CDP
        endpoint, the flag must be False again."""
        mock_browser = MagicMock()
        factory, _ = _mock_playwright([mock_browser])
        with (
            patch("rainier.scrapers.browser.async_playwright", return_value=factory),
            patch.object(BrowserManager, "_force_restart_chrome", new=AsyncMock()),
        ):
            bm = BrowserManager(cdp_url="http://localhost:9222")
            bm._we_spawned_chrome = True  # stale from a prior recovery cycle
            await bm.start()

        assert bm._we_spawned_chrome is False, (
            "start() must clear stale ownership so reused managers do not "
            "kill the operator's Chrome on the next stop()"
        )
