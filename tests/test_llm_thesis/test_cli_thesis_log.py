"""Tests for `rainier thesis log` — codex P1: session-scoped UPDATE.

The CLI command must update exactly the (scan_date, session_name, symbol)
row, not every row for that ticker on that date. Without this scoping a
ticker that surfaces in both morning and afternoon scans would have its
outcome mirrored onto every session row, corrupting the per-scan history.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from rainier.cli import cli


def _patched_session_with_capture():
    """Return (cm, fake_session, captured) where captured holds the executed stmt."""
    fake_session = MagicMock()
    captured: dict = {}

    def _execute(stmt):
        captured["stmt"] = stmt
        result = MagicMock()
        result.rowcount = 1
        return result

    fake_session.execute.side_effect = _execute
    cm = MagicMock()
    cm.__enter__.return_value = fake_session
    cm.__exit__.return_value = False
    return cm, fake_session, captured


def test_thesis_log_requires_session():
    """--session is required so the UPDATE cannot accidentally span sessions."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thesis", "log",
            "--ticker", "NVDA",
            "--date", "2026-05-07",
            "--action", "took",
        ],
    )
    assert result.exit_code != 0
    assert "session" in result.output.lower()


def test_thesis_log_filters_on_session_name():
    """The compiled UPDATE must include a WHERE clause on session_name."""
    cm, _fake_session, captured = _patched_session_with_capture()
    runner = CliRunner()
    with patch("rainier.core.database.get_session", return_value=cm):
        result = runner.invoke(
            cli,
            [
                "thesis", "log",
                "--ticker", "nvda",  # confirm the upper() round-trip too
                "--date", "2026-05-07",
                "--session", "afternoon",
                "--action", "took",
                "--outcome", "+2.3%",
            ],
        )
    assert result.exit_code == 0, result.output
    stmt = captured.get("stmt")
    assert stmt is not None
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    # Must constrain on all three keys.
    assert "session_name" in sql
    assert "afternoon" in sql
    assert "scan_date" in sql
    assert "NVDA" in sql  # uppercased in the WHERE
    assert "Updated 1 row(s) for nvda on 2026-05-07 (afternoon)." in result.output


def test_thesis_log_zero_rows_raises_with_session_in_message():
    """When no row matches, the error must name the session for diagnosability."""
    fake_session = MagicMock()
    fake_session.execute.return_value = MagicMock(rowcount=0)
    cm = MagicMock()
    cm.__enter__.return_value = fake_session
    cm.__exit__.return_value = False

    runner = CliRunner()
    with patch("rainier.core.database.get_session", return_value=cm):
        result = runner.invoke(
            cli,
            [
                "thesis", "log",
                "--ticker", "ZZZ",
                "--date", "2026-05-07",
                "--session", "close",
                "--action", "skipped",
            ],
        )
    assert result.exit_code != 0
    assert "session=close" in result.output


def test_thesis_log_rejects_unknown_session_value():
    """Click Choice gates `--session` to the four scheduler sessions."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thesis", "log",
            "--ticker", "NVDA",
            "--date", "2026-05-07",
            "--session", "premarket",  # not a valid session
            "--action", "took",
        ],
    )
    assert result.exit_code != 0
    assert "premarket" in result.output or "Choice" in result.output


def test_thesis_log_accepts_all_four_scheduler_sessions(tmp_path):
    """The scheduler runs morning/midday/afternoon/close; --session must
    accept all four so a midday-screened row can be logged."""
    fake_session = MagicMock()
    fake_session.execute.return_value = MagicMock(rowcount=1)
    cm = MagicMock()
    cm.__enter__.return_value = fake_session
    cm.__exit__.return_value = False

    runner = CliRunner()
    for session in ("morning", "midday", "afternoon", "close"):
        with patch("rainier.core.database.get_session", return_value=cm):
            result = runner.invoke(
                cli,
                [
                    "thesis", "log",
                    "--ticker", "NVDA",
                    "--date", "2026-05-07",
                    "--session", session,
                    "--action", "watched",
                ],
            )
        assert result.exit_code == 0, f"session {session} failed: {result.output}"
