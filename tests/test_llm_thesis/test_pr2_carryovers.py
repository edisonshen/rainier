"""Regression tests for the three PR2 carry-over P2/P3 review findings
fixed in PR3:

1. [P2] `_fetch_yesterday` in `scheduler.service.run_daily_eval` rolls back
   via TRADING days (skip Sat/Sun) instead of calendar days. On Monday the
   prior trading day is Friday, not Sunday.
2. [P2] The Discord eval renderer (`alerts.discord._format_eval_message`)
   splits afternoon and close session blocks into separate sections so the
   operator can attribute picks to the right scan.
3. [P3] `eval.compute_signal_contribution` accepts an ``eval_date`` keyword
   so the rolling-window cutoff anchors on the caller's logical "today"
   (matters for historical replay / weekly research). Same for
   `compute_verdict_hit_rate`.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from rainier.alerts.discord import _format_eval_message
from rainier.llm_thesis.eval import (
    HitRate,
    compute_signal_contribution,
    compute_verdict_hit_rate,
)


# ---------------------------------------------------------------------------
# Fix 1 — _fetch_yesterday uses trading-day rollback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_daily_eval_yesterday_uses_trading_day_rollback():
    """On Monday eval, _fetch_yesterday must query the prior FRIDAY, not Sunday.

    We patch the LLMThesisConfig + SQLAlchemy session and capture which
    `scan_date` is filtered on. The prior implementation used a calendar
    `timedelta(days=1)`; the fix uses `_trading_days_back(eval_date, 1)`.
    """
    from rainier.core.config import (
        AlertsConfig,
        AppConfig,
        DiscordConfig,
        LLMThesisConfig,
        ScrapingConfig,
        ScrapingSchedule,
        Settings,
    )

    settings = Settings(
        database_url="postgresql://test:test@localhost/test",
        app=AppConfig(timezone="America/Los_Angeles"),
        scraping=ScrapingConfig(
            schedule=ScrapingSchedule(
                morning="08:35", midday="10:35",
                afternoon="12:35", close="14:35",
            )
        ),
        alerts=AlertsConfig(discord=DiscordConfig(enabled=False, webhook_url="")),
        llm_thesis=LLMThesisConfig(),
    )

    captured_filter_args: list = []

    cm = MagicMock()
    chain = (
        cm.__enter__.return_value.query.return_value
        .join.return_value
    )

    def _filter(*args, **_kwargs):
        # Capture the filter arguments so we can introspect what scan_date
        # was used. SQLAlchemy BinaryExpression `.right.value` is the value.
        captured_filter_args.append(args)
        next_chain = MagicMock()
        next_chain.order_by.return_value.all.return_value = []
        return next_chain

    chain.filter.side_effect = _filter
    cm.__exit__.return_value = False

    with (
        patch(
            "rainier.scheduler.service.load_settings_fresh",
            return_value=settings,
        ),
        patch(
            "rainier.llm_thesis.eval.evaluate_horizon", return_value=0
        ),
        patch(
            "rainier.llm_thesis.eval.compute_signal_contribution", return_value=[]
        ),
        patch(
            "rainier.llm_thesis.eval.compute_verdict_hit_rate", return_value={}
        ),
        patch(
            "rainier.alerts.discord.send_eval_report"
        ),
        patch("rainier.core.database.get_session", return_value=cm),
    ):
        from rainier.scheduler.service import run_daily_eval
        # Monday 2026-05-04. Prior trading day is Friday 2026-05-01.
        await run_daily_eval(eval_date_iso="2026-05-04")

    # Pull the scan_date value out of the captured filter args. The call
    # filters on `ThesisEvaluation.scan_date == prior` so the prior date is
    # one of the comparison expressions.
    expected_friday = date(2026, 5, 1)
    sundays = date(2026, 5, 3)

    found_dates: set = set()
    for args in captured_filter_args:
        for expr in args:
            # Compare against the .right.value attribute of a comparator;
            # tolerate non-comparator arguments by ignoring AttributeError.
            try:
                v = expr.right.value
            except AttributeError:
                continue
            if isinstance(v, date):
                found_dates.add(v)

    assert expected_friday in found_dates, (
        f"trading-day rollback should query {expected_friday}; "
        f"saw filters on {found_dates}"
    )
    assert sundays not in found_dates, (
        f"calendar-day rollback regression: must NOT query {sundays}; "
        f"saw filters on {found_dates}"
    )


# ---------------------------------------------------------------------------
# Fix 2 — Discord eval renderer splits sessions
# ---------------------------------------------------------------------------


def _hit_rates_empty():
    return {
        v: [
            HitRate(v, "1d", 0, 0.0, 0.0),
            HitRate(v, "5d", 0, 0.0, 0.0),
            HitRate(v, "10d", 0, 0.0, 0.0),
        ]
        for v in ("setup_long", "watch", "no_setup")
    }


def test_format_eval_message_splits_afternoon_and_close_sections():
    """When yesterday_rows mixes 'afternoon' and 'close', render two
    distinct headers — not one merged block.
    """
    rows = [
        {
            "scan_date": date(2026, 5, 6),
            "session_name": "afternoon",
            "symbol": "NVDA",
            "verdict": "setup_long",
            "llm_confidence": 8,
            "return_pct": 0.02,
            "hit": True,
        },
        {
            "scan_date": date(2026, 5, 6),
            "session_name": "close",
            "symbol": "TSLA",
            "verdict": "watch",
            "llm_confidence": 5,
            "return_pct": -0.01,
            "hit": False,
        },
        {
            "scan_date": date(2026, 5, 6),
            "session_name": "afternoon",
            "symbol": "AAPL",
            "verdict": "no_setup",
            "llm_confidence": 3,
            "return_pct": 0.001,
            "hit": True,
        },
    ]
    msg = _format_eval_message(
        eval_date=date(2026, 5, 7),
        yesterday_rows=rows,
        base_rates=_hit_rates_empty(),
        signal_contribs=[],
    )

    assert "Yesterday's afternoon scan" in msg
    assert "Yesterday's close scan" in msg
    # Both NVDA (afternoon) and TSLA (close) appear with their own picks.
    assert "NVDA" in msg
    assert "TSLA" in msg
    assert "AAPL" in msg

    afternoon_idx = msg.index("Yesterday's afternoon scan")
    close_idx = msg.index("Yesterday's close scan")
    nvda_idx = msg.index("NVDA")
    aapl_idx = msg.index("AAPL")
    tsla_idx = msg.index("TSLA")
    # NVDA + AAPL group under afternoon (before close header); TSLA under close.
    assert afternoon_idx < nvda_idx < close_idx
    assert afternoon_idx < aapl_idx < close_idx
    assert close_idx < tsla_idx


def test_format_eval_message_single_session_unchanged():
    """Single-session rendering should still produce one header (back-compat)."""
    rows = [
        {
            "scan_date": date(2026, 5, 6),
            "session_name": "afternoon",
            "symbol": "NVDA",
            "verdict": "setup_long",
            "llm_confidence": 8,
            "return_pct": 0.02,
            "hit": True,
        }
    ]
    msg = _format_eval_message(
        eval_date=date(2026, 5, 7),
        yesterday_rows=rows,
        base_rates=_hit_rates_empty(),
        signal_contribs=[],
    )
    assert msg.count("Yesterday's afternoon scan") == 1
    assert "Yesterday's close scan" not in msg


# ---------------------------------------------------------------------------
# Fix 3 — compute_signal_contribution honors eval_date
# ---------------------------------------------------------------------------


def test_compute_signal_contribution_eval_date_anchors_window():
    """Passing eval_date=2026-04-30, days=30 must filter on
    scan_date >= (2026-04-30 - 30 days) regardless of date.today().
    """
    captured_cutoffs: list[date] = []

    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False

    def _execute(stmt):
        # Walk through the WHERE clauses to extract the date cutoff.
        # `.whereclause` is a BooleanClauseList for and_(...).
        try:
            clauses = stmt.whereclause.clauses  # type: ignore[attr-defined]
        except AttributeError:
            clauses = []
        for clause in clauses:
            try:
                v = clause.right.value
            except AttributeError:
                continue
            if isinstance(v, date):
                captured_cutoffs.append(v)
        result = MagicMock()
        result.all.return_value = []
        return result

    sess.execute.side_effect = _execute

    with patch("rainier.llm_thesis.eval.get_session", return_value=cm):
        compute_signal_contribution(
            days=30, horizon="5d", eval_date=date(2026, 4, 30),
        )

    # The cutoff stored on the WHERE clause must equal eval_date - 30 days.
    expected_cutoff = date(2026, 4, 30) - __import__("datetime").timedelta(days=30)
    assert expected_cutoff in captured_cutoffs


def test_compute_signal_contribution_default_uses_today():
    """Without eval_date, default behavior anchors on date.today() — the
    backwards-compatible legacy path.
    """
    captured: list[date] = []

    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False

    def _execute(stmt):
        try:
            for clause in stmt.whereclause.clauses:  # type: ignore[attr-defined]
                try:
                    v = clause.right.value
                except AttributeError:
                    continue
                if isinstance(v, date):
                    captured.append(v)
        except AttributeError:
            pass
        result = MagicMock()
        result.all.return_value = []
        return result

    sess.execute.side_effect = _execute

    with patch("rainier.llm_thesis.eval.get_session", return_value=cm):
        compute_signal_contribution(days=30, horizon="5d")

    today = date.today()
    expected_cutoff = today - __import__("datetime").timedelta(days=30)
    assert expected_cutoff in captured


def test_compute_verdict_hit_rate_eval_date_anchors_window():
    """Same eval_date contract for the verdict aggregator."""
    captured: list[date] = []

    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False

    def _execute(stmt):
        try:
            clause = stmt.whereclause
            try:
                v = clause.right.value
            except AttributeError:
                v = None
            if isinstance(v, date):
                captured.append(v)
        except AttributeError:
            pass
        result = MagicMock()
        result.all.return_value = []
        return result

    sess.execute.side_effect = _execute

    with patch("rainier.llm_thesis.eval.get_session", return_value=cm):
        compute_verdict_hit_rate(days=30, eval_date=date(2026, 4, 30))

    expected_cutoff = date(2026, 4, 30) - __import__("datetime").timedelta(days=30)
    assert expected_cutoff in captured
