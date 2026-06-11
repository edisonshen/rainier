"""Phase 3 — weekly missed-winner sweep (TASK-PLAN qu100-miss-sweep-6b63).

Covers: window arithmetic, the shared cohort selector, the +10% inclusive
threshold, four-tier attribution (highest funnel stage wins), the held
predicate, R-B dodged losers, sweep-owned ingest anchoring (never the partial
Friday bar), insight emission, snapshot re-render vs --regenerate, and the
zero-winner clean-empty path. Deterministic fixtures, no network.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select, text

from rainier.core.models import (
    MoneyFlowSnapshot,
    PaperTrade,
    ResearchInsight,
    ScreenedStockRecord,
    StockPrice,
)
from rainier.paper.calendar import DEFAULT_CALENDAR
from rainier.paper.ingest import canonical_instant, get_current_qu100_cohort
from rainier.paper.sweep import (
    BUCKET_NO_PATTERN,
    BUCKET_NOT_TOP5,
    BUCKET_RANK_TOO_LOW,
    BUCKET_VERDICT,
    compute_weekly_payload,
    compute_window,
    persist_weekly_snapshot,
    render_weekly_payload,
    sweep_missed_winners,
)

requires_postgres = pytest.mark.requires_postgres

AS_OF = date(2026, 6, 12)  # Friday — the 09:00 PT run day (in-progress session)
WINDOW_END = date(2026, 6, 11)  # Thursday — last completed priced day
WINDOW_START = date(2026, 5, 28)  # 10 trading days before window_end
CAP_T1 = datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc)
CAP_T2 = datetime(2026, 6, 11, 21, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Window arithmetic (pure — no DB)
# ---------------------------------------------------------------------------


def test_window_friday_run_uses_last_completed_day():
    """A Friday 09:00 run anchors to Thursday — never the in-progress session —
    and spans exactly 10 trading-day intervals (11 closes)."""
    start, end = compute_window(AS_OF)
    assert end == WINDOW_END
    assert start == WINDOW_START
    sessions = DEFAULT_CALENDAR.sessions_between(start, end)
    assert len(sessions) == 11  # 11 closes, 10 intervals


def test_window_weekend_and_monday_runs():
    # Saturday run: Friday is complete → window_end = Friday.
    start, end = compute_window(date(2026, 6, 13))
    assert end == date(2026, 6, 12)
    assert start == date(2026, 5, 29)
    # Monday run: window_end = the prior Friday.
    start, end = compute_window(date(2026, 6, 15))
    assert end == date(2026, 6, 12)


def test_insight_kinds_extended_with_missed_winner():
    from rainier.llm_thesis.research import INSIGHT_KINDS

    assert "missed_winner" in INSIGHT_KINDS


# ---------------------------------------------------------------------------
# DB fixtures / helpers (pg lane)
# ---------------------------------------------------------------------------


def _stock(s, symbol):
    s.execute(
        text("INSERT INTO stocks (symbol) VALUES (:s) ON CONFLICT DO NOTHING"),
        {"s": symbol},
    )


def _mfs(s, symbol, rank, data_date, captured_at, ranking_type="top100"):
    _stock(s, symbol)
    s.add(
        MoneyFlowSnapshot(
            captured_at=captured_at,
            capture_session="close",
            data_date=data_date,
            view_type="daily",
            ranking_type=ranking_type,
            symbol=symbol,
            rank=rank,
        )
    )
    s.commit()


def _seed_cohort(s, symbols, data_date=date(2026, 6, 11), captured_at=CAP_T2):
    for i, sym in enumerate(symbols, start=1):
        _mfs(s, sym, i, data_date, captured_at)


def _close(s, symbol, d, px):
    _stock(s, symbol)
    s.add(
        StockPrice(
            symbol=symbol, date=canonical_instant(d), open=px, high=px,
            low=px, close=px, volume=1,
        )
    )
    s.commit()


def _window_closes(s, symbol, start_px, end_px):
    _close(s, symbol, WINDOW_START, start_px)
    _close(s, symbol, WINDOW_END, end_px)


def _thesis(s, tid, recommendation):
    s.execute(
        text(
            "INSERT INTO analysis_results (id, llm_model, prompt_template, "
            "recommendation) VALUES (:i,'m','t',:r) ON CONFLICT DO NOTHING"
        ),
        {"i": tid, "r": recommendation},
    )
    s.commit()


def _screened(s, symbol, scan_date, *, pattern=None, thesis_id=None,
              session_name="close", rank=10):
    _stock(s, symbol)
    s.add(
        ScreenedStockRecord(
            scan_date=scan_date, session_name=session_name, symbol=symbol,
            rule_rank=rank, composite_score=0.5, pattern_type=pattern,
            thesis_id=thesis_id,
        )
    )
    s.commit()


def _declined(s, tid, symbol, scan_date, verdict="watch"):
    """A top-5 thesis the LLM declined (watch/no_setup) on scan_date."""
    _thesis(s, tid, verdict)
    _screened(s, symbol, scan_date, pattern="w_bottom", thesis_id=tid, rank=1)


def _filled_position(s, tid, symbol, entry_date, *, exit_date=None,
                     status="open"):
    _thesis(s, tid, "setup_long")
    _stock(s, symbol)
    kw = {}
    if status == "closed":
        kw = dict(exit_price=90.0, exit_reason="stop_loss", return_pct=-0.10,
                  pnl=-1000.0)
    s.add(
        PaperTrade(
            thesis_id=tid, symbol=symbol, scan_date=entry_date,
            session_name="close", status=status, planned_entry_price=100.0,
            stop_loss=50.0, target_price=500.0, entry_date=entry_date,
            entry_price=100.0, shares=100, allocated_amount=10000.0,
            residual_cash=0.0, price_basis="adjusted", exit_date=exit_date,
            **kw,
        )
    )
    s.commit()


def _unfilled_position(s, tid, symbol, scan_date, status="pending"):
    _thesis(s, tid, "setup_long")
    _stock(s, symbol)
    s.add(
        PaperTrade(
            thesis_id=tid, symbol=symbol, scan_date=scan_date,
            session_name="close", status=status, planned_entry_price=100.0,
            stop_loss=50.0, target_price=500.0,
        )
    )
    s.commit()


def _flagged(payload):
    return {w["symbol"]: w for w in payload["missed_winners"]}


# ---------------------------------------------------------------------------
# Cohort selector (acceptance 1)
# ---------------------------------------------------------------------------


@requires_postgres
def test_cohort_latest_data_date_then_latest_captured_at(pg_legacy_session):
    s = pg_legacy_session
    dd = date(2026, 6, 10)
    # Early capture of 6/10 (stale membership) + later capture (fresh).
    _mfs(s, "AAA", 1, dd, CAP_T1)
    _mfs(s, "BBB", 2, dd, CAP_T1)
    _mfs(s, "AAA", 1, dd, CAP_T2)
    _mfs(s, "CCC", 2, dd, CAP_T2)
    # Non-top100 partial with a LATER data_date must not win.
    _mfs(s, "ETF", 1, date(2026, 6, 11), CAP_T2, ranking_type="etf")
    # Backfilled older data_date SHARING the latest captured_at must not win.
    _mfs(s, "OLD", 1, date(2026, 6, 9), CAP_T2)

    cohort = get_current_qu100_cohort(AS_OF)
    assert [c["symbol"] for c in cohort] == ["AAA", "CCC"]
    assert all(c["data_date"] == dd for c in cohort)
    assert all(c["captured_at"] == CAP_T2 for c in cohort)
    assert [c["rank"] for c in cohort] == [1, 2]


@requires_postgres
def test_cohort_historical_as_of_picks_past_cohort(pg_legacy_session):
    """--regenerate for a past week selects max data_date <= as_of, not today's."""
    s = pg_legacy_session
    _mfs(s, "OLD", 1, date(2026, 6, 1), CAP_T1)
    _mfs(s, "NEW", 1, date(2026, 6, 8), CAP_T2)
    cohort = get_current_qu100_cohort(date(2026, 6, 5))
    assert [c["symbol"] for c in cohort] == ["OLD"]
    assert cohort[0]["data_date"] == date(2026, 6, 1)


@requires_postgres
def test_cohort_empty_when_no_top100(pg_legacy_session):
    assert get_current_qu100_cohort(AS_OF) == []


# ---------------------------------------------------------------------------
# Threshold boundary (validation: +10% exactly flagged; just below not)
# ---------------------------------------------------------------------------


@requires_postgres
def test_threshold_plus_10_pct_inclusive(pg_legacy_session):
    s = pg_legacy_session
    _seed_cohort(s, ["WINA", "JUST"])
    _window_closes(s, "WINA", 100.0, 110.0)   # +10.0% exactly → flagged
    _window_closes(s, "JUST", 100.0, 109.9)   # just below → not flagged

    payload = compute_weekly_payload(AS_OF)
    flagged = _flagged(payload)
    assert "WINA" in flagged
    assert "JUST" not in flagged
    assert flagged["WINA"]["bucket"] == BUCKET_RANK_TOO_LOW
    assert flagged["WINA"]["return_pct"] == pytest.approx(0.10)
    assert flagged["WINA"]["rank"] == 1


@requires_postgres
def test_threshold_float_epsilon_inclusive_from_below(pg_legacy_session):
    """3.00→3.30 is mathematically exactly +10% but floats to just BELOW 0.10
    (0.09999999999999987) — the _EPS guard must still flag it (review iter-1,
    the miss-side twin of the dodge-side 100→90 case)."""
    s = pg_legacy_session
    assert 3.3 / 3.0 - 1.0 < 0.10  # the float really lands below
    _seed_cohort(s, ["EPSW"])
    _window_closes(s, "EPSW", 3.0, 3.3)
    assert "EPSW" in _flagged(compute_weekly_payload(AS_OF))


# ---------------------------------------------------------------------------
# Attribution tiers (acceptance 4; multi-day = highest stage wins)
# ---------------------------------------------------------------------------


@requires_postgres
def test_attribution_tiers_and_highest_stage_wins(pg_legacy_session):
    s = pg_legacy_session
    syms = ["VRD", "MIX", "PAT", "NOP", "LOW"]
    _seed_cohort(s, syms)
    for sym in syms:
        _window_closes(s, sym, 100.0, 120.0)  # all +20% winners, none held

    # VRD: declined thesis (watch) in-window on day 2 → tier (i). Also
    # unscreened later in the window — the highest stage still wins.
    _declined(s, 901, "VRD", date(2026, 6, 1), verdict="watch")
    # MIX: pattern-screened on 6/3, then declined (no_setup) on 6/5 →
    # highest stage = tier (i), NOT not_in_top5.
    _screened(s, "MIX", date(2026, 6, 3), pattern="bull_flag")
    _declined(s, 902, "MIX", date(2026, 6, 5), verdict="no_setup")
    # PAT: screened with a pattern, never got a thesis → tier (ii).
    _screened(s, "PAT", date(2026, 6, 4), pattern="w_bottom")
    # NOP: screened without a pattern → tier (iii).
    _screened(s, "NOP", date(2026, 6, 4), pattern=None)
    # LOW: never screened → tier (iv).

    payload = compute_weekly_payload(AS_OF)
    flagged = _flagged(payload)
    assert flagged["VRD"]["bucket"] == BUCKET_VERDICT
    assert flagged["MIX"]["bucket"] == BUCKET_VERDICT
    assert flagged["PAT"]["bucket"] == BUCKET_NOT_TOP5
    assert flagged["NOP"]["bucket"] == BUCKET_NO_PATTERN
    assert flagged["LOW"]["bucket"] == BUCKET_RANK_TOO_LOW

    assert payload["bucket_counts"] == {
        BUCKET_VERDICT: 2, BUCKET_NOT_TOP5: 1,
        BUCKET_NO_PATTERN: 1, BUCKET_RANK_TOO_LOW: 1,
    }
    assert payload["dominant_bucket"] == BUCKET_VERDICT
    assert payload["tuning_hypothesis"]  # one documented hypothesis (acceptance 7)


@requires_postgres
def test_attribution_outside_window_rows_ignored(pg_legacy_session):
    s = pg_legacy_session
    _seed_cohort(s, ["OUT"])
    _window_closes(s, "OUT", 100.0, 115.0)
    # Screened + declined BEFORE the window only → rank_too_low.
    _declined(s, 903, "OUT", date(2026, 5, 27), verdict="watch")
    payload = compute_weekly_payload(AS_OF)
    assert _flagged(payload)["OUT"]["bucket"] == BUCKET_RANK_TOO_LOW


# ---------------------------------------------------------------------------
# Held predicate (acceptance 5)
# ---------------------------------------------------------------------------


@requires_postgres
def test_held_predicate_overlap_and_unfilled(pg_legacy_session):
    s = pg_legacy_session
    syms = ["HOPN", "HCLS", "OLDX", "PEND", "EXPD"]
    _seed_cohort(s, syms)
    for sym in syms:
        _window_closes(s, sym, 100.0, 130.0)  # all winners

    # Open position filled in-window → held → excluded.
    _filled_position(s, 911, "HOPN", date(2026, 6, 3), status="open")
    # Closed position whose [entry, exit] touches the window start → held.
    _filled_position(s, 912, "HCLS", date(2026, 5, 20),
                     exit_date=date(2026, 5, 28), status="closed")
    # Closed BEFORE the window → not held → flagged.
    _filled_position(s, 913, "OLDX", date(2026, 5, 18),
                     exit_date=date(2026, 5, 27), status="closed")
    # Pending (never filled) → flagged.
    _unfilled_position(s, 914, "PEND", date(2026, 6, 9), status="pending")
    # Never-filled expired → flagged.
    _unfilled_position(s, 915, "EXPD", date(2026, 6, 2), status="expired")

    flagged = _flagged(compute_weekly_payload(AS_OF))
    assert "HOPN" not in flagged
    assert "HCLS" not in flagged
    assert {"OLDX", "PEND", "EXPD"} <= set(flagged)


# ---------------------------------------------------------------------------
# Missing prices + zero-winner week (validation)
# ---------------------------------------------------------------------------


@requires_postgres
def test_missing_prices_explicit_and_zero_winner_clean(pg_legacy_session):
    s = pg_legacy_session
    _seed_cohort(s, ["FLAT", "GONE"])
    _window_closes(s, "FLAT", 100.0, 101.0)  # not a winner
    # GONE has no prices at all.

    payload = compute_weekly_payload(AS_OF)
    assert payload["missed_winners"] == []
    assert payload["missing_price_symbols"] == ["GONE"]
    assert payload["dominant_bucket"] is None
    # Clean empty render + snapshot persist (no poison retry).
    text_out = render_weekly_payload(payload)
    assert "none" in text_out.lower()
    persist_weekly_snapshot(AS_OF, payload)
    from rainier.paper.report import REPORT_TYPE_WEEKLY, load_snapshot

    assert load_snapshot(REPORT_TYPE_WEEKLY, AS_OF) == payload


# ---------------------------------------------------------------------------
# R-B dodged losers (acceptance 6)
# ---------------------------------------------------------------------------


@requires_postgres
def test_rb_dodged_losers_inclusive_and_held_excluded(pg_legacy_session):
    s = pg_legacy_session
    _seed_cohort(s, ["DODG", "HLOS", "DWIN"])
    _window_closes(s, "DODG", 100.0, 90.0)   # −10% exactly → counted
    _window_closes(s, "HLOS", 100.0, 80.0)   # held loser → excluded
    _window_closes(s, "DWIN", 100.0, 95.0)   # declined, only −5% → not counted

    _declined(s, 921, "DODG", date(2026, 6, 2), verdict="watch")
    _declined(s, 922, "HLOS", date(2026, 6, 2), verdict="no_setup")
    _declined(s, 923, "DWIN", date(2026, 6, 2), verdict="watch")
    # HLOS was nonetheless held in-window (a different thesis filled).
    _filled_position(s, 924, "HLOS", date(2026, 6, 3), status="open")

    payload = compute_weekly_payload(AS_OF)
    assert payload["dodged_losers"]["count"] == 1
    assert [d["symbol"] for d in payload["dodged_losers"]["names"]] == ["DODG"]


# ---------------------------------------------------------------------------
# Sweep-owned ingest anchoring (acceptance 3)
# ---------------------------------------------------------------------------


@requires_postgres
def test_sweep_ingest_anchored_at_window_end_never_partial_friday(
    pg_legacy_session,
):
    s = pg_legacy_session
    _seed_cohort(s, ["ANCH"])

    sessions = DEFAULT_CALENDAR.sessions_between(WINDOW_START, WINDOW_END)
    assert len(sessions) == 11
    calls = []

    def fetch_fn(symbols, start, end):
        calls.append((tuple(symbols), start, end))
        # Source also returns the in-progress Friday bar — the sweep must
        # never persist it (a mid-session upsert would later read complete).
        days = sessions + [AS_OF]
        bars = [
            {"date": d, "open": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i,
             "close": 100.0 + i, "volume": 1}
            for i, d in enumerate(days)
        ]
        return {sym: bars for sym in symbols}

    sweep_missed_winners(as_of=AS_OF, fetch_fn=fetch_fn)

    assert calls, "sweep must run its own cohort ingest"
    _syms, start, end = calls[0]
    assert end == WINDOW_END, "ingest anchored at window_end, not the run day"
    assert start == WINDOW_START, "fetch window must cover all 11 sessions"

    rows = s.execute(
        select(StockPrice.date).where(StockPrice.symbol == "ANCH")
    ).scalars().all()
    got = {d.date() if isinstance(d, datetime) else d for d in rows}
    assert canonical_instant(AS_OF).date() not in got, "partial Friday bar written"
    assert {d for d in sessions} == got  # all 11 anchor-to-end closes present


# ---------------------------------------------------------------------------
# Insight emission (acceptance 8) + snapshot persistence (acceptance 7)
# ---------------------------------------------------------------------------


@requires_postgres
def test_sweep_emits_one_insight_per_week_and_persists_snapshot(
    pg_legacy_session,
):
    s = pg_legacy_session
    _seed_cohort(s, ["WINA"])
    sessions = DEFAULT_CALENDAR.sessions_between(WINDOW_START, WINDOW_END)

    def fetch_fn(symbols, start, end):
        bars = [
            {"date": d, "open": 100.0, "high": 130.0, "low": 99.0,
             "close": 100.0 if d != WINDOW_END else 125.0, "volume": 1}
            for d in sessions
        ]
        return {sym: bars for sym in symbols}

    payload = sweep_missed_winners(as_of=AS_OF, fetch_fn=fetch_fn)
    assert [w["symbol"] for w in payload["missed_winners"]] == ["WINA"]

    iso = AS_OF.isocalendar()
    subject = f"{iso[0]}-W{iso[1]:02d}"
    rows = s.execute(
        select(ResearchInsight).where(ResearchInsight.kind == "missed_winner")
    ).scalars().all()
    assert len(rows) == 1
    ins = rows[0]
    assert ins.subject == subject
    assert ins.severity == "info"
    assert ins.action["kind"] == "noop"
    ev_winners = ins.evidence["winners"]
    assert ev_winners[0]["symbol"] == "WINA"
    assert ev_winners[0]["bucket"] == BUCKET_RANK_TOO_LOW
    assert ev_winners[0]["return_pct"] == pytest.approx(0.25)

    # Snapshot is the durable record.
    from rainier.paper.report import REPORT_TYPE_WEEKLY, load_snapshot

    assert load_snapshot(REPORT_TYPE_WEEKLY, AS_OF) == payload

    # Re-run the same week → still ONE pending insight (UPSERT, not a dup).
    sweep_missed_winners(as_of=AS_OF, fetch_fn=fetch_fn)
    s.expire_all()
    rows = s.execute(
        select(ResearchInsight).where(ResearchInsight.kind == "missed_winner")
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].recurrence_count == 2

    # Coverage disclosures (acceptance 10).
    disclosure = payload["disclosure"].lower()
    assert "coverage" in disclosure
    assert "rank_too_low" in disclosure
    assert "survivorship" in disclosure


def test_weekly_discord_failure_never_leaks_webhook_token(caplog):
    """httpx.HTTPStatusError.__str__ embeds the request URL, and a Discord
    webhook URL carries its secret token. The weekly send's failure log must
    carry status + exception class only — never str(exc) or a traceback.
    Regression for codex [P1] 2026-06-09 (commit 96fbd13)."""
    import logging
    from unittest.mock import MagicMock, patch

    import httpx

    from rainier.paper.sweep import send_weekly_paper_report

    secret = "WEEKLYtokenSECRET777"
    url = f"https://discord.com/api/webhooks/424242/{secret}"

    class _Cfg:
        enabled = True
        webhook_url = url

    # Reproduce the real leak vector: raise_for_status() auto-generates a
    # message embedding the request URL (which carries the token).
    req = httpx.Request("POST", url)
    resp = httpx.Response(401, request=req)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        err = e
    assert secret in str(err)  # the leak vector exists in the exception

    with caplog.at_level(logging.INFO, logger="rainier.paper.sweep"):
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(side_effect=err)
            )
            ok = send_weekly_paper_report({}, _Cfg())

    assert ok is False
    fmt = logging.Formatter()
    blob = "\n".join(fmt.format(r) for r in caplog.records)
    assert secret not in blob, "webhook token leaked into logs"
    assert "webhooks/424242" not in blob
    assert "weekly_sweep_discord_failed" in blob
    assert "401" in blob and "HTTPStatusError" in blob


def test_weekly_discord_skipped_when_disabled_or_no_webhook():
    """No config / disabled / missing webhook → returns False without ever
    touching the network (review iter-1 coverage)."""
    from unittest.mock import patch

    from rainier.paper.sweep import send_weekly_paper_report

    class _Disabled:
        enabled = False
        webhook_url = "https://discord.com/api/webhooks/1/x"

    class _NoUrl:
        enabled = True
        webhook_url = None

    with patch("rainier.paper.sweep.send_daily_report") as mock_send:
        assert send_weekly_paper_report({}, None) is False
        assert send_weekly_paper_report({}, _Disabled()) is False
        assert send_weekly_paper_report({}, _NoUrl()) is False
    mock_send.assert_not_called()


def test_weekly_discord_success_sends_rendered_text():
    from unittest.mock import patch

    from rainier.paper.sweep import send_weekly_paper_report

    class _Cfg:
        enabled = True
        webhook_url = "https://discord.com/api/webhooks/1/x"

    payload = {"as_of_date": "2026-06-12", "missed_winners": [],
               "dodged_losers": {"count": 0, "names": []}}
    cfg = _Cfg()
    with patch("rainier.paper.sweep.send_daily_report") as mock_send:
        assert send_weekly_paper_report(payload, cfg) is True
    mock_send.assert_called_once_with(render_weekly_payload(payload), cfg)


def test_weekly_discord_malformed_payload_never_raises():
    """render runs INSIDE the try (review iter-1): a malformed payload returns
    False instead of raising into a caller that might stringify it."""
    from unittest.mock import patch

    from rainier.paper.sweep import send_weekly_paper_report

    class _Cfg:
        enabled = True
        webhook_url = "https://discord.com/api/webhooks/1/x"

    bad = {"missed_winners": [{}]}  # w["symbol"] raises KeyError in render
    with patch("rainier.paper.sweep.send_daily_report") as mock_send:
        assert send_weekly_paper_report(bad, _Cfg()) is False
    mock_send.assert_not_called()


def test_render_truncates_long_winner_and_missing_lists():
    """>15 winners → '… +N more'; >10 missing symbols → '(+N more)'
    (review iter-1 coverage for the Discord-size truncation arithmetic)."""
    winners = [
        {"symbol": f"W{i:02d}", "rank": i + 1, "return_pct": 0.20,
         "bucket": BUCKET_RANK_TOO_LOW}
        for i in range(16)
    ]
    payload = {
        "as_of_date": "2026-06-12",
        "missed_winners": winners,
        "bucket_counts": {BUCKET_RANK_TOO_LOW: 16},
        "dominant_bucket": BUCKET_RANK_TOO_LOW,
        "tuning_hypothesis": "h",
        "dodged_losers": {"count": 0, "names": []},
        "missing_price_symbols": [f"M{i:02d}" for i in range(11)],
        "disclosure": "d",
    }
    out = render_weekly_payload(payload)
    assert "… +1 more" in out
    assert "W14" in out and "W15" not in out  # exactly 15 winner lines
    assert "(+1 more)" in out
    assert "M09" in out and "M10" not in out  # exactly 10 missing shown


@requires_postgres
def test_sweep_empty_cohort_skips_ingest_and_persists_clean(pg_legacy_session):
    """Zero cohort + zero declined → no fetch at all, a clean size-0 snapshot,
    and one empty missed_winner insight (review iter-1 coverage for the
    empty-cohort guard branch)."""
    s = pg_legacy_session

    def fetch_fn(symbols, start, end):  # noqa: ARG001
        raise AssertionError("empty cohort must never trigger an ingest fetch")

    payload = sweep_missed_winners(as_of=AS_OF, fetch_fn=fetch_fn)
    assert payload["cohort"] == {"size": 0, "data_date": None, "captured_at": None}
    assert payload["missed_winners"] == []
    assert payload["dominant_bucket"] is None

    from rainier.paper.report import REPORT_TYPE_WEEKLY, load_snapshot

    assert load_snapshot(REPORT_TYPE_WEEKLY, AS_OF) == payload
    rows = s.execute(
        select(ResearchInsight).where(ResearchInsight.kind == "missed_winner")
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].evidence["winners"] == []


# ---------------------------------------------------------------------------
# CLI: --week snapshot re-render vs --week --regenerate (validation, separate)
# ---------------------------------------------------------------------------


@requires_postgres
def test_cli_week_rerenders_from_snapshot_only(pg_legacy_session):
    from click.testing import CliRunner

    from rainier.cli import cli

    s = pg_legacy_session
    # Raw inputs that would compute differently than the stored snapshot.
    _seed_cohort(s, ["RAWW"])
    _window_closes(s, "RAWW", 100.0, 150.0)
    sentinel = {
        "report_type": "weekly",
        "as_of_date": AS_OF.isoformat(),
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "cohort": {"size": 1, "data_date": "2026-06-11", "captured_at": None},
        "missed_winners": [
            {"symbol": "SENT", "rank": 1, "return_pct": 0.42,
             "bucket": BUCKET_RANK_TOO_LOW}
        ],
        "bucket_counts": {BUCKET_RANK_TOO_LOW: 1},
        "dominant_bucket": BUCKET_RANK_TOO_LOW,
        "tuning_hypothesis": "sentinel",
        "dodged_losers": {"count": 0, "names": []},
        "missing_price_symbols": [],
        "disclosure": "sentinel disclosure",
    }
    persist_weekly_snapshot(AS_OF, sentinel)

    result = CliRunner().invoke(
        cli, ["paper", "report", "--week", "--date", AS_OF.isoformat()]
    )
    assert result.exit_code == 0, result.output
    assert "SENT" in result.output  # snapshot content
    assert "RAWW" not in result.output  # NOT recomputed from raw


@requires_postgres
def test_cli_week_regenerate_recomputes_from_raw_and_upserts(pg_legacy_session):
    from click.testing import CliRunner

    from rainier.cli import cli
    from rainier.paper.report import REPORT_TYPE_WEEKLY, load_snapshot

    s = pg_legacy_session
    _seed_cohort(s, ["RAWW"])
    _window_closes(s, "RAWW", 100.0, 150.0)
    stale = {"report_type": "weekly", "as_of_date": AS_OF.isoformat(),
             "missed_winners": [{"symbol": "SENT", "rank": 1,
                                 "return_pct": 0.42,
                                 "bucket": BUCKET_RANK_TOO_LOW}]}
    persist_weekly_snapshot(AS_OF, stale)

    from unittest.mock import patch

    with patch("rainier.paper.sweep.send_daily_report") as mock_send:
        result = CliRunner().invoke(
            cli,
            ["paper", "report", "--week", "--regenerate", "--date",
             AS_OF.isoformat()],
        )
    assert result.exit_code == 0, result.output
    assert "RAWW" in result.output  # recomputed from raw inputs
    assert "SENT" not in result.output

    refreshed = load_snapshot(REPORT_TYPE_WEEKLY, AS_OF)
    assert [w["symbol"] for w in refreshed["missed_winners"]] == ["RAWW"]

    # --regenerate is compute + upsert + render ONLY: no insight emission,
    # no Discord (review iter-1 — pins the docstring's promise).
    mock_send.assert_not_called()
    rows = s.execute(
        select(ResearchInsight).where(ResearchInsight.kind == "missed_winner")
    ).scalars().all()
    assert rows == []


@requires_postgres
def test_cli_week_without_snapshot_fails_cleanly(pg_legacy_session):
    from click.testing import CliRunner

    from rainier.cli import cli

    result = CliRunner().invoke(
        cli, ["paper", "report", "--week", "--date", "2031-01-03"]
    )
    assert result.exit_code != 0
    assert "No weekly snapshot" in result.output
