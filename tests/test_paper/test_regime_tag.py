"""R-C — coarse market-regime tag on weekly paper lessons.

Regime = SPY close vs its 200-day SMA on the lesson's as-of date:
above → `bull`, below → `bear`, <200 usable bars → `unknown` (never a
partial-window SMA). The tag lands in the lesson insight's evidence payload
AND its rendered rationale text. SPY history is guaranteed by a one-time
gap-aware ensure-history backfill (window_days ≥ 250); the daily paper
ingest keeps it fresh afterwards.

Postgres lane (stock_prices + research_insights ORM round-trips).
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from rainier.core.models import PaperTrade, StockPrice
from rainier.paper.ingest import canonical_instant

pytestmark = pytest.mark.requires_postgres

AS_OF = date(2026, 3, 20)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seed_spy(session, closes, *, end=AS_OF):
    """Seed one SPY bar per consecutive calendar day ending at ``end``,
    oldest-first, with the given closes."""
    session.execute(
        text("INSERT INTO stocks (symbol) VALUES ('SPY') ON CONFLICT DO NOTHING")
    )
    n = len(closes)
    session.add_all(
        StockPrice(
            symbol="SPY",
            date=canonical_instant(end - timedelta(days=n - 1 - i)),
            open=float(c),
            high=float(c),
            low=float(c),
            close=float(c),
            volume=1,
        )
        for i, c in enumerate(closes)
    )
    session.commit()


def _seed_thesis(session, tid):
    session.execute(
        text(
            "INSERT INTO analysis_results (id, llm_model, prompt_template) "
            "VALUES (:i,'m','t') ON CONFLICT DO NOTHING"
        ),
        {"i": tid},
    )


def _add_closed(session, tid, symbol, *, exit_date, return_pct, pnl, reason):
    _seed_thesis(session, tid)
    session.add(PaperTrade(
        thesis_id=tid, symbol=symbol, scan_date=date(2026, 3, 1),
        session_name="close", status="closed", planned_entry_price=100.0,
        stop_loss=90.0, target_price=120.0, entry_date=date(2026, 3, 2),
        entry_price=100.0, shares=100, residual_cash=0.0, exit_date=exit_date,
        exit_price=100 * (1 + return_pct), exit_reason=reason,
        return_pct=return_pct, pnl=pnl,
    ))
    session.commit()


def _mk_fetch_fn(calls):
    """Deterministic fetch_fn: one synthetic SPY bar per calendar day in the
    requested [start, end] range (ingest keeps only trading sessions)."""

    def fetch(symbols, start, end):
        calls.append((tuple(symbols), start, end))
        bars = []
        d, i = start, 0
        while d <= end:
            px = 100.0 + i * 0.1
            bars.append({
                "date": d, "open": px, "high": px, "low": px,
                "close": px, "volume": 1000,
            })
            d += timedelta(days=1)
            i += 1
        return {"SPY": bars}

    return fetch


# ---------------------------------------------------------------------------
# compute_market_regime — SPY close vs 200-day SMA
# ---------------------------------------------------------------------------


def test_regime_bull_close_above_sma(pg_legacy_session):
    from rainier.llm_thesis.research import compute_market_regime

    # Rising closes 100..299: latest 299 > SMA(200) = 199.5 → bull.
    _seed_spy(pg_legacy_session, range(100, 300))
    assert compute_market_regime(as_of=AS_OF) == "bull"


def test_regime_bear_close_below_sma(pg_legacy_session):
    from rainier.llm_thesis.research import compute_market_regime

    # Falling closes 300..101: latest 101 < SMA(200) = 200.5 → bear.
    _seed_spy(pg_legacy_session, range(300, 100, -1))
    assert compute_market_regime(as_of=AS_OF) == "bear"


def test_regime_boundary_close_equal_sma_is_bear(pg_legacy_session):
    from rainier.llm_thesis.research import compute_market_regime

    # Boundary: 200 identical closes → latest == SMA exactly. Bull requires
    # close STRICTLY above the SMA, so equality classifies as bear.
    _seed_spy(pg_legacy_session, [100.0] * 200)
    assert compute_market_regime(as_of=AS_OF) == "bear"


def test_regime_unknown_under_200_bars(pg_legacy_session):
    from rainier.llm_thesis.research import compute_market_regime

    # 199 bars: NEVER a partial-window SMA — unknown.
    _seed_spy(pg_legacy_session, range(100, 299))
    assert compute_market_regime(as_of=AS_OF) == "unknown"


def test_regime_unknown_no_spy_rows(pg_legacy_session):
    from rainier.llm_thesis.research import compute_market_regime

    assert compute_market_regime(as_of=AS_OF) == "unknown"


def test_regime_ignores_bars_after_as_of(pg_legacy_session):
    from rainier.llm_thesis.research import compute_market_regime

    # Bull series ending AS_OF, plus a post-as-of crash bar that must NOT
    # contaminate the as-of regime.
    _seed_spy(pg_legacy_session, range(100, 300))
    _seed_spy(pg_legacy_session, [1.0], end=AS_OF + timedelta(days=5))
    assert compute_market_regime(as_of=AS_OF) == "bull"


# ---------------------------------------------------------------------------
# lesson insight carries the tag (payload + rendered text)
# ---------------------------------------------------------------------------


def test_lesson_carries_regime_tag(pg_legacy_session):
    from rainier.llm_thesis.research import check_paper_lessons

    _seed_spy(pg_legacy_session, range(100, 300))
    _add_closed(pg_legacy_session, 1, "WIN", exit_date=date(2026, 3, 10),
                return_pct=0.10, pnl=1000.0, reason="target")

    out = check_paper_lessons(eval_date=AS_OF, days=30)
    assert len(out) == 1
    ins = out[0]
    assert ins.evidence["regime"] == "bull"
    assert "[regime: bull]" in ins.rationale


def test_lesson_regime_unknown_without_spy_history(pg_legacy_session):
    """Missing SPY rows → tag degrades to `unknown`, never a crash."""
    from rainier.llm_thesis.research import check_paper_lessons

    _add_closed(pg_legacy_session, 1, "LOS", exit_date=date(2026, 3, 12),
                return_pct=-0.05, pnl=-500.0, reason="stop_loss")

    out = check_paper_lessons(eval_date=AS_OF, days=30)
    assert len(out) == 1
    ins = out[0]
    assert ins.evidence["regime"] == "unknown"
    assert "[regime: unknown]" in ins.rationale


# ---------------------------------------------------------------------------
# ensure_spy_history — one-time ≥250-session backfill, then no-op
# ---------------------------------------------------------------------------


def test_ensure_spy_history_backfills_then_noop(pg_legacy_session):
    from sqlalchemy import func, select

    from rainier.paper.ingest import ensure_spy_history

    calls: list = []
    fetch_fn = _mk_fetch_fn(calls)

    res = ensure_spy_history(as_of=AS_OF, fetch_fn=fetch_fn)
    assert res["backfilled"] is True
    assert len(calls) == 1
    # The backfill window must feed a 200-SMA: ≥ 250 trading sessions.
    symbols, start, end = calls[0]
    assert symbols == ("SPY",)
    n_rows = pg_legacy_session.execute(
        select(func.count()).select_from(StockPrice).where(
            StockPrice.symbol == "SPY",
            StockPrice.date <= canonical_instant(AS_OF),
        )
    ).scalar_one()
    assert n_rows >= 250

    # Coverage now satisfied → second call is a pure no-op (no fetch).
    res2 = ensure_spy_history(as_of=AS_OF, fetch_fn=fetch_fn)
    assert res2["backfilled"] is False
    assert res2["upserted"] == 0
    assert len(calls) == 1


def test_ensure_spy_history_noop_with_existing_coverage(pg_legacy_session):
    from rainier.paper.ingest import ensure_spy_history

    _seed_spy(pg_legacy_session, range(100, 300))  # 200 usable bars
    calls: list = []
    res = ensure_spy_history(as_of=AS_OF, fetch_fn=_mk_fetch_fn(calls))
    assert res["backfilled"] is False
    assert calls == []


# ---------------------------------------------------------------------------
# scheduler wiring — daily ingest keeps SPY fresh; research job ensures history
# ---------------------------------------------------------------------------


def test_daily_ingest_includes_spy(monkeypatch):
    from rainier.paper import ingest as ingest_mod
    from rainier.paper import positions as positions_mod
    from rainier.scheduler import service

    captured: dict = {"symbols": []}

    monkeypatch.setattr(ingest_mod, "active_symbols", lambda: ["AAA"])
    monkeypatch.setattr(ingest_mod, "screened_symbols", lambda as_of: ["BBB"])
    # R-E split-ingest: trading ingest (SPY ∪ active ∪ screened) + a separate
    # cohort-extras ingest. Empty cohort here → only the trading call fires;
    # accumulate across calls so the assertion sees the SPY-bearing one.
    monkeypatch.setattr(
        ingest_mod, "get_current_qu100_cohort", lambda as_of: []
    )

    def fake_ingest(symbols, *, as_of, fetch_fn, **kw):
        captured["symbols"].extend(symbols)
        return {"upserted": 0, "changed": []}

    monkeypatch.setattr(ingest_mod, "ingest_prices", fake_ingest)
    monkeypatch.setattr(
        positions_mod, "fill_pending_positions", lambda **kw: {"filled": 0}
    )
    monkeypatch.setattr(
        positions_mod, "update_open_positions", lambda **kw: {"updated": 0}
    )
    # R-E feature step is failure-isolated in the caller; stub it so the test
    # exercises the ingest wiring without touching the feature DB.
    monkeypatch.setattr(
        "rainier.paper.features.run_daily_feature_step",
        lambda *a, **k: {"computed": 0, "recomputed": 0, "failed": 0},
    )
    monkeypatch.setattr(
        service,
        "load_settings_fresh",
        lambda: SimpleNamespace(
            llm_thesis=SimpleNamespace(learned_time_stop_days=None)
        ),
    )

    service.run_paper_daily_steps(AS_OF)
    assert "SPY" in captured["symbols"]


def test_research_job_ensures_spy_history_before_research(monkeypatch):
    from rainier.alerts import discord as discord_mod
    from rainier.llm_thesis import research as research_mod
    from rainier.paper import ingest as ingest_mod
    from rainier.scheduler import service

    events: list[str] = []

    monkeypatch.setattr(
        ingest_mod,
        "ensure_spy_history",
        lambda **kw: events.append("ensure") or {"upserted": 0, "backfilled": False},
    )
    monkeypatch.setattr(research_mod, "mark_stale", lambda days: 0)
    monkeypatch.setattr(
        research_mod,
        "run_research",
        lambda **kw: events.append("research") or [],
    )
    monkeypatch.setattr(
        discord_mod, "send_research_report", lambda **kw: events.append("report")
    )
    monkeypatch.setattr(
        service,
        "load_settings_fresh",
        lambda: SimpleNamespace(alerts=SimpleNamespace(discord=None)),
    )

    asyncio.run(service.run_research_job(eval_date_iso=AS_OF.isoformat()))
    # SPY coverage must be ensured BEFORE the checks compute the regime.
    assert events == ["ensure", "research", "report"]


def test_research_job_survives_spy_ensure_failure(monkeypatch):
    """yfinance down on Friday → ensure_spy_history raises → the research job
    logs and continues (lessons degrade to regime=unknown, never a crash)."""
    from rainier.alerts import discord as discord_mod
    from rainier.llm_thesis import research as research_mod
    from rainier.paper import ingest as ingest_mod
    from rainier.scheduler import service

    events: list[str] = []

    def boom(**kw):
        raise RuntimeError("yfinance unreachable")

    monkeypatch.setattr(ingest_mod, "ensure_spy_history", boom)
    monkeypatch.setattr(research_mod, "mark_stale", lambda days: 0)
    monkeypatch.setattr(
        research_mod,
        "run_research",
        lambda **kw: events.append("research") or [],
    )
    monkeypatch.setattr(
        discord_mod, "send_research_report", lambda **kw: events.append("report")
    )
    monkeypatch.setattr(
        service,
        "load_settings_fresh",
        lambda: SimpleNamespace(alerts=SimpleNamespace(discord=None)),
    )

    asyncio.run(service.run_research_job(eval_date_iso=AS_OF.isoformat()))
    assert events == ["research", "report"]
