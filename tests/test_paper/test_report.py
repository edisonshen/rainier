"""Step 8 — daily report snapshot + render paths (TEST-SPEC §H)."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select, text

from rainier.core.models import PaperReportSnapshot, PaperTrade, StockPrice
from rainier.paper.ingest import canonical_instant
from rainier.paper.report import (
    compute_daily_payload,
    load_snapshot,
    persist_daily_snapshot,
    render_payload,
)

pytestmark = pytest.mark.requires_postgres

AS_OF = date(2026, 1, 9)


def _seed_thesis(session, tid):
    session.execute(
        text("INSERT INTO analysis_results (id, llm_model, prompt_template) "
             "VALUES (:i,'m','t') ON CONFLICT DO NOTHING"),
        {"i": tid},
    )


def _add(session, tid, symbol, status, **kw):
    _seed_thesis(session, tid)
    defaults = dict(
        thesis_id=tid, symbol=symbol, scan_date=date(2026, 1, 5),
        session_name="close", status=status, planned_entry_price=100.0,
        stop_loss=90.0, target_price=120.0,
    )
    defaults.update(kw)
    session.add(PaperTrade(**defaults))
    session.commit()


def _price(session, symbol, d, o, h, low, c):
    session.execute(
        text("INSERT INTO stocks (symbol) VALUES (:s) ON CONFLICT DO NOTHING"),
        {"s": symbol},
    )
    session.add(StockPrice(symbol=symbol, date=canonical_instant(d), open=o,
                           high=h, low=low, close=c, volume=1000))
    session.commit()


def test_h1_daily_payload_pinned_fields(pg_legacy_session):
    s = pg_legacy_session
    # closed winner (target), closed loser (stop), open, pending, expired.
    _add(s, 1, "WIN", "closed", entry_date=date(2026, 1, 6), entry_price=100,
         shares=100, exit_date=AS_OF, exit_price=120, exit_reason="target",
         return_pct=0.20, pnl=2000.0, residual_cash=0.0)
    _add(s, 2, "LOS", "closed", entry_date=date(2026, 1, 6), entry_price=100,
         shares=100, exit_date=date(2026, 1, 8), exit_price=90,
         exit_reason="stop_loss", return_pct=-0.10, pnl=-1000.0, residual_cash=0.0)
    _add(s, 3, "OPN", "open", entry_date=date(2026, 1, 6), entry_price=100,
         shares=100, residual_cash=5.0)
    _add(s, 4, "PEN", "pending")
    _add(s, 5, "EXP", "expired")
    _price(s, "OPN", AS_OF, 110, 112, 108, 110)  # MTM close = 110

    payload = compute_daily_payload(AS_OF)
    assert payload["counts_by_status"] == {
        "pending": 1, "open": 1, "closed": 2, "expired": 1
    }
    assert payload["realized_pnl"] == pytest.approx(1000.0)  # 2000 - 1000
    # MTM incl open: realized 1000 + open (110-100)*100 = +1000 → 2000.
    assert payload["mtm_including_open_pnl"] == pytest.approx(2000.0)
    assert payload["open_mtm_pnl"] == pytest.approx(1000.0)
    # today's exits keyed by exit_date == as_of (only WIN).
    assert [e["symbol"] for e in payload["todays_exits"]] == ["WIN"]
    # win-rate over closed only: 1 win / 2 closed.
    assert payload["win_rate_closed"] == pytest.approx(0.5)
    assert payload["closed_trades"] == 2
    assert payload["total_residual_cash"] == pytest.approx(5.0)
    assert "same_bar_ambiguous_exits" in payload


def test_h1b_mtm_when_as_of_close_absent(pg_legacy_session):
    s = pg_legacy_session
    _add(s, 1, "OPN", "open", entry_date=date(2026, 1, 6), entry_price=100,
         shares=100, residual_cash=0.0)
    # No bar on as_of; last available close is 1/8 = 115.
    _price(s, "OPN", date(2026, 1, 8), 114, 116, 113, 115)
    payload = compute_daily_payload(AS_OF)
    assert payload["open_mtm_pnl"] == pytest.approx(100 * (115 - 100))
    assert "OPN" in payload["stale_mtm_symbols"]


def test_h1c_regenerate_is_point_in_time_no_lookahead(pg_legacy_session):
    """codex iter-1/2/3 regression — a historical --regenerate must NOT leak
    trades opened or closed AFTER as_of. A position closed on 1/12 (after
    AS_OF=1/9) reads as OPEN at 1/9 (MTM, not realized); a position created on
    1/12 doesn't exist at 1/9 at all."""
    s = pg_legacy_session
    # Closed-in-the-future: entered 1/6, exit booked 1/12 (after as_of 1/9).
    _add(s, 1, "FUT", "closed", entry_date=date(2026, 1, 6), entry_price=100,
         shares=100, exit_date=date(2026, 1, 12), exit_price=130,
         exit_reason="target", return_pct=0.30, pnl=3000.0, residual_cash=0.0)
    # Created-in-the-future: scanned 1/12, didn't exist at 1/9.
    _add(s, 2, "NEW", "pending", scan_date=date(2026, 1, 12))
    _price(s, "FUT", AS_OF, 108, 109, 107, 108)  # as-of close for MTM

    payload = compute_daily_payload(AS_OF)
    # FUT is OPEN at as_of (closed only on 1/12); NEW doesn't exist yet.
    assert payload["counts_by_status"] == {
        "pending": 0, "open": 1, "closed": 0, "expired": 0
    }
    # Its 3000 realized P&L must NOT leak — realized is 0, MTM reflects the open.
    assert payload["realized_pnl"] == pytest.approx(0.0)
    assert payload["open_mtm_pnl"] == pytest.approx(100 * (108 - 100))
    assert payload["closed_trades"] == 0
    assert payload["win_rate_closed"] is None
    assert payload["todays_exits"] == []


def test_h1d_expired_row_reads_pending_before_its_expiry_session(pg_legacy_session):
    """codex iter-5 regression — a never-filled row that LATER expired must read
    as PENDING on a replay date before its derived expiry session, not expired.
    scan=Mon 1/5 → T+1=1/6, deadline=1/7, expires 1/8. A replay at 1/6 (Tue) =>
    still pending; a replay at 1/8 => expired."""
    s = pg_legacy_session
    _add(s, 1, "EXP", "expired", scan_date=date(2026, 1, 5))  # never filled
    tue = compute_daily_payload(date(2026, 1, 6))
    assert tue["counts_by_status"] == {
        "pending": 1, "open": 0, "closed": 0, "expired": 0
    }
    thu = compute_daily_payload(date(2026, 1, 8))
    assert thu["counts_by_status"] == {
        "pending": 0, "open": 0, "closed": 0, "expired": 1
    }


def test_h2_idempotent_upsert(pg_legacy_session):
    s = pg_legacy_session
    p = compute_daily_payload(AS_OF)
    persist_daily_snapshot(AS_OF, p)
    persist_daily_snapshot(AS_OF, p)
    s.expire_all()
    rows = s.execute(
        select(PaperReportSnapshot).where(PaperReportSnapshot.as_of_date == AS_OF)
    ).scalars().all()
    assert len(rows) == 1


def test_h3_plain_render_from_snapshot_not_mutated_data(pg_legacy_session):
    s = pg_legacy_session
    _add(s, 1, "WIN", "closed", entry_date=date(2026, 1, 6), entry_price=100,
         shares=100, exit_date=AS_OF, exit_price=120, exit_reason="target",
         return_pct=0.20, pnl=2000.0, residual_cash=0.0)
    payload = compute_daily_payload(AS_OF)
    persist_daily_snapshot(AS_OF, payload)
    snap_realized = payload["realized_pnl"]
    # Mutate underlying paper_trade AFTER the snapshot.
    s.execute(text("UPDATE paper_trade SET pnl = 99999 WHERE symbol='WIN'"))
    s.commit()
    # Plain re-render reads the snapshot, NOT the mutated row.
    loaded = load_snapshot("daily", AS_OF)
    assert loaded["realized_pnl"] == snap_realized
    txt = render_payload(loaded)
    assert "99,999" not in txt and "99999" not in txt


def test_h4_regenerate_upserts_and_plain_reads_regenerated(pg_legacy_session):
    s = pg_legacy_session
    _add(s, 1, "WIN", "closed", entry_date=date(2026, 1, 6), entry_price=100,
         shares=100, exit_date=AS_OF, exit_price=110, exit_reason="target",
         return_pct=0.10, pnl=1000.0, residual_cash=0.0)
    persist_daily_snapshot(AS_OF, compute_daily_payload(AS_OF))
    # Underlying changes; regenerate recomputes + upserts.
    s.execute(text("UPDATE paper_trade SET pnl = 2500, exit_price = 125 WHERE symbol='WIN'"))
    s.commit()
    regenerated = compute_daily_payload(AS_OF)
    persist_daily_snapshot(AS_OF, regenerated)
    s.expire_all()
    rows = s.execute(
        select(PaperReportSnapshot).where(PaperReportSnapshot.as_of_date == AS_OF)
    ).scalars().all()
    assert len(rows) == 1  # still one row (upsert, no dup)
    # plain read returns the regenerated payload.
    loaded = load_snapshot("daily", AS_OF)
    assert loaded["realized_pnl"] == pytest.approx(2500.0)


def test_h6_discord_non_fatal_no_webhook(pg_legacy_session):
    from rainier.paper.report import send_daily_paper_report

    payload = compute_daily_payload(AS_OF)

    class _Cfg:
        enabled = True
        webhook_url = ""

    # No webhook configured → returns False, does not raise.
    assert send_daily_paper_report(payload, _Cfg()) is False
    assert send_daily_paper_report(payload, None) is False
