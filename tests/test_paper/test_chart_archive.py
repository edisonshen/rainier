"""R-D chart archive tests (task qu100-chart-archive-77f3).

Covers (TASK-PLAN §Validation):

* figure assembly — 120-bar window (not the 60-bar thesis tail), 90-bar config
  respected, appearance markers ("#rank · M/D" labels), trade overlay
  (entry/SL/target lines + entry/exit point annotations);
* byte-determinism of the new renderer (same kaleido pinning + metadata scrub
  as the thesis path — IMPORTED from chart_export, never edited);
* `get_qu100_appearances` per-day dedup (latest captured_at within
  ranking_type='top100') + `--all-sessions` + as-of/window cuts;
* append-only supersession (sha dedup no-op; 3-step single-transaction
  staging; bytes never overwritten; flip-flop renders never violate the 0004
  partial-unique because archive rows keep `scan_date` NULL);
* thesis rows stay OUTSIDE the logical-identity contract (`as_of_date` NULL
  for source='thesis', existing + future);
* migration 0010 up/backfill/idempotency/downgrade;
* close-side capture (`capture_trade_close_charts`) incl. chart_id move on
  supersession;
* render-on-demand byte-identical reproduction;
* CLI registration + threading.

Figure tests are pure (no DB, no kaleido). Render tests skip without kaleido.
DB tests ride the `pg_legacy_engine` (0010 applied) / `pg_chart_mig_engine`
(pre-0010) postgres lane.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from click.testing import CliRunner
from sqlalchemy import select, text

from rainier.paper import chart_archive
from rainier.paper.chart_archive import (
    SOURCE_MISS_SWEEP,
    SOURCE_TRADE_CLOSE,
    TradeOverlay,
    build_archive_figure,
    capture_trade_close_charts,
    load_daily_bars,
    persist_archive_chart,
    regenerate_chart,
    render_archive_chart_png,
)
from rainier.paper.ingest import canonical_instant, get_qu100_appearances

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

AS_OF = date(2026, 6, 5)  # a Friday


def _mk_df(n: int = 200, end: date = AS_OF) -> pd.DataFrame:
    """Synthetic ascending daily OHLCV over the last `n` business days."""
    idx = pd.bdate_range(end=end, periods=n)
    base = [100.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "open": base,
            "high": [b + 1.0 for b in base],
            "low": [b - 1.0 for b in base],
            "close": [b + 0.5 for b in base],
            "volume": [1_000_000] * n,
        },
        index=idx,
    )


def _tail_dates(df: pd.DataFrame, window: int) -> list[date]:
    return [ts.date() for ts in df.tail(window).index]


def _vertical_shapes(fig) -> list:
    return [s for s in fig.layout.shapes if s.x0 == s.x1]


def _horizontal_shapes(fig) -> list:
    return [s for s in fig.layout.shapes if s.y0 == s.y1]


def _annotation_texts(fig) -> list[str]:
    return [a.text for a in fig.layout.annotations]


def _appearance(d: date, rank: int):
    """Minimal (data_date, rank) appearance record the figure builder accepts."""
    return (d, rank)


# ---------------------------------------------------------------------------
# Figure assembly (pure — no DB, no kaleido)
# ---------------------------------------------------------------------------


def test_figure_window_default_120():
    """120-bar window honored — NOT the thesis path's 60-bar tail."""
    fig = build_archive_figure("AAPL", _mk_df(200))
    assert len(fig.data[0].open) == 120


def test_figure_window_90():
    """A 90-day configured window is honored end-to-end in the figure."""
    fig = build_archive_figure("AAPL", _mk_df(200), window=90)
    assert len(fig.data[0].open) == 90


def test_appearance_markers_drawn_with_rank_date_labels():
    df = _mk_df(200)
    dates = _tail_dates(df, 120)
    d1, d2 = dates[10], dates[50]
    fig = build_archive_figure(
        "AAPL", df, appearances=[_appearance(d1, 23), _appearance(d2, 5)]
    )
    verts = _vertical_shapes(fig)
    assert len(verts) == 2
    assert sorted(v.x0 for v in verts) == [10, 50]
    texts = _annotation_texts(fig)
    assert f"#23 · {d1.month}/{d1.day}" in texts
    assert f"#5 · {d2.month}/{d2.day}" in texts


def test_appearance_multiple_markers_same_symbol():
    """Multi-appearance symbol → one marker per appearance date."""
    df = _mk_df(200)
    dates = _tail_dates(df, 120)
    apps = [_appearance(dates[i], i + 1) for i in (5, 30, 60, 100)]
    fig = build_archive_figure("NVDA", df, appearances=apps)
    assert len(_vertical_shapes(fig)) == 4


def test_appearance_outside_window_or_missing_bar_skipped():
    df = _mk_df(200)
    visible = _tail_dates(df, 120)
    older = _tail_dates(df, 200)[0]  # oldest bar — outside the 120 window
    assert older not in visible
    saturday = AS_OF + timedelta(days=1)  # no bar on a weekend
    fig = build_archive_figure(
        "AAPL",
        df,
        appearances=[_appearance(older, 7), _appearance(saturday, 9)],
    )
    assert len(_vertical_shapes(fig)) == 0


def test_trade_overlay_levels_and_markers():
    df = _mk_df(200)
    dates = _tail_dates(df, 120)
    entry_d, exit_d = dates[80], dates[110]
    trade = TradeOverlay(
        entry_date=entry_d,
        entry_price=105.0,
        stop_loss=101.5,
        target_price=112.25,
        exit_date=exit_d,
        exit_price=112.3,
        exit_reason="target",
    )
    fig = build_archive_figure("AAPL", df, trade=trade)
    level_prices = {s.y0 for s in _horizontal_shapes(fig)}
    assert {105.0, 101.5, 112.25} <= level_prices
    # Entry/exit point annotations at the correct (date-pos, price).
    points = {
        (a.x, a.y): a.text
        for a in fig.layout.annotations
        if a.xref == "x" and a.yref == "y" and a.showarrow
    }
    assert (80, 105.0) in points and "105.00" in points[(80, 105.0)]
    assert (110, 112.3) in points and "target" in points[(110, 112.3)]


def test_trade_overlay_without_exit_still_renders():
    """Defensive: overlay with no exit booked draws levels, no exit marker."""
    df = _mk_df(200)
    trade = TradeOverlay(
        entry_date=None,
        entry_price=105.0,
        stop_loss=101.5,
        target_price=112.25,
        exit_date=None,
        exit_price=None,
        exit_reason=None,
    )
    fig = build_archive_figure("AAPL", df, trade=trade)
    assert {105.0, 101.5, 112.25} <= {s.y0 for s in _horizontal_shapes(fig)}
    assert not [a for a in fig.layout.annotations if a.showarrow]


def test_config_chart_lookback_days_default():
    from rainier.core.config import LLMThesisConfig

    assert LLMThesisConfig().chart_lookback_days == 120


# ---------------------------------------------------------------------------
# Renderer determinism (kaleido)
# ---------------------------------------------------------------------------


def test_render_archive_chart_png_deterministic():
    pytest.importorskip("kaleido")
    df = _mk_df(200)
    dates = _tail_dates(df, 120)
    apps = [_appearance(dates[10], 23)]
    trade = TradeOverlay(
        entry_date=dates[80],
        entry_price=105.0,
        stop_loss=101.5,
        target_price=112.25,
        exit_date=dates[110],
        exit_price=112.3,
        exit_reason="target",
    )
    png1, sha1 = render_archive_chart_png("AAPL", df, appearances=apps, trade=trade)
    png2, sha2 = render_archive_chart_png("AAPL", df, appearances=apps, trade=trade)
    assert png1 == png2
    assert sha1 == sha2 == hashlib.sha256(png1).hexdigest()
    assert png1.startswith(b"\x89PNG\r\n\x1a\n")


# ---------------------------------------------------------------------------
# Appearance query (postgres lane)
# ---------------------------------------------------------------------------


def _seed_stock(session, symbol: str) -> None:
    session.execute(
        text(
            "INSERT INTO stocks (symbol) VALUES (:s) ON CONFLICT (symbol) DO NOTHING"
        ),
        {"s": symbol},
    )


def _seed_snapshot(
    session,
    symbol: str,
    data_date: date,
    rank: int,
    captured_at: datetime,
    session_name: str = "close",
    ranking_type: str = "top100",
) -> None:
    session.execute(
        text(
            "INSERT INTO money_flow_snapshots "
            "(captured_at, capture_session, data_date, ranking_type, symbol, rank) "
            "VALUES (:cap, :ses, :dd, :rt, :sym, :rk)"
        ),
        {
            "cap": captured_at,
            "ses": session_name,
            "dd": data_date,
            "rt": ranking_type,
            "sym": symbol,
            "rk": rank,
        },
    )


def _cap(d: date, hour: int) -> datetime:
    return datetime(d.year, d.month, d.day, hour, 0, tzinfo=timezone.utc)


def test_appearances_per_day_dedup_latest_capture(pg_legacy_session):
    """4 sessions/day → ONE (date, rank): the day's latest captured_at batch."""
    s = pg_legacy_session
    _seed_stock(s, "AAPL")
    d = date(2026, 6, 3)
    for hour, rank, ses in [
        (14, 40, "morning"), (16, 30, "midday"), (19, 20, "afternoon"), (21, 10, "close"),
    ]:
        _seed_snapshot(s, "AAPL", d, rank, _cap(d, hour), session_name=ses)
    s.commit()

    apps = get_qu100_appearances("AAPL", as_of=AS_OF, window=10)
    assert [(a.data_date, a.rank) for a in apps] == [(d, 10)]


def test_appearances_all_sessions_exposes_intraday(pg_legacy_session):
    s = pg_legacy_session
    _seed_stock(s, "AAPL")
    d = date(2026, 6, 3)
    for hour, rank in [(14, 40), (21, 10)]:
        _seed_snapshot(s, "AAPL", d, rank, _cap(d, hour))
    s.commit()

    apps = get_qu100_appearances("AAPL", as_of=AS_OF, window=10, all_sessions=True)
    assert [(a.data_date, a.rank) for a in apps] == [(d, 40), (d, 10)]


def test_appearances_as_of_and_window_cuts(pg_legacy_session):
    s = pg_legacy_session
    _seed_stock(s, "AAPL")
    inside = date(2026, 6, 1)
    after = date(2026, 6, 8)  # data_date > as_of — excluded
    way_back = date(2025, 6, 5)  # far outside any reasonable trailing window
    for d, rank in [(inside, 5), (after, 6), (way_back, 7)]:
        _seed_snapshot(s, "AAPL", d, rank, _cap(d, 21))
    s.commit()

    apps = get_qu100_appearances("AAPL", as_of=AS_OF, window=10)
    assert [(a.data_date, a.rank) for a in apps] == [(inside, 5)]


def test_appearances_dedup_is_cohort_consistent(pg_legacy_session):
    """A symbol present in an early batch but absent from the day's LATEST
    top100 batch is NOT an appearance (same rule as the day's cohort);
    --all-sessions still shows the intraday row. Non-top100 rows never count."""
    s = pg_legacy_session
    for sym in ("AAPL", "MSFT"):
        _seed_stock(s, sym)
    d = date(2026, 6, 3)
    _seed_snapshot(s, "AAPL", d, 90, _cap(d, 14))  # morning batch only
    _seed_snapshot(s, "MSFT", d, 90, _cap(d, 21))  # the day's latest batch
    _seed_snapshot(s, "AAPL", d, 1, _cap(d, 22), ranking_type="bottom100")
    s.commit()

    assert get_qu100_appearances("AAPL", as_of=AS_OF, window=10) == []
    apps_all = get_qu100_appearances("AAPL", as_of=AS_OF, window=10, all_sessions=True)
    assert [(a.data_date, a.rank) for a in apps_all] == [(d, 90)]


# ---------------------------------------------------------------------------
# Append-only supersession (postgres lane, 0010 applied)
# ---------------------------------------------------------------------------


def _chart_rows(session, symbol: str):
    from rainier.core.models import ChartImage

    # The production code writes through its own sessions; expire this
    # session's identity map so the re-read sees committed DB state, not
    # cached instances.
    session.expire_all()
    return (
        session.execute(
            select(ChartImage).where(ChartImage.symbol == symbol).order_by(ChartImage.id)
        )
        .scalars()
        .all()
    )


def test_persist_same_bytes_is_dedup_noop(pg_legacy_session):
    _seed_stock(pg_legacy_session, "AAPL")
    pg_legacy_session.commit()
    png = b"\x89PNG\r\n\x1a\nfakebytes-a"
    sha = hashlib.sha256(png).hexdigest()
    id1 = persist_archive_chart(
        symbol="AAPL", as_of=AS_OF, source=SOURCE_TRADE_CLOSE,
        png_bytes=png, sha256=sha, timeframe_days=120,
    )
    id2 = persist_archive_chart(
        symbol="AAPL", as_of=AS_OF, source=SOURCE_TRADE_CLOSE,
        png_bytes=png, sha256=sha, timeframe_days=120,
    )
    assert id1 == id2
    rows = _chart_rows(pg_legacy_session, "AAPL")
    assert len(rows) == 1
    assert rows[0].source == SOURCE_TRADE_CLOSE
    assert rows[0].as_of_date == AS_OF
    assert rows[0].superseded_by is None
    assert rows[0].scan_date is None  # archive rows live on as_of_date, not scan_date


def test_persist_changed_bytes_supersedes_append_only(pg_legacy_session):
    _seed_stock(pg_legacy_session, "AAPL")
    pg_legacy_session.commit()
    png_a, png_b = b"\x89PNG\r\n\x1a\nbytes-a", b"\x89PNG\r\n\x1a\nbytes-b"
    sha_a = hashlib.sha256(png_a).hexdigest()
    sha_b = hashlib.sha256(png_b).hexdigest()
    id_a = persist_archive_chart(
        symbol="AAPL", as_of=AS_OF, source=SOURCE_MISS_SWEEP,
        png_bytes=png_a, sha256=sha_a, timeframe_days=120,
    )
    id_b = persist_archive_chart(
        symbol="AAPL", as_of=AS_OF, source=SOURCE_MISS_SWEEP,
        png_bytes=png_b, sha256=sha_b, timeframe_days=120,
    )
    assert id_b != id_a
    rows = {r.id: r for r in _chart_rows(pg_legacy_session, "AAPL")}
    old, new = rows[id_a], rows[id_b]
    # Old row: superseded, bytes UNCHANGED (append-only; snapshot payloads
    # holding id_a stay valid forever).
    assert old.superseded_by == id_b
    assert bytes(old.image_bytes) == png_a
    # New row: promoted current.
    assert new.superseded_by is None
    assert new.source == SOURCE_MISS_SWEEP
    assert new.as_of_date == AS_OF


def test_persist_flip_flop_render_no_unique_violation(pg_legacy_session):
    """A→B→A sha sequence: the third render re-produces sha_a while a
    superseded row with sha_a already exists. Must not trip either unique
    index (0004's (symbol, scan_date, sha) — archive rows keep scan_date
    NULL — or 0010's logical partial-unique)."""
    _seed_stock(pg_legacy_session, "AAPL")
    pg_legacy_session.commit()
    png_a, png_b = b"\x89PNG\r\n\x1a\nflip-a", b"\x89PNG\r\n\x1a\nflip-b"
    sha_a = hashlib.sha256(png_a).hexdigest()
    sha_b = hashlib.sha256(png_b).hexdigest()
    kw = dict(symbol="AAPL", as_of=AS_OF, source=SOURCE_TRADE_CLOSE, timeframe_days=120)
    id1 = persist_archive_chart(png_bytes=png_a, sha256=sha_a, **kw)
    id2 = persist_archive_chart(png_bytes=png_b, sha256=sha_b, **kw)
    id3 = persist_archive_chart(png_bytes=png_a, sha256=sha_a, **kw)
    assert len({id1, id2, id3}) == 3
    rows = {r.id: r for r in _chart_rows(pg_legacy_session, "AAPL")}
    assert rows[id1].superseded_by == id2
    assert rows[id2].superseded_by == id3
    assert rows[id3].superseded_by is None


def test_persist_race_partial_unique_blocks_second_promote(
    pg_legacy_engine, pg_legacy_session
):
    """Two concurrent-ish persists for ONE logical identity, interleaved on two
    raw connections (the worst-case race the single-threaded daily job never
    produces): staging (NULL key columns) never trips the partial unique while
    both transactions are open, and the loser's PROMOTE is rejected by
    ``idx_chart_images_logical`` — the whole losing transaction rolls back, so
    'one CURRENT row per (symbol, as_of_date, source)' holds at the DB level
    and the old row's ``superseded_by`` ends up pointing at the winner only."""
    from sqlalchemy.exc import IntegrityError

    _seed_stock(pg_legacy_session, "AAPL")
    pg_legacy_session.commit()
    png0 = b"\x89PNG\r\n\x1a\nrace-0"
    cur_id = persist_archive_chart(
        symbol="AAPL", as_of=AS_OF, source=SOURCE_TRADE_CLOSE,
        png_bytes=png0, sha256=hashlib.sha256(png0).hexdigest(), timeframe_days=120,
    )

    ins = text(
        "INSERT INTO chart_images (symbol, scan_date, source, as_of_date, "
        "image_bytes, sha256) VALUES ('AAPL', NULL, NULL, NULL, :png, :sha) "
        "RETURNING id"
    )
    supersede = text("UPDATE chart_images SET superseded_by = :n WHERE id = :o")
    promote = text(
        "UPDATE chart_images SET as_of_date = :d, source = :s WHERE id = :n"
    )
    conn1 = pg_legacy_engine.connect()
    conn2 = pg_legacy_engine.connect()
    try:
        tx1 = conn1.begin()
        new1 = conn1.execute(
            ins, {"png": b"race-1", "sha": hashlib.sha256(b"race-1").hexdigest()}
        ).scalar_one()
        # Interleave: conn2 stages while conn1's transaction is still OPEN —
        # two staged rows + the current row coexist without touching the
        # partial unique (mid-transaction safety of step 1).
        tx2 = conn2.begin()
        new2 = conn2.execute(
            ins, {"png": b"race-2", "sha": hashlib.sha256(b"race-2").hexdigest()}
        ).scalar_one()

        # conn1 wins: supersede + promote + commit (steps 2-3).
        conn1.execute(supersede, {"n": new1, "o": cur_id})
        conn1.execute(promote, {"d": AS_OF, "s": SOURCE_TRADE_CLOSE, "n": new1})
        tx1.commit()

        # conn2 loses: its promote would create a SECOND current row for the
        # identity — idx_chart_images_logical rejects it.
        conn2.execute(supersede, {"n": new2, "o": cur_id})
        with pytest.raises(IntegrityError, match="idx_chart_images_logical"):
            conn2.execute(
                promote, {"d": AS_OF, "s": SOURCE_TRADE_CLOSE, "n": new2}
            )
        tx2.rollback()
    finally:
        conn1.close()
        conn2.close()

    rows = {r.id: r for r in _chart_rows(pg_legacy_session, "AAPL")}
    # Loser's transaction rolled back WHOLE: its staged row and its
    # superseded_by overwrite are both gone; the winner is the only current.
    assert new2 not in rows
    assert rows[cur_id].superseded_by == new1
    current = [
        r for r in rows.values()
        if r.superseded_by is None and r.source == SOURCE_TRADE_CLOSE
    ]
    assert [r.id for r in current] == [new1]


def test_persist_timeframe_days_follows_window(pg_legacy_session):
    _seed_stock(pg_legacy_session, "AAPL")
    pg_legacy_session.commit()
    png = b"\x89PNG\r\n\x1a\nninety"
    cid = persist_archive_chart(
        symbol="AAPL", as_of=AS_OF, source=SOURCE_TRADE_CLOSE,
        png_bytes=png, sha256=hashlib.sha256(png).hexdigest(), timeframe_days=90,
    )
    row = {r.id: r for r in _chart_rows(pg_legacy_session, "AAPL")}[cid]
    assert row.timeframe_days == 90


def test_persist_rejects_thesis_source():
    """Guard: thesis rows go through llm_thesis.chart_persistence, never the
    archive persist path — a 'thesis' (or unknown) source raises before any DB
    work, so the logical-identity contract can't be polluted."""
    png = b"\x89PNG\r\n\x1a\nguard"
    for bad in ("thesis", "bogus"):
        with pytest.raises(ValueError, match="trade_close/miss_sweep"):
            persist_archive_chart(
                symbol="AAPL", as_of=AS_OF, source=bad,
                png_bytes=png, sha256=hashlib.sha256(png).hexdigest(),
                timeframe_days=120,
            )


def test_thesis_rows_stay_outside_logical_contract(pg_legacy_session):
    """Future thesis persists: source='thesis', as_of_date NULL — two same-day
    same-symbol thesis charts coexist (NULLs never collide in the partial
    unique), and the up-to-4-renders/day thesis pattern is unaffected."""
    from rainier.llm_thesis.chart_persistence import persist_chart_image

    _seed_stock(pg_legacy_session, "AAPL")
    pg_legacy_session.commit()
    d = date(2026, 6, 3)
    png_a, png_b = b"\x89PNG\r\n\x1a\nthesis-a", b"\x89PNG\r\n\x1a\nthesis-b"
    id_a = persist_chart_image(
        symbol="AAPL", scan_date=d, image_bytes=png_a,
        sha256=hashlib.sha256(png_a).hexdigest(),
    )
    id_b = persist_chart_image(
        symbol="AAPL", scan_date=d, image_bytes=png_b,
        sha256=hashlib.sha256(png_b).hexdigest(), timeframe_days=90,
    )
    assert id_a is not None and id_b is not None and id_a != id_b
    rows = {r.id: r for r in _chart_rows(pg_legacy_session, "AAPL")}
    for cid in (id_a, id_b):
        assert rows[cid].source == "thesis"
        assert rows[cid].as_of_date is None
        assert rows[cid].superseded_by is None
    # timeframe_days follows the caller's window (90 here), default stays 120.
    assert rows[id_a].timeframe_days == 120
    assert rows[id_b].timeframe_days == 90


# ---------------------------------------------------------------------------
# Close-side capture (postgres lane; renderer faked — figure correctness is
# covered by the pure tests above)
# ---------------------------------------------------------------------------


def _seed_closed_trade(session, symbol: str, exit_d: date) -> int:
    thesis_id = session.execute(
        text("INSERT INTO analysis_results (llm_model, prompt_template) "
             "VALUES ('test', 'test') RETURNING id")
    ).scalar_one()
    trade_id = session.execute(
        text(
            "INSERT INTO paper_trade (thesis_id, symbol, scan_date, session_name, "
            "status, planned_entry_price, stop_loss, target_price, entry_date, "
            "entry_price, shares, exit_date, exit_price, exit_reason, return_pct) "
            "VALUES (:tid, :sym, :sd, 'close', 'closed', 100.0, 95.0, 110.0, :ed, "
            "101.0, 10, :xd, 110.5, 'target', 9.4) RETURNING id"
        ),
        {"tid": thesis_id, "sym": symbol, "sd": exit_d - timedelta(days=7),
         "ed": exit_d - timedelta(days=5), "xd": exit_d},
    ).scalar_one()
    session.commit()
    return trade_id


def _fake_renderer(png: bytes):
    def _render(symbol, df, *, window=120, appearances=(), trade=None, **kw):
        return png, hashlib.sha256(png).hexdigest()

    return _render


def test_capture_trade_close_charts_sets_chart_id(pg_legacy_session, monkeypatch):
    _seed_stock(pg_legacy_session, "AAPL")
    pg_legacy_session.commit()
    trade_id = _seed_closed_trade(pg_legacy_session, "AAPL", AS_OF)
    monkeypatch.setattr(
        chart_archive, "load_daily_bars", lambda symbol, *, as_of, window: _mk_df(130)
    )
    monkeypatch.setattr(
        chart_archive, "render_archive_chart_png",
        _fake_renderer(b"\x89PNG\r\n\x1a\ncapture-1"),
    )

    n = capture_trade_close_charts(as_of=AS_OF)
    assert n == 1
    from rainier.core.models import PaperTrade

    trade = pg_legacy_session.get(PaperTrade, trade_id)
    pg_legacy_session.refresh(trade)
    assert trade.chart_id is not None
    rows = _chart_rows(pg_legacy_session, "AAPL")
    assert len(rows) == 1
    assert rows[0].source == SOURCE_TRADE_CLOSE
    assert rows[0].as_of_date == AS_OF

    # Same-tick re-run → sha dedup, no new row, chart_id unchanged.
    first_chart_id = trade.chart_id
    assert capture_trade_close_charts(as_of=AS_OF) == 1
    pg_legacy_session.refresh(trade)
    assert trade.chart_id == first_chart_id
    assert len(_chart_rows(pg_legacy_session, "AAPL")) == 1

    # Changed inputs (new bytes) → supersession; chart_id MOVES to the new
    # current row; the snapshot-referenced old id still resolves.
    monkeypatch.setattr(
        chart_archive, "render_archive_chart_png",
        _fake_renderer(b"\x89PNG\r\n\x1a\ncapture-2"),
    )
    assert capture_trade_close_charts(as_of=AS_OF) == 1
    pg_legacy_session.refresh(trade)
    rows = {r.id: r for r in _chart_rows(pg_legacy_session, "AAPL")}
    assert len(rows) == 2
    assert trade.chart_id != first_chart_id
    assert rows[first_chart_id].superseded_by == trade.chart_id
    assert bytes(rows[first_chart_id].image_bytes) == b"\x89PNG\r\n\x1a\ncapture-1"


def test_capture_skips_symbol_with_no_bars(pg_legacy_session, monkeypatch):
    _seed_stock(pg_legacy_session, "NOPX")
    pg_legacy_session.commit()
    _seed_closed_trade(pg_legacy_session, "NOPX", AS_OF)
    monkeypatch.setattr(
        chart_archive, "load_daily_bars",
        lambda symbol, *, as_of, window: pd.DataFrame(),
    )
    assert capture_trade_close_charts(as_of=AS_OF) == 0
    assert _chart_rows(pg_legacy_session, "NOPX") == []


# ---------------------------------------------------------------------------
# load_daily_bars + render-on-demand byte-identical reproduction
# ---------------------------------------------------------------------------


def _seed_prices(session, symbol: str, n: int = 130, end: date = AS_OF) -> list[date]:
    days = [ts.date() for ts in pd.bdate_range(end=end, periods=n)]
    for i, d in enumerate(days):
        session.execute(
            text(
                "INSERT INTO stock_prices (symbol, date, open, high, low, close, volume) "
                "VALUES (:sym, :dt, :o, :h, :l, :c, 1000000)"
            ),
            {"sym": symbol, "dt": canonical_instant(d), "o": 100.0 + i * 0.1,
             "h": 101.0 + i * 0.1, "l": 99.0 + i * 0.1, "c": 100.5 + i * 0.1},
        )
    session.commit()
    return days


def test_load_daily_bars_window_and_as_of(pg_legacy_session):
    _seed_stock(pg_legacy_session, "AAPL")
    pg_legacy_session.commit()
    days = _seed_prices(pg_legacy_session, "AAPL", n=130)
    df = load_daily_bars("AAPL", as_of=AS_OF, window=120)
    assert len(df) == 120
    assert [ts.date() for ts in df.index] == days[-120:]
    # as_of strictly bounds the data — a re-render never sees future bars.
    earlier = days[-10]
    df2 = load_daily_bars("AAPL", as_of=earlier, window=120)
    assert [ts.date() for ts in df2.index][-1] == earlier


def test_render_on_demand_reproduces_stored_chart_byte_identically(
    pg_legacy_session, monkeypatch
):
    pytest.importorskip("kaleido")
    _seed_stock(pg_legacy_session, "AAPL")
    pg_legacy_session.commit()
    days = _seed_prices(pg_legacy_session, "AAPL", n=130)
    _seed_snapshot(
        pg_legacy_session, "AAPL", days[-20], 23, _cap(days[-20], 21)
    )
    pg_legacy_session.commit()
    trade_id = _seed_closed_trade(pg_legacy_session, "AAPL", AS_OF)

    n = capture_trade_close_charts(as_of=AS_OF)
    assert n == 1
    from rainier.core.models import PaperTrade

    trade = pg_legacy_session.get(PaperTrade, trade_id)
    pg_legacy_session.refresh(trade)
    stored = {r.id: r for r in _chart_rows(pg_legacy_session, "AAPL")}[trade.chart_id]

    png, sha = regenerate_chart("AAPL", as_of=AS_OF)
    assert sha == stored.sha256
    assert png == bytes(stored.image_bytes)


# ---------------------------------------------------------------------------
# Migration 0010 (pre-0010 fixture; the test applies/downgrades itself)
# ---------------------------------------------------------------------------

from .conftest import MIGRATION_0010_DOWN, MIGRATION_0010_UP, _apply_sql  # noqa: E402


def test_migration_0010_files_exist():
    """Pins the files so conftest's exists() guard can never silently skip."""
    assert MIGRATION_0010_UP.exists()
    assert MIGRATION_0010_DOWN.exists()


def _col_names(engine, table: str) -> set[str]:
    from sqlalchemy import inspect

    return {c["name"] for c in inspect(engine).get_columns(table, schema=engine.rainier_schema)}


def test_migration_0010_backfill_index_idempotency(pg_chart_mig_engine):
    eng = pg_chart_mig_engine
    with eng.begin() as conn:
        conn.execute(text("INSERT INTO stocks (symbol) VALUES ('AAPL')"))
        # Two same-day duplicate thesis charts (the live-DB shape that must
        # survive the migration + backfill without tripping the new index).
        for sfx in ("a", "b"):
            conn.execute(
                text(
                    "INSERT INTO chart_images (symbol, scan_date, image_bytes, sha256) "
                    "VALUES ('AAPL', '2026-06-03', :png, :sha)"
                ),
                {"png": f"png-{sfx}".encode(), "sha": hashlib.sha256(sfx.encode()).hexdigest()},
            )

    _apply_sql(eng, MIGRATION_0010_UP)

    cols = _col_names(eng, "chart_images")
    assert {"source", "as_of_date", "superseded_by"} <= cols
    assert "chart_id" in _col_names(eng, "paper_trade")
    with eng.connect() as conn:
        rows = conn.execute(
            text("SELECT source, as_of_date FROM chart_images ORDER BY id")
        ).all()
        assert rows == [("thesis", None), ("thesis", None)]
        idx = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes WHERE tablename='chart_images' "
                "AND schemaname=:sch"
            ),
            {"sch": eng.rainier_schema},
        ).scalars().all()
        assert "idx_chart_images_logical" in idx

    # Idempotent re-apply (and the backfill never clobbers non-NULL sources).
    with eng.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO chart_images (symbol, scan_date, image_bytes, sha256, "
                "source, as_of_date) VALUES ('AAPL', NULL, 'arc', :sha, "
                "'trade_close', '2026-06-05')"
            ),
            {"sha": hashlib.sha256(b"arc").hexdigest()},
        )
    _apply_sql(eng, MIGRATION_0010_UP)
    with eng.connect() as conn:
        src = conn.execute(
            text("SELECT source FROM chart_images ORDER BY id")
        ).scalars().all()
        assert src == ["thesis", "thesis", "trade_close"]


def test_migration_0010_downgrade_then_up(pg_chart_mig_engine):
    eng = pg_chart_mig_engine
    _apply_sql(eng, MIGRATION_0010_UP)
    _apply_sql(eng, MIGRATION_0010_DOWN)
    cols = _col_names(eng, "chart_images")
    assert not ({"source", "as_of_date", "superseded_by"} & cols)
    assert "chart_id" not in _col_names(eng, "paper_trade")
    # Up again from the downgraded state — round-trip safe.
    _apply_sql(eng, MIGRATION_0010_UP)
    assert {"source", "as_of_date", "superseded_by"} <= _col_names(eng, "chart_images")


# ---------------------------------------------------------------------------
# Daily-job wiring (close-side capture is unconditional, step (v))
# ---------------------------------------------------------------------------


def test_run_paper_close_charts_calls_capture(monkeypatch):
    from rainier.scheduler import service

    called: dict = {}

    def fake_capture(*, as_of):
        called["as_of"] = as_of
        return 2

    monkeypatch.setattr(
        "rainier.paper.chart_archive.capture_trade_close_charts", fake_capture
    )
    service.run_paper_close_charts(AS_OF)
    assert called["as_of"] == AS_OF


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_paper_lists_new_commands():
    from rainier.cli import cli

    result = CliRunner().invoke(cli, ["paper", "--help"])
    assert result.exit_code == 0
    assert "appearances" in result.output
    assert "chart" in result.output


def test_cli_paper_appearances_renders_rows(monkeypatch):
    from rainier.cli import cli
    from rainier.paper.ingest import QU100Appearance

    def fake_appearances(symbol, *, as_of, window, all_sessions=False, calendar=None):
        assert symbol == "AAPL"
        assert as_of == date(2026, 6, 5)
        assert window == 30
        return [
            QU100Appearance(
                data_date=date(2026, 5, 28), rank=23,
                capture_session="close", captured_at=_cap(date(2026, 5, 28), 21),
            )
        ]

    monkeypatch.setattr(
        "rainier.paper.ingest.get_qu100_appearances", fake_appearances
    )
    result = CliRunner().invoke(
        cli,
        ["paper", "appearances", "AAPL", "--as-of", "2026-06-05", "--window", "30"],
    )
    assert result.exit_code == 0, result.output
    assert "2026-05-28" in result.output
    assert "#23" in result.output


def test_cli_paper_chart_writes_png(monkeypatch, tmp_path):
    from rainier.cli import cli

    png = b"\x89PNG\r\n\x1a\ncli-bytes"

    def fake_regen(symbol, *, as_of, window=None):
        assert symbol == "AAPL"
        assert as_of == date(2026, 6, 5)
        return png, hashlib.sha256(png).hexdigest()

    monkeypatch.setattr("rainier.paper.chart_archive.regenerate_chart", fake_regen)
    out = tmp_path / "aapl.png"
    result = CliRunner().invoke(
        cli,
        ["paper", "chart", "AAPL", "--as-of", "2026-06-05", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == png
    assert hashlib.sha256(png).hexdigest()[:12] in result.output
