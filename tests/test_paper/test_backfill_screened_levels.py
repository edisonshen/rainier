"""One-time backfill of historical screened_stocks pattern levels.

The screen->persist path has written entry/stop/target/rr correctly since the
06-15 cutover, but ~510 patterned rows from scan_date 06-03..06-12 carry NULL
levels (written by the pre-cutover live code). This backfill replays the pattern
detector AS-OF each historical scan_date over a FRESH yfinance window (the same
source the live screener uses — the stored ``stock_prices`` corpus is too short
to form a pattern), matches the actionable pattern whose ``pattern_type`` equals
the row's stored type, and coalesce-upserts the four levels. Dry-run by default;
``apply=True`` writes.

The screened rows + Postgres ON CONFLICT need the legacy DB, so these run under
``requires_postgres`` against ``pg_legacy_session``. The PRICE source is yfinance;
each test injects a deterministic ``{symbol: DataFrame}`` by monkeypatching
``_fetch_history`` (autouse ``_patch_fetch_history`` fixture) — no live network.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import select, text

from rainier.core.models import ScreenedStockRecord
from rainier.paper.backfill_screened_levels import (
    _target_rows,
    backfill_screened_levels,
)

pytestmark = pytest.mark.requires_postgres


# ---------------------------------------------------------------------------
# Fixture price path: the same false-breakdown-that-recovers shape the replay
# parity test uses — it yields TWO actionable patterns as-of the last bar:
#   w_bottom (best / top-ranked)  and  false_breakdown.
# This lets us assert the backfill matches the STORED pattern_type (picking
# false_breakdown even though replay's `best` is w_bottom).
# ---------------------------------------------------------------------------

_FB_PRICES = (
    [95.0, 93.0, 91.0, 90.0, 91.0, 93.0, 95.0, 93.0, 91.0, 90.0]
    + [92.0, 93.0, 91.0]  # range above support
    + [89.0, 88.0, 87.0]  # break below support
    + [89.0, 91.0, 92.0]  # recover above support (confirm)
    + [92.5, 92.0, 92.5, 92.0, 92.5]  # hold near entry (still actionable)
)

# StockScreenerConfig knobs that make the fixture's patterns detect (mirrors the
# replay parity test's _CONFIG).
_CFG_OVERRIDES = {
    "swing_lookback": 3,
    "min_pattern_bars": 3,
    "max_pattern_bars": 50,
    "neckline_tolerance_pct": 0.05,
}


# In-memory price corpus the patched ``_fetch_history`` serves. Mirrors the
# yfinance return shape: tz-naive DatetimeIndex, lowercase OHLCV columns. Tests
# populate it via ``_seed_prices``; the autouse ``_patch_fetch_history`` fixture
# resets it per test and routes ``_fetch_history`` to read from it (no network).
_PRICE_FIXTURES: dict[str, pd.DataFrame] = {}


@pytest.fixture(autouse=True)
def _patch_fetch_history(monkeypatch):
    """Route ``_fetch_history`` to the in-memory ``_PRICE_FIXTURES`` map.

    The source swap means the backfill fetches prices from yfinance, not the DB.
    Tests must not hit the network, so this autouse fixture replaces the helper
    with one that returns the deterministic frames ``_seed_prices`` registered —
    honoring the requested symbol set, exactly like the real batch fetch."""
    import rainier.paper.backfill_screened_levels as mod

    _PRICE_FIXTURES.clear()

    def fake_fetch(symbols, *, start, end, min_bars):
        return {s: _PRICE_FIXTURES[s] for s in symbols if s in _PRICE_FIXTURES}

    monkeypatch.setattr(mod, "_fetch_history", fake_fetch)
    yield
    _PRICE_FIXTURES.clear()


def _seed_prices(session, symbol: str, last_date: date, prices: list[float]) -> None:
    """Register a yfinance-shaped OHLCV frame ending ON ``last_date``.

    Bars are one business day apart with a tz-naive DatetimeIndex and lowercase
    columns — the exact shape ``_fetch_history`` returns. The ``session`` arg is
    kept for call-site symmetry but unused (prices no longer live in the DB)."""
    dates = pd.bdate_range(end=pd.Timestamp(last_date), periods=len(prices))
    df = pd.DataFrame(
        {
            "open": [prices[max(0, i - 1)] for i in range(len(prices))],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": list(prices),
            "volume": [1000] * len(prices),
        },
        index=dates,
    )
    _PRICE_FIXTURES[symbol] = df


def _seed_screened(
    session,
    symbol: str,
    scan_date: date,
    *,
    pattern_type: str | None,
    levels_null: bool = True,
    entry=None,
    stop=None,
    target=None,
    rr=None,
    session_name: str = "close",
) -> None:
    session.add(
        ScreenedStockRecord(
            scan_date=scan_date,
            session_name=session_name,
            symbol=symbol,
            rule_rank=1,
            composite_score=0.8,
            pattern_type=pattern_type,
            entry_price=None if levels_null else entry,
            stop_loss=None if levels_null else stop,
            target_price=None if levels_null else target,
            rr_ratio=None if levels_null else rr,
        )
    )
    session.commit()


def _row(session, symbol: str, scan_date: date) -> ScreenedStockRecord:
    return (
        session.execute(
            select(ScreenedStockRecord).where(
                ScreenedStockRecord.symbol == symbol,
                ScreenedStockRecord.scan_date == scan_date,
            )
        )
        .scalars()
        .one()
    )


_SCAN = date(2026, 6, 5)


def test_patterned_null_row_backfilled(pg_legacy_session):
    """A patterned row with NULL levels gets the four levels from the as-of
    replay's matching pattern."""
    _seed_prices(pg_legacy_session, "AAA", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "AAA", _SCAN, pattern_type="false_breakdown")

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "AAA", _SCAN)
    assert r.entry_price is not None
    assert r.stop_loss is not None
    assert r.target_price is not None
    assert r.rr_ratio is not None
    assert res.scanned == 1
    assert res.recovered == 1
    assert res.still_null == 0


def test_matches_stored_pattern_type_not_top_ranked_best(pg_legacy_session):
    """Levels come from the actionable pattern whose type == stored pattern_type,
    even when that is NOT replay's top-ranked `best`. The fixture's `best` is
    w_bottom; the stored row says false_breakdown — the persisted levels must be
    the false_breakdown's (entry ~89.1), never w_bottom's (entry ~93.9)."""
    _seed_prices(pg_legacy_session, "FBK", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "FBK", _SCAN, pattern_type="false_breakdown")

    backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "FBK", _SCAN)
    # false_breakdown entry ~89.1, w_bottom entry ~93.9 — assert the former.
    assert r.entry_price == pytest.approx(89.1, abs=1.0)
    assert r.entry_price < 91.0  # definitively NOT the w_bottom entry


def test_dry_run_writes_nothing(pg_legacy_session):
    _seed_prices(pg_legacy_session, "DRY", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "DRY", _SCAN, pattern_type="false_breakdown")

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=False,  # default: report only
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "DRY", _SCAN)
    assert r.entry_price is None  # nothing written
    assert res.scanned == 1
    assert res.recovered == 1  # counted as would-recover


def test_coalesce_no_clobber_of_set_levels(pg_legacy_session):
    """A row that already has levels is not in the target set (entry NOT NULL),
    so the backfill never touches it."""
    _seed_prices(pg_legacy_session, "SET", _SCAN, _FB_PRICES)
    _seed_screened(
        pg_legacy_session,
        "SET",
        _SCAN,
        pattern_type="false_breakdown",
        levels_null=False,
        entry=55.0,
        stop=50.0,
        target=70.0,
        rr=3.0,
    )

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "SET", _SCAN)
    assert (r.entry_price, r.stop_loss, r.target_price, r.rr_ratio) == (55.0, 50.0, 70.0, 3.0)
    assert res.scanned == 0  # already-set row excluded from target set


def test_patternless_row_untouched(pg_legacy_session):
    _seed_prices(pg_legacy_session, "NOP", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "NOP", _SCAN, pattern_type=None)

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "NOP", _SCAN)
    assert r.entry_price is None
    assert res.scanned == 0  # patternless rows are not in the target set


def test_unreconstructable_pattern_reported_not_errored(pg_legacy_session):
    """A patterned row whose stored type does NOT re-detect as-of is left NULL
    and counted in still_null — never errored, never given wrong-pattern levels."""
    _seed_prices(pg_legacy_session, "MISS", _SCAN, _FB_PRICES)
    # bull_flag is not among the fixture's actionable patterns.
    _seed_screened(pg_legacy_session, "MISS", _SCAN, pattern_type="bull_flag")

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "MISS", _SCAN)
    assert r.entry_price is None
    assert res.scanned == 1
    assert res.recovered == 0
    assert res.still_null == 1
    assert ("MISS", _SCAN) in res.still_null_keys


def test_idempotent_second_run_changes_nothing(pg_legacy_session):
    _seed_prices(pg_legacy_session, "IDEM", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "IDEM", _SCAN, pattern_type="false_breakdown")

    backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )
    pg_legacy_session.expire_all()
    first = _row(pg_legacy_session, "IDEM", _SCAN)
    first_levels = (first.entry_price, first.stop_loss, first.target_price, first.rr_ratio)

    # Second run: the row now has levels → no longer in the target set.
    res2 = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )
    pg_legacy_session.expire_all()
    second = _row(pg_legacy_session, "IDEM", _SCAN)
    assert (
        second.entry_price,
        second.stop_loss,
        second.target_price,
        second.rr_ratio,
    ) == first_levels
    assert res2.scanned == 0  # nothing left to backfill


def test_out_of_window_row_not_scanned(pg_legacy_session):
    """A patterned NULL row OUTSIDE [from,to] is never touched."""
    out_of_window = date(2026, 6, 16)
    _seed_prices(pg_legacy_session, "OOW", out_of_window, _FB_PRICES)
    _seed_screened(pg_legacy_session, "OOW", out_of_window, pattern_type="false_breakdown")

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "OOW", out_of_window)
    assert r.entry_price is None
    assert res.scanned == 0


def test_non_close_session_row_not_backfilled(pg_legacy_session):
    """A patterned NULL row from a non-close session is NEVER backfilled.

    Only the close-session daily bar matches what the live screen saw; an
    earlier session lacked the day's final high/low/close, so replaying its
    completed bar would inject look-ahead. The row is left NULL and excluded
    from the target set entirely (not even counted as still_null)."""
    _seed_prices(pg_legacy_session, "MORN", _SCAN, _FB_PRICES)
    _seed_screened(
        pg_legacy_session,
        "MORN",
        _SCAN,
        pattern_type="false_breakdown",
        session_name="morning",
    )

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "MORN", _SCAN)
    assert r.entry_price is None  # never written
    assert res.scanned == 0  # non-close row not in the target set
    assert res.recovered == 0
    # ...but it IS still a damaged row, surfaced so a dry-run isn't falsely complete.
    assert res.skipped_non_close == 1


def test_no_price_data_row_is_separate_bucket_not_still_null(pg_legacy_session):
    """A patterned NULL row whose symbol returned NO yfinance bars (transient
    fetch failure) lands in `no_price_data`, NOT `still_null`. Conflating them
    would let a fetch blip masquerade as a permanent no-pattern-match, so the
    operator would mark a recoverable row unreconstructable. Seed the screened
    row but NO prices for its symbol → the faked `_fetch_history` omits it."""
    # NB: deliberately do NOT call _seed_prices for NODATA → absent from corpus.
    _seed_screened(pg_legacy_session, "NODATA", _SCAN, pattern_type="false_breakdown")

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    r = _row(pg_legacy_session, "NODATA", _SCAN)
    assert r.entry_price is None  # nothing written
    assert res.scanned == 1
    assert res.recovered == 0
    assert res.still_null == 0  # NOT a permanent no-match
    assert res.no_price_data == 1  # transient: surfaced for re-run
    assert ("NODATA", _SCAN) in res.no_price_data_keys
    assert ("NODATA", _SCAN) not in res.still_null_keys


def test_multiple_scan_dates_each_gets_its_own_window_levels(pg_legacy_session):
    """Two close-session patterned-NULL rows on DIFFERENT scan_dates each get the
    levels derived from THEIR OWN as-of window. The apply path groups candidates
    by (scan_date, session_name) and persists per group; a regression mapping a
    candidate to the wrong scan_date would mis-persist. Distinct price magnitudes
    per symbol prove each row got its own window, not the other's."""
    scan_a = date(2026, 6, 5)
    scan_b = date(2026, 6, 9)
    # Distinct price bases → distinct entry levels, so a cross-mapping is visible.
    _seed_prices(pg_legacy_session, "AAA", scan_a, _FB_PRICES)
    _seed_screened(pg_legacy_session, "AAA", scan_a, pattern_type="false_breakdown")
    _seed_prices(pg_legacy_session, "BBB", scan_b, [p + 100.0 for p in _FB_PRICES])
    _seed_screened(pg_legacy_session, "BBB", scan_b, pattern_type="false_breakdown")

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    ra = _row(pg_legacy_session, "AAA", scan_a)
    rb = _row(pg_legacy_session, "BBB", scan_b)
    assert res.recovered == 2
    assert ra.entry_price is not None and rb.entry_price is not None
    # AAA's window (~89) is the low corpus; BBB's (+100, ~189) is the high one.
    assert ra.entry_price < 100.0
    assert rb.entry_price > 100.0


def test_multiple_same_group_rows_all_persisted(pg_legacy_session):
    """Two recovered rows in the SAME (scan_date, session_name) group are BOTH
    written via the single per-group persist_screened_stocks call — the per-group
    candidate list is fully persisted, not just its first element."""
    _seed_prices(pg_legacy_session, "AAA", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "AAA", _SCAN, pattern_type="false_breakdown")
    _seed_prices(pg_legacy_session, "BBB", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "BBB", _SCAN, pattern_type="false_breakdown")

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    assert res.recovered == 2
    assert _row(pg_legacy_session, "AAA", _SCAN).entry_price is not None
    assert _row(pg_legacy_session, "BBB", _SCAN).entry_price is not None


def test_detector_failure_on_one_row_does_not_abort_run(
    pg_legacy_session, monkeypatch
):
    """A detector exception on one symbol is per-symbol noise: that row lands in
    still_null and the run continues to repair the others (never aborts)."""
    import rainier.paper.backfill_screened_levels as mod

    _seed_prices(pg_legacy_session, "BAD", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "BAD", _SCAN, pattern_type="false_breakdown")
    _seed_prices(pg_legacy_session, "GOOD", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "GOOD", _SCAN, pattern_type="false_breakdown")

    real_replay = mod.replay_pattern_layer

    def flaky_replay(symbol, windowed, config):
        if symbol == "BAD":
            raise RuntimeError("synthetic detector blow-up")
        return real_replay(symbol, windowed, config)

    monkeypatch.setattr(mod, "replay_pattern_layer", flaky_replay)

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    pg_legacy_session.expire_all()
    # The run completed (no abort): BAD → still_null, GOOD → recovered + written.
    assert res.scanned == 2
    assert res.recovered == 1
    assert res.still_null == 1
    assert ("BAD", _SCAN) in res.still_null_keys
    assert _row(pg_legacy_session, "BAD", _SCAN).entry_price is None
    assert _row(pg_legacy_session, "GOOD", _SCAN).entry_price is not None


# ---------------------------------------------------------------------------
# Schema-drift regression (the P1 fix): the live local DB is missing
# `screened_stocks.bearish_invalidation_level` (migration 0012 never applied).
# A full-ORM `select(ScreenedStockRecord)` would SELECT that column and crash
# with UndefinedColumn. The column-scoped `_target_rows` must survive it.
# ---------------------------------------------------------------------------


def _column_exists(session, table: str, column: str) -> bool:
    # to_regclass(table) resolves the table on the session's search_path (the
    # throwaway test schema), so the catalog lookup targets THIS schema's table,
    # not a same-named `public` one.
    return bool(
        session.execute(
            text(
                "SELECT 1 FROM pg_attribute "
                "WHERE attrelid = to_regclass(:t) "
                "AND attname = :c AND NOT attisdropped"
            ),
            {"t": table, "c": column},
        ).first()
    )


def _seed_screened_raw(session, symbol: str, scan_date: date, pattern_type: str) -> None:
    """Insert a patterned, level-NULL `screened_stocks` row via column-scoped SQL.

    The full-ORM `_seed_screened` emits an INSERT over EVERY mapped column,
    including `bearish_invalidation_level` — which the drift DB lacks. Seeding
    here must therefore name only columns that exist pre-0012, or the seed itself
    crashes before the backfill under test even runs.
    """
    session.execute(
        text(
            "INSERT INTO screened_stocks "
            "(scan_date, session_name, symbol, rule_rank, composite_score, pattern_type) "
            "VALUES (:d, 'close', :s, 1, 0.8, :p)"
        ),
        {"d": scan_date, "s": symbol, "p": pattern_type},
    )
    session.commit()


def test_drift_db_missing_model_column_does_not_crash(pg_drift_session):
    """On a DB missing `bearish_invalidation_level` (migration 0012 unapplied),
    the DRY-RUN backfill must scan and report real counts — never crash with
    UndefinedColumn. This is the operator's actual usage (dry-run first; 0012 is
    applied before any `--apply`), and the surface this task hardens: the
    column-scoped read in `_target_rows`. FAILS on the parent commit, whose
    full-ORM `select(ScreenedStockRecord)` SELECTs the missing column."""
    # Precondition: confirm the column really is absent on this drift fixture, so
    # this test is meaningfully exercising the drift path (not a healthy DB).
    assert not _column_exists(
        pg_drift_session, "screened_stocks", "bearish_invalidation_level"
    )

    _seed_prices(pg_drift_session, "DRF", _SCAN, _FB_PRICES)
    _seed_screened_raw(pg_drift_session, "DRF", _SCAN, "false_breakdown")

    # Dry-run (apply=False): the read path under test. (apply=True would write
    # through `persist_screened_stocks`, whose INSERT names the 0012 column — out
    # of scope here; the operator applies 0012 before `--apply`, per the plan.)
    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=False,
        config_overrides=_CFG_OVERRIDES,
    )

    # Scanned and matched the row without crashing on the missing column.
    assert res.scanned == 1
    assert res.recovered == 1  # would-recover (dry-run writes nothing)
    assert res.still_null == 0
    # Column-scoped read-back (a full-ORM `_row` would itself crash on the
    # missing column): the dry-run wrote nothing, so entry_price is still NULL.
    pg_drift_session.expire_all()
    entry = pg_drift_session.execute(
        text(
            "SELECT entry_price FROM screened_stocks "
            "WHERE symbol = :s AND scan_date = :d"
        ),
        {"s": "DRF", "d": _SCAN},
    ).scalar_one()
    assert entry is None


def test_apply_on_drift_db_raises_clear_error_not_undefined_column(pg_drift_session):
    """`--apply` on a pre-0012 DB must fail fast with a clear, actionable message,
    NOT crash mid-repair with an opaque UndefinedColumn from the upsert.

    The write path (`persist_screened_stocks`) names `bearish_invalidation_level`
    in its INSERT ... ON CONFLICT; that column is absent on a drifted DB. A
    dry-run on the same DB succeeds and tells the operator to re-run with
    `--apply` — so without the preflight guard `--apply` would attempt the write
    and abort with a raw psycopg error. The guard raises a RuntimeError that names
    the missing column and migration 0012, and leaves the row UNWRITTEN.

    FAILS on the parent commit: there the apply path reaches the upsert and raises
    a psycopg `UndefinedColumn` (not the guarded RuntimeError)."""
    # Precondition: the column really is absent on this drift fixture.
    assert not _column_exists(
        pg_drift_session, "screened_stocks", "bearish_invalidation_level"
    )

    _seed_prices(pg_drift_session, "DRF", _SCAN, _FB_PRICES)
    _seed_screened_raw(pg_drift_session, "DRF", _SCAN, "false_breakdown")

    with pytest.raises(RuntimeError, match="migration 0012"):
        backfill_screened_levels(
            from_date=date(2026, 6, 3),
            to_date=date(2026, 6, 12),
            apply=True,
            config_overrides=_CFG_OVERRIDES,
        )

    # The guard refused before any write: the row's levels stay NULL.
    pg_drift_session.expire_all()
    entry = pg_drift_session.execute(
        text(
            "SELECT entry_price FROM screened_stocks "
            "WHERE symbol = :s AND scan_date = :d"
        ),
        {"s": "DRF", "d": _SCAN},
    ).scalar_one()
    assert entry is None


def test_apply_on_healthy_db_passes_preflight(pg_legacy_session):
    """On a complete (post-0012) DB the apply preflight passes and the write
    proceeds — the guard must not block a healthy `--apply`."""
    _seed_prices(pg_legacy_session, "OK", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "OK", _SCAN, pattern_type="false_breakdown")

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )
    assert res.recovered == 1

    pg_legacy_session.expire_all()
    entry = pg_legacy_session.execute(
        text(
            "SELECT entry_price FROM screened_stocks "
            "WHERE symbol = :s AND scan_date = :d"
        ),
        {"s": "OK", "d": _SCAN},
    ).scalar_one()
    assert entry is not None


def test_missing_scan_date_bar_is_transient_no_price_data_no_prior_day_levels(
    pg_legacy_session,
):
    """Fidelity gate: when the fetch has NO bar exactly on the row's scan_date (a
    vendor gap), the backfill must NOT replay the prior trading day's bar and write
    stale prior-day levels. The row is left UNWRITTEN.

    Classification (codex iter-N P2): a missing exact-scan_date bar is a TRANSIENT
    gap — a later fetch can supply it — so it belongs in `no_price_data`
    (recoverable, re-run may help), NOT `still_null` (which the CLI labels a
    PERMANENT no-pattern-match). Mislabeling it permanent would tell the operator
    to stop retrying a row a re-run could fix.

    Seed the full price history but ending the business day BEFORE scan_date, so
    the canonical scan_date bar is absent."""
    prior_bday = (pd.Timestamp(_SCAN) - pd.tseries.offsets.BDay(1)).date()
    # Prices end on the prior business day → no bar exactly on _SCAN.
    _seed_prices(pg_legacy_session, "GAP", prior_bday, _FB_PRICES)
    _seed_screened(pg_legacy_session, "GAP", _SCAN, pattern_type="false_breakdown")

    res = backfill_screened_levels(
        from_date=date(2026, 6, 3),
        to_date=date(2026, 6, 12),
        apply=True,
        config_overrides=_CFG_OVERRIDES,
    )

    assert res.scanned == 1
    assert res.recovered == 0
    # Transient (missing exact bar) → no_price_data, NOT permanent still_null.
    assert res.still_null == 0
    assert ("GAP", _SCAN) not in res.still_null_keys
    assert res.no_price_data == 1
    assert ("GAP", _SCAN) in res.no_price_data_keys
    # No prior-day levels written: the row's entry_price is still NULL.
    pg_legacy_session.expire_all()
    assert _row(pg_legacy_session, "GAP", _SCAN).entry_price is None


def test_target_rows_match_orm_path_on_healthy_db(pg_legacy_session):
    """On a complete DB the column-scoped `_target_rows` returns the SAME row set
    (by identity key) the old full-ORM load did — no behavior change when nothing
    is drifted."""
    # Two close-session patterned NULL rows (in target set), plus a set-levels row
    # and a non-close row (both excluded). The scoped query must return exactly
    # the two target rows, in (symbol, scan_date) order.
    _seed_prices(pg_legacy_session, "AAA", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "AAA", _SCAN, pattern_type="false_breakdown")
    _seed_prices(pg_legacy_session, "BBB", _SCAN, _FB_PRICES)
    _seed_screened(pg_legacy_session, "BBB", _SCAN, pattern_type="false_breakdown")
    _seed_screened(
        pg_legacy_session,
        "CCC",
        _SCAN,
        pattern_type="false_breakdown",
        levels_null=False,
        entry=55.0,
        stop=50.0,
        target=70.0,
        rr=3.0,
    )
    _seed_screened(
        pg_legacy_session,
        "DDD",
        _SCAN,
        pattern_type="false_breakdown",
        session_name="morning",
    )

    # Old ORM path: the same WHERE/ORDER over full entities.
    orm_rows = (
        pg_legacy_session.execute(
            select(ScreenedStockRecord)
            .where(
                ScreenedStockRecord.scan_date >= date(2026, 6, 3),
                ScreenedStockRecord.scan_date <= date(2026, 6, 12),
                ScreenedStockRecord.session_name == "close",
                ScreenedStockRecord.pattern_type.isnot(None),
                ScreenedStockRecord.entry_price.is_(None),
            )
            .order_by(
                ScreenedStockRecord.symbol.asc(),
                ScreenedStockRecord.scan_date.asc(),
            )
        )
        .scalars()
        .all()
    )
    orm_keys = [(r.symbol, r.scan_date) for r in orm_rows]

    scoped = _target_rows(pg_legacy_session, date(2026, 6, 3), date(2026, 6, 12))
    scoped_keys = [(r.symbol, r.scan_date) for r in scoped]

    assert scoped_keys == orm_keys == [("AAA", _SCAN), ("BBB", _SCAN)]
    # And the carried columns match the ORM values.
    for scoped_row, orm_row in zip(scoped, orm_rows, strict=True):
        assert scoped_row.session_name == orm_row.session_name
        assert scoped_row.pattern_type == orm_row.pattern_type
        assert scoped_row.sector == orm_row.sector
        assert scoped_row.composite_score == orm_row.composite_score
