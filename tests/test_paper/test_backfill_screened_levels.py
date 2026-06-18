"""One-time backfill of historical screened_stocks pattern levels.

The screen->persist path has written entry/stop/target/rr correctly since the
06-15 cutover, but ~510 patterned rows from scan_date 06-03..06-12 carry NULL
levels (written by the pre-cutover live code). This backfill replays the pattern
detector AS-OF each historical scan_date over stored ``stock_prices``, matches the
actionable pattern whose ``pattern_type`` equals the row's stored type, and
coalesce-upserts the four levels. Dry-run by default; ``apply=True`` writes.

The detector / replay needs the legacy DB + Postgres ON CONFLICT, so these run
under ``requires_postgres`` against ``pg_legacy_session``.
"""

from __future__ import annotations

from datetime import date, timezone

import pandas as pd
import pytest
from sqlalchemy import select, text

from rainier.core.models import ScreenedStockRecord, StockPrice
from rainier.paper.backfill_screened_levels import backfill_screened_levels

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


def _seed_prices(session, symbol: str, last_date: date, prices: list[float]) -> None:
    """Insert OHLCV bars ending ON ``last_date`` (one business day apart)."""
    dates = pd.bdate_range(end=pd.Timestamp(last_date), periods=len(prices))
    for i, (d, p) in enumerate(zip(dates, prices, strict=True)):
        session.add(
            StockPrice(
                symbol=symbol,
                date=d.to_pydatetime().replace(tzinfo=timezone.utc),
                open=prices[max(0, i - 1)],
                high=p * 1.01,
                low=p * 0.99,
                close=p,
                volume=1000,
            )
        )
    session.commit()


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
) -> None:
    session.add(
        ScreenedStockRecord(
            scan_date=scan_date,
            session_name="close",
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
