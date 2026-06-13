"""R-E — daily JSONB feature snapshot per QU100 member (design Appendix C).

One ``qu100_daily_features`` row per (symbol, data_date, ranking_type):
``{vwap, sma5, sma22, sma44, sma60, fractal, volume, vrvp, price_basis,
feature_version, data_gap?}``. JSONB so new attributes ship without a table
migration; the dataset joins against trades and misses for learning.

PINNED FORMULAS — ``feature_version: 1``; ANY change bumps it:

* window — the trailing **120 trading days** of priced ``stock_prices`` daily
  bars ending at the row's ``data_date`` (a bar counts as priced when all of
  o/h/l/c are non-NULL, matching the ingest's gap probe);
* ``vwap`` — that day's typical price ``(high + low + close) / 3`` (single-bar
  daily proxy; no intraday feed — upgradeable later under a new
  feature_version);
* ``sma5/22/44/60`` — close-based simple moving averages; **NULL** when fewer
  than N bars exist (warm-up), never partial-window;
* ``volume`` — that day's share volume;
* ``fractal`` — from ``analysis/pivots.py`` ``detect_pivots`` over the window
  (default ``PivotConfig``, centered lookback): the latest **confirmed**
  pivot. Confirmation lags by the lookback, so it is never a same-day signal;
  the JSON carries the pivot's own date so consumers see the lag;
* ``vrvp`` — 200 uniform price bins spanning ``[min(low), max(high)]`` of the
  window; bin ``i`` is the half-open ``[min + i*w, min + (i+1)*w)``, last bin
  closed. Each day's volume is allocated **proportionally to the overlap** of
  ``[low, high]`` with each bin (not equally across touched bins);
  ``high == low`` puts all volume in that price's bin. Summary only:
  ``{bins, poc, va_high, va_low, vol_above, vol_below}`` — POC = max-volume
  bin midpoint (tie → lower price); value area = expand from POC adding the
  higher-volume adjacent side until ≥70% (tie → lower-price side); above /
  below split at that day's close (the straddling bin splits proportionally);
* NULL-OHLC / missing day — the row is still written, with NULL features and
  ``"data_gap": true`` (a session is never skipped silently);
* ``price_basis`` — recorded per row (same basis the ingest stores). When the
  gap-triggered ingest re-fetches a window, the affected (symbol, date)
  feature rows — the ingest's returned **changed set** — are recomputed in
  the SAME daily run, immediately after ingest (``feature_version``
  unchanged), so features always match the current ``stock_prices`` basis.
  Features are derived views, not provenance (unlike charts).

The daily step is FAILURE-ISOLATED: it runs after price ingest, and an
exception here can never block ingest/fill/exit/eval/report (wired with its
own try/except in ``scheduler/service.py``; per-symbol failures are caught
and counted, never raised).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rainier.core.database import get_session
from rainier.core.models import QU100DailyFeatures, StockPrice

from .ingest import (
    Bar,
    canonical_instant,
    get_current_qu100_cohort,
    normalize_to_trading_date,
)
from .positions import PRICE_BASIS

log = logging.getLogger(__name__)

FEATURE_VERSION = 1
WINDOW_BARS = 120
VRVP_BINS = 200
SMA_PERIODS = (5, 22, 44, 60)
VALUE_AREA_PCT = 0.70
RANKING_TYPE = "top100"

_FEATURE_KEYS = ("vwap", "volume", "sma5", "sma22", "sma44", "sma60",
                 "fractal", "vrvp")


# ---------------------------------------------------------------------------
# Pure compute (no DB)
# ---------------------------------------------------------------------------


def compute_features(window: Sequence[Bar], as_of: date) -> dict[str, Any]:
    """Feature payload for ``as_of`` from its trailing priced-bar ``window``.

    ``window`` is ascending and priced (non-NULL o/h/l/c); its last bar must
    BE the ``as_of`` bar — otherwise the day has no usable bar and the pinned
    data-gap payload (all NULLs + ``"data_gap": true``) is returned.
    """
    if not window or normalize_to_trading_date(window[-1]["date"]) != as_of:
        return _gap_payload()

    bars = list(window)[-WINDOW_BARS:]
    day = bars[-1]
    closes = [float(b["close"]) for b in bars]

    feats: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "price_basis": PRICE_BASIS,
        "vwap": (float(day["high"]) + float(day["low"]) + float(day["close"])) / 3.0,
        "volume": int(day["volume"]) if day["volume"] is not None else None,
    }
    for n in SMA_PERIODS:
        feats[f"sma{n}"] = sum(closes[-n:]) / n if len(closes) >= n else None
    feats["fractal"] = _fractal(bars)
    feats["vrvp"] = _vrvp(bars, day_close=float(day["close"]))
    return feats


def _gap_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "price_basis": PRICE_BASIS,
        "data_gap": True,
    }
    payload.update({k: None for k in _FEATURE_KEYS})
    return payload


def _fractal(bars: list[Bar]) -> dict[str, Any]:
    """Latest CONFIRMED pivots (centered window — confirmation lags by the
    lookback, never a same-day signal). Each pivot carries its OWN date."""
    from rainier.analysis.pivots import detect_pivots

    df = pd.DataFrame(
        {
            "high": [float(b["high"]) for b in bars],
            "low": [float(b["low"]) for b in bars],
            "timestamp": [
                datetime.combine(
                    normalize_to_trading_date(b["date"]), datetime.min.time()
                )
                for b in bars
            ],
        }
    )
    pivots = detect_pivots(df)  # ascending bar order; [] when window < 2*lb+1

    def _as_json(p) -> dict[str, Any]:
        return {"date": p.timestamp.date().isoformat(), "price": float(p.price)}

    last_high = next((p for p in reversed(pivots) if p.is_high), None)
    last_low = next((p for p in reversed(pivots) if not p.is_high), None)
    latest = None
    if pivots:
        # detect_pivots appends in bar order — the final element is the most
        # recently confirmed pivot.
        latest = "high" if pivots[-1].is_high else "low"
    return {
        "last_pivot_high": _as_json(last_high) if last_high else None,
        "last_pivot_low": _as_json(last_low) if last_low else None,
        "latest": latest,
    }


def _vrvp(bars: list[Bar], day_close: float) -> dict[str, Any]:
    """200-bin volume-by-price summary over the window (pinned formula)."""
    lo = min(float(b["low"]) for b in bars)
    hi = max(float(b["high"]) for b in bars)

    if hi == lo:
        # Degenerate window: every bar at one price → single-bin profile.
        # Nothing trades strictly above/below that price (== the close).
        return {
            "bins": VRVP_BINS,
            "poc": lo,
            "va_high": lo,
            "va_low": lo,
            "vol_above": 0.0,
            "vol_below": 0.0,
        }

    w = (hi - lo) / VRVP_BINS

    def _bin_index(price: float) -> int:
        # Half-open bins; the last bin is closed so price == hi lands in it.
        return min(int((price - lo) / w), VRVP_BINS - 1)

    vols = np.zeros(VRVP_BINS)
    for b in bars:
        v = b["volume"]
        if not v:
            continue
        v = float(v)
        bl, bh = float(b["low"]), float(b["high"])
        if bh == bl:
            vols[_bin_index(bl)] += v
            continue
        span = bh - bl
        for i in range(_bin_index(bl), _bin_index(bh) + 1):
            left = lo + i * w
            overlap = min(bh, left + w) - max(bl, left)
            if overlap > 0:
                vols[i] += v * overlap / span

    total = float(vols.sum())

    # POC: max-volume bin midpoint; np.argmax returns the FIRST max → tie
    # breaks to the lower price.
    poc_i = int(np.argmax(vols))

    # Value area: expand from POC, adding the higher-volume adjacent side
    # until cumulative ≥ 70% of total (tie → lower-price side).
    va_lo_i = va_hi_i = poc_i
    cum = float(vols[poc_i])
    threshold = VALUE_AREA_PCT * total
    while cum < threshold:
        down_v = float(vols[va_lo_i - 1]) if va_lo_i > 0 else None
        up_v = float(vols[va_hi_i + 1]) if va_hi_i < VRVP_BINS - 1 else None
        if down_v is None and up_v is None:
            break
        if up_v is None or (down_v is not None and down_v >= up_v):
            va_lo_i -= 1
            cum += down_v
        else:
            va_hi_i += 1
            cum += up_v

    # Above/below split at the day's close; the straddling bin splits
    # proportionally so vol_above + vol_below == total.
    vol_below = 0.0
    for i in range(VRVP_BINS):
        v = float(vols[i])
        if v == 0.0:
            continue
        frac_below = min(max((day_close - (lo + i * w)) / w, 0.0), 1.0)
        vol_below += v * frac_below
    vol_above = total - vol_below

    def _mid(i: int) -> float:
        return lo + (i + 0.5) * w

    return {
        "bins": VRVP_BINS,
        "poc": _mid(poc_i),
        "va_high": _mid(va_hi_i),
        "va_low": _mid(va_lo_i),
        "vol_above": vol_above,
        "vol_below": vol_below,
    }


# ---------------------------------------------------------------------------
# DB step
# ---------------------------------------------------------------------------


def _load_window(session, symbol: str, data_date: date) -> list[Bar]:
    """Trailing ≤120 PRICED bars ending at-or-before ``data_date``, ascending.

    Priced = all of o/h/l/c non-NULL — the same usability probe as the
    ingest's gap detection, so a transient NULL row reads as a gap here too.
    """
    rows = session.execute(
        select(
            StockPrice.date,
            StockPrice.open,
            StockPrice.high,
            StockPrice.low,
            StockPrice.close,
            StockPrice.volume,
        )
        .where(
            StockPrice.symbol == symbol,
            StockPrice.date <= canonical_instant(data_date),
            StockPrice.open.isnot(None),
            StockPrice.high.isnot(None),
            StockPrice.low.isnot(None),
            StockPrice.close.isnot(None),
        )
        .order_by(StockPrice.date.desc())
        .limit(WINDOW_BARS)
    ).all()
    return [
        {
            "date": normalize_to_trading_date(r.date),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in reversed(rows)
    ]


def _upsert_features(
    session,
    symbol: str,
    data_date: date,
    rank: int | None,
    features: dict[str, Any],
    ranking_type: str = RANKING_TYPE,
) -> None:
    stmt = pg_insert(QU100DailyFeatures).values(
        symbol=symbol,
        data_date=data_date,
        ranking_type=ranking_type,
        rank=rank,
        features=features,
        computed_at=datetime.now(timezone.utc),
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_qu100_daily_features_symbol_date_ranking",
        set_={
            "rank": stmt.excluded.rank,
            "features": stmt.excluded.features,
            "computed_at": stmt.excluded.computed_at,
        },
    )
    session.execute(stmt)


def run_daily_feature_step(
    as_of: date,
    changed: Iterable[tuple[str, date]] = (),
    cohort: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Daily R-E step: snapshot today's cohort + recompute the changed set.

    Runs immediately after price ingest in the daily job. ``changed`` is the
    ingest's returned (symbol, trading_date) set — all upserted pairs for
    re-fetched symbols (over-trigger acceptable; recompute is idempotent).
    ``cohort`` lets the caller share one ``get_current_qu100_cohort`` fetch.

    Failure-isolated at TWO levels: per-symbol failures are caught and
    counted here (one bad symbol never starves the rest), and the caller
    wraps the whole step so an exception can never block the trading steps.
    Returns ``{"computed", "recomputed", "failed"}``.
    """
    if cohort is None:
        cohort = get_current_qu100_cohort(as_of)

    computed = recomputed = failed = 0
    done: set[tuple[str, date]] = set()

    # 1) Today's cohort — one row per member at its cohort data_date.
    for member in cohort:
        sym, dd = member["symbol"], member["data_date"]
        try:
            with get_session() as session:
                window = _load_window(session, sym, dd)
                feats = compute_features(window, dd)
                _upsert_features(session, sym, dd, member.get("rank"), feats)
            done.add((sym, dd))
            computed += 1
        except Exception:
            failed += 1
            log.exception("feature_compute_failed symbol=%s date=%s", sym, dd)

    # 2) Changed-set recompute (same run, design Appendix C): only rows that
    # already EXIST — a changed bar for a never-snapshotted day creates
    # nothing. rank is untouched (only the feature payload re-derives).
    for sym, dd in sorted(set(changed) - done):
        try:
            with get_session() as session:
                rows = session.execute(
                    select(QU100DailyFeatures).where(
                        QU100DailyFeatures.symbol == sym,
                        QU100DailyFeatures.data_date == dd,
                    )
                ).scalars().all()
                if not rows:
                    continue
                window = _load_window(session, sym, dd)
                feats = compute_features(window, dd)
                now = datetime.now(timezone.utc)
                for row in rows:
                    row.features = feats
                    row.computed_at = now
            recomputed += 1
        except Exception:
            failed += 1
            log.exception("feature_recompute_failed symbol=%s date=%s", sym, dd)

    log.info(
        "feature_step_done as_of=%s computed=%d recomputed=%d failed=%d",
        as_of.isoformat(), computed, recomputed, failed,
    )
    return {"computed": computed, "recomputed": recomputed, "failed": failed}
