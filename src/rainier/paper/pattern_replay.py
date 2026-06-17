"""Faithful live-ranker PATTERN-LAYER replay over Postgres ``stock_prices``.

The live screen (`analysis/stock_screener.py:screen_stocks`) ranks each QU100
stock by a 3-layer composite. Layer 3 (patterns) is the heavy lever (65% of the
weight) yet its track record is unmeasured. The existing `qu100_portfolio`
backtest engine DIVERGES from the live ranker (2 patterns, top-20,
confidence-only), so it cannot measure what production does.

This module re-implements ONLY the parts the live ranker runs as-of a day `t`,
faithfully:

    detect_patterns(df_up_to_t)  ──►  _filter_actionable(df_up_to_t)  ──►
    best_pattern (= patterns[0] after the filter reorders)  ──►
    3-layer composite (with the live sector double-count)

Why a re-implementation and not `screen_stocks` verbatim: the live Layer 1
reads the *latest* money-flow snapshot behind a `datetime.now()` staleness gate,
and the live Layer 3 does a live `yf.download(period="6mo")` — neither is
reproducible as-of a historical `t`. So we feed the detector the SAME bars the
live path would see (the ~6-month window ending at `t`) and reuse the live
`detect_patterns` / `_filter_actionable` functions unchanged (the detector is
the instrument; altering it invalidates the measurement).

    ASCII flow (one symbol, one as-of day t):

      stock_prices OHLC ── window last ~126 bars ending at t ──► df_t
            df_t ──► detect_patterns ──► _filter_actionable ──► best_pattern
            best_pattern.confidence ─┐
            money-flow signal_strength ├─► composite (sector double-counted)
            sector_boost ─────────────┘

WS1 scope: the **pattern layer** is the 1-year deliverable (needs only the
detector + prices). The **full 3-layer composite** is only reproducible where a
money-flow snapshot exists; `replay_composite` reproduces the live formula
exactly (including the sector double-count) so a parity test can assert it on
recent dates. Building an as-of money-flow selector for a 1-year composite
replay is WS3, NOT this PR.

Corpus source: legacy local TimescaleDB via `core.database.get_session` — NOT
the `data/cache/qu100_backtest` adjusted-close artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from rainier.analysis.sector_analyzer import get_sector_boost
from rainier.analysis.stock_patterns import detect_patterns
from rainier.analysis.stock_screener import _filter_actionable
from rainier.core.config import StockScreenerConfig
from rainier.core.models import StockPrice
from rainier.core.types import PatternSignal, SectorTrend

# Live `_fetch_stock_data` uses yfinance `period="6mo"` — a CALENDAR 6-month
# slice ending at `t`, NOT a fixed bar count. The replay mirrors that exactly
# when the frame is date-indexed (the production `load_prices` path): a fixed
# 126-bar slice would feed `detect_patterns` a different left edge on
# holiday-heavy stretches where 6 calendar months != 126 sessions, and swing
# detection / `max_pattern_bars=120` are boundary-sensitive. Kept here (not in
# StockScreenerConfig) because it mirrors a yfinance fetch param, not a knob.
LIVE_LOOKBACK_MONTHS = 6
# Bar-count fallback for frames WITHOUT a datetime index (e.g. synthetic test
# fixtures): ~126 trading sessions ≈ 6 months. The production path uses the
# calendar window above; this only governs index-less frames.
LIVE_LOOKBACK_BARS = 126

# Columns the detector + filter expect (lowercase), in canonical order.
_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class PatternEmission:
    """One actionable pattern the live ranker would consume as-of ``as_of``.

    ``pattern_contribution`` is the pattern layer's term in the composite
    (``layer_weight_pattern * confidence``) — the slice this audit measures.
    """

    symbol: str
    as_of: pd.Timestamp
    pattern_type: str
    direction: str
    status: str
    confidence: float
    pattern_contribution: float
    entry_price: float
    stop_loss: float
    target_wave1: float
    close_at_t: float


# ---------------------------------------------------------------------------
# Price loading — Postgres stock_prices (legacy engine), deterministic order.
# ---------------------------------------------------------------------------


def load_prices(
    session: Session,
    symbols: list[str],
    *,
    start_date: "pd.Timestamp | None" = None,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV per symbol from ``stock_prices``, sorted ``symbol, date``.

    Returns a dict ``{symbol: DataFrame}`` with a tz-aware ``date`` index and
    lowercase OHLCV columns. The explicit ``ORDER BY symbol, date`` makes the
    corpus byte-deterministic on re-run.

    ``start_date`` (inclusive) bounds the SQL load so a trailing-window audit
    over a multi-year table does not materialize all history. The caller must
    pad it left by the detector lookback so the earliest in-window as-of bar
    still sees its full ~6-month window.
    """
    query = select(
        StockPrice.symbol,
        StockPrice.date,
        StockPrice.open,
        StockPrice.high,
        StockPrice.low,
        StockPrice.close,
        StockPrice.volume,
    ).where(StockPrice.symbol.in_(symbols))
    if start_date is not None:
        query = query.where(StockPrice.date >= start_date)
    rows = session.execute(
        query.order_by(StockPrice.symbol.asc(), StockPrice.date.asc())
    ).all()

    by_symbol: dict[str, list] = {}
    for r in rows:
        by_symbol.setdefault(r.symbol, []).append(r)

    out: dict[str, pd.DataFrame] = {}
    for sym, sym_rows in by_symbol.items():
        df = pd.DataFrame(
            {
                "date": [r.date for r in sym_rows],
                "open": [r.open for r in sym_rows],
                "high": [r.high for r in sym_rows],
                "low": [r.low for r in sym_rows],
                "close": [r.close for r in sym_rows],
                "volume": [r.volume for r in sym_rows],
            }
        )
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        df = df.set_index("date")
        out[sym] = df
    return out


# ---------------------------------------------------------------------------
# As-of windowing — the live ~6-month lookback ending at t (no look-ahead).
# ---------------------------------------------------------------------------


def window_as_of(
    df: pd.DataFrame,
    t_idx: int,
    lookback_bars: int = LIVE_LOOKBACK_BARS,
    lookback_months: int = LIVE_LOOKBACK_MONTHS,
) -> pd.DataFrame:
    """Return the bars the live path would see as-of position ``t_idx``.

    Only bars up to and including ``t_idx`` (no look-ahead). When ``df`` is
    date-indexed (the production path), the left edge is a CALENDAR
    ``lookback_months`` cut ending at ``t`` — byte-faithful to yfinance
    ``period="6mo"``. For an index-less frame (synthetic fixtures), it falls
    back to the last ``lookback_bars`` rows. ``t_idx`` is the 0-based positional
    row; the returned frame's last row IS the as-of bar.
    """
    if t_idx < 0 or t_idx >= len(df):
        raise IndexError(f"t_idx {t_idx} out of range for {len(df)} bars")
    upto = df.iloc[: t_idx + 1]
    if isinstance(df.index, pd.DatetimeIndex):
        t_ts = df.index[t_idx]
        # period="6mo" = bars strictly after (t - 6 months), inclusive of t.
        cutoff = t_ts - pd.DateOffset(months=lookback_months)
        return upto[upto.index > cutoff]
    start = max(0, t_idx + 1 - lookback_bars)
    return upto.iloc[start:]


# ---------------------------------------------------------------------------
# Pattern-layer replay — the faithful live mechanism, as-of t.
# ---------------------------------------------------------------------------


def replay_pattern_layer(
    symbol: str,
    windowed_df: pd.DataFrame,
    config: StockScreenerConfig,
) -> tuple[list[PatternSignal], PatternSignal | None]:
    """Run the live Layer-3 mechanism on the as-of window.

    ``detect_patterns`` → ``_filter_actionable`` → ``best_pattern`` (the first
    actionable after the filter reorders), exactly as `screen_stocks` does.
    Returns ``(actionable_patterns, best_pattern)``.
    """
    all_patterns = detect_patterns(symbol, windowed_df, config)
    actionable = _filter_actionable(all_patterns, windowed_df)
    best = actionable[0] if actionable else None
    return actionable, best


def replay_composite(
    *,
    signal_strength: float,
    sector_boost: float,
    best_confidence: float,
    config: StockScreenerConfig,
) -> float:
    """Reproduce the live composite EXACTLY — including the sector double-count.

    The live formula (`stock_screener.py:99`) is NOT three independent layers:
    ``sector`` is already folded into ``signal.signal_strength``
    (`_apply_sector_boost`) AND ``sector_boost`` is added again here. We mirror
    that — a "cleaner" formula would not match production.

        composite = w_mf * signal_strength    (signal_strength ALREADY +sector)
                  + w_sector * sector_boost    (sector counted a 2nd time)
                  + w_pattern * best_confidence

    Rounded to 4 dp to match the live `round(composite, 4)`.
    """
    composite = (
        config.layer_weight_money_flow * signal_strength
        + config.layer_weight_sector * sector_boost
        + config.layer_weight_pattern * best_confidence
    )
    return round(composite, 4)


def emission_at(
    symbol: str,
    windowed_df: pd.DataFrame,
    config: StockScreenerConfig,
) -> PatternEmission | None:
    """Build a ``PatternEmission`` for ``symbol`` as-of the window's last bar.

    Returns ``None`` when no actionable pattern exists (the live ranker would
    then carry no pattern contribution for this symbol on this day).
    """
    if windowed_df.empty:
        return None
    _, best = replay_pattern_layer(symbol, windowed_df, config)
    if best is None:
        return None
    as_of = windowed_df.index[-1]
    return PatternEmission(
        symbol=symbol,
        as_of=as_of,
        pattern_type=best.pattern_type,
        direction=best.direction,
        status=best.status,
        confidence=best.confidence,
        pattern_contribution=round(
            config.layer_weight_pattern * best.confidence, 6
        ),
        entry_price=best.entry_price,
        stop_loss=best.stop_loss,
        target_wave1=best.target_wave1,
        close_at_t=float(windowed_df["close"].iloc[-1]),
    )


# ---------------------------------------------------------------------------
# Parity helper — composite-ranked selection mirroring screen_stocks, used by
# the recent-date composite parity test (live latest-snapshot path).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    """One ranked row, mirroring the live `screen_stocks` selection."""

    symbol: str
    composite_score: float
    best_pattern_type: str | None
    best_confidence: float


def replay_screen(
    *,
    signals_by_symbol: dict[str, float],
    sectors_by_symbol: dict[str, str],
    sector_trends: list[SectorTrend],
    windows_by_symbol: dict[str, pd.DataFrame],
    config: StockScreenerConfig,
) -> list[ReplaySelection]:
    """Reproduce the live composite ranking over a set of symbols, as-of `t`.

    ``signals_by_symbol`` carries each symbol's money-flow ``signal_strength``
    (already sector-boosted, mirroring the live `_apply_sector_boost`).
    ``windows_by_symbol`` carries each symbol's as-of OHLCV window. Returns the
    composite-descending selection — ordering + best_pattern + score — so a
    parity test can assert it against `screen_stocks`.
    """
    selections: list[ReplaySelection] = []
    for sym, signal_strength in signals_by_symbol.items():
        df = windows_by_symbol.get(sym)
        sector = sectors_by_symbol.get(sym, "Unknown")
        sector_boost = get_sector_boost(sector, sector_trends)
        if df is not None and not df.empty:
            _, best = replay_pattern_layer(sym, df, config)
        else:
            best = None
        best_confidence = best.confidence if best else 0.0
        composite = replay_composite(
            signal_strength=signal_strength,
            sector_boost=sector_boost,
            best_confidence=best_confidence,
            config=config,
        )
        selections.append(
            ReplaySelection(
                symbol=sym,
                composite_score=composite,
                best_pattern_type=best.pattern_type if best else None,
                best_confidence=best_confidence,
            )
        )
    # Mirror the live `results.sort(key=composite_score, reverse=True)`. Python's
    # sort is stable, matching the live insertion order for ties.
    selections.sort(key=lambda s: s.composite_score, reverse=True)
    return selections
