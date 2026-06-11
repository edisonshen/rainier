"""R-D annotated chart archive (DESIGN-qu100-llm-feedback-loop §5/§6, App. B/C).

A 120-day (config ``llm_thesis.chart_lookback_days``) daily-candle PNG for
every closed paper trade — QU100 appearance dates marked (vertical line,
"#rank · M/D" label) and entry/stop/target/exit overlaid — persisted
append-only and regenerable on demand.

    stock_prices ──► load_daily_bars ──┐
                                       ├─► build_archive_figure ─► kaleido PNG
    money_flow_snapshots ─► appearances┘        │                  + metadata
    paper_trade (closed) ─► TradeOverlay ───────┘                  scrub
                                                                     │
                       persist_archive_chart (append-only supersession)
                                                                     │
                                   chart_images + paper_trade.chart_id

DETERMINISM / CACHE CONTRACT: this module lives OUTSIDE the thesis-chart
render path on purpose. ``CHART_RENDER_VERSION`` hashes the raw bytes of
``llm_thesis/chart_render_version.py``, ``llm_thesis/chart_export.py``, and
``viz/charts.py`` — this module IMPORTS from two of them (figure assembly,
PNG metadata scrub) but never edits them, so the thesis Tier-1 cache SHA is
untouched. The kaleido pinning + scrub it imports keep archive renders
byte-deterministic for the same inputs, which is what makes the
zero-stored-pixels render-on-demand policy (`regenerate_chart`) faithful.

STORAGE POLICY (design Appendix C): bytes are stored ONLY for the flagged set
(closed trades now; weekly-sweep names via the follow-up below) — everything
else re-renders from `stock_prices` + `money_flow_snapshots`.

FOLLOW-UP (one commit, after PR #138 qu100-miss-sweep merges): sweep-side
capture — the weekly sweep calls `persist_archive_chart(...,
source=SOURCE_MISS_SWEEP, as_of=<sweep as-of>)` for each flagged name
(missed winners + R-B dodged losers). The renderer + persistence here already
support ``miss_sweep`` end-to-end (tested); only the hook inside
``paper/sweep.py`` is missing because PR #138 is unmerged at the time this
ships — do not block on it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date as date_cls

import pandas as pd
from sqlalchemy import select

from rainier.core.database import get_session
from rainier.core.models import ChartImage, PaperTrade, Stock, StockPrice
from rainier.llm_thesis.chart_export import (
    RENDER_HEIGHT,
    RENDER_SCALE,
    RENDER_WIDTH,
    _strip_png_metadata,
)
from rainier.paper.ingest import canonical_instant, get_qu100_appearances
from rainier.viz.charts import create_static_stock_chart

log = logging.getLogger(__name__)

# chart_images.source values. 'thesis' rows are written by
# llm_thesis/chart_persistence.py and stay OUTSIDE the logical-identity
# contract (as_of_date NULL forever).
SOURCE_THESIS = "thesis"
SOURCE_TRADE_CLOSE = "trade_close"
SOURCE_MISS_SWEEP = "miss_sweep"

# Mirrors LLMThesisConfig.chart_lookback_days — the figure-level default when
# no config is threaded (pure callers/tests).
DEFAULT_CHART_LOOKBACK_DAYS = 120

# Marker/overlay palette. Levels reuse the thesis-path colors (green entry /
# red SL / blue target) so charts read identically across both paths.
_APPEARANCE_COLOR = "#FFD54F"
_LEVEL_STYLE = (
    ("entry_price", "#26a69a", "Entry"),
    ("stop_loss", "#ef5350", "SL"),
    ("target_price", "#42A5F5", "Target"),
)
_CHART_FONT = "DejaVu Sans, sans-serif"  # same bundled-kaleido family


@dataclass(frozen=True)
class TradeOverlay:
    """The booked trade levels/points drawn onto an archive chart."""

    entry_date: date_cls | None
    entry_price: float | None
    stop_loss: float | None
    target_price: float | None
    exit_date: date_cls | None
    exit_price: float | None
    exit_reason: str | None

    @classmethod
    def from_trade(cls, trade: PaperTrade) -> TradeOverlay:
        return cls(
            entry_date=trade.entry_date,
            # Closed trades always carry the booked fill; planned level is the
            # defensive fallback so a malformed row still renders.
            entry_price=trade.entry_price or trade.planned_entry_price,
            stop_loss=trade.stop_loss,
            target_price=trade.target_price,
            exit_date=trade.exit_date,
            exit_price=trade.exit_price,
            exit_reason=trade.exit_reason,
        )


def build_archive_figure(
    symbol: str,
    df: pd.DataFrame,
    *,
    window: int = DEFAULT_CHART_LOOKBACK_DAYS,
    appearances=(),
    trade: TradeOverlay | None = None,
):
    """Assemble the annotated archive figure (no kaleido — pure Plotly).

    Reuses ``create_static_stock_chart`` (UNMODIFIED — it already takes a
    ``bars`` window parameter) for the candles/layout, then decorates:

    * one vertical line + "#rank · M/D" label per QU100 appearance whose
      data_date has a visible bar (dates without a bar, or outside the
      window, are skipped — positional x-axis has nowhere to put them);
    * trade overlay — entry/SL/target horizontal dotted lines across the full
      width, plus arrowed point annotations at (entry_date, entry_price) and
      (exit_date, exit_price) when those dates are visible.

    Deterministic: appearances are drawn sorted by (data_date, rank); no wall
    clock, no random state — same inputs, same figure JSON.
    """
    tail = df.tail(window) if df is not None and not df.empty else df
    fig = create_static_stock_chart(symbol, tail, pattern=None, bars=window)
    if tail is None or tail.empty:
        return fig

    # date -> positional x (the candlestick x-axis is range(len(tail))).
    pos_by_date = {}
    for i, ts in enumerate(tail.index):
        d = ts.date() if hasattr(ts, "date") else ts
        pos_by_date[d] = i

    # --- QU100 appearance markers -----------------------------------------
    drawn = 0
    for app in sorted(appearances, key=lambda a: (a[0], a[1])):
        data_date, rank = app[0], app[1]
        pos = pos_by_date.get(data_date)
        if pos is None:
            continue
        fig.add_shape(
            type="line",
            x0=pos, x1=pos, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=_APPEARANCE_COLOR, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=pos,
            # Alternate two label rows so adjacent markers don't overlap.
            y=1.02 if drawn % 2 == 0 else 0.985,
            xref="x", yref="paper",
            text=f"#{rank} · {data_date.month}/{data_date.day}",
            showarrow=False,
            font=dict(family=_CHART_FONT, size=10, color=_APPEARANCE_COLOR),
        )
        drawn += 1

    # --- Trade overlay ------------------------------------------------------
    if trade is not None:
        x1 = len(tail) - 1
        for field, color, label in _LEVEL_STYLE:
            price = getattr(trade, field)
            if price is None or price <= 0:
                continue
            fig.add_shape(
                type="line",
                x0=0, x1=x1, y0=price, y1=price,
                line=dict(color=color, width=1.5, dash="dot"),
            )
            fig.add_annotation(
                x=x1, y=price, text=f"{label} {price:.2f}",
                showarrow=False, xanchor="right", xshift=-4,
                font=dict(family=_CHART_FONT, size=10, color=color),
            )

        def _point(d: date_cls | None, price: float | None, text: str, color: str):
            if d is None or price is None:
                return
            pos = pos_by_date.get(d)
            if pos is None:
                return
            fig.add_annotation(
                x=pos, y=price, xref="x", yref="y",
                text=text, showarrow=True, arrowhead=2, arrowcolor=color,
                ax=0, ay=-30,
                font=dict(family=_CHART_FONT, size=10, color=color),
            )

        _point(
            trade.entry_date, trade.entry_price,
            f"Entry {trade.entry_price:.2f}" if trade.entry_price else "",
            "#26a69a",
        )
        if trade.exit_price is not None:
            reason = f" ({trade.exit_reason})" if trade.exit_reason else ""
            _point(
                trade.exit_date, trade.exit_price,
                f"Exit {trade.exit_price:.2f}{reason}",
                "#FF8A65",
            )
    return fig


def render_archive_chart_png(
    symbol: str,
    df: pd.DataFrame,
    *,
    window: int = DEFAULT_CHART_LOOKBACK_DAYS,
    appearances=(),
    trade: TradeOverlay | None = None,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
) -> tuple[bytes, str]:
    """Figure → deterministic PNG bytes + sha256 (same pinning as thesis path).

    kaleido engine, pinned scale, post-render metadata scrub (imported from
    ``chart_export`` — the scrub keeps the sha a pure function of the inputs).
    """
    fig = build_archive_figure(
        symbol, df, window=window, appearances=appearances, trade=trade
    )
    raw = fig.to_image(
        format="png", width=width, height=height,
        scale=RENDER_SCALE, engine="kaleido",
    )
    png = _strip_png_metadata(raw)
    return png, hashlib.sha256(png).hexdigest()


def load_daily_bars(
    symbol: str, *, as_of: date_cls, window: int
) -> pd.DataFrame:
    """Last ``window`` usable daily bars for ``symbol`` at-or-before ``as_of``.

    Reads `stock_prices` (canonical 00:00-UTC instants, D9). Only bars with
    full OHLC count (a NULL/partial re-fetch row is not a renderable candle).
    Deterministic: ordered by date; the same DB state always yields the same
    frame — the foundation of the render-on-demand byte-identity contract.
    """
    with get_session() as session:
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
                StockPrice.date <= canonical_instant(as_of),
                StockPrice.open.isnot(None),
                StockPrice.high.isnot(None),
                StockPrice.low.isnot(None),
                StockPrice.close.isnot(None),
            )
            .order_by(StockPrice.date)
        ).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "volume"]
    ).set_index("date")
    return df.tail(window)


def persist_archive_chart(
    *,
    symbol: str,
    as_of: date_cls,
    source: str,
    png_bytes: bytes,
    sha256: str,
    timeframe_days: int,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
) -> int:
    """Persist an archive chart under logical identity (symbol, as_of, source).

    Append-only supersession (TASK-PLAN acceptance 4), ONE transaction:

      identical bytes (current row's sha matches) ─► dedup no-op, same id
      changed bytes ─► 3-step staging:
        1. INSERT the new row with the nullable logical-key columns
           (`as_of_date`/`source`) staged NULL — `symbol` is NOT NULL and set;
        2. point the old current row's `superseded_by` at the new id;
        3. promote: fill the new row's `as_of_date`/`source`.

    The partial unique (symbol, as_of_date, source) WHERE superseded_by IS
    NULL is never violated mid-transaction (the staged row has NULL key
    columns while the old row is still current). Bytes are never overwritten
    or deleted — snapshot payloads keep their original chart id and
    LLM-referenced rows stay valid forever. Archive rows keep `scan_date`
    NULL so a flip-flop re-render can't collide with a superseded row in the
    0004 (symbol, scan_date, sha256) partial unique.

    Returns the CURRENT chart id for the identity.
    """
    if source not in (SOURCE_TRADE_CLOSE, SOURCE_MISS_SWEEP):
        raise ValueError(
            f"archive source must be trade_close/miss_sweep, got {source!r} "
            "(thesis rows go through llm_thesis.chart_persistence)"
        )
    with get_session() as session:
        # Parent stocks row (FK target) — same ensure-pattern as the ingest.
        if (
            session.execute(
                select(Stock.symbol).where(Stock.symbol == symbol)
            ).first()
            is None
        ):
            session.add(Stock(symbol=symbol))
            session.flush()

        current = session.execute(
            select(ChartImage).where(
                ChartImage.symbol == symbol,
                ChartImage.as_of_date == as_of,
                ChartImage.source == source,
                ChartImage.superseded_by.is_(None),
            )
        ).scalar_one_or_none()
        if current is not None and current.sha256 == sha256:
            return int(current.id)

        staged = ChartImage(
            symbol=symbol,
            source=None,  # staged — promoted below, same transaction
            as_of_date=None,
            scan_date=None,  # archive rows live on as_of_date (see docstring)
            image_bytes=png_bytes,
            sha256=sha256,
            file_size_bytes=len(png_bytes),
            width=int(width),
            height=int(height),
            timeframe_days=int(timeframe_days),
        )
        session.add(staged)
        session.flush()  # assigns staged.id
        if current is not None:
            current.superseded_by = staged.id
            session.flush()
        staged.as_of_date = as_of
        staged.source = source
        session.flush()
        return int(staged.id)


def _archive_window(window: int | None) -> int:
    if window is not None:
        return int(window)
    from rainier.core.config import get_settings

    return int(get_settings().llm_thesis.chart_lookback_days)


def capture_trade_close_charts(
    *, as_of: date_cls, window: int | None = None
) -> int:
    """Close-side capture (daily job step (v), unconditional): archive a chart
    for every paper trade CLOSED on ``as_of`` and point its ``chart_id`` at
    the current row.

    Idempotent: a same-tick re-run sha-dedups to the existing row; changed
    inputs supersede append-only and MOVE ``chart_id`` to the new current row
    (snapshot payloads keep the old id). Per-trade failure isolation — one
    bad symbol never starves the rest. Returns the number of trades whose
    chart_id is set/confirmed.
    """
    window = _archive_window(window)
    with get_session() as session:
        trades = session.execute(
            select(PaperTrade.id, PaperTrade.symbol).where(
                PaperTrade.status == "closed",
                PaperTrade.exit_date == as_of,
            )
        ).all()

    captured = 0
    for trade_id, symbol in trades:
        try:
            df = load_daily_bars(symbol, as_of=as_of, window=window)
            if df is None or df.empty:
                log.warning(
                    "close_chart_no_bars symbol=%s as_of=%s", symbol, as_of
                )
                continue
            appearances = get_qu100_appearances(
                symbol, as_of=as_of, window=window
            )
            with get_session() as session:
                trade = session.get(PaperTrade, trade_id)
                overlay = TradeOverlay.from_trade(trade)
            png, sha = render_archive_chart_png(
                symbol, df, window=window, appearances=appearances, trade=overlay
            )
            chart_id = persist_archive_chart(
                symbol=symbol,
                as_of=as_of,
                source=SOURCE_TRADE_CLOSE,
                png_bytes=png,
                sha256=sha,
                timeframe_days=window,
            )
            with get_session() as session:
                trade = session.get(PaperTrade, trade_id)
                trade.chart_id = chart_id
            captured += 1
        except Exception:
            log.exception(
                "close_chart_capture_failed symbol=%s as_of=%s", symbol, as_of
            )
    log.info("close_charts_done as_of=%s captured=%d", as_of, captured)
    return captured


def regenerate_chart(
    symbol: str, *, as_of: date_cls, window: int | None = None
) -> tuple[bytes, str]:
    """Render-on-demand (zero stored pixels): rebuild the archive chart for
    (symbol, as_of) from stored inputs only.

    Reconstructs exactly what the capture path saw — bars ``<= as_of``,
    appearances ``<= as_of``, and the closed trade exited ON ``as_of`` (if
    any) for the overlay — so on unchanged data the PNG is byte-identical to
    a stored trade_close chart for the same identity.
    """
    window = _archive_window(window)
    df = load_daily_bars(symbol, as_of=as_of, window=window)
    if df is None or df.empty:
        raise ValueError(f"no stored daily bars for {symbol} at {as_of}")
    appearances = get_qu100_appearances(symbol, as_of=as_of, window=window)

    overlay: TradeOverlay | None = None
    with get_session() as session:
        trade = session.execute(
            select(PaperTrade).where(
                PaperTrade.symbol == symbol,
                PaperTrade.status == "closed",
                PaperTrade.exit_date == as_of,
            )
        ).scalar_one_or_none()
        if trade is not None:
            overlay = TradeOverlay.from_trade(trade)

    return render_archive_chart_png(
        symbol, df, window=window, appearances=appearances, trade=overlay
    )
