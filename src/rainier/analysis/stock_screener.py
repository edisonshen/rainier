"""Stock screening pipeline — 3-layer QU100 screener.

Layer 1: Money flow screening from QU100 data
Layer 2: Sector trend analysis + boost
Layer 3: Technical pattern detection (Caisen methodology)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from sqlalchemy import func
from sqlalchemy.orm import Session

from rainier.analysis.sector_analyzer import analyze_sectors, get_sector_boost
from rainier.analysis.stock_patterns import detect_patterns
from rainier.core.config import Settings, StockScreenerConfig, get_settings
from rainier.core.database import get_session
from rainier.core.models import MoneyFlowSnapshot, StockCapitalFlow
from rainier.core.types import (
    MoneyFlowSignal,
    PatternSignal,
    SectorTrend,
    StockCandidate,
    StockScreenResult,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def screen_stocks(settings: Settings | None = None) -> list[StockCandidate]:
    """Run the full 3-layer stock screening pipeline.

    Layer 1: Money flow screening from QU100 data
    Layer 2: Sector trend analysis + boost
    Layer 3: Technical pattern detection on each candidate

    Returns list of StockCandidate sorted by composite score descending.
    """
    if settings is None:
        settings = get_settings()
    config = settings.stock_screener

    with get_session() as session:
        # Layer 1 — money flow screening
        signals = _screen_money_flow(session)
        if not signals:
            log.warning("No QU100 candidates after money flow screening")
            return []
        log.info("Layer 1 complete: %d candidates from money flow", len(signals))

        # Layer 2 — sector analysis + boost
        sector_trends = analyze_sectors(session)
        signals = _apply_sector_boost(signals, sector_trends)
        log.info("Layer 2 complete: sector boost applied")

    # Layer 3 — pattern detection (outside DB session, uses yfinance)
    symbols = [s.symbol for s in signals]
    stock_data = _fetch_stock_data(symbols, config.min_daily_bars)
    log.info(
        "Layer 3: fetched data for %d/%d symbols", len(stock_data), len(symbols)
    )

    # Build signal lookup by symbol
    signal_map = {s.symbol: s for s in signals}

    # Detect patterns and filter to actionable setups
    results: list[StockScreenResult] = []
    for sym, df in stock_data.items():
        signal = signal_map[sym]
        try:
            all_patterns = detect_patterns(sym, df, config)
        except Exception:
            log.exception("Pattern detection failed for %s, skipping", sym)
            continue

        # Filter to patterns actionable from today's price
        patterns = _filter_actionable(all_patterns, df)
        # Best = first actionable (sorted by priority), not highest confidence
        best_pattern = patterns[0] if patterns else None
        best_confidence = best_pattern.confidence if best_pattern else 0.0

        # Composite score
        sector_boost = get_sector_boost(signal.sector, sector_trends)
        composite = (
            config.layer_weight_money_flow * signal.signal_strength
            + config.layer_weight_sector * sector_boost
            + config.layer_weight_pattern * best_confidence
        )

        recommendation = _classify(composite, config)

        result = StockScreenResult(
            symbol=sym,
            name="",
            sector=signal.sector,
            money_flow_score=signal.signal_strength,
            long_short=signal.long_short,
            qu100_rank=signal.rank,
            sector_trend=_sector_direction(signal.sector, sector_trends),
            sector_boost=sector_boost,
            patterns=patterns,
            best_pattern=best_pattern,
            composite_score=round(composite, 4),
            recommendation=recommendation,
            entry_price=best_pattern.entry_price if best_pattern else None,
            stop_loss=best_pattern.stop_loss if best_pattern else None,
            target=best_pattern.target_wave1 if best_pattern else None,
            risk_pct=best_pattern.risk_pct if best_pattern else None,
        )
        results.append(result)

    # Also include candidates that had no yfinance data (money flow only)
    for sym in symbols:
        if sym in stock_data:
            continue
        signal = signal_map[sym]
        sector_boost = get_sector_boost(signal.sector, sector_trends)
        composite = (
            config.layer_weight_money_flow * signal.signal_strength
            + config.layer_weight_sector * sector_boost
        )
        recommendation = _classify(composite, config)
        results.append(
            StockScreenResult(
                symbol=sym,
                name="",
                sector=signal.sector,
                money_flow_score=signal.signal_strength,
                long_short=signal.long_short,
                qu100_rank=signal.rank,
                sector_trend=_sector_direction(signal.sector, sector_trends),
                sector_boost=sector_boost,
                patterns=[],
                best_pattern=None,
                composite_score=round(composite, 4),
                recommendation=recommendation,
            )
        )

    # Sort by composite score descending
    results.sort(key=lambda r: r.composite_score, reverse=True)

    # Build price map for actionability context
    price_map: dict[str, float] = {}
    for sym, df in stock_data.items():
        price_map[sym] = float(df["close"].iloc[-1])
        price_map[f"_len_{sym}"] = float(len(df))

    candidates = [_to_candidate(r, signal_map, price_map) for r in results]
    log.info(
        "Screening complete: %d candidates (%d strong_buy, %d buy, %d watch)",
        len(candidates),
        sum(1 for r in results if r.recommendation == "strong_buy"),
        sum(1 for r in results if r.recommendation == "buy"),
        sum(1 for r in results if r.recommendation == "watch"),
    )
    return candidates


# ---------------------------------------------------------------------------
# Layer 1 — Money flow screening
# ---------------------------------------------------------------------------


def _screen_money_flow(session: Session) -> list[MoneyFlowSignal]:
    """Screen stocks from latest QU100 snapshot + capital flow history.

    Step 1: Get latest 'Long in' stocks from MoneyFlowSnapshot.
    Step 2: Enrich with capital flow direction from StockCapitalFlow.

    Returns list of MoneyFlowSignal sorted by signal_strength descending.
    """
    # Step 1: Latest QU100 snapshot — "Long in" stocks only
    latest_ts = session.query(
        func.max(MoneyFlowSnapshot.captured_at)
    ).scalar()
    if latest_ts is None:
        log.warning("No money flow snapshots in database")
        return []

    # Staleness check: reject data older than 24 hours
    now = datetime.now(timezone.utc)
    ts = latest_ts if latest_ts.tzinfo else latest_ts.replace(tzinfo=timezone.utc)
    age_hours = (now - ts).total_seconds() / 3600
    if age_hours > 24:
        log.warning(
            "Stale QU100 data (%.1f hours old, captured_at=%s) — skipping report",
            age_hours,
            latest_ts,
        )
        return []

    rows = (
        session.query(
            MoneyFlowSnapshot.symbol,
            MoneyFlowSnapshot.rank,
            MoneyFlowSnapshot.daily_change,
            MoneyFlowSnapshot.long_short,
            MoneyFlowSnapshot.sector,
            MoneyFlowSnapshot.industry,
        )
        .filter(
            MoneyFlowSnapshot.ranking_type == "top100",
            MoneyFlowSnapshot.long_short == "Long in",
            MoneyFlowSnapshot.captured_at == latest_ts,
        )
        .order_by(MoneyFlowSnapshot.rank.asc())
        .all()
    )

    if not rows:
        log.warning("No 'Long in' stocks found at %s", latest_ts)
        return []

    # Step 2: Enrich each stock with capital flow data
    signals: list[MoneyFlowSignal] = []
    for row in rows:
        symbol = row.symbol
        rank = row.rank
        daily_change = row.daily_change or 0
        long_short = row.long_short or "Long in"
        sector = row.sector or "Unknown"
        industry = row.industry or "Unknown"

        # Get recent capital flow history
        cf_rows = (
            session.query(
                StockCapitalFlow.flow_date,
                StockCapitalFlow.capital_flow_direction,
                StockCapitalFlow.long_short,
                StockCapitalFlow.rank,
            )
            .filter(
                StockCapitalFlow.symbol == symbol,
                StockCapitalFlow.period_type == "daily",
            )
            .order_by(StockCapitalFlow.flow_date.desc())
            .limit(5)
            .all()
        )

        # Determine capital flow direction from most recent entry
        capital_flow_direction = "N"
        if cf_rows:
            capital_flow_direction = cf_rows[0].capital_flow_direction or "N"

        # Count consecutive "+" days
        days_in_top100 = 0
        for cf in cf_rows:
            if cf.capital_flow_direction == "+":
                days_in_top100 += 1
            else:
                break

        # Compute signal strength
        strength = _compute_money_flow_score(
            long_short=long_short,
            capital_flow_direction=capital_flow_direction,
            rank=rank,
            rank_change=daily_change,
            days_in_top100=days_in_top100,
        )

        signals.append(
            MoneyFlowSignal(
                symbol=symbol,
                rank=rank,
                rank_change=daily_change,
                long_short=long_short,
                capital_flow_direction=capital_flow_direction,
                days_in_top100=days_in_top100,
                sector=sector,
                industry=industry,
                signal_strength=round(strength, 4),
            )
        )

    signals.sort(key=lambda s: s.signal_strength, reverse=True)
    return signals


def _compute_money_flow_score(
    *,
    long_short: str,
    capital_flow_direction: str,
    rank: int,
    rank_change: int,
    days_in_top100: int,
) -> float:
    """Compute money flow score from QU100 metrics.

    Scoring formula:
    - long_short == "Long in" → base 0.5
    - capital_flow_direction == "+" → +0.2
    - rank <= 30 → +0.15
    - rank_change > 0 (improving) → +0.1
    - days_in_top100 >= 3 → +0.05
    """
    score = 0.0

    if long_short == "Long in":
        score += 0.5

    if capital_flow_direction == "+":
        score += 0.2

    if rank <= 30:
        score += 0.15

    if rank_change > 0:
        score += 0.1

    if days_in_top100 >= 3:
        score += 0.05

    return score


# ---------------------------------------------------------------------------
# Layer 2 — Sector boost
# ---------------------------------------------------------------------------


def _apply_sector_boost(
    signals: list[MoneyFlowSignal],
    sector_trends: list[SectorTrend],
) -> list[MoneyFlowSignal]:
    """Add 0.1 to signal_strength for stocks in bullish sectors.

    Returns new list of MoneyFlowSignal (frozen dataclass, so we rebuild).
    """
    boosted: list[MoneyFlowSignal] = []
    for signal in signals:
        boost = get_sector_boost(signal.sector, sector_trends)
        if boost > 0:
            signal = MoneyFlowSignal(
                symbol=signal.symbol,
                rank=signal.rank,
                rank_change=signal.rank_change,
                long_short=signal.long_short,
                capital_flow_direction=signal.capital_flow_direction,
                days_in_top100=signal.days_in_top100,
                sector=signal.sector,
                industry=signal.industry,
                signal_strength=round(signal.signal_strength + boost, 4),
            )
        boosted.append(signal)
    return boosted


# ---------------------------------------------------------------------------
# Layer 3 — Data fetching + pattern detection helpers
# ---------------------------------------------------------------------------


def _fetch_stock_data(
    symbols: list[str], min_bars: int
) -> dict[str, pd.DataFrame]:
    """Batch fetch 6mo daily data for all symbols via yfinance."""
    if not symbols:
        return {}

    try:
        raw = yf.download(
            symbols,
            period="6mo",
            interval="1d",
            group_by="ticker",
            progress=False,
        )
    except Exception:
        log.exception("yfinance batch download failed")
        return {}

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            if len(symbols) == 1:
                df = raw.copy()
            else:
                df = raw[sym].copy()
            df = df.dropna(subset=["Close"])
            if len(df) < min_bars:
                log.debug(
                    "Skipping %s: only %d bars (need %d)",
                    sym, len(df), min_bars,
                )
                continue
            # Normalize column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            result[sym] = df
        except (KeyError, TypeError):
            log.debug("No data returned for %s", sym)
            continue
    return result


def _best_pattern(
    patterns: list[PatternSignal],
) -> PatternSignal | None:
    """Return the pattern with highest confidence, or None."""
    if not patterns:
        return None
    return max(patterns, key=lambda p: p.confidence)


def _filter_actionable(
    patterns: list[PatternSignal],
    df: pd.DataFrame,
    max_bars_since_breakout: int = 10,
    entry_proximity_pct: float = 0.05,
) -> list[PatternSignal]:
    """Filter patterns to only those actionable from today's price.

    A pattern is actionable if:
    - FORMING: price is approaching breakout level (within entry_proximity_pct)
    - CONFIRMED recently: breakout happened within last max_bars_since_breakout bars
      AND price hasn't already reached target or been stopped out
      AND current price is still near entry (within entry_proximity_pct)

    Returns patterns sorted by actionability (forming near breakout first,
    then fresh confirmations).
    """
    if not patterns or df.empty:
        return []

    last_idx = len(df) - 1
    current_price = float(df["close"].iloc[-1])
    actionable: list[tuple[PatternSignal, float]] = []

    for p in patterns:
        is_bullish = p.direction == "bullish"

        # Skip if already stopped out
        if is_bullish and current_price < p.stop_loss:
            continue
        if not is_bullish and current_price > p.stop_loss:
            continue

        # Skip if target already reached
        if is_bullish and current_price >= p.target_wave1:
            continue
        if not is_bullish and current_price <= p.target_wave1:
            continue

        if p.status == "forming":
            # Forming: price must be approaching the entry level
            # Use entry_price as reference (= neckline for most patterns)
            ref_price = p.entry_price if p.entry_price > 0 else p.neckline
            if ref_price <= 0:
                continue
            dist_to_entry = abs(current_price - ref_price) / ref_price
            if dist_to_entry > entry_proximity_pct:
                continue
            # Must be approaching from correct side
            if is_bullish and current_price > ref_price:
                continue  # already past breakout level
            if not is_bullish and current_price < ref_price:
                continue
            # Score: closer to entry = higher priority
            priority = 2.0 - dist_to_entry
            actionable.append((p, priority))

        elif p.status == "confirmed":
            # Confirmed: must be recent AND price near entry
            if p.breakout_idx is None:
                continue
            bars_since = last_idx - p.breakout_idx
            if bars_since > max_bars_since_breakout:
                continue

            dist_to_entry = abs(current_price - p.entry_price) / p.entry_price
            if dist_to_entry > entry_proximity_pct:
                # Price has moved too far from entry — not actionable
                continue

            # Score: fresher = higher, closer to entry = higher
            freshness = 1.0 - (bars_since / max_bars_since_breakout)
            proximity = 1.0 - min(dist_to_entry, 1.0)
            priority = freshness * 0.6 + proximity * 0.4
            actionable.append((p, priority))

    # Sort by priority descending
    actionable.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in actionable]


# ---------------------------------------------------------------------------
# Scoring + classification
# ---------------------------------------------------------------------------


def _classify(composite: float, config: StockScreenerConfig) -> str:
    """Classify composite score into recommendation tier."""
    if composite >= config.strong_buy_threshold:
        return "strong_buy"
    if composite >= config.buy_threshold:
        return "buy"
    if composite >= config.watch_threshold:
        return "watch"
    return "avoid"


def _sector_direction(
    sector: str, sector_trends: list[SectorTrend]
) -> str:
    """Get trend direction string for a sector."""
    for st in sector_trends:
        if st.sector == sector:
            return st.trend_direction
    return "neutral"


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------


def _to_candidate(
    result: StockScreenResult,
    signal_map: dict[str, MoneyFlowSignal],
    price_map: dict[str, float],
) -> StockCandidate:
    """Convert StockScreenResult to StockCandidate for downstream use."""
    signal = signal_map.get(result.symbol)
    rank_change = signal.rank_change if signal else 0
    capital_flow_direction = signal.capital_flow_direction if signal else "N"
    bp = result.best_pattern
    current_price = price_map.get(result.symbol)

    # Compute distance to entry
    dist_to_entry = None
    if current_price and bp and bp.entry_price:
        dist_to_entry = round(
            (current_price - bp.entry_price) / bp.entry_price * 100, 2
        )

    # Bars since breakout
    bars_since = None
    total_bars = price_map.get(f"_len_{result.symbol}")
    if bp and bp.breakout_idx is not None and total_bars:
        bars_since = int(total_bars) - 1 - bp.breakout_idx

    return StockCandidate(
        symbol=result.symbol,
        rank=result.qu100_rank,
        rank_change=rank_change,
        long_short=result.long_short,
        capital_flow_direction=capital_flow_direction,
        sector=result.sector,
        signal_strength=result.composite_score,
        pattern_type=bp.pattern_type if bp else None,
        pattern_direction=bp.direction if bp else None,
        pattern_status=bp.status if bp else None,
        pattern_confidence=bp.confidence if bp else None,
        entry_price=result.entry_price,
        stop_loss=result.stop_loss,
        target_price=result.target,
        rr_ratio=bp.rr_ratio if bp else None,
        volume_confirmed=bp.volume_confirmed if bp else False,
        current_price=round(current_price, 2) if current_price else None,
        distance_to_entry_pct=dist_to_entry,
        bars_since_breakout=bars_since,
    )
