"""Daily FNG-TQQQ signal orchestration."""

from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from rainier.core.config import FngTqqqConfig
from rainier.core.database import get_session
from rainier.core.models import StrategySignal
from rainier.core.types import StrategyDay
from rainier.data.fear_greed import load_daily_scores
from rainier.data.prices import warmup_start
from rainier.signals.fng_tqqq import evaluate

log = structlog.get_logger()


_UPSERT_INSERTS = {"postgresql": pg_insert, "sqlite": sqlite_insert}


class StaleInputError(RuntimeError):
    """Raised when daily strategy inputs have not reached the requested session."""


def _last_date(series: pd.Series) -> date | None:
    if series.empty:
        return None
    dates = pd.DatetimeIndex(pd.to_datetime(series.index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.max().date()


def _through_session(series: pd.Series, session: date) -> pd.Series:
    result = series.copy()
    dates = pd.DatetimeIndex(pd.to_datetime(result.index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    result.index = dates.normalize()
    result = result.sort_index()
    return result.loc[result.index <= pd.Timestamp(session)]


def _warmup_sessions(cfg: FngTqqqConfig) -> int:
    return max(cfg.sma_window, cfg.regime_sma_window or 0) + 20


def preview_daily_signal(
    *,
    today: date,
    cfg: FngTqqqConfig,
    session_factory=get_session,
    price_loader,
    calendar,
) -> tuple[StrategyDay, date] | None:
    """Load and evaluate today's inputs without persisting or posting."""
    if not calendar.is_session(today.isoformat()):
        log.info("fng_tqqq_non_session", date=today.isoformat())
        return None

    price_start = warmup_start(cfg.start, _warmup_sessions(cfg))
    qqq_close = price_loader("QQQ", price_start)
    tqqq_close = price_loader("TQQQ", price_start)
    fg = load_daily_scores(session_factory=session_factory)

    latest_qqq = _last_date(qqq_close)
    if latest_qqq != today:
        latest_text = latest_qqq.isoformat() if latest_qqq else "no bars"
        raise StaleInputError(
            f"QQQ is stale: latest bar is {latest_text}; expected {today.isoformat()}"
        )
    latest_tqqq = _last_date(tqqq_close)
    if latest_tqqq != today:
        latest_text = latest_tqqq.isoformat() if latest_tqqq else "no bars"
        raise StaleInputError(
            f"TQQQ is stale: latest bar is {latest_text}; expected {today.isoformat()}"
        )
    latest_fg = _last_date(fg)
    if latest_fg != today:
        latest_text = latest_fg.isoformat() if latest_fg else "no observations"
        raise StaleInputError(
            f"F&G is stale: latest observation is {latest_text}; "
            f"expected {today.isoformat()}"
        )

    days = evaluate(fg, qqq_close, tqqq_close, cfg)
    if not days:
        raise StaleInputError(
            f"No F&G/QQQ/TQQQ strategy rows available on or after {cfg.start.isoformat()}"
        )
    next_session_value = calendar.next_session(today.isoformat())
    next_session = pd.Timestamp(next_session_value).date()
    return days[-1], next_session


def run_daily_signal(
    *,
    today: date,
    cfg: FngTqqqConfig,
    session_factory=get_session,
    price_loader,
    poster,
    calendar,
) -> StrategyDay | None:
    """Persist and post today's signal after validating session input freshness."""
    prepared = preview_daily_signal(
        today=today,
        cfg=cfg,
        session_factory=session_factory,
        price_loader=price_loader,
        calendar=calendar,
    )
    if prepared is None:
        return None
    day, next_session = prepared
    params = {
        "buy_below": cfg.buy_below,
        "sma_window": cfg.sma_window,
        "regime_sma_window": cfg.regime_sma_window,
        "stop_loss_pct": cfg.stop_loss_pct,
        "start": cfg.start.isoformat(),
    }
    with session_factory() as session:
        values = {
            "strategy": "fng-tqqq",
            "signal_date": day.date,
            "fg": day.fg,
            "qqq_close": day.qqq_close,
            "qqq_sma": day.qqq_sma,
            "entry_price": day.entry_price,
            "stop_price": day.stop_price,
            "regime_sma": day.regime_sma,
            "holding": day.holding,
            "armed": day.armed,
            "bear": day.bear,
            "locked": day.locked,
            "action": day.action,
            "params": params,
        }
        update_values = {
            key: value for key, value in values.items()
            if key not in {"strategy", "signal_date"}
        }
        insert = _UPSERT_INSERTS[session.get_bind().dialect.name]
        session.execute(
            insert(StrategySignal)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["strategy", "signal_date"],
                set_=update_values,
            )
        )
        session.commit()

    poster(day, cfg, next_session)
    log.info(
        "fng_tqqq_signal_complete",
        date=day.date.isoformat(),
        action=day.action,
        next_session=next_session.isoformat(),
    )
    return day


def run_stop_check(
    *,
    cfg: FngTqqqConfig,
    session_factory=get_session,
    fg_loader=load_daily_scores,
    price_loader,
    live_price_fn,
    poster,
    calendar,
    clock,
) -> bool:
    """Check the live TQQQ price during the final five minutes of a session."""
    now = pd.Timestamp(clock())
    if now.tzinfo is None:
        raise StaleInputError("Stop-check clock must return a timezone-aware datetime")
    now_utc = now.tz_convert("UTC")
    session_date = now.tz_convert(ZoneInfo("America/New_York")).date()
    if not calendar.is_session(session_date.isoformat()):
        log.info("fng_tqqq_stop_check_non_session", date=session_date.isoformat())
        return False

    close = pd.Timestamp(calendar.session_close(session_date.isoformat()))
    if close.tzinfo is None:
        close = close.tz_localize("UTC")
    else:
        close = close.tz_convert("UTC")
    window_start = close - pd.Timedelta(minutes=5)
    if not window_start <= now_utc < close:
        log.info(
            "fng_tqqq_stop_check_outside_window",
            date=session_date.isoformat(),
            now=now.isoformat(),
            session_close=close.isoformat(),
        )
        return False

    previous_session = pd.Timestamp(
        calendar.previous_session(session_date.isoformat())
    ).date()
    price_start = warmup_start(cfg.start, _warmup_sessions(cfg))
    fg = _through_session(
        fg_loader(session_factory=session_factory), previous_session
    )
    qqq_close = _through_session(
        price_loader("QQQ", price_start), previous_session
    )
    tqqq_close = _through_session(
        price_loader("TQQQ", price_start), previous_session
    )
    for name, series in (
        ("F&G", fg),
        ("QQQ", qqq_close),
        ("TQQQ", tqqq_close),
    ):
        latest = _last_date(series)
        if latest != previous_session:
            latest_text = latest.isoformat() if latest else "no data"
            raise StaleInputError(
                f"{name} is stale for stop-check: latest bar is {latest_text}; "
                f"expected previous session {previous_session.isoformat()}"
            )

    days = evaluate(fg, qqq_close, tqqq_close, cfg)
    if not days or days[-1].date != previous_session:
        raise StaleInputError(
            "No complete strategy inputs for previous session "
            f"{previous_session.isoformat()}"
        )
    day = days[-1]
    if (
        not day.holding
        or day.action in {"BUY", "SELL"}
        or day.stop_price is None
    ):
        log.info(
            "fng_tqqq_stop_check_not_applicable",
            date=day.date.isoformat(),
            action=day.action,
            holding=day.holding,
        )
        return False

    live_price, price_time = live_price_fn("TQQQ")
    price_timestamp = pd.Timestamp(price_time)
    if price_timestamp.tzinfo is None:
        raise StaleInputError("Live TQQQ price timestamp must be timezone-aware")
    age = now_utc - price_timestamp.tz_convert("UTC")
    if age > timedelta(minutes=5):
        raise StaleInputError(
            f"Live TQQQ price is stale: {age.total_seconds():.0f}s old "
            "(maximum 300s)"
        )

    if live_price <= day.stop_price:
        poster(day, float(live_price), price_timestamp.to_pydatetime())
        log.warning(
            "fng_tqqq_stop_check_triggered",
            date=day.date.isoformat(),
            price=float(live_price),
            stop_price=day.stop_price,
            entry_price=day.entry_price,
        )
        return True

    log.info(
        "fng_tqqq_stop_check_clear",
        date=day.date.isoformat(),
        price=float(live_price),
        stop_price=day.stop_price,
    )
    return False
