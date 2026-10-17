"""Daily FNG-TQQQ signal orchestration."""

from __future__ import annotations

from datetime import date

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

    qqq_close = price_loader("QQQ", warmup_start(cfg.start))
    fg = load_daily_scores(session_factory=session_factory)

    latest_qqq = _last_date(qqq_close)
    if latest_qqq != today:
        latest_text = latest_qqq.isoformat() if latest_qqq else "no bars"
        raise StaleInputError(
            f"QQQ is stale: latest bar is {latest_text}; expected {today.isoformat()}"
        )
    latest_fg = _last_date(fg)
    if latest_fg != today:
        latest_text = latest_fg.isoformat() if latest_fg else "no observations"
        raise StaleInputError(
            f"F&G is stale: latest observation is {latest_text}; "
            f"expected {today.isoformat()}"
        )

    days = evaluate(fg, qqq_close, cfg)
    if not days:
        raise StaleInputError(
            f"No F&G/QQQ strategy rows available on or after {cfg.start.isoformat()}"
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
        "start": cfg.start.isoformat(),
    }
    with session_factory() as session:
        values = {
            "strategy": "fng-tqqq",
            "signal_date": day.date,
            "fg": day.fg,
            "qqq_close": day.qqq_close,
            "qqq_sma": day.qqq_sma,
            "holding": day.holding,
            "armed": day.armed,
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
