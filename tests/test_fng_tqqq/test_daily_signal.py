from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from rainier.core.config import DiscordConfig, FngTqqqConfig
from rainier.core.models import FearGreedIndex, StrategySignal
from rainier.core.types import StrategyDay
from rainier.pipeline.fng_tqqq_signal import (
    StaleInputError,
    run_daily_signal,
    run_stop_check,
)
from rainier.signals.fng_tqqq import evaluate

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "fng_tqqq" / "history.csv"
TODAY = date(2021, 1, 27)


class FakeCalendar:
    def __init__(
        self,
        is_session: bool = True,
        *,
        previous_session: date | None = None,
        close=None,
    ):
        self.session = is_session
        self.previous_session_date = previous_session
        self.close_value = close

    def is_session(self, session) -> bool:
        return self.session

    def next_session(self, session):
        return pd.Timestamp(session) + pd.offsets.BDay(1)

    def previous_session(self, session):
        return pd.Timestamp(self.previous_session_date)

    def session_close(self, session):
        return self.close_value


class FakeResponse:
    def raise_for_status(self):
        return None


class CapturingHttp:
    def __init__(self):
        self.posts = []

    def post(self, url, *, json, timeout):
        self.posts.append((url, json, timeout))
        return FakeResponse()


@pytest.fixture
def history():
    return pd.read_csv(FIXTURE, parse_dates=["date"]).set_index("date")


@pytest.fixture
def sqlite_db():
    engine = create_engine("sqlite://")
    FearGreedIndex.__table__.create(engine)
    StrategySignal.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


def _insert_fixture_scores(factory, history, through: date, duplicate_latest: bool = False):
    with factory() as session:
        for timestamp, row in history.iterrows():
            day = timestamp.date()
            if day > through or pd.isna(row.fg):
                continue
            score = float(row.fg)
            if duplicate_latest and day == TODAY:
                session.add(
                    FearGreedIndex(
                        date=day,
                        observed_at=datetime(2021, 1, 27, 14, tzinfo=timezone.utc),
                        score=5.0,
                        raw={},
                        source_version="backfill",
                    )
                )
                score = 29.0
                observed_at = datetime(2021, 1, 27, 15, tzinfo=timezone.utc)
            else:
                observed_at = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
            session.add(
                FearGreedIndex(
                    date=day,
                    observed_at=observed_at,
                    score=score,
                    raw={},
                    source_version="backfill",
                )
            )
        session.commit()


def _price_loader(history, through: date):
    def load(symbol, start):
        assert symbol in {"QQQ", "TQQQ"}
        assert start < date(2021, 1, 22)
        return history.loc[
            history.index.date <= through, f"{symbol.lower()}_close"
        ]

    return load


def _strategy_days(history, cfg=None):
    return evaluate(
        history["fg"].dropna(),
        history["qqq_close"],
        history["tqqq_close"],
        cfg or FngTqqqConfig(),
    )


def _first_holding_day(history):
    return next(
        day
        for day in _strategy_days(history)
        if day.holding and day.action == "HOLD" and day.stop_price is not None
    )


def _run_stop_check(
    history,
    day,
    now,
    live_price,
    poster,
    *,
    session_close,
    is_session=True,
    price_time=None,
    price_loader=None,
    fg_loader=None,
    live_price_fn=None,
):
    calendar = FakeCalendar(
        is_session=is_session,
        previous_session=day.date,
        close=session_close,
    )

    def load_prices(symbol, start):
        return history.loc[
            history.index.date <= day.date, f"{symbol.lower()}_close"
        ]

    return run_stop_check(
        cfg=FngTqqqConfig(),
        fg_loader=fg_loader
        or (lambda session_factory: history.loc[
            history.index.date <= day.date, "fg"
        ].dropna()),
        price_loader=price_loader or load_prices,
        live_price_fn=live_price_fn
        or (
            lambda symbol: (
                live_price,
                price_time or now - timedelta(minutes=1),
            )
        ),
        poster=poster,
        calendar=calendar,
        clock=lambda: now,
    )


def _poster(http):
    from rainier.alerts.discord import send_strategy_signal

    config = DiscordConfig(
        enabled=True,
        webhook_url="https://discord.example/fallback",
        strategy_webhook_url="https://discord.example/strategy",
    )

    def post(day, cfg, next_session):
        return send_strategy_signal(day, cfg, next_session, config)

    return post


def test_fixture_buy_is_persisted_posted_and_same_day_rerun_upserts(
    sqlite_db, history, monkeypatch
):
    import rainier.alerts.discord as discord

    http = CapturingHttp()
    monkeypatch.setattr(discord, "httpx", http)
    _insert_fixture_scores(sqlite_db, history, TODAY, duplicate_latest=True)
    cfg = FngTqqqConfig()

    day = run_daily_signal(
        today=TODAY,
        cfg=cfg,
        session_factory=sqlite_db,
        price_loader=_price_loader(history, TODAY),
        poster=_poster(http),
        calendar=FakeCalendar(),
    )

    assert day is not None
    assert day.action == "BUY"
    assert day.date == TODAY
    with sqlite_db() as session:
        row = session.scalar(select(StrategySignal))
        assert row.strategy == "fng-tqqq"
        assert row.signal_date == TODAY
        assert row.action == "BUY"
        assert row.fg == 29.0
        assert row.qqq_close == pytest.approx(float(history.loc["2021-01-27", "qqq_close"]))

    assert len(http.posts) == 1
    url, payload, timeout = http.posts[0]
    embed = payload["embeds"][0]
    assert url == "https://discord.example/strategy"
    assert timeout == 10
    assert embed["title"] == (
        "FNG-TQQQ: BUY TQQQ at close on 2021-01-28 (MOC)"
    )
    assert "29.00 (buy below 30)" in embed["fields"][0]["value"]

    run_daily_signal(
        today=TODAY,
        cfg=cfg,
        session_factory=sqlite_db,
        price_loader=_price_loader(history, TODAY),
        poster=_poster(http),
        calendar=FakeCalendar(),
    )
    with sqlite_db() as session:
        count = session.scalar(select(func.count()).select_from(StrategySignal))
        assert count == 1
    assert len(http.posts) == 2


def test_daily_stop_signal_persists_stop_state(sqlite_db, history):
    stop_day = next(day for day in _strategy_days(history) if day.action == "STOP")
    _insert_fixture_scores(sqlite_db, history, stop_day.date)
    posted = []

    day = run_daily_signal(
        today=stop_day.date,
        cfg=FngTqqqConfig(),
        session_factory=sqlite_db,
        price_loader=_price_loader(history, stop_day.date),
        poster=lambda *args: posted.append(args),
        calendar=FakeCalendar(),
    )

    assert day is not None
    assert day.action == "STOP"
    assert day.holding is False
    assert day.locked is True
    assert day.entry_price is not None
    assert day.stop_price is not None
    with sqlite_db() as session:
        row = session.scalar(select(StrategySignal))
        assert row.action == "STOP"
        assert row.holding is False
        assert row.locked is True
        assert row.entry_price == pytest.approx(day.entry_price)
        assert row.stop_price == pytest.approx(day.stop_price)
        assert row.regime_sma == pytest.approx(day.regime_sma)
        assert row.bear == day.bear
    assert len(posted) == 1


def test_non_session_returns_without_loading_or_posting(sqlite_db):
    def forbidden_loader(*args):
        pytest.fail("prices must not be loaded on a non-session")

    def forbidden_poster(*args):
        pytest.fail("no Discord post is sent on a non-session")

    result = run_daily_signal(
        today=date(2021, 1, 30),
        cfg=FngTqqqConfig(),
        session_factory=sqlite_db,
        price_loader=forbidden_loader,
        poster=forbidden_poster,
        calendar=FakeCalendar(is_session=False),
    )
    assert result is None


def test_stale_fear_greed_raises_and_cli_exits_nonzero(
    sqlite_db, history, monkeypatch
):
    from click.testing import CliRunner

    import rainier.cli.strategy as cli_strategy
    from rainier.cli import cli

    _insert_fixture_scores(sqlite_db, history, TODAY)
    with pytest.raises(StaleInputError, match="F&G is stale"):
        run_daily_signal(
            today=date(2021, 1, 28),
            cfg=FngTqqqConfig(),
            session_factory=sqlite_db,
            price_loader=_price_loader(history, date(2021, 1, 28)),
            poster=lambda *args: pytest.fail("stale inputs must not post"),
            calendar=FakeCalendar(),
        )

    def fail_stale(**kwargs):
        raise StaleInputError("F&G is stale: latest observation is 2021-01-27")

    monkeypatch.setattr(cli_strategy, "_legacy_db_for_config", lambda ctx: nullcontext())
    monkeypatch.setattr(cli_strategy, "run_daily_signal", fail_stale)
    result = CliRunner().invoke(
        cli,
        ["strategy", "fng-tqqq", "signal", "--no-discord"],
    )
    assert result.exit_code != 0
    assert "F&G is stale" in result.output


def test_cli_discord_disabled_fails_after_persisting_signal(
    sqlite_db, history, monkeypatch
):
    from click.testing import CliRunner

    import rainier.cli as cli_module
    import rainier.cli.strategy as cli_strategy
    from rainier.cli import cli
    from rainier.core import database
    from rainier.core.config import load_settings

    class FixedDate(date):
        @classmethod
        def today(cls):
            return TODAY

    settings = load_settings(ROOT / "config" / "settings.yaml")
    settings.alerts.discord.enabled = False
    monkeypatch.setattr(cli_module, "load_settings", lambda path: settings)
    monkeypatch.setattr(
        cli_strategy, "_legacy_db_for_config", lambda ctx: nullcontext()
    )
    monkeypatch.setattr(cli_strategy, "date", FixedDate)
    monkeypatch.setattr(cli_strategy.xcals, "get_calendar", lambda _: FakeCalendar())
    monkeypatch.setattr(
        cli_strategy, "load_daily_close", _price_loader(history, TODAY)
    )
    monkeypatch.setattr(database, "get_session", sqlite_db)
    _insert_fixture_scores(sqlite_db, history, TODAY)

    result = CliRunner().invoke(cli, ["strategy", "fng-tqqq", "signal"])

    assert result.exit_code != 0
    assert (
        "Discord strategy webhook not configured/enabled; "
        "signal BUY for 2021-01-27 was persisted but not posted"
    ) in result.output
    with sqlite_db() as session:
        row = session.scalar(select(StrategySignal))
        assert row is not None
        assert row.action == "BUY"
        assert row.signal_date == TODAY


def test_stop_check_posts_when_price_is_below_stop(history):
    day = _first_holding_day(history)
    close = datetime(2021, 5, 3, 13, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    now = close - timedelta(minutes=2)
    posted = []

    result = _run_stop_check(
        history,
        day,
        now,
        day.stop_price - 0.01,
        lambda *args: posted.append(args),
        session_close=close,
    )

    assert result is True
    assert len(posted) == 1
    assert posted[0][0] == day


def test_stop_check_above_stop_does_not_post(history):
    day = _first_holding_day(history)
    close = datetime(2021, 5, 3, 13, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    now = close - timedelta(minutes=2)
    posted = []

    result = _run_stop_check(
        history,
        day,
        now,
        day.stop_price + 0.01,
        lambda *args: posted.append(args),
        session_close=close,
    )

    assert result is False
    assert posted == []


def test_stop_check_outside_window_and_non_session_are_noops(history):
    day = _first_holding_day(history)
    close = datetime(2021, 5, 3, 13, 0, tzinfo=ZoneInfo("America/Los_Angeles"))

    def forbidden(*args, **kwargs):
        pytest.fail("inputs must not be loaded")

    def poster(*args):
        pytest.fail("no stop alert should be posted")

    outside = _run_stop_check(
        history,
        day,
        close - timedelta(minutes=6),
        day.stop_price - 1,
        poster,
        session_close=close,
        fg_loader=forbidden,
        price_loader=forbidden,
    )
    non_session = _run_stop_check(
        history,
        day,
        close - timedelta(minutes=2),
        day.stop_price - 1,
        poster,
        session_close=close,
        is_session=False,
        fg_loader=forbidden,
        price_loader=forbidden,
    )

    assert outside is False
    assert non_session is False


def test_stop_check_uses_half_day_close_window(history):
    day = _first_holding_day(history)
    close = datetime(2021, 11, 26, 10, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    now = datetime(2021, 11, 26, 9, 58, tzinfo=ZoneInfo("America/Los_Angeles"))
    posted = []

    result = _run_stop_check(
        history,
        day,
        now,
        day.stop_price - 0.01,
        lambda *args: posted.append(args),
        session_close=close,
    )

    assert result is True
    assert len(posted) == 1


def test_stop_check_skips_buy_and_flat_days(history):
    days = _strategy_days(history)
    candidates = [
        next(day for day in days if day.action == "BUY"),
        next(day for day in days if not day.holding and day.action == "WAIT"),
    ]
    close = datetime(2021, 5, 3, 13, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    now = close - timedelta(minutes=2)

    for day in candidates:
        result = _run_stop_check(
            history,
            day,
            now,
            0.0,
            lambda *args: pytest.fail("not holding, no alert should be posted"),
            session_close=close,
            live_price_fn=lambda symbol: pytest.fail("live price is not needed"),
        )
        assert result is False


def test_stop_check_rejects_stale_live_price(history):
    day = _first_holding_day(history)
    close = datetime(2021, 5, 3, 13, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    now = close - timedelta(minutes=2)

    with pytest.raises(StaleInputError, match="Live TQQQ price is stale"):
        _run_stop_check(
            history,
            day,
            now,
            day.stop_price - 0.01,
            lambda *args: pytest.fail("stale price must not post"),
            session_close=close,
            price_time=now - timedelta(minutes=6),
        )


def test_stop_check_disabled_discord_exits_nonzero(history, monkeypatch):
    from click.testing import CliRunner

    import rainier.cli as cli_module
    import rainier.cli.strategy as cli_strategy
    import rainier.data.fear_greed as fg_data
    import rainier.data.prices as prices
    from rainier.cli import cli
    from rainier.core.config import load_settings

    day = _first_holding_day(history)
    close = datetime(2021, 5, 3, 13, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    now = close - timedelta(minutes=2)
    settings = load_settings(ROOT / "config" / "settings.yaml")
    settings.alerts.discord.enabled = False

    class FixedDatetime:
        @staticmethod
        def now(tz=None):
            return now.astimezone(tz)

    monkeypatch.setattr(cli_module, "load_settings", lambda path: settings)
    monkeypatch.setattr(
        cli_strategy, "_legacy_db_for_config", lambda ctx: nullcontext()
    )
    monkeypatch.setattr(cli_strategy, "datetime", FixedDatetime)
    monkeypatch.setattr(
        cli_strategy.xcals,
        "get_calendar",
        lambda _: FakeCalendar(previous_session=day.date, close=close),
    )
    monkeypatch.setattr(
        fg_data,
        "load_daily_scores",
        lambda session_factory: history["fg"].dropna(),
    )

    def load_prices(symbol, start):
        return history[f"{symbol.lower()}_close"]

    monkeypatch.setattr(prices, "load_daily_close", load_prices)
    monkeypatch.setattr(
        prices, "load_last_price", lambda symbol: (day.stop_price - 0.01, now - timedelta(minutes=1))
    )

    result = CliRunner().invoke(cli, ["strategy", "fng-tqqq", "stop-check"])

    assert result.exit_code != 0
    assert "Discord strategy webhook not configured/enabled" in result.output


def test_strategy_discord_hold_wait_and_stop_messages(monkeypatch):
    import rainier.alerts.discord as discord

    http = CapturingHttp()
    monkeypatch.setattr(discord, "httpx", http)
    cfg = FngTqqqConfig()
    config = DiscordConfig(enabled=True, strategy_webhook_url="https://discord.example")

    holding = StrategyDay(
        date=TODAY,
        fg=25.0,
        qqq_close=300.0,
        qqq_sma=295.0,
        holding=True,
        armed=True,
        action="HOLD",
        entry_price=100.0,
        stop_price=90.0,
        tqqq_close=95.0,
    )
    waiting = StrategyDay(
        date=TODAY,
        fg=25.0,
        qqq_close=280.0,
        qqq_sma=295.0,
        holding=False,
        armed=False,
        action="WAIT",
        regime_sma=285.0,
        bear=True,
    )
    stopped = StrategyDay(
        date=TODAY,
        fg=25.0,
        qqq_close=280.0,
        qqq_sma=295.0,
        holding=False,
        armed=False,
        action="STOP",
        entry_price=100.0,
        stop_price=90.0,
        tqqq_close=89.0,
        locked=True,
    )

    discord.send_strategy_signal(holding, cfg, date(2021, 1, 28), config)
    discord.send_strategy_signal(waiting, cfg, date(2021, 1, 28), config)
    discord.send_strategy_signal(stopped, cfg, date(2021, 1, 28), config)

    hold_embed = http.posts[0][1]["embeds"][0]
    hold_fields = {field["name"]: field["value"] for field in hold_embed["fields"]}
    assert hold_fields["Entry price"] == "100.00"
    assert hold_fields["Stop price"] == "90.00"

    wait_embed = http.posts[1][1]["embeds"][0]
    wait_fields = {field["name"]: field["value"] for field in wait_embed["fields"]}
    assert wait_fields["Buy status"] == "QQQ below SMA180 (bear filter)"

    stop_embed = http.posts[2][1]["embeds"][0]
    assert stop_embed["color"] == 0xE74C3C
    assert "close 89.00 ≤ stop 90.00" in stop_embed["description"]
    assert "if it wasn't sold, sell at the next session" in stop_embed["description"]

    alert_time = datetime(
        2021, 1, 27, 12, 58, tzinfo=ZoneInfo("America/Los_Angeles")
    )
    assert discord.send_strategy_stop_alert(stopped, 89.0, alert_time, config)
    alert_embed = http.posts[3][1]["embeds"][0]
    assert alert_embed["title"] == (
        "FNG-TQQQ: STOP — sell TQQQ now before the close"
    )
    assert alert_embed["color"] == 0xE74C3C
    alert_fields = {field["name"]: field["value"] for field in alert_embed["fields"]}
    assert alert_fields["Live TQQQ price"] == "89.00"
    assert alert_fields["Stop price"] == "90.00"
    assert alert_fields["Entry price"] == "100.00"


def test_discord_failed_post_raises(monkeypatch):
    import rainier.alerts.discord as discord

    class FailedResponse:
        def raise_for_status(self):
            raise RuntimeError("HTTP 500")

    class FailedHttp:
        def post(self, *args, **kwargs):
            return FailedResponse()

    monkeypatch.setattr(discord, "httpx", FailedHttp())
    with pytest.raises(RuntimeError, match="HTTP 500"):
        discord.send_strategy_signal(
            StrategyDay(
                date=TODAY,
                fg=29.0,
                qqq_close=308.93,
                qqq_sma=300.0,
                holding=False,
                armed=False,
                action="BUY",
            ),
            FngTqqqConfig(),
            date(2021, 1, 28),
            DiscordConfig(webhook_url="https://discord.example", enabled=True),
        )
