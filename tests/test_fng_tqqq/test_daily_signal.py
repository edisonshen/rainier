from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from rainier.core.config import DiscordConfig, FngTqqqConfig
from rainier.core.models import FearGreedIndex, StrategySignal
from rainier.core.types import StrategyDay
from rainier.pipeline.fng_tqqq_signal import StaleInputError, run_daily_signal

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "fng_tqqq" / "history.csv"
TODAY = date(2021, 1, 27)


class FakeCalendar:
    def __init__(self, is_session: bool = True):
        self.session = is_session

    def is_session(self, session) -> bool:
        return self.session

    def next_session(self, session):
        return pd.Timestamp(session) + pd.offsets.BDay(1)


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
        assert symbol == "QQQ"
        assert start < date(2021, 1, 22)
        return history.loc[history.index.date <= through, "qqq_close"]

    return load


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
