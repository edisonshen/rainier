"""Spec: `rainier thesis collect SYM` writes an agent-ready pack for ANY ticker
(QU capitalflow + daily-thesis evidence + 200-bar chart), with no LLM call."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rainier.cli import cli
from rainier.core.types import MoneyFlowSignal, StockCandidate
from rainier.llm_thesis.schemas import EvidencePack
from rainier.scrapers.qu.parsers import TickerFlow, parse_ticker_bars, parse_ticker_ranks
from rainier.scrapers.qu.scraper import QUScraper, _auth_rejected

SUMMARY = {
    "data": {
        "daily": [
            {"label": "2026-10-16", "long_short": "No dominance", "rank": "5",
             "sentiment": "N", "total": "1717"},
            {"label": "2026-10-15", "long_short": "Long in", "rank": "1",
             "sentiment": "+", "total": "1720"},
        ],
        "weekly": [
            {"label": "2026-10-12 to 2026-10-16", "long_short": "No Dominance",
             "rank": "1", "sentiment": "+", "total": "2567"},
        ],
    },
    "message": "Last updated on 2026-10-16 17:43 Eastern Time",
}
DAILY_PLOT = {"data": [{"near": 4.94, "total": 17.86, "xlabel": "2026-10-05"},
                       {"near": -2.41, "total": 3.11, "xlabel": "2026-10-16"}]}
WEEKLY_PLOT = {"data": [{"near": 1.54, "total": 100, "xlabel": "2026-10-12 ~ 2026-10-16"}]}


def _ohlcv(bars: int = 260) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=bars)
    close = 100 + np.cumsum(np.random.default_rng(7).normal(0, 1, bars))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close,
         "volume": 1_000_000},
        index=idx,
    )


def _candidate(symbol: str = "NVDA") -> StockCandidate:
    return StockCandidate(
        symbol=symbol, rank=0, rank_change=0, long_short="Not in QU100",
        capital_flow_direction="N", sector="Technology", signal_strength=0.1,
        current_price=101.0,
    )


class _FakePage:
    def __init__(self):
        self.urls: list[str] = []

    async def evaluate(self, _js, arg):
        url = arg["url"]
        self.urls.append(url)
        if "mfsummary" in url:
            body = SUMMARY
        elif "frequency=weekly" in url:
            body = WEEKLY_PLOT
        else:
            body = DAILY_PLOT
        return {"status": 200, "body": body, "text_excerpt": ""}


async def _flow() -> TickerFlow:
    scraper = QUScraper.__new__(QUScraper)
    scraper._page = _FakePage()
    return await scraper.fetch_ticker_flow("NVDA", date(2026, 10, 16))


def test_fetch_ticker_flow_parses_capitalflow_endpoints():
    scraper = QUScraper.__new__(QUScraper)
    scraper._page = page = _FakePage()
    flow = asyncio.run(scraper.fetch_ticker_flow("NVDA", date(2026, 10, 16)))

    assert page.urls == [
        "/api/v3/mfsummary?ticker=NVDA&date=2026-10-16",
        "/api/v3/mfplot?ticker=NVDA&date=2026-10-16&frequency=daily",
        "/api/v3/mfplot?ticker=NVDA&date=2026-10-16&frequency=weekly",
    ]
    assert flow.daily_ranks[1].rank == 1 and flow.daily_ranks[1].total == 1720
    assert flow.weekly_bars[0].total == 100.0
    section = flow.prompt_section()
    assert "2026-10-15: 1/1720, Long in, +" in section
    assert "2026-10-16: -2.41 / +3.11" in section


def test_unknown_ticker_renders_as_no_data():
    flow = TickerFlow("ZZZZQ", "x", parse_ticker_ranks([]), [], parse_ticker_bars(None), [])
    assert flow.empty
    assert "no capital-flow data" in flow.prompt_section()


@pytest.mark.parametrize(
    ("result", "rejected"),
    [
        ({"status": 403, "body": None}, True),
        ({"status": 201, "body": {"message": "Not logged in"}}, True),
        ({"status": 200, "body": SUMMARY}, False),
    ],
)
def test_not_logged_in_body_triggers_reauth(result, rejected):
    assert _auth_rejected(result) is rejected


def test_collect_writes_pack_without_llm_or_db_writes(tmp_path):
    pack = EvidencePack(
        symbol="NVDA", scan_date=date.today().isoformat(), session_name="afternoon",
        candidate={"symbol": "NVDA"}, signals={"rank_trajectory": {"days": 3}},
    )

    async def _fake_evidence(*_a, **_k):
        return pack, ["rank_trajectory: up 3 days"]

    async def _fake_flow(*_a, **_k):
        return await _flow()

    out = tmp_path / "pack"
    with (
        patch("rainier.analysis.stock_screener.screen_ticker",
              return_value=(_candidate(), _ohlcv())),
        patch("rainier.cli.thesis._fetch_ticker_flow", side_effect=_fake_flow),
        patch("rainier.llm_thesis.ticker_pack.assemble_evidence", side_effect=_fake_evidence),
        patch("rainier.paper.calibration.load_latest_calibration", return_value=None),
        patch("rainier.paper.reflection.reflection_prompt_section", return_value=""),
        patch("litellm.completion") as llm,
        patch("rainier.llm_thesis.chart_persistence.persist_chart_image") as persist,
    ):
        result = CliRunner().invoke(cli, ["thesis", "collect", "nvda", "--out", str(out)])

    assert result.exit_code == 0, result.output
    llm.assert_not_called()
    persist.assert_not_called()
    prompt = (out / "prompt.md").read_text()
    assert "## System prompt" in prompt and "## User message" in prompt
    assert "rank_trajectory: up 3 days" in prompt
    assert "--- QU capital flow (Last updated on 2026-10-16" in prompt
    assert prompt.rstrip().endswith("Produce the JSON object now.")
    assert (out / "chart.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    evidence = json.loads((out / "evidence.json").read_text())
    assert evidence["qu_capital_flow"]["daily_ranks"][0]["total"] == 1717


def test_collect_continues_when_qu_fetch_fails(tmp_path):
    pack = EvidencePack(symbol="NVDA", scan_date="2026-10-16", session_name="afternoon",
                        candidate={}, signals={})

    async def _fake_evidence(*_a, **_k):
        return pack, []

    async def _boom(*_a, **_k):
        raise RuntimeError("cloudflare")

    with (
        patch("rainier.analysis.stock_screener.screen_ticker",
              return_value=(_candidate(), _ohlcv())),
        patch("rainier.cli.thesis._fetch_ticker_flow", side_effect=_boom),
        patch("rainier.llm_thesis.ticker_pack.assemble_evidence", side_effect=_fake_evidence),
        patch("rainier.paper.calibration.load_latest_calibration", return_value=None),
        patch("rainier.paper.reflection.reflection_prompt_section", return_value=""),
    ):
        result = CliRunner().invoke(cli, ["thesis", "collect", "NVDA", "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "QU capitalflow fetch failed" in result.output
    assert "QU capital flow" not in (tmp_path / "prompt.md").read_text()


def test_screen_ticker_scores_a_ticker_outside_qu100():
    from rainier.analysis import stock_screener as ss

    neutral = MoneyFlowSignal(
        symbol="ZZZ", rank=0, rank_change=0, long_short="Not in QU100",
        capital_flow_direction="N", days_in_top100=0, sector="Unknown",
        industry="Unknown", signal_strength=0.0,
    )
    df = _ohlcv()
    with (
        patch.object(ss, "get_session"),
        patch.object(ss, "_screen_money_flow_as_of", return_value=[]),
        patch.object(ss, "_unranked_signal", return_value=neutral),
        patch.object(ss, "analyze_sectors", return_value=[]),
        patch.object(ss, "_fetch_stock_data", return_value={"ZZZ": df}) as fetch,
    ):
        candidate, history = ss.screen_ticker("zzz")

    fetch.assert_called_once()
    assert fetch.call_args.args[0] == ["ZZZ"]
    assert candidate.symbol == "ZZZ" and candidate.long_short == "Not in QU100"
    assert history is df


def test_screen_ticker_returns_none_when_pattern_detection_fails():
    from rainier.analysis import stock_screener as ss

    with (
        patch.object(ss, "get_session"),
        patch.object(ss, "_screen_money_flow_as_of", return_value=[]),
        patch.object(ss, "_unranked_signal", return_value=MoneyFlowSignal(
            symbol="ZZZ", rank=0, rank_change=0, long_short="Not in QU100",
            capital_flow_direction="N", days_in_top100=0, sector="Unknown",
            industry="Unknown", signal_strength=0.0)),
        patch.object(ss, "analyze_sectors", return_value=[]),
        patch.object(ss, "_fetch_stock_data", return_value={"ZZZ": _ohlcv()}),
        patch.object(ss, "detect_patterns", side_effect=ValueError("bad pivots")),
    ):
        assert ss.screen_ticker("ZZZ") == (None, None)


def test_weekly_only_flow_is_still_rendered():
    flow = TickerFlow("NEWCO", "x", [], [], [], parse_ticker_bars(WEEKLY_PLOT["data"]))
    assert not flow.empty
    assert "+1.54 / +100.00" in flow.prompt_section()


def test_failed_qu_setup_still_closes_the_tab():
    from rainier.cli.thesis import _fetch_ticker_flow

    class _Browser:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

    teardowns = []

    class _Scraper:
        def __init__(self, _browser):
            pass

        async def setup(self):
            raise RuntimeError("cloudflare stuck")

        async def teardown(self):
            teardowns.append(True)

    with (
        patch("rainier.scrapers.browser.BrowserManager", _Browser),
        patch("rainier.scrapers.qu.scraper.QUScraper", _Scraper),
        pytest.raises(RuntimeError, match="cloudflare"),
    ):
        asyncio.run(_fetch_ticker_flow("NVDA", date(2026, 10, 16), "http://x"))
    assert teardowns == [True]
