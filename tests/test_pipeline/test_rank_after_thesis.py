"""Spec: the afternoon scan theses every "Long in" name, then Claude picks 10 to post.

Scenario (selection_stage: after_thesis): the 12:45 PT afternoon scan screens
~40 "Long in" names. Every one gets an Opus thesis through a single Anthropic
Message Batch (half price), every thesis persists and feeds paper trading, then
one text-only Claude call ranks the finished theses and only its top 10 post to
Discord, best first, each with the ranking reason. A ranking outage falls back
to screener order; the close scan runs no LLM at all.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from rainier.llm_thesis.schemas import EvidencePack
from tests.test_llm_thesis.test_close_reuse_and_batch import (
    MODEL,
    SCAN_DATE,
    FakeAnthropicBatches,
    _sync_completion,
)
from tests.test_pipeline.test_post_scrape import _candidates, _make_settings

N_CANDIDATES = 40


def _settings():
    s = _make_settings(enabled_sessions=("afternoon",), llm_selection_enabled=True)
    s.anthropic_api_key = "k"
    cfg = s.llm_thesis
    cfg.model, cfg.max_usd_per_scan, cfg.batch_poll_seconds = MODEL, 6.0, 0
    cfg.batch_sessions = ["afternoon"]
    cfg.selection_stage, cfg.top_n, cfg.selection_pool_size = "after_thesis", 10, 100
    return s


RANKED = ["S031", "S004", "S017", "BOGUS", "s022", "S004", "S000", "S039", "S012", "S025"]


def _ranking_reply(picks: list[str]) -> dict:
    body = json.dumps({"picks": [{"symbol": s, "reason": f"{s} best setup"} for s in picks]})
    return {
        "choices": [{"message": {"content": body}}],
        "usage": {"prompt_tokens": 12000, "completion_tokens": 600},
    }


def _run_afternoon(session: str = "afternoon", *, ranking_fails: bool = False):
    """Real pipeline + thesis service + batch client + Discord renderer; only the
    screener, DB, chart render and the network edges are faked."""
    candidates = _candidates(N_CANDIDATES)
    ranking_calls: list[dict] = []

    def completion(**kwargs):
        if "finished trade theses" in kwargs["messages"][0]["content"]:
            ranking_calls.append(kwargs)
            if ranking_fails:
                raise RuntimeError("credit balance is too low")
            return _ranking_reply(RANKED)
        return _sync_completion(**kwargs)

    async def assemble(candidate, *_args):
        pack = EvidencePack(symbol=candidate.symbol, scan_date=SCAN_DATE.isoformat(),
                            session_name=session, candidate={"rank": candidate.rank})
        return pack, []

    persisted_ids = iter(range(1000, 2000))
    embeds: list[dict] = []
    api = FakeAnthropicBatches()
    with (
        patch("rainier.pipeline.post_scrape.screen_stocks",
              return_value=(candidates, {c.symbol: MagicMock() for c in candidates})),
        patch("rainier.pipeline.post_scrape.persist_screened_stocks") as persist_screened,
        patch("rainier.llm_thesis.service._tier1_lookup", return_value=None),
        patch("rainier.llm_thesis.service.assemble_evidence", side_effect=assemble),
        patch("rainier.llm_thesis.service.build_user_message",
              side_effect=lambda **kw: f"{kw['symbol']} evidence"),
        patch("rainier.llm_thesis.service._persist_thesis",
              side_effect=lambda **_: next(persisted_ids)),
        patch("rainier.llm_thesis.service.update_with_thesis"),
        patch("rainier.llm_thesis.service.get_session"),
        patch("rainier.llm_thesis.chart_export.render_chart_png", return_value=(b"PNG", "sha")),
        patch("rainier.llm_thesis.chart_persistence.persist_chart_image", return_value=None),
        patch("rainier.paper.calibration.load_latest_calibration", return_value=None),
        patch("rainier.paper.reflection.reflection_prompt_section", return_value=""),
        patch("rainier.paper.positions.create_positions_for_theses") as paper,
        patch("rainier.paper.positions.create_shadow_positions_for_theses"),
        patch("rainier.llm_thesis.batch.httpx", api),
        patch("litellm.completion", side_effect=completion) as llm,
        patch("rainier.alerts.discord.httpx.post") as summary_post,
        patch("rainier.alerts.discord._post_thesis_embed",
              side_effect=lambda **kw: embeds.append(kw["embed"])),
        patch("rainier.pipeline.post_scrape.send_thesis_failure_alert") as failure_alert,
    ):
        from rainier.pipeline.post_scrape import run_post_scrape_pipeline

        run_post_scrape_pipeline(_settings(), session, scan_date=SCAN_DATE)

    return {
        "api": api, "llm": llm, "ranking_calls": ranking_calls, "embeds": embeds,
        "paper": paper, "persist_screened": persist_screened,
        "summary_post": summary_post, "failure_alert": failure_alert,
    }


def _symbol(embed: dict) -> str:
    return embed["title"].split(" · ")[1]


def _field(embed: dict, name: str) -> str | None:
    return next((f["value"] for f in embed["fields"] if f["name"] == name), None)


def test_afternoon_theses_all_candidates_in_one_batch_and_posts_ranked_top_10():
    run = _run_afternoon()

    # Every candidate gets a screened row and a thesis, in one half-price batch.
    assert len(run["persist_screened"].call_args.args[0]) == N_CANDIDATES
    assert [len(r) for r in run["api"].submitted] == [N_CANDIDATES]
    assert len(run["ranking_calls"]) == 1 and run["llm"].call_count == 1
    run["failure_alert"].assert_not_called()

    # The ranker sees every finished thesis, not just the screener rows.
    table = run["ranking_calls"][0]["messages"][1]["content"]
    assert table.count("thesis setup_long quality 7/10") == N_CANDIDATES
    assert "S039 radar." in table

    # Exactly 10 thesis embeds, in ranking order; bogus/duplicate picks are
    # dropped and the shortfall filled in screener order.
    posted = [_symbol(e) for e in run["embeds"]]
    assert posted == ["S031", "S004", "S017", "S022", "S000", "S039", "S012", "S025",
                      "S001", "S002"]
    assert _field(run["embeds"][0], "PICKED BY LLM") == "S031 best setup"
    assert _field(run["embeds"][-1], "PICKED BY LLM") is None

    # Paper trading gets every thesis, not just the 10 posted.
    paper_theses = run["paper"].call_args.args[0]
    assert len(paper_theses) == N_CANDIDATES
    assert {t["verdict"] for t in paper_theses} == {"setup_long"}

    # The stock-channel top-20 summary still goes out.
    run["summary_post"].assert_called()


def test_ranking_outage_posts_screener_top_10_and_keeps_every_thesis():
    run = _run_afternoon(ranking_fails=True)

    assert [len(r) for r in run["api"].submitted] == [N_CANDIDATES]
    assert [_symbol(e) for e in run["embeds"]] == [f"S{i:03d}" for i in range(10)]
    assert len(run["paper"].call_args.args[0]) == N_CANDIDATES


def test_close_scan_runs_no_llm():
    run = _run_afternoon("close")

    assert run["api"].submitted == [] and run["llm"].call_count == 0
    assert run["embeds"] == [] and run["paper"].call_count == 0
    run["summary_post"].assert_called()


def test_thesis_budget_leaves_room_for_the_ranking_call():
    from rainier.llm_thesis.selection import ranking_reserve_usd
    from rainier.llm_thesis.service import ThesisBatch
    from rainier.pipeline.post_scrape import pick_and_thesis

    settings = _settings()
    candidates = _candidates(N_CANDIDATES)
    with (
        patch("rainier.pipeline.post_scrape.compute_theses_and_persist",
              return_value=ThesisBatch({}, {})) as compute,
        patch("rainier.pipeline.post_scrape.rank_theses") as rank,
    ):
        pick_and_thesis(candidates, {}, scan_date=SCAN_DATE, session_name="afternoon",
                        settings=settings)

    reserve = compute.call_args.kwargs["already_spent_usd"]
    assert reserve == ranking_reserve_usd(N_CANDIDATES, settings) > 0
    assert reserve < settings.llm_thesis.max_usd_per_scan / 10
    rank.assert_called_once()
