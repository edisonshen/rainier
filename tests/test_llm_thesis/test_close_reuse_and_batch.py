"""Spec: the close scan reuses unchanged afternoon theses and batches the rest.

Scenario: the afternoon scan already wrote theses. At close, a name whose
pattern, levels, price (<=2%) and QU rank (<=10) are unchanged reuses the
afternoon thesis with no LLM call; every other name is analyzed through
Anthropic's Message Batches API at half price. A batch that outlasts its
timeout is cancelled; the replies it already finished (and billed) are kept
and only the cancelled calls run synchronously, so the close post lands.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from rainier.core.config import DiscordConfig, LLMThesisConfig, Settings
from rainier.core.types import StockCandidate
from rainier.llm_thesis.schemas import EvidencePack
from rainier.llm_thesis.service import (
    _estimate_cost_usd,
    compute_theses_and_persist,
    thesis_reuse_basis,
)

SCAN_DATE = date(2026, 10, 14)
MODEL = "claude-opus-5-5"
PROMPT_TOKENS, COMPLETION_TOKENS = 9000, 6000


def _settings(**overrides) -> Settings:
    s = Settings(database_url="postgresql://test:test@localhost/test", anthropic_api_key="k")
    s.llm_thesis = LLMThesisConfig(
        model=MODEL, max_usd_per_scan=3.5, batch_poll_seconds=0, **overrides
    )
    return s


def _candidate(symbol: str, *, price: float = 100.0, rank: int = 10,
               status: str = "confirmed", stop: float = 95.0) -> StockCandidate:
    return StockCandidate(
        symbol=symbol, rank=rank, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
        pattern_type="w_bottom", pattern_status=status, entry_price=101.0,
        stop_loss=stop, target_price=120.0, current_price=price,
    )


def _thesis_json(symbol: str) -> str:
    return json.dumps({
        "verdict": "setup_long", "setup_quality": 7, "llm_confidence": 7,
        "paragraph_radar": f"{symbol} radar.", "paragraph_evidence": "Evidence.",
        "paragraph_invalidation": "Invalidation.", "risks": [], "watch_items": [],
        "evidence_used": [], "signals_used": [], "patterns_in_chart_not_in_indicators": "none",
    })


def _afternoon_row(candidate: StockCandidate, record_id: int) -> tuple[int, dict]:
    return record_id, {
        **json.loads(_thesis_json(candidate.symbol)),
        "_session_name": "afternoon",
        "_reuse_basis": thesis_reuse_basis(candidate),
    }


class FakeAnthropicBatches:
    """Message Batches API double: records submissions, answers from `reply`."""

    def __init__(self, reply=_thesis_json, *, ends: bool = True, poll_fails: bool = False,
                 results_failures: int = 0, finished_before_cancel: set[str] | None = None,
                 drain_poll_failures: int = 0, cancel_rejections: int = 0):
        self.reply = reply
        self.ends = ends
        # When set, a cancel ends the batch: these symbols succeeded, the rest are canceled.
        self.finished_before_cancel = finished_before_cancel
        self.poll_fails = poll_fails
        self.results_failures = results_failures
        self.drain_poll_failures = drain_poll_failures
        self.cancel_rejections = cancel_rejections
        self.submitted: list[list[dict]] = []
        self.cancelled: list[str] = []

    @staticmethod
    def _resp(payload=None, text=""):
        resp = MagicMock()
        resp.json.return_value = payload
        resp.text = text
        return resp

    def post(self, url, headers=None, json=None, timeout=None):
        if url.endswith("/cancel"):
            if self.cancel_rejections:
                self.cancel_rejections -= 1
                raise RuntimeError("500 from cancel endpoint")
            self.cancelled.append(url)
            return self._resp({})
        self.submitted.append(json["requests"])
        return self._resp({"id": f"b{len(self.submitted)}", "processing_status": "in_progress"})

    def get(self, url, headers=None, timeout=None, follow_redirects=False):
        batch_id = url.rsplit("/", 2)[-2] if url.endswith("/results") else url.rsplit("/", 1)[-1]
        if not url.endswith("/results"):
            if self.poll_fails:
                raise RuntimeError("503 from batch status endpoint")
            if self.cancelled and self.drain_poll_failures:
                self.drain_poll_failures -= 1
                raise RuntimeError("503 from batch status endpoint")
            cancel_landed = self.cancelled and self.finished_before_cancel is not None
            status = "ended" if self.ends or cancel_landed else "in_progress"
            return self._resp({"id": batch_id, "processing_status": status,
                               "results_url": f"https://x/batches/{batch_id}/results"})
        if self.results_failures:
            self.results_failures -= 1
            raise RuntimeError("502 downloading batch results")
        requests = self.submitted[int(batch_id[1:]) - 1]
        lines = []
        for req in requests:
            symbol = req["params"]["messages"][0]["content"][0]["text"].split()[0]
            if self.cancelled and symbol not in (self.finished_before_cancel or set()):
                lines.append(json_dumps({"custom_id": req["custom_id"],
                                         "result": {"type": "canceled"}}))
                continue
            lines.append(json_dumps({
                "custom_id": req["custom_id"],
                "result": {"type": "succeeded", "message": {
                    "content": [{"type": "text", "text": self.reply(symbol)}],
                    "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS},
                }},
            }))
        return self._resp(text="\n".join(lines))


json_dumps = json.dumps


def _sync_completion(**kwargs):
    symbol = kwargs["messages"][1]["content"][0]["text"].split()[0]
    return {
        "choices": [{"message": {"content": _thesis_json(symbol)}}],
        "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS},
    }


def _run_close_scan(
    candidates, *, settings, afternoon_rows=None, close_rows=None, batch_api=None
):
    """Real close-scan thesis pipeline; DB, chart render and network are faked."""
    rows_by_session = {"afternoon": afternoon_rows or {}, "close": close_rows or {}}
    persisted: list[dict] = []

    def tier1(symbol, scan_date, prompt_version, *, session_name, **_):
        return rows_by_session.get(session_name, {}).get(symbol)

    async def assemble(candidate, *_args):
        pack = EvidencePack(symbol=candidate.symbol, scan_date=SCAN_DATE.isoformat(),
                            session_name="close", candidate={"rank": candidate.rank})
        return pack, []

    def persist(**kwargs):
        persisted.append(kwargs)
        return 1000 + len(persisted)

    batch_api = batch_api or FakeAnthropicBatches()
    with (
        patch("rainier.llm_thesis.service._tier1_lookup", side_effect=tier1),
        patch("rainier.llm_thesis.service.assemble_evidence", side_effect=assemble),
        patch("rainier.llm_thesis.service.build_user_message",
              side_effect=lambda **kw: f"{kw['symbol']} evidence"),
        patch("rainier.llm_thesis.service._persist_thesis", side_effect=persist),
        patch("rainier.llm_thesis.service.update_with_thesis"),
        patch("rainier.llm_thesis.service.get_session"),
        patch("rainier.llm_thesis.chart_export.render_chart_png",
              return_value=(b"PNG", "sha")),
        patch("rainier.llm_thesis.chart_persistence.persist_chart_image", return_value=None),
        patch("rainier.paper.calibration.load_latest_calibration", return_value=None),
        patch("rainier.paper.reflection.reflection_prompt_section", return_value=""),
        patch("rainier.paper.positions.create_positions_for_theses"),
        patch("rainier.paper.positions.create_shadow_positions_for_theses"),
        patch("rainier.llm_thesis.batch.httpx", batch_api),
        patch("litellm.completion", side_effect=_sync_completion) as sync_llm,
    ):
        batch = compute_theses_and_persist(
            candidates, {c.symbol: MagicMock() for c in candidates},
            scan_date=SCAN_DATE, session_name="close", settings=settings,
        )
    return batch, persisted, batch_api, sync_llm


def _full_cost() -> float:
    return _estimate_cost_usd(PROMPT_TOKENS, COMPLETION_TOKENS, MODEL)


def test_close_reuses_unchanged_afternoon_thesis_and_batches_the_rest():
    unchanged = _candidate("AAA", price=101.5, rank=14)  # +1.5%, rank +4
    afternoon = {"AAA": _afternoon_row(_candidate("AAA"), 77),
                 "BBB": _afternoon_row(_candidate("BBB", status="forming"), 78)}
    candidates = [unchanged, _candidate("BBB"), _candidate("CCC")]

    batch, persisted, api, sync_llm = _run_close_scan(
        candidates, settings=_settings(), afternoon_rows=afternoon
    )

    assert set(batch.theses) == {"AAA", "BBB", "CCC"} and not batch.failures
    assert batch.theses["AAA"]["_reused_from_session"] == "afternoon"
    assert batch.theses["AAA"]["_thesis_id"] == 77
    assert "_reused_from_session" not in batch.theses["BBB"]

    # One batch with just the two names that needed a fresh analysis, sent as a
    # native Messages request with adaptive thinking and the chart image.
    assert len(api.submitted) == 1
    params = [r["params"] for r in api.submitted[0]]
    assert sorted(p["messages"][0]["content"][0]["text"] for p in params) == [
        "BBB evidence", "CCC evidence"]
    assert params[0]["model"] == MODEL and params[0]["thinking"] == {"type": "adaptive"}
    assert params[0]["messages"][0]["content"][1]["type"] == "image"
    sync_llm.assert_not_called()

    # Batch results are billed at half price and stamp the reuse basis.
    by_symbol = {p["symbol"]: p for p in persisted}
    assert sorted(by_symbol) == ["BBB", "CCC"]
    assert by_symbol["BBB"]["cost_usd"] == pytest.approx(_full_cost() / 2)
    assert by_symbol["BBB"]["reuse_basis"] == thesis_reuse_basis(_candidate("BBB"))


@pytest.mark.parametrize(
    "close_candidate",
    [
        pytest.param(_candidate("AAA", price=103.0), id="price-moved-3pct"),
        pytest.param(_candidate("AAA", rank=25), id="rank-moved-15"),
        pytest.param(_candidate("AAA", stop=96.0), id="stop-changed"),
        pytest.param(_candidate("AAA", status="forming"), id="pattern-status-flipped"),
    ],
)
def test_close_reanalyzes_when_evidence_changed(close_candidate):
    batch, persisted, api, _ = _run_close_scan(
        [close_candidate], settings=_settings(),
        afternoon_rows={"AAA": _afternoon_row(_candidate("AAA"), 77)},
    )
    assert "_reused_from_session" not in batch.theses["AAA"]
    assert len(api.submitted) == 1 and [p["symbol"] for p in persisted] == ["AAA"]


def test_invalid_batch_reply_is_retried_in_a_second_batch_round():
    replies = {"BBB": iter(["not json", _thesis_json("BBB")])}
    api = FakeAnthropicBatches(
        reply=lambda s: next(replies[s]) if s in replies else _thesis_json(s)
    )
    batch, _, api, sync_llm = _run_close_scan(
        [_candidate("AAA"), _candidate("BBB")], settings=_settings(), batch_api=api
    )
    assert set(batch.theses) == {"AAA", "BBB"}
    assert [len(round_) for round_ in api.submitted] == [2, 1]
    sync_llm.assert_not_called()


def test_batch_timeout_keeps_finished_replies_and_syncs_only_the_cancelled():
    # A 45-name batch outlasted the timeout with 39 already done; those were billed
    # at batch price, then all 45 were paid again at full price via sync.
    api = FakeAnthropicBatches(ends=False, finished_before_cancel={"AAA", "BBB"})
    batch, persisted, api, sync_llm = _run_close_scan(
        [_candidate("AAA"), _candidate("BBB"), _candidate("CCC")],
        settings=_settings(batch_timeout_minutes=0.0001), batch_api=api,
    )
    assert set(batch.theses) == {"AAA", "BBB", "CCC"} and not batch.failures
    assert len(api.cancelled) == 1
    assert sync_llm.call_count == 1
    assert sorted(p["cost_usd"] for p in persisted) == pytest.approx(
        [_full_cost() / 2, _full_cost() / 2, _full_cost()]
    )


@pytest.mark.parametrize("fault", ["drain_poll_failures", "cancel_rejections"])
def test_batch_timeout_survives_a_transient_cancel_or_drain_error(fault):
    api = FakeAnthropicBatches(ends=False, finished_before_cancel={"AAA", "BBB"},
                               **{fault: 1})
    batch, persisted, api, sync_llm = _run_close_scan(
        [_candidate("AAA"), _candidate("BBB"), _candidate("CCC")],
        settings=_settings(batch_timeout_minutes=0.0001), batch_api=api,
    )
    assert set(batch.theses) == {"AAA", "BBB", "CCC"} and not batch.failures
    assert sync_llm.call_count == 1  # only the cancelled CCC is paid again


def test_batch_whose_cancel_never_lands_falls_back_to_full_price_sync_calls():
    api = FakeAnthropicBatches(ends=False)
    with patch("rainier.llm_thesis.batch.CANCEL_DRAIN_TIMEOUT_S", 0):
        batch, persisted, api, sync_llm = _run_close_scan(
            [_candidate("AAA"), _candidate("BBB")],
            settings=_settings(batch_timeout_minutes=0.0001), batch_api=api,
        )
    assert set(batch.theses) == {"AAA", "BBB"} and not batch.failures
    assert len(api.cancelled) == 1
    assert sync_llm.call_count == 2
    assert persisted[0]["cost_usd"] == pytest.approx(_full_cost())


def test_reused_thesis_is_labelled_in_discord():
    from rainier.alerts.discord import send_stock_candidates

    candidate = _candidate("AAA")
    thesis = {**json.loads(_thesis_json("AAA")), "_reused_from_session": "afternoon"}
    embeds: list[dict] = []
    with (
        patch("rainier.alerts.discord.httpx.post"),
        patch("rainier.alerts.discord._post_thesis_embed",
              side_effect=lambda **kw: embeds.append(kw["embed"])),
    ):
        send_stock_candidates(
            [candidate], DiscordConfig(webhook_url="https://x/y", enabled=True),
            theses={"AAA": thesis}, thesis_candidates=[candidate],
        )
    fields = {f["name"]: f["value"] for f in embeds[0]["fields"]}
    assert fields["REUSED"].startswith("afternoon thesis")


def test_close_rerun_keeps_its_own_close_thesis_instead_of_the_afternoon_one():
    candidate = _candidate("AAA")
    close_id, close_output = _afternoon_row(candidate, 91)
    close_output["_session_name"] = "close"

    batch, persisted, api, sync_llm = _run_close_scan(
        [candidate], settings=_settings(),
        afternoon_rows={"AAA": _afternoon_row(candidate, 77)},
        close_rows={"AAA": (close_id, close_output)},
    )

    assert batch.theses["AAA"]["_thesis_id"] == 91
    assert "_reused_from_session" not in batch.theses["AAA"]
    assert api.submitted == [] and persisted == []
    sync_llm.assert_not_called()


def test_batch_poll_error_cancels_before_falling_back_to_sync_calls():
    api = FakeAnthropicBatches(poll_fails=True)
    batch, _, api, sync_llm = _run_close_scan(
        [_candidate("AAA"), _candidate("BBB")], settings=_settings(), batch_api=api
    )
    assert set(batch.theses) == {"AAA", "BBB"} and not batch.failures
    assert len(api.cancelled) == 1  # the original batch can't also be billed
    assert sync_llm.call_count == 2


def test_ended_batch_results_download_is_retried_not_paid_twice():
    api = FakeAnthropicBatches(results_failures=1)
    batch, persisted, api, sync_llm = _run_close_scan(
        [_candidate("AAA"), _candidate("BBB")], settings=_settings(), batch_api=api
    )
    assert set(batch.theses) == {"AAA", "BBB"} and not batch.failures
    assert api.cancelled == []
    sync_llm.assert_not_called()
    assert persisted[0]["cost_usd"] == pytest.approx(_full_cost() / 2)
