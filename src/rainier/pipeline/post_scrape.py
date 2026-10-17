"""Post-scrape pipeline shared by the CLI and the APScheduler service.

This is the single source of truth for what happens AFTER a successful
QU100 scrape, regardless of how that scrape was triggered:

    screen_stocks      -> 3-layer screener (returns candidates + ohlcv)
    persist top-50     -> shadow-validation dataset (PR2 carry-over)
    LLM pick + thesis  -> only on sessions in `enabled_sessions`: the LLM picks
                          top_n of the pool, then theses them (before_thesis), or
                          theses the whole pool and ranks the theses to pick
                          the top_n to post (after_thesis)
    Discord top-20     -> rich per-ticker embeds when theses dict is non-empty

Before this module existed, the CLI path (``rainier scrape qu``, called
from cron) only ran the regular top-20 screener post — the LLM thesis
embeds shipped via PR4/PR5 only fired from the ``rainier run`` daemon.
Centralising the post-scrape steps here means the cron-driven path and
the long-running scheduler emit identical Discord output.

The function is synchronous: the CLI calls it directly (the CLI's own
async wrapper is one-shot, so blocking is fine), and the scheduler wraps
the whole call in a single ``asyncio.to_thread`` rather than per-step
threading.
"""

from __future__ import annotations

from datetime import date
from typing import Any, NamedTuple

import structlog

from rainier.alerts.discord import send_stock_candidates, send_thesis_failure_alert
from rainier.analysis.stock_screener import screen_stocks
from rainier.core.config import Settings
from rainier.core.types import StockCandidate
from rainier.llm_thesis.persistence import persist_screened_stocks
from rainier.llm_thesis.selection import (
    ThesisSelection,
    rank_theses,
    ranking_reserve_usd,
    select_thesis_targets,
)
from rainier.llm_thesis.service import ThesisBatch, compute_theses_and_persist

log = structlog.get_logger()

_PERSIST_WINDOW = 50


def scan_window(all_candidates: list[StockCandidate], settings: Settings) -> list[StockCandidate]:
    """Candidates persisted as screened rows: top-50, widened to the LLM pool
    so every thesis links to a screened row."""
    return all_candidates[: max(_PERSIST_WINDOW, settings.llm_thesis.selection_pool_size)]


class ThesisScan(NamedTuple):
    """Outcome of one scan's LLM step."""

    # Candidates whose thesis embeds post to Discord, best first.
    posted: list[StockCandidate]
    theses: dict[str, dict[str, Any]]
    failures: dict[str, str]
    requested: int
    # The pick (before_thesis) or the ranking of finished theses (after_thesis).
    selection: ThesisSelection


def _screener_order(candidates: list[StockCandidate], n: int, exc: Exception) -> ThesisSelection:
    return ThesisSelection(candidates[:n], {}, "screener", 0.0, str(exc))


def _compute_or_fail(
    targets: list[StockCandidate], ohlcv_by_symbol: dict, **kwargs: Any
) -> ThesisBatch:
    try:
        return compute_theses_and_persist(targets, ohlcv_by_symbol, **kwargs)
    except Exception as exc:
        log.error(
            "compute_theses_unexpected_failure",
            session=kwargs.get("session_name"),
            error=str(exc),
        )
        return ThesisBatch({}, {c.symbol: f"compute_theses crashed: {exc}" for c in targets})


def pick_and_thesis(
    scan_candidates: list[StockCandidate],
    ohlcv_by_symbol: dict,
    *,
    scan_date: date,
    session_name: str,
    settings: Settings,
    top_n: int | None = None,
) -> ThesisScan:
    """The LLM step of a scan, per ``llm_thesis.selection_stage``.

    before_thesis: the LLM picks top_n of the pool and only those get a thesis.
    after_thesis: the whole pool gets a thesis, then the LLM ranks the finished
    theses and the top_n post. Either LLM pick falls back to screener order.
    """
    cfg = settings.llm_thesis
    n = cfg.top_n if top_n is None else top_n
    compute_kwargs = {
        "scan_date": scan_date, "session_name": session_name, "settings": settings,
    }

    if cfg.selection_stage == "after_thesis":
        targets = scan_candidates[: cfg.selection_pool_size]
        # Reserve the ranking call's budget up front so it stays under the cap.
        batch = _compute_or_fail(
            targets,
            ohlcv_by_symbol,
            **compute_kwargs,
            already_spent_usd=ranking_reserve_usd(len(targets), settings, top_n=n),
        )
        try:
            ranking = rank_theses(targets, batch.theses, settings, top_n=n)
        except Exception as exc:
            thesised = [c for c in targets if c.symbol in batch.theses]
            ranking = _screener_order(thesised, n, exc)
        for symbol, reason in ranking.reasons.items():
            batch.theses[symbol]["_selection_reason"] = reason
        return ThesisScan(ranking.targets, batch.theses, batch.failures, len(targets), ranking)

    try:
        selection = select_thesis_targets(scan_candidates, settings, top_n=n)
    except Exception as exc:
        selection = _screener_order(scan_candidates, n, exc)
    batch = _compute_or_fail(
        selection.targets,
        ohlcv_by_symbol,
        **compute_kwargs,
        already_spent_usd=selection.cost_usd,
        selection_reasons=selection.reasons,
    )
    return ThesisScan(
        selection.targets, batch.theses, batch.failures, len(selection.targets), selection
    )


def run_post_scrape_pipeline(
    settings: Settings, session_name: str, scan_date: date | None = None,
) -> None:
    """Execute the post-scrape pipeline for one session.

    Steps (mirrors the pre-extraction service.py block verbatim):

    1. ``screen_stocks(settings)`` — returns ``(all_candidates, ohlcv_by_symbol)``.
    2. ``persist_screened_stocks(top-50)`` — write a DB row per candidate so the
       30-day shadow validation has an unbiased dataset. Top-20 alone biases
       per-rank correlation toward the upper tail. Failures are logged and
       swallowed — persistence must not block the Discord post.
    3. If ``settings.llm_thesis.enabled`` AND ``session_name`` is in
       ``settings.llm_thesis.enabled_sessions`` AND there are candidates,
       ``pick_and_thesis`` runs the LLM pick + theses per
       ``llm_thesis.selection_stage`` (screener order on selector failure).
       Failures fall back to an empty ``theses`` dict so the regular top-20
       post still goes out.
    4. ``send_stock_candidates(top-20, theses or None, dashboard_base_url)`` —
       Discord display set is always top-20; passing ``theses=None`` triggers
       the legacy top-20-only embed; passing a non-empty dict adds the
       per-ticker thesis embeds shipped in PR5.

    Args:
        settings: Already-loaded ``Settings``. Caller is responsible for
            calling ``load_settings_fresh()`` if it wants hot-reloaded YAML.
        session_name: ``"morning" | "midday" | "afternoon" | "close"``.
        scan_date: the TRADING day these derived artifacts belong to. Defaults to
            ``date.today()`` for the live scheduled/cron path (the scrape and the
            pipeline run on the same trading day). ``recover`` passes the RECOVERED
            day explicitly so a post-midnight or cross-timezone replay stamps
            ``ScreenedStockRecord`` / thesis rows with the day the data is from,
            not the wall-clock date the pipeline happened to run (Codex P1).

    Returns:
        None. Logs structured events at each step boundary.
    """
    log.info("post_scrape_pipeline_starting", session=session_name)

    # 1. Screener — pure function, returns the inputs for everything below.
    all_candidates, ohlcv_by_symbol = screen_stocks(settings)

    # Windows on the same sorted list:
    #   - top-50 (or the LLM pool, if wider) -> persistence (shadow dataset)
    #   - top-20 -> Discord display
    #   - the LLM pool -> LLM pick + thesis (expensive; cost cap)
    scan_candidates = scan_window(all_candidates, settings)
    candidates = all_candidates[:20]

    if scan_date is None:
        scan_date = date.today()
    log.info(
        "post_scrape_screener_done",
        session=session_name,
        total_candidates=len(all_candidates),
        persist_window=len(scan_candidates),
        discord_window=len(candidates),
    )

    # 2. Persist top-50. Wrapped in try/except so a DB blip doesn't block the
    # Discord post — losing a row of telemetry is recoverable; losing the
    # operator's alert isn't.
    try:
        persist_screened_stocks(
            scan_candidates,
            scan_date=scan_date,
            session_name=session_name,
        )
    except Exception as exc:
        log.error(
            "persist_screened_stocks_failed",
            session=session_name,
            error=str(exc),
        )

    # 3. LLM pick + thesis — gated by enabled flag + session allowlist +
    # non-empty candidates.
    theses: dict[str, dict] = {}
    llm_gate_open = (
        settings.llm_thesis.enabled
        and session_name in settings.llm_thesis.enabled_sessions
        and bool(candidates)
    )
    log.info(
        "post_scrape_llm_gate",
        session=session_name,
        gate_open=llm_gate_open,
        thesis_enabled=settings.llm_thesis.enabled,
        session_in_allowlist=session_name in settings.llm_thesis.enabled_sessions,
        has_candidates=bool(candidates),
    )
    thesis_failures: dict[str, str] = {}
    thesis_targets: list = []
    if llm_gate_open:
        scan = pick_and_thesis(
            scan_candidates,
            ohlcv_by_symbol,
            scan_date=scan_date,
            session_name=session_name,
            settings=settings,
        )
        theses, thesis_failures, thesis_targets = scan.theses, scan.failures, scan.posted
        log.info(
            "post_scrape_thesis_selection",
            session=session_name,
            stage=settings.llm_thesis.selection_stage,
            source=scan.selection.source,
            picks=[c.symbol for c in thesis_targets],
            theses=len(theses),
            cost_usd=round(scan.selection.cost_usd, 4),
            error=scan.selection.error,
        )
        if thesis_failures:
            log.error(
                "post_scrape_theses_failed",
                session=session_name,
                requested=scan.requested,
                failures=thesis_failures,
            )
            send_thesis_failure_alert(
                settings.alerts.discord,
                session=session_name,
                requested=scan.requested,
                failures=thesis_failures,
            )

    # 4. Discord. ``theses or None`` preserves the legacy top-20-only path
    # when theses is empty (gate disabled, or compute failed). dashboard URL
    # is forwarded so the PR5 per-ticker embeds can deep-link into Streamlit.
    # ``session=session_name`` restores the session-label suffix on the
    # summary embed title that the pre-extraction CLI used (via
    # _build_payloads(candidates, session=session)) — without it, the four
    # daily scans become indistinguishable in the channel.
    send_result = send_stock_candidates(
        candidates,
        settings.alerts.discord,
        theses=theses or None,
        dashboard_base_url=settings.llm_thesis.dashboard_base_url,
        session=session_name,
        thesis_candidates=thesis_targets,
    )
    # discord_sent reflects what actually LANDED, not len(candidates). The
    # summary embed (payload idx 0) lists ALL candidates, so summary_ok is the
    # "did the report show up" signal: it stays len(candidates) on a partial
    # failure where a later detail payload dropped but the table landed, and
    # goes 0 only when the summary itself failed. discord_payloads_failed>0
    # still flags partial drops. A silent dead-webhook now reads discord_sent=0
    # instead of a misleading "sent 20" while the channel stayed empty.
    log.info(
        "post_scrape_pipeline_done",
        session=session_name,
        discord_sent=len(candidates) if send_result.summary_ok else 0,
        discord_payloads_ok=send_result.candidate_payloads_ok,
        discord_payloads_failed=send_result.candidate_payloads_failed,
        theses_attached=len(theses),
        theses_generation_failed=len(thesis_failures),
        theses_sent=send_result.thesis_ok,
        theses_failed=send_result.thesis_failed,
    )
