"""Post-scrape pipeline shared by the CLI and the APScheduler service.

This is the single source of truth for what happens AFTER a successful
QU100 scrape, regardless of how that scrape was triggered:

    screen_stocks      -> 3-layer screener (returns candidates + ohlcv)
    persist top-50     -> shadow-validation dataset (PR2 carry-over)
    LLM thesis top-5   -> only on sessions in `enabled_sessions`
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

import structlog

from rainier.alerts.discord import send_stock_candidates
from rainier.analysis.stock_screener import screen_stocks
from rainier.core.config import Settings
from rainier.llm_thesis.persistence import persist_screened_stocks
from rainier.llm_thesis.service import compute_theses_and_persist

log = structlog.get_logger()


def run_post_scrape_pipeline(settings: Settings, session_name: str) -> None:
    """Execute the post-scrape pipeline for one session.

    Steps (mirrors the pre-extraction service.py block verbatim):

    1. ``screen_stocks(settings)`` — returns ``(all_candidates, ohlcv_by_symbol)``.
    2. ``persist_screened_stocks(top-50)`` — write a DB row per candidate so the
       30-day shadow validation has an unbiased dataset. Top-20 alone biases
       per-rank correlation toward the upper tail. Failures are logged and
       swallowed — persistence must not block the Discord post.
    3. If ``settings.llm_thesis.enabled`` AND ``session_name`` is in
       ``settings.llm_thesis.enabled_sessions`` AND there are candidates,
       call ``compute_theses_and_persist`` on the top-5. Failures fall back
       to an empty ``theses`` dict so the regular top-20 post still goes out.
    4. ``send_stock_candidates(top-20, theses or None, dashboard_base_url)`` —
       Discord display set is always top-20; passing ``theses=None`` triggers
       the legacy top-20-only embed; passing a non-empty dict adds the
       per-ticker thesis embeds shipped in PR5.

    Args:
        settings: Already-loaded ``Settings``. Caller is responsible for
            calling ``load_settings_fresh()`` if it wants hot-reloaded YAML.
        session_name: ``"morning" | "midday" | "afternoon" | "close"``.

    Returns:
        None. Logs structured events at each step boundary.
    """
    log.info("post_scrape_pipeline_starting", session=session_name)

    # 1. Screener — pure function, returns the inputs for everything below.
    all_candidates, ohlcv_by_symbol = screen_stocks(settings)

    # Three windows on the same sorted list:
    #   - top-50 -> persistence (unbiased shadow dataset)
    #   - top-20 -> Discord display
    #   - top-5  -> LLM thesis (expensive; per-call cost cap)
    scan_candidates = all_candidates[:50]
    candidates = all_candidates[:20]

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

    # 3. LLM thesis — gated by enabled flag + session allowlist + non-empty
    # candidates. compute_theses_and_persist itself enforces the top-5 cap,
    # but we slice here too so the call signature matches the historic
    # service.py contract and so the test layer can assert on the slice.
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
    if llm_gate_open:
        try:
            theses = compute_theses_and_persist(
                candidates[:5],
                ohlcv_by_symbol,
                scan_date=scan_date,
                session_name=session_name,
                settings=settings,
            )
        except Exception as exc:
            log.error(
                "compute_theses_unexpected_failure",
                session=session_name,
                error=str(exc),
            )
            theses = {}

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
        theses_sent=send_result.thesis_ok,
        theses_failed=send_result.thesis_failed,
    )
