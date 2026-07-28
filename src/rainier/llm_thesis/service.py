"""LLM thesis service — evidence assembly (parallel signals) + thesis generation
(Tier 1/Tier 2 cache + 3-retry validation + cost kill switch) + scheduler entry.

  ┌──────────────────────┐
  │ compute_theses_and_  │   called by scheduler/service.run_qu_scrape
  │      persist         │
  └────────┬─────────────┘
           │ for each candidate (top-5):
           ▼
  ┌──────────────────────┐    Tier-1 cache (cheap)
  │   generate_thesis    │──► SELECT id, structured_output FROM
  └────────┬─────────────┘    analysis_results WHERE date=today
           │ miss            AND target_symbols=[sym] AND prompt_template=?
           ▼ Tier-2 path
   evidence_provider() ─► assemble_evidence (parallel signals) ─► chart_export
                                                              ─► LiteLLM call
                                                              ─► Pydantic validate
                                                              ─► INSERT ON CONFLICT
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import func

from rainier.core.config import LLMThesisConfig, Settings
from rainier.core.database import get_session
from rainier.core.models import LLMAnalysisRecord
from rainier.core.types import PatternSignal, StockCandidate

from .persistence import update_with_thesis
from .prompt import SYSTEM_PROMPT, build_user_message
from .schemas import EvidencePack, TradeThesis, compute_input_hash
from .signals import REGISTRY
from .signals.base import SignalContext

log = logging.getLogger(__name__)

# Static catalog of cost rates so callers can score-keep without LiteLLM telemetry.
_DEFAULT_INPUT_RATE = 3.0   # USD per 1M tokens (Sonnet 4.6 input)
_DEFAULT_OUTPUT_RATE = 15.0  # USD per 1M tokens (Sonnet 4.6 output)

# Anthropic requires max_tokens > thinking.budget_tokens. We reserve this many
# tokens ABOVE the thinking budget for the final JSON answer. The thesis JSON is
# ~650 output tokens in practice; 2000 is comfortable headroom.
_FINAL_ANSWER_HEADROOM_TOKENS = 2000


# ---------------------------------------------------------------------------
# Evidence assembly
# ---------------------------------------------------------------------------


async def assemble_evidence(
    candidate: StockCandidate,
    ohlcv_df,  # pd.DataFrame | None — kept loose to avoid pandas import here
    scan_date: date,
    session_name: str,
    thesis_cfg: LLMThesisConfig,
) -> tuple[EvidencePack, list[str]]:
    """Run every enabled signal in parallel, build EvidencePack.

    Eng review D4: signals run via asyncio.gather + asyncio.to_thread so the
    slowest blocking signal (yfinance, ~1s) caps total wall time. Failures are
    isolated via return_exceptions=True — one bad signal does not kill the
    thesis.

    Returns (pack, signal_renders) where renders are the
    `signal.render_for_prompt(value)` strings in REGISTRY order, ready to
    stitch into the LLM user message.
    """
    enabled: list[Any] = []
    contexts: list[SignalContext] = []
    for name, sig_cls in REGISTRY.items():
        cfg = thesis_cfg.signals.get(name)
        if cfg is None or not cfg.enabled:
            continue
        signal_instance = sig_cls()
        ctx = SignalContext(
            symbol=candidate.symbol,
            scan_date=scan_date,
            session_name=session_name,
            candidate=candidate,
            ohlcv_df=ohlcv_df,
            params=dict(cfg.params),
        )
        enabled.append(signal_instance)
        contexts.append(ctx)

    if not enabled:
        pack = EvidencePack(
            symbol=candidate.symbol,
            scan_date=scan_date.isoformat(),
            session_name=session_name,
            candidate=asdict(candidate),
            signals={},
        )
        return pack, []

    tasks = [asyncio.to_thread(sig.compute, ctx) for sig, ctx in zip(enabled, contexts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    signals: dict[str, dict[str, Any]] = {}
    renders: list[str] = []
    for sig, value in zip(enabled, results):
        if isinstance(value, Exception):
            log.warning(
                "signal_failed signal=%s symbol=%s error=%s",
                sig.name,
                candidate.symbol,
                value,
            )
            continue
        if value is None:
            continue
        signals[sig.name] = value
        try:
            renders.append(sig.render_for_prompt(value))
        except Exception:
            log.exception("signal_render_failed signal=%s", sig.name)

    pack = EvidencePack(
        symbol=candidate.symbol,
        scan_date=scan_date.isoformat(),
        session_name=session_name,
        candidate=asdict(candidate),
        signals=signals,
    )
    return pack, renders


# ---------------------------------------------------------------------------
# Tier-1 cache lookup
# ---------------------------------------------------------------------------


def _tier1_lookup(
    symbol: str,
    scan_date: date,
    prompt_version: str,
    *,
    session_name: str,
    llm_model: str,
    enabled_signals: list[str] | None = None,
) -> tuple[int, dict[str, Any]] | None:
    """Cheap cache lookup — match (date, symbol, prompt, session, model).

    We narrow the key on session_name + llm_model so the close-of-day rerun
    does NOT reuse the afternoon thesis (P1 from codex review): the close
    scan should reflect newer chart + signal data, and a v2 multi-model A/B
    needs each model's row to be independent. The matching INSERT path
    (idx_llm_analysis_idempotent) already keys on (day, symbols, model,
    prompt, input_hash) so true duplicates still collapse to one row via
    Tier 2's input_hash equality.

    PR2 carry-over P2 #2: when ``enabled_signals`` is provided, reject any
    cached row whose ``signals_used`` (as recorded in structured_output) does
    not match the current set. This guarantees that toggling a signal off in
    settings.yaml causes the next scan to compute a fresh thesis instead of
    serving a stale one built with the old signal set.

    PR2 carry-over P3 #6: filter on the dedicated ``session_name`` column so
    the SQL WHERE clause (not post-Python on a JSONB key) discriminates rows.
    The previous post-filter dropped a same-session lookup to None whenever a
    cross-session row landed earlier in the partition (order_by id desc).
    Falling back to the legacy ``_session_name`` JSONB key keeps pre-PR2 rows
    reusable.
    """
    start = datetime.combine(scan_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    enabled_set = (
        frozenset(enabled_signals) if enabled_signals is not None else None
    )
    with get_session() as session:
        # Strict path: filter on session_name column. Legacy rows (PR1) have
        # session_name NULL, so include those in the candidate set as well —
        # the post-filter on _session_name JSONB key handles per-session reuse
        # for those legacy rows.
        candidates = (
            session.query(
                LLMAnalysisRecord.id,
                LLMAnalysisRecord.structured_output,
                LLMAnalysisRecord.session_name,
            )
            .filter(
                LLMAnalysisRecord.created_at >= start,
                LLMAnalysisRecord.created_at < end,
                LLMAnalysisRecord.target_symbols == [symbol],
                LLMAnalysisRecord.prompt_template == prompt_version,
                LLMAnalysisRecord.llm_model == llm_model,
            )
            .order_by(LLMAnalysisRecord.id.desc())
            .all()
        )
        for rec_id, output, row_session in candidates:
            if output is None:
                continue
            # Session match: prefer the dedicated column; fall back to the
            # legacy JSONB stamp on rows persisted before this column landed.
            if row_session is not None:
                if row_session != session_name:
                    continue
            elif isinstance(output, dict):
                recorded = output.get("_session_name")
                if recorded is not None and recorded != session_name:
                    continue
            # Signal-set drift check (PR2 carry-over P2 #2). When the caller
            # specifies the current enabled-signals list, only reuse a cached
            # thesis whose own signals_used set matches; otherwise the user
            # toggled a signal between scans and the LLM was reasoning over
            # a stale evidence pack.
            if enabled_set is not None and isinstance(output, dict):
                cached_signals = output.get("signals_used")
                if isinstance(cached_signals, list):
                    if frozenset(cached_signals) != enabled_set:
                        continue
            return int(rec_id), dict(output)
        return None


# ---------------------------------------------------------------------------
# Cost utility — used to enforce the per-scan kill switch
# ---------------------------------------------------------------------------


def _estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * _DEFAULT_INPUT_RATE
        + completion_tokens / 1_000_000 * _DEFAULT_OUTPUT_RATE
    )


# ---------------------------------------------------------------------------
# LLM call (Tier 2)
# ---------------------------------------------------------------------------


def _call_llm(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_bytes: bytes | None,
    thinking_budget_tokens: int,
) -> tuple[str, int, int]:
    """Single LiteLLM completion. Returns (content, prompt_tokens, completion_tokens).

    Image bytes are passed as base64 data URL when present.

    Extended thinking ("xhigh") is enabled via the deterministic explicit form
    ``thinking={"type": "enabled", "budget_tokens": N}`` — LiteLLM's
    ``reasoning_effort`` (low/medium/high) maps to much smaller anthropic budgets,
    so we pass the budget directly. When thinking is on, anthropic enforces two
    request invariants, both handled here:
      * ``temperature`` MUST be 1.0 (anthropic 400s on any other value), and
      * ``max_tokens`` MUST be strictly greater than ``budget_tokens``.

    The response carries reasoning under a separate ``reasoning_content`` /
    ``thinking_blocks`` field; ``message["content"]`` is still the final answer
    text (the JSON thesis _parse_thesis expects), so we keep reading ``content``
    and thinking text never leaks into the parsed thesis.

    Model gate: the ``thinking={"type": "enabled", "budget_tokens": N}`` payload
    is ANTHROPIC-specific (OpenAI reasoning models use ``reasoning_effort``), so
    we attach it (and the temperature==1.0 / max_tokens invariants it requires)
    only when the resolved provider is anthropic AND the model supports
    reasoning. If the thesis model is ever reconfigured to another provider, we
    fall back to the legacy plain call (temperature=0.2, no thinking) instead of
    letting LiteLLM raise ``UnsupportedParamsError``.

    Cost note: anthropic bills thinking tokens as OUTPUT tokens and folds them
    into ``usage.output_tokens``, which LiteLLM surfaces as ``completion_tokens``.
    So ``completion_tokens`` already includes the thinking spend — any
    ``reasoning_tokens`` LiteLLM reports in ``completion_tokens_details`` is a
    SUBSET of it (adding it would double-count). The per-scan kill switch bills
    ``completion_tokens`` at $15/M and therefore reflects true thinking spend.
    """
    import base64

    import litellm  # local import — tests mock this without pulling LiteLLM at import time

    content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    completion_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
    }
    try:
        _provider = litellm.get_llm_provider(model)[1]
    except Exception:
        _provider = ""
    anthropic_thinking = _provider == "anthropic" and litellm.supports_reasoning(
        model=model
    )
    if anthropic_thinking:
        # Extended thinking: temperature MUST be 1.0 and max_tokens MUST exceed
        # the thinking budget (final-answer headroom on top).
        completion_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget_tokens,
        }
        completion_kwargs["temperature"] = 1.0
        completion_kwargs["max_tokens"] = (
            thinking_budget_tokens + _FINAL_ANSWER_HEADROOM_TOKENS
        )
    else:
        # Non-reasoning provider: keep the original deterministic-ish call.
        completion_kwargs["temperature"] = 0.2

    resp = litellm.completion(**completion_kwargs)
    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    # completion_tokens already includes thinking tokens (billed as output).
    completion_tokens = int(usage.get("completion_tokens", 0))
    return text, prompt_tokens, completion_tokens


def _parse_thesis(raw: str) -> TradeThesis:
    """Parse the LLM's JSON output. Tolerates surrounding code fences."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip any ```json ... ``` fence.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return TradeThesis.model_validate_json(text)


# ---------------------------------------------------------------------------
# Public API: generate_thesis (single ticker)
# ---------------------------------------------------------------------------


async def generate_thesis(
    *,
    symbol: str,
    scan_date: date,
    session_name: str,
    evidence_provider: Callable[[], "tuple[EvidencePack, list[str], bytes | None]"],
    settings: Settings,
    max_usd_remaining: float,
) -> tuple[TradeThesis | None, float, int | None]:
    """Generate one thesis with two-tier cache + 3-retry validation.

    Returns `(thesis, cost_usd_charged, llm_record_id)`. `cost_usd_charged` is
    the marginal cost paid on this call; cache hits and kill-switch aborts return 0.
    """
    thesis_cfg = settings.llm_thesis
    prompt_version = thesis_cfg.prompt_version

    # PR2 carry-over P2 #2: include the current enabled-signals set so a
    # YAML toggle invalidates the cache for the next scan.
    enabled_signal_names = sorted(
        name for name, cfg in thesis_cfg.signals.items() if cfg.enabled
    )

    # Tier 1: cache hit on same (date, symbol, prompt_template, session, model)?
    cached = _tier1_lookup(
        symbol,
        scan_date,
        prompt_version,
        session_name=session_name,
        llm_model=thesis_cfg.model,
        enabled_signals=enabled_signal_names,
    )
    if cached is not None:
        record_id, raw_output = cached
        try:
            thesis = TradeThesis.model_validate(raw_output)
            log.info(
                "thesis_cache_hit symbol=%s scan_date=%s record_id=%s",
                symbol,
                scan_date,
                record_id,
            )
            return thesis, 0.0, record_id
        except Exception:
            log.warning(
                "thesis_cache_invalid_skip symbol=%s record_id=%s — falling through to Tier 2",
                symbol,
                record_id,
            )

    # Tier 2: budget gate BEFORE any expensive work.
    if max_usd_remaining <= 0:
        log.warning("thesis_killed_budget_exhausted symbol=%s", symbol)
        return None, 0.0, None

    pack, renders, image_bytes = await asyncio.to_thread(evidence_provider)
    input_hash = compute_input_hash(pack, image_bytes)

    candidate_summary = json.dumps(pack.candidate, sort_keys=True, default=str)
    # D7a: inject the calibration section (how prior theses graded out). Loaded
    # best-effort from the latest persisted PaperCalibration row; a failure or a
    # missing row (Phase-0 / fresh DB) yields no section. This text is invisible
    # to compute_input_hash, so PROMPT_VERSION ("v2") is what busts the Tier-1
    # cache — see prompt.py.
    calibration_section = ""
    try:
        from rainier.paper.calibration import (
            load_latest_calibration,
            render_calibration_section,
        )

        # strict_before: a scan on day D must only see calibration from a PRIOR
        # day. Day D's own paper_calibration row is written end-of-day D (after
        # that day's eval/paper outcomes), so a same-day rerun / replay must not
        # inject it — that would leak same-day hindsight (codex iter-3 [P2]).
        calibration_section = render_calibration_section(
            load_latest_calibration(scan_date, strict_before=True)
        )
    except Exception:
        log.warning("thesis_calibration_load_failed symbol=%s", symbol, exc_info=True)

    # R-A: append the rolling last-K post-exit reflections to the calibration
    # section. Same strictly-before discipline (a reflection for a trade exited
    # on day D is written end-of-day D — a day-D scan must not see it) and the
    # same best-effort isolation: a load failure costs the block, not the
    # thesis. Like the calibration text this is invisible to compute_input_hash,
    # so PROMPT_VERSION busts the Tier-1 cache instead.
    try:
        from rainier.paper.reflection import reflection_prompt_section

        reflections_block = reflection_prompt_section(scan_date)
        if reflections_block:
            calibration_section = (
                f"{calibration_section}\n\n{reflections_block}"
                if calibration_section
                else reflections_block
            )
    except Exception:
        log.warning("thesis_reflections_load_failed symbol=%s", symbol, exc_info=True)

    user_prompt = build_user_message(
        symbol=symbol,
        scan_date=pack.scan_date,
        session_name=session_name,
        candidate_summary=candidate_summary,
        signal_renders=renders,
        calibration_section=calibration_section,
    )

    last_error: str | None = None
    cost_charged = 0.0
    for attempt in range(3):
        attempt_user_prompt = user_prompt
        if last_error:
            attempt_user_prompt = (
                user_prompt
                + "\n\nNOTE: previous response failed validation: "
                + last_error
                + "\nReturn the JSON object that strictly matches the schema."
            )
        try:
            content, p_tok, c_tok = await asyncio.to_thread(
                _call_llm,
                model=thesis_cfg.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=attempt_user_prompt,
                image_bytes=image_bytes,
                thinking_budget_tokens=thesis_cfg.thinking_budget_tokens,
            )
        except Exception as exc:
            last_error = f"llm_call_failed: {exc}"
            log.warning(
                "thesis_llm_call_failed symbol=%s attempt=%s error=%s",
                symbol,
                attempt,
                exc,
            )
            continue

        attempt_cost = _estimate_cost_usd(p_tok, c_tok)
        cost_charged += attempt_cost
        if cost_charged > max_usd_remaining:
            log.warning(
                "thesis_killed_budget_overrun symbol=%s charged=%.4f remaining=%.4f",
                symbol,
                cost_charged,
                max_usd_remaining,
            )
            return None, cost_charged, None

        try:
            thesis = _parse_thesis(content)
        except Exception as exc:
            last_error = f"validation_failed: {exc}"
            log.warning(
                "thesis_validation_failed symbol=%s attempt=%s error=%s",
                symbol,
                attempt,
                exc,
            )
            continue

        # Persist to LLMAnalysisRecord (idempotent).
        record_id = _persist_thesis(
            symbol=symbol,
            thesis=thesis,
            settings=settings,
            input_hash=input_hash,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=attempt_cost,
            session_name=session_name,
        )
        return thesis, cost_charged, record_id

    log.warning(
        "thesis_validation_retries_exhausted symbol=%s last_error=%s",
        symbol,
        last_error,
    )
    return None, cost_charged, None


def _persist_thesis(
    *,
    symbol: str,
    thesis: TradeThesis,
    settings: Settings,
    input_hash: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    session_name: str,
) -> int | None:
    """Insert one LLMAnalysisRecord row; return its id, or None on conflict.

    The unique partial index `idx_llm_analysis_idempotent` keys on
    (date(created_at), target_symbols, llm_model, prompt_template, input_hash)
    — Tier-2 races and re-runs collapse to a single row.
    """
    payload = thesis.model_dump()
    # Stash the originating session inside structured_output so Tier-1
    # can refuse cross-session reuse on later same-day scans.
    payload["_session_name"] = session_name
    with get_session() as session:
        rec = LLMAnalysisRecord(
            llm_provider="anthropic",
            llm_model=settings.llm_thesis.model,
            prompt_template=settings.llm_thesis.prompt_version,
            target_symbols=[symbol],
            recommendation=thesis.verdict,
            confidence=float(thesis.llm_confidence),
            reasoning=thesis.paragraph_evidence,
            structured_output=payload,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost_usd=cost_usd,
            input_hash=input_hash,
            signals_used=list(thesis.signals_used or []),
            session_name=session_name,
        )
        session.add(rec)
        try:
            session.flush()
            rec_id = int(rec.id)
        except Exception:
            session.rollback()
            log.warning("thesis_persist_conflict symbol=%s — re-reading existing row", symbol)
            row = (
                session.query(LLMAnalysisRecord.id)
                .filter(
                    func.date(LLMAnalysisRecord.created_at) == datetime.now(timezone.utc).date(),
                    LLMAnalysisRecord.target_symbols == [symbol],
                    LLMAnalysisRecord.llm_model == settings.llm_thesis.model,
                    LLMAnalysisRecord.prompt_template == settings.llm_thesis.prompt_version,
                    LLMAnalysisRecord.input_hash == input_hash,
                )
                .first()
            )
            return int(row[0]) if row else None
    return rec_id


# ---------------------------------------------------------------------------
# Top-level orchestrator (called by scheduler)
# ---------------------------------------------------------------------------


def compute_theses_and_persist(
    candidates: list[StockCandidate],
    ohlcv_by_symbol: dict[str, Any],
    *,
    scan_date: date,
    session_name: str,
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """Run thesis generation across the top-N candidates synchronously.

    Synchronous wrapper so `scheduler/service.run_qu_scrape` can call it via
    `asyncio.to_thread` — keeps the async event loop free.

    Returns a `dict[symbol -> thesis_dict]` ready for Discord rendering. Each
    successful thesis also patches the matching ScreenedStockRecord row with
    `llm_confidence`, `shadow_combined_score`, `would_be_combined_rank`,
    `thesis_id`, `patterns_in_chart_not_in_indicators_count`.
    """
    if not candidates:
        return {}

    return asyncio.run(
        _compute_theses_async(
            candidates,
            ohlcv_by_symbol,
            scan_date=scan_date,
            session_name=session_name,
            settings=settings,
        )
    )


def _candidate_to_pattern_signal(
    candidate: StockCandidate,
) -> PatternSignal | None:
    """Reconstruct the PatternSignal that fired during screening (PR2 fix #4).

    The screener's :class:`PatternSignal` is collapsed onto the frozen
    :class:`StockCandidate` as a handful of flat fields (``pattern_type``,
    ``entry_price``, ``stop_loss``, ``target_price``, etc.). We rebuild the
    minimal PatternSignal needed by ``viz.charts.create_static_stock_chart``
    so the chart sent to the LLM gets the entry / SL / target overlays. If
    the candidate had no pattern, return ``None`` and the chart renders as a
    plain candle plot.
    """
    if (
        candidate.pattern_type is None
        or candidate.entry_price is None
        or candidate.stop_loss is None
        or candidate.target_price is None
    ):
        return None
    return PatternSignal(
        symbol=candidate.symbol,
        pattern_type=candidate.pattern_type,
        direction=candidate.pattern_direction or "bullish",
        status=candidate.pattern_status or "forming",
        confidence=float(candidate.pattern_confidence or 0.0),
        entry_price=float(candidate.entry_price),
        stop_loss=float(candidate.stop_loss),
        target_wave1=float(candidate.target_price),
        rr_ratio=float(candidate.rr_ratio or 0.0),
        volume_confirmed=bool(candidate.volume_confirmed),
    )


async def _compute_theses_async(
    candidates: list[StockCandidate],
    ohlcv_by_symbol: dict[str, Any],
    *,
    scan_date: date,
    session_name: str,
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    from .chart_export import render_chart_png
    from .chart_persistence import attach_chart_id_to_thesis, persist_chart_image

    thesis_cfg = settings.llm_thesis
    max_usd = float(thesis_cfg.max_usd_per_scan)
    cost_used = 0.0
    out: dict[str, dict[str, Any]] = {}

    # Pre-rank by composite_score so we can compute would_be_combined_rank
    # within just the LLM-augmented set (top N — typically 5).
    ranked: list[tuple[StockCandidate, dict[str, Any]]] = []
    # PR5: track the persisted chart_id per symbol so the post-thesis hook
    # can attach it to LLMAnalysisRecord.chart_image_ids. This stays bounded
    # to top-N candidates, so dict size is trivial.
    chart_ids_by_symbol: dict[str, int] = {}

    for candidate in candidates:
        if cost_used >= max_usd:
            log.warning(
                "thesis_budget_exhausted_skipping_remaining symbol=%s cost_used=%.4f",
                candidate.symbol,
                cost_used,
            )
            break

        symbol = candidate.symbol

        # PR1 carry-over P2 #1: Build the evidence provider as a deferred
        # closure. We DO NOT pre-resolve evidence here — generate_thesis runs
        # the Tier-1 cache lookup first and only invokes this provider on
        # cache miss. Without this deferral every "free rerun" still pays for
        # chart export + signal compute + (possibly) yfinance, defeating the
        # idempotency promise.
        df = ohlcv_by_symbol.get(symbol)
        pattern = _candidate_to_pattern_signal(candidate)

        async def _async_provider(c=candidate, s=symbol, _df=df, _pattern=pattern):
            image_bytes: bytes | None = None
            digest: str | None = None
            try:
                if _df is not None:
                    image_bytes, digest = render_chart_png(s, _df, pattern=_pattern)
            except Exception:
                log.exception("chart_export_failed symbol=%s", s)
                image_bytes = None
                digest = None
            # PR5: persist the chart bytes alongside their sha256 digest so
            # the Discord renderer + dashboard can serve the exact PNG that
            # went to the LLM. Idempotent on (symbol, scan_date, sha256).
            if image_bytes is not None and digest is not None:
                try:
                    chart_id = persist_chart_image(
                        symbol=s,
                        scan_date=scan_date,
                        image_bytes=image_bytes,
                        sha256=digest,
                    )
                    if chart_id is not None:
                        chart_ids_by_symbol[s] = chart_id
                except Exception:
                    log.exception("persist_chart_image_failed symbol=%s", s)
            pack, renders = await assemble_evidence(
                c, _df, scan_date, session_name, thesis_cfg
            )
            return pack, renders, image_bytes

        def _sync_provider(_async_provider=_async_provider):
            # generate_thesis calls this via asyncio.to_thread, so we must
            # synchronously drive the async assembly. asyncio.run is safe here
            # because we're running inside a worker thread (not the event
            # loop). Failure bubbles up — generate_thesis catches LLM call
            # failures, but evidence_provider failures are caller-fatal so we
            # let the per-ticker try/except below handle them.
            return asyncio.run(_async_provider())

        try:
            thesis, cost_charged, record_id = await generate_thesis(
                symbol=symbol,
                scan_date=scan_date,
                session_name=session_name,
                evidence_provider=_sync_provider,
                settings=settings,
                max_usd_remaining=max_usd - cost_used,
            )
        except Exception as exc:
            log.exception("thesis_unexpected_failure symbol=%s error=%s", symbol, exc)
            continue

        cost_used += cost_charged
        if thesis is None:
            continue

        thesis_dict = thesis.model_dump()
        # PR5: stamp the LLMAnalysisRecord id on the thesis dict so the
        # Discord renderer can build a dashboard deep-link without making
        # an extra DB lookup. Underscore-prefixed key — not part of the
        # Pydantic schema, just transport metadata for downstream
        # rendering. The leading underscore matches the existing
        # `_session_name` precedent in service._persist_thesis.
        if record_id is not None:
            thesis_dict["_thesis_id"] = int(record_id)
        out[symbol] = thesis_dict
        ranked.append((candidate, thesis_dict))

        # PR5: attach the persisted chart_id (if any) to the thesis record so
        # the Discord renderer + dashboard can read the exact bytes back. We
        # do this AFTER the thesis succeeds — a cache hit returns no chart
        # bytes (we never invoked the provider) and that's fine: an earlier
        # scan's row already carries chart_image_ids. This is also why the
        # mutation is idempotent at the data layer (skip if already present).
        chart_id = chart_ids_by_symbol.get(symbol)
        if chart_id is not None and record_id is not None:
            try:
                attach_chart_id_to_thesis(
                    thesis_id=record_id, chart_id=chart_id
                )
            except Exception:
                log.exception(
                    "attach_chart_id_failed symbol=%s thesis_id=%s chart_id=%s",
                    symbol,
                    record_id,
                    chart_id,
                )

        # Persist LLM-side fields onto ScreenedStockRecord.
        composite = float(candidate.signal_strength or 0.0)
        shadow_combined = 0.6 * composite + 0.4 * (thesis.llm_confidence / 10.0)
        patterns_count = (
            len(thesis.patterns_in_chart_not_in_indicators)
            if isinstance(thesis.patterns_in_chart_not_in_indicators, list)
            else 0
        )
        try:
            update_with_thesis(
                symbol=symbol,
                scan_date=scan_date,
                session_name=session_name,
                llm_confidence=int(thesis.llm_confidence),
                shadow_combined_score=round(shadow_combined, 4),
                would_be_combined_rank=None,  # patched below once all theses computed
                thesis_id=record_id,
                patterns_count=patterns_count,
            )
        except Exception:
            log.exception("update_with_thesis_failed symbol=%s", symbol)

    # Now assign would_be_combined_rank by sorting on shadow_combined_score.
    if ranked:
        from sqlalchemy import update as sql_update

        from rainier.core.models import ScreenedStockRecord

        scored: list[tuple[str, float]] = [
            (
                cand.symbol,
                0.6 * float(cand.signal_strength or 0.0)
                + 0.4 * (int(td.get("llm_confidence", 0)) / 10.0),
            )
            for cand, td in ranked
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        for rank, (symbol, _score) in enumerate(scored, start=1):
            try:
                with get_session() as session:
                    session.execute(
                        sql_update(ScreenedStockRecord)
                        .where(
                            ScreenedStockRecord.scan_date == scan_date,
                            ScreenedStockRecord.session_name == session_name,
                            ScreenedStockRecord.symbol == symbol,
                        )
                        .values(would_be_combined_rank=rank)
                    )
            except Exception:
                log.exception("would_be_combined_rank_update_failed symbol=%s", symbol)

    # Paper-tracker (design §5(1)): create a pending paper_trade for each
    # `setup_long` passing the confidence + session gate. Levels are read from
    # the persisted screened row inside create_positions_for_theses (D4), so
    # this runs AFTER update_with_thesis + persist_screened_stocks have landed
    # the row. Each insert is in its own get_session() scope; failures here are
    # non-fatal to thesis rendering.
    paper_theses: list[dict[str, Any]] = []
    try:
        paper_theses = [
            {
                "thesis_id": td["_thesis_id"],
                "symbol": cand.symbol,
                "verdict": td.get("verdict"),
                "llm_confidence": td.get("llm_confidence"),
                "session_name": session_name,
            }
            for cand, td in ranked
            if td.get("_thesis_id") is not None
        ]
        if paper_theses:
            from rainier.paper.positions import create_positions_for_theses

            create_positions_for_theses(
                paper_theses,
                scan_date=scan_date,
                learned_time_stop_days=thesis_cfg.learned_time_stop_days,
            )
    except Exception:
        log.exception("paper_position_creation_failed scan_date=%s", scan_date)

    # WS A — shadow WATCH-buy (measurement only). A `watch` verdict (conf >= T,
    # long-shape levels) opens a SHADOW paper_trade through the real engine,
    # excluded from every live read. Default ON; the live buy-flip (A2) is a
    # separate, gated change. Failure-isolated and never touches the live book.
    try:
        if settings.paper.watch_buy_shadow and paper_theses:
            from rainier.paper.positions import (
                create_shadow_positions_for_theses,
            )

            create_shadow_positions_for_theses(
                paper_theses,
                scan_date=scan_date,
                min_confidence=settings.paper.watch_buy_min_confidence,
            )
    except Exception:
        log.exception("shadow_position_creation_failed scan_date=%s", scan_date)

    return out
