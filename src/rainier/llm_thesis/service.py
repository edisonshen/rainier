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
from rainier.core.types import StockCandidate

from .persistence import update_with_thesis
from .prompt import SYSTEM_PROMPT, build_user_message
from .schemas import EvidencePack, TradeThesis, compute_input_hash
from .signals import REGISTRY
from .signals.base import SignalContext

log = logging.getLogger(__name__)

# Static catalog of cost rates so callers can score-keep without LiteLLM telemetry.
_DEFAULT_INPUT_RATE = 3.0   # USD per 1M tokens (Sonnet 4.6 input)
_DEFAULT_OUTPUT_RATE = 15.0  # USD per 1M tokens (Sonnet 4.6 output)


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
) -> tuple[int, dict[str, Any]] | None:
    """Cheap cache lookup — match (date, symbol, prompt, session, model).

    We narrow the key on session_name + llm_model so the close-of-day rerun
    does NOT reuse the afternoon thesis (P1 from codex review): the close
    scan should reflect newer chart + signal data, and a v2 multi-model A/B
    needs each model's row to be independent. The matching INSERT path
    (idx_llm_analysis_idempotent) already keys on (day, symbols, model,
    prompt, input_hash) so true duplicates still collapse to one row via
    Tier 2's input_hash equality.
    """
    start = datetime.combine(scan_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    with get_session() as session:
        row = (
            session.query(
                LLMAnalysisRecord.id, LLMAnalysisRecord.structured_output
            )
            .filter(
                LLMAnalysisRecord.created_at >= start,
                LLMAnalysisRecord.created_at < end,
                LLMAnalysisRecord.target_symbols == [symbol],
                LLMAnalysisRecord.prompt_template == prompt_version,
                LLMAnalysisRecord.llm_model == llm_model,
            )
            .order_by(LLMAnalysisRecord.id.desc())
            .first()
        )
        if row is None:
            return None
        rec_id, output = row
        if output is None:
            return None
        # Reuse only when this row was produced for the SAME session.
        if isinstance(output, dict):
            recorded_session = output.get("_session_name")
            if recorded_session is not None and recorded_session != session_name:
                return None
        return int(rec_id), dict(output)


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
) -> tuple[str, int, int]:
    """Single LiteLLM completion. Returns (content, prompt_tokens, completion_tokens).

    Image bytes are passed as base64 data URL when present.
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

    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        temperature=0.2,
    )
    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
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

    # Tier 1: cache hit on same (date, symbol, prompt_template, session, model)?
    cached = _tier1_lookup(
        symbol,
        scan_date,
        prompt_version,
        session_name=session_name,
        llm_model=thesis_cfg.model,
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
    user_prompt = build_user_message(
        symbol=symbol,
        scan_date=pack.scan_date,
        session_name=session_name,
        candidate_summary=candidate_summary,
        signal_renders=renders,
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


async def _compute_theses_async(
    candidates: list[StockCandidate],
    ohlcv_by_symbol: dict[str, Any],
    *,
    scan_date: date,
    session_name: str,
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    from .chart_export import render_chart_png

    thesis_cfg = settings.llm_thesis
    max_usd = float(thesis_cfg.max_usd_per_scan)
    cost_used = 0.0
    out: dict[str, dict[str, Any]] = {}

    # Pre-rank by composite_score so we can compute would_be_combined_rank
    # within just the LLM-augmented set (top N — typically 5).
    ranked: list[tuple[StockCandidate, dict[str, Any]]] = []

    for candidate in candidates:
        if cost_used >= max_usd:
            log.warning(
                "thesis_budget_exhausted_skipping_remaining symbol=%s cost_used=%.4f",
                candidate.symbol,
                cost_used,
            )
            break

        symbol = candidate.symbol

        async def _provider_factory(c=candidate, s=symbol):
            df = ohlcv_by_symbol.get(s)
            pattern = None  # PatternSignal would be reconstructed if needed
            try:
                image_bytes, _digest = (
                    render_chart_png(s, df, pattern=pattern) if df is not None else (None, None)
                )
            except Exception:
                log.exception("chart_export_failed symbol=%s", s)
                image_bytes = None
            pack, renders = await assemble_evidence(
                c, df, scan_date, session_name, thesis_cfg
            )
            return pack, renders, image_bytes

        # The provider must be sync from generate_thesis's POV (called via to_thread).
        # Resolve the async assembly here and hand a thunk that just returns the result.
        try:
            pack, renders, image_bytes = await _provider_factory()
        except Exception as exc:
            log.exception("evidence_assembly_failed symbol=%s error=%s", symbol, exc)
            continue

        def _sync_provider(_pack=pack, _renders=renders, _image=image_bytes):
            return _pack, _renders, _image

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
        out[symbol] = thesis_dict
        ranked.append((candidate, thesis_dict))

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

    return out
