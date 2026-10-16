"""On-demand single-ticker thesis pack.

Builds the same evidence + prompt the daily thesis call sends, for ANY ticker,
and writes it to disk so an agent (Devin / Claude Code) can do the analysis.
No LLM call and no DB writes — reads only.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from rainier.core.config import Settings
from rainier.core.types import StockCandidate

from .chart_export import THESIS_CHART_BARS, render_chart_png
from .prompt import SYSTEM_PROMPT
from .service import _candidate_to_pattern_signal, assemble_evidence, build_thesis_user_prompt

PROMPT_FILE = "prompt.md"
CHART_FILE = "chart.png"
EVIDENCE_FILE = "evidence.json"


@dataclass(frozen=True)
class TickerPack:
    symbol: str
    system_prompt: str
    user_prompt: str
    chart_png: bytes | None
    evidence: dict[str, Any]


def build_ticker_pack(
    candidate: StockCandidate,
    ohlcv_df,
    *,
    scan_date: date,
    session_name: str,
    settings: Settings,
    extra_sections: list[str] | None = None,
    extra_evidence: dict[str, Any] | None = None,
) -> TickerPack:
    pack, renders = asyncio.run(
        assemble_evidence(candidate, ohlcv_df, scan_date, session_name, settings.llm_thesis)
    )
    chart_png, _digest = render_chart_png(
        candidate.symbol, ohlcv_df, pattern=_candidate_to_pattern_signal(candidate)
    )
    user_prompt = build_thesis_user_prompt(
        pack, renders, scan_date=scan_date, extra_sections=extra_sections
    )
    evidence = pack.model_dump(mode="json")
    evidence.update(extra_evidence or {})
    return TickerPack(candidate.symbol, SYSTEM_PROMPT, user_prompt, chart_png, evidence)


def write_ticker_pack(pack: TickerPack, out_dir: Path) -> list[Path]:
    """Write prompt.md, chart.png and evidence.json; returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_note = (
        f"The chart is `{CHART_FILE}` in this folder: the last {THESIS_CHART_BARS} daily "
        "bars with the detected pattern's entry/SL/target lines."
        if pack.chart_png
        else "No chart (price history unavailable)."
    )
    prompt = (
        f"# {pack.symbol} thesis pack\n\n{chart_note}\n\n"
        f"## System prompt\n\n{pack.system_prompt}\n\n"
        f"## User message\n\n{pack.user_prompt}\n"
    )
    written = [out_dir / PROMPT_FILE, out_dir / EVIDENCE_FILE]
    written[0].write_text(prompt, encoding="utf-8")
    written[1].write_text(
        json.dumps(pack.evidence, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    if pack.chart_png:
        (out_dir / CHART_FILE).write_bytes(pack.chart_png)
        written.append(out_dir / CHART_FILE)
    return written
