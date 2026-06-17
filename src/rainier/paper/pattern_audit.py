"""Pattern forward-return audit (WS B of the pattern-tuning design).

Builds, from the faithful pattern-layer replay (`paper/pattern_replay.py`), a
corpus of actionable pattern emissions tagged with:

  * forward return at 5 / 10 / 20 trading days (``close[t+H]/close[t] - 1``;
    NULL — never 0 — when fewer than H bars remain after ``t``),
  * the coarse market regime on ``t`` (`compute_market_regime`: SPY vs 200-SMA →
    bull / bear / unknown), and
  * the pattern's composite contribution.

Then aggregates per ``(pattern_type, regime, horizon)``: n, win-rate, mean /
median forward return, directional-correctness (does a bearish pattern actually
precede a decline?). The `unknown` regime cell is REPORTED, never dropped.

The corpus is a regenerable Parquet cache (feature-store convention; off Neon).
The headline question — "which patterns predict?" — needs only the pattern
layer + prices, so it spans the full 1-year window. The full 3-layer composite
replay (needs an as-of money-flow selector) is WS3, NOT this module.

    per symbol × trading day t with ≥ min_bars history:
        window last ~126 bars ending at t ──► emission_at ──► best pattern
        attach: fwd_return(5/10/20) + regime(t) + contribution
                          │
                          ▼
        corpus Parquet ──► aggregate by (pattern, regime, horizon)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from rainier.core.config import StockScreenerConfig
from rainier.paper.pattern_replay import (
    LIVE_LOOKBACK_BARS,
    emission_at,
)

log = logging.getLogger(__name__)

# Regenerable Parquet cache (feature-store convention; off Neon). Added to the
# CLAUDE.md disk-hygiene list. `rainier pattern-audit` re-derives it from
# stock_prices each run.
DEFAULT_CORPUS_DIR = Path("data/cache/qu100_pattern_audit")
CORPUS_FILENAME = "corpus.parquet"

# Forward-return horizons (trading days).
HORIZONS: tuple[int, ...] = (5, 10, 20)

# A symbol needs at least this many bars of history before t for the detector
# (mirrors the live min_daily_bars=60 floor; the window itself trims to ~126).
MIN_HISTORY_BARS = 60

CORPUS_COLUMNS: tuple[str, ...] = (
    "symbol",
    "as_of",
    "pattern_type",
    "direction",
    "status",
    "confidence",
    "pattern_contribution",
    "regime",
    "close_at_t",
    "fwd_return_5d",
    "fwd_return_10d",
    "fwd_return_20d",
)


# ---------------------------------------------------------------------------
# Forward return — close[t+H]/close[t] - 1; NULL (not 0) near the window end.
# ---------------------------------------------------------------------------


def forward_return(df: pd.DataFrame, t_idx: int, horizon: int) -> float | None:
    """Return ``close[t+H]/close[t] - 1`` or ``None`` if < H bars remain.

    NULL near the window end is a deliberate contract: a 0 there would be a
    fake "no move" that pollutes the win-rate. ``df`` is the FULL price frame
    (so future bars exist to look up); ``t_idx`` is the as-of row.
    """
    if t_idx < 0 or t_idx >= len(df):
        raise IndexError(f"t_idx {t_idx} out of range for {len(df)} bars")
    target_idx = t_idx + horizon
    if target_idx >= len(df):
        return None
    c0 = df["close"].iloc[t_idx]
    cH = df["close"].iloc[target_idx]
    if c0 is None or cH is None or c0 == 0:
        return None
    return float(cH) / float(c0) - 1.0


def directional_correct(direction: str, fwd_return: float | None) -> bool | None:
    """Did the pattern's direction match the realized move?

    bullish + return > 0 → True; bearish + return < 0 → True; exactly 0 or a
    NULL forward return → None (no directional read).
    """
    if fwd_return is None or fwd_return == 0.0:
        return None
    moved_up = fwd_return > 0
    if direction == "bullish":
        return moved_up
    if direction == "bearish":
        return not moved_up
    return None


# ---------------------------------------------------------------------------
# Corpus build — one row per actionable emission, fwd returns + regime.
# ---------------------------------------------------------------------------

# The regime tag is supplied by a callable mapping an as-of date to a regime
# string. Default uses the live `compute_market_regime`; injectable so the
# corpus build is testable without a live SPY history table.
def _default_regime_fn(as_of: date) -> str:
    from rainier.llm_thesis.research import compute_market_regime

    return compute_market_regime(as_of=as_of)


def build_corpus(
    prices_by_symbol: dict[str, pd.DataFrame],
    *,
    config: StockScreenerConfig,
    regime_fn=_default_regime_fn,
    min_history_bars: int = MIN_HISTORY_BARS,
    lookback_bars: int = LIVE_LOOKBACK_BARS,
) -> pd.DataFrame:
    """Replay the pattern layer over every (symbol, trading day) and attach
    forward returns + a regime tag.

    ``prices_by_symbol`` is the full per-symbol OHLCV frame (tz-aware date
    index, lowercase columns) — typically from
    `pattern_replay.load_prices`. Rows are emitted ONLY where an actionable
    pattern exists as-of ``t`` (matching what the live ranker would consume).
    The result is sorted by ``(symbol, as_of)`` for byte-deterministic output.
    """
    # The regime for a calendar date is invariant across symbols, but the
    # default `compute_market_regime` opens a fresh DB session + SPY query per
    # call. Memoize by date within this run so a 1-year × 100-symbol audit
    # issues one SPY query per distinct date, not one per emission. Per-run
    # (local) so an injected `regime_fn` and a fresh DB snapshot are honored.
    regime_cache: dict[date, str] = {}

    def _regime(as_of_date: date) -> str:
        cached = regime_cache.get(as_of_date)
        if cached is None:
            cached = regime_fn(as_of_date)
            regime_cache[as_of_date] = cached
        return cached

    rows: list[dict] = []
    for symbol in sorted(prices_by_symbol):
        df = prices_by_symbol[symbol]
        n = len(df)
        if n < min_history_bars:
            continue
        # As-of every bar from the min-history floor to the last bar. The window
        # trims to the live lookback; forward returns look INTO future bars of
        # the full frame, so the loop runs to the end (late rows get NULL fwd).
        for t_idx in range(min_history_bars - 1, n):
            window = df.iloc[max(0, t_idx + 1 - lookback_bars) : t_idx + 1]
            # The live screener wraps detect_patterns in try/except and skips
            # bad symbols; mirror that so one malformed window over a 1-year
            # replay drops that window, not the whole audit.
            try:
                emission = emission_at(symbol, window, config)
            except Exception:
                log.exception(
                    "pattern replay failed for %s @ t_idx=%d, skipping window",
                    symbol, t_idx,
                )
                continue
            if emission is None:
                continue
            as_of_ts = df.index[t_idx]
            as_of_date = (
                as_of_ts.date() if hasattr(as_of_ts, "date") else as_of_ts
            )
            row = {
                "symbol": symbol,
                "as_of": as_of_date,
                "pattern_type": emission.pattern_type,
                "direction": emission.direction,
                "status": emission.status,
                "confidence": emission.confidence,
                "pattern_contribution": emission.pattern_contribution,
                "regime": _regime(as_of_date),
                "close_at_t": emission.close_at_t,
            }
            for h in HORIZONS:
                row[f"fwd_return_{h}d"] = forward_return(df, t_idx, h)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(CORPUS_COLUMNS))
    corpus = pd.DataFrame(rows, columns=list(CORPUS_COLUMNS))
    return corpus.sort_values(["symbol", "as_of"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation — per (pattern_type, regime, horizon): n, win-rate, mean/median,
# directional-correctness. The `unknown` regime cell is reported, not dropped.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditCell:
    pattern_type: str
    regime: str
    horizon: int
    n: int  # emissions with a non-null fwd return at this horizon
    win_rate: float | None  # share with fwd return > 0
    mean_fwd_return: float | None
    median_fwd_return: float | None
    directional_correctness: float | None  # share where direction matched move
    thin: bool  # n < 30 → flagged thin, never over-tuned


THIN_N_FLOOR = 30


def aggregate(corpus: pd.DataFrame) -> list[AuditCell]:
    """Aggregate the corpus into per-(pattern_type, regime, horizon) cells.

    Cells are sorted by (pattern_type, regime, horizon) for stable output.
    Emissions whose fwd return is NULL at a horizon are excluded from THAT
    horizon's n (so n is honest per-horizon), but the same emission may count
    at a shorter horizon. The `unknown` regime is kept as its own cell.
    """
    cells: list[AuditCell] = []
    if corpus.empty:
        return cells

    groups = corpus.groupby(["pattern_type", "regime"], sort=True)
    for (pattern_type, regime), grp in groups:
        for horizon in HORIZONS:
            col = f"fwd_return_{horizon}d"
            vals = grp[col].dropna()
            n = int(len(vals))
            if n == 0:
                cells.append(
                    AuditCell(
                        pattern_type=pattern_type,
                        regime=regime,
                        horizon=horizon,
                        n=0,
                        win_rate=None,
                        mean_fwd_return=None,
                        median_fwd_return=None,
                        directional_correctness=None,
                        thin=True,
                    )
                )
                continue
            win_rate = float((vals > 0).mean())
            mean_ret = float(vals.mean())
            median_ret = float(vals.median())
            # Directional correctness over the same non-null subset.
            sub = grp.loc[vals.index]
            correct = [
                directional_correct(d, r)
                for d, r in zip(sub["direction"], sub[col], strict=True)
            ]
            judged = [c for c in correct if c is not None]
            dir_correct = (
                float(sum(judged) / len(judged)) if judged else None
            )
            cells.append(
                AuditCell(
                    pattern_type=pattern_type,
                    regime=regime,
                    horizon=horizon,
                    n=n,
                    win_rate=win_rate,
                    mean_fwd_return=mean_ret,
                    median_fwd_return=median_ret,
                    directional_correctness=dir_correct,
                    thin=n < THIN_N_FLOOR,
                )
            )
    return cells


def aggregate_to_frame(corpus: pd.DataFrame) -> pd.DataFrame:
    """`aggregate` as a DataFrame (one row per cell) for report rendering."""
    cells = aggregate(corpus)
    if not cells:
        return pd.DataFrame(
            columns=[
                "pattern_type", "regime", "horizon", "n", "win_rate",
                "mean_fwd_return", "median_fwd_return",
                "directional_correctness", "thin",
            ]
        )
    return pd.DataFrame([asdict(c) for c in cells])


# ---------------------------------------------------------------------------
# Corpus persistence + orchestration (CLI entry point).
# ---------------------------------------------------------------------------


def corpus_path(corpus_dir: Path | None = None) -> Path:
    base = corpus_dir if corpus_dir is not None else DEFAULT_CORPUS_DIR
    return base / CORPUS_FILENAME


def write_corpus(corpus: pd.DataFrame, corpus_dir: Path | None = None) -> Path:
    """Write the corpus to the regenerable Parquet cache."""
    path = corpus_path(corpus_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(path, index=False)
    return path


def universe_symbols(session) -> list[str]:
    """Audit universe = symbols present in `money_flow_snapshots` (the scraped
    QU100 names over history). Disclosed survivorship: historical membership
    isn't reconstructed; this is a fixed-universe-over-history approximation."""
    from sqlalchemy import select

    from rainier.core.models import MoneyFlowSnapshot

    rows = session.execute(
        select(MoneyFlowSnapshot.symbol).distinct()
    ).all()
    return sorted({r[0] for r in rows})


def run_pattern_audit(
    *,
    config: StockScreenerConfig,
    symbols: list[str] | None = None,
    corpus_dir: Path | None = None,
    min_history_bars: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Build the corpus from `stock_prices`, write Parquet, aggregate.

    Returns ``(corpus, aggregate_frame, corpus_path)``. Reads the legacy local
    TimescaleDB via `pattern_replay.load_prices`; the regime tag uses the live
    `compute_market_regime` (SPY vs 200-SMA). Deterministic for a fixed DB
    snapshot (explicit ORDER BY symbol, date in the price query).

    ``min_history_bars`` defaults to ``config.min_daily_bars`` (the live
    screener's data gate) so the replay starts emitting on the SAME bar floor
    the live ranker would — a champion that raises ``min_daily_bars`` must not
    leave the audit including emissions the live screen would have skipped.
    """
    from rainier.core.database import get_session
    from rainier.paper.pattern_replay import load_prices

    if min_history_bars is None:
        min_history_bars = config.min_daily_bars

    with get_session() as session:
        syms = symbols if symbols is not None else universe_symbols(session)
        prices = load_prices(session, syms)

    corpus = build_corpus(
        prices, config=config, min_history_bars=min_history_bars
    )
    agg = aggregate_to_frame(corpus)
    path = write_corpus(corpus, corpus_dir)
    return corpus, agg, path


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def render_report_markdown(
    corpus: pd.DataFrame, agg: pd.DataFrame, *, window_label: str
) -> str:
    """Render a concise markdown hit-rate table from the aggregate frame.

    One row per (pattern_type, regime, horizon). Thin cells (n<30) are flagged.
    The `unknown` regime is shown, never hidden. This is the body the CLI writes
    to `docs/REPORT-qu100-pattern-hit-rate.md`.
    """
    lines = [
        "# QU100 pattern forward-return audit",
        "",
        "*WS1 deliverable of the QU100-LLM pattern-tuning design. Auto-generated "
        "by `rainier pattern-audit`.*",
        "",
        "## The problem this answers",
        "",
        "Our daily stock screen tags each QU100 name with a chart pattern and "
        "ranks by a 3-layer score where the **pattern shape carries 65% of the "
        "weight**. Those pattern weights were hand-set and **never checked "
        "against what prices actually did next**. This audit replays the live "
        "pattern detector over one year of daily prices, records every "
        "actionable emission, and measures the forward return at 5 / 10 / 20 "
        "trading days — so weight tuning can finally trace to evidence.",
        "",
        "How to read the table:",
        "- **win-rate** = share of emissions where price went UP over the "
        "horizon. It is direction-agnostic (a bearish pattern with a high "
        "win-rate means price rose — i.e. the bearish call was *wrong*).",
        "- **dir-correct** = share where the move matched the pattern's "
        "direction (bullish→up, bearish→down) — the honest "
        "\"did the pattern predict?\" column.",
        "- **regime** = SPY vs its 200-day SMA on the as-of day "
        "(bull / bear / unknown).",
        "",
        "## Tuning hypotheses (one per later axis — NOT applied in WS1)",
        "",
        "- **WS2 (per-pattern weights):** down-weight / exclude bearish patterns "
        "whose dir-correct sits below ~50%; up-weight patterns with a "
        "monotone-improving bullish edge. The per-pattern weight is only ~35% of "
        "`score_pattern`, so true suppression needs an exclusion/threshold, not "
        "weight=0.",
        "- **WS3 (3-layer rebalance):** where pattern bull-regime cells are near "
        "coin-flip, the 0.65 pattern weight may be too high — shift toward "
        "money-flow (IC diagnostic + held-out grid-sweep). Needs the as-of "
        "money-flow selector (WS3).",
        "- **WS4 (LLM presentation):** surface each pattern's measured "
        "regime-conditional record (this table) to the LLM so an inverted "
        "bearish tag arrives pre-flagged.",
        "",
        "## Provenance",
        "",
        f"- Window: {window_label}",
        f"- Total emissions: {len(corpus)}",
        "- Universe = symbols scraped into `money_flow_snapshots` over history "
        "(fixed-universe-over-history; survivorship disclosed).",
        "- Regime = SPY vs 200-day SMA on the as-of day (bull / bear / unknown). "
        "`unknown` = fewer than 200 SPY bars at/before the as-of day; reported, "
        "never dropped.",
        "- `thin` = n < 30 at that horizon — reported, never over-tuned.",
        "- Reproduce: `rainier pattern-audit` (re-derives corpus + this report "
        "from Postgres `stock_prices`; deterministic for a fixed DB snapshot). "
        "Parity-tested vs live `screen_stocks` "
        "(`tests/test_paper/test_pattern_replay.py`).",
        "",
        "## Per-(pattern, regime, horizon) hit-rate",
        "",
        "| pattern | regime | H | n | win-rate | mean fwd | median fwd | "
        "dir-correct | thin |",
        "|---|---|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    if agg.empty:
        lines.append("| _(no emissions)_ | | | | | | | | |")
        return "\n".join(lines) + "\n"
    ordered = agg.sort_values(["pattern_type", "regime", "horizon"])
    for _, r in ordered.iterrows():
        lines.append(
            f"| {r['pattern_type']} | {r['regime']} | {int(r['horizon'])} | "
            f"{int(r['n'])} | {_fmt(r['win_rate'], pct=True)} | "
            f"{_fmt(r['mean_fwd_return'], pct=True)} | "
            f"{_fmt(r['median_fwd_return'], pct=True)} | "
            f"{_fmt(r['directional_correctness'], pct=True)} | "
            f"{'⚠' if r['thin'] else ''} |"
        )
    return "\n".join(lines) + "\n"
