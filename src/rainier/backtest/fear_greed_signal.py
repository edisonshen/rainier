"""Fear & Greed + QQQ-MA market-entry backtest (Phase 2, offline analysis).

Combines a trend filter (QQQ vs its own SMA) with the CNN Fear & Greed Index
under three level-state combine logics and asks ONE question: does adding F&G
beat a QQQ-MA-only baseline out-of-sample? No live surface — this gates Phase 3.

Execution convention (no lookahead), mirroring ``tqqq_sma_sweep.py``'s
carry-then-apply order: the mask value for day ``D`` — computed from ``F&G[D]``
and ``close[D]``, both known only after ``D``'s close — gates the position that
earns ``return[D+1]``. Day D itself earns nothing from a signal that turns on at
D. The one-day lag lives in :func:`equity_curve`::

    for d in 1..n-1:
        if mask[d-1]:  equity *= 1 + ret[d]     # position set yesterday
        if mask[d] != mask[d-1]:  equity *= 1 - slip   # trade at today's close

The three logics (all require ``QQQ > SMA(window)`` AND):
    CONTRARIAN    F&G <= low_thresh      (buy fear in an uptrend)
    MOMENTUM      F&G >= high_thresh     (sentiment confirms trend)
    GREED_AVOID   F&G <  extreme_thresh  (step aside only on froth)

Selection is train-only; the gate metric is read from a single untouched OOS
holdout (NOT the full-sample selection ``walk_forward_top_n`` does). Masks are
built on the FULL price history so an SMA(200) has proper warmup in the OOS
slice; the split only partitions which return periods score into which window.

Data reads: prices via ``tqqq_sma_sweep.fetch_prices`` (adjusted-close parquet,
deep history); F&G via the legacy ``core.database`` engine. All current F&G rows
are ``source_version=backfill`` (revised values) so the whole span is
research-grade — surfaced in the report, never silently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rainier.core.database import get_session
from rainier.core.models import FearGreedIndex

_TRADING_DAYS_PER_YEAR = 252.0


class Logic(str, Enum):
    """The three level-state combine logics (all gated on ``QQQ > SMA``)."""

    CONTRARIAN = "contrarian"     # in when F&G <= threshold
    MOMENTUM = "momentum"         # in when F&G >= threshold
    GREED_AVOID = "greed_avoid"   # in when F&G <  threshold


# Pre-registered grid (task plan). Kept small on purpose — the OOS/overfitting
# bar in the design gates on a clear (x1.15) edge, not a marginal one.
MA_WINDOWS: tuple[int, ...] = (20, 50, 100, 200)
THRESHOLD_GRID: dict[Logic, tuple[float, ...]] = {
    Logic.CONTRARIAN: (20.0, 25.0, 30.0),
    Logic.MOMENTUM: (55.0, 60.0, 65.0, 75.0),
    Logic.GREED_AVOID: (75.0, 80.0),
}


@dataclass(frozen=True)
class Combo:
    """One (logic, MA window, F&G threshold) point in the grid."""

    logic: Logic
    ma: int
    threshold: float


@dataclass(frozen=True)
class ArmMetrics:
    """Performance summary for one backtest arm over one date window."""

    final_value: float
    total_return: float
    cagr: float
    max_dd: float
    calmar: float
    sharpe: float
    exposure: float
    switches: int
    hit_rate: float
    n_days: int


@dataclass(frozen=True)
class ComboResult:
    """A grid combo evaluated on the train slice, the OOS slice, and full span."""

    combo: Combo
    train: ArmMetrics
    oos: ArmMetrics
    full: ArmMetrics


@dataclass(frozen=True)
class BenchArm:
    """A benchmark arm (buy-and-hold or MA-only) across the three windows."""

    label: str
    train: ArmMetrics
    oos: ArmMetrics
    full: ArmMetrics


@dataclass(frozen=True)
class BacktestResult:
    """Everything the report renders + the auto-verdict inputs."""

    symbol: str
    start: date
    end: date
    split_date: date
    research_grade: bool
    daily_boundary: datetime | None
    slippage_bp: float
    combo_results: list[ComboResult]
    ma_only: dict[int, BenchArm]          # keyed by MA window
    buy_and_hold: BenchArm
    selections: dict[Logic, ComboResult]  # train-selected, per logic
    verdicts: dict[Logic, bool]           # OOS gate outcome, per logic
    eligible_logics: list[Logic]


# --------------------------------------------------------------------------
# Signal masks
# --------------------------------------------------------------------------


def sma(close: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average; ``NaN`` for the first ``window-1`` warmup days."""
    close = np.asarray(close, dtype=np.float64)
    n = close.shape[0]
    out = np.full(n, np.nan)
    if window < 1 or window > n:
        return out
    csum = np.concatenate([[0.0], np.cumsum(close)])
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def ma_mask(close: np.ndarray, window: int) -> np.ndarray:
    """Trend filter: ``True`` where ``close > SMA(window)`` (warmup days False)."""
    close = np.asarray(close, dtype=np.float64)
    ma = sma(close, window)
    mask = np.zeros(close.shape[0], dtype=bool)
    valid = ~np.isnan(ma)
    mask[valid] = close[valid] > ma[valid]
    return mask


def logic_mask(
    close: np.ndarray, fng: np.ndarray, logic: Logic, threshold: float, window: int
) -> np.ndarray:
    """Combined in/out mask: trend filter AND the logic's F&G condition."""
    fng = np.asarray(fng, dtype=np.float64)
    trend = ma_mask(close, window)
    if logic is Logic.CONTRARIAN:
        cond = fng <= threshold
    elif logic is Logic.MOMENTUM:
        cond = fng >= threshold
    elif logic is Logic.GREED_AVOID:
        cond = fng < threshold
    else:  # pragma: no cover - exhaustive enum
        raise ValueError(f"unknown logic {logic!r}")
    # A NaN F&G reading is "no signal" -> stay out.
    cond = np.where(np.isnan(fng), False, cond)
    return trend & cond


def buy_and_hold_mask(n: int) -> np.ndarray:
    """Always-in mask (the buy-and-hold benchmark)."""
    return np.ones(n, dtype=bool)


# --------------------------------------------------------------------------
# Equity + metrics
# --------------------------------------------------------------------------


def equity_curve(
    trade_close: np.ndarray, mask: np.ndarray, slippage_bp: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(equity, strat_ret)`` applying the D->D+1 lag + per-transition slip.

    ``equity[0] = 1.0``. The position earning ``ret[d]`` is ``mask[d-1]`` (set at
    the prior close). Slippage is charged once per adjacent state change; an
    all-in mask (buy-and-hold) therefore pays nothing and equals ``close`` scaled
    to 1.0 at the start.
    """
    trade_close = np.asarray(trade_close, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = trade_close.shape[0]
    ret = np.zeros(n)
    ret[1:] = trade_close[1:] / trade_close[:-1] - 1.0
    slip = slippage_bp * 1e-4

    equity = np.empty(n)
    equity[0] = 1.0
    e = 1.0
    for d in range(1, n):
        if mask[d - 1]:
            e *= 1.0 + ret[d]
        if mask[d] != mask[d - 1]:
            e *= 1.0 - slip
        equity[d] = e

    strat_ret = np.zeros(n)
    strat_ret[1:] = equity[1:] / equity[:-1] - 1.0
    return equity, strat_ret


def compute_metrics(
    trade_close: np.ndarray, mask: np.ndarray, slippage_bp: float = 5.0
) -> ArmMetrics:
    """Full metric summary for one arm (annualized at 252 days, rf=0)."""
    trade_close = np.asarray(trade_close, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = trade_close.shape[0]
    equity, strat_ret = equity_curve(trade_close, mask, slippage_bp)

    final_value = float(equity[-1])
    total_return = final_value - 1.0

    years = n / _TRADING_DAYS_PER_YEAR
    if final_value > 0.0 and years > 0.0:
        cagr = final_value ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    peak = np.maximum.accumulate(equity)
    max_dd = float((1.0 - equity / peak).max())

    std = float(strat_ret.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(strat_ret.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR)) if std > 0.0 else 0.0

    calmar = cagr / max_dd if max_dd > 0.0 else float("nan")

    exposure = float(mask.mean()) if n > 0 else 0.0
    switches = int(np.count_nonzero(mask[1:] != mask[:-1])) if n > 1 else 0

    # Hit-rate: of the return periods the arm actually held a position, the
    # fraction where the price moved up. In-position for ret[d] is mask[d-1].
    ret = np.zeros(n)
    ret[1:] = trade_close[1:] / trade_close[:-1] - 1.0
    in_pos = mask[:-1]
    held = int(np.count_nonzero(in_pos))
    hit_rate = float(np.count_nonzero((ret[1:] > 0.0) & in_pos) / held) if held else float("nan")

    return ArmMetrics(
        final_value=final_value,
        total_return=total_return,
        cagr=cagr,
        max_dd=max_dd,
        calmar=calmar,
        sharpe=sharpe,
        exposure=exposure,
        switches=switches,
        hit_rate=hit_rate,
        n_days=n,
    )


# --------------------------------------------------------------------------
# Selection + verdict
# --------------------------------------------------------------------------


def _train_rank_key(cr: ComboResult) -> float:
    """Rank key for train selection: Calmar, with NaN (no-drawdown) as worst.

    A NaN Calmar means no drawdown was observed on the train slice — usually a
    near-always-cash degenerate combo, not a robust winner — so it sorts last.
    """
    c = cr.train.calmar
    return c if np.isfinite(c) else float("-inf")


def select_best_on_train(
    combo_results: Sequence[ComboResult], logic: Logic
) -> ComboResult | None:
    """Pick the combo of ``logic`` with the highest TRAIN Calmar (OOS untouched)."""
    candidates = [cr for cr in combo_results if cr.combo.logic is logic]
    if not candidates:
        return None
    return max(candidates, key=_train_rank_key)


def logic_eligible(
    logic_oos: ArmMetrics, ma_only_oos: ArmMetrics, calmar_mult: float = 1.15
) -> bool:
    """The pre-registered 3-inequality promote bar, on the OOS holdout.

    Eligible iff ALL THREE hold vs the paired MA-only arm:
        (i)   OOS Calmar  >= MA-only OOS Calmar * 1.15
        (ii)  OOS Sharpe  >= MA-only OOS Sharpe
        (iii) OOS MaxDD   <= MA-only OOS MaxDD   (no deeper drawdown)
    A non-finite Calmar on either side fails (i) — Calmar is too unstable to
    stake a promotion on when a denominator vanished.
    """
    calmar_ok = (
        np.isfinite(logic_oos.calmar)
        and np.isfinite(ma_only_oos.calmar)
        and logic_oos.calmar >= ma_only_oos.calmar * calmar_mult
    )
    sharpe_ok = logic_oos.sharpe >= ma_only_oos.sharpe
    maxdd_ok = logic_oos.max_dd <= ma_only_oos.max_dd
    return bool(calmar_ok and sharpe_ok and maxdd_ok)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


SessionFactory = Callable[[], AbstractContextManager[Session]]


def load_fng_series(
    session_factory: SessionFactory = get_session,
) -> tuple[pd.Series, datetime | None]:
    """Read the point-in-time F&G composite as a date-indexed score series.

    Per the Phase-1 contract: for each ``date`` take the EARLIEST observation
    (min ``observed_at``, then min ``id``) — closest to real-time knowledge. The
    live-capture boundary is DERIVED as ``MIN(observed_at) WHERE
    source_version='daily'`` (``None`` when no live rows exist yet — the whole
    span is then research-grade). Returns ``(score_series, daily_boundary)``.
    """
    with session_factory() as session:
        rows = session.execute(
            select(FearGreedIndex.date, FearGreedIndex.score)
            .order_by(
                FearGreedIndex.date,
                FearGreedIndex.observed_at.asc(),
                FearGreedIndex.id.asc(),
            )
        ).all()
        boundary = session.execute(
            select(func.min(FearGreedIndex.observed_at))
            .where(FearGreedIndex.source_version == "daily")
        ).scalar_one_or_none()

    by_date: dict[date, float] = {}
    for d, score in rows:
        if d not in by_date:  # first row per date == earliest observation
            by_date[d] = float(score)

    series = pd.Series(by_date, dtype=float).sort_index()
    series.index = pd.to_datetime(series.index)
    return series, boundary


def align_prices_and_fng(
    prices: pd.DataFrame, fng: pd.Series, symbol: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Align prices to the F&G span, forward-filling F&G onto trading days.

    Returns ``(signal_close, trade_close, fng_values, dates)``. ``signal_close``
    is always QQQ (the trend + F&G are market-level per the design); ``trade_close``
    is the traded instrument. F&G is forward-filled onto any trading day the
    endpoint skipped — a stale (past) reading, never a future one, so the fill
    cannot reintroduce lookahead. Rows before F&G's first date are dropped.
    """
    sym = symbol.lower()
    if sym not in prices.columns:
        raise ValueError(f"prices frame has no column {sym!r} (have {list(prices.columns)})")
    if "qqq" not in prices.columns:
        raise ValueError("prices frame must include a 'qqq' column for the signal")

    idx = prices.index
    fng_aligned = fng.reindex(idx.union(fng.index)).sort_index().ffill().reindex(idx)
    joined = pd.DataFrame(
        {
            "qqq": prices["qqq"],
            "trade": prices[sym],
            "fng": fng_aligned,
        },
        index=idx,
    ).dropna()

    return (
        joined["qqq"].to_numpy(dtype=np.float64),
        joined["trade"].to_numpy(dtype=np.float64),
        joined["fng"].to_numpy(dtype=np.float64),
        joined.index,
    )


# --------------------------------------------------------------------------
# Backtest driver
# --------------------------------------------------------------------------


def _all_combos() -> list[Combo]:
    return [
        Combo(logic, ma, thr)
        for logic in Logic
        for ma in MA_WINDOWS
        for thr in THRESHOLD_GRID[logic]
    ]


def _split_index(dates: pd.DatetimeIndex, split_date) -> int:
    return int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(split_date))))


def _eval_windows(
    trade_close: np.ndarray, mask: np.ndarray, split: int, slippage_bp: float
) -> tuple[ArmMetrics, ArmMetrics, ArmMetrics]:
    """(train, oos, full) metrics — masks come from full history, sliced here."""
    train = compute_metrics(trade_close[:split], mask[:split], slippage_bp)
    oos = compute_metrics(trade_close[split:], mask[split:], slippage_bp)
    full = compute_metrics(trade_close, mask, slippage_bp)
    return train, oos, full


def run_backtest(
    signal_close: np.ndarray,
    trade_close: np.ndarray,
    fng: np.ndarray,
    dates: pd.DatetimeIndex,
    split_date,
    slippage_bp: float = 5.0,
    combos: Sequence[Combo] | None = None,
    ma_windows: Sequence[int] | None = None,
    symbol: str = "QQQ",
    research_grade: bool = True,
    daily_boundary: datetime | None = None,
) -> BacktestResult:
    """Run the grid + benchmarks with a single untouched OOS holdout.

    Selection is train-only (:func:`select_best_on_train`); the gate metric per
    logic is read from the OOS slice and scored against the MA-only arm sharing
    the selected combo's MA window (:func:`logic_eligible`).
    """
    combos = list(combos) if combos is not None else _all_combos()
    ma_windows = list(ma_windows) if ma_windows is not None else list(MA_WINDOWS)
    split = _split_index(dates, split_date)

    # A degenerate split (0 or n) leaves one window empty, which would crash
    # compute_metrics on a size-0 slice. Fail loudly with an actionable message.
    n = int(trade_close.shape[0])
    if not 0 < split < n:
        raise ValueError(
            f"split_date {pd.Timestamp(split_date).date()} yields a degenerate "
            f"train/OOS split (index {split} of {n}); it must fall strictly "
            f"inside the aligned span {dates[0].date()}..{dates[-1].date()} "
            "(check --split-date / --oos-months vs the data range)."
        )

    combo_results: list[ComboResult] = []
    for combo in combos:
        mask = logic_mask(signal_close, fng, combo.logic, combo.threshold, combo.ma)
        train, oos, full = _eval_windows(trade_close, mask, split, slippage_bp)
        combo_results.append(ComboResult(combo=combo, train=train, oos=oos, full=full))

    ma_only: dict[int, BenchArm] = {}
    for w in ma_windows:
        mask = ma_mask(signal_close, w)
        train, oos, full = _eval_windows(trade_close, mask, split, slippage_bp)
        ma_only[w] = BenchArm(label=f"MA-only({w})", train=train, oos=oos, full=full)

    bh_mask = buy_and_hold_mask(trade_close.shape[0])
    bh_train, bh_oos, bh_full = _eval_windows(trade_close, bh_mask, split, slippage_bp)
    buy_and_hold = BenchArm(label="buy-and-hold", train=bh_train, oos=bh_oos, full=bh_full)

    selections: dict[Logic, ComboResult] = {}
    verdicts: dict[Logic, bool] = {}
    for logic in Logic:
        picked = select_best_on_train(combo_results, logic)
        if picked is None:
            continue
        selections[logic] = picked
        paired = ma_only.get(picked.combo.ma)
        verdicts[logic] = (
            logic_eligible(picked.oos, paired.oos) if paired is not None else False
        )

    eligible = [logic for logic, ok in verdicts.items() if ok]

    return BacktestResult(
        symbol=symbol,
        start=dates[0].date(),
        end=dates[-1].date(),
        split_date=pd.Timestamp(split_date).date(),
        research_grade=research_grade,
        daily_boundary=daily_boundary,
        slippage_bp=slippage_bp,
        combo_results=combo_results,
        ma_only=ma_only,
        buy_and_hold=buy_and_hold,
        selections=selections,
        verdicts=verdicts,
        eligible_logics=eligible,
    )
