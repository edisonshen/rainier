"""TQQQ/SQQQ rotation sweep over a QQQ-SMA grid.

State machine (mutually exclusive states):

    CASH        --(QQQ > SMA(buy_T))--> LONG_TQQQ
    LONG_TQQQ   --(QQQ < SMA(sell_T))--> CASH
    CASH        --(QQQ < SMA(buy_S))--> SHORT_SQQQ
    SHORT_SQQQ  --(QQQ > SMA(sell_S))--> CASH

Same-day flip is allowed: on a given bar, the order of operations is

    1. mark-to-market the carried position with today's return
    2. if in a position, check the matching exit signal -> may go CASH
    3. if now CASH, check the entry signals -> may go LONG_TQQQ or SHORT_SQQQ

Phase 1 enforces ``sell_T >= buy_T`` and ``sell_S >= buy_S`` (trend-following
only). Phase 2 (anti-trend) is intentionally out of scope but reads the same
parquet schema, so adding it later is a separate sweep call, not a migration.

Wall-clock budget: < 20 min for ~3.35M combos on a single laptop, single-pass.
Inner backtest is ``@njit`` and is fed precomputed SMA boolean matrices so the
per-combo cost is a single sequential scan + four boolean column slices.

ASCII flow of the inner backtest loop::

    for d in 1..n_days:
        # mark to market
        if state == LONG_TQQQ:  equity *= 1 + tqqq_ret[d-1]
        if state == SHORT_SQQQ: equity *= 1 + sqqq_ret[d-1]

        # exit signal first
        if state == LONG_TQQQ  and not above_sellT[d]:  state = CASH; pay slippage
        if state == SHORT_SQQQ and       above_sellS[d]: state = CASH; pay slippage

        # entry signal if now flat
        if state == CASH and       above_buyT[d]:  state = LONG_TQQQ;  pay slippage
        if state == CASH and not above_buyS[d]:    state = SHORT_SQQQ; pay slippage
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a hard dep in pyproject
    def njit(*_args: Any, **_kwargs: Any):  # type: ignore[misc]
        def deco(fn):
            return fn
        return deco

# ---------------------------------------------------------------------------
# State constants (kept module-level so tests + callers can import them)
# ---------------------------------------------------------------------------

CASH: int = 0
LONG_TQQQ: int = 1
SHORT_SQQQ: int = 2

# Trading-day count for annualization (~252 / yr)
_TRADING_DAYS_PER_YEAR = 252.0


# ---------------------------------------------------------------------------
# Combo generator
# ---------------------------------------------------------------------------


def iter_phase1_combos(max_window: int = 60) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(buy_T, sell_T, buy_S, sell_S)`` tuples obeying Phase-1 constraints.

    Phase 1 = trend-following only: ``sell_T >= buy_T`` AND ``sell_S >= buy_S``.
    With ``max_window=60`` the count is ``1830 * 1830 = 3_348_900``
    (1830 = 60*61/2 ordered pairs per side).

    The iterator is materialized eagerly into a list to keep the hot loop in C;
    callers that need streaming should ``itertools.islice`` on it.
    """
    pairs = [
        (buy, sell)
        for buy in range(1, max_window + 1)
        for sell in range(buy, max_window + 1)
    ]
    for bT, sT in pairs:
        for bS, sS in pairs:
            yield (bT, sT, bS, sS)


# ---------------------------------------------------------------------------
# Precomputed signal matrix
# ---------------------------------------------------------------------------


def precompute_sma_signals(qqq_close: np.ndarray, max_window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Return two ``(n_days, max_window)`` bool matrices: ``(above, valid)``.

    ``above[d, w-1]`` = ``QQQ_close[d] > SMA_w[d]`` (long-signal truthy / short-signal falsy).
    ``valid[d, w-1]`` = the SMA for window ``w`` is computable at day ``d``.

    SMA is implemented with a cumulative-sum trick so the cost is O(n_days)
    per window, vectorized across windows. Days where the SMA cannot be
    computed (window not yet filled) yield ``above = False`` AND
    ``valid = False`` — the inner backtest must check ``valid`` before
    interpreting either polarity of ``above`` as a tradable signal.
    Without the ``valid`` mask, ``not above[...]`` would spuriously fire the
    SHORT_SQQQ entry during warmup and on the SMA(1) column (where the
    signal is structurally False, not "QQQ below SMA").
    """
    qqq_close = np.ascontiguousarray(qqq_close, dtype=np.float64)
    n = qqq_close.shape[0]
    above = np.zeros((n, max_window), dtype=np.bool_)
    valid = np.zeros((n, max_window), dtype=np.bool_)
    if max_window < 1:
        return above, valid
    # SMA1 = the price itself → "above" is the trivially-false signal AND
    # the signal is not meaningful (price never strictly above itself), so
    # column 0 stays valid=False — callers must not use buy_T/sell_T/buy_S/
    # sell_S = 1 expecting a real signal.
    # Cumulative sum with a leading zero so SMA_n[d] = (csum[d+1] - csum[d+1-n]) / n
    csum = np.concatenate([[0.0], np.cumsum(qqq_close)])
    for w in range(2, max_window + 1):
        if w > n:
            break
        sma = (csum[w:] - csum[:-w]) / w
        # First valid index is w-1 (days 0..w-2 don't have a full window)
        above[w - 1:, w - 1] = qqq_close[w - 1:] > sma
        valid[w - 1:, w - 1] = True
    return above, valid


# ---------------------------------------------------------------------------
# Inner backtest (njit)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def run_backtest(
    above: np.ndarray,
    valid: np.ndarray,
    tqqq_ret: np.ndarray,
    sqqq_ret: np.ndarray,
    buy_T: int,
    sell_T: int,
    buy_S: int,
    sell_S: int,
    slippage_bp: float = 5.0,
) -> tuple[float, float, float, float, int, float, float, float]:
    """Run one (buy_T, sell_T, buy_S, sell_S) combo on the precomputed signals.

    Parameters
    ----------
    above, valid:
        Bool matrices from :func:`precompute_sma_signals`, shape ``(n_days, W)``.
        ``valid[d, w-1]`` is True when SMA(w) is computable at day d. The
        backtest treats ``valid=False`` as "no signal" — neither a long
        trigger nor a short trigger fires. Without this check, the negated
        long-signal ``not above[...]`` would spuriously enter SHORT_SQQQ
        during warmup or on the structural-False SMA(1) column.
    tqqq_ret, sqqq_ret:
        Daily simple returns of TQQQ and SQQQ, shape ``(n_days - 1,)``.
        Element ``i`` is the return from day ``i`` close to day ``i+1`` close.
    buy_T, sell_T, buy_S, sell_S:
        1-based SMA windows. Indexed into ``above``/``valid`` as ``w - 1``.
    slippage_bp:
        Round-trip slippage in basis points, paid on each state transition.
        5 bp = 0.0005 of equity per transition.

    Returns
    -------
    tuple
        ``(final_value, sharpe, max_dd, calmar, n_trades, time_long, time_short, time_cash)``
        where ``time_*`` are in [0, 1].

        ``sharpe`` is annualized using sqrt(252) on daily strategy returns;
        ``max_dd`` is in [0, 1] (e.g. 0.42 == 42% peak-to-trough drawdown);
        ``calmar`` = annualized return / max_dd (NaN if no drawdown observed).
    """
    n_days = above.shape[0]
    slip = slippage_bp * 1e-4  # bps → fraction of equity per transition

    # State columns. v_bT/v_sT/... mask the warmup period and SMA(1).
    col_bT = above[:, buy_T - 1]
    col_sT = above[:, sell_T - 1]
    col_bS = above[:, buy_S - 1]
    col_sS = above[:, sell_S - 1]
    v_bT = valid[:, buy_T - 1]
    v_sT = valid[:, sell_T - 1]
    v_bS = valid[:, buy_S - 1]
    v_sS = valid[:, sell_S - 1]

    state = CASH
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    n_trades = 0
    long_days = 0
    short_days = 0
    cash_days = 0

    # Track daily strategy returns for Sharpe
    daily = np.empty(n_days, dtype=np.float64)
    daily[0] = 0.0

    # Day 0: no prior return; just attempt entry on day-0 signals (we only
    # attempt entry, not exit). Entries gated on validity — an SMA that
    # hasn't warmed up is "no signal", not "signal-false".
    if v_bT[0] and col_bT[0]:
        state = LONG_TQQQ
        equity *= 1.0 - slip
        n_trades += 1
    elif v_bS[0] and not col_bS[0]:
        state = SHORT_SQQQ
        equity *= 1.0 - slip
        n_trades += 1
    daily[0] = equity - 1.0  # captures any open-day slippage

    if state == LONG_TQQQ:
        long_days += 1
    elif state == SHORT_SQQQ:
        short_days += 1
    else:
        cash_days += 1

    for d in range(1, n_days):
        prev_equity = equity

        # 1) Mark to market using yesterday->today return
        if state == LONG_TQQQ:
            equity *= 1.0 + tqqq_ret[d - 1]
        elif state == SHORT_SQQQ:
            equity *= 1.0 + sqqq_ret[d - 1]
        # CASH: no change

        # 2) Exit signal — only fire when the exit SMA is computable.
        # Treat an invalid exit SMA as "stay in position" (we'd rather hold
        # than exit on a non-signal).
        if state == LONG_TQQQ and v_sT[d] and not col_sT[d]:
            state = CASH
            equity *= 1.0 - slip
            n_trades += 1
        elif state == SHORT_SQQQ and v_sS[d] and col_sS[d]:
            state = CASH
            equity *= 1.0 - slip
            n_trades += 1

        # 3) Entry signal if now flat — entry SMAs must be computable.
        if state == CASH:
            if v_bT[d] and col_bT[d]:
                state = LONG_TQQQ
                equity *= 1.0 - slip
                n_trades += 1
            elif v_bS[d] and not col_bS[d]:
                state = SHORT_SQQQ
                equity *= 1.0 - slip
                n_trades += 1

        # Daily return + DD tracking
        if prev_equity > 0:
            daily[d] = equity / prev_equity - 1.0
        else:
            daily[d] = 0.0
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = 1.0 - equity / peak
            if dd > max_dd:
                max_dd = dd

        if state == LONG_TQQQ:
            long_days += 1
        elif state == SHORT_SQQQ:
            short_days += 1
        else:
            cash_days += 1

    # Sharpe (annualized, rf=0)
    mean = 0.0
    for d in range(n_days):
        mean += daily[d]
    mean /= n_days
    var = 0.0
    for d in range(n_days):
        diff = daily[d] - mean
        var += diff * diff
    var /= max(n_days - 1, 1)
    std = math.sqrt(var)
    if std > 0.0:
        sharpe = mean / std * math.sqrt(_TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0

    # Calmar = annualized return / max_dd
    years = n_days / _TRADING_DAYS_PER_YEAR
    if equity > 0.0 and years > 0.0:
        ann_ret = equity ** (1.0 / years) - 1.0
    else:
        ann_ret = 0.0
    if max_dd > 0.0:
        calmar = ann_ret / max_dd
    else:
        calmar = math.nan

    inv_n = 1.0 / n_days
    return (
        equity,
        sharpe,
        max_dd,
        calmar,
        n_trades,
        long_days * inv_n,
        short_days * inv_n,
        cash_days * inv_n,
    )


# ---------------------------------------------------------------------------
# Data fetch + cache
# ---------------------------------------------------------------------------


PRICES_CACHE_PATH = Path("data/cache/tqqq_sma/prices.parquet")
RESULTS_CACHE_PATH = Path("data/cache/tqqq_sma/results.parquet")
TQQQ_INCEPTION = "2010-02-11"


def _hash_frame(df: pd.DataFrame) -> str:
    """Stable SHA-256 of the parquet bytes of ``df`` (sorted by index)."""
    buf = df.sort_index().to_parquet()
    return hashlib.sha256(buf).hexdigest()


def fetch_prices(
    start: str = TQQQ_INCEPTION,
    end: str | None = None,
    refresh: bool = False,
    cache_path: Path = PRICES_CACHE_PATH,
) -> pd.DataFrame:
    """Fetch QQQ/TQQQ/SQQQ adjusted closes, with a parquet cache.

    Cache layout: columns ``qqq``, ``tqqq``, ``sqqq``; index = ``DatetimeIndex``
    of trading days (UTC-naive). The companion ``.hash`` file stores the
    SHA-256 of the cache content; mismatch invalidates the cache.
    """
    cache_path = Path(cache_path)
    hash_path = cache_path.with_suffix(".sha256")
    if not refresh and cache_path.exists() and hash_path.exists():
        df = pd.read_parquet(cache_path)
        if hash_path.read_text().strip() == _hash_frame(df):
            return df

    import yfinance as yf  # imported lazily — tests don't need network

    tickers = ["QQQ", "TQQQ", "SQQQ"]
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # yfinance returns either a wide multiindex (multi-ticker) or flat
    cols = {}
    for t in tickers:
        if (t, "Close") in raw.columns:
            cols[t.lower()] = raw[(t, "Close")]
        elif t in raw.columns:
            cols[t.lower()] = raw[t]["Close"] if "Close" in raw[t].columns else raw[t]
        else:
            raise RuntimeError(f"yfinance did not return Close for {t}")
    df = pd.DataFrame(cols).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    hash_path.write_text(_hash_frame(df))
    return df


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PoolCtx:
    """Read-only context shared with worker processes via initializer."""

    above: np.ndarray
    valid: np.ndarray
    tqqq_ret: np.ndarray
    sqqq_ret: np.ndarray
    slippage_bp: float


# Globals populated by the pool initializer; each worker has its own
# interpreter so this is safe.
_WORKER_CTX: _PoolCtx | None = None


def _worker_init(
    above: np.ndarray,
    valid: np.ndarray,
    tqqq_ret: np.ndarray,
    sqqq_ret: np.ndarray,
    slip: float,
) -> None:
    global _WORKER_CTX
    _WORKER_CTX = _PoolCtx(
        above=above, valid=valid, tqqq_ret=tqqq_ret, sqqq_ret=sqqq_ret, slippage_bp=slip
    )
    # Warm up the JIT once per worker by running on a 1x1 combo
    if above.shape[0] >= 2:
        run_backtest(above, valid, tqqq_ret, sqqq_ret, 1, 1, 1, 1, slippage_bp=slip)


def compute_strategy_id(
    final_value: float,
    n_trades: int,
    time_in_long: float,
    time_in_short: float,
) -> int:
    """Deterministic uint64 fingerprint of an equity curve.

    Two combos that produce the same final value, trade count, and time-in-leg
    fractions (rounded to 2 dp on final_value and 4 dp on the time fractions)
    almost certainly share the same equity curve — in practice this happens
    whenever a dormant parameter varies freely (e.g. sell_S when buy_S=1
    structurally never fires).

    The hash is the SHA-256 of the canonical tuple representation, truncated
    to 64 bits. Pure-python deterministic — does NOT rely on Python's runtime
    hash randomization.

    ASCII flow::

        (fv≈, n, tL≈, tS≈) → repr → utf-8 → SHA-256 → first 8 bytes → uint64
    """
    key = (
        round(float(final_value), 2),
        int(n_trades),
        round(float(time_in_long), 4),
        round(float(time_in_short), 4),
    )
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    # First 8 bytes interpreted big-endian → uint64. Plenty of entropy for
    # leaderboard-level dedup; collisions on distinct curves are astronomical.
    return int.from_bytes(digest[:8], "big", signed=False)


def _worker_run(combo: tuple[int, int, int, int]) -> tuple[int, int, int, int, float, float, float, float, int, float, float, float, int]:
    ctx = _WORKER_CTX
    assert ctx is not None, "worker not initialized"
    bT, sT, bS, sS = combo
    out = run_backtest(
        ctx.above, ctx.valid, ctx.tqqq_ret, ctx.sqqq_ret,
        bT, sT, bS, sS, slippage_bp=ctx.slippage_bp,
    )
    final_value, _sharpe, _mdd, _calmar, n_trades, t_long, t_short, _t_cash = out
    sid = compute_strategy_id(final_value, n_trades, t_long, t_short)
    return (bT, sT, bS, sS, *out, sid)


_RESULT_COLUMNS = [
    "buy_T", "sell_T", "buy_S", "sell_S",
    "final_value", "sharpe", "max_dd", "calmar",
    "n_trades", "time_in_long", "time_in_short", "time_in_cash",
    "strategy_id",
]


def _flush_chunk(rows: list[tuple], results_path: Path) -> None:
    """Append ``rows`` to ``results_path`` atomically (write tmp, then concat+rename)."""
    if not rows:
        return
    df_new = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    df_new = df_new.astype({
        "buy_T": np.int16, "sell_T": np.int16, "buy_S": np.int16, "sell_S": np.int16,
        "n_trades": np.int32,
        # uint64 strategy_id — equity-curve fingerprint, used for leaderboard
        # dedup downstream. Pandas/Arrow round-trips uint64 cleanly.
        "strategy_id": np.uint64,
    })
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        existing = pd.read_parquet(results_path)
        merged = pd.concat([existing, df_new], ignore_index=True)
    else:
        merged = df_new
    tmp = results_path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, results_path)


def _existing_combo_set(results_path: Path) -> set[tuple[int, int, int, int]]:
    if not results_path.exists():
        return set()
    df = pd.read_parquet(results_path, columns=["buy_T", "sell_T", "buy_S", "sell_S"])
    return {(int(r.buy_T), int(r.sell_T), int(r.buy_S), int(r.sell_S)) for r in df.itertuples()}


# Bumped whenever the results.parquet schema changes in a way that an old
# parquet must be rejected (new column, changed dtype, changed semantics of
# an existing column). Read into the sweep fingerprint so a stale parquet
# from a prior schema is refused at sweep time.
_RESULTS_SCHEMA_VERSION = "v2-strategy_id"


def _sweep_fingerprint(prices: pd.DataFrame, slippage_bp: float, max_window: int) -> str:
    """SHA-256 of the sweep inputs that materially affect any combo's result.

    Used to invalidate ``results.parquet`` when the user reruns with different
    prices (`--refresh-data`), slippage, window bounds, or the parquet schema
    itself — otherwise the resumability path would silently mix stale rows
    with new ones, poisoning every downstream report.

    The schema version is part of the fingerprint so adding columns (e.g.
    ``strategy_id``) automatically forces a clean rerun on existing caches.
    """
    h = hashlib.sha256()
    # Stable price hash: parquet bytes of qqq/tqqq/sqqq sorted by index
    h.update(prices[["qqq", "tqqq", "sqqq"]].sort_index().to_parquet())
    h.update(f"|slippage_bp={slippage_bp:.6f}|max_window={max_window}".encode())
    h.update(f"|schema={_RESULTS_SCHEMA_VERSION}".encode())
    return h.hexdigest()


def _read_sweep_fingerprint(results_path: Path) -> str | None:
    fp_path = results_path.with_suffix(".fingerprint.txt")
    if not fp_path.exists():
        return None
    return fp_path.read_text().strip() or None


def _write_sweep_fingerprint(results_path: Path, fp: str) -> None:
    fp_path = results_path.with_suffix(".fingerprint.txt")
    tmp = fp_path.with_suffix(".txt.tmp")
    tmp.write_text(fp)
    os.replace(tmp, fp_path)


class SweepInputMismatchError(RuntimeError):
    """Raised when ``results.parquet`` exists but was produced from different
    prices, slippage, or max_window than the current call."""


def run_sweep(
    prices: pd.DataFrame,
    results_path: Path = RESULTS_CACHE_PATH,
    max_window: int = 60,
    n_workers: int | None = None,
    slippage_bp: float = 5.0,
    flush_every: int = 50_000,
    max_combos: int | None = None,
    progress: bool = False,
) -> Path:
    """Run the Phase-1 sweep, appending results to a parquet cache.

    Resumable: if ``results_path`` already has rows for some combos, those are
    skipped. On crash mid-sweep, the most recent completed flush is on disk.

    Cache safety: a SHA-256 fingerprint of ``(prices, slippage_bp, max_window)``
    is written alongside ``results_path`` as ``<name>.fingerprint.txt``. If a
    pre-existing ``results.parquet`` was produced from different inputs, the
    sweep aborts with :class:`SweepInputMismatchError` — never silently mixes
    rows. Delete the parquet (and its companion fingerprint) to force a clean
    rerun, or write to a different path.

    Parameters
    ----------
    prices:
        Frame with columns ``qqq``, ``tqqq``, ``sqqq`` indexed by date.
    results_path:
        Output parquet. Parent dir created if missing.
    max_window:
        Top SMA window. Phase 1 default 60.
    n_workers:
        Pool size. ``None`` → ``os.cpu_count()``. ``1`` runs in-process for
        easy debugging.
    slippage_bp:
        Round-trip slippage on each state transition, in basis points.
    flush_every:
        Flush rows to disk every N completed backtests. Crash-safety knob.
    max_combos:
        For testing — cap the total combo count. ``None`` runs the whole grid.

    Returns
    -------
    Path
        The output parquet path (same as ``results_path``).
    """
    qqq = prices["qqq"].to_numpy(dtype=np.float64)
    tqqq = prices["tqqq"].to_numpy(dtype=np.float64)
    sqqq = prices["sqqq"].to_numpy(dtype=np.float64)
    if qqq.shape != tqqq.shape or qqq.shape != sqqq.shape:
        raise ValueError("qqq/tqqq/sqqq columns must be the same length")
    if qqq.shape[0] < max_window + 2:
        raise ValueError(f"need at least {max_window + 2} bars; got {qqq.shape[0]}")

    # Cache-key the sweep on (prices, slippage, max_window). If an existing
    # results parquet was produced with different inputs, refuse to mix rows.
    fp = _sweep_fingerprint(prices, slippage_bp=slippage_bp, max_window=max_window)
    if results_path.exists():
        prior_fp = _read_sweep_fingerprint(results_path)
        if prior_fp is None:
            # Legacy parquet without a fingerprint companion — refuse to
            # extend it; the user must opt in by deleting it.
            raise SweepInputMismatchError(
                f"{results_path} exists but has no fingerprint file. "
                "Delete it (or write to a different --results-path) to start fresh."
            )
        if prior_fp != fp:
            raise SweepInputMismatchError(
                f"{results_path} was produced from different inputs (prices, "
                f"slippage_bp, or max_window). Delete it (and its companion "
                f"{results_path.with_suffix('.fingerprint.txt').name}) to start fresh, "
                "or write to a different --results-path."
            )

    above, valid = precompute_sma_signals(qqq, max_window=max_window)
    tqqq_ret = (tqqq[1:] / tqqq[:-1]) - 1.0
    sqqq_ret = (sqqq[1:] / sqqq[:-1]) - 1.0

    done = _existing_combo_set(results_path)
    todo: list[tuple[int, int, int, int]] = []
    for combo in iter_phase1_combos(max_window=max_window):
        if combo in done:
            continue
        todo.append(combo)
        if max_combos is not None and len(todo) >= max_combos:
            break
    if not todo:
        # Still write/refresh the fingerprint so a later resume catches drift
        _write_sweep_fingerprint(results_path, fp)
        return results_path

    n_workers = n_workers if n_workers is not None else (os.cpu_count() or 1)
    pending: list[tuple] = []
    start = time.time()
    done_count = 0

    if n_workers == 1:
        # In-process — simpler stack trace for debugging tests
        _worker_init(above, valid, tqqq_ret, sqqq_ret, slippage_bp)
        for combo in todo:
            pending.append(_worker_run(combo))
            done_count += 1
            if len(pending) >= flush_every:
                _flush_chunk(pending, results_path)
                pending.clear()
                if progress:
                    rate = done_count / max(time.time() - start, 1e-3)
                    print(f"  swept {done_count}/{len(todo)} ({rate:,.0f}/s)")
    else:
        ctx = mp.get_context("spawn")  # safer on macOS than fork
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(above, valid, tqqq_ret, sqqq_ret, slippage_bp),
        ) as pool:
            # imap_unordered to surface progress as workers complete
            for row in pool.imap_unordered(_worker_run, todo, chunksize=256):
                pending.append(row)
                done_count += 1
                if len(pending) >= flush_every:
                    _flush_chunk(pending, results_path)
                    pending.clear()
                    if progress:
                        rate = done_count / max(time.time() - start, 1e-3)
                        print(f"  swept {done_count}/{len(todo)} ({rate:,.0f}/s)")

    _flush_chunk(pending, results_path)
    # Stamp the parquet with the input fingerprint so future runs can detect
    # parameter drift instead of silently mixing stale rows.
    _write_sweep_fingerprint(results_path, fp)
    return results_path


# ---------------------------------------------------------------------------
# Walk-forward robustness check
# ---------------------------------------------------------------------------


def dedup_by_strategy_id(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse equity-curve duplicates by ``strategy_id``.

    For each unique ``strategy_id``, the representative row is the one with
    the lexicographically smallest ``(buy_T, sell_T, buy_S, sell_S)`` tuple
    — deterministic and independent of input row order. Adds an
    ``n_equivalent`` column equal to the number of grid combos in each
    group; the sum across the returned frame equals ``len(df)``.

    Rationale: with phase-1 trend-following constraints, many grid combos
    produce IDENTICAL equity curves because one or both legs are dormant
    (e.g. buy_S=1 makes the SHORT_SQQQ entry condition
    ``QQQ_close < SMA(1) = QQQ_close`` structurally False — sell_S varies
    with zero effect on the curve). Without dedup, the top-50 leaderboard
    can contain ~50 copies of one underlying strategy.

    Raises
    ------
    KeyError
        If ``strategy_id`` is missing — the caller must run the sweep with
        the v2 schema first.
    """
    if "strategy_id" not in df.columns:
        raise KeyError(
            "strategy_id column missing — re-run `rainier sma-sweep` after "
            "deleting the existing results.parquet to produce v2 schema rows."
        )
    counts = df.groupby("strategy_id", sort=False).size().rename("n_equivalent")
    # Lexicographic min on the 4-tuple → pick the first row after a stable
    # sort by (buy_T, sell_T, buy_S, sell_S).
    sorted_df = df.sort_values(["buy_T", "sell_T", "buy_S", "sell_S"], kind="stable")
    reps = sorted_df.drop_duplicates(subset="strategy_id", keep="first")
    out = reps.merge(counts, on="strategy_id", how="left").reset_index(drop=True)
    return out


def walk_forward_top_n(
    prices: pd.DataFrame,
    results_path: Path = RESULTS_CACHE_PATH,
    top_n: int = 100,
    split_date: str = "2019-01-01",
    slippage_bp: float = 5.0,
    max_window: int = 60,
) -> pd.DataFrame:
    """Rerun the deduped top-N by ``final_value`` on a train/test split.

    The dedup step is applied BEFORE the top-N selection so that the
    walk-forward set is N truly distinct strategies (one per equity curve)
    rather than N near-copies of the same strategy with dormant parameters
    varying. Adds ``final_value_train`` and ``final_value_test`` columns to
    the returned DataFrame (only for the top-N rows, not the whole 3.35M).
    """
    df = pd.read_parquet(results_path)
    df = dedup_by_strategy_id(df)
    top = df.nlargest(top_n, "final_value").reset_index(drop=True)

    split = pd.Timestamp(split_date)
    train = prices.loc[prices.index < split]
    test = prices.loc[prices.index >= split]

    def _value_for(slice_df: pd.DataFrame, combo: tuple[int, int, int, int]) -> float:
        if len(slice_df) < max_window + 2:
            return float("nan")
        qqq = slice_df["qqq"].to_numpy(dtype=np.float64)
        tqqq = slice_df["tqqq"].to_numpy(dtype=np.float64)
        sqqq = slice_df["sqqq"].to_numpy(dtype=np.float64)
        above, valid = precompute_sma_signals(qqq, max_window=max_window)
        tqqq_ret = (tqqq[1:] / tqqq[:-1]) - 1.0
        sqqq_ret = (sqqq[1:] / sqqq[:-1]) - 1.0
        bT, sT, bS, sS = combo
        return run_backtest(
            above, valid, tqqq_ret, sqqq_ret, bT, sT, bS, sS, slippage_bp=slippage_bp
        )[0]

    train_vals = []
    test_vals = []
    for _, row in top.iterrows():
        combo = (int(row.buy_T), int(row.sell_T), int(row.buy_S), int(row.sell_S))
        train_vals.append(_value_for(train, combo))
        test_vals.append(_value_for(test, combo))
    top["final_value_train"] = train_vals
    top["final_value_test"] = test_vals
    return top
