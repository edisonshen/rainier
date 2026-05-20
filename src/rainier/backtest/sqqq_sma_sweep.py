"""SQQQ-only SMA sweep — bet on QQQ going down via long SQQQ, otherwise cash.

State machine (2 states, no TQQQ involvement at all):

    CASH       --(QQQ < SMA(buy_S))--> LONG_SQQQ
    LONG_SQQQ  --(QQQ > SMA(sell_S))--> CASH

Two parameters: ``buy_S`` (1..max_window), ``sell_S`` (1..max_window). The
grid is unconstrained — no ``sell >= buy`` requirement — so at the default
``max_window=60`` the sweep is 60×60 = 3,600 combos. The inner backtest is
``@njit`` and reuses the precomputed SMA validity-aware boolean matrices
from :mod:`rainier.backtest.tqqq_sma_sweep` (``precompute_sma_signals``).

The structural difference from the TQQQ/SQQQ rotation sweep is that there's
no long leg: this strategy never holds TQQQ, only SQQQ-or-cash. As a result
the 4-tuple ``(buy_T, sell_T, buy_S, sell_S)`` collapses to a 2-tuple
``(buy_S, sell_S)``, and ``time_in_long`` is gone from the returned metrics.

SMA(1) degeneracy (mirrors the TQQQ sweep, slightly different polarity):
    - ``buy_S=1``: entry signal ``QQQ < SMA(1) = QQQ < QQQ`` is structurally
      False AND the validity mask forces the signal off. Strategy stays in
      cash → ``n_trades = 0``, ``final_value = 1.0``.
    - ``sell_S=1``: exit signal ``QQQ > SMA(1) = QQQ > QQQ`` is structurally
      False AND the validity mask forces it off. If the strategy enters, it
      never exits → ``n_trades = 1``, equity = SQQQ B&H from entry forward.

Either case is a "degenerate one-shot" or "all-cash" outcome that the report
flags with a DEGENERATE banner. The dedup pass collapses all ``buy_S=1``
combos (regardless of ``sell_S``) into a single equity-curve equivalence
class — they're all the all-cash strategy.

ASCII flow of the inner backtest loop::

    for d in 1..n_days:
        # mark to market
        if state == LONG_SQQQ:  equity *= 1 + sqqq_ret[d-1]

        # exit signal first (then entry if now flat)
        if state == LONG_SQQQ and valid(sell_S, d) and QQQ > SMA(sell_S):
            state = CASH; pay slippage
        if state == CASH and valid(buy_S, d) and QQQ < SMA(buy_S):
            state = LONG_SQQQ; pay slippage
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

# Reuse the SMA precompute from the TQQQ sweep — same QQQ signal source,
# same validity gating semantics. Importing keeps the module a thin layer
# on top of the existing infrastructure rather than a parallel reimplementation.
from rainier.backtest.tqqq_sma_sweep import precompute_sma_signals

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

CASH: int = 0
LONG_SQQQ: int = 1

_TRADING_DAYS_PER_YEAR = 252.0


# ---------------------------------------------------------------------------
# Combo generator
# ---------------------------------------------------------------------------


def iter_sqqq_combos(max_window: int = 60) -> Iterator[tuple[int, int]]:
    """Yield every ``(buy_S, sell_S)`` in ``{1..max_window}^2``.

    Unconstrained 2-D grid: ``max_window^2`` combos. At ``max_window=60``
    this is 3,600 — trivial to enumerate single-process. We don't enforce
    ``sell_S >= buy_S`` because (a) the count is small enough that pruning
    is pointless and (b) anti-trend strategies (``sell_S < buy_S``) are a
    legitimate part of the search surface for the SQQQ-only universe.
    """
    for bS in range(1, max_window + 1):
        for sS in range(1, max_window + 1):
            yield (bS, sS)


# ---------------------------------------------------------------------------
# Inner backtest (njit)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def run_backtest(
    above: np.ndarray,
    valid: np.ndarray,
    sqqq_ret: np.ndarray,
    buy_S: int,
    sell_S: int,
    slippage_bp: float = 5.0,
) -> tuple[float, float, float, float, int, float, float, np.uint64]:
    """Run one ``(buy_S, sell_S)`` combo on the precomputed signals.

    Parameters
    ----------
    above, valid:
        Bool matrices from :func:`precompute_sma_signals`, shape ``(n_days, W)``.
        ``valid[d, w-1]`` is True when SMA(w) is computable at day d AND
        SMA(w) is a meaningful signal (column 0 == SMA(1) stays valid=False
        because ``QQQ > QQQ`` is structurally False, not a tradable signal).
    sqqq_ret:
        Daily simple returns of SQQQ, shape ``(n_days - 1,)``. Element ``i``
        is the return from day ``i`` close to day ``i+1`` close.
    buy_S, sell_S:
        1-based SMA windows. Indexed into ``above``/``valid`` as ``w - 1``.
    slippage_bp:
        Round-trip slippage in basis points, paid on each state transition.

    Returns
    -------
    tuple
        ``(final_value, sharpe, max_dd, calmar, n_trades,
          time_in_short, time_in_cash, curve_hash)`` where ``time_*`` are
        in [0, 1].

        No ``time_in_long`` field — the LONG_TQQQ state does not exist in
        this strategy. ``curve_hash`` is a uint64 FNV-1a polynomial rolling
        hash over the daily ``(state, n_trades)`` pair sequence. Two combos
        with identical ``(state, n_trades)`` trajectories produce the same
        hash; distinct trajectories collide with probability ~2**-64. This
        is the path-identity input for :func:`compute_strategy_id`.

        Mixing ``n_trades`` into the hash matters because a combo can exit
        and re-enter the same side on the same bar (e.g. sell_S fires then
        buy_S fires within day d), producing the same end-of-day state as
        a no-exit combo but paying 2× slippage. The cumulative trade count
        captures that divergence — same reasoning as the TQQQ sweep
        ``run_backtest`` (codex review iter-2 [P1]).
    """
    n_days = above.shape[0]
    slip = slippage_bp * 1e-4

    col_bS = above[:, buy_S - 1]
    col_sS = above[:, sell_S - 1]
    v_bS = valid[:, buy_S - 1]
    v_sS = valid[:, sell_S - 1]

    state = CASH
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    n_trades = 0
    short_days = 0
    cash_days = 0

    daily = np.empty(n_days, dtype=np.float64)
    daily[0] = 0.0

    # FNV-1a 64-bit. Same construction as the TQQQ sweep — see that module's
    # run_backtest for the iter-2 [P1] rationale on why we mix n_trades.
    curve_hash = np.uint64(0xcbf29ce484222325)
    _fnv_prime = np.uint64(0x100000001b3)

    # Day 0: only entry can fire (no prior return). Entry gated on validity:
    # an SMA that hasn't warmed up — or SMA(1), which is structurally False —
    # is "no signal", not "signal-false".
    if v_bS[0] and not col_bS[0]:
        state = LONG_SQQQ
        equity *= 1.0 - slip
        n_trades += 1
    daily[0] = equity - 1.0

    if state == LONG_SQQQ:
        short_days += 1
    else:
        cash_days += 1
    curve_hash = (curve_hash ^ np.uint64(state)) * _fnv_prime
    curve_hash = (curve_hash ^ np.uint64(n_trades)) * _fnv_prime

    for d in range(1, n_days):
        prev_equity = equity

        # 1) Mark to market using yesterday->today return
        if state == LONG_SQQQ:
            equity *= 1.0 + sqqq_ret[d - 1]

        # 2) Exit signal — only fire when sell_S is computable. Treating an
        # invalid exit SMA as "stay" matches the TQQQ sweep convention.
        if state == LONG_SQQQ and v_sS[d] and col_sS[d]:
            state = CASH
            equity *= 1.0 - slip
            n_trades += 1

        # 3) Entry signal if now flat — entry SMA must be computable.
        if state == CASH and v_bS[d] and not col_bS[d]:
            state = LONG_SQQQ
            equity *= 1.0 - slip
            n_trades += 1

        # Daily return + drawdown tracking
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

        if state == LONG_SQQQ:
            short_days += 1
        else:
            cash_days += 1
        curve_hash = (curve_hash ^ np.uint64(state)) * _fnv_prime
        curve_hash = (curve_hash ^ np.uint64(n_trades)) * _fnv_prime

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
        short_days * inv_n,
        cash_days * inv_n,
        curve_hash,
    )


# ---------------------------------------------------------------------------
# Cache paths (separate from TQQQ; prices may be shared)
# ---------------------------------------------------------------------------


# Reuse the TQQQ prices cache: it already contains QQQ + SQQQ, which is what
# this sweep needs. TQQQ column is just ignored by load_price_data() below.
# Pointing at the same parquet means a single ``rainier sma-sweep`` + ``rainier
# sqqq-sma-sweep`` workflow doesn't double-fetch.
SQQQ_PRICES_CACHE_PATH = Path("data/cache/tqqq_sma/prices.parquet")
SQQQ_RESULTS_CACHE_PATH = Path("data/cache/sqqq_sma/results.parquet")


def load_price_data(
    cache_path: Path = SQQQ_PRICES_CACHE_PATH,
) -> pd.DataFrame:
    """Load the QQQ/SQQQ price cache produced by the TQQQ sweep.

    We deliberately reuse the TQQQ sweep's parquet — it already has QQQ and
    SQQQ. We drop the TQQQ column (not needed). If the cache is missing the
    user should run ``uv run rainier sma-sweep`` first to populate it (or
    pass ``--refresh-data`` to the SQQQ sweep CLI to trigger a fresh fetch
    via the TQQQ helper).
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Price cache {cache_path} not found. Run "
            "`uv run rainier sma-sweep` first to populate it, or run "
            "`uv run rainier sqqq-sma-sweep --refresh-data` to fetch."
        )
    df = pd.read_parquet(cache_path)
    # Keep only what we need; the TQQQ column is harmless but irrelevant here.
    keep = [c for c in ("qqq", "sqqq") if c in df.columns]
    if "qqq" not in keep or "sqqq" not in keep:
        raise RuntimeError(
            f"Price cache {cache_path} missing qqq or sqqq column; "
            f"found {list(df.columns)}"
        )
    return df[keep]


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PoolCtx:
    """Read-only context shared with worker processes via initializer."""

    above: np.ndarray
    valid: np.ndarray
    sqqq_ret: np.ndarray
    slippage_bp: float


_WORKER_CTX: _PoolCtx | None = None


def _worker_init(
    above: np.ndarray,
    valid: np.ndarray,
    sqqq_ret: np.ndarray,
    slip: float,
) -> None:
    global _WORKER_CTX
    _WORKER_CTX = _PoolCtx(above=above, valid=valid, sqqq_ret=sqqq_ret, slippage_bp=slip)
    # Warm the JIT once per worker
    if above.shape[0] >= 2:
        run_backtest(above, valid, sqqq_ret, 1, 1, slippage_bp=slip)


def compute_strategy_id(curve_hash: int) -> int:
    """Deterministic uint64 fingerprint of an equity curve.

    Passthrough on the FNV-1a curve_hash emitted by :func:`run_backtest`.
    Identical ``(state, n_trades)`` daily trajectories → identical id;
    distinct trajectories collide with probability ~2**-64. Same contract
    and rationale as the TQQQ sweep — keeping a stable public API so
    downstream tools (leaderboard, walk-forward) don't reach into the
    backtest return tuple directly.
    """
    return int(np.uint64(curve_hash))


def _worker_run(combo: tuple[int, int]) -> tuple[int, int, float, float, float, float, int, float, float, int]:
    ctx = _WORKER_CTX
    assert ctx is not None, "worker not initialized"
    bS, sS = combo
    out = run_backtest(ctx.above, ctx.valid, ctx.sqqq_ret, bS, sS, slippage_bp=ctx.slippage_bp)
    *metrics, curve_hash = out
    sid = compute_strategy_id(curve_hash)
    return (bS, sS, *metrics, sid)


_RESULT_COLUMNS = [
    "buy_S", "sell_S",
    "final_value", "sharpe", "max_dd", "calmar",
    "n_trades", "time_in_short", "time_in_cash",
    "strategy_id",
]


def _flush_chunk(rows: list[tuple], results_path: Path) -> None:
    """Append ``rows`` to ``results_path`` atomically (tmp + rename)."""
    if not rows:
        return
    df_new = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    df_new = df_new.astype({
        "buy_S": np.int16, "sell_S": np.int16,
        "n_trades": np.int32,
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


def _existing_combo_set(results_path: Path) -> set[tuple[int, int]]:
    if not results_path.exists():
        return set()
    df = pd.read_parquet(results_path, columns=["buy_S", "sell_S"])
    return {(int(r.buy_S), int(r.sell_S)) for r in df.itertuples()}


_RESULTS_SCHEMA_VERSION = "v1-sqqq-only-with-strategy_id"


def _sweep_fingerprint(prices: pd.DataFrame, slippage_bp: float, max_window: int) -> str:
    """SHA-256 of the sweep inputs that materially affect any combo's result.

    Same anti-contamination pattern as the TQQQ sweep. Mismatched inputs →
    refuse to extend the parquet (don't silently mix stale rows with new ones).
    """
    h = hashlib.sha256()
    h.update(prices[["qqq", "sqqq"]].sort_index().to_parquet())
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
    """Raised when ``results.parquet`` exists but came from different inputs."""


def run_sweep(
    prices: pd.DataFrame,
    results_path: Path = SQQQ_RESULTS_CACHE_PATH,
    max_window: int = 60,
    n_workers: int | None = 1,
    slippage_bp: float = 5.0,
    flush_every: int = 1000,
    max_combos: int | None = None,
    progress: bool = False,
) -> Path:
    """Run the SQQQ-only SMA sweep, appending results to a parquet cache.

    Resumable: if ``results_path`` already has rows for some combos, those are
    skipped. Cache-safe: an SHA-256 fingerprint of ``(prices, slippage_bp,
    max_window, schema)`` is written alongside the parquet; mismatch raises
    :class:`SweepInputMismatchError` instead of silently mixing rows.

    At default ``max_window=60`` this is 3,600 combos — trivially single-pass
    on one core in ~1 second after the JIT warms. ``n_workers=1`` is the
    default (no point spinning up a process pool for that small a grid),
    but ``n_workers>1`` is supported for symmetry with the TQQQ sweep.
    """
    qqq = prices["qqq"].to_numpy(dtype=np.float64)
    sqqq = prices["sqqq"].to_numpy(dtype=np.float64)
    if qqq.shape != sqqq.shape:
        raise ValueError("qqq/sqqq columns must be the same length")
    if qqq.shape[0] < max_window + 2:
        raise ValueError(f"need at least {max_window + 2} bars; got {qqq.shape[0]}")

    fp = _sweep_fingerprint(prices, slippage_bp=slippage_bp, max_window=max_window)
    if results_path.exists():
        prior_fp = _read_sweep_fingerprint(results_path)
        if prior_fp is None:
            raise SweepInputMismatchError(
                f"{results_path} exists but has no fingerprint file. "
                "Delete it (or write to a different --results-path) to start fresh."
            )
        if prior_fp != fp:
            raise SweepInputMismatchError(
                f"{results_path} was produced from different inputs (prices, "
                f"slippage_bp, or max_window). Delete it (and "
                f"{results_path.with_suffix('.fingerprint.txt').name}) to start "
                "fresh, or write to a different --results-path."
            )

    above, valid = precompute_sma_signals(qqq, max_window=max_window)
    sqqq_ret = (sqqq[1:] / sqqq[:-1]) - 1.0

    done = _existing_combo_set(results_path)
    todo: list[tuple[int, int]] = []
    for combo in iter_sqqq_combos(max_window=max_window):
        if combo in done:
            continue
        todo.append(combo)
        if max_combos is not None and len(todo) >= max_combos:
            break
    if not todo:
        _write_sweep_fingerprint(results_path, fp)
        return results_path

    # Crash-safety: stamp the fingerprint BEFORE the first parquet flush so a
    # mid-sweep interrupt can still be resumed. Without this, the very first
    # _flush_chunk creates results.parquet, then a SIGINT/SIGKILL before the
    # tail _write_sweep_fingerprint(...) leaves the cache in a "parquet exists
    # but fingerprint missing" state — which run_sweep rejects as
    # SweepInputMismatchError on the next invocation, defeating the resume path.
    # (codex review iter-1 [P1])
    _write_sweep_fingerprint(results_path, fp)

    n_workers = n_workers if n_workers is not None else 1
    pending: list[tuple] = []
    start = time.time()
    done_count = 0

    if n_workers == 1:
        _worker_init(above, valid, sqqq_ret, slippage_bp)
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
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(above, valid, sqqq_ret, slippage_bp),
        ) as pool:
            for row in pool.imap_unordered(_worker_run, todo, chunksize=64):
                pending.append(row)
                done_count += 1
                if len(pending) >= flush_every:
                    _flush_chunk(pending, results_path)
                    pending.clear()
                    if progress:
                        rate = done_count / max(time.time() - start, 1e-3)
                        print(f"  swept {done_count}/{len(todo)} ({rate:,.0f}/s)")

    _flush_chunk(pending, results_path)
    _write_sweep_fingerprint(results_path, fp)
    return results_path


# ---------------------------------------------------------------------------
# Dedup + walk-forward
# ---------------------------------------------------------------------------


def dedup_by_strategy_id(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse equity-curve duplicates by ``strategy_id``.

    For each unique ``strategy_id``, the representative row is the one with
    the lexicographically smallest ``(buy_S, sell_S)`` tuple — deterministic
    and independent of input row order. Adds an ``n_equivalent`` column
    equal to the size of each equivalence class.

    The SMA(1) degeneracy is the load-bearing case here: every ``buy_S=1``
    combo (60 of them at ``max_window=60``) produces the same all-cash
    equity curve regardless of ``sell_S``, so they collapse into a single
    representative row with ``n_equivalent = 60``.
    """
    if "strategy_id" not in df.columns:
        raise KeyError(
            "strategy_id column missing — re-run `rainier sqqq-sma-sweep` "
            "after deleting the existing results.parquet."
        )
    counts = df.groupby("strategy_id", sort=False).size().rename("n_equivalent")
    sorted_df = df.sort_values(["buy_S", "sell_S"], kind="stable")
    reps = sorted_df.drop_duplicates(subset="strategy_id", keep="first")
    out = reps.merge(counts, on="strategy_id", how="left").reset_index(drop=True)
    return out


def walk_forward_top_n(
    prices: pd.DataFrame,
    results_path: Path = SQQQ_RESULTS_CACHE_PATH,
    top_n: int = 50,
    split_date: str = "2019-01-01",
    slippage_bp: float = 5.0,
    max_window: int = 60,
) -> pd.DataFrame:
    """Rerun the deduped top-N by ``final_value`` on a train/test split.

    Dedup is applied BEFORE top-N selection so the walk-forward set is N
    truly distinct equity curves, not N near-copies with dormant parameters
    varying. Returns the top-N rows with ``final_value_train`` and
    ``final_value_test`` columns appended.
    """
    df = pd.read_parquet(results_path)
    df = dedup_by_strategy_id(df)
    top = df.nlargest(top_n, "final_value").reset_index(drop=True)

    split = pd.Timestamp(split_date)
    train = prices.loc[prices.index < split]
    test = prices.loc[prices.index >= split]

    def _value_for(slice_df: pd.DataFrame, combo: tuple[int, int]) -> float:
        if len(slice_df) < max_window + 2:
            return float("nan")
        qqq = slice_df["qqq"].to_numpy(dtype=np.float64)
        sqqq = slice_df["sqqq"].to_numpy(dtype=np.float64)
        above, valid = precompute_sma_signals(qqq, max_window=max_window)
        sqqq_ret = (sqqq[1:] / sqqq[:-1]) - 1.0
        bS, sS = combo
        return run_backtest(above, valid, sqqq_ret, bS, sS, slippage_bp=slippage_bp)[0]

    train_vals = []
    test_vals = []
    for _, row in top.iterrows():
        combo = (int(row.buy_S), int(row.sell_S))
        train_vals.append(_value_for(train, combo))
        test_vals.append(_value_for(test, combo))
    top["final_value_train"] = train_vals
    top["final_value_test"] = test_vals
    return top
