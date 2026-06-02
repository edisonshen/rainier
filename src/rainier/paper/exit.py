"""Pure path-dependent exit evaluator (design D5).

`evaluate_exit` is a pure function over (position-levels, ordered daily OHLC
rows) → an ExitResult or None (still open). It is the deterministic, unit-
testable core that `update_open_positions` calls per open position.

Rules (design D5 / TEST-SPEC §F):
  * Walk days from `entry_date` **inclusive** (the position is filled at the
    entry-day open, so an intraday SL/TP can trigger the same day).
  * **stop-loss** if `day_low <= stop_loss`; **target** if `day_high >= target`
    (equality inclusive, F7).
  * **same-day SL+TP → SL first** (conservative; the report counts these as
    `same_bar_ambiguous_exits`).
  * **gap-through:** if the day *opens* beyond the level (`open <= stop` or
    `open >= target`) exit at the **open**, not the level (F4/F5). Combined with
    SL-first: an open==stop bar that also gaps the target exits at the stop open
    (F7b).
  * **time_stop** at the `time_stop_days`-th trading session (entry day =
    session 1) when `time_stop_days` is set; NULL → no time-exit (F9/F10).
  * **no look-ahead:** rows are only considered up to `as_of` (F12) and never
    before `entry_date` (F11). NULL-OHLC days are no-data: skipped, never a
    phantom trigger (F15).
  * P&L: `return_pct = (exit-entry)/entry` (a fraction); `pnl = shares*(exit-
    entry)` — share-based, NOT `return_pct * notional` (F14).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class PriceBar:
    date: date
    open: float | None
    high: float | None
    low: float | None
    close: float | None


@dataclass(frozen=True)
class ExitResult:
    exit_date: date
    exit_price: float
    exit_reason: str  # stop_loss | target | time_stop
    return_pct: float
    pnl: float
    same_bar_ambiguous: bool  # SL+TP on the same bar (downward-bias disclosure)


def _coerce_bar(row: Any) -> PriceBar:
    if isinstance(row, PriceBar):
        return row
    # dict-like or ORM row with .date/.open/.high/.low/.close
    if isinstance(row, dict):
        return PriceBar(
            date=row["date"], open=row.get("open"), high=row.get("high"),
            low=row.get("low"), close=row.get("close"),
        )
    return PriceBar(
        date=_as_date(row.date), open=row.open, high=row.high,
        low=row.low, close=row.close,
    )


def _as_date(d: Any) -> date:
    from datetime import datetime

    if isinstance(d, datetime):
        return d.date()
    return d


def evaluate_exit(
    *,
    entry_date: date,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    shares: int,
    price_rows: list[Any],
    as_of: date,
    time_stop_days: int | None = None,
) -> ExitResult | None:
    """Return the exit (or None if still open) walking from entry_date inclusive
    up to (and including) ``as_of`` — never past it (no look-ahead)."""
    bars = [_coerce_bar(r) for r in price_rows]
    # Order by date, drop rows before entry_date or after as_of (look-ahead
    # guard on BOTH ends), and drop no-data days lazily during the walk.
    bars = [b for b in bars if entry_date <= _as_date(b.date) <= as_of]
    bars.sort(key=lambda b: _as_date(b.date))

    session_n = 0
    for bar in bars:
        bd = _as_date(bar.date)
        # NULL-OHLC day = no data → skip, do not advance the time-stop session
        # count (we can't know the position survived a day we never saw).
        if bar.high is None or bar.low is None:
            continue
        session_n += 1

        open_px = bar.open
        hit_stop = bar.low <= stop_loss
        hit_target = bar.high >= target_price
        same_bar = hit_stop and hit_target

        # --- gap-through AT THE OPEN takes priority (the open prints first, so a
        # level the open already gapped through was hit before any intraday
        # high/low). At most one can apply since stop_loss < target_price. This
        # resolves the gap-up-through-target-then-falls-to-stop case as a target
        # exit @ open (not a phantom stop), and the symmetric gap-down case as a
        # stop @ open.
        if open_px is not None and open_px <= stop_loss:
            return _result(bd, open_px, "stop_loss", entry_price, shares, same_bar)
        if open_px is not None and open_px >= target_price:
            return _result(bd, open_px, "target", entry_price, shares, same_bar)

        # --- no open-gap: intraday touches. Conservative same-bar SL+TP → SL
        # (F3 downward-bias convention, disclosed via same_bar_ambiguous). ---
        if hit_stop:
            return _result(bd, stop_loss, "stop_loss", entry_price, shares, same_bar)
        if hit_target:
            return _result(bd, target_price, "target", entry_price, shares, same_bar)

        # --- time-stop at the Nth trading session (entry day = session 1) ---
        if time_stop_days is not None and session_n >= time_stop_days:
            close_px = bar.close if bar.close is not None else entry_price
            return _result(bd, close_px, "time_stop", entry_price, shares, False)

    return None


def _result(
    exit_date: date, exit_price: float, reason: str, entry_price: float,
    shares: int, same_bar: bool,
) -> ExitResult:
    return_pct = (exit_price - entry_price) / entry_price
    pnl = shares * (exit_price - entry_price)
    return ExitResult(
        exit_date=exit_date, exit_price=exit_price, exit_reason=reason,
        return_pct=return_pct, pnl=pnl, same_bar_ambiguous=same_bar,
    )
