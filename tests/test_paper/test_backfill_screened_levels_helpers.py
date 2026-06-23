"""Pure-helper unit tests for the screened-levels backfill (no Postgres).

The end-to-end backfill tests are Postgres-gated (``requires_postgres``) and
skip when no DB is reachable. But three load-bearing helpers are pure functions
that need no DB — and one of them (``_match_pattern``'s same-type tie-break) is
not exercised by the e2e fixture at all (it yields two patterns of DIFFERENT
types). These ungated tests pin that logic so it has coverage on every run.
"""

from __future__ import annotations

from datetime import date, timezone

import pandas as pd
import pytest

from rainier.core.types import PatternSignal
from rainier.paper.backfill_screened_levels import (
    _as_of_idx,
    _candidate_with_levels,
    _fetch_history,
    _match_for_row,
    _match_pattern,
    _Outcome,
    _TargetRow,
    backfill_screened_levels,
)


def _sig(
    pattern_type: str,
    *,
    confidence: float,
    entry_price: float,
    stop_loss: float = 0.0,
    target_wave1: float = 0.0,
    key_points: dict | None = None,
) -> PatternSignal:
    return PatternSignal(
        symbol="X",
        pattern_type=pattern_type,
        direction="bullish",
        status="confirmed",
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_wave1=target_wave1,
        key_points=key_points,
    )


def _target(pattern_type: str) -> _TargetRow:
    return _TargetRow(
        symbol="X",
        scan_date=date(2026, 6, 5),
        session_name="close",
        pattern_type=pattern_type,
        sector="Tech",
        composite_score=0.8,
    )


# --- _match_pattern --------------------------------------------------------


def test_match_pattern_no_match_returns_none():
    actionable = [_sig("w_bottom", confidence=0.9, entry_price=100.0)]
    assert _match_pattern(actionable, "false_breakdown") is None


def test_match_pattern_empty_returns_none():
    assert _match_pattern([], "w_bottom") is None


def test_match_pattern_picks_only_matching_type_not_top_ranked():
    # actionable[0] (the live `best_pattern`) is a non-matching type; the
    # matching one ranks lower but must still be chosen because we match by
    # stored type, not by top-ranking.
    actionable = [
        _sig("w_bottom", confidence=0.95, entry_price=100.0),
        _sig("false_breakdown", confidence=0.40, entry_price=89.0),
    ]
    got = _match_pattern(actionable, "false_breakdown")
    assert got is not None
    assert got.pattern_type == "false_breakdown"
    assert got.entry_price == 89.0


def test_match_pattern_same_type_takes_first_in_priority_order():
    # Two SAME-type patterns: the FIRST in _filter_actionable priority order
    # wins — that is the setup the live screener would have written as
    # best_pattern. We must NOT re-sort by confidence: here the first entry has
    # LOWER confidence but higher actionability priority, so it must win.
    actionable = [
        _sig("w_bottom", confidence=0.50, entry_price=80.0),  # higher priority
        _sig("w_bottom", confidence=0.90, entry_price=120.0),  # lower priority
    ]
    got = _match_pattern(actionable, "w_bottom")
    assert got is not None
    # First-in-order wins despite its lower confidence (no confidence re-sort).
    assert got.confidence == 0.50
    assert got.entry_price == 80.0


# --- _as_of_idx ------------------------------------------------------------


def _df(dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"close": range(len(dates))}, index=pd.DatetimeIndex(dates))


def test_as_of_idx_hits_exact_scan_date_bar():
    dates = [
        pd.Timestamp("2026-06-01", tz="UTC"),
        pd.Timestamp("2026-06-03", tz="UTC"),
        pd.Timestamp("2026-06-05", tz="UTC"),
        pd.Timestamp("2026-06-09", tz="UTC"),  # after scan_date
    ]
    # scan_date 06-05 → the EXACT 06-05 bar (positional index 2) is the as-of bar.
    assert _as_of_idx(_df(dates), date(2026, 6, 5)) == 2


def test_as_of_idx_none_when_scan_date_bar_missing_no_prior_day_fallback():
    """Fidelity gate (codex P1): a missing scan_date bar must NOT silently fall
    back to the prior trading day's bar. Replaying the prior day would write stale
    prior-day levels into a row meant to be reconstructed AS OF scan_date — so the
    row stays still_null instead. Returns None when the exact bar is absent."""
    dates = [
        pd.Timestamp("2026-06-01", tz="UTC"),
        pd.Timestamp("2026-06-03", tz="UTC"),
        pd.Timestamp("2026-06-08", tz="UTC"),
    ]
    # scan_date 06-05 has NO bar → None (no 06-03 fallback). Was index 1 pre-fix.
    assert _as_of_idx(_df(dates), date(2026, 6, 5)) is None


def test_as_of_idx_none_when_all_bars_after_scan_date():
    dates = [
        pd.Timestamp("2026-06-10", tz="UTC"),
        pd.Timestamp("2026-06-11", tz="UTC"),
    ]
    assert _as_of_idx(_df(dates), date(2026, 6, 5)) is None


def test_as_of_idx_handles_tz_naive_index():
    # A tz-naive index (synthetic data) must still compare cleanly.
    dates = [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-05")]
    assert _as_of_idx(_df(dates), date(2026, 6, 5)) == 1


def test_as_of_idx_includes_midnight_utc_scan_date_bar():
    # Production bars are normalized to midnight UTC; the scan_date bar itself
    # must be included (no off-by-one excluding it).
    bar = pd.Timestamp(date(2026, 6, 5), tz=timezone.utc)
    assert _as_of_idx(_df([bar]), date(2026, 6, 5)) == 0


def test_as_of_idx_matches_scan_date_under_non_utc_session_tz():
    """A midnight-UTC bar read back under a BEHIND-UTC session tz must still match
    its scan_date — the bar's UTC calendar date is canonical, not its local date.

    psycopg returns timestamptz in the Postgres session TimeZone GUC, which is NOT
    pinned to UTC on the local TimescaleDB. The 06-05 00:00 UTC bar presents as
    06-04 17:00-07:00 under America/Los_Angeles; taking the LOCAL date would mis-
    date it to 06-04 and the row would never match scan_date 06-05 — silently
    making the whole repair a 0-recovered no-op. _as_of_idx must convert to UTC
    first. FAILS on a local-tz `.normalize().date` implementation."""
    # The same instant the DB stores (midnight UTC on scan_date), but presented
    # in a behind-UTC zone the way a non-UTC psycopg session would return it.
    bar = pd.Timestamp(date(2026, 6, 5), tz=timezone.utc).tz_convert(
        "America/Los_Angeles"
    )  # -> 2026-06-04 17:00-07:00, LOCAL date 06-04
    assert _as_of_idx(_df([bar]), date(2026, 6, 5)) == 0


# --- _match_for_row outcome classification ---------------------------------


def _cfg(min_bars: int = 0):
    """A minimal stand-in config carrying just the knob _match_for_row reads past
    the early returns (``min_daily_bars``). The detector itself is monkeypatched
    in these tests, so no other config field is touched."""
    from types import SimpleNamespace

    return SimpleNamespace(min_daily_bars=min_bars)


def _ohlcv_at(scan: date, periods: int = 5) -> pd.DataFrame:
    """A tiny tz-naive OHLCV frame whose LAST bar is exactly on ``scan``."""
    dates = pd.bdate_range(end=pd.Timestamp(scan), periods=periods)
    return pd.DataFrame(
        {
            "open": range(periods),
            "high": range(1, periods + 1),
            "low": range(periods),
            "close": range(periods),
            "volume": [1] * periods,
        },
        index=dates,
    )


def test_match_for_row_no_bars_is_no_as_of_data():
    outcome, pat = _match_for_row(_target("false_breakdown"), None, config=object())
    assert outcome is _Outcome.NO_AS_OF_DATA
    assert pat is None


def test_match_for_row_missing_exact_bar_is_no_as_of_data():
    # Frame present but ending BEFORE the row's scan_date → no exact bar.
    row = _target("false_breakdown")  # scan_date 2026-06-05
    df = _ohlcv_at(date(2026, 6, 3))  # last bar 06-03, no 06-05
    outcome, pat = _match_for_row(row, df, config=object())
    assert outcome is _Outcome.NO_AS_OF_DATA
    assert pat is None


def test_match_for_row_detector_raise_is_detector_error(monkeypatch):
    """A detector exception classifies DETECTOR_ERROR (its own bucket), NOT
    NO_MATCH — a crash is not a proven-permanent no-match (codex P2)."""
    import rainier.paper.backfill_screened_levels as mod

    def boom(symbol, windowed, config):
        raise RuntimeError("synthetic detector blow-up")

    monkeypatch.setattr(mod, "replay_pattern_layer", boom)
    row = _target("false_breakdown")
    outcome, pat = _match_for_row(row, _ohlcv_at(date(2026, 6, 5)), config=_cfg())
    assert outcome is _Outcome.DETECTOR_ERROR
    assert pat is None


def test_match_for_row_clean_run_no_type_match_is_no_match(monkeypatch):
    """Detector runs clean but no actionable pattern has the stored type →
    NO_MATCH (permanent)."""
    import rainier.paper.backfill_screened_levels as mod

    # Returns an actionable of a DIFFERENT type than the row's stored type.
    monkeypatch.setattr(
        mod,
        "replay_pattern_layer",
        lambda s, w, c: ([_sig("w_bottom", confidence=0.9, entry_price=100.0)], None),
    )
    row = _target("false_breakdown")
    outcome, pat = _match_for_row(row, _ohlcv_at(date(2026, 6, 5)), config=_cfg())
    assert outcome is _Outcome.NO_MATCH
    assert pat is None


def test_match_for_row_type_match_is_recovered(monkeypatch):
    import rainier.paper.backfill_screened_levels as mod

    match = _sig("false_breakdown", confidence=0.5, entry_price=89.0)
    monkeypatch.setattr(
        mod, "replay_pattern_layer", lambda s, w, c: ([match], None)
    )
    row = _target("false_breakdown")
    outcome, pat = _match_for_row(row, _ohlcv_at(date(2026, 6, 5)), config=_cfg())
    assert outcome is _Outcome.RECOVERED
    assert pat is match


def test_match_for_row_window_below_min_daily_bars_is_no_match(monkeypatch):
    """Codex P2 fidelity gate: an as-of window SHORTER than min_daily_bars (a
    recent IPO/spinoff that only reaches the floor later in the range) is rejected
    BEFORE the detector runs — the live screener would have skipped that symbol on
    scan_date, so writing levels would fabricate a never-screened setup. NO_MATCH
    (permanent: the as-of window is fixed by scan_date + to_date)."""
    import rainier.paper.backfill_screened_levels as mod

    def must_not_run(*a, **k):
        raise AssertionError("detector must not run on a too-short as-of window")

    monkeypatch.setattr(mod, "replay_pattern_layer", must_not_run)
    row = _target("false_breakdown")
    # 5-bar window, but the floor is 60 → rejected as NO_MATCH, detector skipped.
    outcome, pat = _match_for_row(
        row, _ohlcv_at(date(2026, 6, 5), periods=5), config=_cfg(min_bars=60)
    )
    assert outcome is _Outcome.NO_MATCH
    assert pat is None


# --- _candidate_with_levels (bearish_invalidation_level backfill) ----------


def test_candidate_carries_false_breakout_invalidation_level():
    """A recovered `false_breakout` must carry `bearish_invalidation_level` (the
    pattern's `false_high`) so persist_screened_stocks fills it and the reclaim
    flow (which filters on `bearish_invalidation_level IS NOT NULL`) can re-enqueue
    the repaired row. Mirrors the live screener's false_breakout-only derivation."""
    pat = _sig(
        "false_breakout",
        confidence=0.7,
        entry_price=90.0,
        stop_loss=95.0,
        target_wave1=80.0,
        key_points={"false_high": 94.5},
    )
    cand = _candidate_with_levels(_target("false_breakout"), pat)
    assert cand.bearish_invalidation_level == 94.5
    # The four trade levels still flow through.
    assert cand.entry_price == 90.0
    assert cand.stop_loss == 95.0


def test_candidate_invalidation_level_none_for_non_false_breakout():
    """Only `false_breakout` carries a trap-high. A bullish pattern (e.g.
    `false_breakdown`) leaves `bearish_invalidation_level` NULL even if its
    key_points happen to contain a `false_high` key — matches live behavior."""
    pat = _sig(
        "false_breakdown",
        confidence=0.7,
        entry_price=89.0,
        key_points={"false_high": 94.5},  # ignored for a non-false_breakout
    )
    cand = _candidate_with_levels(_target("false_breakdown"), pat)
    assert cand.bearish_invalidation_level is None


def test_candidate_invalidation_level_none_when_false_high_absent():
    """A `false_breakout` whose key_points lack `false_high` (or None key_points)
    leaves the level NULL rather than crashing — defensive against a malformed
    pattern (e.g. false_breakout_hs, which has no false_high)."""
    assert (
        _candidate_with_levels(
            _target("false_breakout"),
            _sig("false_breakout", confidence=0.7, entry_price=90.0, key_points={}),
        ).bearish_invalidation_level
        is None
    )
    assert (
        _candidate_with_levels(
            _target("false_breakout"),
            _sig("false_breakout", confidence=0.7, entry_price=90.0, key_points=None),
        ).bearish_invalidation_level
        is None
    )


# --- _fetch_history (yfinance source, mirrors _fetch_stock_data normalize) -


def _ohlcv_frame(dates: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    """A yfinance-shaped per-symbol OHLCV frame (capitalized columns)."""
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [base + i for i in range(n)],
            "High": [base + i + 1 for i in range(n)],
            "Low": [base + i - 1 for i in range(n)],
            "Close": [base + i for i in range(n)],
            "Volume": [1000] * n,
        },
        index=dates,
    )


def _multi_download(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """A multi-symbol ``yf.download(group_by="ticker")`` shape: MultiIndex
    columns (symbol, field) over a shared date index."""
    return pd.concat(frames, axis=1)


def test_fetch_history_multi_symbol_normalizes_lowercase_and_indexes(monkeypatch):
    import rainier.paper.backfill_screened_levels as mod

    dates = pd.date_range("2026-01-02", periods=70, freq="B")
    raw = _multi_download(
        {"AAA": _ohlcv_frame(dates, 100.0), "BBB": _ohlcv_frame(dates, 50.0)}
    )

    def fake_download(symbols, **kwargs):
        # mirror the live screener: 1d interval, grouped by ticker, adjusted.
        assert kwargs["interval"] == "1d"
        assert kwargs["group_by"] == "ticker"
        assert kwargs["auto_adjust"] is True
        # explicit start/end (NOT period=) so the left edge is reachable.
        assert "start" in kwargs and "end" in kwargs and "period" not in kwargs
        return raw

    monkeypatch.setattr(mod.yf, "download", fake_download)

    out = _fetch_history(
        ["AAA", "BBB"], start=date(2026, 1, 1), end=date(2026, 6, 1), min_bars=60
    )

    assert set(out) == {"AAA", "BBB"}
    # lowercase OHLCV columns, date-indexed.
    assert list(out["AAA"].columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(out["AAA"].index, pd.DatetimeIndex)
    assert len(out["AAA"]) == 70


def test_fetch_history_single_symbol_flat_columns(monkeypatch):
    import rainier.paper.backfill_screened_levels as mod

    dates = pd.date_range("2026-01-02", periods=70, freq="B")
    raw = _ohlcv_frame(dates, 100.0)  # flat (non-MultiIndex) columns for 1 symbol

    monkeypatch.setattr(mod.yf, "download", lambda symbols, **kw: raw)

    out = _fetch_history(
        ["AAA"], start=date(2026, 1, 1), end=date(2026, 6, 1), min_bars=60
    )
    assert set(out) == {"AAA"}
    assert list(out["AAA"].columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_history_single_symbol_multiindex_columns(monkeypatch):
    """Codex [P1] regression: with group_by='ticker', a SINGLE-ticker yfinance
    download in 1.2.x still returns a MultiIndex (('AAA','Close')). The old
    `raw.copy()` kept those tuple columns so `_normalize_symbol_frame` found no
    'Close' and returned None — silently misclassifying every one-symbol fetch as
    no-data and DEFEATING the per-symbol solo-retry recovery path. _extract must
    flatten the single-ticker MultiIndex. FAILS pre-fix (out would be empty)."""
    import rainier.paper.backfill_screened_levels as mod

    dates = pd.date_range("2026-01-02", periods=70, freq="B")
    # Single-ticker MultiIndex, exactly what yf.download(['AAA'], group_by='ticker')
    # returns: columns are ('AAA', 'Open'), ('AAA', 'Close'), ...
    raw = _multi_download({"AAA": _ohlcv_frame(dates, 100.0)})
    assert isinstance(raw.columns, pd.MultiIndex)  # precondition: it IS MultiIndex

    monkeypatch.setattr(mod.yf, "download", lambda symbols, **kw: raw)

    out = _fetch_history(
        ["AAA"], start=date(2026, 1, 1), end=date(2026, 6, 1), min_bars=60
    )
    assert set(out) == {"AAA"}  # recovered, NOT dropped to no-data
    assert list(out["AAA"].columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_history_solo_retry_returns_multiindex_recovers(monkeypatch):
    """Codex [P1] regression on the retry path: a symbol dropped from the batch is
    re-fetched solo, and that solo call returns a single-ticker MultiIndex. The
    retry must still extract it (pre-fix it normalized to None → no_price_data)."""
    import rainier.paper.backfill_screened_levels as mod

    dates = pd.date_range("2026-01-02", periods=70, freq="B")
    batch = _multi_download({"AAA": _ohlcv_frame(dates, 100.0)})  # LLY absent
    solo = _multi_download({"LLY": _ohlcv_frame(dates, 80.0)})  # MultiIndex, 1 sym

    def fake_download(symbols, **kw):
        syms = symbols if isinstance(symbols, list) else [symbols]
        return solo if syms == ["LLY"] else batch

    monkeypatch.setattr(mod.yf, "download", fake_download)

    out = _fetch_history(
        ["AAA", "LLY"], start=date(2026, 1, 1), end=date(2026, 6, 1), min_bars=60
    )
    assert set(out) == {"AAA", "LLY"}  # LLY recovered from the MultiIndex solo retry


def test_fetch_history_skips_symbol_below_min_bars(monkeypatch):
    import rainier.paper.backfill_screened_levels as mod

    dates = pd.date_range("2026-01-02", periods=70, freq="B")
    short = pd.date_range("2026-01-02", periods=10, freq="B")
    raw = _multi_download(
        {"AAA": _ohlcv_frame(dates, 100.0), "SHORT": _ohlcv_frame(short, 50.0)}
    )
    monkeypatch.setattr(mod.yf, "download", lambda symbols, **kw: raw)

    out = _fetch_history(
        ["AAA", "SHORT"], start=date(2026, 1, 1), end=date(2026, 6, 1), min_bars=60
    )
    assert set(out) == {"AAA"}  # SHORT (10 bars) dropped below the 60-bar floor


def test_fetch_history_per_symbol_retry_on_transient_batch_drop(monkeypatch):
    """A symbol absent from the batch frame (observed LLY hiccup) is re-fetched
    solo; a network blip must never be mislabeled as no-data."""
    import rainier.paper.backfill_screened_levels as mod

    dates = pd.date_range("2026-01-02", periods=70, freq="B")
    # Batch returns AAA only; LLY is missing from the grouped frame.
    batch = _multi_download({"AAA": _ohlcv_frame(dates, 100.0)})
    solo = _ohlcv_frame(dates, 80.0)  # LLY solo retry succeeds

    calls: list[list[str]] = []

    def fake_download(symbols, **kw):
        calls.append(list(symbols) if isinstance(symbols, list) else [symbols])
        if symbols == ["LLY"] or symbols == "LLY":
            return solo
        return batch

    monkeypatch.setattr(mod.yf, "download", fake_download)

    out = _fetch_history(
        ["AAA", "LLY"], start=date(2026, 1, 1), end=date(2026, 6, 1), min_bars=60
    )
    assert set(out) == {"AAA", "LLY"}  # LLY recovered via the solo retry
    assert ["LLY"] in calls  # the per-symbol retry actually fired


def test_fetch_history_batch_exception_falls_back_to_per_symbol(monkeypatch):
    """If the whole batch download raises, each symbol is still attempted solo
    so a single bad batch does not zero out the corpus."""
    import rainier.paper.backfill_screened_levels as mod

    dates = pd.date_range("2026-01-02", periods=70, freq="B")
    solo = {"AAA": _ohlcv_frame(dates, 100.0), "BBB": _ohlcv_frame(dates, 50.0)}

    def fake_download(symbols, **kw):
        syms = symbols if isinstance(symbols, list) else [symbols]
        if len(syms) > 1:
            raise RuntimeError("synthetic batch failure")
        return solo[syms[0]]

    monkeypatch.setattr(mod.yf, "download", fake_download)

    out = _fetch_history(
        ["AAA", "BBB"], start=date(2026, 1, 1), end=date(2026, 6, 1), min_bars=60
    )
    assert set(out) == {"AAA", "BBB"}


def test_fetch_history_symbol_absent_after_solo_retry_omitted_not_crashed(monkeypatch):
    """The TERMINAL no-data branch: a symbol missing from the batch AND whose solo
    retry also returns nothing is simply omitted from the output map — never
    crashes, never falsely present. A well-formed sibling still comes through."""
    import rainier.paper.backfill_screened_levels as mod

    dates = pd.date_range("2026-01-02", periods=70, freq="B")
    batch = _multi_download({"AAA": _ohlcv_frame(dates, 100.0)})  # GONE absent

    def fake_download(symbols, **kw):
        syms = symbols if isinstance(symbols, list) else [symbols]
        if syms == ["GONE"]:
            return None  # solo retry also yields nothing (delisting / outage)
        return batch

    monkeypatch.setattr(mod.yf, "download", fake_download)

    out = _fetch_history(
        ["AAA", "GONE"], start=date(2026, 1, 1), end=date(2026, 6, 1), min_bars=60
    )
    assert set(out) == {"AAA"}  # GONE dropped (doubly-failed), AAA preserved


def test_fetch_history_empty_symbols_returns_empty(monkeypatch):
    import rainier.paper.backfill_screened_levels as mod

    def boom(*a, **k):
        raise AssertionError("must not download for an empty symbol list")

    monkeypatch.setattr(mod.yf, "download", boom)
    assert _fetch_history([], start=date(2026, 1, 1), end=date(2026, 6, 1), min_bars=60) == {}


# --- input validation ------------------------------------------------------


def test_backfill_rejects_from_after_to():
    with pytest.raises(ValueError, match="from_date.*>.*to_date"):
        backfill_screened_levels(
            from_date=date(2026, 6, 12),
            to_date=date(2026, 6, 3),
        )
