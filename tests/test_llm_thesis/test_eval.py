"""Tests for daily evaluation: evaluate_horizon idempotent, signal
contribution Mann-Whitney U, verdict hit rate.

The eval functions touch tables (LLMAnalysisRecord, StockPrice) whose
column types (Postgres ARRAY, composite-PK on a hypertable) don't compile
on SQLite — so we stub the SQL boundaries (`get_session` / the price
helper) with in-test fakes. This keeps the unit tests fast and decoupled
from a live Postgres / TimescaleDB instance.

Postgres integration tests (gated `@pytest.mark.requires_postgres`)
exercise the full pipeline; those run on demand against a real DB.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rainier.llm_thesis.eval import (
    HORIZONS,
    HitRate,
    SignalContribution,
    compute_signal_contribution,
    compute_verdict_hit_rate,
    evaluate_horizon,
)

# ---------------------------------------------------------------------------
# Fake session helpers
# ---------------------------------------------------------------------------


class _FakeRow:
    """Mimic SQLAlchemy 2.x Row — supports attribute + index access."""

    def __init__(self, **fields: Any):
        self.__dict__.update(fields)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover
        raise AttributeError(item)


class _FakeSession:
    """Records inserts; lets execute() drive a scripted return queue."""

    def __init__(
        self,
        *,
        select_rows: list[Any] | None = None,
        aggregate_rows: list[Any] | None = None,
    ):
        self.select_rows = list(select_rows or [])
        self.aggregate_rows = list(aggregate_rows or [])
        self.inserts: list[dict[str, Any]] = []
        # Track whether we already returned the SELECT page so subsequent
        # execute() calls return aggregate_rows.
        self._selects_returned = 0

    def query(self, *args, **kwargs):
        # Used by _close_on_or_before in the production code; tests stub
        # _close_on_or_before directly so this never runs.
        return MagicMock()

    def execute(self, stmt):  # noqa: ARG002
        # First execute() in evaluate_horizon returns the candidate-list;
        # subsequent ones (the inserts) need a no-op result with rowcount=1.
        if self._selects_returned == 0 and self.select_rows:
            self._selects_returned += 1
            return MagicMock(all=lambda: list(self.select_rows))
        if self.aggregate_rows:
            return MagicMock(all=lambda: list(self.aggregate_rows))
        # Default: an INSERT — capture its values + report 1 row affected.
        result = MagicMock()
        result.rowcount = 1
        # Try to capture the values dict from a pg_insert(...).values(...)
        try:
            params = stmt.compile().params  # SQLAlchemy 2.x
            self.inserts.append(dict(params))
        except Exception:
            pass
        return result


class _CMSession:
    """Context-manager wrapper around _FakeSession for `with get_session():`."""

    def __init__(self, sess: _FakeSession):
        self.sess = sess

    def __enter__(self):
        return self.sess

    def __exit__(self, *_args):
        return False


# ---------------------------------------------------------------------------
# evaluate_horizon
# ---------------------------------------------------------------------------


def _candidate_row(
    *,
    rec_id: int,
    symbol: str,
    scan_date: date,
    verdict: str = "setup_long",
    confidence: float = 7.0,
    signals: list[str] | None = None,
    screened_id: int = 1,
):
    """Match the Row shape evaluate_horizon expects from its SELECT join."""
    return _FakeRow(
        id=rec_id,
        target_symbols=[symbol],
        recommendation=verdict,
        confidence=confidence,
        signals_used=list(signals or []),
        screened_id=screened_id,
        scan_date=scan_date,
        symbol=symbol,
    )


def test_evaluate_horizon_inserts_one_row_per_thesis():
    today = date(2026, 5, 7)
    scan = today - timedelta(days=5)
    candidate_session = _FakeSession(
        select_rows=[
            _candidate_row(rec_id=42, symbol="NVDA", scan_date=scan),
        ]
    )
    insert_session = _FakeSession()

    sessions = iter([candidate_session, insert_session])

    def _get_session():
        return _CMSession(next(sessions))

    with patch("rainier.llm_thesis.eval.get_session", side_effect=_get_session), patch(
        "rainier.llm_thesis.eval._close_on_or_before",
        side_effect=lambda sym, d: (d, 100.0) if d == scan else (d, 110.0),
    ):
        n = evaluate_horizon(today, "5d")

    assert n == 1
    # The INSERT was issued on the second session.
    assert len(insert_session.inserts) == 1
    payload = insert_session.inserts[0]
    assert payload["thesis_id"] == 42
    assert payload["entry_price"] == 100.0
    assert payload["exit_price"] == 110.0
    assert abs(payload["return_pct"] - 0.10) < 1e-9
    assert payload["hit"] is True


def test_evaluate_horizon_idempotent():
    """When ON CONFLICT DO NOTHING reports rowcount=0, the inserted counter
    stays at 0. The second pass over the same theses produces no new rows.
    """
    today = date(2026, 5, 7)
    scan = today - timedelta(days=5)

    # Session 1: candidate SELECT returns one row. Subsequent INSERT reports
    # rowcount=0 (conflict). evaluate_horizon should count it as 0.
    class _ConflictSession(_FakeSession):
        def execute(self, stmt):
            if self._selects_returned == 0 and self.select_rows:
                self._selects_returned += 1
                return MagicMock(all=lambda: list(self.select_rows))
            r = MagicMock()
            r.rowcount = 0  # conflict
            return r

    sess = _ConflictSession(
        select_rows=[
            _candidate_row(rec_id=42, symbol="NVDA", scan_date=scan),
        ]
    )

    def _get_session():
        return _CMSession(sess)

    with patch("rainier.llm_thesis.eval.get_session", side_effect=_get_session), patch(
        "rainier.llm_thesis.eval._close_on_or_before",
        side_effect=lambda sym, d: (d, 100.0) if d == scan else (d, 105.0),
    ):
        n = evaluate_horizon(today, "5d")

    assert n == 0


def test_evaluate_horizon_skips_when_prices_missing(caplog):
    """No StockPrice rows → log a warning and skip; no eval row inserted."""
    today = date(2026, 5, 7)
    scan = today - timedelta(days=5)

    candidate_session = _FakeSession(
        select_rows=[_candidate_row(rec_id=1, symbol="ZZZ", scan_date=scan)]
    )

    def _get_session():
        return _CMSession(candidate_session)

    import logging

    with patch("rainier.llm_thesis.eval.get_session", side_effect=_get_session), patch(
        "rainier.llm_thesis.eval._close_on_or_before",
        return_value=None,
    ), caplog.at_level(logging.WARNING, logger="rainier.llm_thesis.eval"):
        n = evaluate_horizon(today, "5d")

    assert n == 0
    assert any(
        "evaluate_horizon_skip_missing_prices" in r.message
        for r in caplog.records
    )


def test_evaluate_horizon_setup_long_loss_is_miss():
    today = date(2026, 5, 7)
    scan = today - timedelta(days=1)
    cand_sess = _FakeSession(
        select_rows=[
            _candidate_row(
                rec_id=1, symbol="MISS", scan_date=scan, verdict="setup_long"
            )
        ]
    )
    ins_sess = _FakeSession()
    sessions = iter([cand_sess, ins_sess])

    with patch(
        "rainier.llm_thesis.eval.get_session",
        side_effect=lambda: _CMSession(next(sessions)),
    ), patch(
        "rainier.llm_thesis.eval._close_on_or_before",
        side_effect=lambda sym, d: (d, 100.0) if d == scan else (d, 95.0),
    ):
        evaluate_horizon(today, "1d")

    assert ins_sess.inserts[0]["hit"] is False


def test_evaluate_horizon_no_setup_decline_is_hit():
    """no_setup verdict + price decline → hit (LLM was right not to buy)."""
    today = date(2026, 5, 7)
    scan = today - timedelta(days=1)
    cand_sess = _FakeSession(
        select_rows=[
            _candidate_row(
                rec_id=2, symbol="FALL", scan_date=scan, verdict="no_setup"
            )
        ]
    )
    ins_sess = _FakeSession()
    sessions = iter([cand_sess, ins_sess])

    with patch(
        "rainier.llm_thesis.eval.get_session",
        side_effect=lambda: _CMSession(next(sessions)),
    ), patch(
        "rainier.llm_thesis.eval._close_on_or_before",
        side_effect=lambda sym, d: (d, 100.0) if d == scan else (d, 90.0),
    ):
        evaluate_horizon(today, "1d")

    assert ins_sess.inserts[0]["hit"] is True


def test_evaluate_horizon_rejects_unknown_horizon():
    with pytest.raises(ValueError):
        evaluate_horizon(date(2026, 5, 7), "3d")  # type: ignore[arg-type]


def test_evaluate_horizon_zero_entry_price_skips():
    """Defensive: a 0 entry price would divide-by-zero — the helper bails."""
    today = date(2026, 5, 7)
    scan = today - timedelta(days=1)
    cand_sess = _FakeSession(
        select_rows=[
            _candidate_row(rec_id=3, symbol="BAD", scan_date=scan)
        ]
    )

    def _get_session():
        return _CMSession(cand_sess)

    with patch("rainier.llm_thesis.eval.get_session", side_effect=_get_session), patch(
        "rainier.llm_thesis.eval._close_on_or_before",
        side_effect=lambda sym, d: (d, 0.0),
    ):
        n = evaluate_horizon(today, "1d")
    assert n == 0


# ---------------------------------------------------------------------------
# compute_signal_contribution
# ---------------------------------------------------------------------------


def _eval_row(signals: list[str], return_pct: float):
    """The shape compute_signal_contribution sees: (signals_used, return_pct).

    The eval code unpacks two-tuples directly (`for sigs, ret in rows:`), so
    we pass plain tuples here instead of attribute-bearing rows.
    """
    return (list(signals), return_pct)


def test_compute_signal_contribution_finds_lift():
    rows = [
        _eval_row(["signal_a", "signal_b"], 0.05),
        _eval_row(["signal_a", "signal_b"], 0.06),
        _eval_row(["signal_a", "signal_b"], 0.04),
        _eval_row(["signal_a", "signal_b"], 0.07),
        _eval_row(["signal_a", "signal_b"], 0.05),
        _eval_row(["signal_a", "signal_b"], 0.06),
        _eval_row(["signal_b"], -0.02),
        _eval_row(["signal_b"], -0.03),
        _eval_row(["signal_b"], -0.025),
        _eval_row(["signal_b"], -0.015),
        _eval_row(["signal_b"], -0.01),
        _eval_row(["signal_b"], -0.04),
    ]
    sess = _FakeSession()
    sess.execute = lambda stmt: MagicMock(all=lambda: list(rows))  # type: ignore[assignment]

    with patch(
        "rainier.llm_thesis.eval.get_session",
        side_effect=lambda: _CMSession(sess),
    ):
        contribs = compute_signal_contribution(days=30)

    by_name = {c.name: c for c in contribs}
    assert "signal_a" in by_name
    a = by_name["signal_a"]
    assert a.n_used == 6
    assert a.n_absent == 6
    assert a.lift > 0.05
    assert a.p_value is not None
    assert a.p_value < 0.05

    # signal_b appears in every row → no absent bucket → p_value None.
    b = by_name["signal_b"]
    assert b.n_used == 12
    assert b.n_absent == 0
    assert b.p_value is None


def test_compute_signal_contribution_empty_window():
    sess = _FakeSession()
    sess.execute = lambda stmt: MagicMock(all=lambda: [])  # type: ignore[assignment]
    with patch(
        "rainier.llm_thesis.eval.get_session",
        side_effect=lambda: _CMSession(sess),
    ):
        out = compute_signal_contribution(days=30)
    assert out == []


def test_compute_signal_contribution_dataclass_shape():
    sc = SignalContribution(
        name="x", n_used=2, n_absent=2, mean_used=0.05,
        mean_absent=0.0, lift=0.05, p_value=0.04,
    )
    assert sc.lift == 0.05
    assert sc.p_value == 0.04


def test_compute_signal_contribution_rejects_unknown_horizon():
    with pytest.raises(ValueError):
        compute_signal_contribution(days=30, horizon="3d")


# ---------------------------------------------------------------------------
# compute_verdict_hit_rate
# ---------------------------------------------------------------------------


def _aggregate_row(verdict: str, horizon: str, n: int, hits: int, avg_ret: float):
    """Match the SELECT verdict, horizon, count, sum(hit), avg(return) shape."""
    return (verdict, horizon, n, hits, avg_ret)


def test_compute_verdict_hit_rate_buckets():
    aggregate_rows = [
        # 4 setup_long: 3 wins out of 4, avg 0.02
        _aggregate_row("setup_long", "5d", 4, 3, 0.02),
        # 2 watch: 2 wins (both negative returns), avg -0.02
        _aggregate_row("watch", "5d", 2, 2, -0.02),
    ]
    sess = _FakeSession()
    sess.execute = lambda stmt: MagicMock(all=lambda: list(aggregate_rows))  # type: ignore[assignment]

    with patch(
        "rainier.llm_thesis.eval.get_session",
        side_effect=lambda: _CMSession(sess),
    ):
        out = compute_verdict_hit_rate(days=30)

    long_rates = {r.horizon: r for r in out["setup_long"]}
    setup_5d = long_rates["5d"]
    assert setup_5d.n == 4
    assert abs(setup_5d.win_rate - 0.75) < 1e-9
    assert abs(setup_5d.avg_return_pct - 0.02) < 1e-9

    watch_5d = {r.horizon: r for r in out["watch"]}["5d"]
    assert watch_5d.n == 2
    assert watch_5d.win_rate == 1.0


def test_compute_verdict_hit_rate_includes_zero_buckets():
    sess = _FakeSession()
    sess.execute = lambda stmt: MagicMock(all=lambda: [])  # type: ignore[assignment]
    with patch(
        "rainier.llm_thesis.eval.get_session",
        side_effect=lambda: _CMSession(sess),
    ):
        out = compute_verdict_hit_rate(days=30)
    assert set(out.keys()) == {"setup_long", "watch", "no_setup"}
    for v in out:
        horizons = {hr.horizon for hr in out[v]}
        assert horizons == set(HORIZONS)
        for hr in out[v]:
            assert hr.n == 0
            assert hr.win_rate == 0.0


def test_hit_rate_dataclass_shape():
    hr = HitRate(
        verdict="setup_long", horizon="5d", n=10,
        win_rate=0.6, avg_return_pct=0.02,
    )
    assert hr.n == 10
    assert hr.win_rate == 0.6
