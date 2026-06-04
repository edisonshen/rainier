"""Tests for the auto-research module — emit_insight UPSERT, mark_stale,
and each of the 6 check classes against synthetic data.

The eval/research code touches Postgres-only ARRAY + JSONB columns, so we
mock `get_session` boundaries with in-test fakes (same pattern as
test_eval.py). Postgres integration tests live in `tests/integration/` and
run on demand against a real DB.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rainier.core.models import ResearchInsight
from rainier.llm_thesis.research import (
    INSIGHT_KINDS,
    SEVERITIES,
    apply_action,
    check_calibration_off,
    check_new_pattern_discovered,
    check_prompt_regression,
    check_signal_overperform,
    check_signal_underperform,
    check_verdict_drift,
    emit_insight,
    mark_stale,
    run_research,
)

# ---------------------------------------------------------------------------
# In-memory ResearchInsight session — supports emit_insight() UPSERT
# ---------------------------------------------------------------------------


class _MemSession:
    """A tiny in-memory store for ResearchInsight rows used by emit_insight().

    Implements just enough of the SQLAlchemy 2.x session interface to satisfy
    the production code: query().filter(...).first(), add(), flush(), and
    execute(update(...)) — the last for mark_stale().
    """

    def __init__(self):
        self.rows: list[ResearchInsight] = []
        self._next_id = 1
        self.committed = False

    # ----- query path used by emit_insight to look up an existing pending row
    def query(self, model):  # noqa: ARG002
        return _Query(self.rows)

    def add(self, row: ResearchInsight) -> None:
        if row.id is None:
            row.id = self._next_id
            self._next_id += 1
        if row.created_at is None:
            row.created_at = datetime.now(timezone.utc)
        if row.updated_at is None:
            row.updated_at = datetime.now(timezone.utc)
        if row.recurrence_count is None:
            row.recurrence_count = 1
        if row.status is None:
            row.status = "pending"
        self.rows.append(row)

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        return None

    # ----- update path used by mark_stale
    def execute(self, stmt):
        # mark_stale uses sqlalchemy.update(...).where(...).values(...).
        # We only support that one shape.
        try:
            params = stmt.compile().params
        except Exception:
            params = {}
        result = MagicMock()
        # Determine which rows match: status='pending' and created_at < cutoff.
        cutoff = None
        new_status = None
        # Best-effort introspection.
        try:
            new_status = stmt._values["status"].value
        except Exception:
            new_status = "stale"
        try:
            for clause in stmt._where_criteria:
                v = clause.right.value
                if isinstance(v, datetime):
                    cutoff = v
        except Exception:
            pass

        rowcount = 0
        if cutoff is not None:
            for r in self.rows:
                if r.status == "pending" and r.created_at < cutoff:
                    r.status = new_status or "stale"
                    rowcount += 1
        result.rowcount = rowcount
        _ = params  # unused, kept for symmetry
        return result

    # ----- context manager
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Query:
    def __init__(self, rows):
        self._rows = rows
        self._filters: list = []

    def filter(self, *args):
        self._filters.extend(args)
        return self

    def first(self):
        candidates = list(self._rows)
        for f in self._filters:
            try:
                col_name = f.left.key
                expected = f.right.value
            except AttributeError:
                continue
            candidates = [r for r in candidates if getattr(r, col_name, None) == expected]
        return candidates[0] if candidates else None

    def all(self):
        return list(self._rows)


# ---------------------------------------------------------------------------
# emit_insight — UPSERT semantics
# ---------------------------------------------------------------------------


def test_emit_insight_inserts_first_row():
    sess = _MemSession()
    row = emit_insight(
        kind="signal_underperform",
        subject="rank_trajectory",
        severity="warn",
        evidence={"lift": -0.001, "p_value": 0.02},
        action={"kind": "disable_signal", "target": "rank_trajectory", "params": {}},
        rationale="No lift",
        db_session=sess,
    )
    assert row.recurrence_count == 1
    assert row.status == "pending"
    assert row.kind == "signal_underperform"
    assert row.subject == "rank_trajectory"
    assert len(sess.rows) == 1


def test_emit_insight_upsert_increments_recurrence():
    sess = _MemSession()
    emit_insight(
        kind="signal_underperform",
        subject="rank_trajectory",
        severity="warn",
        evidence={"lift": -0.001},
        action={"kind": "disable_signal", "target": "rank_trajectory", "params": {}},
        rationale="No lift v1",
        db_session=sess,
    )
    second = emit_insight(
        kind="signal_underperform",
        subject="rank_trajectory",
        severity="warn",
        evidence={"lift": -0.002},  # refreshed evidence
        action={"kind": "disable_signal", "target": "rank_trajectory", "params": {}},
        rationale="No lift v2",
        db_session=sess,
    )
    # Same row, recurrence bumped to 2, evidence + rationale refreshed.
    assert len(sess.rows) == 1
    assert second.recurrence_count == 2
    assert second.evidence == {"lift": -0.002}
    assert second.rationale == "No lift v2"


def test_emit_insight_after_decided_creates_new_pending_row():
    """When a previous insight is decided (accepted/rejected/stale), the
    next emit for the same (kind, subject) must INSERT a fresh pending
    row — not refresh the decided one.
    """
    sess = _MemSession()
    first = emit_insight(
        kind="signal_overperform",
        subject="sector_momentum",
        severity="info",
        evidence={"lift": 0.006},
        action={"kind": "raise_signal_weight", "target": "sector_momentum", "params": {}},
        rationale="lift",
        db_session=sess,
    )
    # Operator marks the row accepted offline.
    first.status = "accepted"
    first.decided_at = datetime.now(timezone.utc)

    # Re-emit the same insight: the helper must NOT bump recurrence on the
    # accepted row; it should INSERT a fresh pending row.
    second = emit_insight(
        kind="signal_overperform",
        subject="sector_momentum",
        severity="info",
        evidence={"lift": 0.007},
        action={"kind": "raise_signal_weight", "target": "sector_momentum", "params": {}},
        rationale="lift again",
        db_session=sess,
    )
    assert len(sess.rows) == 2
    assert second.id != first.id
    assert second.status == "pending"
    assert second.recurrence_count == 1


def test_emit_insight_rejects_unknown_kind():
    sess = _MemSession()
    with pytest.raises(ValueError, match="Unknown insight kind"):
        emit_insight(
            kind="not_a_real_kind",
            subject="x",
            severity="warn",
            evidence={},
            action={"kind": "noop", "target": "x", "params": {}},
            rationale="r",
            db_session=sess,
        )


def test_emit_insight_rejects_unknown_severity():
    sess = _MemSession()
    with pytest.raises(ValueError, match="Unknown severity"):
        emit_insight(
            kind="signal_underperform",
            subject="x",
            severity="catastrophic",
            evidence={},
            action={"kind": "noop", "target": "x", "params": {}},
            rationale="r",
            db_session=sess,
        )


def test_emit_insight_rejects_action_without_kind():
    sess = _MemSession()
    with pytest.raises(ValueError, match="must be a dict with at least"):
        emit_insight(
            kind="signal_underperform",
            subject="x",
            severity="warn",
            evidence={},
            action={"target": "x"},  # no 'kind'
            rationale="r",
            db_session=sess,
        )


def test_insight_kinds_and_severities_are_complete():
    """Sanity check the public enums match the design's fixed set."""
    assert set(INSIGHT_KINDS) == {
        "signal_underperform",
        "signal_overperform",
        "verdict_drift",
        "calibration_off",
        "prompt_regression",
        "new_pattern_discovered",
        # Phase 2 (D6 / D7c).
        "time_stop_discovered",
        "paper_lessons",
    }
    assert set(SEVERITIES) == {"info", "warn", "critical"}


# ---------------------------------------------------------------------------
# mark_stale
# ---------------------------------------------------------------------------


def test_mark_stale_flips_old_pending_rows():
    sess = _MemSession()
    now = datetime.now(timezone.utc)
    # Old pending row (40 days back).
    old = ResearchInsight(
        id=1, kind="signal_underperform", subject="old", severity="warn",
        evidence={}, action={"kind": "disable_signal", "target": "old", "params": {}},
        rationale="r", recurrence_count=1, status="pending",
        created_at=now - timedelta(days=40),
        updated_at=now - timedelta(days=40),
    )
    sess.rows.append(old)
    # Fresh pending row (5 days back).
    fresh = ResearchInsight(
        id=2, kind="signal_underperform", subject="fresh", severity="warn",
        evidence={}, action={"kind": "disable_signal", "target": "fresh", "params": {}},
        rationale="r", recurrence_count=1, status="pending",
        created_at=now - timedelta(days=5),
        updated_at=now - timedelta(days=5),
    )
    sess.rows.append(fresh)
    # Already-accepted old row — must not be touched.
    accepted = ResearchInsight(
        id=3, kind="signal_underperform", subject="acc", severity="warn",
        evidence={}, action={"kind": "disable_signal", "target": "acc", "params": {}},
        rationale="r", recurrence_count=1, status="accepted",
        created_at=now - timedelta(days=40),
        updated_at=now - timedelta(days=40),
    )
    sess.rows.append(accepted)

    with patch(
        "rainier.llm_thesis.research.get_session",
        return_value=sess,
    ):
        n = mark_stale(days=30)

    assert n == 1
    statuses = {r.id: r.status for r in sess.rows}
    assert statuses[1] == "stale"
    assert statuses[2] == "pending"
    assert statuses[3] == "accepted"


# ---------------------------------------------------------------------------
# Synthetic eval-row helpers — drive the check_* functions
# ---------------------------------------------------------------------------


def _scan_iso(d: date) -> str:
    return d.isoformat()


def _eval_rows(
    *,
    rows: list[tuple[list[str], float, str, int | None, date]],
):
    """Match the shape `_fetch_eval_window` returns from the SQL boundary."""
    return [(sigs, ret, verdict, conf, _scan_iso(scan)) for sigs, ret, verdict, conf, scan in rows]


# ---------------------------------------------------------------------------
# check_signal_underperform
# ---------------------------------------------------------------------------


def test_check_signal_underperform_emits_when_signal_flat():
    sess = _MemSession()
    today = date(2026, 5, 7)
    # signal "flat" used: returns ~0; absent: returns ~0.05. Lift very negative
    # and signal is statistically distinct from absent set.
    rows = []
    for _ in range(10):
        rows.append((["flat", "other"], 0.0, "setup_long", 7, today))
    for _ in range(10):
        rows.append((["other"], 0.05, "setup_long", 7, today))
    fetched = _eval_rows(rows=rows)

    with patch(
        "rainier.llm_thesis.research._fetch_eval_window", return_value=fetched
    ):
        out = check_signal_underperform(
            eval_date=today, days=30, db_session=sess
        )

    names = {r.subject for r in out}
    assert "flat" in names
    flat = next(r for r in out if r.subject == "flat")
    assert flat.severity == "warn"
    assert flat.action["kind"] == "disable_signal"
    assert flat.action["target"] == "flat"
    assert flat.evidence["lift"] < 0


def test_check_signal_underperform_no_emit_when_signal_lifts():
    sess = _MemSession()
    today = date(2026, 5, 7)
    rows = []
    for _ in range(10):
        rows.append((["winner"], 0.05, "setup_long", 7, today))
    for _ in range(10):
        rows.append(([], 0.0, "setup_long", 7, today))
    fetched = _eval_rows(rows=rows)

    with patch(
        "rainier.llm_thesis.research._fetch_eval_window", return_value=fetched
    ):
        out = check_signal_underperform(
            eval_date=today, days=30, db_session=sess
        )
    assert out == []


# ---------------------------------------------------------------------------
# check_signal_overperform
# ---------------------------------------------------------------------------


def test_check_signal_overperform_emits_when_signal_lifts_strongly():
    sess = _MemSession()
    today = date(2026, 5, 7)
    rows = []
    for _ in range(15):
        rows.append((["star"], 0.04, "setup_long", 7, today))
    for _ in range(15):
        rows.append(([], 0.0, "setup_long", 7, today))
    fetched = _eval_rows(rows=rows)

    with patch(
        "rainier.llm_thesis.research._fetch_eval_window", return_value=fetched
    ):
        out = check_signal_overperform(
            eval_date=today, days=30, db_session=sess
        )

    names = {r.subject for r in out}
    assert "star" in names
    star = next(r for r in out if r.subject == "star")
    assert star.severity == "info"
    assert star.action["kind"] == "raise_signal_weight"
    assert star.action["params"].get("factor") == 1.2


# ---------------------------------------------------------------------------
# check_verdict_drift
# ---------------------------------------------------------------------------


def test_check_verdict_drift_emits_when_spread_collapses():
    sess = _MemSession()
    today = date(2026, 5, 7)

    # Long-window: setup_long is 80% hit (positive returns), no_setup is 20% hit
    # (mostly positive returns -> losing the no_setup bet). Spread = 0.6.
    long_rows = []
    for _ in range(20):
        long_rows.append(([], 0.02, "setup_long", 7, today - timedelta(days=20)))
    for _ in range(5):
        long_rows.append(([], -0.01, "setup_long", 7, today - timedelta(days=21)))
    for _ in range(20):
        long_rows.append(([], 0.02, "no_setup", 3, today - timedelta(days=22)))
    for _ in range(5):
        long_rows.append(([], -0.01, "no_setup", 3, today - timedelta(days=23)))
    # Short-window subset (last 14 days): each verdict ~50% hit. Spread ~0.
    short_rows = []
    for _ in range(5):
        short_rows.append(([], 0.01, "setup_long", 7, today - timedelta(days=2)))
    for _ in range(5):
        short_rows.append(([], -0.01, "setup_long", 7, today - timedelta(days=3)))
    for _ in range(5):
        short_rows.append(([], -0.01, "no_setup", 3, today - timedelta(days=4)))
    for _ in range(5):
        short_rows.append(([], 0.01, "no_setup", 3, today - timedelta(days=5)))

    def _fetch(*, days: int, horizon: str, eval_date):
        return _eval_rows(
            rows=long_rows if days >= 30 else short_rows
        )

    with patch(
        "rainier.llm_thesis.research._fetch_eval_window", side_effect=_fetch
    ), patch("rainier.llm_thesis.research.get_session") as gs:
        # The verdict_drift check looks up the latest prompt_template via
        # get_session — return a no-op chain so the lookup yields None.
        cm = MagicMock()
        cm.__enter__.return_value.execute.return_value.first.return_value = None
        cm.__exit__.return_value = False
        gs.return_value = cm

        out = check_verdict_drift(eval_date=today, db_session=sess)

    assert len(out) == 1
    assert out[0].kind == "verdict_drift"
    assert out[0].severity == "critical"
    assert out[0].action["kind"] == "bump_prompt_version"
    assert out[0].evidence["drop"] >= 0.10


def test_check_verdict_drift_no_emit_when_stable():
    sess = _MemSession()
    today = date(2026, 5, 7)
    rows = []
    for _ in range(20):
        rows.append(([], 0.02, "setup_long", 7, today))
    for _ in range(20):
        rows.append(([], 0.02, "no_setup", 3, today))

    def _fetch(*, days, horizon, eval_date):
        return _eval_rows(rows=rows)

    with patch(
        "rainier.llm_thesis.research._fetch_eval_window", side_effect=_fetch
    ), patch("rainier.llm_thesis.research.get_session") as gs:
        cm = MagicMock()
        cm.__enter__.return_value.execute.return_value.first.return_value = None
        cm.__exit__.return_value = False
        gs.return_value = cm

        out = check_verdict_drift(eval_date=today, db_session=sess)
    assert out == []


# ---------------------------------------------------------------------------
# check_calibration_off
# ---------------------------------------------------------------------------


def test_check_calibration_off_emits_on_overconfidence():
    sess = _MemSession()
    today = date(2026, 5, 7)
    # confidence 9 -> hit rate ~0.1 (massively over-confident); confidence 7 ->
    # hit rate ~0.2; confidence 5 -> hit rate ~0.5. Slope way below 1.0.
    rows = []
    rows.extend([([], 0.0, "setup_long", 9, today)] * 9)  # 9 misses
    rows.extend([([], 0.05, "setup_long", 9, today)])     # 1 hit
    rows.extend([([], 0.0, "setup_long", 7, today)] * 8)  # 8 misses
    rows.extend([([], 0.05, "setup_long", 7, today)] * 2) # 2 hits
    rows.extend([([], 0.0, "setup_long", 5, today)] * 5)
    rows.extend([([], 0.05, "setup_long", 5, today)] * 5)
    fetched = _eval_rows(rows=rows)

    with patch(
        "rainier.llm_thesis.research._fetch_eval_window", return_value=fetched
    ):
        out = check_calibration_off(eval_date=today, db_session=sess)

    assert len(out) == 1
    insight = out[0]
    assert insight.kind == "calibration_off"
    assert insight.severity == "warn"
    assert insight.action["kind"] == "noop"
    assert insight.evidence["deviation_from_unity"] > 0.3


def test_check_calibration_off_no_emit_when_calibrated():
    sess = _MemSession()
    today = date(2026, 5, 7)
    rows = []
    # confidence 9 -> 9/10 hits, confidence 5 -> 5/10 hits, confidence 3 -> 3/10
    rows.extend([([], 0.05, "setup_long", 9, today)] * 9)
    rows.extend([([], 0.0, "setup_long", 9, today)])
    rows.extend([([], 0.05, "setup_long", 5, today)] * 5)
    rows.extend([([], 0.0, "setup_long", 5, today)] * 5)
    rows.extend([([], 0.05, "setup_long", 3, today)] * 3)
    rows.extend([([], 0.0, "setup_long", 3, today)] * 7)
    fetched = _eval_rows(rows=rows)

    with patch(
        "rainier.llm_thesis.research._fetch_eval_window", return_value=fetched
    ):
        out = check_calibration_off(eval_date=today, db_session=sess)
    assert out == []


def test_check_calibration_off_skips_when_too_few_bins():
    sess = _MemSession()
    today = date(2026, 5, 7)
    rows = [([], 0.0, "setup_long", 5, today)] * 10
    fetched = _eval_rows(rows=rows)
    with patch(
        "rainier.llm_thesis.research._fetch_eval_window", return_value=fetched
    ):
        out = check_calibration_off(eval_date=today, db_session=sess)
    assert out == []


# ---------------------------------------------------------------------------
# check_new_pattern_discovered
# ---------------------------------------------------------------------------


def test_check_new_pattern_discovered_emits_for_recurring_winner():
    sess = _MemSession()
    today = date(2026, 5, 7)
    # 4 rows mention "ascending_triangle" + 1 mentions "noisy_pattern".
    # All ascending_triangle rows have positive returns; noisy_pattern is below
    # the min_occurrences threshold.
    fake_rows = [
        (1, {"patterns_in_chart_not_in_indicators": ["ascending_triangle"]}, 0.04),
        (2, {"patterns_in_chart_not_in_indicators": ["ascending_triangle", "noisy_pattern"]}, 0.03),
        (3, {"patterns_in_chart_not_in_indicators": ["ascending_triangle"]}, 0.02),
        (4, {"patterns_in_chart_not_in_indicators": ["ascending_triangle"]}, 0.05),
        (5, {"patterns_in_chart_not_in_indicators": []}, 0.0),
    ]
    cm = MagicMock()
    cm.__enter__.return_value.execute.return_value.all.return_value = fake_rows
    cm.__exit__.return_value = False

    with patch(
        "rainier.llm_thesis.research.get_session", return_value=cm
    ):
        out = check_new_pattern_discovered(
            eval_date=today, days=30, db_session=sess
        )

    subjects = {r.subject for r in out}
    assert "ascending_triangle" in subjects
    assert "noisy_pattern" not in subjects
    insight = next(r for r in out if r.subject == "ascending_triangle")
    assert insight.action["kind"] == "noop"
    assert insight.evidence["occurrences"] == 4


def test_check_new_pattern_discovered_skips_negative_mean():
    sess = _MemSession()
    today = date(2026, 5, 7)
    fake_rows = [
        (1, {"patterns_in_chart_not_in_indicators": ["loser"]}, -0.04),
        (2, {"patterns_in_chart_not_in_indicators": ["loser"]}, -0.03),
        (3, {"patterns_in_chart_not_in_indicators": ["loser"]}, -0.02),
    ]
    cm = MagicMock()
    cm.__enter__.return_value.execute.return_value.all.return_value = fake_rows
    cm.__exit__.return_value = False

    with patch(
        "rainier.llm_thesis.research.get_session", return_value=cm
    ):
        out = check_new_pattern_discovered(
            eval_date=today, days=30, db_session=sess
        )
    assert out == []


# ---------------------------------------------------------------------------
# check_prompt_regression
# ---------------------------------------------------------------------------


def test_check_prompt_regression_emits_when_new_prompt_worse():
    sess = _MemSession()
    today = date(2026, 5, 7)
    older_at = datetime(2026, 4, 1, tzinfo=timezone.utc)
    newer_at = datetime(2026, 4, 25, tzinfo=timezone.utc)
    # v1 -> consistent +5% returns; v2 -> consistent -3% returns.
    fake_rows = [("v1", older_at, 0.05)] * 12 + [("v2", newer_at, -0.03)] * 12

    cm = MagicMock()
    cm.__enter__.return_value.execute.return_value.all.return_value = fake_rows
    cm.__exit__.return_value = False

    with patch(
        "rainier.llm_thesis.research.get_session", return_value=cm
    ):
        out = check_prompt_regression(
            eval_date=today, days=30, db_session=sess
        )

    assert len(out) == 1
    insight = out[0]
    assert insight.kind == "prompt_regression"
    assert insight.severity == "critical"
    assert insight.action["kind"] == "bump_prompt_version"
    assert insight.action["target"] == "v1"  # rollback to older
    assert insight.subject == "v2"


def test_check_prompt_regression_no_emit_when_stable():
    sess = _MemSession()
    today = date(2026, 5, 7)
    older_at = datetime(2026, 4, 1, tzinfo=timezone.utc)
    newer_at = datetime(2026, 4, 25, tzinfo=timezone.utc)
    fake_rows = [("v1", older_at, 0.02)] * 10 + [("v2", newer_at, 0.025)] * 10
    cm = MagicMock()
    cm.__enter__.return_value.execute.return_value.all.return_value = fake_rows
    cm.__exit__.return_value = False
    with patch(
        "rainier.llm_thesis.research.get_session", return_value=cm
    ):
        out = check_prompt_regression(
            eval_date=today, days=30, db_session=sess
        )
    assert out == []


def test_check_prompt_regression_no_emit_when_single_prompt():
    sess = _MemSession()
    today = date(2026, 5, 7)
    older_at = datetime(2026, 4, 1, tzinfo=timezone.utc)
    fake_rows = [("v1", older_at, 0.02)] * 10
    cm = MagicMock()
    cm.__enter__.return_value.execute.return_value.all.return_value = fake_rows
    cm.__exit__.return_value = False
    with patch(
        "rainier.llm_thesis.research.get_session", return_value=cm
    ):
        out = check_prompt_regression(
            eval_date=today, days=30, db_session=sess
        )
    assert out == []


# ---------------------------------------------------------------------------
# run_research orchestrator — calls all 6 checks, isolates failures
# ---------------------------------------------------------------------------


def test_run_research_runs_all_checks_and_collects_emissions():
    """Each individual check is patched to emit a single insight; the
    orchestrator should collect 8 total (6 PR3 + 2 Phase-2)."""
    sentinel_rows: list[Any] = []

    def _fake(name):
        def fn(**_kwargs):
            row = ResearchInsight(
                id=len(sentinel_rows) + 1,
                kind=name,
                subject="x",
                severity="warn",
                evidence={},
                action={"kind": "noop", "target": "x", "params": {}},
                rationale="r",
                recurrence_count=1,
                status="pending",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            sentinel_rows.append(row)
            return [row]
        return fn

    with (
        patch(
            "rainier.llm_thesis.research.check_signal_underperform",
            side_effect=_fake("signal_underperform"),
        ),
        patch(
            "rainier.llm_thesis.research.check_signal_overperform",
            side_effect=_fake("signal_overperform"),
        ),
        patch(
            "rainier.llm_thesis.research.check_verdict_drift",
            side_effect=_fake("verdict_drift"),
        ),
        patch(
            "rainier.llm_thesis.research.check_calibration_off",
            side_effect=_fake("calibration_off"),
        ),
        patch(
            "rainier.llm_thesis.research.check_new_pattern_discovered",
            side_effect=_fake("new_pattern_discovered"),
        ),
        patch(
            "rainier.llm_thesis.research.check_prompt_regression",
            side_effect=_fake("prompt_regression"),
        ),
        patch(
            "rainier.llm_thesis.research.discover_time_stop",
            side_effect=_fake("time_stop_discovered"),
        ),
        patch(
            "rainier.llm_thesis.research.check_paper_lessons",
            side_effect=_fake("paper_lessons"),
        ),
    ):
        out = run_research(eval_date=date(2026, 5, 7))

    assert len(out) == 8
    assert {r.kind for r in out} == set(INSIGHT_KINDS)


def test_run_research_isolates_failures():
    """One check raising must not crash the orchestrator."""
    def _ok(**_kwargs):
        return []

    def _boom(**_kwargs):
        raise RuntimeError("intentional")

    with (
        patch(
            "rainier.llm_thesis.research.check_signal_underperform",
            side_effect=_boom,
        ),
        patch(
            "rainier.llm_thesis.research.check_signal_overperform",
            side_effect=_ok,
        ),
        patch(
            "rainier.llm_thesis.research.check_verdict_drift", side_effect=_ok
        ),
        patch(
            "rainier.llm_thesis.research.check_calibration_off", side_effect=_ok
        ),
        patch(
            "rainier.llm_thesis.research.check_new_pattern_discovered",
            side_effect=_ok,
        ),
        patch(
            "rainier.llm_thesis.research.check_prompt_regression", side_effect=_ok
        ),
        patch(
            "rainier.llm_thesis.research.discover_time_stop", side_effect=_ok
        ),
        patch(
            "rainier.llm_thesis.research.check_paper_lessons", side_effect=_ok
        ),
    ):
        out = run_research(eval_date=date(2026, 5, 7))
    # No crash; the failing check just contributed nothing.
    assert out == []


# ---------------------------------------------------------------------------
# apply_action smoke tests live in test_yaml_mutation.py — re-import here
# only to confirm the symbol is wired in.
# ---------------------------------------------------------------------------


def test_apply_action_is_callable():
    """Sanity: the dispatcher is exported from research.py."""
    assert callable(apply_action)


# ---------------------------------------------------------------------------
# /review iter-1 fixes
# ---------------------------------------------------------------------------


def test_emit_insight_owns_session_via_contextmanager():
    """[P0 from review] When the caller does NOT pass `db_session`, the helper
    must drive the session through `get_session()`'s contextmanager so a
    SQL-level exception triggers rollback() + close() instead of leaking an
    open transaction. Previously the code called `get_session().__enter__()`
    directly, which bypassed the @contextmanager exception handling.
    """
    cm_calls: dict[str, int] = {"enter": 0, "exit": 0, "rollback": 0, "commit": 0}

    class _SpySession:
        def __init__(self):
            self.committed = False
            self.rolled_back = False

        def query(self, *_a, **_kw):
            return _Query([])

        def add(self, _r):
            pass

        def flush(self):
            return None

        def commit(self):
            cm_calls["commit"] += 1
            self.committed = True

        def rollback(self):
            cm_calls["rollback"] += 1
            self.rolled_back = True

        def close(self):
            pass

    spy = _SpySession()

    class _CM:
        def __enter__(self):
            cm_calls["enter"] += 1
            return spy

        def __exit__(self, exc_type, exc, tb):
            cm_calls["exit"] += 1
            if exc_type is not None:
                spy.rollback()
            else:
                spy.commit()
            return False

    with patch("rainier.llm_thesis.research.get_session", return_value=_CM()):
        emit_insight(
            kind="signal_underperform",
            subject="rank_trajectory",
            severity="warn",
            evidence={},
            action={"kind": "disable_signal", "target": "x", "params": {}},
            rationale="r",
        )

    # The helper must have entered AND exited the contextmanager exactly once.
    assert cm_calls["enter"] == 1
    assert cm_calls["exit"] == 1
    assert cm_calls["commit"] == 1
    assert cm_calls["rollback"] == 0


def test_check_prompt_regression_emits_at_most_one_per_newer_prompt():
    """[P2 from review] When three prompts exist (v1 older, v2 mid, v3
    newest) and v3 is statistically worse than BOTH v1 and v2, the previous
    implementation called emit_insight twice (once per pair), each UPSERT-ing
    the same `subject=v3` row. That silently dropped earlier evidence and
    inflated recurrence_count. The fix picks the strongest pair (lowest p)
    and calls emit_insight exactly once per `newer`.
    """
    sess = _MemSession()
    today = date(2026, 5, 7)

    v1_at = datetime(2026, 4, 1, tzinfo=timezone.utc)
    v2_at = datetime(2026, 4, 15, tzinfo=timezone.utc)
    v3_at = datetime(2026, 4, 25, tzinfo=timezone.utc)

    # v1 returns +0.05; v2 returns +0.04 (slightly worse); v3 returns -0.03
    # (much worse). v3 is significantly worse than BOTH older prompts.
    fake_rows = (
        [("v1", v1_at, 0.05)] * 12
        + [("v2", v2_at, 0.04)] * 12
        + [("v3", v3_at, -0.03)] * 12
    )

    cm = MagicMock()
    cm.__enter__.return_value.execute.return_value.all.return_value = fake_rows
    cm.__exit__.return_value = False

    with patch(
        "rainier.llm_thesis.research.get_session", return_value=cm
    ):
        out = check_prompt_regression(
            eval_date=today, days=30, db_session=sess
        )

    # v2 is barely worse than v1 (likely no significant p), v3 is much worse
    # than both. So we expect at most one emission for v3 — never one for v2
    # AND v3 from the same `newer=v3` group.
    v3_emissions = [r for r in out if r.subject == "v3"]
    assert len(v3_emissions) == 1, (
        "v3 must produce a single insight (best-pair semantics), "
        "not one per older predecessor"
    )
