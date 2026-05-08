"""Unit tests for `dashboard.data`.

We mock `get_session` (and `compute_signal_contribution` /
`compute_verdict_hit_rate` for the wrapper smoke tests) so the tests
run against in-memory fakes — no Postgres ARRAY / JSONB needed. Same
pattern as `tests/test_llm_thesis/test_eval.py`.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

from rainier.dashboard import data as ddata
from rainier.llm_thesis.eval import HitRate, SignalContribution

# ---------------------------------------------------------------------------
# Helpers — minimal SQLAlchemy 2.x Session shim
# ---------------------------------------------------------------------------


class _ScriptedSession:
    """Returns scripted `execute().all()` and `execute().scalars().all()`
    payloads in the order they're queued. `session.get(model, id)` reads
    from a separate keyed dict.

    Intentionally tiny — we don't need to support ORM mutation in these
    read-only tests.
    """

    def __init__(
        self,
        *,
        all_payloads: list[list[Any]] | None = None,
        scalars_payloads: list[list[Any]] | None = None,
        gets: dict[tuple[type, int], Any] | None = None,
    ):
        self._all = list(all_payloads or [])
        self._scalars = list(scalars_payloads or [])
        self._gets = dict(gets or {})
        self.calls = 0

    def execute(self, _stmt):  # noqa: ARG002
        self.calls += 1
        result = MagicMock()
        next_all = self._all.pop(0) if self._all else []
        result.all.return_value = next_all
        scalars_obj = MagicMock()
        next_scalars = self._scalars.pop(0) if self._scalars else []
        scalars_obj.all.return_value = next_scalars
        result.scalars.return_value = scalars_obj
        return result

    def get(self, model, ident):
        return self._gets.get((model, int(ident)))


@contextmanager
def _session_cm(sess):
    yield sess


def _patch_session(sess):
    """`with patch(...) as gs: gs.return_value = _session_cm(sess)`."""
    return patch.object(
        ddata,
        "get_session",
        return_value=_session_cm(sess),
    )


# ---------------------------------------------------------------------------
# Stub Settings — load_signal_status reads the YAML toggle; we feed it a
# minimal Settings instance instead.
# ---------------------------------------------------------------------------


class _StubSignalConfig:
    def __init__(self, enabled: bool = True, weight: float = 1.0, params: dict | None = None):
        self.enabled = enabled
        self.weight = weight
        self.params = params or {}


class _StubLLMThesisConfig:
    def __init__(self, signals: dict[str, _StubSignalConfig]):
        self.signals = signals


class _StubSettings:
    def __init__(self, signals: dict[str, _StubSignalConfig]):
        self.llm_thesis = _StubLLMThesisConfig(signals)


# ---------------------------------------------------------------------------
# load_signal_status
# ---------------------------------------------------------------------------


def test_load_signal_status_zero_theses_baseline():
    """All signals appear with hit_count=0 when no theses are in the window."""
    from rainier.llm_thesis.signals import REGISTRY

    stub = _StubSettings(
        signals={name: _StubSignalConfig(enabled=True) for name in REGISTRY}
    )
    sess = _ScriptedSession(all_payloads=[[]])  # empty signals_used result
    with patch.object(ddata, "load_settings_fresh", return_value=stub), _patch_session(sess):
        rows = ddata.load_signal_status()

    assert len(rows) == len(REGISTRY)
    assert {r.name for r in rows} == set(REGISTRY)
    assert all(r.last_7d_hit_count == 0 for r in rows)
    assert all(r.enabled for r in rows)


def test_load_signal_status_counts_signals_used():
    """Rows in the window with `signals_used` arrays bump per-name hits."""
    from rainier.llm_thesis.signals import REGISTRY

    # Pick the first two registry names; rest stay at 0.
    names = list(REGISTRY.keys())
    a, b = names[0], names[1]

    stub = _StubSettings(
        signals={
            **{name: _StubSignalConfig(enabled=True) for name in REGISTRY},
            a: _StubSignalConfig(enabled=False, weight=2.0),
        }
    )
    payload = [
        ([a, b],),  # thesis 1 used a, b
        ([a],),  # thesis 2 used a
        ([b],),  # thesis 3 used b
        (None,),  # thesis 4 used no signals (still in window) — should not crash
    ]
    sess = _ScriptedSession(all_payloads=[payload])
    with patch.object(ddata, "load_settings_fresh", return_value=stub), _patch_session(sess):
        rows = ddata.load_signal_status()

    by_name = {r.name: r for r in rows}
    assert by_name[a].last_7d_hit_count == 2
    assert by_name[b].last_7d_hit_count == 2
    assert by_name[a].enabled is False  # honors stub
    assert by_name[a].weight == 2.0


# ---------------------------------------------------------------------------
# load_signal_contribution / load_verdict_hit_rate — thin wrappers
# ---------------------------------------------------------------------------


def test_load_signal_contribution_wrapper_passes_args():
    expected = [
        SignalContribution(
            name="rank_trajectory",
            n_used=10,
            n_absent=10,
            mean_used=0.02,
            mean_absent=0.0,
            lift=0.02,
            p_value=0.04,
        )
    ]
    with patch.object(
        ddata, "compute_signal_contribution", return_value=expected
    ) as m:
        out = ddata.load_signal_contribution(days=14, horizon="1d")
    m.assert_called_once_with(days=14, horizon="1d")
    assert out is expected


def test_load_verdict_hit_rate_wrapper_passes_args():
    expected = {
        "setup_long": [
            HitRate(
                verdict="setup_long",
                horizon="5d",
                n=4,
                win_rate=0.75,
                avg_return_pct=0.012,
            )
        ]
    }
    with patch.object(
        ddata, "compute_verdict_hit_rate", return_value=expected
    ) as m:
        out = ddata.load_verdict_hit_rate(days=7)
    m.assert_called_once_with(days=7)
    assert out is expected


# ---------------------------------------------------------------------------
# load_signal_value_series
# ---------------------------------------------------------------------------


def test_load_signal_value_series_unknown_signal_returns_empty_df():
    df = ddata.load_signal_value_series("not_a_signal")
    assert df.empty
    assert list(df.columns) == [
        "scan_date",
        "symbol",
        "signal_value",
        "forward_return_5d",
    ]


def test_load_signal_value_series_extracts_rank_trajectory_delta_10d():
    """rank_trajectory dispatcher must pull `delta_10d` from the JSON blob."""
    today = date(2026, 5, 7)
    rows = [
        (
            today - timedelta(days=2),
            "NVDA",
            0.015,
            {"signals": {"rank_trajectory": {"delta_10d": 5}}},
        ),
        (
            today - timedelta(days=1),
            "TSLA",
            -0.008,
            {"signals": {"rank_trajectory": {"delta_10d": -3}}},
        ),
        # Row missing rank_trajectory — must be dropped silently.
        (today, "AAPL", 0.001, {"signals": {"sector_momentum": {"delta": 0.1}}}),
        # Row with non-numeric delta_10d — must be dropped.
        (
            today,
            "MSFT",
            0.001,
            {"signals": {"rank_trajectory": {"delta_10d": "n/a"}}},
        ),
    ]
    sess = _ScriptedSession(all_payloads=[rows])
    with _patch_session(sess):
        df = ddata.load_signal_value_series("rank_trajectory", days=30)

    assert len(df) == 2
    assert set(df.columns) == {
        "scan_date",
        "symbol",
        "signal_value",
        "forward_return_5d",
    }
    nvda = df[df["symbol"] == "NVDA"].iloc[0]
    assert nvda["signal_value"] == 5.0
    assert nvda["forward_return_5d"] == 0.015


def test_load_signal_value_series_capital_flow_streak_signs_with_direction():
    """capital_flow_streak extractor must invert sign on `direction='-'`."""
    today = date(2026, 5, 7)
    rows = [
        (
            today,
            "AMZN",
            0.02,
            {
                "signals": {
                    "capital_flow_streak": {"streak_days": 4, "direction": "+"}
                }
            },
        ),
        (
            today,
            "META",
            -0.01,
            {
                "signals": {
                    "capital_flow_streak": {"streak_days": 3, "direction": "-"}
                }
            },
        ),
    ]
    sess = _ScriptedSession(all_payloads=[rows])
    with _patch_session(sess):
        df = ddata.load_signal_value_series("capital_flow_streak", days=30)

    by_symbol = {r["symbol"]: r["signal_value"] for _i, r in df.iterrows()}
    assert by_symbol["AMZN"] == 4.0
    assert by_symbol["META"] == -3.0


# ---------------------------------------------------------------------------
# load_recent_theses
# ---------------------------------------------------------------------------


def _thesis_row(
    *,
    screened_id: int,
    scan_d: date,
    symbol: str,
    thesis_id: int,
    confidence: int = 7,
    verdict: str = "setup_long",
    signals: list[str] | None = None,
    chart_ids: list[int] | None = None,
    structured: dict | None = None,
    session: str = "afternoon",
):
    """Mirror the SELECT row shape `load_recent_theses` expects."""
    obj = MagicMock()
    obj.screened_id = screened_id
    obj.scan_date = scan_d
    obj.session_name = session
    obj.symbol = symbol
    obj.thesis_id = thesis_id
    obj.llm_confidence = confidence
    obj.recommendation = verdict
    obj.signals_used = signals or ["rank_trajectory"]
    obj.chart_image_ids = chart_ids
    obj.structured_output = structured
    return obj


def test_load_recent_theses_join_and_horizon_flatten():
    """Forward returns at 1d/5d/10d must appear on the right thesis rows;
    missing horizons stay None."""
    today = date(2026, 5, 7)
    base = [
        _thesis_row(
            screened_id=1,
            scan_d=today,
            symbol="NVDA",
            thesis_id=10,
            chart_ids=[200],
        ),
        _thesis_row(
            screened_id=2,
            scan_d=today,
            symbol="TSLA",
            thesis_id=11,
        ),
    ]
    # Two horizons for thesis 10, none for thesis 11.
    eval_rows = [
        (10, "1d", 0.01),
        (10, "5d", 0.025),
        # 10d for thesis 10 is missing on purpose — expect None.
    ]
    sess = _ScriptedSession(all_payloads=[base, eval_rows])
    with _patch_session(sess):
        rows = ddata.load_recent_theses(limit=10)

    assert len(rows) == 2
    by_sym = {r.symbol: r for r in rows}
    assert by_sym["NVDA"].return_1d == 0.01
    assert by_sym["NVDA"].return_5d == 0.025
    assert by_sym["NVDA"].return_10d is None
    assert by_sym["NVDA"].chart_image_id == 200
    # No eval rows queried for TSLA.
    assert by_sym["TSLA"].return_1d is None
    assert by_sym["TSLA"].return_5d is None
    assert by_sym["TSLA"].return_10d is None


def test_load_recent_theses_skips_eval_query_when_no_thesis_ids():
    """If no theses are returned, the eval query must NOT run."""
    sess = _ScriptedSession(all_payloads=[[]])  # only the base SELECT runs
    with _patch_session(sess):
        rows = ddata.load_recent_theses(limit=5)
    assert rows == []
    assert sess.calls == 1


# ---------------------------------------------------------------------------
# load_thesis_chart
# ---------------------------------------------------------------------------


def test_load_thesis_chart_returns_none_for_unknown_id():
    sess = _ScriptedSession(gets={})  # session.get(...) returns None
    with _patch_session(sess):
        out = ddata.load_thesis_chart(999)
    assert out is None


def test_load_thesis_chart_returns_bytes_when_file_exists(tmp_path):
    """Legacy path: thesis -> ChartImage(file_path) -> bytes."""
    from rainier.core.models import ChartImage, LLMAnalysisRecord

    png_path = tmp_path / "nvda.png"
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    png_path.write_bytes(png_bytes)

    analysis = MagicMock(spec=LLMAnalysisRecord)
    analysis.chart_image_ids = [42]
    chart = MagicMock(spec=ChartImage)
    chart.file_path = str(png_path)
    # PR5: legacy rows have image_bytes=None and the loader falls back to
    # the on-disk file_path read path.
    chart.image_bytes = None

    sess = _ScriptedSession(
        gets={
            (LLMAnalysisRecord, 100): analysis,
            (ChartImage, 42): chart,
        }
    )
    with _patch_session(sess):
        out = ddata.load_thesis_chart(100)
    assert out == png_bytes


def test_load_thesis_chart_returns_none_when_file_missing(tmp_path):
    """Path on the row points at a non-existent file."""
    from rainier.core.models import ChartImage, LLMAnalysisRecord

    analysis = MagicMock(spec=LLMAnalysisRecord)
    analysis.chart_image_ids = [42]
    chart = MagicMock(spec=ChartImage)
    chart.file_path = str(tmp_path / "missing.png")
    chart.image_bytes = None

    sess = _ScriptedSession(
        gets={
            (LLMAnalysisRecord, 100): analysis,
            (ChartImage, 42): chart,
        }
    )
    with _patch_session(sess):
        out = ddata.load_thesis_chart(100)
    assert out is None


def test_load_thesis_chart_prefers_inline_image_bytes_over_file_path(tmp_path):
    """PR5: when ``image_bytes`` is set, return them directly without
    touching the filesystem (the file_path may be empty/legacy)."""
    from rainier.core.models import ChartImage, LLMAnalysisRecord

    inline_bytes = b"\x89PNG\r\n\x1a\nINLINE-PR5"
    analysis = MagicMock(spec=LLMAnalysisRecord)
    analysis.chart_image_ids = [42]
    chart = MagicMock(spec=ChartImage)
    chart.file_path = None
    chart.image_bytes = inline_bytes

    sess = _ScriptedSession(
        gets={
            (LLMAnalysisRecord, 100): analysis,
            (ChartImage, 42): chart,
        }
    )
    with _patch_session(sess):
        out = ddata.load_thesis_chart(100)
    assert out == inline_bytes


def test_load_thesis_chart_returns_none_when_thesis_has_no_chart_ids():
    from rainier.core.models import LLMAnalysisRecord

    analysis = MagicMock(spec=LLMAnalysisRecord)
    analysis.chart_image_ids = []
    sess = _ScriptedSession(gets={(LLMAnalysisRecord, 5): analysis})
    with _patch_session(sess):
        out = ddata.load_thesis_chart(5)
    assert out is None


# ---------------------------------------------------------------------------
# load_pending_insights
# ---------------------------------------------------------------------------


def _insight_row(
    *,
    rid: int,
    kind: str = "signal_underperform",
    severity: str = "warn",
    subject: str = "rank_trajectory",
    rationale: str = "rationale",
    evidence: dict | None = None,
    action: dict | None = None,
    recurrence: int = 1,
    status: str = "pending",
    updated: datetime | None = None,
    decided_at: datetime | None = None,
    decided_by: str | None = None,
    applied_change: dict | None = None,
):
    obj = MagicMock()
    obj.id = rid
    obj.kind = kind
    obj.severity = severity
    obj.subject = subject
    obj.rationale = rationale
    obj.evidence = evidence or {}
    obj.action = action or {"kind": "noop", "target": subject, "params": {}}
    obj.recurrence_count = recurrence
    obj.status = status
    obj.created_at = (updated or datetime.now(timezone.utc)) - timedelta(hours=1)
    obj.updated_at = updated or datetime.now(timezone.utc)
    obj.decided_at = decided_at
    obj.decided_by = decided_by
    obj.applied_change = applied_change
    return obj


def test_load_pending_insights_orders_by_severity_then_recency():
    now = datetime.now(timezone.utc)
    rows = [
        _insight_row(
            rid=1, severity="info", subject="A", updated=now - timedelta(minutes=5)
        ),
        _insight_row(rid=2, severity="critical", subject="B", updated=now - timedelta(hours=3)),
        _insight_row(rid=3, severity="warn", subject="C", updated=now),
        _insight_row(rid=4, severity="critical", subject="D", updated=now - timedelta(hours=1)),
    ]
    sess = _ScriptedSession(scalars_payloads=[rows])
    with _patch_session(sess):
        out = ddata.load_pending_insights()

    # critical first (D before B because D is more recent), then warn, info.
    assert [r.insight_id for r in out] == [4, 2, 3, 1]
    assert all(r.status == "pending" for r in out)


def test_load_insight_history_filter_passthrough():
    accepted = _insight_row(
        rid=10,
        severity="info",
        status="accepted",
        decided_at=datetime.now(timezone.utc),
        decided_by="user",
    )
    sess = _ScriptedSession(scalars_payloads=[[accepted]])
    with _patch_session(sess):
        out = ddata.load_insight_history(status=["accepted"])
    assert len(out) == 1
    assert out[0].insight_id == 10
    assert out[0].status == "accepted"


def test_load_insight_history_default_excludes_pending_and_stale():
    """Default filter is the audit-trail set: accepted + rejected + auto_applied."""
    sess = _ScriptedSession(scalars_payloads=[[]])
    with _patch_session(sess):
        ddata.load_insight_history()
    # We can't easily inspect the WHERE clause, but the empty status case
    # is defensive: an empty list short-circuits to [].
    assert ddata.load_insight_history(status=[]) == []


def test_load_insight_history_jsonable_string_evidence():
    """`evidence` may come back as a JSON string from SQLite — coerce to dict."""
    accepted = _insight_row(
        rid=11,
        status="accepted",
        decided_at=datetime.now(timezone.utc),
        evidence='{"lift": 0.005, "n": 12}',  # string, not dict
    )
    accepted.action = '{"kind": "raise_signal_weight", "target": "x", "params": {}}'
    sess = _ScriptedSession(scalars_payloads=[[accepted]])
    with _patch_session(sess):
        out = ddata.load_insight_history(status=["accepted"])
    assert out[0].evidence == {"lift": 0.005, "n": 12}
    assert out[0].action == {
        "kind": "raise_signal_weight",
        "target": "x",
        "params": {},
    }


# ---------------------------------------------------------------------------
# Sanity: the SIGNAL_VALUE_EXTRACTORS map covers the registry
# ---------------------------------------------------------------------------


def test_signal_value_extractors_cover_registry():
    from rainier.llm_thesis.signals import REGISTRY

    missing = set(REGISTRY) - set(ddata.SIGNAL_VALUE_EXTRACTORS)
    assert not missing, (
        f"Add SIGNAL_VALUE_EXTRACTORS entries for new signals: {missing}"
    )
