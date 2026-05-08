"""Tests for the PR5 dashboard deep-link query-param handling.

Per the design doc Section G, Streamlit page rendering is excluded from
unit tests — we test the data-loading helpers used by the deep-link path.
That keeps the test stable across Streamlit version churn.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from unittest.mock import MagicMock, patch

from rainier.dashboard import data as ddata


@contextmanager
def _patch_session(session_obj):
    @contextmanager
    def _ctx():
        yield session_obj

    with patch("rainier.dashboard.data.get_session", _ctx):
        yield


class _ScriptedSession:
    def __init__(self, *, all_payloads=None, gets=None):
        self._all = list(all_payloads or [])
        self._gets = dict(gets or {})

    def execute(self, _stmt):  # noqa: ARG002
        result = MagicMock()
        result.all.return_value = self._all.pop(0) if self._all else []
        return result

    def get(self, model, key):
        return self._gets.get((model, int(key)))


# ---------------------------------------------------------------------------
# load_recent_theses with thesis_id deep-link scenario
# ---------------------------------------------------------------------------


class TestDeepLinkLoad:
    """The Streamlit page passes a wide ``since`` window when the
    ``thesis_id`` query param is present so the referenced row can't be
    filtered out by a recent date window. We unit-test the loader's
    ability to surface a specific thesis given that wide window.
    """

    def _row(self, thesis_id: int, scan_date: date, symbol: str = "NVDA"):
        row = MagicMock()
        row.screened_id = thesis_id * 10
        row.scan_date = scan_date
        row.session_name = "afternoon"
        row.symbol = symbol
        row.thesis_id = thesis_id
        row.llm_confidence = 7
        row.recommendation = "setup_long"
        row.signals_used = ["rank_trajectory"]
        row.chart_image_ids = [thesis_id]
        row.structured_output = {"verdict": "setup_long"}
        return row

    def test_load_recent_theses_returns_row_for_deep_link_thesis_id(self):
        # Wide window — covers the requested thesis from a year ago.
        sess = _ScriptedSession(
            all_payloads=[
                [self._row(thesis_id=42, scan_date=date(2025, 8, 1))],
                [],  # eval_rows query
            ]
        )
        with _patch_session(sess):
            rows = ddata.load_recent_theses(
                limit=50,
                since=date(2024, 1, 1),  # wide window simulates deep-link expansion
                until=date.today(),
            )
        assert len(rows) == 1
        assert rows[0].thesis_id == 42

    def test_load_thesis_chart_resolves_inline_bytes_for_deep_link(self):
        from rainier.core.models import ChartImage, LLMAnalysisRecord

        analysis = MagicMock(spec=LLMAnalysisRecord)
        analysis.chart_image_ids = [99]
        chart = MagicMock(spec=ChartImage)
        chart.image_bytes = b"PNG-BYTES-FROM-DB"
        chart.file_path = None

        sess = _ScriptedSession(
            gets={
                (LLMAnalysisRecord, 42): analysis,
                (ChartImage, 99): chart,
            }
        )
        with _patch_session(sess):
            out = ddata.load_thesis_chart(42)
        assert out == b"PNG-BYTES-FROM-DB"
