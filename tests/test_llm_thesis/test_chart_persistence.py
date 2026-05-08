"""Tests for ``llm_thesis.chart_persistence`` (PR5).

Covers idempotent persist + attach helpers. The DB session is mocked so
these tests don't require Postgres.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date as date_cls
from unittest.mock import MagicMock, patch

from rainier.llm_thesis.chart_persistence import (
    attach_chart_id_to_thesis,
    persist_chart_image,
)


@contextmanager
def _patch_session(session_obj):
    @contextmanager
    def _ctx():
        yield session_obj

    with patch(
        "rainier.llm_thesis.chart_persistence.get_session", _ctx
    ):
        yield


class _ScriptedSession:
    """Minimal SQLAlchemy 2.x Session shim — scripted execute() responses
    + session.get() reads from a keyed dict.
    """

    def __init__(self, *, execute_scripts=None, gets=None):
        self._execute_scripts = list(execute_scripts or [])
        self._gets = dict(gets or {})

    def execute(self, _stmt):  # noqa: ARG002
        result = MagicMock()
        if self._execute_scripts:
            first_value = self._execute_scripts.pop(0)
            result.first.return_value = first_value
        else:
            result.first.return_value = None
        return result

    def get(self, model, key):
        return self._gets.get((model, int(key)))

    # Context-manager protocol support not needed because the helper uses
    # ``get_session()`` directly.


class TestPersistChartImage:

    def test_returns_existing_id_when_sha_already_persisted(self):
        # First execute() = SELECT probe returns existing row id.
        sess = _ScriptedSession(execute_scripts=[(7,)])
        with _patch_session(sess):
            out = persist_chart_image(
                symbol="NVDA",
                scan_date=date_cls(2026, 5, 8),
                image_bytes=b"FAKE",
                sha256="a" * 64,
            )
        assert out == 7

    def test_inserts_new_row_when_no_existing(self):
        # First execute = SELECT (None), second execute = INSERT returning id.
        sess = _ScriptedSession(execute_scripts=[None, (123,)])
        with _patch_session(sess):
            out = persist_chart_image(
                symbol="NVDA",
                scan_date=date_cls(2026, 5, 8),
                image_bytes=b"FAKE",
                sha256="b" * 64,
            )
        assert out == 123

    def test_idempotent_on_same_sha256(self):
        """Two calls with identical bytes return the same id."""
        sess1 = _ScriptedSession(execute_scripts=[None, (50,)])
        sess2 = _ScriptedSession(execute_scripts=[(50,)])  # second call hits SELECT
        with _patch_session(sess1):
            first = persist_chart_image(
                symbol="NVDA",
                scan_date=date_cls(2026, 5, 8),
                image_bytes=b"X",
                sha256="c" * 64,
            )
        with _patch_session(sess2):
            second = persist_chart_image(
                symbol="NVDA",
                scan_date=date_cls(2026, 5, 8),
                image_bytes=b"X",
                sha256="c" * 64,
            )
        assert first == second == 50

    def test_returns_none_on_invalid_args(self):
        out = persist_chart_image(
            symbol="NVDA",
            scan_date=date_cls(2026, 5, 8),
            image_bytes=b"",
            sha256="",
        )
        assert out is None


class TestAttachChartIdToThesis:

    def test_appends_when_not_present(self):
        from rainier.core.models import LLMAnalysisRecord

        record = MagicMock(spec=LLMAnalysisRecord)
        record.chart_image_ids = [1, 2]

        sess = _ScriptedSession(gets={(LLMAnalysisRecord, 100): record})
        with _patch_session(sess):
            mutated = attach_chart_id_to_thesis(thesis_id=100, chart_id=3)
        assert mutated is True
        assert record.chart_image_ids == [1, 2, 3]

    def test_skips_when_already_present(self):
        from rainier.core.models import LLMAnalysisRecord

        record = MagicMock(spec=LLMAnalysisRecord)
        record.chart_image_ids = [1, 2, 3]

        sess = _ScriptedSession(gets={(LLMAnalysisRecord, 100): record})
        with _patch_session(sess):
            mutated = attach_chart_id_to_thesis(thesis_id=100, chart_id=3)
        assert mutated is False

    def test_starts_fresh_when_array_is_none(self):
        from rainier.core.models import LLMAnalysisRecord

        record = MagicMock(spec=LLMAnalysisRecord)
        record.chart_image_ids = None

        sess = _ScriptedSession(gets={(LLMAnalysisRecord, 100): record})
        with _patch_session(sess):
            mutated = attach_chart_id_to_thesis(thesis_id=100, chart_id=42)
        assert mutated is True
        assert record.chart_image_ids == [42]

    def test_returns_false_when_thesis_missing(self):
        sess = _ScriptedSession(gets={})
        with _patch_session(sess):
            mutated = attach_chart_id_to_thesis(thesis_id=999, chart_id=1)
        assert mutated is False
