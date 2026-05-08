"""Unit tests for `dashboard.actions`.

`toggle_signal` writes YAML; `accept_insight` / `reject_insight` mutate
DB rows. We point the helpers at a temp YAML and a fake session so
nothing escapes the test isolation.

`test_signal` is exercised at the dispatch level — we mock the screener
+ registry so the test stays fast and doesn't fetch yfinance.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rainier.dashboard import actions as dactions

# ---------------------------------------------------------------------------
# Fake session — same shape as test_data.py but supports flush() + setattr
# ---------------------------------------------------------------------------


class _FakeRow:
    """Mutable stand-in for a SQLAlchemy row — supports ORM attribute writes."""

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MutableSession:
    def __init__(self, *, gets: dict[tuple[type, int], Any] | None = None):
        self._gets = dict(gets or {})
        self.flushed = 0

    def get(self, model, ident):
        return self._gets.get((model, int(ident)))

    def flush(self):
        self.flushed += 1


@contextmanager
def _session_cm(sess):
    yield sess


def _patch_session(sess):
    return patch.object(
        dactions, "get_session", return_value=_session_cm(sess)
    )


# ---------------------------------------------------------------------------
# toggle_signal
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, contents: str) -> None:
    path.write_text(contents)


def test_toggle_signal_disables_existing_entry(tmp_path):
    yaml_path = tmp_path / "settings.yaml"
    _write_yaml(
        yaml_path,
        """
# Top comment preserved
llm_thesis:
  prompt_version: v1
  signals:
    rank_trajectory:
      enabled: true
      params: {days: 10}
      weight: 1.0
""".lstrip(),
    )

    out = dactions.toggle_signal(
        "rank_trajectory", False, settings_path=yaml_path
    )
    assert out["enabled"] is False
    body = yaml_path.read_text()
    assert "enabled: false" in body
    # Ruamel preserves comments + key order.
    assert "Top comment preserved" in body
    assert body.index("prompt_version") < body.index("signals")


def test_toggle_signal_creates_missing_section(tmp_path):
    yaml_path = tmp_path / "settings.yaml"
    _write_yaml(yaml_path, "# nothing here\n")
    dactions.toggle_signal("rank_trajectory", True, settings_path=yaml_path)
    body = yaml_path.read_text()
    assert "llm_thesis" in body
    assert "rank_trajectory" in body
    assert "enabled: true" in body


def test_toggle_signal_rejects_unknown_name(tmp_path):
    yaml_path = tmp_path / "settings.yaml"
    _write_yaml(yaml_path, "{}\n")
    with pytest.raises(ValueError, match="Unknown signal"):
        dactions.toggle_signal("not_a_signal", True, settings_path=yaml_path)


# ---------------------------------------------------------------------------
# accept_insight
# ---------------------------------------------------------------------------


def test_accept_insight_dispatches_action_and_marks_row_accepted(tmp_path):
    """Happy path mirrors the CLI flow: validate, apply YAML, mark row."""
    from rainier.core.models import ResearchInsight

    settings = tmp_path / "settings.yaml"
    settings.write_text(
        """
llm_thesis:
  signals:
    rank_trajectory:
      enabled: true
      params: {days: 10}
      weight: 1.0
""".lstrip()
    )

    row = _FakeRow(
        id=1,
        status="pending",
        action={"kind": "disable_signal", "target": "rank_trajectory", "params": {}},
        decided_at=None,
        decided_by=None,
        applied_change=None,
    )
    sess = _MutableSession(gets={(ResearchInsight, 1): row})

    diff_payload = {"signal": "rank_trajectory", "field": "enabled", "new_value": False}
    with _patch_session(sess), patch.object(
        dactions, "apply_action", return_value=diff_payload
    ) as mock_apply:
        out = dactions.accept_insight(1, settings_path=settings)

    mock_apply.assert_called_once()
    args, _kwargs = mock_apply.call_args
    assert args[0]["kind"] == "disable_signal"
    assert isinstance(args[1], Path)

    assert out["status"] == "accepted"
    assert out["action_kind"] == "disable_signal"
    assert out["applied_change"] == diff_payload
    assert row.status == "accepted"
    assert isinstance(row.decided_at, datetime)
    assert row.decided_by == "dashboard"
    assert row.applied_change == diff_payload
    assert sess.flushed == 1


def test_accept_insight_lookup_error_for_missing_row(tmp_path):
    from rainier.core.models import ResearchInsight  # noqa: F401

    sess = _MutableSession(gets={})
    with _patch_session(sess), pytest.raises(LookupError):
        dactions.accept_insight(999, settings_path=tmp_path / "x.yaml")


def test_accept_insight_rejects_non_pending_row(tmp_path):
    from rainier.core.models import ResearchInsight

    row = _FakeRow(id=2, status="accepted", action={"kind": "noop", "target": "x"})
    sess = _MutableSession(gets={(ResearchInsight, 2): row})
    with _patch_session(sess), pytest.raises(ValueError, match="not pending"):
        dactions.accept_insight(2, settings_path=tmp_path / "x.yaml")


def test_accept_insight_propagates_apply_action_error(tmp_path):
    """A bad action.kind must surface BEFORE the row is marked accepted."""
    from rainier.core.models import ResearchInsight

    row = _FakeRow(
        id=3,
        status="pending",
        action={"kind": "not_real", "target": "x", "params": {}},
        decided_at=None,
        decided_by=None,
        applied_change=None,
    )
    sess = _MutableSession(gets={(ResearchInsight, 3): row})
    with _patch_session(sess), patch.object(
        dactions, "apply_action", side_effect=ValueError("Unknown action kind")
    ):
        with pytest.raises(ValueError, match="Unknown action kind"):
            dactions.accept_insight(3, settings_path=tmp_path / "x.yaml")
    # Status untouched.
    assert row.status == "pending"


# ---------------------------------------------------------------------------
# reject_insight
# ---------------------------------------------------------------------------


def test_reject_insight_marks_row_with_reason():
    from rainier.core.models import ResearchInsight

    row = _FakeRow(
        id=5,
        status="pending",
        decided_at=None,
        decided_by=None,
    )
    sess = _MutableSession(gets={(ResearchInsight, 5): row})
    with _patch_session(sess):
        out = dactions.reject_insight(5, "noisy signal — wait for more data")

    assert out["status"] == "rejected"
    assert row.status == "rejected"
    assert isinstance(row.decided_at, datetime)
    assert "noisy signal" in (row.decided_by or "")


def test_reject_insight_requires_reason():
    with pytest.raises(ValueError, match="non-empty reason"):
        dactions.reject_insight(1, "   ")


def test_reject_insight_lookup_error_on_missing():
    sess = _MutableSession(gets={})
    with _patch_session(sess), pytest.raises(LookupError):
        dactions.reject_insight(404, "stale")


def test_reject_insight_rejects_non_pending():
    from rainier.core.models import ResearchInsight

    row = _FakeRow(id=6, status="rejected", decided_at=None, decided_by=None)
    sess = _MutableSession(gets={(ResearchInsight, 6): row})
    with _patch_session(sess), pytest.raises(ValueError, match="not pending"):
        dactions.reject_insight(6, "double tap")


# ---------------------------------------------------------------------------
# test_signal — Tab 1 expander
# ---------------------------------------------------------------------------


def test_test_signal_runs_compute_and_returns_render(tmp_path):
    """Mocks the screener + signal class so we don't fetch yfinance."""
    from rainier.core.types import StockCandidate

    candidate = StockCandidate(
        symbol="NVDA",
        rank=1,
        rank_change=0,
        long_short="Long in",
        capital_flow_direction="N",
        sector="Tech",
        signal_strength=0.8,
    )

    fake_signal = MagicMock()
    fake_signal.compute.return_value = {"streak_days": 4, "direction": "+"}
    fake_signal.render_for_prompt.return_value = "4-day positive streak"
    fake_cls = MagicMock(return_value=fake_signal)

    fake_settings = MagicMock()
    fake_signal_cfg = MagicMock()
    fake_signal_cfg.params = {"days": 10}
    fake_settings.llm_thesis.signals.get.return_value = fake_signal_cfg

    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text("{}\n")

    with patch.object(
        dactions, "REGISTRY", {"capital_flow_streak": fake_cls}
    ), patch.object(
        dactions, "load_settings_fresh", return_value=fake_settings
    ), patch(
        "rainier.analysis.stock_screener.screen_stocks",
        return_value=([candidate], {"NVDA": MagicMock()}),
    ):
        out = dactions.test_signal(
            "capital_flow_streak", "NVDA", settings_path=yaml_path
        )

    assert out["signal"] == "capital_flow_streak"
    assert out["symbol"] == "NVDA"
    assert out["value"] == {"streak_days": 4, "direction": "+"}
    assert out["render_for_prompt"] == "4-day positive streak"
    fake_signal.compute.assert_called_once()


def test_test_signal_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown signal"):
        dactions.test_signal("not_a_signal", "NVDA")


def test_test_signal_builds_stub_when_symbol_not_in_screener(tmp_path):
    """Symbol not in screener output -> use a default-built StockCandidate."""
    fake_signal = MagicMock()
    fake_signal.compute.return_value = {"pe_forward": 18.5}
    fake_signal.render_for_prompt.return_value = "Forward P/E: 18.5"
    fake_cls = MagicMock(return_value=fake_signal)

    fake_settings = MagicMock()
    fake_settings.llm_thesis.signals.get.return_value = MagicMock(params={})

    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text("{}\n")

    with patch.object(
        dactions, "REGISTRY", {"fundamentals": fake_cls}
    ), patch.object(
        dactions, "load_settings_fresh", return_value=fake_settings
    ), patch(
        "rainier.analysis.stock_screener.screen_stocks",
        return_value=([], {}),  # screener returned no candidates
    ):
        out = dactions.test_signal(
            "fundamentals", "ZZZZ", settings_path=yaml_path
        )

    assert out["symbol"] == "ZZZZ"
    assert out["render_for_prompt"] == "Forward P/E: 18.5"
    # Verify a stub candidate was passed (compute() called once).
    fake_signal.compute.assert_called_once()
    ctx_arg = fake_signal.compute.call_args.args[0]
    assert ctx_arg.symbol == "ZZZZ"
    assert ctx_arg.candidate.symbol == "ZZZZ"
