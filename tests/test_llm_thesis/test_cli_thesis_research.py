"""Tests for `rainier thesis research` CLI subcommands — list, accept,
reject, run, signals, verdicts.

Covers the human-in-the-loop accept/reject flow including the YAML-mutation
side effect (handled via a tmp_path settings file).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rainier.cli import cli
from rainier.core.models import ResearchInsight

# ---------------------------------------------------------------------------
# Fake session helpers
# ---------------------------------------------------------------------------


def _ins(
    *,
    id: int,
    kind: str = "signal_underperform",
    subject: str = "rank_trajectory",
    severity: str = "warn",
    rationale: str = "Default rationale.",
    recurrence: int = 1,
    action: dict | None = None,
    status: str = "pending",
) -> ResearchInsight:
    return ResearchInsight(
        id=id,
        kind=kind,
        subject=subject,
        severity=severity,
        evidence={"sample": 1},
        action=action or {"kind": "disable_signal", "target": subject, "params": {}},
        rationale=rationale,
        recurrence_count=recurrence,
        status=status,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


SAMPLE_YAML = """\
# top comment
app:
  name: rainier
  data_dir: ./data

llm_thesis:
  enabled: true
  prompt_version: "v1"
  signals:
    rank_trajectory:
      enabled: true
      params: {days: 10}
      weight: 1.0
    fundamentals:
      enabled: true
      params: {}
      weight: 1.0
"""


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    p = tmp_path / "settings.yaml"
    p.write_text(SAMPLE_YAML)
    return p


# ---------------------------------------------------------------------------
# `rainier thesis research insights list`
# ---------------------------------------------------------------------------


def test_insights_list_filters_by_status(settings_path: Path):
    rows = [
        _ins(id=1, severity="warn", status="pending"),
        _ins(id=2, severity="critical", status="accepted"),
    ]
    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False
    sess.execute.return_value.scalars.return_value.all.return_value = [rows[0]]

    runner = CliRunner()
    with patch("rainier.core.database.get_session", return_value=cm):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "insights", "list",
                "--status", "pending",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "ID" in result.output
    assert "rank_trajectory" in result.output
    assert "warn" in result.output


def test_insights_list_no_rows(settings_path: Path):
    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False
    sess.execute.return_value.scalars.return_value.all.return_value = []
    runner = CliRunner()
    with patch("rainier.core.database.get_session", return_value=cm):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "insights", "list",
                "--status", "pending",
            ],
        )
    assert result.exit_code == 0
    assert "No pending insights" in result.output


# ---------------------------------------------------------------------------
# `rainier thesis research insights accept`
# ---------------------------------------------------------------------------


def test_insights_accept_applies_yaml_change_and_marks_accepted(
    settings_path: Path,
):
    """End-to-end: accept a disable_signal insight → YAML flipped + DB row
    moves to status=accepted with applied_change populated.
    """
    row = _ins(
        id=42, severity="warn", subject="rank_trajectory",
        action={"kind": "disable_signal", "target": "rank_trajectory", "params": {}},
        status="pending",
    )
    captured: dict = {}

    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False
    sess.get.return_value = row

    def _flush():
        captured["status"] = row.status
        captured["applied_change"] = row.applied_change
        captured["decided_at"] = row.decided_at

    sess.flush.side_effect = _flush

    runner = CliRunner()
    with patch("rainier.core.database.get_session", return_value=cm):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "insights", "accept", "42",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Accepted insight #42" in result.output
    # YAML mutated.
    text = settings_path.read_text()
    rt_section = text.split("rank_trajectory:")[1].split("fundamentals:")[0]
    assert "enabled: false" in rt_section
    # DB updates flushed.
    assert captured["status"] == "accepted"
    assert captured["applied_change"]["new_value"] is False
    assert captured["decided_at"] is not None


def test_insights_accept_rejects_non_pending(settings_path: Path):
    """An already-accepted/rejected/stale row must error out clearly."""
    row = _ins(id=42, severity="warn", status="accepted")
    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False
    sess.get.return_value = row

    runner = CliRunner()
    with patch("rainier.core.database.get_session", return_value=cm):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "insights", "accept", "42",
            ],
        )
    assert result.exit_code != 0
    assert "not pending" in result.output


def test_insights_accept_missing_row(settings_path: Path):
    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False
    sess.get.return_value = None

    runner = CliRunner()
    with patch("rainier.core.database.get_session", return_value=cm):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "insights", "accept", "999",
            ],
        )
    assert result.exit_code != 0
    assert "No ResearchInsight with id=999" in result.output


def test_insights_accept_unknown_action_kind(settings_path: Path):
    """If an old insight has an action.kind no longer in ACTION_EXECUTORS,
    surface a clear error rather than partial side effect.
    """
    row = _ins(
        id=42,
        severity="warn",
        action={"kind": "delete_universe", "target": "x", "params": {}},
        status="pending",
    )
    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False
    sess.get.return_value = row
    original = settings_path.read_text()
    runner = CliRunner()
    with patch("rainier.core.database.get_session", return_value=cm):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "insights", "accept", "42",
            ],
        )
    assert result.exit_code != 0
    assert "Could not apply action" in result.output
    # YAML untouched.
    assert settings_path.read_text() == original


# ---------------------------------------------------------------------------
# `rainier thesis research insights reject`
# ---------------------------------------------------------------------------


def test_insights_reject_marks_rejected_with_reason(settings_path: Path):
    row = _ins(id=42, severity="warn", status="pending")
    captured: dict = {}

    sess = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = sess
    cm.__exit__.return_value = False
    sess.get.return_value = row

    def _flush():
        captured["status"] = row.status
        captured["decided_by"] = row.decided_by

    sess.flush.side_effect = _flush

    original = settings_path.read_text()
    runner = CliRunner()
    with patch("rainier.core.database.get_session", return_value=cm):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "insights", "reject", "42",
                "--reason", "noise",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Rejected insight #42" in result.output
    assert captured["status"] == "rejected"
    assert captured["decided_by"] == "noise"
    # YAML untouched on rejection.
    assert settings_path.read_text() == original


def test_insights_reject_requires_reason(settings_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config", str(settings_path),
            "thesis", "research", "insights", "reject", "42",
        ],
    )
    assert result.exit_code != 0
    assert "reason" in result.output.lower()


# ---------------------------------------------------------------------------
# `rainier thesis research run`
# ---------------------------------------------------------------------------


def test_research_run_invokes_scheduler_entry(settings_path: Path):
    runner = CliRunner()
    captured: dict = {}

    async def _fake_run(eval_date_iso=None):
        captured["eval_date_iso"] = eval_date_iso

    with patch(
        "rainier.scheduler.service.run_research_job",
        side_effect=_fake_run,
    ):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "run",
                "--eval-date", "2026-05-08",
            ],
        )
    assert result.exit_code == 0, result.output
    assert captured["eval_date_iso"] == "2026-05-08"


# ---------------------------------------------------------------------------
# `rainier thesis research signals` and `verdicts`
# ---------------------------------------------------------------------------


def test_research_signals_prints_contributions(settings_path: Path):
    from rainier.llm_thesis.eval import SignalContribution

    contribs = [
        SignalContribution(
            "rank_trajectory", n_used=20, n_absent=18,
            mean_used=0.012, mean_absent=0.004, lift=0.008, p_value=0.03,
        ),
    ]
    runner = CliRunner()
    with patch(
        "rainier.llm_thesis.eval.compute_signal_contribution",
        return_value=contribs,
    ):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "signals", "--days", "30",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "rank_trajectory" in result.output
    assert "0.030" in result.output  # p-value


def test_research_verdicts_prints_hit_rates(settings_path: Path):
    from rainier.llm_thesis.eval import HitRate

    rates = {
        "setup_long": [
            HitRate("setup_long", "1d", 4, 0.75, 0.01),
            HitRate("setup_long", "5d", 4, 0.75, 0.02),
            HitRate("setup_long", "10d", 4, 0.75, 0.03),
        ],
        "watch": [
            HitRate("watch", "1d", 0, 0.0, 0.0),
            HitRate("watch", "5d", 0, 0.0, 0.0),
            HitRate("watch", "10d", 0, 0.0, 0.0),
        ],
        "no_setup": [
            HitRate("no_setup", "1d", 0, 0.0, 0.0),
            HitRate("no_setup", "5d", 0, 0.0, 0.0),
            HitRate("no_setup", "10d", 0, 0.0, 0.0),
        ],
    }
    runner = CliRunner()
    with patch(
        "rainier.llm_thesis.eval.compute_verdict_hit_rate", return_value=rates,
    ):
        result = runner.invoke(
            cli,
            [
                "--config", str(settings_path),
                "thesis", "research", "verdicts", "--days", "30",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "setup_long" in result.output
    assert "75.00%" in result.output or "75%" in result.output
