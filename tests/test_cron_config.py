"""Cron-config invariants for config/cron.yaml.

These assert the fengshen-publish retirement (task retire-fengshen-dashboar-aec0):
the legacy git-push-HTML publish steps are retired while every market.* PG-write
step is preserved. fengshen's PG-backed /trading is now the canonical source, so
no enabled cron job may invoke a publish-*dashboard*.sh / publish-market-breadth.sh
script (those git-push rendered HTML into the fengshen-site checkout and caused the
daily "target working tree dirty" publish failures).

Deterministic: parses the committed YAML with yaml.safe_load — no network, no cron
install.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CRON_PATH = ROOT / "config" / "cron.yaml"

# Any script that git-pushes rendered HTML into the fengshen-site checkout. None
# of these may appear in an *enabled* job's command after the retirement.
PUBLISH_SCRIPT_RE = re.compile(r"publish-(?:[a-z-]*dashboard|market-breadth)\.sh")


@pytest.fixture(scope="module")
def jobs() -> dict[str, dict]:
    """Return {job_name: job_dict} parsed from the committed cron.yaml."""
    data = yaml.safe_load(CRON_PATH.read_text())
    return {job["name"]: job for job in data["jobs"]}


def test_etf_dashboard_publish_disabled(jobs: dict[str, dict]) -> None:
    """The pure-publish ETF job is retired (its PG write lives in thematic-ranks-daily)."""
    assert jobs["etf-dashboard-publish"]["enabled"] is False


def test_market_breadth_keeps_compute_drops_publish(jobs: dict[str, dict]) -> None:
    """market-breadth-daily stays enabled and keeps both PG-write compute steps,
    but no longer invokes the publish-market-breadth.sh tail."""
    job = jobs["market-breadth-daily"]
    assert job["enabled"] is True
    cmd = job["command"]
    assert "backfill-ohlcv" in cmd
    assert "compute-indicators" in cmd
    assert "publish-market-breadth.sh" not in cmd
    assert not PUBLISH_SCRIPT_RE.search(cmd)


def test_no_enabled_job_publishes_to_fengshen(jobs: dict[str, dict]) -> None:
    """Net invariant: zero enabled jobs git-push HTML to fengshen-site."""
    offenders = [
        name
        for name, job in jobs.items()
        if job.get("enabled") and PUBLISH_SCRIPT_RE.search(job.get("command", ""))
    ]
    assert offenders == [], f"enabled jobs still invoke a fengshen publish script: {offenders}"


def test_trading_dashboard_stays_disabled(jobs: dict[str, dict]) -> None:
    """The combined-dashboard publisher was already disabled; verify it wasn't flipped on."""
    assert jobs["trading-dashboard"]["enabled"] is False


def test_pg_write_jobs_preserved(jobs: dict[str, dict]) -> None:
    """The market.* PG-write producers fengshen reads from stay enabled + intact."""
    thematic = jobs["thematic-ranks-daily"]
    assert thematic["enabled"] is True
    assert "thematic backfill" in thematic["command"]
    assert "run-daily" in thematic["command"]

    names = jobs["etf-names-refresh"]
    assert names["enabled"] is True
    assert "backfill-names" in names["command"]

    breadth = jobs["market-breadth-daily"]
    assert "backfill-ohlcv" in breadth["command"]
    assert "compute-indicators" in breadth["command"]
