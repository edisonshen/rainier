"""The daily Fear & Greed cron slot is DEFINED but NOT armed.

A missing `enabled` key defaults to True in scheduler/jobs.py, so the flag must
be present-and-false or the next `rainier jobs sync` would install it into the
live crontab. Operator arms it deliberately.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CRON_PATH = ROOT / "config" / "cron.yaml"


@pytest.fixture(scope="module")
def jobs() -> dict[str, dict]:
    data = yaml.safe_load(CRON_PATH.read_text())
    return {job["name"]: job for job in data["jobs"]}


def test_fear_greed_slot_present(jobs):
    assert "fear-greed-daily" in jobs, "fear-greed daily cron slot missing"


def test_fear_greed_slot_disabled(jobs):
    job = jobs["fear-greed-daily"]
    # Present-and-false — NOT a missing key (which defaults True → auto-arms).
    assert "enabled" in job, "enabled key must be explicit, not defaulted"
    assert job["enabled"] is False


def test_fear_greed_slot_runs_fetch(jobs):
    assert "fear-greed fetch" in jobs["fear-greed-daily"]["command"]
