"""Structural sanity checks — the F&G table shape and its cron slot.

Two lightweight, deterministic (no live DB) checks, separate from the
core-use-case ingest tests:
  1. fear_greed_index is a PLAIN table, not a hypertable.
  2. the daily cron slot is defined but NOT armed (`enabled: false`).
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_fear_greed_is_a_plain_table_not_a_hypertable():
    """CONTRACT: fear_greed_index stays a PLAIN table. A hypertable would force
    the partition column into every unique index (composite PK) and defeat the
    append-on-change `MIN(observed_at)` PIT scan — so it must be absent from
    HYPERTABLES and the migration must issue no `create_hypertable`."""
    from rainier.core.models import HYPERTABLES

    assert "fear_greed_index" not in HYPERTABLES
    sql = (ROOT / "migrations" / "0014_fear_greed_index.sql").read_text().lower()
    assert "create table if not exists fear_greed_index" in sql
    assert "create_hypertable" not in sql


def test_fear_greed_cron_slot_defined_but_disabled():
    """CONTRACT: the daily F&G slot exists but is `enabled: false` — present, not
    missing (a missing key defaults True in scheduler/jobs.py and would auto-arm
    on the next `jobs sync`). The operator arms it deliberately."""
    data = yaml.safe_load((ROOT / "config" / "cron.yaml").read_text())
    slot = {job["name"]: job for job in data["jobs"]}["fear-greed-daily"]
    assert "enabled" in slot  # explicit, not defaulted
    assert slot["enabled"] is False
    assert "fear-greed fetch" in slot["command"]
