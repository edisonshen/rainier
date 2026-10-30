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


def test_market_breadth_refreshes_benchmark_ohlcv(jobs: dict[str, dict]) -> None:
    """REGRESSION (benchmark-ohlcv-spy-stale): the daily breadth cron must invoke
    `backfill-spy`, the ONLY command that writes market.benchmark_ohlcv (via
    _dual_write_benchmark_pg). Before this fix the daily job ran backfill-ohlcv +
    compute-indicators but never backfill-spy, so benchmark_ohlcv (SPY price pane
    fengshen reads) silently went stale — breadth/thematic kept current while the
    SPY OHLCV table lagged from 2026-05-29 onward.

    The job is enabled and the command runs backfill-spy --incremental so today's
    SPY bar lands in market.benchmark_ohlcv every trading day."""
    job = jobs["market-breadth-daily"]
    assert job["enabled"] is True
    cmd = job["command"]
    assert "backfill-spy" in cmd, (
        "market-breadth-daily must run `backfill-spy` to keep market.benchmark_ohlcv "
        "fresh — without it the SPY price pane fengshen reads goes stale"
    )
    assert "--incremental" in cmd


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


@pytest.mark.parametrize("symbol", ["MES", "MNQ"])
def test_weekly_backfill_scheduled(jobs: dict[str, dict], symbol: str) -> None:
    """yfinance keeps only 60 days of 5m, so each charted symbol must be backfilled at
    least weekly or its 5m history develops permanent gaps."""
    job = jobs[f"backfill-{symbol.lower()}-weekly"]
    assert job["enabled"] is True
    assert f"rainier fetch --symbol {symbol}" in job["command"]
    dow = job["schedule"].split()[4]
    assert dow != "*"


def _fires(expr: str) -> set[tuple[int, int, int]]:
    """(dow, hour, minute) slots a simple 5-field cron expression fires in (dom/month = *)."""
    def field(f: str, lo: int, hi: int) -> set[int]:
        out: set[int] = set()
        for part in f.split(","):
            rng, _, step = part.partition("/")
            a, b = (lo, hi) if rng == "*" else map(int, (rng.split("-") * 2)[:2])
            out |= set(range(a, b + 1, int(step or 1)))
        return out

    m, h, dom, mon, dow = expr.split()
    assert dom == mon == "*"
    return {(d, hh, mm) for d in field(dow, 0, 6) for hh in field(h, 0, 23)
            for mm in field(m, 0, 59)}


@pytest.mark.parametrize("symbol", ["MES", "MNQ"])
def test_chart_fetch_runs_only_while_cme_equity_futures_trade(
    jobs: dict[str, dict], symbol: str,
) -> None:
    """MES/MNQ trade Sun 15:00 -> Fri 14:00 PT; the chart needs no pushes over the weekend
    (user request), but must catch Friday's last bars, which arrive ~10-15 min late."""
    from rainier.scheduler.jobs import schedules

    job = jobs[f"fetch-{symbol.lower()}"]
    assert job["enabled"] is True
    assert (f"rainier fetch --symbol {symbol} && uv run rainier sr-levels --symbol {symbol}"
            in job["command"]) and "--publish-kv" in job["command"]
    slots = set().union(*(_fires(s) for s in schedules(job)))
    sun, fri, sat = 0, 5, 6
    assert not any(d == sat for d, _, _ in slots)
    assert not any(d == fri and (h, m) > (14, 25) for d, h, m in slots)
    assert not any(d == sun and h < 15 for d, h, _ in slots)
    every_5 = {(h, m) for h in range(24) for m in range(0, 60, 5)}
    for d in (1, 2, 3, 4):  # Mon-Thu all day; the halt hour is skipped by the KV no-op
        assert {(h, m) for dd, h, m in slots if dd == d} == every_5
    assert {(fri, 13, 55), (fri, 14, 25), (sun, 15, 0), (sun, 23, 55)} <= slots


def test_sync_installs_one_crontab_line_per_schedule(tmp_path, monkeypatch) -> None:
    from rainier.scheduler import jobs as jobs_mod

    tab = {"text": "0 1 * * * echo keep # unrelated\n"}
    monkeypatch.setattr(jobs_mod, "_read_crontab", lambda: tab["text"])
    monkeypatch.setattr(jobs_mod, "_write_crontab", lambda t: tab.update(text=t))
    cfg = tmp_path / "cron.yaml"

    def write(schedule) -> None:
        cfg.write_text(yaml.safe_dump({"jobs": [
            {"name": "fetch-x", "schedule": schedule, "command": "echo hi"},
            {"name": "single", "schedule": "0 9 * * 6", "command": "echo one"},
        ]}))

    write(["*/5 * * * 1-4", "*/5 15-23 * * 0"])
    assert jobs_mod.sync(cfg, tmp_path) == {"fetch-x": "added", "single": "added"}
    crons = sorted(j["cron"] for j in jobs_mod.list_active() if j["name"] == "fetch-x")
    assert crons == ["*/5 * * * 1-4", "*/5 15-23 * * 0"] and "echo keep" in tab["text"]
    assert jobs_mod.sync(cfg, tmp_path) == {"fetch-x": "unchanged", "single": "unchanged"}

    write(["*/5 * * * 1-4"])
    assert jobs_mod.sync(cfg, tmp_path)["fetch-x"] == "updated"
    assert [j["cron"] for j in jobs_mod.list_active() if j["name"] == "fetch-x"] == [
        "*/5 * * * 1-4"]


def test_jobs_list_flags_a_missing_schedule_line(tmp_path, monkeypatch) -> None:
    from click.testing import CliRunner

    from rainier.cli import cli
    from rainier.scheduler import jobs as jobs_mod

    tab = {"text": ""}
    monkeypatch.setattr(jobs_mod, "_read_crontab", lambda: tab["text"])
    monkeypatch.setattr(jobs_mod, "_write_crontab", lambda t: tab.update(text=t))
    jobs_mod.sync(project_dir=tmp_path)

    def status() -> str:
        out = CliRunner().invoke(cli, ["jobs", "list"]).output
        return next(ln for ln in out.splitlines() if ln.strip().startswith("fetch-mes"))

    assert status().endswith("ACTIVE")
    tab["text"] = "".join(ln + "\n" for ln in tab["text"].splitlines()
                          if not (ln.startswith("*/5 15-23 * * 0") and "fetch-mes" in ln))
    assert status().endswith("OUT OF SYNC")
