"""Manage scheduled jobs defined in config/cron.yaml via system crontab."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml  # type: ignore[import-untyped]

JOB_TAG = "# rainier:"
DEFAULT_CONFIG = Path("config/cron.yaml")
WRAPPER_SCRIPT = Path("scripts/cron-wrapper.sh")


def load_config(path: Path = DEFAULT_CONFIG) -> list[dict]:
    """Load job definitions from cron.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("jobs", [])


def schedules(job: dict) -> list[str]:
    """``schedule`` is one cron expression or a list of them (one crontab line each)."""
    s = job["schedule"]
    return [s] if isinstance(s, str) else list(s)


def _load_discord_on_failure(path: Path = DEFAULT_CONFIG) -> bool:
    """Check if discord_on_failure is enabled in cron.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return bool(data.get("discord_on_failure", False))


def _load_discord_webhook(project_dir: Path) -> str:
    """Load DISCORD_WEBHOOK_URL from .env file."""
    env_file = project_dir / ".env"
    if not env_file.exists():
        return ""
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("DISCORD_WEBHOOK_URL=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def list_active() -> list[dict[str, str]]:
    """List rainier-managed jobs currently in system crontab."""
    crontab = _read_crontab()
    jobs = []
    for line in crontab.splitlines():
        if JOB_TAG in line:
            name = line.split(JOB_TAG)[-1].strip()
            before_tag = line.split(JOB_TAG)[0].strip()
            parts = before_tag.split(None, 5)
            cron_expr = " ".join(parts[:5]) if len(parts) >= 5 else before_tag
            command = parts[5] if len(parts) > 5 else ""
            jobs.append({"name": name, "cron": cron_expr, "command": command})
    return jobs


def sync(config_path: Path = DEFAULT_CONFIG, project_dir: Path | None = None) -> dict[str, str]:
    """Sync cron.yaml → system crontab. Returns {name: action} for each job."""
    if project_dir is None:
        project_dir = Path.cwd()

    jobs = load_config(config_path)
    active: dict[str, list[dict[str, str]]] = {}
    for j in list_active():
        active.setdefault(j["name"], []).append(j)
    actions: dict[str, str] = {}

    # Check if we should use the wrapper with Discord alerts
    discord_on_failure = _load_discord_on_failure(config_path)
    discord_webhook = _load_discord_webhook(project_dir) if discord_on_failure else ""
    wrapper_path = project_dir / WRAPPER_SCRIPT

    # Remove jobs no longer in config (or disabled)
    config_names = {j["name"] for j in jobs if j.get("enabled", True)}
    for name in list(active):
        if name not in config_names:
            _remove_job(name)
            actions[name] = "removed"

    # Add/update enabled jobs
    for job in jobs:
        name = job["name"]
        if not job.get("enabled", True):
            if name in active:
                _remove_job(name)
                actions[name] = "disabled"
            continue

        log = job.get("log", "/dev/null")
        log_path = project_dir / log if not log.startswith("/") else Path(log)

        # Resolve bare "uv" to full path so cron (minimal PATH) can find it
        cmd = job["command"]
        uv_path = shutil.which("uv")
        if uv_path:
            cmd = cmd.replace("uv run", f"{uv_path} run")

        # Build the cron command — wrapper does cd to project_dir automatically
        if wrapper_path.exists() and discord_webhook:
            command = (
                f"{wrapper_path} {name} {log_path} "
                f"{discord_webhook} "
                f'"{cmd}"'
            )
        else:
            command = f"cd {project_dir} && {cmd} >> {log_path} 2>&1"

        crons = schedules(job)
        cron_lines = "".join(f"{c} {command} {JOB_TAG} {name}\n" for c in crons)

        existing = active.get(name, [])
        if (sorted(e["cron"] for e in existing) == sorted(crons)
                and all(e["command"] == command for e in existing)):
            actions[name] = "unchanged"
        else:
            _remove_job(name)
            crontab = _read_crontab()
            if crontab.strip():
                crontab = crontab.rstrip("\n") + "\n" + cron_lines
            else:
                crontab = cron_lines
            _write_crontab(crontab)
            actions[name] = "added" if name not in active else "updated"

    return actions


def _remove_job(name: str) -> None:
    crontab = _read_crontab()
    lines = [
        line for line in crontab.splitlines()
        if not (JOB_TAG in line and line.strip().endswith(name))
    ]
    _write_crontab("\n".join(lines) + "\n" if lines else "")


def _read_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def _write_crontab(content: str) -> None:
    subprocess.run(["crontab", "-"], input=content, capture_output=True, text=True, check=True)
