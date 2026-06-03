"""Unit tests for scripts/cron-wrapper.sh log size capping.

fetch-mes runs every 5 minutes and appends to data/fetch.log, which grew to
52 MB unbounded. cron-wrapper.sh now truncates the log to its trailing
CRON_LOG_MAX_BYTES (default 10 MiB) before appending — bounding file size
WITHOUT changing log content (only the oldest bytes past the cap are dropped).

These tests drive the real bash script as a subprocess with a tiny cap and a
trivial command (no webhook → the Discord/uv path is never reached), so they
are fast, deterministic, and need no network.

Test plan pinned by docs/TASK-PLAN-rainier-slim-audit.md R3.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CRON_WRAPPER = ROOT / "scripts" / "cron-wrapper.sh"


def _run_wrapper(log_file: Path, cap_bytes: int, command: str = "true") -> int:
    """Invoke cron-wrapper.sh <job> <log> <webhook=""> <command> with a cap.

    Empty webhook arg → the failure-alert branch never shells out to uv, so
    the wrapper stays self-contained and offline.
    """
    proc = subprocess.run(
        ["bash", str(CRON_WRAPPER), "test-job", str(log_file), "", command],
        env={"CRON_LOG_MAX_BYTES": str(cap_bytes), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    return proc.returncode


def test_oversized_log_is_truncated_to_tail(tmp_path: Path):
    """A log already larger than the cap is shrunk to roughly the cap, and the
    MOST RECENT content (the tail) is what survives."""
    log_file = tmp_path / "fetch.log"
    cap = 4096
    # Seed an oversized log: a unique 'OLD' prefix then enough filler to exceed
    # the cap, then a unique 'RECENT' marker right at the end (the tail).
    old_marker = "OLDEST_LINE_SHOULD_BE_DROPPED\n"
    filler = ("x" * 79 + "\n") * 200  # ~16 KiB, well over the 4 KiB cap
    recent_marker = "RECENT_TAIL_MUST_SURVIVE\n"
    log_file.write_text(old_marker + filler + recent_marker)
    assert log_file.stat().st_size > cap

    rc = _run_wrapper(log_file, cap)
    assert rc == 0

    after = log_file.read_text()
    # Capped: original tail (<= cap) + the wrapper's own [START]/[OK] lines.
    # The dropped old prefix means total stays bounded near the cap.
    assert log_file.stat().st_size <= cap + 512
    # Tail content preserved; oldest line dropped.
    assert "RECENT_TAIL_MUST_SURVIVE" in after
    assert "OLDEST_LINE_SHOULD_BE_DROPPED" not in after
    # Wrapper still logged its structured markers (content semantics unchanged).
    assert "[START] test-job" in after
    assert "[OK] test-job" in after


def test_undersized_log_is_left_intact(tmp_path: Path):
    """A log under the cap is not truncated — existing content is fully kept
    and the wrapper appends as usual."""
    log_file = tmp_path / "fetch.log"
    existing = "PRE_EXISTING_CONTENT_KEPT_VERBATIM\n"
    log_file.write_text(existing)

    rc = _run_wrapper(log_file, cap_bytes=1_000_000)
    assert rc == 0

    after = log_file.read_text()
    assert after.startswith(existing)
    assert "[START] test-job" in after
    assert "[OK] test-job" in after


def test_missing_log_is_created_without_rotation(tmp_path: Path):
    """No pre-existing log → no truncation attempted, wrapper creates it."""
    log_file = tmp_path / "fetch.log"
    assert not log_file.exists()

    rc = _run_wrapper(log_file, cap_bytes=1024)
    assert rc == 0
    assert log_file.exists()
    assert "[START] test-job" in log_file.read_text()
    # No stray temp truncation files left behind.
    assert list(tmp_path.glob("*.trunc.*")) == []


def test_failing_command_still_caps_and_records_fail(tmp_path: Path):
    """Rotation runs regardless of command exit status, and a failure with no
    webhook records [FAIL] without shelling out to uv/Discord."""
    log_file = tmp_path / "fetch.log"
    cap = 2048
    log_file.write_text(("y" * 79 + "\n") * 100)  # ~8 KiB, over the cap
    assert log_file.stat().st_size > cap

    rc = _run_wrapper(log_file, cap, command="exit 3")
    assert rc == 3

    after = log_file.read_text()
    assert log_file.stat().st_size <= cap + 512
    assert "[FAIL exit=3] test-job" in after


@pytest.mark.skipif(not CRON_WRAPPER.exists(), reason="cron-wrapper.sh missing")
def test_wrapper_script_exists():
    assert CRON_WRAPPER.is_file()
