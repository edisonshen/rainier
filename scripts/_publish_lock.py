#!/usr/bin/env python3
"""Portable shared publish lock for fengshen-site publishers.

All scripts that push to the shared fengshen-site checkout (the dashboard
publishers + the docs publisher) must hold a *single* mutual-exclusion lock
keyed by the RESOLVED target path. Two crons could otherwise both pass the
clean-tree check and then race the worktree / index / push.

Why not `flock(1)`? The cron host is macOS, which ships no `flock` binary
(docs/DESIGN-docs-publisher.md §3.2, Codex round 3 P1). We use `fcntl.flock`
on an exclusive lock file. The lock file lives OUTSIDE any git worktree (in
`$TMPDIR`/`/tmp`) so it can never show up as an untracked file and trip a
publisher's dirty-tree guard.

Usage (from a shell publisher):

    LOCKED_CMD="$SCRIPT_DIR/_publish_lock.py"
    python3 "$LOCKED_CMD" "$RESOLVED_TARGET" -- bash -c '...the publish body...'

`_publish_lock.py` acquires an exclusive lock on a path derived ONLY from the
resolved target, then exec/runs the given command. The lock is released when
this process exits (fcntl locks are tied to the fd / process), so a crash can
never leave a stale lock that blocks future publishes.

Sub-commands:
  <target> --path                 print the lock path for <target> and exit 0
                                   (lets shells + tests assert path equality)
  <target> [--timeout SECS] -- CMD...
                                   acquire the lock (blocking up to SECS, or
                                   forever if SECS<0) then run CMD; exit with
                                   CMD's exit code. On timeout: exit 75
                                   (EX_TEMPFAIL) WITHOUT running CMD.

ASCII flow:

    resolve target ─▶ sha1(realpath)[:12] ─▶ $TMPDIR/fengshen-publish-<sha>.lock
                                                      │
                                          fcntl.flock(LOCK_EX)
                                                      │
                                  ┌───────────────────┴──────────────────┐
                              acquired                                blocked
                                  │                                       │
                            run CMD; exit rc            wait up to timeout; if it
                                                        expires → exit 75, CMD never runs
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
import time

# Exit code used when the lock could not be acquired within the timeout.
# Mirrors BSD sysexits EX_TEMPFAIL so callers can distinguish "busy" from a
# real command failure.
EX_TEMPFAIL = 75


def lock_path_for(target: str) -> str:
    """Deterministic lock-file path for a target directory.

    Keyed by the RESOLVED (realpath) target so two scripts pointed at the same
    checkout — even via different symlinks / relative paths — derive the same
    lock. Lives in TMPDIR so it never dirties a git worktree.
    """
    resolved = os.path.realpath(target)
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    tmp = os.environ.get("TMPDIR", "/tmp")
    return os.path.join(tmp, f"fengshen-publish-{digest}.lock")


def _usage() -> int:
    sys.stderr.write(
        "usage: _publish_lock.py <target> --path\n"
        "       _publish_lock.py <target> [--timeout SECS] -- CMD...\n"
    )
    return 2


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return _usage()

    target = argv[1]
    rest = argv[2:]

    if rest == ["--path"]:
        sys.stdout.write(lock_path_for(target) + "\n")
        return 0

    # Parse optional --timeout, then a literal `--`, then the command.
    timeout = -1.0  # block forever by default
    i = 0
    if i < len(rest) and rest[i] == "--timeout":
        if i + 1 >= len(rest):
            return _usage()
        try:
            timeout = float(rest[i + 1])
        except ValueError:
            return _usage()
        i += 2
    if i >= len(rest) or rest[i] != "--":
        return _usage()
    cmd = rest[i + 1 :]
    if not cmd:
        return _usage()

    path = lock_path_for(target)
    # Open (create) the lock file. Mode 0o600 — operator-only.
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        acquired = _acquire(fd, timeout)
        if not acquired:
            sys.stderr.write(
                f"_publish_lock: timed out acquiring {path} after {timeout}s\n"
            )
            return EX_TEMPFAIL
        # Lock held for the lifetime of CMD. fcntl releases it when this
        # process exits (incl. crash), so no stale-lock cleanup is needed.
        completed = subprocess.run(cmd)
        return completed.returncode
    finally:
        os.close(fd)


def _acquire(fd: int, timeout: float) -> bool:
    """Acquire an exclusive flock on fd.

    timeout < 0  → block until acquired (returns True).
    timeout >= 0 → poll non-blocking until acquired or the deadline; returns
                   False on timeout. Polling (vs. a blocking flock + alarm)
                   keeps this portable and signal-safe across macOS/Linux.
    """
    if timeout < 0:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return True

    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
