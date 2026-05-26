"""Unit tests for scripts/publish-dashboard.sh.

Generic dashboard publisher — copies `out/dashboards/<name>.html` into a
target git repo's `public/trading/<name>/index.html` and pushes when
the content actually changed.

Tests use temporary git repos as both the source dashboard dir and
the publish target (no writes to ~/projects/fengshen-site). The
script is selected via the `DASHBOARD_PUBLISH_TARGET_DIR` env var
so CI never touches the operator's real fengshen-site checkout.

Test plan pinned by docs/TASK-PLAN-etf-dashboard-publish-cr-31aa.md §Tests.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publish-dashboard.sh"
CRON_WRAPPER = ROOT / "scripts" / "cron-wrapper.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path, check: bool = True, env: dict | None = None) -> str:
    """Run git with deterministic identity so commits don't pull from the
    operator's global config."""
    full_env = os.environ.copy()
    full_env.update(
        {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    if env:
        full_env.update(env)
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=full_env,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout


@pytest.fixture
def target_repo(tmp_path: Path) -> Path:
    """A bare-ish git repo that simulates the fengshen-site checkout.

    We create both a 'remote' bare repo and a 'local' working clone, so the
    publish script can `git push` against a real remote (locally) without
    touching the network or the operator's real repo.
    """
    remote = tmp_path / "fengshen-site-remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )

    local = tmp_path / "fengshen-site"
    subprocess.run(
        ["git", "clone", str(remote), str(local)], check=True, capture_output=True
    )

    # Need an initial commit on `main` so the push has something to fast-forward
    # from. Astro project layout: `public/` is the static asset root.
    (local / "public").mkdir()
    (local / "public" / ".gitkeep").write_text("")
    _git("checkout", "-B", "main", cwd=local)
    _git("add", "public/.gitkeep", cwd=local)
    _git("commit", "-m", "init", cwd=local)
    _git("push", "-u", "origin", "main", cwd=local)
    return local


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """Simulated rainier `out/dashboards/` directory."""
    d = tmp_path / "out" / "dashboards"
    d.mkdir(parents=True)
    return d


def _run_publisher(
    name: str,
    *,
    source_dir: Path,
    target_repo: Path,
    extra_env: dict | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "DASHBOARD_SOURCE_DIR": str(source_dir),
            "DASHBOARD_PUBLISH_TARGET_DIR": str(target_repo),
            # Deterministic git identity inside the script as well.
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SCRIPT), name],
        capture_output=True,
        text=True,
        env=env,
    )


# ---------------------------------------------------------------------------
# Test 1 — no-op when rendered HTML is identical to what's already published.
# ---------------------------------------------------------------------------


def test_publish_no_op_on_clean(source_dir: Path, target_repo: Path):
    """Re-running with byte-identical HTML must exit 0 without a new commit."""
    name = "etf-ranks"
    html = "<html><body>identical</body></html>"
    (source_dir / f"{name}.html").write_text(html)

    # Pre-seed the destination with the same content + commit it so the next
    # publish is a true no-op.
    dest_dir = target_repo / "public" / "trading" / name
    dest_dir.mkdir(parents=True)
    (dest_dir / "index.html").write_text(html)
    _git("add", f"public/trading/{name}/index.html", cwd=target_repo)
    _git("commit", "-m", f"{name}: pre-seed", cwd=target_repo)
    _git("push", "origin", "main", cwd=target_repo)

    head_before = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    result = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    head_after = _git("rev-parse", "HEAD", cwd=target_repo).strip()

    assert result.returncode == 0, (
        f"expected clean exit on no-op, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert head_before == head_after, "no-op must not create a new commit"
    # Friendly log so the operator can grep cron logs for "no-op".
    assert "no-op" in result.stdout.lower() or "no change" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Test 2 — fresh target gets dir created + file copied + committed.
# ---------------------------------------------------------------------------


def test_publish_creates_dir(source_dir: Path, target_repo: Path):
    """First-ever publish creates `public/trading/<name>/` and lands index.html."""
    name = "etf-ranks"
    html = "<html><body>first publish</body></html>"
    (source_dir / f"{name}.html").write_text(html)

    dest = target_repo / "public" / "trading" / name / "index.html"
    assert not dest.exists(), "test precondition: destination should not exist"

    result = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    assert dest.exists(), "destination index.html must be created"
    assert dest.read_text() == html

    # Commit must have happened.
    log = _git(
        "log", "--oneline", "-n", "1", "--", f"public/trading/{name}/index.html",
        cwd=target_repo,
    )
    assert name in log, f"commit log missing dashboard name: {log!r}"


# ---------------------------------------------------------------------------
# Test 3 — when the source HTML differs, script commits with expected
# message format and runs `git push`.
# ---------------------------------------------------------------------------


def test_publish_commits_when_changed(source_dir: Path, target_repo: Path):
    """A modified source HTML produces a commit `<name>: YYYY-MM-DD daily render`
    and the commit lands on origin/main."""
    name = "etf-ranks"
    # Seed an initial publish.
    (source_dir / f"{name}.html").write_text("<html>v1</html>")
    r1 = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert r1.returncode == 0, r1.stderr

    # Now modify the source and re-publish.
    (source_dir / f"{name}.html").write_text("<html>v2 updated</html>")
    r2 = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert r2.returncode == 0, (
        f"rc={r2.returncode}\nstdout: {r2.stdout}\nstderr: {r2.stderr}"
    )

    # Latest commit message: `<name>: YYYY-MM-DD daily render`
    head_msg = _git("log", "-1", "--pretty=%s", cwd=target_repo).strip()
    assert re.match(
        rf"^{re.escape(name)}: \d{{4}}-\d{{2}}-\d{{2}} daily render$", head_msg
    ), f"unexpected commit subject: {head_msg!r}"

    # And the push made it to the remote.
    local_head = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    remote_head = _git("rev-parse", "origin/main", cwd=target_repo).strip()
    assert local_head == remote_head, "publisher must `git push` after commit"


# ---------------------------------------------------------------------------
# Test 4 — bail (non-zero, no commit) when target has uncommitted changes.
# ---------------------------------------------------------------------------


def test_publish_bails_on_dirty_target(source_dir: Path, target_repo: Path):
    """Dirty target working tree → script exits non-zero and does NOT commit
    or stash anything."""
    name = "etf-ranks"
    (source_dir / f"{name}.html").write_text("<html>clean source</html>")

    # Dirty the target: untracked file + modified tracked file.
    (target_repo / "public" / ".gitkeep").write_text("modified\n")
    (target_repo / "stray.txt").write_text("untracked\n")

    head_before = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    result = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    head_after = _git("rev-parse", "HEAD", cwd=target_repo).strip()

    assert result.returncode != 0, (
        "publisher must exit non-zero when target is dirty\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert head_before == head_after, "publisher must not commit when bailing"

    # Stray + modified files must still be there — no auto-stash.
    assert (target_repo / "stray.txt").exists(), "must not stash untracked files"
    assert (target_repo / "public" / ".gitkeep").read_text() == "modified\n"

    # Clear, actionable error message on stdout or stderr.
    combined = (result.stdout + result.stderr).lower()
    assert "dirty" in combined or "uncommitted" in combined, (
        f"expected dirty/uncommitted in output, got:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 5 — missing source HTML produces a clean non-zero exit (no copy, no
# commit). `set -euo pipefail` + the early `[ ! -f "$SRC" ]` guard must turn
# this into a meaningful error, NOT a silent success.
# ---------------------------------------------------------------------------


def test_publish_errors_when_source_missing(source_dir: Path, target_repo: Path):
    """No rendered HTML at the source path → exit non-zero, no commit."""
    name = "etf-ranks"
    # Intentionally do NOT write `etf-ranks.html` into source_dir.
    assert not (source_dir / f"{name}.html").exists()

    head_before = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    result = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    head_after = _git("rev-parse", "HEAD", cwd=target_repo).strip()

    assert result.returncode != 0, (
        f"missing source must produce non-zero exit, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert head_before == head_after, "must not commit when source is missing"

    combined = (result.stdout + result.stderr).lower()
    assert "missing" in combined or "no such" in combined, (
        f"expected actionable error mentioning missing source, got:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    # And no file should have been published.
    dst = target_repo / "public" / "trading" / name / "index.html"
    assert not dst.exists(), "must not create destination when source is missing"


# ---------------------------------------------------------------------------
# Test 6 — target is not a git checkout → clean non-zero exit, no file copy.
# Protects against misconfiguration (DASHBOARD_PUBLISH_TARGET_DIR pointing at
# a plain directory instead of the fengshen-site clone).
# ---------------------------------------------------------------------------


def test_publish_errors_when_target_not_git(tmp_path: Path, source_dir: Path):
    """Non-git target → exit non-zero, no file written."""
    name = "etf-ranks"
    (source_dir / f"{name}.html").write_text("<html>v1</html>")

    # Plain directory, no .git/ inside.
    not_a_git_repo = tmp_path / "fengshen-site-but-not-really"
    not_a_git_repo.mkdir()
    assert not (not_a_git_repo / ".git").exists()

    result = _run_publisher(
        name, source_dir=source_dir, target_repo=not_a_git_repo
    )
    assert result.returncode != 0, (
        f"non-git target must produce non-zero exit, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    combined = (result.stdout + result.stderr).lower()
    assert "not a git" in combined or "git checkout" in combined, (
        f"expected error mentioning git checkout, got:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    # And no destination directory created.
    dst = not_a_git_repo / "public" / "trading" / name
    assert not dst.exists(), "must not create destination on non-git target"


# ---------------------------------------------------------------------------
# Test 7 — pending unpushed commits are recovered on the no-op path.
# Regression for the "yesterday push failed + today identical render"
# corner case where the local repo silently drifts ahead of origin.
# ---------------------------------------------------------------------------


def test_publish_pushes_pending_commits_on_noop(
    source_dir: Path, target_repo: Path
):
    """After a prior commit is left local-only (simulating a failed push),
    a subsequent no-op render must detect the ahead-of-upstream state and
    push the pending commit instead of exiting silently."""
    name = "etf-ranks"
    html = "<html>v1</html>"
    (source_dir / f"{name}.html").write_text(html)

    # First run: lands a commit and pushes.
    r1 = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert r1.returncode == 0, r1.stderr

    # Simulate "yesterday's push failed" by rewinding the remote one commit.
    # Resolve the parent SHA in the working clone (where the reflog lives),
    # then point the bare remote's main ref at it directly. We avoid
    # `HEAD~1` against the bare repo because its HEAD is symbolic-only.
    remote_url = _git("config", "remote.origin.url", cwd=target_repo).strip()
    parent_sha = _git("rev-parse", "HEAD~1", cwd=target_repo).strip()
    _git(
        "update-ref",
        "refs/heads/main",
        parent_sha,
        cwd=Path(remote_url),
    )
    # Confirm we are now ahead of origin/main.
    _git("fetch", "origin", cwd=target_repo)
    ahead = _git(
        "rev-list", "--count", "origin/main..HEAD", cwd=target_repo
    ).strip()
    assert ahead == "1", f"setup error: expected 1 ahead, got {ahead}"

    # Re-run with identical source HTML → the no-op path triggers, but the
    # pending commit must still be pushed.
    r2 = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert r2.returncode == 0, (
        f"rc={r2.returncode}\nstdout: {r2.stdout}\nstderr: {r2.stderr}"
    )

    # Local and origin should be back in sync.
    _git("fetch", "origin", cwd=target_repo)
    local_head = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    origin_head = _git("rev-parse", "origin/main", cwd=target_repo).strip()
    assert local_head == origin_head, (
        f"pending commit was not recovered: local={local_head[:8]} "
        f"origin={origin_head[:8]}"
    )


# ---------------------------------------------------------------------------
# Test 8 — cron-wrapper integration: invokes render then publish in order.
# ---------------------------------------------------------------------------


def test_cron_wrapper_invokes_render_then_publish(tmp_path: Path):
    """`cron-wrapper.sh` runs the supplied command verbatim and logs phases.

    We construct a 'fake render && fake publish' command that drops two files
    in order, then assert the file order matches (render first, publish
    second) and the wrapper's [OK] line appears in the log.

    This is the integration shape used by the new cron entry; the wrapper is
    already generic so we're verifying it cleanly hosts the new chained
    command without bespoke per-job code.
    """
    project = tmp_path / "rainier"
    (project / "scripts").mkdir(parents=True)
    # Drop the real wrapper into a clone of the project layout cron-wrapper.sh
    # expects (it cd's into `dirname(scripts)/`).
    wrapper_dst = project / "scripts" / "cron-wrapper.sh"
    wrapper_dst.write_text(CRON_WRAPPER.read_text())
    wrapper_dst.chmod(0o755)

    marker_render = tmp_path / "render.marker"
    marker_publish = tmp_path / "publish.marker"
    log = tmp_path / "cron.log"

    # The command argument is `eval`d by cron-wrapper, exactly as it would be
    # from cron.yaml. Order matters: render must touch its marker FIRST, then
    # `&&` chains to the publisher.
    command = (
        f"date >> {marker_render} && "
        # tiny sleep so mtimes differ even on coarse filesystems
        f"sleep 0.01 && "
        f"date >> {marker_publish}"
    )

    result = subprocess.run(
        [
            "bash",
            str(wrapper_dst),
            "etf-dashboard-publish",  # job name
            str(log),                  # log file
            "",                        # webhook (empty disables Discord)
            command,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"wrapper failed: rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
        f"log: {log.read_text() if log.exists() else '(no log)'}"
    )
    assert marker_render.exists(), "render step never ran"
    assert marker_publish.exists(), "publish step never ran"

    # Render's marker must be older than publish's marker — proves order.
    assert marker_render.stat().st_mtime <= marker_publish.stat().st_mtime, (
        "render must run before publish"
    )

    log_text = log.read_text()
    assert "[START] etf-dashboard-publish" in log_text
    assert "[OK] etf-dashboard-publish" in log_text
