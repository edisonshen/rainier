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
# Test 5 — cron-wrapper integration: invokes render then publish in order.
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
