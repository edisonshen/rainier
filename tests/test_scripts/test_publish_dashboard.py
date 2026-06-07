"""Unit tests for scripts/publish-dashboard.sh — self-isolating publisher.

The publisher copies `out/dashboards/<name>.html` into a target git repo's
`public/trading/<name>/index.html` and pushes it to the deploy branch on
`origin`. It NEVER operates on the target checkout directly: it sources an
ephemeral detached worktree on `origin/<branch>`, commits + pushes there, and
removes the worktree afterwards (trap on EXIT). This makes the publish robust
to whatever state the shared fengshen-site checkout is parked in (a different
branch, a dirty tree).

Harness: a BARE origin repo + a working clone (TARGET_DIR). The script is
pointed at the clone via `DASHBOARD_PUBLISH_TARGET_DIR`, and the deploy branch
is overridden to `main` (the seeded branch) via `DASHBOARD_PUBLISH_BRANCH` so
CI never touches the operator's real fengshen-site checkout.

Test plan pinned by docs/TASK-PLAN-etf-dashboard-publish-cr-31aa.md §Tests and
the Part B robustness fix (extend-133).
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

# The harness seeds the bare origin on `main`; the script defaults to `master`.
# Override so the publish targets the branch the harness actually created.
PUBLISH_BRANCH = "main"


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
def remote_repo(tmp_path: Path) -> Path:
    """A BARE origin repo seeded with one commit on `main`. The publisher
    pushes here; tests read `origin/main` from a fresh fetch to assert what
    actually landed upstream (the whole point of the self-isolating flow)."""
    remote = tmp_path / "fengshen-site-remote.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(remote)],
        check=True,
        capture_output=True,
    )
    # Seed an initial commit on main via a scratch clone, then discard it.
    seed = tmp_path / "_seed"
    subprocess.run(
        ["git", "clone", str(remote), str(seed)], check=True, capture_output=True
    )
    (seed / "public").mkdir()
    (seed / "public" / ".gitkeep").write_text("")
    _git("checkout", "-B", "main", cwd=seed)
    _git("add", "public/.gitkeep", cwd=seed)
    _git("commit", "-m", "init", cwd=seed)
    _git("push", "-u", "origin", "main", cwd=seed)
    return remote


@pytest.fixture
def target_repo(tmp_path: Path, remote_repo: Path) -> Path:
    """A working clone of the bare origin — simulates the fengshen-site
    checkout the script is pointed at (DASHBOARD_PUBLISH_TARGET_DIR)."""
    local = tmp_path / "fengshen-site"
    subprocess.run(
        ["git", "clone", str(remote_repo), str(local)],
        check=True,
        capture_output=True,
    )
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
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "DASHBOARD_SOURCE_DIR": str(source_dir),
            "DASHBOARD_PUBLISH_TARGET_DIR": str(target_repo),
            "DASHBOARD_PUBLISH_BRANCH": PUBLISH_BRANCH,
            # Deterministic git identity inside the script as well.
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    if extra_env:
        env.update(extra_env)
    argv = ["bash", str(SCRIPT), name]
    if extra_args:
        argv.extend(extra_args)
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=env,
    )


def _published_html(remote_repo: Path, rel: str) -> str:
    """Read a file as it exists on the bare origin's HEAD (deploy branch)."""
    return subprocess.run(
        ["git", "-C", str(remote_repo), "show", f"{PUBLISH_BRANCH}:{rel}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _origin_head(remote_repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(remote_repo), "rev-parse", PUBLISH_BRANCH],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Test 1 — publish lands the rendered HTML on origin's deploy branch.
# ---------------------------------------------------------------------------


def test_publish_lands_on_origin(
    source_dir: Path, target_repo: Path, remote_repo: Path
):
    """A fresh publish copies the HTML, commits, and pushes to origin/<branch>."""
    name = "etf-ranks"
    html = "<html><body>first publish</body></html>"
    (source_dir / f"{name}.html").write_text(html)

    result = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    rel = f"public/trading/{name}/index.html"
    assert _published_html(remote_repo, rel) == html, "HTML must land on origin"


# ---------------------------------------------------------------------------
# Test 2 — THE KEY TEST: the publish succeeds even when the target checkout is
# on a DIFFERENT branch and DIRTY. This proves the publisher no longer depends
# on the shared checkout's state (Part B, extend-133).
# ---------------------------------------------------------------------------


def test_publish_succeeds_when_checkout_dirty_and_off_branch(
    source_dir: Path, target_repo: Path, remote_repo: Path
):
    """Park the TARGET_DIR clone on a different branch AND make it dirty; the
    publish must STILL land on origin's deploy branch."""
    name = "market-breadth"
    html = "<html><body>breadth published from isolated worktree</body></html>"
    (source_dir / f"{name}.html").write_text(html)

    # Park the checkout on a worker branch and dirty both a tracked and an
    # untracked file — exactly the shared-tree state that broke the old flow.
    _git("checkout", "-b", "worker/some-other-work", cwd=target_repo)
    (target_repo / "public" / ".gitkeep").write_text("hand-edited\n")
    (target_repo / "stray-uncommitted.txt").write_text("WIP\n")

    head_before = _origin_head(remote_repo)
    result = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert result.returncode == 0, (
        "publish must succeed regardless of checkout state\n"
        f"rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    rel = f"public/trading/{name}/index.html"
    assert _published_html(remote_repo, rel) == html, (
        "HTML must land on origin even from a dirty / off-branch checkout"
    )
    assert _origin_head(remote_repo) != head_before, "origin must have advanced"

    # The shared checkout's dirt is untouched — we never operated on it.
    assert (target_repo / "stray-uncommitted.txt").exists()
    assert (target_repo / "public" / ".gitkeep").read_text() == "hand-edited\n"
    current_branch = _git(
        "rev-parse", "--abbrev-ref", "HEAD", cwd=target_repo
    ).strip()
    assert current_branch == "worker/some-other-work", (
        "publish must not move the checkout off its branch"
    )


# ---------------------------------------------------------------------------
# Test 3 — modified source HTML produces a `<name>: YYYY-MM-DD daily render`
# commit on origin.
# ---------------------------------------------------------------------------


def test_publish_commits_when_changed(
    source_dir: Path, target_repo: Path, remote_repo: Path
):
    name = "etf-ranks"
    (source_dir / f"{name}.html").write_text("<html>v1</html>")
    r1 = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert r1.returncode == 0, r1.stderr

    (source_dir / f"{name}.html").write_text("<html>v2 updated</html>")
    r2 = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert r2.returncode == 0, (
        f"rc={r2.returncode}\nstdout: {r2.stdout}\nstderr: {r2.stderr}"
    )

    head_msg = subprocess.run(
        ["git", "-C", str(remote_repo), "log", "-1", "--pretty=%s", PUBLISH_BRANCH],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert re.match(
        rf"^{re.escape(name)}: \d{{4}}-\d{{2}}-\d{{2}} daily render$", head_msg
    ), f"unexpected commit subject: {head_msg!r}"

    rel = f"public/trading/{name}/index.html"
    assert _published_html(remote_repo, rel) == "<html>v2 updated</html>"


# ---------------------------------------------------------------------------
# Test 4 — no-op when rendered HTML is byte-identical to what's published.
# ---------------------------------------------------------------------------


def test_publish_no_op_on_identical(
    source_dir: Path, target_repo: Path, remote_repo: Path
):
    name = "etf-ranks"
    html = "<html><body>identical</body></html>"
    (source_dir / f"{name}.html").write_text(html)

    r1 = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert r1.returncode == 0, r1.stderr
    head_after_first = _origin_head(remote_repo)

    # Second run, identical content → no commit, no push, clean exit.
    r2 = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert r2.returncode == 0, (
        f"rc={r2.returncode}\nstdout: {r2.stdout}\nstderr: {r2.stderr}"
    )
    assert "no-op" in r2.stdout.lower() or "no change" in r2.stdout.lower()
    assert _origin_head(remote_repo) == head_after_first, (
        "identical render must not create a new commit on origin"
    )


# ---------------------------------------------------------------------------
# Test 5 — ephemeral worktree is cleaned up after the run (success path).
# No leftover worktrees registered, tmp dir gone.
# ---------------------------------------------------------------------------


def test_publish_cleans_up_worktree(
    source_dir: Path, target_repo: Path, remote_repo: Path
):
    name = "etf-ranks"
    (source_dir / f"{name}.html").write_text("<html>v1</html>")
    result = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert result.returncode == 0, result.stderr

    # `git worktree list` should show only the main checkout (no leftovers).
    wt_list = _git("worktree", "list", cwd=target_repo)
    lines = [ln for ln in wt_list.splitlines() if ln.strip()]
    assert len(lines) == 1, (
        f"expected only the main worktree, got:\n{wt_list}"
    )
    # And no stray publish-dashboard.* temp dirs survived.
    assert not list(Path("/tmp").glob("publish-dashboard.*")) or all(
        not p.exists() for p in Path("/tmp").glob("publish-dashboard.*")
    ), "ephemeral worktree tmp dir must be removed"


# ---------------------------------------------------------------------------
# Test 6 — concurrency: a non-fast-forward push (origin advanced under us) is
# retried by re-fetching + rebasing the ephemeral worktree, then re-pushing.
# ---------------------------------------------------------------------------


def test_publish_retries_on_non_fast_forward(
    source_dir: Path, target_repo: Path, remote_repo: Path, tmp_path: Path
):
    """Force a REAL non-fast-forward on the publisher's first push, then assert
    it re-fetches, rebases its ephemeral worktree onto the new tip, and lands
    on the second attempt — without clobbering the concurrent commit.

    Mechanism: a `pre-receive` hook on the bare origin advances the branch with
    a concurrent commit during the publisher's FIRST push, so that push is
    rejected non-FF; on the SECOND push the hook is inert (one-shot via a flag
    file) and the push succeeds. This deterministically exercises the retry
    loop instead of relying on wall-clock timing."""
    name = "etf-ranks"
    (source_dir / f"{name}.html").write_text("<html>ours</html>")

    # One-shot pre-receive hook: on its first invocation it injects a
    # concurrent commit onto the branch (advancing origin under the pushing
    # client, which makes the client's update non-FF and rejected), then drops
    # a flag so subsequent pushes pass through untouched.
    hooks = remote_repo / "hooks"
    hooks.mkdir(exist_ok=True)
    flag = tmp_path / "hook-fired.flag"
    concurrent_tree_repo = tmp_path / "_hook_worker"
    subprocess.run(
        ["git", "clone", str(remote_repo), str(concurrent_tree_repo)],
        check=True,
        capture_output=True,
    )
    (concurrent_tree_repo / "CONCURRENT.txt").write_text("from another agent\n")
    _git("add", "CONCURRENT.txt", cwd=concurrent_tree_repo)
    _git("commit", "-m", "concurrent: unrelated change", cwd=concurrent_tree_repo)
    concurrent_sha = _git("rev-parse", "HEAD", cwd=concurrent_tree_repo).strip()

    # Seed the concurrent commit's objects into the bare repo (a hidden ref)
    # BEFORE the hook is installed, so the hook's update-ref can reach the SHA.
    _git("push", "origin", f"{concurrent_sha}:refs/heads/_seed_objects",
         cwd=concurrent_tree_repo)

    # Modern git runs pre-receive inside a quarantine env where ref updates are
    # forbidden; unset the quarantine vars and target the bare git-dir directly
    # so the hook can advance the branch (simulating a concurrent push that
    # landed between the publisher's fetch and its push).
    pre_receive = hooks / "pre-receive"
    pre_receive.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        f'if [ ! -f "{flag}" ]; then\n'
        f'  touch "{flag}"\n'
        "  env -u GIT_QUARANTINE_PATH -u GIT_OBJECT_DIRECTORY "
        "-u GIT_ALTERNATE_OBJECT_DIRECTORIES \\\n"
        f'    git --git-dir="{remote_repo}" update-ref '
        f"refs/heads/{PUBLISH_BRANCH} {concurrent_sha}\n"
        '  echo "hook: injected concurrent commit, rejecting this push" >&2\n'
        "  exit 1\n"
        "fi\n"
        "exit 0\n"
    )
    pre_receive.chmod(0o755)

    result = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert result.returncode == 0, (
        "publish must recover from a non-FF rejection via retry\n"
        f"rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert flag.exists(), "the non-FF hook must have fired (race not exercised)"
    combined = (result.stdout + result.stderr).lower()
    assert "non-fast-forward" in combined or "rejected" in combined, (
        "publisher should log the non-FF retry; "
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Both the concurrent file AND our render must exist on origin (no clobber).
    rel = f"public/trading/{name}/index.html"
    assert _published_html(remote_repo, rel) == "<html>ours</html>"
    concurrent = subprocess.run(
        ["git", "-C", str(remote_repo), "show", f"{PUBLISH_BRANCH}:CONCURRENT.txt"],
        capture_output=True,
        text=True,
    )
    assert concurrent.returncode == 0, (
        "concurrent agent's commit must survive (publish must not clobber it)"
    )


# ---------------------------------------------------------------------------
# Test 7 — missing source HTML produces a clean non-zero exit (no copy, no
# commit, nothing pushed).
# ---------------------------------------------------------------------------


def test_publish_errors_when_source_missing(
    source_dir: Path, target_repo: Path, remote_repo: Path
):
    name = "etf-ranks"
    assert not (source_dir / f"{name}.html").exists()

    head_before = _origin_head(remote_repo)
    result = _run_publisher(name, source_dir=source_dir, target_repo=target_repo)
    assert result.returncode != 0, (
        f"missing source must produce non-zero exit, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert _origin_head(remote_repo) == head_before, "must not push when source missing"
    combined = (result.stdout + result.stderr).lower()
    assert "missing" in combined or "no such" in combined


# ---------------------------------------------------------------------------
# Test 8 — target is not a git checkout → clean non-zero exit, nothing pushed.
# ---------------------------------------------------------------------------


def test_publish_errors_when_target_not_git(tmp_path: Path, source_dir: Path):
    name = "etf-ranks"
    (source_dir / f"{name}.html").write_text("<html>v1</html>")

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
    assert "not a git" in combined or "git checkout" in combined


# ---------------------------------------------------------------------------
# Test 8b — a SUBDIRECTORY inside a repo (not the worktree root) is rejected.
# The checkout guard requires the target to be the worktree top-level so the
# publish path can't accidentally nest under a subdir. (Carried from #133.)
# ---------------------------------------------------------------------------


def test_publish_rejects_subdir_of_repo(source_dir: Path, target_repo: Path):
    name = "etf-ranks"
    (source_dir / f"{name}.html").write_text("<html>v1</html>")

    subdir = target_repo / "public"  # inside the work tree, but not its root
    assert subdir.is_dir()

    result = _run_publisher(name, source_dir=source_dir, target_repo=subdir)
    assert result.returncode != 0, (
        f"subdir target must be rejected, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "not a git" in combined or "git checkout" in combined


# ---------------------------------------------------------------------------
# Test 8c — a linked git worktree (`git worktree add`) is still a valid target.
# Its `.git` is a FILE (gitdir pointer), not a directory; the checkout guard
# must accept it and proceed. (Carried from #133 — the publisher now sources
# its OWN worktree from this target, so the target itself may legitimately be
# a worktree.)
# ---------------------------------------------------------------------------


def test_publish_accepts_linked_worktree(
    tmp_path: Path, source_dir: Path, target_repo: Path, remote_repo: Path
):
    name = "etf-ranks"
    html = "<html><body>worktree publish</body></html>"
    (source_dir / f"{name}.html").write_text(html)

    # Create a linked worktree of the target clone (its `.git` is a FILE).
    worktree = tmp_path / "fengshen-site-worktree"
    _git(
        "worktree", "add", "-b", "publish-wt", str(worktree), PUBLISH_BRANCH,
        cwd=target_repo,
    )
    assert (worktree / ".git").is_file(), (
        "test precondition: a linked worktree's .git must be a FILE, not a dir"
    )

    result = _run_publisher(name, source_dir=source_dir, target_repo=worktree)
    combined = (result.stdout + result.stderr).lower()
    assert "target is not a git checkout" not in combined, (
        "checkout guard wrongly rejected a linked worktree\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.returncode == 0, (
        f"publish from a linked-worktree target must succeed, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    rel = f"public/trading/{name}/index.html"
    assert _published_html(remote_repo, rel) == html


# ---------------------------------------------------------------------------
# Test 9 — `--root` flag publishes to `public/trading/index.html`.
# ---------------------------------------------------------------------------


def test_publish_root_flag_lands_at_trading_root(
    source_dir: Path, target_repo: Path, remote_repo: Path
):
    name = "dashboard"
    html = "<html><body>combined v1</body></html>"
    (source_dir / f"{name}.html").write_text(html)

    result = _run_publisher(
        name, source_dir=source_dir, target_repo=target_repo, extra_args=["--root"]
    )
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert _published_html(remote_repo, "public/trading/index.html") == html
    # The nested per-name path must NOT also be created.
    nested = subprocess.run(
        ["git", "-C", str(remote_repo), "show",
         f"{PUBLISH_BRANCH}:public/trading/{name}/index.html"],
        capture_output=True,
        text=True,
    )
    assert nested.returncode != 0, "--root must not also create the nested path"


def test_publish_root_flag_rejects_unknown_args(
    source_dir: Path, target_repo: Path
):
    name = "dashboard"
    (source_dir / f"{name}.html").write_text("<html>v1</html>")
    result = _run_publisher(
        name, source_dir=source_dir, target_repo=target_repo, extra_args=["--bogus"]
    )
    assert result.returncode != 0, (
        f"unknown flag must produce non-zero exit, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "unknown" in combined or "usage" in combined


def test_publish_rejects_path_traversal_name(
    source_dir: Path, target_repo: Path
):
    """A `<name>` with `/` or `..` is rejected before any git op (carried)."""
    (source_dir / "etf-ranks.html").write_text("<html>v1</html>")
    for bad in ["../escape", "a/b", ".."]:
        result = _run_publisher(bad, source_dir=source_dir, target_repo=target_repo)
        assert result.returncode != 0, f"name {bad!r} must be rejected"
        combined = (result.stdout + result.stderr).lower()
        assert "invalid" in combined or "usage" in combined


# ---------------------------------------------------------------------------
# Test 10 — cron-wrapper integration: invokes render then publish in order.
# ---------------------------------------------------------------------------


def test_cron_wrapper_invokes_render_then_publish(tmp_path: Path):
    """`cron-wrapper.sh` runs the supplied command verbatim and logs phases."""
    project = tmp_path / "rainier"
    (project / "scripts").mkdir(parents=True)
    wrapper_dst = project / "scripts" / "cron-wrapper.sh"
    wrapper_dst.write_text(CRON_WRAPPER.read_text())
    wrapper_dst.chmod(0o755)

    marker_render = tmp_path / "render.marker"
    marker_publish = tmp_path / "publish.marker"
    log = tmp_path / "cron.log"

    command = (
        f"date >> {marker_render} && "
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
    assert marker_render.stat().st_mtime <= marker_publish.stat().st_mtime, (
        "render must run before publish"
    )

    log_text = log.read_text()
    assert "[START] etf-dashboard-publish" in log_text
    assert "[OK] etf-dashboard-publish" in log_text
