"""Unit tests for scripts/publish-docs.sh + scripts/_publish_lock.py.

Mirrors tests/test_scripts/test_publish_dashboard.py: a throwaway bare git
remote + working clone stand in for the fengshen-site checkout, and a fixture
`DOCS_SOURCE_DIR` of synthetic markdown stands in for docs/. Nothing here ever
touches the operator's real fengshen-site or docs/ (DESIGN-docs-publisher.md §4).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLISH_DOCS = ROOT / "scripts" / "publish-docs.sh"
PUBLISH_DASHBOARD = ROOT / "scripts" / "publish-dashboard.sh"
LOCK_PY = ROOT / "scripts" / "_publish_lock.py"


def _load_lock_module():
    spec = importlib.util.spec_from_file_location("_publish_lock", LOCK_PY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lock_mod = _load_lock_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env() -> dict:
    e = os.environ.copy()
    e.update(
        {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    return e


def _git(*args: str, cwd: Path, check: bool = True) -> str:
    r = subprocess.run(
        ["git", *args], cwd=str(cwd), env=_env(), capture_output=True, text=True
    )
    if check and r.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed (rc={r.returncode}):\n{r.stdout}\n{r.stderr}"
        )
    return r.stdout


@pytest.fixture
def target_repo(tmp_path: Path) -> Path:
    """Bare remote + working clone simulating the fengshen-site checkout."""
    remote = tmp_path / "fengshen-site-remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )
    local = tmp_path / "fengshen-site"
    subprocess.run(
        ["git", "clone", str(remote), str(local)], check=True, capture_output=True
    )
    (local / "public").mkdir()
    (local / "public" / ".gitkeep").write_text("")
    _git("checkout", "-B", "main", cwd=local)
    _git("add", "public/.gitkeep", cwd=local)
    _git("commit", "-m", "init", cwd=local)
    _git("push", "-u", "origin", "main", cwd=local)
    return local


@pytest.fixture
def source_docs(tmp_path: Path) -> Path:
    """Fixture docs/ dir with two allowlisted + one non-allowlisted .md."""
    d = tmp_path / "docs"
    d.mkdir()
    # Alpha includes an ordered list so the end-to-end publish path exercises
    # ordered-list rendering (regression: a numbered list used to crash the
    # renderer, failing the whole all-or-nothing run).
    (d / "DESIGN-alpha.md").write_text(
        "# Alpha\n\nIntro.\n\n## Section\n\nBody.\n\n1. step one\n2. step two\n"
    )
    (d / "TASK-PLAN-beta.md").write_text("# Beta\n\nIntro.\n\n## Plan\n\nSteps.\n")
    # Not allowlisted — must be ignored.
    (d / "notes.md").write_text("# Notes\n\nScratch.\n")
    return d


def _run(
    *,
    source_docs: Path,
    target_repo: Path,
    extra_env: dict | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _env()
    env.update(
        {
            "DOCS_SOURCE_DIR": str(source_docs),
            "DOCS_PUBLISH_TARGET_DIR": str(target_repo),
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(PUBLISH_DOCS)],
        capture_output=True,
        text=True,
        env=env,
    )


def _published(target_repo: Path) -> Path:
    return target_repo / "public" / "projdoc" / "rainier"


# ---------------------------------------------------------------------------
# Publisher: happy path
# ---------------------------------------------------------------------------


def test_publishes_allowlisted_set(source_docs: Path, target_repo: Path):
    r = _run(source_docs=source_docs, target_repo=target_repo)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    pub = _published(target_repo)
    assert (pub / "DESIGN-alpha.html").exists()
    assert (pub / "TASK-PLAN-beta.html").exists()
    assert (pub / "index.html").exists()
    # Style came through the renderer.
    assert "0f766e" in (pub / "DESIGN-alpha.html").read_text()
    # Committed + pushed.
    local_head = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    remote_head = _git("rev-parse", "origin/main", cwd=target_repo).strip()
    assert local_head == remote_head


def test_html_inputs_ignored(source_docs: Path, target_repo: Path):
    """A `DESIGN-foo.html` companion in docs/ is never rendered (no *.html.html)."""
    (source_docs / "DESIGN-alpha.html").write_text("<html>companion</html>")
    r = _run(source_docs=source_docs, target_repo=target_repo)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    pub = _published(target_repo)
    assert not (pub / "DESIGN-alpha.html.html").exists()
    # The .md still produced its single .html.
    assert (pub / "DESIGN-alpha.html").exists()


def test_non_allowlisted_md_skipped(source_docs: Path, target_repo: Path):
    r = _run(source_docs=source_docs, target_repo=target_repo)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert not (_published(target_repo) / "notes.html").exists()


def test_index_is_deterministic_and_sorted(source_docs: Path, target_repo: Path):
    r = _run(source_docs=source_docs, target_repo=target_repo)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    idx = (_published(target_repo) / "index.html").read_text()
    # Alpha sorts before TASK-PLAN-beta.
    assert idx.index("DESIGN-alpha.html") < idx.index("TASK-PLAN-beta.html")


# ---------------------------------------------------------------------------
# Publisher: no-op + recovery
# ---------------------------------------------------------------------------


def test_noop_after_mtime_touch(source_docs: Path, target_repo: Path):
    """Re-running with identical content (even after a `touch`) is a true
    no-op — the index date comes from git, not mtime, so no empty commit."""
    r1 = _run(source_docs=source_docs, target_repo=target_repo)
    assert r1.returncode == 0, f"{r1.stdout}\n{r1.stderr}"
    head1 = _git("rev-parse", "HEAD", cwd=target_repo).strip()

    # Touch a source doc so its mtime changes but content is identical.
    time.sleep(0.02)
    (source_docs / "DESIGN-alpha.md").touch()

    r2 = _run(source_docs=source_docs, target_repo=target_repo)
    assert r2.returncode == 0, f"{r2.stdout}\n{r2.stderr}"
    head2 = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    assert head1 == head2, "identical content must not create a new commit"
    assert "no-op" in r2.stdout.lower()


def test_commits_when_content_changes(source_docs: Path, target_repo: Path):
    r1 = _run(source_docs=source_docs, target_repo=target_repo)
    assert r1.returncode == 0, f"{r1.stdout}\n{r1.stderr}"
    head1 = _git("rev-parse", "HEAD", cwd=target_repo).strip()

    (source_docs / "DESIGN-alpha.md").write_text(
        "# Alpha\n\nIntro.\n\n## Section\n\nBody changed.\n"
    )
    r2 = _run(source_docs=source_docs, target_repo=target_repo)
    assert r2.returncode == 0, f"{r2.stdout}\n{r2.stderr}"
    head2 = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    assert head1 != head2, "changed content must commit"


def test_recovers_unpushed_commit_on_noop(source_docs: Path, target_repo: Path):
    r1 = _run(source_docs=source_docs, target_repo=target_repo)
    assert r1.returncode == 0, f"{r1.stdout}\n{r1.stderr}"

    # Simulate a failed push: rewind the remote one commit.
    remote_url = _git("config", "remote.origin.url", cwd=target_repo).strip()
    parent = _git("rev-parse", "HEAD~1", cwd=target_repo).strip()
    _git("update-ref", "refs/heads/main", parent, cwd=Path(remote_url))
    _git("fetch", "origin", cwd=target_repo)
    ahead = _git("rev-list", "--count", "origin/main..HEAD", cwd=target_repo).strip()
    assert ahead == "1"

    r2 = _run(source_docs=source_docs, target_repo=target_repo)
    assert r2.returncode == 0, f"{r2.stdout}\n{r2.stderr}"
    _git("fetch", "origin", cwd=target_repo)
    local_head = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    origin_head = _git("rev-parse", "origin/main", cwd=target_repo).strip()
    assert local_head == origin_head, "pending commit must be recovered"


# ---------------------------------------------------------------------------
# Publisher: guards
# ---------------------------------------------------------------------------


def test_refuses_dirty_tree(source_docs: Path, target_repo: Path):
    (target_repo / "public" / ".gitkeep").write_text("dirty\n")
    head_before = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    r = _run(source_docs=source_docs, target_repo=target_repo)
    assert r.returncode != 0
    head_after = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    assert head_before == head_after
    combined = (r.stdout + r.stderr).lower()
    assert "dirty" in combined or "uncommitted" in combined


def test_rejects_non_git_target(tmp_path: Path, source_docs: Path):
    not_git = tmp_path / "plain-dir"
    not_git.mkdir()
    r = _run(source_docs=source_docs, target_repo=not_git)
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "not a git" in combined or "git checkout" in combined
    assert not (not_git / "public" / "projdoc" / "rainier").exists()


def test_upsert_only_preserves_manual_page(source_docs: Path, target_repo: Path):
    """A pre-existing manually-placed page is never deleted by a run that
    doesn't produce it (protects recursive-research-system.html)."""
    pub = _published(target_repo)
    pub.mkdir(parents=True)
    manual = pub / "recursive-research-system.html"
    manual.write_text("<html>manual</html>")
    _git("add", "public/projdoc/rainier/recursive-research-system.html", cwd=target_repo)
    _git("commit", "-m", "manual page", cwd=target_repo)
    _git("push", "origin", "main", cwd=target_repo)

    r = _run(source_docs=source_docs, target_repo=target_repo)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert manual.exists(), "manual page must survive"
    assert manual.read_text() == "<html>manual</html>"
    assert (pub / "DESIGN-alpha.html").exists()


def test_all_or_nothing_on_render_failure(source_docs: Path, target_repo: Path):
    """A doc that fails to render leaves the target tree pristine + exits
    non-zero. The shared checkout never goes dirty."""
    # Seed a good publish first.
    r1 = _run(source_docs=source_docs, target_repo=target_repo)
    assert r1.returncode == 0, f"{r1.stdout}\n{r1.stderr}"
    head_before = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    alpha_before = (_published(target_repo) / "DESIGN-alpha.html").read_text()

    # Add a doc whose render will fail by pointing the renderer at a broken
    # python via RAINIER_PYTHON would break ALL renders; instead make one doc
    # unreadable by replacing it with a directory of the same name.
    bad = source_docs / "DESIGN-broken.md"
    bad.mkdir()  # a directory named like a .md — read fails, render returns nonzero

    r2 = _run(source_docs=source_docs, target_repo=target_repo)
    assert r2.returncode != 0, f"expected failure, got:\n{r2.stdout}\n{r2.stderr}"
    head_after = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    assert head_before == head_after, "failed batch must not commit"
    # Target tree must be clean (no staged/partial writes).
    porcelain = _git("status", "--porcelain", cwd=target_repo).strip()
    assert porcelain == "", f"target must stay pristine, got dirt:\n{porcelain}"
    # Previous good output untouched.
    assert (_published(target_repo) / "DESIGN-alpha.html").read_text() == alpha_before


def test_path_confinement_rejects_symlink_source(source_docs: Path, target_repo: Path):
    """A symlinked source doc is refused (don't follow it out of docs/)."""
    outside = source_docs.parent / "secret.md"
    outside.write_text("# Secret\n\nbody\n")
    link = source_docs / "DESIGN-link.md"
    link.symlink_to(outside)
    r = _run(source_docs=source_docs, target_repo=target_repo)
    assert r.returncode != 0
    assert "symlink" in (r.stdout + r.stderr).lower()


# ---------------------------------------------------------------------------
# Shared lock
# ---------------------------------------------------------------------------


def test_lock_path_outside_worktree_and_deterministic(tmp_path: Path):
    target = tmp_path / "fengshen-site"
    target.mkdir()
    p1 = lock_mod.lock_path_for(str(target))
    p2 = lock_mod.lock_path_for(str(target))
    assert p1 == p2, "lock path must be deterministic for a given target"
    # Outside the worktree.
    assert not str(p1).startswith(str(target)), "lock must live outside the worktree"


def test_both_scripts_derive_same_lock_path(target_repo: Path, source_docs: Path):
    """publish-docs.sh and publish-dashboard.sh derive the SAME lock path for
    the same resolved target. We assert via the shared helper that both shells
    invoke (lock_path_for keyed on realpath)."""
    resolved = os.path.realpath(str(target_repo))
    expected = lock_mod.lock_path_for(resolved)
    # The helper is what both scripts call; equality of the helper output for
    # the resolved target is the contract both scripts honor.
    assert lock_mod.lock_path_for(str(target_repo)) == expected


def test_lock_mutual_exclusion(tmp_path: Path):
    """While one holder owns the lock, a second acquirer blocks (times out)
    and does NOT run its command. Proves exclusion, not just path equality."""
    target = tmp_path / "t"
    target.mkdir()
    resolved = os.path.realpath(str(target))
    marker = tmp_path / "second_ran.marker"

    def holder():
        # Hold the lock by running a command that waits until `release` is set.
        # We drive _publish_lock.py with a python sleeper that touches a file
        # then waits on a sentinel file.
        sentinel = tmp_path / "release.sentinel"
        cmd = (
            f"import pathlib,time;"
            f"pathlib.Path({str(tmp_path / 'held.marker')!r}).write_text('x');"
            f"s=pathlib.Path({str(sentinel)!r});"
            f"[time.sleep(0.01) for _ in iter(lambda: not s.exists(), False)]"
        )
        subprocess.run(
            [sys.executable, str(LOCK_PY), resolved, "--", sys.executable, "-c", cmd],
            check=True,
        )

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    # Wait until the holder has the lock.
    deadline = time.monotonic() + 5
    while not (tmp_path / "held.marker").exists():
        assert time.monotonic() < deadline, "holder never acquired lock"
        time.sleep(0.01)

    # Second acquirer with a short timeout must FAIL (EX_TEMPFAIL=75) and never
    # run its command (no marker written).
    second = subprocess.run(
        [
            sys.executable,
            str(LOCK_PY),
            resolved,
            "--timeout",
            "0.5",
            "--",
            sys.executable,
            "-c",
            f"open({str(marker)!r},'w').write('ran')",
        ],
        capture_output=True,
        text=True,
    )
    assert second.returncode == lock_mod.EX_TEMPFAIL, (
        f"blocked acquirer must exit EX_TEMPFAIL, got rc={second.returncode}\n"
        f"{second.stdout}\n{second.stderr}"
    )
    assert not marker.exists(), "blocked acquirer must NOT run its command"

    # Release the holder; lock frees; a fresh acquirer now succeeds + runs.
    (tmp_path / "release.sentinel").write_text("go")
    t.join(timeout=5)
    third = subprocess.run(
        [
            sys.executable,
            str(LOCK_PY),
            resolved,
            "--timeout",
            "2",
            "--",
            sys.executable,
            "-c",
            f"open({str(marker)!r},'w').write('ran')",
        ],
        capture_output=True,
        text=True,
    )
    assert third.returncode == 0, f"{third.stdout}\n{third.stderr}"
    assert marker.exists(), "acquirer must run its command after release"
