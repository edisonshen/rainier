"""Unit tests for scripts/publish-etf-dashboard.sh.

The wrapper computes today's ASOF, invokes the renderer via `$RAINIER_UV run
rainier dashboard render-etf-html`, guards against the renderer's
"valid-but-empty" output (`Universe: 0 sector/theme ETFs`), and chains to
the generic `publish-dashboard.sh etf-ranks` publisher.

Tests stub `RAINIER_UV` so we never invoke the real `uv` / `rainier` /
parquet pipeline. The stub writes deterministic HTML to the expected
output path, then the wrapper makes the publish decision.

Why this wrapper exists (re-stated for reviewers reading the test alone):

  1. The crontab line cannot contain `%` characters — cron's command-field
     parser treats unescaped `%` as a line terminator. Putting
     `$(date +%Y-%m-%d)` directly in cron.yaml made the job silently break
     once `rainier jobs sync` installed it into the system crontab. The
     wrapper computes ASOF in plain shell, off the crontab line.

  2. The renderer's no-data path returns a valid-but-empty HTML page on
     market holidays / stale OHLCV cache. Without a guard, publish-
     dashboard.sh would overwrite the last good public dashboard with an
     empty one. The wrapper checks for "Universe: 0 " in the rendered
     HTML and bails before handing off.
"""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publish-etf-dashboard.sh"
CRON_CONFIG = ROOT / "config" / "cron.yaml"

# Sentinel string the renderer emits when no features exist for ASOF.
EMPTY_MARKER = "Universe: 0 sector/theme ETFs"
HAPPY_MARKER = "Universe: 12 sector/theme ETFs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path, check: bool = True, env: dict | None = None) -> str:
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
    """Bare remote + working clone, mirroring fengshen-site's layout."""
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
def source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "out" / "dashboards"
    d.mkdir(parents=True)
    return d


def _write_uv_stub(tmp_path: Path, source_dir: Path, content: str) -> Path:
    """Build a stand-in for `$RAINIER_UV` that mimics `uv run rainier
    dashboard render-etf-html` by writing the given HTML content to the
    --output path the wrapper passes in.

    The stub also asserts that the wrapper passed `--asof` in YYYY-MM-DD
    form so we'd catch a regression in date formatting.
    """
    stub = tmp_path / "uv-stub.sh"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            # Args from wrapper:
            #   $1 = "run"
            #   $2 = "rainier"
            #   $3 = "dashboard"
            #   $4 = "render-etf-html"
            #   $5..$N = flag pairs (--asof DATE --rendered-at-pt HH:MM
            #                        --output PATH ...)
            set -euo pipefail
            OUTPUT=""
            ASOF=""
            shift 4
            while [ $# -gt 0 ]; do
                case "$1" in
                    --output) OUTPUT="$2"; shift 2 ;;
                    --asof) ASOF="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            # Sanity: wrapper should have passed YYYY-MM-DD.
            if ! [[ "$ASOF" =~ ^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$ ]]; then
                echo "uv-stub: bad --asof: $ASOF" >&2
                exit 99
            fi
            mkdir -p "$(dirname "$OUTPUT")"
            cat > "$OUTPUT" <<'HTML_EOF'
            {content}
            HTML_EOF
            """
        )
    )
    stub.chmod(0o755)
    return stub


def _run_wrapper(
    *,
    tmp_path: Path,
    source_dir: Path,
    target_repo: Path,
    uv_stub: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "RAINIER_UV": str(uv_stub),
            "DASHBOARD_SOURCE_DIR": str(source_dir),
            "DASHBOARD_PUBLISH_TARGET_DIR": str(target_repo),
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
    )


# ---------------------------------------------------------------------------
# Test 1 — cron.yaml command field is `%`-free.
#
# Cron's command-field parser treats unescaped `%` as a line terminator
# (everything after the first `%` is sent on stdin). The etf-dashboard-publish
# job MUST NOT contain `%` in its command — date computation moved into a
# wrapper script. Regression for the originally-committed inline `$(date
# +%Y-%m-%d)` form that codex flagged as P1.
# ---------------------------------------------------------------------------


def test_cron_command_field_is_percent_free():
    with open(CRON_CONFIG) as f:
        data = yaml.safe_load(f)
    jobs = [j for j in data["jobs"] if j["name"] == "etf-dashboard-publish"]
    assert len(jobs) == 1, "etf-dashboard-publish job missing from cron.yaml"
    cmd = jobs[0]["command"]
    assert "%" not in cmd, (
        f"etf-dashboard-publish.command must not contain `%` (crontab splits on "
        f"it). Move date computation into a wrapper script. Got: {cmd!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — empty render is NOT published.
#
# Renderer emits a valid-but-empty page on market holidays / stale cache
# ("Universe: 0 sector/theme ETFs"). Without a guard, the empty page would
# replace the last good public dashboard. Wrapper must detect + bail.
# ---------------------------------------------------------------------------


def test_empty_render_is_not_published(
    tmp_path: Path, source_dir: Path, target_repo: Path
):
    empty_html = f"<html><body><p>{EMPTY_MARKER}</p></body></html>"
    uv_stub = _write_uv_stub(tmp_path, source_dir, empty_html)

    head_before = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    result = _run_wrapper(
        tmp_path=tmp_path,
        source_dir=source_dir,
        target_repo=target_repo,
        uv_stub=uv_stub,
    )
    head_after = _git("rev-parse", "HEAD", cwd=target_repo).strip()

    assert result.returncode != 0, (
        f"empty render must exit non-zero, got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert head_before == head_after, "must not commit on empty render"

    dst = target_repo / "public" / "trading" / "etf-ranks" / "index.html"
    assert not dst.exists(), "must not publish empty HTML"

    combined = (result.stdout + result.stderr).lower()
    assert "empty" in combined or "no features" in combined, (
        f"expected actionable error mentioning empty render, got:\n"
        f"{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 3 — non-empty render flows through to publish-dashboard.sh and lands a
# commit + push.
# ---------------------------------------------------------------------------


def test_happy_path_publishes(
    tmp_path: Path, source_dir: Path, target_repo: Path
):
    happy_html = f"<html><body><p>{HAPPY_MARKER}</p><p>data here</p></body></html>"
    uv_stub = _write_uv_stub(tmp_path, source_dir, happy_html)

    result = _run_wrapper(
        tmp_path=tmp_path,
        source_dir=source_dir,
        target_repo=target_repo,
        uv_stub=uv_stub,
    )
    assert result.returncode == 0, (
        f"happy path failed: rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    dst = target_repo / "public" / "trading" / "etf-ranks" / "index.html"
    assert dst.exists(), "publish step did not land index.html"
    # Read against the rendered content (textwrap.dedent in the stub adds a
    # trailing newline, so use `in` rather than equality).
    assert HAPPY_MARKER in dst.read_text()

    # Commit subject from publish-dashboard.sh.
    head_msg = _git("log", "-1", "--pretty=%s", cwd=target_repo).strip()
    assert re.match(
        r"^etf-ranks: \d{4}-\d{2}-\d{2} daily render$", head_msg
    ), f"unexpected commit subject: {head_msg!r}"

    # And pushed to origin.
    local_head = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    remote_head = _git("rev-parse", "origin/main", cwd=target_repo).strip()
    assert local_head == remote_head, "wrapper did not push to origin"


# ---------------------------------------------------------------------------
# Test 4 — renderer failure halts the chain (publish-dashboard.sh is NOT
# invoked). `set -euo pipefail` ensures non-zero `uv run ...` propagates.
# ---------------------------------------------------------------------------


def test_renderer_failure_halts_chain(
    tmp_path: Path, source_dir: Path, target_repo: Path
):
    # Stub that always exits 1 without writing the output.
    stub = tmp_path / "uv-fail.sh"
    stub.write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            echo "uv-fail-stub: simulated render failure" >&2
            exit 1
            """
        )
    )
    stub.chmod(0o755)

    head_before = _git("rev-parse", "HEAD", cwd=target_repo).strip()
    result = _run_wrapper(
        tmp_path=tmp_path,
        source_dir=source_dir,
        target_repo=target_repo,
        uv_stub=stub,
    )
    head_after = _git("rev-parse", "HEAD", cwd=target_repo).strip()

    assert result.returncode != 0, "renderer failure must propagate"
    assert head_before == head_after, "no commit on renderer failure"

    dst = target_repo / "public" / "trading" / "etf-ranks" / "index.html"
    assert not dst.exists(), "must not publish when render failed"
