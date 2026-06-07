#!/bin/bash
# publish-dashboard.sh — copy a rendered dashboard HTML into the fengshen-site
# Astro `public/` tree, commit + push so Cloudflare Pages auto-deploys.
#
# Usage:
#   scripts/publish-dashboard.sh <name>           # publishes to /trading/<name>/
#   scripts/publish-dashboard.sh <name> --root    # publishes to /trading/ (root)
#
# <name> is the dashboard slug (e.g. `etf-ranks`, `market-breadth`, `dashboard`).
# The same script is reused across dashboards — keep it generic.
#
# The `--root` flag promotes the dashboard to the `/trading/` landing page
# rather than nesting it under `/trading/<name>/`. Used by the combined
# trading dashboard (DESIGN-trading-dashboard-combined-v1.md §4 D1 —
# "the url will be /trading"). The source HTML is still read from
# `$DASHBOARD_SOURCE_DIR/<name>.html`; only the destination path changes.
#
# SELF-ISOLATING PUBLISH (no dependency on the target checkout's state)
# ---------------------------------------------------------------------
# $TARGET_DIR (DASHBOARD_PUBLISH_TARGET_DIR, e.g. the operator's fengshen-site)
# is a SHARED working tree — other agents leave it parked on their own worker
# branches, sometimes dirty. Committing/pushing from it would either trip the
# dirty-guard or push the WRONG branch. So this script NEVER operates on that
# checkout. It treats $TARGET_DIR purely as "the repo to source a worktree
# from": it fetches origin and publishes via an EPHEMERAL detached worktree on
# origin/master, then removes it (trap on EXIT, so cleanup runs on failure too).
#
# ASCII flow:
#
#   rainier render CLI ─▶ $DASHBOARD_SOURCE_DIR/<name>.html
#                                │
#   git -C $TARGET_DIR fetch origin                  (no checkout mutation)
#   WT=$(mktemp -d); git -C $TARGET_DIR worktree add --detach $WT origin/master
#                                │   trap: worktree remove --force + prune (EXIT)
#                                ▼   cp into $WT/$DST_REL (creates dirs)
#                    default:    $WT/public/trading/<name>/index.html
#                    with --root: $WT/public/trading/index.html
#                                │
#                                ▼   staged diff vs HEAD (origin/master) ?
#                          ┌──────┴───────┐
#                       no │              │ yes
#                          ▼              ▼
#                       exit 0     git -C $WT commit "<name>: YYYY-MM-DD daily render"
#                       (no-op)         → push origin HEAD:refs/heads/master
#                                          (non-FF? re-fetch, reset $WT to new
#                                           origin/master, re-apply, retry ≤3x)
#                                                       │
#                                                       ▼
#                                            Cloudflare Pages auto-deploys
#
# Environment overrides (defaults match the operator's machine):
#   DASHBOARD_SOURCE_DIR          rendered HTML lives here ($HOME/projects/rainier/out/dashboards)
#   DASHBOARD_PUBLISH_TARGET_DIR  fengshen-site repo to SOURCE the worktree from
#                                 ($HOME/projects/fengshen-site)
#
# Bootstrap note: the first publish for a brand-new <name> creates
# `public/trading/<name>/` inside the ephemeral worktree automatically — no
# manual `.gitkeep` pre-step required. Astro serves `public/` verbatim, so the
# rendered HTML lives at `https://fengshen.dev/trading/<name>/` (or
# `https://fengshen.dev/trading/` when `--root` is set).

set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s [publish-dashboard] %s\n' "$(ts)" "$*"; }

if [ $# -lt 1 ]; then
    echo "usage: $(basename "$0") <name> [--root]" >&2
    exit 2
fi

NAME="$1"
ROOT_PUBLISH=0
shift
while [ $# -gt 0 ]; do
    case "$1" in
        --root)
            ROOT_PUBLISH=1
            shift
            ;;
        *)
            echo "error: unknown flag: $1" >&2
            echo "usage: $(basename "$0") <name> [--root]" >&2
            exit 2
            ;;
    esac
done

# Sanity: slug must be filesystem-safe (no path traversal).
case "$NAME" in
    */*|*..*|"")
        echo "error: invalid dashboard name: $NAME" >&2
        exit 2
        ;;
esac

SOURCE_DIR="${DASHBOARD_SOURCE_DIR:-$HOME/projects/rainier/out/dashboards}"
TARGET_DIR="${DASHBOARD_PUBLISH_TARGET_DIR:-$HOME/projects/fengshen-site}"
SRC="$SOURCE_DIR/$NAME.html"
if [ "$ROOT_PUBLISH" -eq 1 ]; then
    DST_REL="public/trading/index.html"
else
    DST_REL="public/trading/$NAME/index.html"
fi

# Branch on origin that fengshen-site deploys from. fengshen-site deploys
# `master`; overridable so the test harness (bare origin seeded on `main`)
# can exercise the same flow.
PUBLISH_BRANCH="${DASHBOARD_PUBLISH_BRANCH:-master}"

log "name=$NAME source=$SRC target=$TARGET_DIR branch=$PUBLISH_BRANCH"

if [ ! -f "$SRC" ]; then
    log "ERROR rendered HTML missing: $SRC"
    exit 1
fi

# Accept both a regular checkout (.git is a directory) and a linked worktree
# (.git is a FILE — a gitdir pointer created by `git worktree add`). The old
# `-d "$TARGET_DIR/.git"` test wrongly rejected worktrees, breaking the
# shared-tree publish path.
#
# We require the target to be the worktree ROOT, not merely inside one: ask git
# for the worktree top-level and compare it to TARGET_DIR. `--show-toplevel`
# prints the root for both a regular checkout and a linked worktree, and errors
# (empty output) for a non-repo — so this still rejects non-repos AND a
# misconfigured subdir like `fengshen-site/public` (where DST_REL would
# otherwise nest under the subdir and commit `public/public/trading/...`).
TARGET_TOPLEVEL="$(git -C "$TARGET_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
# Resolve both to physical paths so a symlinked or trailing-slash TARGET_DIR
# still matches git's canonical top-level.
TARGET_ABS="$(cd "$TARGET_DIR" 2>/dev/null && pwd -P || true)"
TARGET_TOP_ABS=""
if [ -n "$TARGET_TOPLEVEL" ]; then
    # `|| true` so a permission/race failure on cd degrades to the friendly
    # error below instead of a bare `set -e` abort with no log line.
    TARGET_TOP_ABS="$(cd "$TARGET_TOPLEVEL" 2>/dev/null && pwd -P || true)"
fi
if [ -z "$TARGET_TOP_ABS" ] || [ "$TARGET_ABS" != "$TARGET_TOP_ABS" ]; then
    log "ERROR target is not a git checkout: $TARGET_DIR"
    exit 1
fi

# --- Self-isolating publish via an ephemeral origin/<branch> worktree -------
#
# We never `cd` into $TARGET_DIR's checkout or touch its branch. Instead we
# fetch origin and add a throwaway DETACHED worktree pinned at
# origin/$PUBLISH_BRANCH, do all the copy/commit/push there, then remove it.
# This makes the publish independent of whatever state the shared checkout is
# parked in (different branch, dirty tree).

log "fetching origin/$PUBLISH_BRANCH"
git -C "$TARGET_DIR" fetch --quiet origin "$PUBLISH_BRANCH"

WT="$(mktemp -d "${TMPDIR:-/tmp}/publish-dashboard.XXXXXX")"

# Always reap the ephemeral worktree — on success AND on any failure/early-exit
# (operator rule: cleanup is the last step, even on the failure path). `prune`
# clears the parent repo's registry entry if `remove` couldn't (e.g. the dir
# was already gone).
cleanup() {
    git -C "$TARGET_DIR" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
    git -C "$TARGET_DIR" worktree prune 2>/dev/null || true
}
trap cleanup EXIT

git -C "$TARGET_DIR" worktree add --quiet --detach "$WT" "origin/$PUBLISH_BRANCH"

DST="$WT/$DST_REL"

# Bounded non-fast-forward retry. The etf + market-breadth publishers run at
# ~the same minute, and other agents push to master too, so origin can advance
# between our fetch and our push. On a non-FF rejection we re-fetch, hard-reset
# the EPHEMERAL worktree onto the NEW origin tip, re-apply the file, and retry.
#
# NOTE: `git reset --hard` here is SAFE and intentional — it operates ONLY on
# $WT, this script's own throwaway detached worktree. It NEVER touches the
# shared $TARGET_DIR checkout (which is forbidden). Do not "fix" this into a
# shared-tree reset.
MAX_ATTEMPTS=3
attempt=1
while :; do
    mkdir -p "$(dirname "$DST")"
    cp "$SRC" "$DST"

    # Stage the (possibly new) file so the staged-vs-HEAD diff sees the intent.
    git -C "$WT" add -- "$DST_REL"

    # No-op: staged tree matches origin/$PUBLISH_BRANCH HEAD → nothing to do.
    # Safe for cron retries (identical render shouldn't churn commits).
    if git -C "$WT" diff --cached --quiet -- "$DST_REL"; then
        log "no-op: rendered HTML matches the published copy (no commit)"
        exit 0
    fi

    DATE_TAG="$(date -u +%Y-%m-%d)"
    MSG="$NAME: $DATE_TAG daily render"
    log "committing: $MSG (attempt $attempt/$MAX_ATTEMPTS)"
    git -C "$WT" commit -m "$MSG" --quiet -- "$DST_REL"

    log "pushing to origin/$PUBLISH_BRANCH"
    # Push the detached HEAD straight to the remote branch ref — no local
    # branch is created or moved anywhere.
    if git -C "$WT" push --quiet origin "HEAD:refs/heads/$PUBLISH_BRANCH"; then
        log "done"
        exit 0
    fi

    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        log "ERROR push rejected after $MAX_ATTEMPTS attempts (origin/$PUBLISH_BRANCH kept advancing); giving up"
        exit 1
    fi

    log "push rejected (non-fast-forward); re-fetching and rebasing onto new origin/$PUBLISH_BRANCH"
    git -C "$TARGET_DIR" fetch --quiet origin "$PUBLISH_BRANCH"
    # Reset the EPHEMERAL worktree only (see NOTE above) onto the new tip,
    # discarding our just-made commit; the loop re-applies the file + re-commits.
    git -C "$WT" reset --hard --quiet "origin/$PUBLISH_BRANCH"
    attempt=$((attempt + 1))
done
