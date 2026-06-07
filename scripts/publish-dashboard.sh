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
# ASCII flow:
#
#   rainier render CLI ─▶ $DASHBOARD_SOURCE_DIR/<name>.html
#                                │
#                                ▼   cp (creates dirs)
#                    default:    $DASHBOARD_PUBLISH_TARGET_DIR/public/trading/<name>/index.html
#                    with --root: $DASHBOARD_PUBLISH_TARGET_DIR/public/trading/index.html
#                                │
#                                ▼   git diff --quiet ?
#                          ┌──────┴───────┐
#                       yes│              │no
#                          ▼              ▼
#                       exit 0     git add → commit "<name>: YYYY-MM-DD daily render"
#                       (no-op)                 → git push (origin/main)
#                                                       │
#                                                       ▼
#                                            Cloudflare Pages auto-deploys
#
# Environment overrides (defaults match the operator's machine):
#   DASHBOARD_SOURCE_DIR          rendered HTML lives here ($HOME/projects/rainier/out/dashboards)
#   DASHBOARD_PUBLISH_TARGET_DIR  fengshen-site checkout    ($HOME/projects/fengshen-site)
#
# Bootstrap note: the first publish for a brand-new <name> creates
# `public/trading/<name>/` inside fengshen-site automatically — no manual
# `.gitkeep` pre-step required. Astro serves `public/` verbatim, so the
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
DST="$TARGET_DIR/$DST_REL"

log "name=$NAME source=$SRC target=$TARGET_DIR"

if [ ! -f "$SRC" ]; then
    log "ERROR rendered HTML missing: $SRC"
    exit 1
fi

# Accept both a regular checkout (.git is a directory) and a linked worktree
# (.git is a FILE — a gitdir pointer created by `git worktree add`). The old
# `-d "$TARGET_DIR/.git"` test wrongly rejected worktrees, breaking the
# shared-tree publish path. Ask git itself, and compare the printed value
# ("true") rather than relying on exit code, which can be 0 even in bare/.git
# internal dirs where this prints "false".
if [ "$(git -C "$TARGET_DIR" rev-parse --is-inside-work-tree 2>/dev/null)" != "true" ]; then
    log "ERROR target is not a git checkout: $TARGET_DIR"
    exit 1
fi

cd "$TARGET_DIR"

# Bail if the working tree has unrelated dirt — don't auto-stash.
#
# The dirty decision is based on RELEVANT changes only: modified/staged/deleted
# TRACKED files and stray untracked files. git-IGNORED paths are explicitly
# disregarded — tool droppings such as `.gstack/browse-audit.jsonl` (left by the
# gstack browse tooling) must never block a publish. This was the 2026-06-04 P0:
# an untracked-but-ignored `.gstack/` tripped the guard and the breadth
# dashboard never went live.
#
# `git status --porcelain` already omits ignored files by default, but we make
# the contract explicit (and log it) so a future git/config change to that
# default can't silently re-break the publish. We additionally enumerate the
# ignored paths that WERE present, purely for an operator-visible log line.
RELEVANT_STATUS="$(git status --porcelain --untracked-files=normal)"
if [ -n "$RELEVANT_STATUS" ]; then
    log "ERROR target working tree dirty (uncommitted changes); refusing to publish"
    git status --short >&2
    exit 1
fi

# At this point the tree is clean except (possibly) for git-ignored paths.
# Surface them so cron logs show the guard ran and chose to disregard them.
# `--ignored` lists ignored entries as `!! <path>`; we keep only those lines.
IGNORED_COUNT="$(git status --porcelain --ignored \
    | grep -c '^!! ' || true)"
if [ "${IGNORED_COUNT:-0}" -gt 0 ]; then
    log "disregarding $IGNORED_COUNT git-ignored path(s) in target tree (not dirt)"
fi

mkdir -p "$(dirname "$DST")"
cp "$SRC" "$DST"

# Track new files explicitly so `git diff --quiet --` sees the intent below.
git add -- "$DST_REL"

# `git diff --cached --quiet` returns 0 when staged tree matches HEAD, 1 otherwise.
if git diff --cached --quiet -- "$DST_REL"; then
    # Reset the noop stage to keep the working tree pristine.
    git reset --quiet HEAD -- "$DST_REL" >/dev/null 2>&1 || true
    # Recovery: if a prior run's commit never made it upstream (e.g., a
    # non-fast-forward `git push` failure from an unrelated remote advance),
    # the local branch is ahead of @{u}. Without this check, today's no-op
    # would silently abandon that unpushed commit. Detect + push it now.
    AHEAD="$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)"
    if [ "$AHEAD" -gt 0 ]; then
        log "no-op render, but $AHEAD local commit(s) ahead of upstream — pushing"
        git push --quiet
        log "done (recovered unpushed commits)"
        exit 0
    fi
    log "no-op: rendered HTML matches the published copy (no commit)"
    exit 0
fi

DATE_TAG="$(date -u +%Y-%m-%d)"
MSG="$NAME: $DATE_TAG daily render"
log "committing: $MSG"
git commit -m "$MSG" --quiet -- "$DST_REL"

log "pushing to origin"
git push --quiet
log "done"
