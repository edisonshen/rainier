#!/bin/bash
# publish-dashboard.sh — copy a rendered dashboard HTML into the fengshen-site
# Astro `public/` tree, commit + push so Cloudflare Pages auto-deploys.
#
# Usage:
#   scripts/publish-dashboard.sh <name>
#
# <name> is the dashboard slug (e.g. `etf-ranks`, `market-breadth`).
# The same script is reused across dashboards — keep it generic.
#
# ASCII flow:
#
#   rainier render CLI ─▶ $DASHBOARD_SOURCE_DIR/<name>.html
#                                │
#                                ▼   cp (creates dirs)
#                    $DASHBOARD_PUBLISH_TARGET_DIR/public/trading/<name>/index.html
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
# rendered HTML lives at `https://fengshen.dev/trading/<name>/`.

set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s [publish-dashboard] %s\n' "$(ts)" "$*"; }

if [ $# -lt 1 ]; then
    echo "usage: $(basename "$0") <name>" >&2
    exit 2
fi

NAME="$1"

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
DST_REL="public/trading/$NAME/index.html"
DST="$TARGET_DIR/$DST_REL"

log "name=$NAME source=$SRC target=$TARGET_DIR"

if [ ! -f "$SRC" ]; then
    log "ERROR rendered HTML missing: $SRC"
    exit 1
fi

if [ ! -d "$TARGET_DIR/.git" ]; then
    log "ERROR target is not a git checkout: $TARGET_DIR"
    exit 1
fi

cd "$TARGET_DIR"

# Bail if working tree has unrelated dirt — don't auto-stash.
if [ -n "$(git status --porcelain)" ]; then
    log "ERROR target working tree dirty (uncommitted changes); refusing to publish"
    git status --short >&2
    exit 1
fi

mkdir -p "$(dirname "$DST")"
cp "$SRC" "$DST"

# Track new files explicitly so `git diff --quiet --` sees the intent below.
git add -- "$DST_REL"

# `git diff --cached --quiet` returns 0 when staged tree matches HEAD, 1 otherwise.
if git diff --cached --quiet -- "$DST_REL"; then
    # Reset the noop stage to keep the working tree pristine.
    git reset --quiet HEAD -- "$DST_REL" >/dev/null 2>&1 || true
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
