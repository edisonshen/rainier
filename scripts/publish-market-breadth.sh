#!/bin/bash
# publish-market-breadth.sh — render today's S&P 500 market-breadth HTML
# and publish to fengshen.dev/trading/market-breadth/.
#
# Mirrors scripts/publish-etf-dashboard.sh in shape:
#
#   1. Compute the asof date + the Pacific wall-clock HH:MM HERE (not in
#      crontab) so `%` chars never reach cron's command field.
#   2. Render the HTML via `rainier market-breadth render-html`.
#   3. Hand off to the generic publisher (`publish-dashboard.sh market-breadth`).
#
# Usage:
#   scripts/publish-market-breadth.sh
#
# Environment overrides (defaults match the operator's machine):
#   DASHBOARD_SOURCE_DIR  rendered HTML lives here ($HOME/projects/rainier/out/dashboards)
#   RAINIER_UV            uv binary to use (auto-detected via `command -v uv`)
#
# Fails fast on any step; cron-wrapper.sh converts non-zero exit into a
# Discord alert (config/cron.yaml: discord_on_failure: true).

set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s [publish-market-breadth] %s\n' "$(ts)" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SOURCE_DIR="${DASHBOARD_SOURCE_DIR:-$HOME/projects/rainier/out/dashboards}"
OUTPUT="$SOURCE_DIR/market-breadth.html"
mkdir -p "$SOURCE_DIR"

RENDERED_AT_PT="$(date +%H:%M)"

# Resolve uv binary. cron's PATH is minimal, so prefer an explicit override or
# the operator's standard install location before falling back to PATH.
UV="${RAINIER_UV:-}"
if [ -z "$UV" ]; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        UV="$HOME/.local/bin/uv"
    elif command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
    else
        log "ERROR uv not found (set RAINIER_UV=/path/to/uv or install uv)"
        exit 1
    fi
fi

cd "$PROJECT_DIR"

# Don't pass --asof — the renderer pulls the latest row from the parquet
# itself. That keeps this script agnostic to weekend / market-holiday timing
# (the most recent data wins; no `% Y%m%d` string in the crontab).
log "render rendered_at_pt=$RENDERED_AT_PT output=$OUTPUT"
"$UV" run rainier market-breadth render-html \
    --rendered-at-pt "$RENDERED_AT_PT" \
    --output "$OUTPUT"

log "publish slug=market-breadth"
"$SCRIPT_DIR/publish-dashboard.sh" market-breadth
log "done"
