#!/bin/bash
# publish-etf-dashboard.sh — render today's ETF ranks HTML and publish to
# fengshen.dev. Single entry point for the cron job, isolating two concerns
# from the cron command field:
#
#   1. `%` characters in `$(date +%Y-%m-%d)` would be split by cron's
#      command-field parser (everything after `%` becomes stdin). Moving the
#      date computation inside a script keeps the crontab line `%`-free.
#   2. The renderer's "no data" path produces a valid-but-empty HTML page
#      (`Universe: 0 sector/theme ETFs`). Publishing that would overwrite a
#      good dashboard on a market holiday or stale cache. Guard against it
#      here before handing off to the generic publisher.
#
# Usage:
#   scripts/publish-etf-dashboard.sh
#
# Environment overrides (defaults match the operator's machine):
#   DASHBOARD_SOURCE_DIR  rendered HTML lives here ($HOME/projects/rainier/out/dashboards)
#   RAINIER_UV            uv binary to use (auto-detected via `command -v uv`)
#
# Fails fast on any step; cron-wrapper.sh converts non-zero exit into a
# Discord alert (config/cron.yaml: discord_on_failure: true).

set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s [publish-etf-dashboard] %s\n' "$(ts)" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SOURCE_DIR="${DASHBOARD_SOURCE_DIR:-$HOME/projects/rainier/out/dashboards}"
OUTPUT="$SOURCE_DIR/etf-ranks.html"
mkdir -p "$SOURCE_DIR"

ASOF="$(date +%Y-%m-%d)"
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

log "render asof=$ASOF rendered_at_pt=$RENDERED_AT_PT output=$OUTPUT"
"$UV" run rainier dashboard render-etf-html \
    --asof "$ASOF" \
    --rendered-at-pt "$RENDERED_AT_PT" \
    --output "$OUTPUT"

# Empty-render guard: the renderer produces a valid-but-empty HTML page when
# no features exist for ASOF (market holiday, stale OHLCV cache, etc).
# That page contains the literal "Universe: 0" — match against it so we don't
# overwrite the last good dashboard with an empty one.
#
# The renderer's comment in render_etf.py explicitly delegates this guard to
# the publish step:
#   "Best-effort: render empty page so the cron output is still a valid file
#    (the publish step can detect emptiness via byte size)."
if grep -q 'Universe: 0 ' "$OUTPUT"; then
    log "ERROR rendered ETF dashboard is empty (no features for asof=$ASOF) — refusing to publish"
    exit 1
fi

log "publish slug=etf-ranks"
"$SCRIPT_DIR/publish-dashboard.sh" etf-ranks
log "done"
