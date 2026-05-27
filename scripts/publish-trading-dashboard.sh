#!/bin/bash
# publish-trading-dashboard.sh — render the combined trading dashboard
# (breadth + ETF ranks) and publish to fengshen.dev/trading/.
#
# Mirrors scripts/publish-market-breadth.sh + scripts/publish-etf-dashboard.sh
# in shape:
#
#   1. Compute the rendered-at HH:MM HERE (not in crontab) so `%` chars never
#      reach cron's command field.
#   2. Render the HTML via `rainier dashboard render-combined`.
#   3. Hand off to the generic publisher (`publish-dashboard.sh dashboard --root`).
#      The `--root` flag promotes the file to `public/trading/index.html`
#      (i.e. `/trading/`) instead of the default `public/trading/dashboard/index.html`
#      — DESIGN-trading-dashboard-combined-v1.md §4 D1 (operator override
#      2026-05-27 "the url will be /trading").
#
# Cron sequencing (config/cron.yaml):
#   - market-breadth-daily   refreshes data/cache/sp500_breadth_daily.parquet
#                            (+ data/cache/spy_history.parquet) and renders
#                            the standalone /trading/market-breadth/ page.
#   - etf-dashboard-publish  reads data/cache/thematic_features_daily.parquet
#                            and renders the standalone /trading/etf-ranks/.
#   - trading-dashboard      runs LAST and composes both feeds into one HTML
#                            at /trading/.
#
# DESIGN-trading-dashboard-combined-v1.md §5: standalone URLs continue to
# publish — D4 keeps all three URLs without 301.
#
# Usage:
#   scripts/publish-trading-dashboard.sh
#
# Environment overrides (defaults match the operator's machine):
#   DASHBOARD_SOURCE_DIR   rendered HTML lives here ($HOME/projects/rainier/out/dashboards)
#   RAINIER_UV             uv binary to use (auto-detected via `command -v uv`)
#
# Fails fast on any step; cron-wrapper.sh converts non-zero exit into a
# Discord alert (config/cron.yaml: discord_on_failure: true).

set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s [publish-trading-dashboard] %s\n' "$(ts)" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SOURCE_DIR="${DASHBOARD_SOURCE_DIR:-$HOME/projects/rainier/out/dashboards}"
OUTPUT="$SOURCE_DIR/dashboard.html"
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

# Don't pass --asof — the renderer pulls the latest row from the breadth
# parquet itself (same auto-resolution as `market-breadth render-html`).
# Keeps this script agnostic to weekend / holiday timing.
log "render rendered_at_pt=$RENDERED_AT_PT output=$OUTPUT"
"$UV" run rainier dashboard render-combined \
    --rendered-at-pt "$RENDERED_AT_PT" \
    --output "$OUTPUT"

# Empty-render guard — combined page is degenerate when BOTH sub-renders are
# empty. We can detect that by matching either of the per-section "no data"
# markers; if both are present the page is unpublishable. A single empty
# section is still useful (the operator can read the other tab).
if grep -q 'No breadth data available' "$OUTPUT" \
   && grep -q 'Universe: 0 ' "$OUTPUT"; then
    log "ERROR combined dashboard has no usable data in either section — refusing to publish"
    exit 1
fi

log "publish slug=dashboard root=1"
"$SCRIPT_DIR/publish-dashboard.sh" dashboard --root
log "done"
