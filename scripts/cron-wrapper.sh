#!/bin/bash
# Cron job wrapper — structured logging + Discord alert on failure.
# Usage: cron-wrapper.sh <job-name> <log-file> <webhook-url> <command...>
#
# Example crontab entry:
#   45 8 * * 1-5 /path/to/cron-wrapper.sh qu-morning /path/to/log.log https://discord.com/... "cd /proj && uv run rainier scrape qu ..."

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UV="/Users/pinkbear/.local/bin/uv"

JOB_NAME="$1"; shift
LOG_FILE="$1"; shift
WEBHOOK="$1"; shift

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(ts) [START] $JOB_NAME" >> "$LOG_FILE"

# Run command in subshell, capture output to temp file (avoids shell-escaping issues)
TMPFILE=$(mktemp)
( eval "$@" ) > "$TMPFILE" 2>&1
RC=$?

cat "$TMPFILE" >> "$LOG_FILE"

if [ $RC -eq 0 ]; then
    echo "$(ts) [OK] $JOB_NAME" >> "$LOG_FILE"
else
    echo "$(ts) [FAIL exit=$RC] $JOB_NAME" >> "$LOG_FILE"

    # Send Discord alert via project Python (system python3 gets 403 from Discord)
    if [ -n "$WEBHOOK" ]; then
        cd "$PROJECT_DIR" && $UV run python3 - "$JOB_NAME" "$RC" "$WEBHOOK" "$TMPFILE" <<'PYEOF'
import sys, json, httpx
from datetime import datetime, timezone

name, rc, webhook, tmpfile = sys.argv[1:5]
with open(tmpfile) as f:
    output = f.read()[-1500:]

embed = {
    "title": f"\u274c Cron Job Failed: {name}",
    "description": (
        f"**Exit code:** {rc}\n"
        f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M PT')}\n"
        f"```\n{output}\n```"
    ),
    "color": 0xFF1744,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
try:
    r = httpx.post(webhook, json={"embeds": [embed]}, timeout=10)
    r.raise_for_status()
except Exception as e:
    print(f"Discord notify failed: {e}", file=sys.stderr)
PYEOF
    fi
fi

rm -f "$TMPFILE"
exit $RC
