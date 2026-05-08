#!/bin/bash
# Ensure Chrome is running with remote debugging for CDP scraping.
#
# Restarts Chrome if:
#   - it is not currently listening on the debug port, OR
#   - the running process's Chrome version differs from the on-disk binary
#     (this happens when Chrome auto-updates in place while the process
#     keeps running — the version skew breaks CDP context management
#     on newer Playwright versions), OR
#   - --force is passed (used by the scraper's CDP-connect retry path).
#
# Chrome otherwise stays running between scrapes to preserve the login session.

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT=9222
PROFILE="/tmp/chrome-debug-profile"
FORCE=0

[ "$1" = "--force" ] && FORCE=1

extract_version() {
    echo "$1" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1
}

is_running() {
    curl -s "http://127.0.0.1:$PORT/json/version" > /dev/null 2>&1
}

kill_chrome() {
    pkill -f "remote-debugging-port=$PORT" 2>/dev/null
    for i in $(seq 1 10); do
        sleep 1
        is_running || return 0
    done
    return 1
}

start_chrome() {
    rm -f "$PROFILE/SingletonLock" 2>/dev/null
    # Headed mode is required to pass Cloudflare bot challenges.
    # On macOS the window opens but stays in the background.
    "$CHROME" \
        --remote-debugging-port=$PORT \
        --user-data-dir="$PROFILE" \
        --no-first-run \
        >/dev/null 2>&1 &
    for i in $(seq 1 15); do
        is_running && return 0
        sleep 1
    done
    return 1
}

if is_running; then
    if [ "$FORCE" -eq 1 ]; then
        echo "Force restart requested — stopping Chrome on port $PORT"
        kill_chrome || { echo "ERROR: Chrome failed to stop"; exit 1; }
    else
        RUN_INFO=$(curl -s "http://127.0.0.1:$PORT/json/version" 2>/dev/null)
        RUN_VER=$(extract_version "$(echo "$RUN_INFO" | grep Browser)")
        BIN_VER=$(extract_version "$("$CHROME" --version 2>/dev/null)")
        if [ -n "$RUN_VER" ] && [ -n "$BIN_VER" ] && [ "$RUN_VER" = "$BIN_VER" ]; then
            echo "Chrome already running on port $PORT (version $RUN_VER)"
            exit 0
        fi
        echo "Chrome version skew (running=$RUN_VER, binary=$BIN_VER) — restarting"
        kill_chrome || { echo "ERROR: Chrome failed to stop"; exit 1; }
    fi
fi

if start_chrome; then
    echo "Chrome started on port $PORT"
    exit 0
fi

echo "ERROR: Chrome failed to start on port $PORT"
exit 1
