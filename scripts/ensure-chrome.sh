#!/bin/bash
# Ensure Chrome is running with remote debugging for CDP scraping.
# Starts Chrome only if not already listening on port 9222.
# Chrome stays running between scrapes to preserve the login session.

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT=9222
PROFILE="/tmp/chrome-debug-profile"

# Check if Chrome is already listening on the debug port
if curl -s "http://127.0.0.1:$PORT/json/version" > /dev/null 2>&1; then
    echo "Chrome already running on port $PORT"
    exit 0
fi

# Clean up stale state if Chrome crashed
rm -f "$PROFILE/SingletonLock" 2>/dev/null

# Start Chrome in background WITHOUT --headless.
# Headed mode is required to pass Cloudflare bot challenges.
# On macOS the window opens but stays in the background.
"$CHROME" \
    --remote-debugging-port=$PORT \
    --user-data-dir="$PROFILE" \
    --no-first-run \
    >/dev/null 2>&1 &

# Wait for Chrome to be ready (up to 15 seconds)
for i in $(seq 1 15); do
    if curl -s "http://127.0.0.1:$PORT/json/version" > /dev/null 2>&1; then
        echo "Chrome started on port $PORT"
        exit 0
    fi
    sleep 1
done

echo "ERROR: Chrome failed to start on port $PORT"
exit 1
