#!/bin/bash
# Shutdown a server and wait for it to die
# Usage: shutdown_server.sh <url> [timeout_seconds]

URL="${1:-http://localhost:8000}"
TIMEOUT="${2:-30}"

curl -s -X POST "$URL/shutdown" > /dev/null 2>&1

for i in $(seq 1 $TIMEOUT); do
    if ! curl -s "$URL/health" > /dev/null 2>&1; then
        exit 0
    fi
    sleep 1
done
echo "Timeout waiting for server at $URL to stop after ${TIMEOUT}s" >&2
exit 1
