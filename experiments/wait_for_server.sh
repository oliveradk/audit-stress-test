#!/bin/bash
# Wait for a server to be ready by polling /health endpoint
# Usage: wait_for_server.sh <url> [timeout_seconds]

URL="${1:-http://localhost:8000}"
TIMEOUT="${2:-120}"

for i in $(seq 1 $TIMEOUT); do
    if curl -s "$URL/health" > /dev/null 2>&1; then
        exit 0
    fi
    sleep 1
done
echo "Timeout waiting for server at $URL after ${TIMEOUT}s" >&2
exit 1
