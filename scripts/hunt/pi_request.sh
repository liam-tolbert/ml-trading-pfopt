#!/usr/bin/env bash
# The PC's side of a hunt request, over the ssh channel the hunt already uses:
#     bash scripts/hunt/pi_request.sh claim             # prints the request JSON; exit 3 = none
#     bash scripts/hunt/pi_request.sh status FILE       # copies a local status.json to the Pi
# `claim` renames request.json on the Pi before reading it, so two pollers cannot both run
# the same request. HUNT_PI, HUNT_PI_KEY and HUNT_PI_DIR override the host, key and the
# Pi's data/cockpit as in pull_scan.sh.
set -euo pipefail

PI="${HUNT_PI:-lct-raspi@192.168.1.230}"
KEY="${HUNT_PI_KEY:-$HOME/.ssh/claude_key}"
REMOTE="${HUNT_PI_DIR:-Documents/ml-trading-pfopt/data/cockpit}/hunt"
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=8)

case "${1:-}" in
    claim)
        # mv is atomic on one filesystem; it fails when there is no request (exit 3 here).
        ssh "${SSH_OPTS[@]}" "$PI" \
            "cd '$REMOTE' 2>/dev/null && mv request.json request.claimed.json 2>/dev/null \
             && cat request.claimed.json && rm -f request.claimed.json" \
            || exit 3
        ;;
    status)
        FILE="${2:?usage: pi_request.sh status FILE}"
        scp -q "${SSH_OPTS[@]}" "$FILE" "$PI:$REMOTE/status.json.incoming"
        ssh "${SSH_OPTS[@]}" "$PI" "mv '$REMOTE/status.json.incoming' '$REMOTE/status.json'"
        ;;
    *)
        echo "usage: pi_request.sh claim | status FILE" >&2
        exit 2
        ;;
esac
