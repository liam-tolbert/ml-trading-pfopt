#!/usr/bin/env bash
# The PC's side of a hunt request, over the ssh channel the hunt already uses:
#     bash scripts/hunt/pi_request.sh claim             # prints the request JSON; exit 3 = none
#     bash scripts/hunt/pi_request.sh wait SECONDS      # one login; waits up to SECONDS for one
#     bash scripts/hunt/pi_request.sh status FILE       # copies a local status.json to the Pi
# Both claim forms rename request.json on the Pi before reading it, so two pollers cannot
# run the same request. `wait` keeps one session open and checks every 5 s on the Pi
# itself, so the Pi's journal sees one login per wait, not one per check. HUNT_PI,
# HUNT_PI_KEY and HUNT_PI_DIR override the host, key and the Pi's data/cockpit as in
# pull_scan.sh.
set -euo pipefail

PI="${HUNT_PI:-lct-raspi@192.168.1.230}"
KEY="${HUNT_PI_KEY:-$HOME/.ssh/claude_key}"
REMOTE="${HUNT_PI_DIR:-Documents/ml-trading-pfopt/data/cockpit}/hunt"
# ServerAlive: a session that waits for half an hour MUST notice a Pi that went away.
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30
          -o ServerAliveCountMax=3)
# mv is atomic on one filesystem; it fails when there is no request. A subshell, so the
# cd does not persist across the wait loop's iterations (the path is home-relative).
CLAIM="( cd '$REMOTE' 2>/dev/null && mv request.json request.claimed.json 2>/dev/null \
       && cat request.claimed.json && rm -f request.claimed.json )"

case "${1:-}" in
    claim)
        ssh "${SSH_OPTS[@]}" "$PI" "$CLAIM" || exit 3
        ;;
    wait)
        SECS="${2:?usage: pi_request.sh wait SECONDS}"
        ssh "${SSH_OPTS[@]}" "$PI" \
            "end=\$((SECONDS + $SECS)); while [ \$SECONDS -lt \$end ]; do \
               if $CLAIM; then exit 0; fi; sleep 5; done; exit 3" \
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
