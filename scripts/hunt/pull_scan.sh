#!/usr/bin/env bash
# Copy the Pi's last scan and watchlist into data/cockpit/. Reads the Pi, writes nothing there.
#     bash scripts/hunt/pull_scan.sh
# Waits up to HUNT_PI_WAIT seconds (default 120) for the Pi to answer: after a wake from
# sleep the network takes a moment. HUNT_PI, HUNT_PI_KEY and HUNT_PI_DIR override the host,
# the key and the Pi's data/cockpit directory.
set -euo pipefail

PI="${HUNT_PI:-lct-raspi@192.168.1.230}"
KEY="${HUNT_PI_KEY:-$HOME/.ssh/claude_key}"
REMOTE="${HUNT_PI_DIR:-Documents/ml-trading-pfopt/data/cockpit}"
WAIT="${HUNT_PI_WAIT:-120}"
FILES=(last_scan.pkl watchlist.json)
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=8)

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DEST=data/cockpit
mkdir -p "$DEST"

until ssh "${SSH_OPTS[@]}" "$PI" true 2>/dev/null; do
    if [ "$SECONDS" -ge "$WAIT" ]; then
        echo "pull_scan.sh: $PI did not answer within ${WAIT}s" >&2
        exit 1
    fi
    sleep 5
done

# Both files MUST arrive before either replaces the local copy: a scan paired with another
# day's watchlist would audit the wrong names.
for f in "${FILES[@]}"; do
    scp -q "${SSH_OPTS[@]}" "$PI:$REMOTE/$f" "$DEST/$f.incoming"
done
for f in "${FILES[@]}"; do
    if [ -f "$DEST/$f" ]; then
        cp -p "$DEST/$f" "$DEST/$f.bak"
    fi
    mv -f "$DEST/$f.incoming" "$DEST/$f"
done
echo "pull_scan.sh: pulled ${FILES[*]} from $PI"
