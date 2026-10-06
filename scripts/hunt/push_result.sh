#!/usr/bin/env bash
# Copy a finished hunt folder to the Pi, where the cockpit's Weekend Hunt page reads it:
#     bash scripts/hunt/push_result.sh 2026-10-10
# data/cockpit/hunt/<date>/ lands as the same path under the Pi's data/cockpit/ (gitignored
# there, so the deploy's dirty-checkout halt never sees it). The folder arrives whole: it is
# copied to <date>.incoming and renamed, so the app never reads a half-pushed hunt. The
# reviewer's working files (review_*.txt, verdicts_batch_*.csv) stay here.
# HUNT_PI, HUNT_PI_KEY and HUNT_PI_DIR override the host, the key and the Pi's data/cockpit.
set -euo pipefail

DATE="${1:?usage: push_result.sh YYYY-MM-DD}"
PI="${HUNT_PI:-lct-raspi@192.168.1.230}"
KEY="${HUNT_PI_KEY:-$HOME/.ssh/claude_key}"
REMOTE="${HUNT_PI_DIR:-Documents/ml-trading-pfopt/data/cockpit}"
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=8)

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SRC="data/cockpit/hunt/$DATE"
if [ ! -f "$SRC/report.html" ] || [ ! -f "$SRC/verdicts.csv" ]; then
    echo "push_result.sh: $SRC has no finished hunt (report.html + verdicts.csv)" >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$DATE"
cp -r "$SRC"/. "$STAGE/$DATE/"
rm -f "$STAGE/$DATE"/review_*.txt "$STAGE/$DATE"/verdicts_batch_*.csv

ssh "${SSH_OPTS[@]}" "$PI" "mkdir -p '$REMOTE/hunt' && rm -rf '$REMOTE/hunt/$DATE.incoming'"
scp -q -r "${SSH_OPTS[@]}" "$STAGE/$DATE" "$PI:$REMOTE/hunt/$DATE.incoming"
ssh "${SSH_OPTS[@]}" "$PI" "rm -rf '$REMOTE/hunt/$DATE' && mv '$REMOTE/hunt/$DATE.incoming' '$REMOTE/hunt/$DATE'"
echo "push_result.sh: $DATE pushed to $PI:$REMOTE/hunt/$DATE"
