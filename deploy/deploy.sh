#!/usr/bin/env bash
# Pull-based CD for the SEPA cockpit. Fired off-hours by cockpit-deploy.timer;
# safe to run by hand. "Deployed" is defined by the cockpit.sha label on the
# cockpit:live image — the git checkout may sit AHEAD of the deployed image
# after a blocked deploy, which is harmless: nothing executes from the
# checkout; app and trigger only ever run cockpit:live.
set -euo pipefail

REPO=/home/pi/ml-trading-pfopt
cd "$REPO"

exec 9>/tmp/cockpit-deploy.lock
flock -n 9 || { echo "deploy already running"; exit 0; }

# -uno: stray untracked files must not block deploys; a real incoming-file
# conflict still fails the ff-only merge loudly below.
[ -z "$(git status --porcelain -uno)" ] || { echo "DEPLOY HALT: dirty checkout" >&2; exit 1; }

git fetch origin
REMOTE=$(git rev-parse origin/main)
DEPLOYED=$(docker image inspect cockpit:live \
             --format '{{index .Config.Labels "cockpit.sha"}}' 2>/dev/null || echo none)
[ "$REMOTE" = "$DEPLOYED" ] && exit 0

git checkout -q main
git merge --ff-only origin/main   # divergence/force-push halts loudly here

SHA=$(git rev-parse HEAD)
SHORT=${SHA:0:12}

docker build --label "cockpit.sha=$SHA" -t "cockpit:sha-$SHORT" .

# Test gate: fresh container, NO volumes (live state physically unmountable by
# the suite) and NO network (proves the suite is truly offline). Never promote
# an image whose tests are red.
if ! docker run --rm --network none "cockpit:sha-$SHORT" python tests/test_cockpit.py; then
    echo "DEPLOY BLOCKED: tests failed at $SHA — still serving $DEPLOYED" >&2
    exit 1
fi

docker image inspect cockpit:live >/dev/null 2>&1 && docker tag cockpit:live cockpit:prev
docker tag "cockpit:sha-$SHORT" cockpit:live
docker compose up -d app

# Health-poll the new app; roll back to prev if it never comes up.
ok=
for _ in $(seq 30); do
    curl -fsS http://localhost:8501/_stcore/health >/dev/null 2>&1 && ok=1 && break
    sleep 2
done
if [ -z "$ok" ]; then
    echo "DEPLOY ROLLBACK: health check failed at $SHA — restoring $DEPLOYED" >&2
    docker tag cockpit:prev cockpit:live
    docker compose up -d app
    exit 1
fi

# Prune sha- tags not referenced by live/prev; cap builder cache (32 GB card).
KEEP=$(docker image inspect cockpit:live cockpit:prev --format '{{.Id}}' 2>/dev/null | sort -u)
for tag in $(docker images cockpit --format '{{.Tag}}' | grep '^sha-' || true); do
    id=$(docker image inspect "cockpit:$tag" --format '{{.Id}}')
    echo "$KEEP" | grep -q "$id" || docker rmi "cockpit:$tag" >/dev/null
done
docker image prune -f >/dev/null
docker builder prune -f --keep-storage=2GB >/dev/null 2>&1 || true

echo "DEPLOYED $SHA (previous: $DEPLOYED)"
