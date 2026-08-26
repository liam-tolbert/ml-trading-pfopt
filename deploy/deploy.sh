#!/usr/bin/env bash
# Pull-based CD for the SEPA cockpit. Fired off-hours by cockpit-deploy.timer;
# safe to run by hand. "Deployed" is defined by the cockpit.sha label on the
# cockpit:live image — the git checkout may sit AHEAD of the deployed image
# after a blocked deploy, which is harmless: nothing executes from the
# checkout; app and trigger only ever run cockpit:live.
set -euo pipefail

# Self-locating: never edit this on the box — a locally modified tracked file
# makes every future deploy halt at the dirty-checkout gate.
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# One lock for BOTH callers — the timer and a hand-run must contend, or a manual deploy
# can land on top of a scheduled one mid-promotion. Opening it is the first thing that
# can fail for a reason the raw bash error ("Permission denied") does not explain, so
# say what is actually wrong: the usual cause is a lock left behind root-owned by a
# `sudo ./deploy/deploy.sh`, after which every later run as the repo owner is locked out.
LOCK=/tmp/cockpit-deploy.lock
if ! exec 9>"$LOCK"; then
    echo "DEPLOY HALT: cannot open $LOCK for writing (running as $(id -un))." >&2
    ls -l "$LOCK" >&2 2>/dev/null || true
    echo "  If it is owned by another user: sudo rm -f $LOCK" >&2
    echo "  Then re-run as the repo owner — never with sudo: a root deploy leaves" >&2
    echo "  root-owned files in data/, which the container (uid 1000) cannot write." >&2
    exit 1
fi
flock -n 9 || { echo "deploy already running"; exit 0; }

# -uno: stray untracked files must not block deploys; a real incoming-file
# conflict still fails the ff-only merge loudly below.
[ -z "$(git status --porcelain -uno)" ] || { echo "DEPLOY HALT: dirty checkout" >&2; exit 1; }

git fetch origin
REMOTE=$(git rev-parse origin/main)
DEPLOYED=$(docker image inspect cockpit:live \
             --format '{{index .Config.Labels "cockpit.sha"}}' 2>/dev/null || echo none)
[ "$REMOTE" = "$DEPLOYED" ] && { echo "up to date: cockpit:live is already $DEPLOYED"; exit 0; }

git checkout -q main
git merge --ff-only origin/main   # divergence/force-push halts loudly here

SHA=$(git rev-parse HEAD)
SHORT=${SHA:0:12}

docker build --label "cockpit.sha=$SHA" -t "cockpit:sha-$SHORT" .

# Test gate: fresh container, NO volumes (live state physically unmountable by
# the suite) and NO network (proves the suite is truly offline). Never promote
# an image whose tests are red. Both suites gate — test_hunt.py pins the weekend
# hunt's rule boundaries (buy zone, RS floor, earnings block), which are exactly
# the numbers that got mis-remembered when that review was done by hand. A suite
# added here must also be un-ignored in .dockerignore or it is absent from the
# image and the run fails as "can't open file".
for suite in tests/test_cockpit.py tests/test_hunt.py; do
    if ! docker run --rm --network none "cockpit:sha-$SHORT" python "$suite"; then
        echo "DEPLOY BLOCKED: $suite failed at $SHA — still serving $DEPLOYED" >&2
        exit 1
    fi
done

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

# Steady state is exactly TWO tags: cockpit:live and cockpit:prev. The sha- tag is only
# scaffolding between build and promote — the commit is recorded in the image's
# cockpit.sha LABEL (what DEPLOYED below reads), so the tag carries no information once
# live points at that image. Dropping it here removes a TAG, never the image.
docker rmi "cockpit:sha-$SHORT" >/dev/null 2>&1 || true

# Any other sha- tag is a leftover from an older deploy (or from one that rolled back
# before reaching this point). NOTHING in this cleanup may be fatal: a tag is
# undeletable while any stopped container still references its image — one stray
# `docker run` without --rm pins an image forever — and on 2026-08-25 exactly that
# aborted a deploy AFTER it had promoted, so `DEPLOYED` and the units-changed reminder
# below never printed and systemd logged a good deploy as failed. Report and continue.
KEEP=$(docker image inspect cockpit:live cockpit:prev --format '{{.Id}}' 2>/dev/null | sort -u)
for tag in $(docker images cockpit --format '{{.Tag}}' | grep '^sha-' || true); do
    id=$(docker image inspect "cockpit:$tag" --format '{{.Id}}' 2>/dev/null) || continue
    echo "$KEEP" | grep -q "$id" && continue
    docker rmi "cockpit:$tag" >/dev/null 2>&1 || {
        holder=$(docker ps -aq --filter "ancestor=cockpit:$tag" | tr '\n' ' ')
        echo "NOTE: kept cockpit:$tag — still referenced by container(s): ${holder:-unknown}"
        echo "      remove it with: docker rm ${holder:-<id>}"
    }
done
docker image prune -f >/dev/null 2>&1 || true
# --reserved-space is the current spelling; --keep-storage is its deprecated alias and is
# accepted for now. Try the new one first so this keeps capping the cache when the alias
# is finally dropped, instead of silently going uncapped behind the `|| true`.
docker builder prune -f --reserved-space=2GB >/dev/null 2>&1 \
    || docker builder prune -f --keep-storage=2GB >/dev/null 2>&1 || true

# systemd units can't be applied from here (needs root; deliberately not passwordless
# sudo for repo-sourced code) — detect a change and tell the human exactly what to run.
if [ "$DEPLOYED" != "none" ] \
        && ! git diff --quiet "$DEPLOYED..$SHA" -- deploy/units deploy/install-units.sh \
        2>/dev/null; then
    echo "NOTE: systemd units changed in this deploy — apply them with:"
    echo "      sudo $REPO/deploy/install-units.sh"
fi

echo "DEPLOYED $SHA (previous: $DEPLOYED)"
