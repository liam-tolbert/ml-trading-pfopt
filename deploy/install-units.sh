#!/usr/bin/env bash
# Sync the repo's systemd units into /etc/systemd/system. Run with sudo after any
# pull that changed deploy/units/ (deploy.sh prints a reminder when one does).
# Idempotent: installs every cockpit-* unit, rewrites the shipped pi user/home for
# the invoking user, removes installed cockpit-* units the repo no longer ships,
# reloads, and (re)enables + restarts every repo timer — a changed OnCalendar only
# takes effect on restart. Deliberately NOT run automatically by deploy.sh: that
# would need passwordless sudo for repo-sourced code, a root path onto the DNS box.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run with sudo: sudo $0" >&2; exit 1; }

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/deploy/units"
DST=/etc/systemd/system

# Units ship with user pi / /home/pi; rewrite for whoever owns the deployment.
RUN_USER="${SUDO_USER:-$(stat -c %U "$REPO")}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
[ -n "$RUN_HOME" ] || { echo "cannot resolve home dir for $RUN_USER" >&2; exit 1; }

# Remove installed cockpit-* units the repo no longer ships (timers disabled first).
for f in "$DST"/cockpit-*.service "$DST"/cockpit-*.timer; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    if [ ! -e "$SRC/$base" ]; then
        case "$base" in
            *.timer) systemctl disable --now "$base" >/dev/null 2>&1 || true ;;
        esac
        rm -f "$f"
        echo "removed $base (no longer in repo)"
    fi
done

install -m 644 "$SRC"/cockpit-*.service "$SRC"/cockpit-*.timer "$DST"/
sed -i "s|/home/pi|$RUN_HOME|g; s|^User=pi$|User=$RUN_USER|" "$DST"/cockpit-*.service

systemctl daemon-reload
for t in "$SRC"/cockpit-*.timer; do
    base="$(basename "$t")"
    systemctl enable "$base" >/dev/null
    systemctl restart "$base"
    echo "enabled + restarted $base"
done

echo
systemctl list-timers 'cockpit-*' --no-pager
