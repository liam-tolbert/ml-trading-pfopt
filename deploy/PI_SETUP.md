# SEPA Cockpit on the Raspberry Pi — one-time setup

The Pi becomes the cockpit's **only** home: Streamlit app + half-hourly data
refreshes + all state live here; the laptop (or phone) is just a browser on
`http://<pi>:8501`. Never run the app or the refresh on two machines — two
diverging watchlists/caches is a lost-update race across hosts.

Architecture at a glance:

- One Docker image (`cockpit`), built **on the Pi** by `deploy/deploy.sh`.
- The app runs as a compose service (`restart: unless-stopped` — survives
  reboots with no systemd unit of its own).
- The data refresh runs as a fresh one-shot container every 30 min,
  09:30–16:30 ET weekdays, fired by `cockpit-refresh.timer`. Each run tops up
  daily bars for the whole universe and then evaluates the watchlist triggers.
  It does **not** screen: the scan table is rebuilt only by the app's explicit
  Re-scan button, so screening never runs on a schedule.
- An EOD systemd timer (17:30 ET weekdays + Sat 10:00 ET) runs `deploy.sh`:
  `git pull --ff-only` → build image → run the full offline test suite inside
  the new image (no volumes, no network) → promote to `cockpit:live` only on
  green, with automatic rollback if the new app fails its health check.
- All mutable state is the single bind mount `./data` (mostly `data/cockpit/`).
  Deploys never touch it.

## 0. Prerequisites

- Raspberry Pi 4 (4 GB), ethernet, ~32 GB storage. Steady-state footprint is
  roughly 3–4 GB of Docker images/cache plus ~120 MB of state.
- **64-bit OS is a hard requirement** (aarch64 wheels for pyarrow/numpy/pandas):

  ```
  uname -m        # MUST print aarch64
  ```

  If it prints `armv7l`, the OS is 32-bit and must be reflashed with 64-bit
  Raspberry Pi OS (Lite is fine) before anything else. On the Pi-hole box that
  means DNS downtime and re-installing Pi-hole/unbound — decide deliberately.
- Pi-hole coexistence: the cockpit uses port 8501 (no clash with DNS 53 or the
  Pi-hole web UI). Make sure the blocklists don't cover
  `query1.finance.yahoo.com` / `query2.finance.yahoo.com`,
  `paper-api.alpaca.markets`, `data.sec.gov`, `www.sec.gov`, or
  `www.nasdaqtrader.com`.
- **Container DNS on a Pi-hole host:** the host's `/etc/resolv.conf` likely
  points at `127.0.0.1`; Docker strips loopback nameservers from containers
  and silently falls back to 8.8.8.8 — the cockpit's traffic would bypass
  Pi-hole, and would break entirely if the router blocks outside DNS. Fix:
  uncomment the `dns:` lines in `docker-compose.yml` with the Pi's **LAN IP**
  (e.g. `192.168.1.2`, never `127.0.0.1`). Verify after first start:
  `docker compose run --rm oneshot python -c "import socket; print(socket.gethostbyname('query1.finance.yahoo.com'))"`
  should succeed, and the query should appear in the Pi-hole query log.

## 1. Install Docker

```
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER     # then log out/in
sudo systemctl enable --now docker
docker compose version            # any version is fine as long as this exact command works
                                  # (the compose CLI plugin, v2+; NOT the legacy python docker-compose v1)
```

## 2. Clone the repo

As your normal login user, **without sudo** — a sudo clone makes the tree
root-owned and every later step (scp, deploys, container writes) fails with
permission denied:

```
git clone https://github.com/liam-tolbert/ml-trading-pfopt.git ~/ml-trading-pfopt
```

`deploy.sh` locates the repo from its own path, and the shipped units carry a
literal `pi` user and `/home/pi/ml-trading-pfopt` path that step 7 rewrites to
the invoking user and the checkout's real location — so the clone can live
anywhere, under any directory name, not just directly in `$HOME`. Never edit
tracked files inside the checkout itself: a modified tracked file trips the
deploy's dirty-checkout halt forever after.

## 3. Secrets

Copy the laptop's `.env` into the repo root on the Pi (from Git Bash on the
laptop, in the repo root):

```
scp .env <user>@<pi>:ml-trading-pfopt/.env
ssh <user>@<pi> chmod 600 ml-trading-pfopt/.env
```

(Remote paths without a leading `/` are relative to your home on the Pi.)

Only the Alpaca paper keys are used (`ALPACA_API_KEY_MINERVINI` /
`ALPACA_API_KEY_SECRET_MINERVINI`, plus the shared fallbacks) — but these are
credentials on the DNS box; keep the file 600 and LAN-only.

## 4. Seed the state (strongly recommended)

Skipping this works but the first scan re-downloads ~2 years of history for
~4,200 tickers from the Pi's IP. From the laptop repo root:

```
scp -r data/cockpit <user>@<pi>:ml-trading-pfopt/data/
scp data/tickers.txt <user>@<pi>:ml-trading-pfopt/data/   # optional fallback universe
```

~120 MB. Afterwards the app restart is instant (`last_scan.pkl`) and scans are
incremental top-ups.

## 5. First deploy (by hand)

```
cd ~/ml-trading-pfopt
./deploy/deploy.sh
```

Expect: image build (~5–10 min, pip downloads only — nothing compiles), then
the 125-test offline suite inside the image (several minutes on a Pi 4), then
`DEPLOYED <sha>`. Browse `http://<pi>:8501` — the scan table should render
from the seeded cache and the SEPA Guide page should show content.

## 6. Refresh smoke test

`docker compose run` treats extra args as a **replacement** for the service
command, so the manual invocation needs the full command line:

```
docker compose run --rm oneshot python src/stock_screener/cockpit/refresh_job.py --no-write
```

The report should print; `--no-write` guarantees nothing is persisted.

## 7. Install the timers

One command installs and arms everything — it copies every `deploy/units/`
unit, rewrites the shipped `pi` user/home for **your** user in the installed
copies (the repo copies stay untouched — editing those would dirty the
checkout and halt deploys), removes any cockpit unit the repo no longer ships,
reloads systemd, and (re)enables + restarts every timer:

```
sudo ./deploy/install-units.sh
```

Re-run it any time the units change — it's idempotent, and `deploy.sh` prints
a reminder with this exact command whenever a deploy touched the unit files.

Sanity-check the schedules (they are written in ET and stay correct across
DST):

```
systemd-analyze calendar 'Mon..Fri *-*-* 09:30:00 America/New_York'
systemd-analyze calendar 'Mon..Fri *-*-* 10..16:00,30:00 America/New_York'
systemd-analyze calendar 'Mon..Fri *-*-* 17:30:00 America/New_York'
```

Refresh fires: 09:30, then every 30 min through 16:30 ET (the 16:30 run is the
settled-close report the daily 16:35 ritual reads). Deploy fires: 17:30 ET
weekdays + Sat 10:00 ET — always outside market hours, so a mid-day push never
changes trading behavior mid-session.

### Auto-sell timers (optional — P1-P4 sell automation)

Two more units automate the sell doctrine: `cockpit-sellplan.timer` (16:40 ET
weekdays) evaluates the sell pillars on every holding and writes a *plan*;
`cockpit-sellexec.timer` (09:25 ET weekdays) submits any still-planned full
exits so they fill at the open. Between the two, the Positions page shows the
plan with per-order **Veto** buttons — your overnight veto window.

`install-units.sh` above installs and enables these two timers along with the
rest — that's safe, because the morning executor is **disarmed by default**:
it does nothing until you add
`AUTOSELL=1` to `.env`. Run order of a first test: let one evening plan
generate, check it on the Positions page, then
`docker compose run --rm oneshot python src/stock_screener/cockpit/sell_job.py execute --dry-run`
before arming. What it will and won't do: only name-specific hard fails
(P1 breakout broken / P2 template broken two closes running / P4 earnings
without cushion) sell, always the full position; P3 (market regime) and all
warns are report-only; vetoed orders are never submitted; a stale plan
(evaluation older than the last session) is refused.

### Armed entries (buy-side automation)

The buy-side mirror: at the evening ritual you build a **limit** trade plan in
the app and click **"Arm for open"** instead of submitting; `cockpit-buyexec.timer`
(09:26 ET weekdays, installed by `install-units.sh` like the rest) then submits
**at most one** still-armed row — limit at the buy-zone top with the GTC OTO
stop — after re-checking the progressive-exposure gate fresh (unknown state =
no buys; the unattended path fails closed). Disarm any row in the app
overnight; un-executed rows expire with the plan.

Also **disarmed by default**: nothing submits until `AUTOBUY=1` is in `.env`.
First test:

```
docker compose run --rm oneshot python src/stock_screener/cockpit/entry_job.py execute --dry-run
```

## 8. CUTOVER (do this only after steps 5–7 succeed)

On the **laptop**:

1. Disable the Windows Task Scheduler job **"SEPA Intraday Trigger"**.
2. Never run `streamlit run ...` or `refresh_job.py` on the laptop again.

The Pi is now the cockpit's only home; the laptop is a browser.

## 9. Drills (worth doing once)

- **Test-gate rollback:** push a commit to `main` with a deliberately failing
  test, run `./deploy/deploy.sh` — it must print `DEPLOY BLOCKED`, exit
  non-zero, and keep serving the old code. Revert, re-run, goes green.
- **Health rollback:** push a commit that breaks app startup — the deploy must
  print `DEPLOY ROLLBACK` and restore the previous image.
- **Reboot:** `sudo reboot` — the app container returns on its own
  (restart policy); `systemctl list-timers` shows both timers re-armed.

## 10. Operations

- Logs: `journalctl -u cockpit-refresh -u cockpit-deploy` (refresh stdout goes
  to journald; the dated JSON reports in `data/cockpit/triggers/` are
  unchanged and feed the app sidebar).
- What's deployed:
  `docker image inspect cockpit:live --format '{{index .Config.Labels "cockpit.sha"}}'`
  (the git checkout may legitimately sit ahead after a blocked deploy).
- Manual rollback: `docker tag cockpit:prev cockpit:live && docker compose up -d app`.
- Backup: rsync `data/cockpit/` off-box weekly — at minimum `watchlist.json`.
- SD wear: `data/cockpit/prices/` is the hot write path (~94 MB rewritten per
  full scan). Fine on a decent card at daily cadence; if a USB SSD is handy,
  move `data/` there and change the one `volumes:` line in
  `docker-compose.yml` (e.g. `/mnt/ssd/data:/app/data`).
- Memory: 4 GB is enough including full-US scans. Optional headroom:
  `sudo apt install zram-tools`.
