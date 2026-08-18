# SEPA Cockpit on the Raspberry Pi — one-time setup

The Pi becomes the cockpit's **only** home: Streamlit app + half-hourly trigger
checks + all state live here; the laptop (or phone) is just a browser on
`http://<pi>:8501`. Never run the app or trigger on two machines — two
diverging watchlists/caches is a lost-update race across hosts.

Architecture at a glance:

- One Docker image (`cockpit`), built **on the Pi** by `deploy/deploy.sh`.
- The app runs as a compose service (`restart: unless-stopped` — survives
  reboots with no systemd unit of its own).
- The trigger check runs as a fresh one-shot container every 30 min,
  09:30–16:30 ET weekdays, fired by a systemd timer.
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

## 1. Install Docker

```
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker pi        # then log out/in
sudo systemctl enable --now docker
docker compose version            # must be the v2 plugin ("Docker Compose version v2...")
```

## 2. Clone the repo

```
git clone https://github.com/liam-tolbert/ml-trading-pfopt.git /home/pi/ml-trading-pfopt
```

(`deploy.sh` and the systemd units assume this exact path; edit them if you
put it elsewhere.)

## 3. Secrets

Copy the laptop's `.env` into the repo root on the Pi (from Git Bash on the
laptop, in the repo root):

```
scp .env pi@<pi>:/home/pi/ml-trading-pfopt/.env
ssh pi@<pi> chmod 600 /home/pi/ml-trading-pfopt/.env
```

Only the Alpaca paper keys are used (`ALPACA_API_KEY_MINERVINI` /
`ALPACA_API_KEY_SECRET_MINERVINI`, plus the shared fallbacks) — but these are
credentials on the DNS box; keep the file 600 and LAN-only.

## 4. Seed the state (strongly recommended)

Skipping this works but the first scan re-downloads ~2 years of history for
~4,200 tickers from the Pi's IP. From the laptop repo root:

```
scp -r data/cockpit pi@<pi>:/home/pi/ml-trading-pfopt/data/
scp data/tickers.txt pi@<pi>:/home/pi/ml-trading-pfopt/data/   # optional fallback universe
```

~120 MB. Afterwards the app restart is instant (`last_scan.pkl`) and scans are
incremental top-ups.

## 5. First deploy (by hand)

```
cd /home/pi/ml-trading-pfopt
./deploy/deploy.sh
```

Expect: image build (~5–10 min, pip downloads only — nothing compiles), then
the 125-test offline suite inside the image (several minutes on a Pi 4), then
`DEPLOYED <sha>`. Browse `http://<pi>:8501` — the scan table should render
from the seeded cache and the SEPA Guide page should show content.

## 6. Trigger smoke test

`docker compose run` treats extra args as a **replacement** for the service
command, so the manual invocation needs the full command line:

```
docker compose run --rm trigger python src/stock_screener/cockpit/eod_trigger.py --no-write
```

The report should print; `--no-write` guarantees nothing is persisted.

## 7. Install the timers

```
sudo cp deploy/units/* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cockpit-trigger.timer cockpit-deploy.timer
systemctl list-timers 'cockpit-*'
```

Sanity-check the schedules (they are written in ET and stay correct across
DST):

```
systemd-analyze calendar 'Mon..Fri *-*-* 09:30:00 America/New_York'
systemd-analyze calendar 'Mon..Fri *-*-* 10..16:00,30:00 America/New_York'
systemd-analyze calendar 'Mon..Fri *-*-* 17:30:00 America/New_York'
```

Trigger fires: 09:30, then every 30 min through 16:30 ET (the 16:30 run is the
settled-close report the daily 16:35 ritual reads). Deploy fires: 17:30 ET
weekdays + Sat 10:00 ET — always outside market hours, so a mid-day push never
changes trading behavior mid-session.

## 8. CUTOVER (do this only after steps 5–7 succeed)

On the **laptop**:

1. Disable the Windows Task Scheduler job **"SEPA Intraday Trigger"**.
2. Never run `streamlit run ...` or `eod_trigger.py` on the laptop again.

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

- Logs: `journalctl -u cockpit-trigger -u cockpit-deploy` (trigger stdout goes
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
