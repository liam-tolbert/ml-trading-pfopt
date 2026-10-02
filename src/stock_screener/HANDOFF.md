# HANDOFF — Stock-Screener (Minervini) venture

**Scope:** the *classical* stock-screening track. Separate from `docs/HANDOFF.md` (the parked ML
cross-sectional track). The momentum-*factor* experiment is closed; its essentials are in
`HANDOFF_HISTORY.md` §5 and its standalone write-up was never committed.

**Status (2026-10-01):** live paper trading on a dedicated Raspberry Pi. Weekly `full_us` hunt →
frozen-pivot watchlist → half-hourly refresh + trigger checks → GTC-stopped entries. Sell automation is **armed** on the
Pi (`AUTOSELL=1` since 2026-10-02); armed entries are built but disarmed (`AUTOBUY` unset). A SEPA-fidelity
audit (the local, gitignored `SEPA_AUDIT.md`) drove §6.72–§6.99. §6.100 trimmed what this stage
doesn't use, and the audit is frozen until the journal has ~20 more trades (§12). The offline
suites `tests/test_cockpit.py` and `tests/test_hunt.py` gate every deploy and print their own
counts.

**Research verdict (2026-06-29) — no out-of-sample alpha.** A strong in-sample result (α t=2.49) was
overfit; OOS collapsed it to t=0.47. Risk management is real; selection is not. The user trades this
deliberately for **execution practice**, judged on execution, not P&L.

**Pivot (2026-06-29) — automation → human-in-the-loop "cockpit."** The backtest only exercised SEPA
*Step 1* (the 8-pt Trend Template, as a hard gate); Steps 2–4 are *discretionary* in Minervini's
hands. So the tool does the mechanical filtering and hands the user charts to judge. **The user is
the judge.**

Two halves under `src/stock_screener/`: `minervini_screener/` (vendored third-party *rules*),
`backtest_daily/` (event-driven daily *simulator*). The cockpit (`cockpit/`) is a third, live track.

---

## 1. Research: what was tested and what it showed

Moved to `HANDOFF_HISTORY.md` §1. The verdict is in the header: no out-of-sample alpha; risk
management is real, selection is not.

## 2. Methodology lessons

- **★ The in-sample tuning trap.** A knob chosen after seeing the outcome manufactured a fake
  t=2.49. **Any positive result from a post-hoc knob is suspect until OOS-validated.**
- **The momentum Phase-0 "STOP" measured the wrong object** — an ungated, L/S, large-cap,
  survivorship-biased *factor*. It does not generalize to the real screener.
- **Vendored package eager-loaded the live data layer** (`screening/__init__` → `data.storage` →
  sqlalchemy). Fixed by dropping the import; the dead layer was later deleted outright. A
  vendored tree is not free — unreachable modules still shape the image and every grep.
- **Don't fight the import-sorter.** It enforces `from src.X` with the repo ROOT on `sys.path`.
- **`pytest` isn't installed** — tests run as plain scripts, matching repo style.
- **The engine was unusably slow** because capacity was checked by *position count* while risk-sizing
  filled cash at ~10 names, so the expensive VCP scan re-ran while fully invested.

## 3. Open research questions

Moved to `HANDOFF_HISTORY.md` §3. The one genuinely open question is whether the screen's score
predicts which names delist.

## 4. Rules — never do these

**Research:**
- Don't conflate the momentum Phase-0 STOP with the real screener.
- Don't backtest it as market-neutral L/S or a periodic top-N rebalance — it's long-only,
  position-based, with stops and a cash state.
- Don't run it on top-250 large-caps. Minervini lives in small/mid-cap growth.
- Don't trust long-only breakout numbers on survivorship-biased data — buying breakouts to new highs
  is the *most* survivorship-sensitive signal there is.
- Don't let the vendored package pull the live layer into the harness path.
- Don't edit vendored business logic; keep `LICENSE`/`PROVENANCE.md` accurate.
- Don't break the leak contract — every rule call through `cache.ohlcv_upto(t)`, fundamentals lagged
  to `rdq`, delisting realized only on its date. `test_engine_decisions_leak_free` is the guard.
- Don't re-tune `confirm_days` (or any risk knob) chasing alpha — the binding constraint is the
  signal, and the train optimum is unstable (15 on 2003–13 vs 25 full-history).
- Don't cite the in-sample 11.1% / t=2.49. The validated number is 7.95% / t=0.47.

**Live trading — each learned at real P&L:**
- **Don't batch-enter.** Six positions in four minutes (2026-07-27) = one bet on that day's tape.
  Progressive exposure runs the other way: 1–2 pilots, add only after banked wins.
- **Don't act on an intraday trigger.** PEBK's 15:27 intraday trigger faded to a settled close BELOW
  the pivot the same day; bought at 15:58 → instant failed breakout. The close decides.
- **Don't chase past pivot×1.05.** PKG filled +5.4% above pivot and gave back ~4%.
- **Don't open inside the ~21-day earnings window.** STRW, bought 14 days before its report at double
  size, had to be trimmed then exited.
- **Don't size micro-caps without an ADV look.** STRW's ~$100k order moved the tape 1.5% (~12% of ADV).
  Enforced since §6.81: one order stays within `MAX_ORDER_ADV_PCT` (2%) of the 20-day dollar volume.
- **Don't design around a "known" API constraint that was never verified live.** "Alpaca market OTOs
  can't be GTC" was assumed for a month and was simply false — the disproof was a five-minute
  1-share probe, and the assumption cost an unprotected overnight position.
- **Don't let tests depend on wall-clock/market state.** Any time-coupled feature needs its test
  bypass designed in, not discovered.

## 5. Momentum factor (closed 2026-06-04)

Moved to `HANDOFF_HISTORY.md` §5. Verdict: STOP; the ranking carries no market-neutral information.

---

## 6. The cockpit — architecture

A local **Streamlit** app running the SEPA funnel as decision support. `src/stock_screener/cockpit/`,
live yfinance data (not CRSP). Reuses ONLY the pure rule functions from `minervini_screener/screening/`.

**Funnel:** universe (`full_us`, ~4,120 names — the ONLY universe since §6.32) → **Step 1** hard gate,
full **8/8** trend template → RS rating (IBD-style weighted multi-horizon, §6.17) → **Step 2**
fundamental highlight (F, eight checks, §6.89; Code 33, last-report reaction, estimates, funds) → **Step 3** VCP tier (cockpit detector, §6.1) →
**Step 4** advisory levels (pivot / buy zone / stop / target) + sizing. A regime/breadth banner gates
the discipline.

**Module map:**

| Module | Role |
|---|---|
| `data_feed.py` | yfinance layer: universe, price cache, incremental top-ups, fundamentals (yfinance quarters, estimates, holders; EDGAR facts and 8-K release dates) |
| `scan.py` / `vcp.py` / `indicators.py` | the funnel, the VCP tier detector, RMV/BBWP/squeeze |
| `scan_worker.py` | background scan thread + process-wide result store (`last_scan.pkl`) |
| `triggers.py` | pure trigger evaluation; `export.py` the watchlist store |
| `trade.py` | Alpaca paper submit path, stops, the exposure gate, the journal + Loss Adjustment Exercise |
| `advisories.py` | display-only SEPA reads: stop room, post-breakout violations/follow-through, regime tier, weak-tape advice, market turn, SPY re-entry streak (breadth-aware with history) |
| `breadth_store.py` | `data/cockpit/breadth.csv`: one row per settled session (n_scanned, phase2_pct, new highs, new lows); appended by `screen_job` only |
| `sectors.py` | sector/industry per symbol from yfinance `Ticker.info`, cached in `data/cockpit/sectors.json` (180 days; a miss retried after 7); fetched by the scan for 8/8 passers, read cache-only everywhere else |
| `doctrine.py` | the shared rule numbers and the two promotion switches (imports nothing) |
| `sells.py` / `entries.py` | P1–P4 sell planner; armed-entry plans |
| `refresh_job.py` / `sell_job.py` / `entry_job.py` | the three headless CLIs the timers invoke |
| `runlog.py` | dated run logs, 14-day retention |
| `app.py` + `pages/` | Streamlit surfaces (scan, SEPA Guide, Positions, Journal) |

**Key data semantics:**
- **Frozen pivots.** Watchlist entries carry `judged_pivot` — the detected pivot drifts every scan, so
  a trigger against a recomputed level would move under your feet. 📌 sets the level you judged; the
  refresh job auto-freezes the rest on first sight (`pivot_source="auto"`).
- **Two pivots exist.** The app pivot (`_entry_levels`, 52-wk-high based) is the ONLY user-facing one;
  the detector's `pivot_price` stays internal to tier classification.
- **Price cache** is one parquet per ticker, incremental: fresh / top-up-since-last-bar / full
  re-baseline (cold, gap >10 days, or a >0.5% split divergence over the overlap window).
- **Settled-close serve.** A cache written with no market session since is current *regardless of
  age* — evenings, weekends, pre-open cost zero network. This is why an EOD sweep makes every later
  read free.
- **yfinance races.** Concurrent single-ticker `yf.download` calls corrupt each other's results via
  shared global state. Use the batch download; `_YF_LOCK` serializes in-process. Never reintroduce a
  ThreadPool over per-ticker `yf.download`.
- **Account and order-history reads are per browser session** (§6.71). The Positions page's account
  read (`_session_positions`) and the shared order history (`journal_cache.cached_fills`) live in
  `st.session_state`, keyed by the page's nonce (`pos_nonce` / `jr_nonce`) and reused for at most
  **60 s** (`POS_MAX_AGE_S` / `FILLS_MAX_AGE_S`). Widget reruns reuse the read; Refresh, sell and
  re-arm bump the nonce; age catches "came back after trading on another page", because session
  state outlives page switches. Failures are never memoized, so Refresh always retries. The
  Positions caption shows `as of HH:MM:SS ET`.

## 7. Doctrine — how it is traded

**Cadence: two jobs, two frequencies.**
- **Weekly (weekend) = HUNT.** Re-scan `full_us` for new Tier-A bases; prune names inside the 21d
  earnings window and >5% drifters. Bases form over weeks — daily re-hunting is noise.
- **Daily 16:10 ET = TRIGGER + STOPS (5 min).** Read the trigger report, read the Positions pillars,
  write down tomorrow's orders.

**THE RULES** (from the §6.53 execution audit — the answer to "what am I missing"):
1. **Market hours execute yesterday's decisions — no new decisions intraday.** Covers chasing,
   impulse entries, and late sells at once.
2. **A red/yellow pillar = decision TONIGHT, order at the NEXT OPEN** — never "watch one more day."
3. **Max one new entry per day** — kills batching, forces each buy through the exposure gate.
4. **Weekend = hunt AND prune**, not just add.
5. **Pilot size (0.5% risk)** until the last-10 numbers improve.

**Step 1, the gate.** The book's eight trend-template criteria: the vendored template's seven
price criteria plus an RS rating ≥ `doctrine.RS_FLOOR` (70). The vendored eighth, a Stage-2
slope check, is not one of the book's and no longer counts (§6.80). P2 judges a holding by the
same eight, reading the RS from the last scan; seven passing with no scan is unknown, and
unknown never trades. Two reads beside the gate, advisory: the RS line's 6/13-week trend
and how many months the 200-day has risen.

**Entry.** Buy zone = pivot..pivot×1.05 (no chasing). Trigger = settled close above the frozen pivot
on **≥1.5× the 50-day average volume**. Stop 7.5% below the pivot by default, and **never more than
10% below the price paid** (§6.72): a buy's stop is raised to 10% below its worst-case fill (the
limit, else the price), re-checked at submit and at arming. From the 5th cockpit win the default
becomes the **derived stop**, ½ the average win from the fill, clamped 4–10% (§6.74). Judge the
stop against the name's **typical day**. Under 2 of them is inside noise (§6.75). Target pivot×1.25.
Skip inside ~21 days of earnings. Size for ~1% account risk (0.5% while probing); the
10%-of-equity single-order cap clamps risk-mode quantities, and every buy is clamped to 2% of
the name's 20-day dollar volume (§6.81), re-checked at submit and at arming.

**Sell — the four pillars (P1–P4).** A breakout buy's thesis is P1 resolution (closed above the pivot
on volume and STAYS above) ∧ P2 Stage-2 structure (8/8, rising 50-SMA) ∧ P3 risk-on tape ∧ P4 no
unpriced binary. **Any pillar failing kills the trade; the stop is the disaster floor for what happens
between checks, never the sell signal.** Decisions on settled closes only; execution at the next open.
- Day 0: entered intraday, closed back below the pivot → the breakout never happened, sell next open.
- Decisive close below the pivot → sell. A close below the breakout bar's low sells only when it is
  also back below the pivot; while the pivot holds it warns (§6.95).
- Laggard clock: no +3% cushion by ~day 10 → sell into strength; flat-to-red at day 15–20 → exit.
- Earnings: a LOSS is never carried into a ≤21d report; a small gain is trimmed to hold-through size.
- Automation acts on hard ❌ of P1/P2/P4 only. P3 and all warns are report-only. P2 needs **two
  consecutive** failing closes (the strict template has one-day SMA noise flips).
- **Post-breakout violations** warn in P1 (§6.77): a close under the 20-day line in the first
  month, a heavy down day after a light-volume breakout, lower lows, more down or lower-half
  closes, a gain given back. `VIOLATIONS_CAN_FAIL` (off) would make 3 of them a P1 fail.
- **The tape** (§6.78): SPY entering Stage 4 → the plan notes "reduce"; `MARKET_TURN_CAN_TRADE`
  (off) would also order half of each position sold. After a break, no new buys until SPY has
  been back in Stage 1–2 for 15 sessions. In a weak tape the book's 5–6% stops / 10–12%
  profits are shown beside the plan's, never applied.
- **Selling into strength** (§6.98): from +20%, the books' climax signs show on the Positions page
  and in the plan notes. `STRENGTH_CAN_TRADE` (off) would make 3 signs on one close a planned sale
  of half, the rest stopped at breakeven, once per position.

**Progressive exposure.** Gate is open when flat, or when every position in the newest-day cohort is
at breakeven-or-better AND tagged net open P&L ≥ 0. Scope is cockpit-**tagged** positions only, so
manual legacy holdings can never poison "flat". Two consecutive tagged losses → half-size advisory.
**Fail direction is asymmetric by design:** unknown → manual path OPEN (the human judges), unattended
executor CLOSED.

**Judging a base.** RMV is the tightness discriminator (<25 tight, >45 loose) — tightening legs and
`fund_score` can flatter a loose base. Read raw rev/EPS YoY, not `fund_score` (it counts n/a as a
fail). Never hard-gate on fundamentals: patchy yfinance data would drop *thin-data* names, not weak
ones — the same never-miss failure the VCP gate had. Shrink the list with **Tier A + RS** instead.
**F (`fund_score`) is 0–8** (§6.89): revenue ≥ 20%, EPS ≥ 20%, EPS accelerating, margin expanding,
Code 33, annual EPS up, the current-year estimate raised ≥ 5% over 90 days, and the last report
held (no ≥ 5% drop on ≥ 1.5× volume). A foreign filer can't pass Code 33, annual EPS or the
reaction (no EDGAR data), so its F tops out near 5. The slider and `--min-fund` default to 0.

**Live scorecard (13 closed, 2026-10-01):** 1W/12L, expectancy −1.9%, −$14.7k. Loss control
WORKED (avg loss −2.4%, worst −6%, avg win +4.0%); what failed was follow-through. Every exit was
a manual sale; none was a stop-out. As of §6.53 (10 closed), every dollar lost traced to
*entry*-rule violations.

## 8. Deployment and operations

**The Pi is the cockpit's only home.** Never run the app or the refresh on two machines — two
diverging watchlists is a lost-update race across hosts.

- **Access:** `192.168.1.230`, user `lct-raspi`, repo at **`~/Documents/ml-trading-pfopt`** (NOT
  `~/ml-trading-pfopt`). **No passwordless sudo**, deliberately — anything needing root is run by hand.
- **Hardware:** Pi 4 Model B with **1.8 GB usable RAM** (a 2 GB board) booting from a **57 GB USB
  disk**, not an SD card. **Memory is the binding constraint**, with 1.8 GB zram swap active. An
  image build competes with the running app — expect 15+ minutes, and don't hand-start one while a
  universe sweep is in flight.
- **App** runs as a compose service (`restart: unless-stopped`). Note that policy does NOT revive a
  container you stopped by hand, not even across a reboot.
- **`oneshot`** is the generic short-lived compose service behind every scheduled CLI job.

**Timers** (all ET, installed by `sudo deploy/install-units.sh`):

| Unit | When | Runs |
|---|---|---|
| `cockpit-refresh` | 09:30, :00/:30 to 15:30, 16:10 | watchlist + held-name price top-up, then trigger check |
| `cockpit-sellplan` | 16:15 weekdays | evening sell plan (overnight veto window) |
| `cockpit-eod` | 16:20 weekdays | **two sequential steps in one unit**: full-universe price top-up (arms the settled-close serve), then the universe screen that rebuilds `last_scan.pkl` and appends the session's row to `data/cockpit/breadth.csv` |
| `cockpit-sellexec` | 09:25 weekdays | submit still-planned sells for the open |
| `cockpit-buyexec` | 09:26 weekdays | submit at most ONE armed entry |
| `cockpit-deploy` | hourly, 17:00–09:00 daily | `deploy.sh` |

`Persistent=false` on everything but deploy: a missed buy/sell must never replay late against a stale
plan. Deploy is `Persistent=true` (catch-up is harmless off-hours).

**Deploy pipeline (`deploy.sh`).** Pull-based, no inbound anything. `flock` → dirty-checkout halt →
fetch → compare `origin/main` to the **`cockpit.sha` label on `cockpit:live`** → ff-only merge →
build → **both test suites** in a fresh container (`--network none`, no volumes) → tag `prev`,
promote `live` → health-poll 60s → roll back on failure → prune to exactly two tags → print the
`install-units.sh` reminder if units changed.

- **"Deployed" is the image label, never the checkout.** The checkout may sit ahead; nothing executes
  from it. **Diagnose from the label**, or you will misread a three-day outage as healthy:
  `docker image inspect cockpit:live --format '{{index .Config.Labels "cockpit.sha"}}'`
- **Units only change when you run `install-units.sh`.** Committing and deploying is not enough.
- **Never run `deploy.sh` with sudo.** It must run as the repo owner.

**Logs.** `journalctl -u 'cockpit-*'` is where every scheduled run goes. `data/cockpit/logs/
cockpit_<date>.log` holds dated run logs (14-day retention) and **survives deploys**, where
`docker logs` does not — the container is destroyed and recreated on every promotion.

**Health check one-liner:** `journalctl -u 'cockpit-*' --since -7d | grep -c 'Failed with result'`

**Friday hunt task — on the Windows box, never the Pi (§6.101).** The scheduled task "SEPA Weekend
Hunt" (`scripts/hunt/register_task.ps1`; Fridays 18:00 local, wakes the PC from sleep) runs
`scripts/hunt/weekend_hunt.ps1`. It copies the Pi's `last_scan.pkl` and `watchlist.json` down
(read-only on the Pi), runs the `/weekend-hunt` skill headless (`claude -p`), and checks the result
itself.

- **Output:** `docs/hunt/<date>/report.html` with its `charts/`, the deliverable: the reviewer's
  `narrative.md` under "Reviewer's read", the gated tables, every chart sheet under its verdicts.
  Working state stays in `data/cockpit/hunt/<date>/` (`summary.md`, verdicts, `FAILED.txt`
  naming the reason when a run did not finish). Log: `data/cockpit/hunt/logs/<date>.log`.
- **The allowlist is the safety boundary** (`scripts/hunt/unattended_settings.json`): the hunt
  CLI, file reads, and writes inside `data/cockpit/hunt/`. No ssh, no cockpit or trade code.
- **Needs:** the standalone `claude` CLI on PATH and `CLAUDE_CODE_OAUTH_TOKEN` in the user's
  environment (`claude setup-token`). The desktop app's login is not usable from a task.
- **Wake:** from sleep or hibernate only, with the user logged in. After a shutdown or logout the
  missed run starts at the next logon. The PC sleeps again only if the run woke it and nobody has
  touched it since.
- **By hand:** `scripts/hunt/weekend_hunt.ps1 -NoSleep`. `register_task.ps1 -WakeTestInMinutes 5`
  proves the wake without spending a review; `-Unregister` removes the task.
- **Artifact:** the task cannot publish one (no Artifact tool outside a session). Ask a session
  to publish a run; the skill's step 10 has the call (`report.html` plus its sheets). Printing
  an artifact from the claude.ai viewer cuts it off; print `report.html` from a browser instead.

**App hung?** (a page spins forever and Refresh does nothing). Separate "the account is unreachable"
from "the app process is wedged" by running the page's read in a **fresh** process inside the same
container. `-i` is required or the heredoc never reaches Python and the call silently does nothing:

    docker exec -i -w /app cockpit-app python - <<'PY'
    from src.stock_screener.cockpit import trade; print(trade.fetch_positions()["account"])
    PY

If that answers in seconds while the page still spins, the long-running app is stuck:
`docker compose restart app` (no sudo; safe in market hours — every timer job runs in its own
`oneshot` container, and the watchlist, scan and plans are on disk). Since §6.71 every Alpaca call
times out, and since §6.79 a log line can no longer lock out other threads, so a true hang should
not happen. If one does, dump the app's threads before restarting (`py-spy dump --pid 1` inside
the container, if installed). A thread parked in `logging/__init__.py ... acquire` is a
handler lock. An idle ESTABLISHED socket to `paper-api` with nothing in flight is NOT evidence of a
stuck Alpaca call: a finished request leaves exactly that.

## 9. Conventions

- **Source comments say WHY the code must be this way, never which bug/review/date produced it**
  (§6.47). The incident ledger lives in test docstrings and `HANDOFF_HISTORY.md`. `CLAUDE.md`
  at the repo root holds the full comment and commit rules.
- **`breadth.csv` is appended only by the scheduled screen.** An in-app scan mid-session would
  write a provisional row; `screen_job` runs after the settle. A same-evening rerun replaces
  the day's row.
- **`RS_FLOOR` is one constant.** The scan gate, the app's slider default, P2 and the hunt read
  `doctrine.RS_FLOOR`. The gate's RS leg lives inside `screen_universe`, so `filter_candidates`
  needs no mirror for it; its own `min_rs` is the slider's extra filter above the floor.
- **Tests run as plain scripts**, no pytest. `python tests/test_cockpit.py` is the gate's entry point;
  the suites live in `tests/cockpit/test_<category>.py` and each runs standalone.
- **Exit-code contract: 0 for anything normal — including "nothing to do" and "disabled" — and 1 only
  for a real failure.** This is what makes a disarmed executor show green in systemd.
- **A suite added to the gate must reach the image.** `.dockerignore` is a whitelist; `tests/cockpit/**`
  is a glob so new category files need no second edit.
- **Tests must not depend on wall-clock or market state.** Time-coupled features need an explicit
  bypass (e.g. the negative-`max_age_days` sentinel).
- **Never `st.cache_data` per-account or per-visitor state.** It is ONE cache per server process,
  and a key built from a per-session counter that starts at 1 is shared by every visitor. They get
  the first visitor's snapshot with no expiry and queue behind its in-flight fetch (§6.71). Memoize
  in `st.session_state` with a nonce plus a max age instead. `st.cache_data` stays right only for
  data that is genuinely the same for everyone.
- **Every outbound network call carries a timeout**, and a timed-out *read* raises
  (`TradeUnavailable`). It never degrades to an empty result that looks like a real answer: an empty
  stops list reads as "unprotected" (§6.71).
- **Never reuse a stdlib hook's name for something else.** `DatedFileHandler.release()` meant "close
  the file" to its author and "release the lock" to `logging`, which called it after every record
  (§6.79). Before naming a method on a subclass, check the base class doesn't already call it.
- **The user commits all code themselves.** Claude leaves the tree dirty for review — their push is
  the human gate in front of the Pi's auto-deploy.
- **Don't edit tracked files inside the Pi's checkout** — a modified tracked file trips the
  dirty-checkout halt forever after. Stage work in `/tmp` instead.
- **AppTest gotchas:** `at.session_state` has no `.get()`/`.setdefault()`; widget refs go stale after
  each `.run()`; there is no `at.download_button`/`at.file_uploader` accessor — test pure helpers
  directly instead, which is why sizing/parsing live in `export.py`/`trade.py`, not inline in `app.py`.
  Two `AppTest` instances are two *sessions* that share one process's `st.cache_data`, which makes
  them a faithful model of two browsers on the Pi. That is how the §6.71 tests prove isolation.

## 10. Reference — hard-won specifics

Constants and API facts that are expensive to rediscover. Change these only with the benchmark green.

**VCP detector (`vcp.py`), calibrated against the 200-chart benchmark:**
- **Multi-threshold:** up to 4 ZigZag thresholds — long-history, recent-window (~2 mo), 0.7× recent
  (floor 2%), and a fixed 3.5%. A VCP by definition ends *quieter* than its history, so one
  history-calibrated threshold goes blind at the tight ending; the recent window gets polluted by the
  breakout burst itself (WERN read 9.6% while its coil legs were 5–7%). Best read wins.
- **RMV veto is conditional** — only while price is *below* the pivot (a breakout IS a volatility
  burst; SMBC read RMV 100 mid-breakout). Cutoff **30**. Removing the below-pivot veto was measured
  and rejected: +6 real / +14 junk.
- **Sanity rules:** leg ≥ **2** bars (quiet climbers have genuine 2-day final shakeouts; 1-bar gap
  legs are junk anchors) · base ≥ **2.0** weeks (length is measured over the *selected* legs, which
  under-reads — VRA's ~6-week base measures 2.1) · newest leg ≤ **13** weeks · **dead tape** = median
  daily true-range% over the last 42 bars < **1%** (median, not max — one pop-day defeats a max rule;
  all 7 deal-zombies ≤0.95%, quietest real setup 1.64%). Dead-tape runs in adaptive mode only, so
  pinned `thr=` keeps synthetic H=L=C tests deterministic.
- **Tightness:** final leg ≤12% AND (≤0.8× first leg OR ≤6.5% absolute — uniform quiet shelves can't
  shrink 20% further but ARE tight).
- **Tiers:** **A** = valid base within −10%..+10% of the *detected* pivot · **B** = forming or
  extended, never hidden · **C** = safe exclusions only (dead tape / no pullbacks / stale), reason
  recorded. **Benchmark contract (`test_vcp_benchmark_200_charts`)** — what is ASSERTED: every YES
  lands in A or B (**C contains zero YES**) and A-recall ≥ 45. The split it PRINTS for reference is
  A=79 (53 YES, precision 67%), B=114 (19 YES), C=7; those numbers drift and are not enforced.
  **C containing zero YES is the never-miss contract** — squeezing A below the true setup count
  reintroduces misses.

**`pct_to_pivot` sign convention:** negative = price ABOVE the pivot (into/past the buy zone);
positive = BELOW it (not yet triggered). Sweet spot ≈ 0 to −5%; deeply negative = chasing.

**Stops and the book's numbers (`doctrine.py`, §6.72–§6.78):**
- `fill_floor(fill)` = `ceil(round(fill × 0.9 × 100, 6)) / 100`: 10% below the fill, rounded UP to
  the cent (the inner `round` stops 42 × 0.9 = 37.800000000000004 ceiling to 37.81). The builder,
  submit and arming all use it, so a builder-made stop always passes the guards.
- Typical day = median true range over the last **42** sessions (the dead-tape window), as % of
  price. Room = (fill − stop) / fill ÷ typical day; under **2.0** = inside noise. The benchmark's
  quietest real setup runs 1.64%/day and its wildest (IMUX) 7.73%.
- Derived stop: tagged closed trades, **≥5 wins**, ½ × average win, clamped [4%, 10%]; the 4% floor
  came from the §6.73 pre-registered rule on ONE winner.
- Post-breakout window: **20** sessions for the 20-day line; violation cluster for the switch = **3**.
- Re-entry lag: **15** sessions of SPY in Stage 1–2 (the OOS-validated value; SPY-only, no breadth).
- **Promotion switches, all OFF:** `VIOLATIONS_CAN_FAIL`, `MARKET_TURN_CAN_TRADE` and
  `STRENGTH_CAN_TRADE`. Read at call time (`doctrine.X`), so tests patch the module attribute.
- **Strength signs (`advisories.strength_signs`)**, read from `STRENGTH_MIN_GAIN` (+20%) on the latest
  settled close since the entry: a 25% rise within 15 sessions; 7 of the last 10 sessions up; the
  biggest up day since the entry; a gap up (low over the prior high) in the last 3 sessions; ≥ 1.5×
  volume with the close within 1% of the prior; the biggest down day since the entry. The two
  "biggest day" signs need 5 sessions since the entry.

**`suggest_stop` bases (Positions page, auto mode):** `trade.book_stop_level`, the books' triggers
(§6.96). With none, the initial stop (8% below entry, or the derived stop). Breakeven once the rising
50-day reaches the cost, at `BREAKEVEN_R` (3R), or at `BREAKEVEN_AVG_WIN_MULT` (2×) the average win;
that last also sets the back stop at `entry × (1 + average win)`. With the 50-day above the cost the
stop trails `sma_50 × 0.99`. The highest level wins. The average-win rules wait for
`DERIVED_STOP_MIN_WINS` tagged wins. Floored at the in-force stop (ratchet-safe). `None` = underwater →
manual row. `position_stage` reads the same function: underwater · initial · breakeven · back stop ·
trailing 50-day.

**Breadth (`scan._new_high_low`, `breadth_store`):**
- New high = the last bar's High ≥ `classify_phase`'s 252-bar `week_52_high` (rounded to 2 dp,
  hence a 0.005 slack); new low the mirror on Low. Counted over every name with 200 rows,
  before the 8/8 gate, so the counts describe the universe, not the candidates.
- `nh_nl_expanding` = today's spread > the spread `SPREAD_LOOKBACK = 10` settled sessions back
  (rows dated before the session). The breadth-aware re-entry lag requires `phase2_pct ≥
  BREADTH_MIN_PHASE2` on every counted session; a session without a row counts on SPY alone and
  sets `partial`. Today's own breadth is added to the map at screen time, since its row is
  appended afterwards.

**Depth vs market (`advisories.depth_vs_market`):**
- Window = the first selected contraction's `peak_date` to the last bar, for the stock and
  SPY alike. Depth = the deepest fall from a running high inside it (High/Low, Close when
  absent). Ratio = stock ÷ SPY; flagged at `DEPTH_VS_MARKET_MAX = 3.0`. None when SPY fell
  under 1% in the window: a ratio to a flat market means nothing. Computed after
  `detect_vcp`, never inside it, so the tiers and pivots can't move.

**Step-3 reads (`advisories.volume_dryup` and the reads after it, §6.91+):** all run after
`detect_vcp` from its `contractions`, like depth vs market, and gate nothing.
- **Dry-up.** Window = the final contraction's peak to the last bar, cut before the first close
  above that peak (a broken-out name is judged on its tight area). Baseline = the 50-bar
  average before the peak. `avg_ratio` = window mean ÷ baseline; `quiet_days` = bars at
  ≤ `DRYUP_QUIET_RATIO = 0.5` of it. Numbers only, no verdict (§6.100). Zero-volume bars are
  gaps and are dropped.
- **Halving.** `book_tightening` gives each dip's depth over the one before and the base length,
  numbers only (§6.100).
- Shakeouts and V recovery were removed in §6.100 after failing their checks (§6.92, §6.93).

**Sector labels (`sectors.py`):**
- `Ticker.info["sector"]` is one of Yahoo's 11 sectors; `["industry"]` one of ~145 industries.
  The group reads use industry. `.info` is the slowest yfinance call (~0.5–1 s) and is
  rate-limited, so only 8/8 passers are labelled (the scan's injected `get_sector`), and
  every page reads cache-only. The first evening after deploy labels ~400 names; later
  evenings only new passers. A held name that was never a passer shows no industry.

**Company numbers (`data_feed.get_fundamentals`, §6.85):**
- yfinance gives ~5 quarters; EDGAR company facts back-fill the older YoY, annual EPS and the
  3-quarter read. yfinance wins. A growth pair (latest YoY, the quarter before) MUST come from
  one source (`_merge_edgar`): EDGAR's 10-Q lags the release, so its newest quarter can be one
  behind yfinance's.
- EDGAR tags: the tag whose newest period ends latest wins (companies switch tags and the old one
  stays in the facts, frozen). A quarterly series older than 200 days, or annual older than 500,
  is dropped. A fiscal Q4 exists only inside the 10-K's year, so it is derived as FY − (Q1+Q2+Q3)
  when exactly three quarters sit inside that year; for EPS that is approximate (share counts
  drift).
- yfinance 0.2.65's `earnings_dates` stops at May 2025 for every name checked, so the surprise is
  kept only when its report is ≤ 120 days old, which today means never. The next report date comes
  from `Ticker.calendar`, which is current.
- A cache is refetched after 7 days, or once its `next_earnings` date has passed if it was written
  on or before that date.
- Only 10-Q/10-K facts count: a proxy's pay-versus-performance table tags net income in millions.
- Code 33 (`_code33`) uses three consecutive quarters ending at the newest EPS quarter: EPS and
  sales YoY growth (a year-ago value ≤ 0 gives no growth figure, so a loss in the window makes
  it unknown) and net margin (`NetIncomeLoss` ÷ revenue, same period end). Each must rise every
  quarter. A seasonal business rarely passes the margin leg; that is the books' strictness,
  not a bug.
- The last release date (`_edgar_last_report`) is the newest 8-K with item 2.02, from the
  submissions endpoint (~150 KB per name, on the backfill's 7-day cycle and after a report).
  Its acceptance time in New York picks the reaction bar (`advisories.earnings_reaction`):
  at or after 16:00 → the next session, otherwise the release day. A before-open release
  whose 8-K is accepted after the close would be read a day late; none has been seen.
  Foreign filers report on 6-K and read None.
- Estimates (`_estimate_revisions`) come from `Ticker.eps_trend`, the "0y" row: the
  consensus for the current fiscal year against 30 and 90 days ago. Around a fiscal year-end
  the row changes year, so the 90-day figure can jump (MLAB +85% over 90, −14% over 30 on
  2026-09-25); the panel shows both. Holders (`_institutional`) come from
  `Ticker.major_holders`; `institutionsPercentHeld` can exceed 100% (13F double counting:
  HALO 112%). The count is a snapshot; `inst_history` keeps its changes across refetches.
  Both are two more Yahoo calls per weekly fundamentals refetch.

**Alpaca facts (alpaca-py 0.43.4):**
- Keys are per-account. Canonical names: `ALPACA_API_KEY_MINERVINI` /
  `ALPACA_API_KEY_SECRET_MINERVINI`, shared fallback `ALPACA_API_KEY_PAPER1` /
  `ALPACA_API_SECRET_PAPER1`. Always `paper=True`. **`ALPACA_BASE_URL` is unused** — the SDK derives
  the endpoint from `paper=True` and appends `/v2` itself.
- `OrderClass` has `SIMPLE/OTO/BRACKET/OCO` — **no OTOCO**. OTO/STOP legs require **whole-share qty**.
- The stop leg **inherits the parent's TIF** — the mechanism behind both the §6.38 bug and its fix.
- The API exposes the account *number*, not the dashboard's friendly name; the UI confirms on last-4.
- `client_order_id` prefixes: `SEPAoto-` (buy+stop), `SEPAstop-` (held stop), `SEPAcockpit-` (naked
  buy). Millisecond timestamps avoid duplicate-id rejects on fast resubmit.
- **Protective stops are exempt** from the $50 floor and the 10%-equity cap — risk-reducing actions
  must never be blocked by a size guard.
- **alpaca-py sends no request timeout**, so a silently dropped connection blocks forever. Every
  cockpit client comes from `trade._connect_paper()`, which calls `_with_timeout`: an adapter
  mounted on the SDK's private `client._session` that fills in **`ALPACA_TIMEOUT_S = (5, 15)`** s
  (connect, read) unless the caller set one. `test_connect_paper_installs_timeout` pins that hook
  against the image's alpaca-py pin, so an upgrade that drops `_session` fails the deploy gate
  instead of shipping without a timeout. (`alpaca_trader.connect()`, the All-Weather mirror, is not
  covered.)
- **A timeout on a submit is ambiguous** — the order may have reached Alpaca and only the response
  been lost. Check the account before retrying a buy/sell that timed out. `execute_sell_plan`
  already refuses to auto-retry a failed order for this reason.

**Universe filter (`full_us`).** Built from NASDAQ Trader `nasdaqlisted.txt` + `otherlisted.txt` over
**HTTPS** (upstream's `ftp://` is commonly blocked). The warrant/right/unit drop **must stay anchored
to `^[A-Z]{4}[WRU]$`** — an earlier unanchored `(?:W|R|U)$` silently removed every ordinary 4-letter
name ending in W/R/U (PLTR, SNOW, UBER, LULU, TROW, DOW, LOW, EMR, KR) and single-letter `U` from the
*only* discovery universe: exactly the high-RS leaders the screen targets. Regression:
`test_get_universe_full_us_offline`. Known limitation: dotted class shares (BRK.B/BF.B) are dropped.

**Rate-limit reality.** `_download_batch`'s backoff is 0.5 s / 1.0 s then it gives up on the batch —
far shorter than Yahoo's actual 429 cooldown (tens of seconds to minutes), so it does NOT recover a
*sustained* limit; it fails politely and those names retry next sweep. The durable fix is the
incremental cache plus keeping sweeps rare (§6.64). If sustained limits return, the options are an
adaptive cooldown (all-empty batch → sleep 30–60 s once) or an Alpaca daily-bars backend for the cold
scan. **IP/proxy rotation was raised and rejected** — ToS-violating and the wrong tool here.

## 11. Change ledger

Moved to `HANDOFF_HISTORY.md` §11, where the `§6.NN` anchors cited in test docstrings and
source comments live. A new entry goes there, numbered after the last one.

## 12. Open items

- **The SEPA audit is frozen (§6.100).** No new feature is built from `SEPA_AUDIT.md` until the
  journal has about 20 more closed trades; then re-read it against the record. A change MUST
  come from a real trade that would have gone differently, or from a gap the journal shows.
- **The pyramid add is parked** on branch `park/pyramid-add` (§6.99 lives there). Merge it when a
  real winner sits in a new buy zone; its ledger entry goes into `HANDOFF_HISTORY.md`.
- **~95 orphan parquets** (~2.2 MB) for names that left the universe. **Do not prune by universe
  membership** — `SPY` is filtered out of `full_us` as an ETF and is the benchmark behind every RS
  rating and the regime banner. No orphan is older than 90 days, so a staleness rule catches nothing
  either. The interesting question is the opposite one: why `ACN`/`BRK-B` fall out of
  `_filter_us_symbols` at all.
- **`AUTOSELL=1` on the Pi since 2026-10-02**: the evening plan's full exits (hard P1/P2/P4
  fails) submit at 09:25 unless vetoed on the Positions page. `AUTOBUY` is unset.
- **P2's RS leg can flip daily.** The RS rating is a rank across the universe, so a holding near
  70 can fail P2 on one close and pass the next. The two-consecutive-closes rule is the only
  hysteresis. If RS-only P2 fails prove noisy on live positions, a band (fail under 65, warn
  65–70) is a separate decision, not a tune. Also: the 16:15 sell plan reads the RS from the
  previous evening's scan, because `cockpit-eod` screens at 16:20; a same-day RS dip reaches the
  plan one evening late.
- **When to flip the two switches.** `VIOLATIONS_CAN_FAIL`: after a few weeks of watching the P1
  violation warnings on live positions. Would 3-at-once have exited earlier than the existing P1
  rules, and at a better price? `MARKET_TURN_CAN_TRADE`: only once a turn has been seen and
  handled by hand at least once. `STRENGTH_CAN_TRADE`: after the signs have been read against a
  few live runs to +20% and beyond. All are one-line changes in `doctrine.py`.
- **Re-apply the §6.73 floor rule at the 5th win.** The derived stop turns on by itself at 5 cockpit
  wins, but its 4% floor rests on one. Re-run the sweep then (the Journal page shows it) and
  update `DERIVED_STOP_FLOOR` only by the pre-registered rule: the smallest grid stop ≥ 3% at
  which no closed winner becomes a loss.
- **Base count (§6.76) needs a new, pre-registered design** before any code: the two failure
  mechanisms (single-threshold blindness, the transition base) are written up there. The demotion
  would have changed no benchmark tier, so recall is not what is at stake.
- **Deferred: a tier-C exclusion for wild movers (≥8%/day typical range)** — see §6.75 for why.
- **Step-3 reads that failed or don't separate (§6.91–§6.94).** Shakeouts and V recovery were
  removed; dry-up and halving show numbers only (§6.100). A second attempt at any of them needs
  a new design and a new pre-registration, not a threshold change.
- **P1's older checks still read today's provisional bar on live calls.** Day-0 below the pivot,
  the decisive close, the second close and the breakout-bar low use `last_close`. Only the §6.77
  reads drop an unsettled bar. The evening plan runs after the settle, so the automation is
  unaffected; the Positions page mid-session can show a P1 ❌ that the close then clears.
- **The earnings surprise has no working source (§6.85).** yfinance 0.2.65's `earnings_dates`
  stops at May 2025. A newer yfinance may fix it, but the pin also governs every price download,
  so an upgrade is its own change with its own test run on the Pi. Until then the surprise reads
  n/a everywhere. Once it works, it is the natural ninth F check (the guide's "earnings
  surprises are a green light").
- **Leaked Yahoo connections.** After 20 days up, the app held ~15 CLOSE_WAIT sockets to
  `query1/2.finance.yahoo.com` (the server closed them; the process never did). It's harmless at that
  count and was not addressed in §6.71. If the app is ever up for months, count them
  (`/proc/1/net/tcp`, remote port `01BB`) before suspecting anything else.

## Files (this venture)

- **Vendored rules:** `minervini_screener/` — `screening/{phase_indicators,signal_engine,benchmark,indicators}.py`; `LICENSE`, `PROVENANCE.md`. That is the whole package: the live-only modules (`data/`, `notifications/`, `analysis/`, batch processors, `quant_engine.py`, `screener.py`) were deleted 2026-09-02.
- **Harness:** `backtest_daily/` — config, providers (synthetic + WRDS), cache_io, indicators_cache, signals, regime, sizing, portfolio, metrics, engine, `run_backtest.py --wrds`.
- **Cockpit:** `cockpit/` — see the module map in §6. Deployment in `deploy/` (`deploy.sh`, `install-units.sh`, `units/`, `PI_SETUP.md`).
- **Weekend hunt:** `hunt/` — deterministic Step-3 review pipeline; the `/weekend-hunt` skill judges the charts. `scripts/hunt/` (repo root) holds the env wrapper `hunt.sh` and the Friday task (§8).
- **Method:** `SEPA_METHODOLOGY.md` — the books' rules for sessions, with a rule-to-code map (§9 of that file) and the deviations; `minervini_sepa_system.md` is the user-facing guide the app renders. A rule change updates both, plus the ledger in `HANDOFF_HISTORY.md`.
- **History:** `HANDOFF_HISTORY.md` — the research record (§1, §3, §5), the change ledger (§11) and parked ideas.
- **Tests:** `tests/test_cockpit.py` (runner) + `tests/cockpit/` · `tests/test_hunt.py` · `tests/test_backtest_daily.py` · `tests/test_wrds_provider.py` · `tests/test_momentum_lib.py`. Run as plain scripts. **Only the first two gate** — the parked-track suites run in neither CI nor `deploy.sh`.
- **WRDS pull:** `ingest_wrds.py` → `data/wrds/*.parquet` (gitignored). Backtest outputs saved as `data/wrds/_bt_*.csv` — start the delisting work from these.
