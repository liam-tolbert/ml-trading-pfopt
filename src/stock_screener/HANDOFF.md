# HANDOFF — Stock-Screener (Minervini) venture

**Scope:** the *classical* stock-screening track. Separate from `docs/HANDOFF.md` (the parked ML
cross-sectional track). The momentum-*factor* experiment is closed; its essentials are folded
into §5 and its standalone write-up was never committed.

**Status (2026-08-26):** live paper trading on a dedicated Raspberry Pi. Weekly `full_us` hunt →
frozen-pivot watchlist → half-hourly refresh + trigger checks → GTC-stopped entries, with sell
automation and armed entries built but **disarmed** (`AUTOSELL`/`AUTOBUY` unset). Suite **151 cockpit
+ 33 hunt**, offline, both gating deploys.

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

The sibling `stock-screener` repo is a **complete, faithful Minervini system** — 8-point Trend
Template as a hard gate, Stage 1–4 phases, VCP, breakout+volume, the SEPA fundamental leg, stops,
R:R≥2, sell signals, a market-regime cash gate. It emits hard buy/sell signals, but is **live-scan
only — no backtest**. Vendored at `minervini_screener/` (MIT, © 2024 Ryan Hamby, upstream `397e555`;
`LICENSE` + `PROVENANCE.md` kept). **No business logic was ever changed**: the edits are
`from src.` → relative imports, a `calculate_sma` dedup, and the re-export list. The unreachable
23 of its 27 modules (`data/`, `notifications/`, `analysis/`, both batch processors,
`screener.py`, `quant_engine.py`) were deleted 2026-09-02 — see PROVENANCE.md.

`backtest_daily/` (16 modules) replays history one day at a time through a single leak-safe slice
(`ohlcv_upto(t)`), with stops, sell signals, a regime cash gate, risk-based sizing, and CAGR/Sharpe/
maxDD/CAPM α-β reporting. Data behind provider interfaces: a synthetic provider for tests, a WRDS
provider for the real pull.

**Data (survivorship-free):** top-3000-by-cap quarterly, 2003→2024-12-31 (CRSP is annual-update).
**8,462** names, **20.8M** daily rows, **4,823 in-universe delistings**, **341K** Compustat rows.
A `$5` floor on the **unadjusted** price (`raw_close`, so split-cheap early AAPL isn't excluded) plus
a despike for cfacpr/penny artifacts.

**Results:**
- **Baseline:** CAGR 3.90% vs SPY 8.84%, Sharpe 0.29 vs 0.55, maxDD −65.9%, α +0.22% (t=0.06).
- **−66% DD diagnosed:** 2008 rode the crash ~100% exposed (regime gate blocked new buys but
  `exit_on_regime_flip=False` meant no liquidation); 2009–10/2020/2023 whipsaw re-entries.
- **Two fixes → strong in-sample:** `exit_on_regime_flip=True` + `regime_confirm_days=25` →
  CAGR 11.1%, Sharpe 0.72, maxDD −40%, **α +7.98%/yr t=2.49**. *But 25 was chosen by looking at the
  full history's bad years.*
- **★ OOS killed it.** Pre-registered sweep on TRAIN 2003–13 (best = 15, a sharp peak, not a
  plateau) applied to held-out 2014–24: CAGR 7.95% vs SPY 11.18%, **α +2.18%/yr t=0.47**.
- **What survived:** the confirm-lag beat no-lag OOS (Sharpe 0.22→0.51, DD −43.6%→−35.6%) —
  "wait for the market to confirm before re-entering" is genuine **damage control**, not alpha.
  Even that is regime-dependent (OOS DD −35.6% ≈ SPY's −34.1%).

Engine notes: ~45 min on the full 8,462×5,537 cache; a cash-aware entry gate (skip the scan when free
cash <2% of equity) made it tractable. Verified by `tests/test_backtest_daily.py` **12/12**, including
`test_engine_decisions_leak_free` (a run truncated at D_mid reproduces a full run bit-for-bit).

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

The live pull, the backtest, and OOS validation are **done**. "Does Minervini beat the market" is
answered: no OOS alpha. What remains:

- **★ Delisting-avoidance — the one genuinely open, orthogonal question.** Does the screen's *score*
  rank which names subsequently DELIST? Distress prediction is a different hypothesis that could be
  real even with no long-only alpha, and only this dataset (**4,823 delistings**) can test it. Start
  from `data/wrds/_bt_*.csv` — NOT in this checkout (gitignored, never committed), so re-run
  `ingest_wrds.py` then `run_backtest.py --wrds` to regenerate them first.
- **Where alpha could actually be.** Selection AND timing on broad US equity are exhausted — ML,
  momentum factor, equal-weight, and the faithful screener all return no OOS alpha, and the residue
  every time is risk management. To find alpha, change the **signal or the arena** (a less-efficient
  niche), not the tuning.
- **Reconcile the buy threshold** (`is_buy = score>=60` vs docstring ≥70) via
  `BacktestConfig.buy_score_min` — minor, open.

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
- **Don't design around a "known" API constraint that was never verified live.** "Alpaca market OTOs
  can't be GTC" was assumed for a month and was simply false — the disproof was a five-minute
  1-share probe, and the assumption cost an unprotected overnight position.
- **Don't let tests depend on wall-clock/market state.** Any time-coupled feature needs its test
  bypass designed in, not discovered.

## 5. Momentum factor (closed 2026-06-04)

Ported Minervini *ideas* into continuous scorers, tested as a factor on a 240-ticker weekly panel
2018→2026. **Phase 0 verdict: STOP** — every beta-neutral L/S α small-negative and insignificant
(|t|<0.9), gross α≈0, so the ranking carries no market-neutral info (an uninformative null, unlike
the ML L/S which was significantly negative). Long-only top30 Sharpe 0.825 / α +11.4% but **t=1.58**
and survivorship-suspect. Code: `momentum_lib.py` + `tests/test_momentum_lib.py`.

---

## 6. The cockpit — architecture

A local **Streamlit** app running the SEPA funnel as decision support. `src/stock_screener/cockpit/`,
live yfinance data (not CRSP). Reuses ONLY the pure rule functions from `minervini_screener/screening/`.

**Funnel:** universe (`full_us`, ~4,120 names — the ONLY universe since §6.32) → **Step 1** hard gate,
full **8/8** trend template → RS rating (IBD-style weighted multi-horizon, §6.17) → **Step 2**
fundamental highlight (rev/EPS YoY/QoQ + margins) → **Step 3** VCP tier (cockpit detector, §6.1) →
**Step 4** advisory levels (pivot / buy zone / stop / target) + sizing. A regime/breadth banner gates
the discipline.

**Module map:**

| Module | Role |
|---|---|
| `data_feed.py` | yfinance layer: universe, price cache, incremental top-ups, EDGAR/fundamentals |
| `scan.py` / `vcp.py` / `indicators.py` | the funnel, the VCP tier detector, RMV/BBWP/squeeze |
| `scan_worker.py` | background scan thread + process-wide result store (`last_scan.pkl`) |
| `triggers.py` | pure trigger evaluation; `export.py` the watchlist store |
| `trade.py` | Alpaca paper submit path, stops, the exposure gate |
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

**Entry.** Buy zone = pivot..pivot×1.05 (no chasing). Trigger = settled close above the frozen pivot
on **≥1.5× the 50-day average volume**. Stop 7–8% below the *pivot*, **10% hard max** (clamped in
`_entry_levels`). Target pivot×1.25. Skip inside ~21 days of earnings. Size for ~1% account risk
(0.5% while probing); the 10%-of-equity single-order cap clamps risk-mode quantities.

**Sell — the four pillars (P1–P4).** A breakout buy's thesis is P1 resolution (closed above the pivot
on volume and STAYS above) ∧ P2 Stage-2 structure (8/8, rising 50-SMA) ∧ P3 risk-on tape ∧ P4 no
unpriced binary. **Any pillar failing kills the trade; the stop is the disaster floor for what happens
between checks, never the sell signal.** Decisions on settled closes only; execution at the next open.
- Day 0: entered intraday, closed back below the pivot → the breakout never happened, sell next open.
- Until cushioned: decisive close below the pivot, or below the breakout bar's low → sell.
- Laggard clock: no +3% cushion by ~day 10 → sell into strength; flat-to-red at day 15–20 → exit.
- Earnings: a LOSS is never carried into a ≤21d report; a small gain is trimmed to hold-through size.
- Automation acts on hard ❌ of P1/P2/P4 only. P3 and all warns are report-only. P2 needs **two
  consecutive** failing closes (the strict template has one-day SMA noise flips).

**Progressive exposure.** Gate is open when flat, or when every position in the newest-day cohort is
at breakeven-or-better AND tagged net open P&L ≥ 0. Scope is cockpit-**tagged** positions only, so
manual legacy holdings can never poison "flat". Two consecutive tagged losses → half-size advisory.
**Fail direction is asymmetric by design:** unknown → manual path OPEN (the human judges), unattended
executor CLOSED.

**Judging a base.** RMV is the tightness discriminator (<25 tight, >45 loose) — tightening legs and
`fund_score` can flatter a loose base. Read raw rev/EPS YoY, not `fund_score` (it counts n/a as a
fail). Never hard-gate on fundamentals: patchy yfinance data would drop *thin-data* names, not weak
ones — the same never-miss failure the VCP gate had. Shrink the list with **Tier A + RS** instead.

**Live scorecard (10 closed, as of §6.53):** 1W/9L, expectancy −1.7%, −$10.4k ≈ −1.1% of equity over
a window where SPY made +3.2%. Loss control WORKED (avg loss −2.3%, worst −6%, avg win +4.0%); what
failed was follow-through. Every dollar lost traced to *entry*-rule violations. The sell side, when
run, worked.

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
| `cockpit-eod` | 16:20 weekdays | **two sequential steps in one unit**: full-universe price top-up (arms the settled-close serve), then the universe screen that rebuilds `last_scan.pkl` |
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

## 9. Conventions

- **Source comments say WHY the code must be this way, never which bug/review/date produced it**
  (§6.47). The incident ledger lives in test docstrings and this file.
- **Tests run as plain scripts**, no pytest. `python tests/test_cockpit.py` is the gate's entry point;
  the suites live in `tests/cockpit/test_<category>.py` and each runs standalone.
- **Exit-code contract: 0 for anything normal — including "nothing to do" and "disabled" — and 1 only
  for a real failure.** This is what makes a disarmed executor show green in systemd.
- **A suite added to the gate must reach the image.** `.dockerignore` is a whitelist; `tests/cockpit/**`
  is a glob so new category files need no second edit.
- **Tests must not depend on wall-clock or market state.** Time-coupled features need an explicit
  bypass (e.g. the negative-`max_age_days` sentinel).
- **The user commits all code themselves.** Claude leaves the tree dirty for review — their push is
  the human gate in front of the Pi's auto-deploy.
- **Don't edit tracked files inside the Pi's checkout** — a modified tracked file trips the
  dirty-checkout halt forever after. Stage work in `/tmp` instead.
- **AppTest gotchas:** `at.session_state` has no `.get()`/`.setdefault()`; widget refs go stale after
  each `.run()`; there is no `at.download_button`/`at.file_uploader` accessor — test pure helpers
  directly instead, which is why sizing/parsing live in `export.py`/`trade.py`, not inline in `app.py`.

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

**`suggest_stop` bases (Positions page, auto mode):** fresh (gain < `BREAKEVEN_GAIN` 0.16) → 8% below
entry · working → breakeven · well in profit (≥ `TRAIL_GAIN` 0.20) with a 50-day → trail
`sma_50 × 0.99`. Floored at the in-force stop (ratchet-safe). `None` = underwater → manual row.

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

**Windows Task Scheduler (legacy laptop path, if ever re-registered):** clear the battery flags or an
unplugged laptop silently skips runs with no error anywhere.

```powershell
Get-ScheduledTask -TaskName "SEPA Intraday Trigger" | ForEach-Object {
  $_.Settings.DisallowStartIfOnBatteries = $false
  $_.Settings.StopIfGoingOnBatteries = $false
  Set-ScheduledTask -InputObject $_ }
```

## 11. Change ledger

Anchors for the `§6.NN` references in test docstrings and source comments. Detail is in git history.

- **§6.1** Recall-first VCP rework + 200-chart blind benchmark (72 YES/128 NO). Tiers A/B/C, multi-threshold ZigZag, RMV veto only *below* the pivot. Contract: **tier C contains zero YES**.
- **§6.2** Watchlist, CSV/txt export, Alpaca paper-trade panel, layout reflow. Reverted: right-arrow row nav (version-brittle DOM injection — don't re-attempt).
- **§6.3** Auto-attach protective stop on submit (OTO); credential-name bugs that were reaching *no* account. `_first_env` must treat a bare `str` as ONE name.
- **§6.4** Earnings-date awareness (`earnings_in`, 21-day window). Advisory only.
- **§6.5** Cockpit backlog from a full SEPA fidelity review — nearly every gap was *after* the buy button. Fully closed 2026-07-16.
- **§6.6** GTC stops + never-lower ratchet. Supersedes §6.3's DAY choice.
- **§6.7** Risk-to-stop sizing mode. Anchor = current price; cap clamps rather than skips.
- **§6.8** Stop clamped to 10% below the pivot (Minervini's hard max).
- **§6.9** Positions page — the daily stop-management habit; ratchet extracted to one shared helper.
- **§6.10** Re-scan force-refresh parity bug (`nonce % 2` fired every *other* click).
- **§6.11** Weekend Tier-A review workflow, trade cadence, and (2026-08-05) the Step-E sell procedure.
- **§6.12** Persistent watchlist across runs.
- **§6.13** Trade journal from `client_order_id` tags.
- **§6.14** Frozen-pivot watchlist + the EOD trigger check.
- **§6.15** Regime warning at the point of action.
- **§6.16** Table declutter, target locked to +25%, scan progress bar. One pivot for display.
- **§6.17** IBD-style RS + up/down volume bars; EDGAR shelved.
- **§6.18** Half-hourly intraday trigger checks; nightly prewarm removed.
- **§6.19** First-week operating notes: PECO case study, post-breakout freeze semantics, EDGAR precedence (EDGAR does **not** override yfinance).
- **§6.20** Multi-agent code review — 34 verified findings (2 high, 13 med, 19 low), backlog tracked in a scratch file that was never committed. Both HIGHs fixed here.
- **§6.21** Comment-trim pass to "moderate" density.
- **§6.22–§6.23** Phase 1 (items 1–10). Includes the staleness guard (`max_bar_age_days`) and `freshen_prices`, so a plan sizes on current bars rather than the scan memo's possibly days-old closes.
- **§6.24** Item 11 — watchlist lost-update race; saves are now atomic (tmp + `os.replace`) and merge into a fresh read of the file, so a concurrent app save is never clobbered by the refresh job's auto-freeze.
- **§6.25** Item 12 — the new-day 2y-refetch avalanche. Root cause: provisional intraday bars poisoned the split check, so every name looked re-adjusted and re-baselined. Scans are now always incremental; Re-scan = top-up, Advanced ⟳ = full re-download.
- **§6.26** Per-buy checkboxes in the trade plan.
- **§6.27** Earnings-aware sell advisories + manual selling on the Positions page.
- **§6.28** Scan download-transparency log (per-name fetch labels).
- **§6.29** 30-minute scan freshness window. *(Later superseded by §6.37's settled-close serve and §6.64's scope split.)*
- **§6.30** Phase 2 (items 13–17). Item 15: `make_entry` applies the yfinance dash convention to dotted tickers.
- **§6.31** Phase 3 (items 18–22). Filter tweaks became instant — the scan runs ONCE at the loosest gates and sliders apply via pure `scan.filter_candidates` over the memoized result (never mutated). Deleting the guarded vendored `.screener` import took **~1.7 s off every process start**. BBWP and the VCP hot loop were hoisted; the 200-chart benchmark line stayed byte-identical.
- **§6.32** Phase 4 (items 23–32) — **review CLOSED**: 30 fixed with tests, 2 kept per PROVENANCE, 2 refuted.
  - **24 (user decision): `full_us` is the ONLY universe** — the selectbox is gone; sp500 remains as the offline fallback, and `run_scan`/`get_universe` now REQUIRE the
    argument so a bare call cannot screen a different one.
  - **23 (the medium, A/B-gated):** `scan.detect_breakout_prior_high` makes the Base/Pivot Breakout branches reachable via prior-bar 60/20-day highs (vendored file untouched; the backtest keeps old behavior). A/B over the 200 fixtures: **12/200 pivots changed, all moved DOWN** (median −5.3%, max −15%) from the 52-wk-high fallback to real base highs. Already-frozen 📌 pivots untouched.
  - **29:** `STATUSES` is load-bearing — a new trigger status must be registered there or the report test fails.
- **§6.33** Watchlist pills picker — and **the test that wiped the real `watchlist.json`**. Tests must never touch live state.
- **§6.34** `untracked` trigger status + sidebar chart-jump buttons.
- **§6.35** Manual "Check triggers now" button (the escape hatch when a scheduled run is missed).
- **§6.36** `crossed` status — above the pivot *without* volume confirmation. Loud, but **not a buy signal**.
- **§6.37** Settled-close cache serve — no session elapsed ⇒ cache current at any age.
- **§6.38** OTO buys GTC end-to-end. The expiring-DAY-leg incident: a fill at 15:58 lost its stop at the 16:00 close.
- **§6.39** RS-line-at-new-high flag; held-name stop gap at build closed.
- **§6.40** Background scan worker — the scan runs in a daemon thread that never touches Streamlit APIs, so a page switch kills the script run, not the fetch.
- **§6.41** Limit buys in the trade panel (the no-chase cap becomes the entry mechanic).
- **§6.42** Process-wide scan store, stale-while-refresh UI, warm-path speedups. One worker per browser session; `run_scan` serialized process-wide so two LAN sessions can't race yfinance.
- **§6.43** Review round 2 + its HIGH fix: **build-time intent is binding.** A row stamped `rearm_only` (held at build) or `stop_only` (zero-share stop carrier) is SKIPPED if its position later closes — never silently converted into a buy the user didn't consent to. Rebuild the plan to buy such a name.
- **§6.44** Review round 2 closed — all mediums + lows.
- **§6.45** Last-scan persistence (`last_scan.pkl`) so a server restart serves instantly. The entry keeps the ORIGINAL scan time; `completed_mono = -inf` so an in-flight run can never adopt it.
- **§6.46** Positions page decoupled from the bulk pipeline. Live incident: the user couldn't sell NMM while a refresh held the yfinance lock — interactive pages now check `data_feed.network_busy()` and serve cache-only rather than queueing behind a multi-minute sweep. **Selling must never wait on a 4,000-name download.**
- **§6.47** Comment de-verbosing — the convention in §9.
- **§6.49** Trigger task un-blocked on battery (Task Scheduler defaults silently skipped runs).
- **§6.50** `pullback` status — the low-risk secondary entry.
- **§6.51** Progressive-exposure guidance at the point of sizing.
- **§6.52** P1–P4 sell pillars on the Positions page.
- **§6.53** Execution audit + THE RULES (§7). Findings: intraday decisions, 2–5 day sell latency, batch entries, watchlist drift, dark windows.
- **§6.54** Pi migration + pull-based Docker deploy pipeline.
- **§6.55** Auto-sell from the pillars (evening plan + morning submit, ships dark).
- **§6.56** Scan refreshes: interaction-driven TTL → scheduled. Page interaction never downloads.
- **§6.57** `install-units.sh` — one-command unit sync.
- **§6.58** Exposure gate, armed entries, free-roll (GH #23/#24/#19).
- **§6.59** **The units were installed at a path that did not exist — 3 days of silent no-deploys.** `install-units.sh` rewrote only the `/home/pi` prefix, so a clone at `~/Documents/…` produced a dead path. 29 fires, 0 successes, hidden because `cockpit:live` kept serving and the checkout looked current.
- **§6.60** Run logging — dated files, 14-day retention, one summary line per sweep (never per ticker). `cached` is the field parquet mtimes cannot give you.
- **§6.61** Deploy gate covers `test_hunt.py` too. **Rule: a suite added to the gate must be un-ignored in `.dockerignore`**, or it fails as "can't open file" rather than as a red test.
- **§6.62** Deploy-timer cadence.
- **§6.63** `_cli` → `_job` renames, `cockpit-trigger` → `cockpit-refresh`, compose `trigger` → `oneshot`, and the in-app refresh scheduler **deleted**. That thread was invisible to `list-timers` and died with the container — a deploy landing after its 16:05 slot silently cost that day's scan. Every scheduled thing now lives in systemd.
- **§6.64** **Refresh scope split.** The half-hourly job was topping up all ~4,120 names — ~12 min per run, ~2.8 h/day, and yfinance started returning `YFRateLimitError` + `database is locked`. Root cause: `max_age_days=0.0` makes the freshness branch *unreachable* (no file is ever ≤0 days old) and `_cache_settled` is false by definition during a session, so every name fell through to a fetch — hence `cached 0` on every sweep. Now: intraday = watchlist ∪ **held positions** (~6 names, ~1 s); universe once at 17:00. `REFRESH_MAX_AGE_DAYS = 0.01` is a duplicate-work valve, kept under **half** the cadence because age is measured from a file's *write* time, so the real margin is the interval minus the run's duration.
- **§6.65** **The caption was lying.** "data as of HH:MM" read `completed_wall` from the last *screen*, which only Re-scan advances now — so it sat a day stale while prices were minutes old. Now `scan <t> · prices <t>`, price freshness from the trigger report's `generated_at`, with the date shown once a stamp isn't from today.
- **§6.66** **Deploy hardening.** A six-day-old stopped container (`docker run` without `--rm`) pinned an image, `docker rmi` failed, and under `set -e` that **aborted a deploy that had already promoted** — so `DEPLOYED` and the units-changed reminder never printed and systemd logged a good deploy as failed. Prune is now non-fatal and names the holding container; steady state is exactly two tags (`live`, `prev`); `--reserved-space` replaces the deprecated `--keep-storage`.
- **§6.67** **`sudo bash deploy.sh` left root-owned `/tmp/cockpit-deploy.lock`, `.git/HEAD`, `.git/index`, `.git/ORIG_HEAD`** — locking out every later run *and* the scheduled deploy. The lock open now fails with a diagnostic instead of a bare "Permission denied". `docker ps` hides the containers that cause this class of problem; use `docker ps -a`.
- **§6.70** **Debt/staleness sweep.** An audit of the whole venture, then nine staged fixes.
  - **The test fixture was shipping two parked research tracks to the Pi.** `tests/cockpit/_common.py` imports `backtest_daily.synthetic_provider` for its offline prices; the package `__init__` re-exported the engine, so that import pulled `engine → metrics → momentum_lib + ml_stock_prediction/backtest_lib` and `.dockerignore` had to whitelist all of it. The `__init__` now re-exports NOTHING (no consumer used the package-level names) and the whitelist is four files. `test_synthetic_fixture_isolated_from_engine_chain` is the guard.
  - **23 of the 27 vendored modules were unreachable and were deleted** (`data/`, `notifications/`, `analysis/`, both batch processors, `screener.py`, `quant_engine.py`) — recorded in PROVENANCE.md as MIT requires. `notifications/` had in fact been **import-broken since 2026-07-22**: `scheduler.py` imports `screen_candidates`, which stopped being re-exported when the `.screener` import was dropped. Nothing noticed, because nothing imported it.
  - **★ One doctrine rule had two values.** The weekend hunt confirmed breakouts at **1.4×** over the vendored **20-day** window while `triggers.py` used **1.5×** over the prior **50**, so the same rule read differently depending on which surface you asked. Constants now live in `cockpit/doctrine.py` (imports nothing, so `trade.py` and `hunt/` can both read it) and the single `indicators.volume_ratio` is the only implementation. The lesson is the general one: **a duplicated constant is a fork with a delayed fuse** — `EARNINGS_SOON_DAYS` existed in four files, `MAX_STOP_FROM_PIVOT` in two.
  - `gates()` now confirms only inside the **buy zone** — a heavy-volume close 6% past the pivot is chasing, not a confirmation, and `report.py` (which already scoped it that way) now calls `gates()` instead of re-deriving the buckets with a `min_fund` divergence.
  - **`cockpit-sellexec` logged red on a normal skip:** `sell_job` returned exit 1 on `stale`, which is an ordinary morning outcome (a holiday, a missed evening fire). Fixed to the §9 contract, with `test_sell_job_execute_stale_exits_zero`. Storage for both plans merged into `plan_store.py`; the two copies had already drifted (only one cleaned up its temp file on a failed write).
  - **Docs described a schedule that no longer existed** — 16:30/16:40 fire times in seven docstrings, a `REFRESH_SCHEDULE_ET` constant deleted in §6.63, "tops up the whole universe" (the §6.64 misconception, still asserted in source), and a **Sat 10:30 timer that never existed** named in an error message the user is told to act on. `PI_SETUP.md` was wrong in six places including the clone path — the §6.59 outage's root cause.
  - Dead members removed: `FREE_ROLL_FRACTION`, `cache.TICKERS_TXT` (the file never existed, so the "offline fallback" was always `[]`), and the vcp payload's `breakout_volume_ratio`/`near_52w_high`/`distance_from_52w_high_pct` — the first computed per ticker on a ~4,100-name hot path for no reader. `run_scan`/`get_universe` no longer DEFAULT to `sp500`: the argument is required, so a bare call fails at the call site instead of silently screening a different universe.
- **§6.68** **Test suite split.** 6,411 lines / 144 tests → a 70-line runner plus 12 category suites under `tests/cockpit/`. Split by an AST script that aborts unless every test is assigned exactly once. `_common.py` holds the 12 shared fixtures and must re-export them via `__all__` (`import *` skips underscore names). Two path traps: `ROOT` moved to `parents[2]`, and `vcp_labels` is imported *bare*, which only resolved while the suite ran as a script from `tests/`.
- **§6.69** **The two EOD units became one.** The first-ever `cockpit-screen-eod` run (2026-08-28) exposed both halves of the problem at once. (a) It crashed at the last line with `ValueError: The truth value of a DataFrame is ambiguous` — `screen_job.py` did `getattr(res, "candidates", []) or []`, and `candidates` is a DataFrame. The crash landed *after* `store.put`, so the scan table was correct and only the exit code lied; **never `or []` a DataFrame**. (b) The sweep ran 16:20:20→16:35:43 (15m23s) while the screen fired at 16:25, so the two contended for yfinance and the screen re-fetched what the sweep had not reached — 30 minutes against the ~5 a warm cache costs. A clock gap cannot enforce ordering against a job whose runtime varies 11–18 min, so `cockpit-refresh-eod` + `cockpit-screen-eod` collapsed into `cockpit-eod`: `Type=oneshot` with two `ExecStart=` lines, which systemd runs serially and abandons if the first fails. `TimeoutStartSec` is **per-unit, not per-ExecStart** — hence 6000, the sum of the old two. `install-units.sh` removes installed `cockpit-*` units the repo no longer ships, so the old pair disappears on the next `sudo deploy/install-units.sh` with no manual cleanup.


## 12. Open items

- **Auto-arming entries — discussed, not built.** `triggers.py` already computes `triggered` and
  `entry_job.py` already submits one capped, stopped limit order at 09:26. The missing link is the
  buy-side equivalent of `cockpit-sellplan`: turn settled-close `triggered` names into armed rows,
  leaving the overnight disarm window intact. **Never drive it from INTRADAY triggers** — the volume
  ratio is provisional and only clears 1.5× late in the session, so it would systematically buy near
  the close.
- ~~Nothing advances the scan automatically~~ — **resolved (§6.69).** The screen is step 2 of
  `cockpit-eod`. The caption's `scan <t>` stamp should now move every weekday evening; if it stops,
  check `systemctl status cockpit-eod` before suspecting the app.
- ~~§6.62 follow-up: confirm no unit overlaps the evening ritual~~ — **resolved (§6.69)** by
  collapsing the sweep and the screen into one unit. The remaining adjacency is `cockpit-deploy` at
  17:00, which can fire while a slow `cockpit-eod` is still screening; that is benign (the deploy
  touches no market data, and `store.put` is tmp + `os.replace`), but it is the next thing to look at
  if the evening ever misbehaves.
- **~95 orphan parquets** (~2.2 MB) for names that left the universe. **Do not prune by universe
  membership** — `SPY` is filtered out of `full_us` as an ETF and is the benchmark behind every RS
  rating and the regime banner. No orphan is older than 90 days, so a staleness rule catches nothing
  either. The interesting question is the opposite one: why `ACN`/`BRK-B` fall out of
  `_filter_us_symbols` at all.
- **`AUTOBUY` / `AUTOSELL` unset** on both boxes — no automation is armed.

## Files (this venture)

- **Vendored rules:** `minervini_screener/` — `screening/{phase_indicators,signal_engine,benchmark,indicators}.py`; `LICENSE`, `PROVENANCE.md`. That is the whole package: the live-only modules (`data/`, `notifications/`, `analysis/`, batch processors, `quant_engine.py`, `screener.py`) were deleted 2026-09-02.
- **Harness:** `backtest_daily/` — config, providers (synthetic + WRDS), cache_io, indicators_cache, signals, regime, sizing, portfolio, metrics, engine, `run_backtest.py --wrds`.
- **Cockpit:** `cockpit/` — see the module map in §6. Deployment in `deploy/` (`deploy.sh`, `install-units.sh`, `units/`, `PI_SETUP.md`).
- **Weekend hunt:** `hunt/` — deterministic Step-3 review pipeline; the `/weekend-hunt` skill judges the charts.
- **Tests:** `tests/test_cockpit.py` (runner) + `tests/cockpit/` (12 suites, 151 tests) · `tests/test_hunt.py` (33 assertions) · `tests/test_backtest_daily.py` (12) · `tests/test_wrds_provider.py` · `tests/test_momentum_lib.py`. Run as plain scripts. **Only the first two gate** — the parked-track suites run in neither CI nor `deploy.sh`.
- **WRDS pull:** `ingest_wrds.py` → `data/wrds/*.parquet` (gitignored). Backtest outputs saved as `data/wrds/_bt_*.csv` — start the delisting work from these.
