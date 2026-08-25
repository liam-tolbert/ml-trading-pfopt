# HANDOFF — Stock-Screener (Minervini) venture

**Scope:** the *classical* stock-screening track — validating Mark Minervini's screener on a
proper, survivorship-free, daily backtest. This is a SEPARATE document from `docs/HANDOFF.md`
(the parked ML cross-sectional track) and from `docs/MOMENTUM.md` (the momentum-*factor*
experiment, Phase 0, closed — its essentials are folded in at the bottom).

**Status (current, 2026-08-05):** the cockpit is in **live paper trading** — weekly `full_us`
hunt → frozen-pivot watchlist → half-hourly trigger checks → GTC-stopped entries; suite **95/95
offline**. Latest increments (the only uncommitted work on `main`): §6.36 "crossed" trigger
status, §6.37 settled-close cache serve, §6.38 GTC OTO stops (the expiring-DAY-leg incident),
§6.39 RS-line flag + held-stop gap. The 2026-07-16 multi-agent review is fully closed (§6.32).
Research verdict below stands unchanged.

**Research verdict (2026-06-29): full survivorship-free backtest RUN — no out-of-sample alpha** (a strong
in-sample result, α t=2.49, was OVERFIT — an OOS test collapsed it to t=0.47). Risk-management rules are
real; selection is not. Venture continuing (see §3 for live directions).

**PIVOT (2026-06-29) — from full automation to a human-in-the-loop "cockpit."** The backtest only ever
exercised SEPA *Step 1* (the 8-pt Trend Template, as a hard gate); Steps 2–4 (fundamental acceleration,
VCP, breakout entry) were soft inputs to a composite score, never gates — and Minervini's edge in those
steps is *discretionary*. So rather than re-automate, we built a decision-support tool that does the
mechanical filtering (Step 1 list + Step 2 fundamental highlights) and hands the *user* interactive charts
to judge the VCP (Step 3) and advisory entry levels (Step 4). The user is the judge. See §6.

The two halves of this venture live under `src/stock_screener/`:
- `minervini_screener/` — the vendored, third-party Minervini screener (the *rules*).
- `backtest_daily/` — a new event-driven daily backtest harness (the *simulator*) that drives
  those rules over point-in-time data.

---

## 1. What happened

- **Found that the sibling `stock-screener` repo is a complete, faithful Minervini system** — not
  the stripped momentum factor we'd tested before. It has the real 8-point Trend Template as a
  HARD gate (≥7/8), Stage 1–4 phase classification, VCP detection, breakout+volume confirmation,
  the SEPA fundamental leg (revenue/EPS/inventory), stop-losses + R:R≥2, sell signals, and a
  market-regime cash gate. It emits **hard buy/sell signals** (entry, stop, target, severity) —
  not just a watchlist. **But it is LIVE-SCAN-ONLY: it has no backtest** (the README lists
  backtesting as a future feature).
- **Reframed the prior "STOP" verdict.** `docs/MOMENTUM.md`'s Phase-0 STOP was on a *stripped*
  object: a continuous, **ungated** score, no fundamentals/stops/regime, tested as a market-neutral
  **L/S factor** on a **survivorship-biased large-cap** weekly panel. That does NOT condemn the real
  screener, which is long-only, position-based, hard-gated, with stops and a cash state — and which
  has never actually been backtested.
- **Vendored the screener into the repo** at `src/stock_screener/minervini_screener/` (27 Python
  files; the upstream `src/` library only — no tests/CI/scripts/notify entrypoints). MIT-licensed
  (© 2024 Ryan Hamby, upstream commit `397e555`); `LICENSE` + `PROVENANCE.md` kept. Only two
  documented edits, **no business logic changed**: `from src.` → relative imports, and a guard in
  `screening/__init__.py` so the pure rule logic imports without the live data-layer extras
  (SQLAlchemy/yfinance).
- **Built a new event-driven DAILY backtest harness** at `src/stock_screener/backtest_daily/`
  (14 modules). It replays history one day at a time, calls the vendored buy/sell rules through a
  single leak-safe slice (`ohlcv_upto(t)`), enters/exits with stops + sell signals + a regime cash
  gate, sizes positions risk-based (Minervini default), and reports CAGR/Sharpe/maxDD/CAPM α-β +
  screener stats (win rate, payoff, avg hold, % time in cash). Data source is abstracted behind
  provider interfaces, with a **synthetic provider** so it runs/tests today and a **stubbed WRDS
  provider** for the real pull.
- **Verified:** `python tests/test_backtest_daily.py` → **12/12 pass**, including the load-bearing
  `test_engine_decisions_leak_free` (a run on data truncated at D_mid reproduces a full run
  bit-for-bit up to D_mid), point-in-time fundamentals, universe/delisting forced-exit, stop/sell/
  cash behavior, cost & sizing identities, report-matches-helpers, reproducibility. `run_backtest.py`
  prints a coherent report on synthetic data (and incidentally shows the intended "go to cash in a
  bad tape" behavior). **These synthetic numbers are wiring proof, NOT a result.**
- **Built the WRDS ingestion** (`backtest_daily/ingest_wrds.py` + cache-reading `wrds_provider.py`
  + `cache_io.py`): a one-time `wrds`-package pull → parquet cache (`data/wrds/`) of the PIT top-3000
  universe (`crsp.msf`/`msenames`), daily CRSP prices (`crsp.dsf`, split-adjusted), delisting returns
  (`crsp.dsedelist`, Shumway-imputed), Compustat fundamentals via CCM (RDQ-lagged), and SPY. The engine
  consumes it through the same provider interfaces as the synthetic run (`--wrds` flag on
  `run_backtest.py`). **Verified on a fixture parquet/CSV cache** (`tests/test_wrds_provider.py`) — the
  providers satisfy the contract and the engine runs end-to-end. Deps added (`environment.yml`: `wrds`,
  `pyarrow`); `*.parquet` + `data/wrds/` gitignored. **The live pull itself is unrun here** (no
  `wrds`/credentials in the build env) — it is the user's step.
- **WRDS access regained.** (Corrects the earlier note that WRDS closed on graduation.) Universe
  decision for this venture: **broad, point-in-time top-N by market cap incl. small/mid-cap**, daily,
  2003→present.

## RESULTS (2026-06-29) — survivorship-free backtest run; NO out-of-sample alpha

- **Full dataset pulled** (top-3000-by-cap quarterly, 2003→**2024-12-31**; CRSP is annual-update so data
  ends 2024): **8,462** distinct names, **20.8M** daily price rows, **4,823 in-universe delistings** (the
  survivorship signal — survivor-only data would have dropped every one), **341K** Compustat rows. Sanity
  passed (Berkshire's $724k is real; delistings cluster at GFC + 2022–23). Added a **$5 floor on the
  UNADJUSTED price** (`raw_close`, so split-cheap winners like early AAPL aren't wrongly excluded) + a
  **despike** for cfacpr/penny artifacts.
- **Baseline (no fixes):** CAGR **3.90%** vs SPY 8.84%, Sharpe **0.29** vs 0.55, **maxDD −65.9%**,
  α +0.22% (t0.06). Underperforms badly. Artifact check clean (max daily +7%/−12%, max trade +420%) →
  the numbers are real, not data-contaminated.
- **−66% DD diagnosed:** (1) 2008 the book rode the crash ~100% exposed — the regime gate blocked new
  buys but `exit_on_regime_flip=False` meant NO liquidation, so it bled out on individual stops; and
  (2) 2009–10 / 2020 / 2023 whipsaw — re-entered choppy recoveries into failing breakouts while SPY ripped.
- **Two fixes → strong IN-SAMPLE result:** `exit_on_regime_flip=True` + `regime_confirm_days=25`. Full
  period: CAGR **11.1%**, Sharpe **0.72**, maxDD −40%, **α +7.98%/yr, t=2.49** — first significant positive
  in the whole project. *But the 25-day lag was chosen by looking at the full history's bad years.*
- **★ OOS test KILLED it.** Pre-registered: swept `confirm_days ∈ {0,15,25,40,60}` on TRAIN 2003–2013
  (best by Sharpe = **15**; a sharp peak, NOT a plateau — and 25 was the full-period *snoop*, not the
  train-best), then applied 15 to held-out **2014–2024**: CAGR 7.95% vs SPY 11.18%, Sharpe 0.51 vs 0.70,
  **α +2.18%/yr, t=0.47 (insignificant).** The t=2.49 was in-sample overfit. **No out-of-sample alpha.**
- **What survived OOS (real but modest):** the confirm-lag beat the no-lag (`confirm=0`) version OOS
  (Sharpe 0.22→0.51, DD −43.6%→−35.6%) — "wait for the market to confirm before re-entering" is a genuine
  *damage-control* rule. It's risk management, NOT alpha; the ceiling is lower-beta equity exposure. (Also:
  the OOS DD −35.6% ≈ SPY's −34.1%, so even the protection is regime-dependent — huge in 2008, marginal
  in the 2014–24 sharp-but-short selloffs.)
- **Engine notes:** ~45 min on the full 8,462×5,537 cache; a **cash-aware entry gate** (skip the scan when
  free cash <2% of equity) + progress logging made it tractable/observable. New `BacktestConfig` knobs:
  `min_price`, `exit_on_regime_flip`, `regime_confirm_days`, `min_free_cash_frac`, `progress_every`;
  `engine.run(cache=...)` reuses one cache across configs.

## 2. What went wrong (pitfalls hit, now resolved)

- **The momentum Phase-0 test measured the wrong thing.** It tested a *factor* (ungated, L/S,
  large-cap, biased data), then the "STOP" got mentally generalized to "Minervini doesn't work."
  It doesn't generalize — see §1. Lesson baked into §4.
- **Vendored package eager-loaded the live data layer.** `screening/__init__.py` imported
  `.screener → data.storage → sqlalchemy`, which isn't in the `ml-trading` env, so `import
  ...screening` crashed. Fixed with a `try/except ImportError` guard; pure rule modules now import
  on numpy/pandas alone.
- **Import-convention churn.** A save-time import-sorter enforces `from src.X` with the **repo root**
  on `sys.path` (matching `src/lib.py`'s `from src.ml_stock_prediction import indicators`). My initial
  `src`-on-path style kept getting rewritten (to broken relative forms). Resolved by aligning the
  whole harness to `from src.X` + repo root on path. **Don't fight the sorter.**
- **`pytest` not installed** in the `ml-trading` env → tests run as a plain script
  (`python tests/test_backtest_daily.py`), matching the repo's existing test style. The file is also
  pytest-compatible if pytest is ever added.
- **Minor:** the synthetic stop-loss test first fired a SELL instead of STOP because a −30% gap was
  shallower than the entry stop for a low-priced fill; deepened the crash so it unambiguously breaches.
- **★ The in-sample tuning trap (the big methodological lesson).** After the baseline failed, the
  regime-exit + a re-entry lag were added and `regime_confirm_days=25` was chosen by looking at the
  *full* history's bad years (2009/2020) — then scored on those same years, manufacturing a fake α
  t=2.49. The OOS test (re-fit the lag on 2003–13 only, evaluate untouched on 2014–24) exposed it
  (t→0.47). **Any positive result from a knob chosen after seeing the outcome is suspect until OOS-validated.**
- **Engine was unusably slow at first** (~26 min and projected indefinite): the entry scan re-ran the
  expensive VCP evaluation every 5 days even when fully invested, because capacity was checked by
  *position count* while risk-sizing fills cash at ~10 names. Fixed with the cash-aware entry gate.

## 3. What is planned / open

The live pull, the survivorship-free backtest, and the OOS validation are DONE (see RESULTS). The
"does Minervini beat the market" question is answered: **no out-of-sample alpha.** What's left:

- **★ Delisting-avoidance test — the one genuinely-open, orthogonal question.** Does the screen's
  *score* rank which names subsequently DELIST? Distress prediction is a *different* hypothesis that
  could be real even though the long-only strategy has no alpha — and only this survivorship-free
  dataset (**4,823 in-universe delistings**) can test it. The saved daily/blotter for the full runs are
  in `data/wrds/_bt_*.csv` (start from there). This is the highest-value next step for mining the dataset.
- **Bigger picture — where alpha could actually be (the user wants to keep working).** Selection AND
  timing on the efficient broad-US-equity universe are now exhausted: the ML model, the momentum factor,
  equal-weight, and now the *faithful* Minervini screener ALL return no OOS alpha; the durable residue
  every time is **risk management / low-beta exposure**, not alpha. To find real alpha, change the
  **signal or the arena** — a less-efficient niche (microcap / crypto / a domain the user knows better
  than the crowd), not a better-tuned rule on this universe.
- **Don't bother walk-forward-adapting `confirm_days`** hoping it becomes alpha (considered 2026-06-29):
  it's a *risk* knob, not a signal; the train optimum is unstable (15 on 2003–13 vs 25 on the full
  history, a sharp peak) so walk-forward would chase a moving target; and you cannot tune a risk dial
  into selection alpha the screen doesn't have. (If you want empirical closure, a *pre-registered*
  walk-forward over 2014–24 is ~2–3 hr — expected result: modestly better than fixed, still no α.)
- **Reconcile the buy threshold** (`is_buy = score>=60` vs docstring ≥70) via `BacktestConfig.buy_score_min` — minor, still open.
- **Optional robustness** if revisiting the risk overlay: reverse-direction OOS (fit late / test early)
  + a timing-vs-selection decomposition (apply the same regime overlay to a no-selection book, to
  confirm how much of any effect is just market timing vs the screen).

## 4. Things to remember NOT to do

- **Don't conflate the momentum Phase-0 "STOP" with the real screener.** Different object (ungated
  factor / L-S / biased large-caps). It does not preclude this venture.
- **Don't backtest the screener as a market-neutral L/S or a periodic top-N rebalance.** It is
  long-only, position-based, with stops + a cash state. Use `backtest_daily`'s event-driven engine.
- **Don't run it on top-250 large-caps only.** Minervini lives in small/mid-cap growth; the efficient
  large-cap arena is exhausted for selection (see §5). Use a broad universe.
- **Don't trust long-only breakout/momentum numbers on survivorship-biased data.** Buying breakouts
  to new highs is the *most* survivorship-sensitive signal there is; the +11.4% top30 momentum α was a
  survivorship + low-β artifact. The clean WRDS data is the whole point.
- **Don't let the vendored package pull the live layer into the backtest.** Import the pure rule
  modules only; never the `data/`, `notifications/`, `analysis/`, batch-processor, or `quant_engine`
  modules in the harness path.
- **Don't fight the import-sorter.** Use `from src.X import ...` and put the repo ROOT on `sys.path`.
- **Don't edit the vendored business logic.** `minervini_screener/` is third-party MIT — keep
  `LICENSE`/`PROVENANCE.md` accurate; only the documented import/guard changes are allowed.
- **Don't break the leak contract.** Every rule call must go through `cache.ohlcv_upto(t)` and
  fundamentals lagged to `rdq`; delisting is realized only on its date (an outcome, never a feature).
  `test_engine_decisions_leak_free` is the guard — keep it green.
- **Don't invert the momentum L/S** (from MOMENTUM.md): it's an insignificant null, not a significant
  negative — nothing to invert.
- **Don't trust an in-sample-tuned result — OOS-validate it.** Any knob chosen after seeing the data's
  bad years must be re-fit on a training split and tested on untouched data (recall t=2.49 → t=0.47).
- **Don't re-tune `confirm_days` (or any risk knob) chasing alpha.** It's damage control; the train
  optimum is unstable and the binding constraint is the *signal*, not the parameter — walk-forward won't fix it.
- **Don't cite the in-sample 11.1% / t=2.49 as a result.** The validated OOS number is 7.95% / t=0.47 (no alpha).

**Live-trading additions (2026-08-05, from the first paper month — each learned at real P&L):**

- **Don't batch-enter.** Six positions in four minutes (2026-07-27) = one bet on that day's tape;
  progressive exposure runs the other way — 1-2 pilot buys, add only after banked wins. Being ten
  positions deep on an 0-for-4 batting average is unearned exposure by definition.
- **Don't act on an intraday trigger.** The cadence is EOD-confirm for a reason: PEBK's 15:27
  intraday trigger faded to a settled close BELOW the pivot the same day (2026-08-04) — bought at
  15:58 anyway → instant failed breakout. The half-hourly reports are context; the close decides.
  (The gate's track record when respected: EBAY 0.8× and XMTR 0.9× quiet crosses refused, both then
  fell; WST's −15% earnings reaction dodged entirely — all 2026-07-20..24.)
- **Don't chase past pivot×1.05.** PKG filled +5.4% above its pivot and immediately gave back ~4%
  with no support underneath; extended entries attack both the win rate and the R:R at once.
- **Don't open inside the ~21-day earnings window.** STRW was bought 14 days before its report at
  double intended size — forced to trim at slippage cost, then to exit before the print.
- **Don't size micro-caps without an ADV look.** STRW: a ~$100k order moved the tape 1.5% on an
  instant correction flip (~12% of its ADV); ICCC/PEBK carry the same exit-slippage risk.
- **Don't design around a "known" API constraint that was never verified live.** "Alpaca market
  OTOs can't be GTC" was assumed for a month and was simply false (§6.38) — the disproof was a
  five-minute canceled 1-share probe; the assumption cost an unprotected overnight position.
- **Don't let tests depend on wall-clock/market state.** §6.37's settled-cache gate would have
  broken every top-up test run after hours or on weekends; the negative-`max_age_days` sentinel
  exists so the suite is deterministic at any hour. Any future time-coupled feature needs its
  explicit test bypass designed in, not discovered.

---

## 5. Carried over from `docs/MOMENTUM.md` (the momentum-FACTOR experiment — Phase 0, CLOSED 2026-06-04)

The original momentum track ported Minervini *ideas* (Stage-2 trend template, RS slope, distance from
52w high, breakout/volume) into continuous scorers and tested them through the weekly `backtest_lib`
engine as a factor. **Staged plan:** Phase 0 (cheap large-cap gate) → Phase 1 (regime/drawdown overlay)
→ Phase 2 (broad small/mid-cap) → Phase 3 (stops + live). Phase 0 was the pre-registered gate.

**Phase 0 verdict — STOP (no market-neutral selection α on large-caps).** Same 240-ticker weekly panel,
2018→2026, equal-weight, quarterly, `spread_per_share=0.02`, seasoning 52w. Wiring validated
(`universe_ew` Sharpe **0.863**, α +3.6% t1.25). Decisive beta-neutral L/S (classic 12-1 AND screener
composite): every hedged α small-negative and **insignificant** (|t|<0.9), gross α≈0 (|t|<0.35) → not a
turnover artifact, the ranking carries no market-neutral info. Unlike the ML L/S (significantly negative,
t≈−2.3, inverted), momentum's is an **uninformative null**. Long-only context: classic top30 Sharpe
0.825 / CAGR 20.9% / α +11.4% but **t1.58 (insignificant)**, β0.86; top100 0.868 / α +4.7% t1.29 — all
just reproduce the low-β/EW bar, and the +11.4% is survivorship-suspect.

**Still open (and now in scope for this venture with clean data):**
- **Phase 2 (broad small/mid-cap)** — a different, survivorship-fraught arena; the legitimate next bet,
  not a continuation of large-cap evidence.
- **Phase 1 (regime/drawdown overlay)** — cheap, orthogonal to selection; cheapest first test is whether
  the aggregate signal / `Bull_Prob` predicts forward SPY at all.

Factor-experiment code (separate from this venture): `src/stock_screener/momentum_lib.py` +
`tests/test_momentum_lib.py`; runner `scripts/run_momentum_phase0.py` (gitignored scratch).

---

## 6. SEPA Cockpit — human-in-the-loop tool (2026-06-29, NEW, working)

A local **Streamlit** app that runs the SEPA funnel as decision support, not automation. Lives in
`src/stock_screener/cockpit/` and is fully separate from the WRDS backtest (live yfinance data, not CRSP).

- **What it does:** scan a universe (S&P 500 first; `get_universe()` is pluggable for full-US) →
  **Step 1** hard gate via the vendored `validate_minervini_trend_template` (full **8/8** criteria) → RS rating
  (percentile of trailing-6mo return across the scanned set) → **Step 2** fundamental highlight (rev/EPS
  YoY/QoQ + margins from yfinance, green/amber checks) → **Step 3** `detect_vcp_pattern` hint → **Step 4**
  advisory entry levels (pivot / no-chase buy-zone / 7–8% stop / 20–25% target / "breakout today on Nx
  vol") + position sizing. A regime/breadth banner (`analyze_spy_trend`/`calculate_market_breadth`/
  `should_generate_signals`) gates the discipline. The user judges the VCP and places orders themselves.
  *(This §6 intro is the 2026-06-29 snapshot; nearly everything in it was reworked by §6.1–§6.39 —
  trust the later sections where they differ. Headlines: the recall-first tier detector replaced the
  raw `detect_vcp_pattern` hint (§6.1), `full_us` is now the ONLY universe (§6.32 item 24), the
  watchlist/triggers/positions/journal machinery of §6.11–§6.27 didn't exist yet, and the suite is
  now 95 tests, not 9.)*
- **Run:** `streamlit run src/stock_screener/cockpit/app.py` (from repo root). First S&P-500 scan ~10-30s
  (prices ~10s + fundamentals ~0.5s/passer); same-day re-scans ~1-2s off the `data/cockpit/` cache
  (prices parquet, ~1-day staleness; fundamentals JSON, ~7-day staleness). Table is click-to-chart with a
  ticker search box.
- **Reuse vs new:** reuses ONLY the pure rule functions from `minervini_screener/screening/` (they import
  on numpy/pandas; re-exported via the guarded `screening/__init__.py`). Does **not** touch the vendored
  `data/` package — its `__init__` eager-loads SQLAlchemy (absent from `ml-trading`), which would crash;
  we wrote a thin yfinance layer instead (`cockpit/data_feed.py`). No vendored edits.
- **Gotcha fixed:** concurrent single-ticker `yf.download` calls RACE on yfinance's shared global state and
  return the *wrong ticker's* data. `get_many_prices` uses yfinance's own batch download
  (`group_by='ticker'`, internal threads) in chunks instead. Don't reintroduce a ThreadPool over per-ticker
  `yf.download`.
- **Help/UX:** every step has an **ℹ️ popover** ("how to use") + a **SEPA Guide** page
  (`cockpit/pages/1_SEPA_Guide.py`) that renders `minervini_sepa_system.md` live (single source of truth),
  linked from the sidebar. Table is click-to-chart with a ticker search box; the chart has a
  **time-range zoom** (3M-2Y) that scopes candles + y-axis to the window (SMAs stay full-history) so a
  tight VCP base is actually visible.
- **Tests:** `python tests/test_cockpit.py` (9/9 as of 2026-07-02) — offline synthetic-fixture funnel, chart-figure asserts,
  Step-2 logic, a subprocess check that `cockpit.data_feed` never imports the vendored `data/` layer, and
  Streamlit `AppTest`s that render both app.py (run_scan patched to an offline result) and the Guide page.
- **Limits / next:** yfinance often exposes only ~4 quarters, so Step-2 YoY can be n/a (QoQ used as
  fallback); breakout email/Slack alerts (vendored notifiers) remain out of scope. (full-US universe:
  DONE 2026-07-02 — see below.)

### Cockpit updates — 2026-07-02 (chart hints, RMV, full-US universe, incremental cache)

Six cockpit enhancements this session (all in `cockpit/`, no vendored edits; tests now **12/12**):

- **VCP contraction hover.** The orange contraction shading (`add_vrect`) is a layout *shape* and can't
  carry a tooltip, so `charts.py` overlays an invisible hoverable scatter across each contraction's span;
  under `hovermode="x unified"` it surfaces depth / peak→trough / duration / volume-ratio on hover.
- **RMV (Relative Measured Volatility) — new indicator + pane.** `cockpit/indicators.py` (kept OUT of the
  vendored screening package — RMV isn't upstream): true-range-as-%-of-price, smoothed, min-max normalized
  over a trailing window to 0–100 (**low = tight base**). Rendered as a **3rd chart subplot** (volume pane
  compressed) with a shaded `<25` "tight zone", and shown in **Step 4 as an advisory tile** (tight/normal/
  loose). **Display only — it does NOT move the pivot/stop/target math** (the cockpit hints, never decides).
  Note RMV (price tightness) is orthogonal to the breakout's *volume* surge — they don't conflict.
- **`full_us` universe wired (~3–4k names).** `get_universe("full_us")` (was a `NotImplementedError` stub)
  now re-implements the upstream `minervini_screener/data/universe_fetcher.py` **inside** `data_feed.py`
  (can't import it — that package eager-loads SQLAlchemy): fetches NASDAQ Trader `nasdaqlisted.txt` +
  `otherlisted.txt` over **HTTPS** (upstream's `ftp://` is commonly blocked), drops test issues + ETF-flag
  rows + the footer, filters to clean common stock (`^[A-Z]{1,5}$`, warrant/right/unit issues dropped only
  when they match the nasdaqtrader `^[A-Z]{4}[WRU]$` base+suffix shape, no fund names). Cached to
  `data/cockpit/us_common_universe.csv` (1-day). ~~Selectable in the sidebar; sp500 stays the default~~
  **[since §6.32 item 24: `full_us` is the ONLY universe — the selectbox is gone; sp500/tickers
  fetchers remain programmatic].** **Known limitation:** dotted class shares (BRK.B/BF.B) are dropped, so
  full_us differs slightly from sp500 (which keeps BRK-B).
  **Fixed 2026-07-17:** the old plain `(?:W|R|U)$` suffix drop was unanchored and silently removed every
  ordinary 4-letter name ending in W/R/U (PLTR, SNOW, UBER, LULU, TROW, DOW, LOW, EMR, KR…) and single-letter
  `U` (Unity) from the *only* enabled discovery universe — exactly the high-RS leaders the screen targets.
  Now anchored to the 5-char base+suffix shape (`^[A-Z]{4}[WRU]$`), so those names are kept while genuine
  SPAC warrants/rights/units (base+W/R/U, e.g. `CVIIW`/`CVIIU`) are still dropped; a 3-char-base warrant
  (4 chars, e.g. `ABCW`) can slip through but just fails the trend template. Regression covered by
  `test_get_universe_full_us_offline`.
- **Incremental price cache.** `get_many_prices`/`get_prices` no longer discard a stale parquet and re-pull
  2y — they partition tickers into **fresh** (use cache) / **incremental** (fetch only bars since the last
  cached date, one shared `start=`) / **full** (cold, or gap >`max_gap_days`=10 → re-baseline);
  `_merge_incremental` appends dedup'd. **Split/dividend hazard handled:** `auto_adjust=True` re-adjusts
  history on a corp action, so a naive append would splice two adjustment bases — if the overlap days
  diverge >0.5% (`SPLIT_TOL`) it forces a full re-baseline. This is the real rate-limit mitigation:
  steady-state scans fetch a handful of new bars, not 2y × ~4k.
- **Throttle/retry.** `_download_batch` adds bounded retry + exponential backoff (0.5s, 1.0s) and a 0.5s
  inter-batch pause so a cold 3–4k scan isn't rate-limited into silently-dropped batches.
- **VCP contraction detector rebuilt (`cockpit/vcp.py`, NEW).** The vendored `detect_vcp_pattern`
  collapses on a broad universe (**`cc=0` for 84% of the 546 `full_us` candidates** — bad base
  anchoring, drops the last contraction, mis-matches peaks↔troughs). Replaced in the cockpit only
  (vendored untouched, edit-restricted) with a volatility-adaptive **ZigZag** detector (strictly-
  alternating H/L pivots, base = every contraction under a flat top; `is_vcp` judges tightening
  separately). Same dict schema = drop-in. **This first cut was then reworked into the recall-first
  TIER system — see §6.1 for the current detector; the threshold/sanity-gate tuning here is superseded.**

**Rate-limit reality (honest, for whoever tunes this next):** the backoff is only 0.5s/1.0s then it gives
up on the batch and moves on — far shorter than Yahoo's actual 429 cooldown (tens of seconds to minutes),
so it does NOT recover a *sustained* limit; it fails politely and those ~100 names retry next scan
(partial batches don't retry at all — any non-empty result returns immediately). The durable fix is the
incremental cache; the first cold scan of ~3.7k names is the only real exposure. **Recommended next:** an
adaptive cooldown (detect an all-empty batch → sleep 30–60s once, or bail with a clear message), and/or an
**Alpaca daily-bars backend** for the cold full_us scan (keyed API instead of scraping Yahoo — the user
already uses Alpaca for the auto-trader). IP/proxy rotation was raised and rejected: ToS-violating,
fragile, and the wrong tool for periodic personal research.

**Files touched:** `cockpit/` — `charts.py` (hover + RMV pane), `indicators.py`/`vcp.py` (NEW),
`data_feed.py` (full_us + incremental cache + throttle), `scan.py`, `app.py`; +5 offline tests.

### 6.1 Recall-first detector rework + 200-chart benchmark (2026-07-04)

**Reframe.** The detector's job changed from "is this a VCP? yes/no" to a **safety net that
never drops a live setup**: the user reviews a shortlist knowing misses are impossible-by-test,
and false alarms only cost a glance. Driven by two blind audit sessions (40 charts) that found
the old gate missed real setups (VRA, SMBC) while flagging junk (EQ, ELSE, TWO).

**Benchmark (the durable asset).** `tests/fixtures/vcp_bench/` = 200 price histories frozen at
2026-07-03 (git-committed parquet; `CON` stored as `CON_.parquet` — Windows reserved name);
`tests/vcp_labels.py` = 200 hand labels (**72 YES / 128 NO**) judged **blind** from raw
weekly/daily OHLCV tables with no detector output visible, each with a one-line structural
reason. YES = live/actionable setup (forming / at / within the buy zone of its pivot);
textbook-but-spent bases (already ran 15%+ past the pivot) are NO. Note the vintage: the label
date sat in a broad breakout wave (small-bank bases everywhere), hence the high YES rate.

**Detector changes (`cockpit/vcp.py`), each validated on the benchmark:**
- **Multi-threshold detection** (the VRA/SMBC fix): a VCP by definition ends *quieter* than the
  stock's history, so one history-calibrated ZigZag threshold goes blind at the tight ending.
  Now runs at up to 4 thresholds — long-history, recent-window (~2 mo), 0.7× recent (floor 2%),
  and a fixed 3.5% (the recent window gets polluted by the breakout burst itself: WERN's
  "recent" read 9.6% while its coil legs were 5–7%) — best read wins (strict pass ▸ tier ▸ quality).
- **RMV veto is conditional**: vetoes only while price is *below* the pivot (a breakout IS a
  volatility burst — RMV read 100 on SMBC mid-breakout). Cutoff 25→**30** (sweep: rescues the
  25.04-boundary class (JAKK/CDNA) for ~1 false alarm; past 30 each rescue costs ~2.5 junk).
  Removing the below-pivot veto entirely was measured and REJECTED: +6 real / +14 junk.
- **Sanity rules restored** (dropped in 133f53a) **with benchmark-recalibrated details**:
  leg ≥ **2** bars (not 3 — quiet climbers have genuine 2-day final shakeouts, RNST class;
  1-bar gap legs are still junk anchors), base ≥ **2.0** weeks (not 3 — length is measured over
  the *selected* legs, which under-reads: VRA's ~6-week base measures 2.1), newest leg ≤ 13
  weeks (TWO/MNST stale class), **dead tape** = median daily true-range% over last 42 bars
  < **1%** (median, not max — KORE's one pop-day defeats a max rule; calibration: all 7
  deal-zombies ≤ 0.95%, quietest real setup (SPG) 1.64%). Dead-tape only runs in adaptive mode
  (pinned `thr=` keeps synthetic H=L=C tests deterministic).
- **Tightness**: final ≤ 12% AND (≤ 0.8× first leg OR ≤ 6.5% absolute — uniform quiet shelves
  (TRIN 4.4→3.8%) can't shrink 20% further but ARE tight).
- **Tiers replace the lone boolean** (schema adds `tier`, `tier_reason`, `pivot_price`,
  `zz_threshold`; `is_vcp` unchanged for compat): **A** = valid base within −10%..+10% of the
  detected pivot (+10% ≈ hand-labeled "≤5–8% past the *real* pivot" because the detected pivot
  sits low; −10% floor kills the falling-away-from-base class, EQ/WSC); **B** = forming or
  extended, never hidden; **C** = safe exclusions only (dead tape / no pullbacks at any
  threshold / stale base), reason recorded. `scan.py` sorts A-first-by-quality; `app.py` shows
  Tier + Tier reason columns.

**Benchmark results** (enforced by `test_vcp_benchmark_200_charts`): **A = 79** (53 YES,
precision 67%), **B = 114** (19 YES — reviewable, sorted right below A), **C = 7** (**zero
YES** — the hard never-miss contract; also asserts A-recall ≥ 45). Old detector on the same
labels: missed 2/11 YES outright and only ~9/19 flags were real.

**Honest budget note.** Live `full_us` (2026-07-04): 767 template passers → **A=314 / B=430 /
C=23**. The "sift ~40" target assumed a normal tape; on THIS tape ~36% of passers are genuinely
live setups by the same hand-standard (72/200 blind-verified), so a recall-first A-list is
necessarily ~hundreds. A is quality-sorted — walk it top-down as far as time allows; on a
quieter tape A shrinks naturally. Squeezing A below the true setup count = reintroducing misses.

**Five shape regression tests** pin the failure modes (`test_vcp_multi_threshold_sees_quiet_
taper_after_loud_history`, `_finds_tight_final_leg_after_wide_start`, `_two_day_spike_is_not_
a_base`, `_deal_pinned_stock_is_dead_tape`, `_extended_breakout_is_watch_not_review`), and
`test_rmv_gate_vetoes_below_pivot_only` replaces the old always-veto RMV test (which encoded
exactly the SMBC bug). Suite: **21/21 offline.**

### 6.2 Watchlist, CSV/txt export, Alpaca paper-trade, layout (2026-07-05/06)

A batch of cockpit workflow features (commits `25fc318`, `cc1a96b`, `c70c1fa`, `5d2babb`).
Suite now **25/25 offline**.

**Watchlist (per-session shortlist).** Canonical store is a plain ordered list in
`st.session_state["watchlist"]` — deliberately NOT a widget key, so button callbacks and the
picker/uploader can all mutate it without fighting over widget ownership. Add a name three
ways: the ⭐ button under any chart (`wl_toggle`), a sidebar multiselect (`_wl_add_from_picker`
on_change), or a `.txt` upload. It's session-scoped — the *export* is what persists it.

**Export/import (`cockpit/export.py`, pure + unit-tested).**
- `watchlist_list_csv()` → the shortlist with its decision columns in add-order; a stale pick
  (from another universe) survives as a ticker-only row, never silently dropped.
- `watchlist_ohlcv_csv()` → long-format daily OHLCV for every watchlisted name, `Date,Ticker,…`.
- `parse_ticker_list()` → tokenizes an uploaded `.txt` on **commas AND any whitespace/newlines**,
  upper-cases, drops blanks, de-dupes keeping first-seen order. Backs the `.txt` uploader; the
  `.txt` download (`",".join`) is its exact round-trip.

**Alpaca paper-trade from the watchlist (`cockpit/trade.py`, NEW).** Two-step preview→submit
in the sidebar. Reuses the existing `portfolio_experimentation/alpaca_trader.py` primitives
(`validate_tradable`, `MAX_ORDER_PCT` 10%-equity cap) but has its **own** connection so it can
target a *different* paper account.
- `build_buy_plan(tickers, payloads, *, mode, amount, equity=None, asof=None, max_bar_age_days=None)`
  *(later gained `pivots` — §6.22 — and `held` — §6.39)*
  — pure, unit-tested. Four sizing modes per name: `"pct"` (% of equity, needs equity),
  `"dollars"` ($ each), `"shares"` (explicit count), `"risk"` (risk-to-stop, §6.7). The $50 order
  floor applies to the dollar-denominated modes only — `"shares"` is exempt. Skips names not in
  the scan, no price, <1 share, or (dollar modes) sub-$50. `extended` flags price > pivot×1.05.
  When `max_bar_age_days` is set, also skips a name whose freshest bar is > that many *trading*
  days old (staleness guard, §6.20).
- `freshen_prices(tickers, payloads)` (NEW, §6.20) — re-pulls the watchlist names' latest bars
  (cheap incremental top-up, `max_age_days=0`) and overlays them so the plan sizes on current
  prices, not the scan memo's possibly days-old closes; the app calls it at Build.
- `submit_buy_plan(plan)` — market BUYs (protective-stop attach added §6.3; the whole OTO is
  GTC since §6.38), one bad symbol doesn't abort the rest; returns
  `{equity, cash, account_number, using_dedicated, results}`.
- `_connect_paper()` — **the account-selection mechanism.** Each Alpaca paper account has its
  OWN key pair, so we pick the "Minervini Trader" account via its dedicated keys, falling back
  to the shared pair (which the All-Weather mirror owns — don't hijack those). Always
  `paper=True`. **[Env names corrected §6.3/§6.23 — the canonical spellings actually in `.env`
  are `ALPACA_API_KEY_MINERVINI` / `ALPACA_API_KEY_SECRET_MINERVINI` (shared fallback
  `ALPACA_API_KEY_PAPER1`/`ALPACA_API_SECRET_PAPER1`); the `ALPACA_MINERVINI_API_*` names this
  section originally documented never existed.]**
- **Alpaca facts learned:** `ALPACA_BASE_URL` in `.env` is **unused** — nothing reads it; it's a
  doc comment. `TradingClient(key, secret, paper=True)` derives the endpoint from `paper=True`
  and the SDK appends `/v2/…` itself, so a `/v2` on the base-URL line is irrelevant (and you'd
  never pass `/v2` to `url_override`). The API exposes the account *number*, not the dashboard's
  friendly name, so the UI confirms with the number's last 4 digits — eyeball that vs the
  dashboard once.

**Layout reflow (`cc1a96b`).** Step 2 (fundamentals) moved out of a full-width box into the
right column, stacked above the Step-3 controls, beside a large chart on the left
(`st.columns([3,1])`). Step 2 condensed to 3 lines (Rev/EPS YoY·QoQ, Op-margin, the 4 checks +
score) so the two side boxes are ~equal height. **Ordering gotcha:** the side column is written
FIRST in code (so the chart-control checkbox values exist before `build_chart`), but still
renders on the right because it's the second column returned — Streamlit lays out by creation
order, not write order.

**MISTAKE — reverted: right-arrow row navigation** (`components.html` keydown → hidden-button
hack). **Don't re-attempt** unless replacing `st.dataframe` with a custom selection UI: the
parent-DOM injection is version-brittle AND the dataframe's blue row-highlight can't be moved
programmatically, so it desyncs from the chart regardless. (Scroll-preservation via
sessionStorage died in the same revert.)

**AppTest gotchas (bit us writing tests):** `at.session_state` supports item access + `in` but
**NOT `.get()`/`.setdefault()`**; widget refs go **stale after each `.run()`** (re-query
`at.button`/etc. every time); there is **no `at.download_button` / `at.file_uploader`** accessor
— cover those paths by asserting the app reruns without raising, and test the pure helpers
directly (why export/trade parsing/sizing live in `export.py`/`trade.py`, not inline in `app.py`).

**Decisions / guidance given this session (not code — record so it isn't re-litigated):**
- **Don't hard-gate on `min_fund` (fund_score).** The checks count a *missing* (n/a) value as a
  fail, and yfinance data is patchy — so `min-fund ≥ 2` silently drops names with *thin data*,
  not just weak ones (same never-miss failure mode as the VCP gate). Use fund_score to **sort /
  size conviction**, not to filter. Shrink the review list with **Tier A + RS** instead (RS is
  price-based, never n/a). If a real fundamentals gate is ever wanted, first exempt names whose
  fundamentals are `available == False` (the flag exists) so it only cuts *present-and-weak*.
- **Prioritization of Steps 2–4 is gates + a ranking, not a weighted average.** Regime (master
  switch) → valid base / Tier A (Step 3, the discretionary edge) → live trigger (Step 4) are
  necessary gates; fundamentals (Step 2) rank conviction & size *among* names that pass. Great
  earnings never justify buying an extended/non-triggering chart.
- **Cadence:** hunt weekly (bases form over weeks — a weekend session), check triggers + open-
  position stops **daily at end-of-day** (yfinance daily bars only finalize after the close;
  midday reads are stale). Regime dials frequency down in a weak tape. The daily habit that
  matters most is **stop management on open positions**, not re-hunting.
- **`pct_to_pivot` sign convention:** negative = price ABOVE the detected pivot (into/past the
  buy zone); positive = BELOW it (not triggered). Tier A spans −10%..+10% of the *detected*
  pivot (which sits low vs the true pivot). Sweet spot ≈ 0 to −5%; deeply negative = chasing.
- **RMV is the base-tightness discriminator — always pull the chart before endorsing.** The
  scan's "tightening legs" + fund_score can flatter a *loose* base. Live example this session:
  RSI had the best fundamentals on an 82-name list but **RMV 59** (wide 8–19% weekly ranges) —
  the volatility never contracted; only pulling the chart caught it. The clean fuel-plus-tight-
  base names were ECPG/PGC/FTNT (RMV 25/31/21); ARMK had the tightest base (RMV 12) but the
  weakest fuel — setup and fuel don't always come together.

### 6.3 Auto-attach protective STOP on paper submit + credential-name fixes (2026-07-07)

Suite now **29/29 offline**. Two credential bugs fixed, then the buy+stop feature added.

**Credential-name bugs (were silently mis-routing orders — fixed first).** The paper-trade path
was reaching **no** account: `.env` names the keys `ALPACA_API_KEY_MINERVINI` /
`ALPACA_API_KEY_SECRET_MINERVINI` (and the shared/All-Weather pair `ALPACA_API_KEY_PAPER1` /
`ALPACA_API_SECRET_PAPER1`), but the code read `ALPACA_MINERVINI_API_KEY` / `_SECRET` (and bare
`ALPACA_API_KEY`) — **none of which exist** — so `_connect_paper()` raised `TradeUnavailable`
and Submit failed into `{"error": …}`. Fix: `trade.py` now resolves each credential from a list
of accepted spellings via a small `_first_env(names)` helper; `alpaca_trader.connect()` likewise
falls back to the `*_PAPER1` names. **Correction to §6.2's "Setup required":** the dedicated keys
are read from the `ALPACA_API_KEY_MINERVINI` / `ALPACA_API_KEY_SECRET_MINERVINI` names actually in
`.env`. (An earlier version of this note claimed BOTH those and the `ALPACA_MINERVINI_API_*` form
were accepted; the phantom form was never in `.env`, so per §6.23 the single canonical spelling is
used and the docs corrected to match.) Landmine noted: `_first_env` must treat a bare `str` as ONE name
(an intermediate edit passed a plain string; `for n in names` then iterated *characters*, matched
the one-char env var `$_` = python's path, and auth'd with garbage → 401). It now wraps `str` →
`(str,)`. Verified live: reaches Minervini paper acct `…SZOE`, `using_dedicated=True`.

**Feature: every submitted buy gets a protective stop (`trade.py`, `app.py`).** Enforces the
SEPA "never hold without a stop" rule directly from the cockpit. Per watchlisted name at submit:
- **not held** → market BUY as an Alpaca **OTO** order carrying a `StopLossRequest` leg
  (`MarketOrderRequest(order_class=OrderClass.OTO, stop_loss=StopLossRequest(stop_price=…))`).
  Chosen over "buy then a separate stop" because you can't place a sell stop for shares you don't
  yet own, and a buy submitted after the close is queued to next open — OTO defers the stop
  leg until the primary fills, so it attaches atomically regardless of market state. **[TIF is
  GTC end-to-end since §6.38 — the original DAY OTO let an intraday fill's stop leg expire at
  that day's close.]**
- **already held** → NO buy; place a standalone DAY `StopOrderRequest` to sell the WHOLE held
  position, first cancelling the ticker's open sell stop(s) so exactly one stop remains (replace).

**User decisions (locked, don't re-litigate):** stop price is **editable, defaulting to the
app-computed `levels["stop"]`** (~7-8% below pivot) — a per-ticker `number_input` in the plan
preview keyed `stop_{ticker}_{build_nonce}`; stop lifetime is **`TimeInForce.DAY`** (his choice —
so it expires at the close and must be re-armed each session; overnight/gap risk is uncovered);
existing stop on a held name is **replaced**. **[SUPERSEDED 2026-07-09 — §6.6: held-position
stops are now GTC and replaced only to RAISE them, never lower (Minervini ratchet).]**

**Key mechanics / API facts (alpaca-py 0.43.4):**
- `build_buy_plan` stays pure and now carries `stop_price` per entry (or `None`). New pure
  `stop_is_valid(stop, price)` = `0 < stop < price` (Alpaca rejects a sell stop at/above market
  and an OTO stop-loss leg not below entry). Used in the UI (live red-flag) AND re-checked in
  submit against the last close.
- `submit_buy_plan(plan, *, attach_stop=True)` fetches `held = {sym:int(float(qty))}` from
  `get_all_positions()` (mirrors `get_account_state`), branches per name, keeps the per-ticker
  `try/except` (one bad symbol never aborts the batch). New result status `"stop_only"` (🛑);
  success line counts `submitted`+`stop_only`.
- Replace = `get_orders(GetOrdersRequest(status=OPEN, side=SELL, symbols=[t]))` → cancel those of
  type STOP/STOP_LIMIT/TRAILING_STOP (leave manual limit sells alone), each cancel guarded. With
  no `nested`, a triggered OTO stop leg surfaces as its OWN top-level SELL order, so this flat
  query catches both standalone stops and prior OTO legs.
- **OTO / STOP legs require whole-share qty** (we already floor everywhere). `OrderClass` has
  `SIMPLE/OTO/BRACKET/OCO` — **no OTOCO**. `client_order_id` conventions: `SEPAoto-` (buy+stop),
  `SEPAstop-` (held stop), `SEPAcockpit-` (naked buy, toggle off); millisecond timestamps avoid
  duplicate-id rejects on fast resubmit.
- **Guardrails:** buy leg keeps `MIN_TRADE_USD` (build) + 10%-equity cap (submit); the held
  stop-only path is **exempt** from both — a protective stop is risk-reducing and must never be
  blocked by a size floor/cap.

**Known limitation (flagged to user).** A held name only reaches the stop-only path if it
survived `build_buy_plan`'s buy-sizing; if its buy would round below $50 (dollar/% modes) or <1
share it's dropped at build and gets no stop. Normal sizing (5% equity, or a share count) never
hits this. `build_buy_plan` is pure/network-free so it can't see holdings — fixing this means
threading holdings in or always-including watchlist names. ~~Deferred.~~ **Closed 2026-08-05 —
§6.39** (optional `held` map → zero-share `stop_only` rows).

**Other server-side risks (can't unit-test).** A pre-breakout watchlist name trading BELOW its
pivot has `levels["stop"]` ABOVE current price → invalid sell stop (UI flags, submit skips) —
common, since the default stop sits below the pivot. A gap between last close and live price can
flip a stop we accepted to a server reject → surfaces as `failed`.

**Testing pattern.** `submit_buy_plan` is now unit-tested offline with a hand-rolled `FakeClient`
+ `monkeypatch trade._connect_paper` — the fake records the request objects, and the test builds
the REAL alpaca-py OTO/stop requests, so the "will the server-shape construct" risk is covered at
the client layer (tests `test_submit_buy_plan_stop_logic`, `test_stop_is_valid`,
`test_trade_plan_preview_renders_stop_controls`). **NOT yet verified:** a live paper submission
(would place real orders in `…SZOE`) — left for the user.

### 6.4 Earnings-date awareness (2026-07-07)

Suite now **30/30 offline**. Encodes Minervini's "never open a fresh, cushion-less position
into an earnings report" rule as data + advisory flags — the first item off the workflow-review
backlog in §6.5 (which records the rest, ranked).

- **Data:** `data_feed._next_earnings_date(tk)` reads `yf.Ticker.calendar` (dict on modern
  yfinance — verified live: AAPL/NVDA/PGC all dicts; DataFrame shape of older versions also
  handled; earliest date of Yahoo's 2-day window). Stored as `next_earnings` ('YYYY-MM-DD')
  in the fundamentals JSON — added AFTER the `_jsonable` float-coercion (which would None a
  string out). Same `yf.Ticker` object as the statements = no extra network pass. Cached
  fundamentals missing the KEY refetch once (schema upgrade); a present-but-None stays cached.
- **Scan:** `scan._days_to_earnings(fund, today=None)` → calendar days to the report
  (negative = just reported — Yahoo can list the last date until the next is scheduled;
  None = unknown). `today` overridable for deterministic tests. Emitted as an `earnings_in`
  column (Entry group, first) and a top-level payload key.
- **UI (`app.py`):** `EARNINGS_SOON_DAYS = 21` + `_earnings_flag(days)` ('⚠︎ earnings in Nd'
  only for 0..21). Surfaced in: the table column (+ header tooltip), the Step-2 box
  ("**Earnings:** 2026-07-30 (in 23d)"), the trade-plan preview (per-order flag + footnote),
  and INFO_STEP2/INFO_STEP4. Advisory only — nothing is gated/skipped on it, per the
  recall-first philosophy.
- **Trade:** `build_buy_plan` copies `payload["earnings_in"]` into each plan entry untouched
  (stays pure/date-free).
- **Guide:** `minervini_sepa_system.md` Step 4 gains the earnings-calendar bullet + checklist
  item (the in-app SEPA Guide page renders this file).
- **Test:** `test_earnings_date_plumbing` — calendar parsing (dict/frame/empty/raising),
  day-count math (pinned today), funnel integration, plan carry-through.

### 6.5 Cockpit backlog — workflow review (2026-07-07, ranked)

Full fidelity/efficiency review of the cockpit against Minervini's SEPA process (books + the
in-repo guide). Verdict: the buy-side funnel is solid; nearly all gaps are **after the buy
button**. Framing note (aligns with the backtest verdict): the durable residue of every
experiment is risk management, not selection — so the risk/process items below are the
highest-leverage work, not further detector tuning. Item 1 was earnings-date awareness (DONE,
§6.4). Remaining, in recommended order:

1. **[DONE — §6.7]** Risk-to-stop sizing mode (+ stale docstring fix).
2. **[DONE — §6.8]** Stop-to-pivot clamp (10% hard max from the pivot).
3. **[DONE — §6.9]** Positions page (holdings, advisories, one-click GTC stop re-arm).
4. **[DONE — §6.13]** Trade journal from the SEPA client_order_id tags.
5. **[DONE — §6.14]** EOD trigger check on frozen pivots (option (a), a resting buy-stop
   at the pivot, was rejected — it can't check breakout volume).
6. **[DONE — §6.15]** Regime warning at the point of action (Build-plan panel).
7. **Fidelity details (cheap, independent):**
   - **[DONE 2026-07-14 — §6.17] RS rating**: plain 6-mo return percentile → IBD-style
     weighted multi-horizon (2×3-mo + 6-mo + 9-mo + 12-mo). The optional "RS line at new
     high before price" chart flag was ~~NOT built (still open, low priority)~~ **built
     2026-08-05 — §6.39**.
   - **[IMPLEMENTED THEN SHELVED 2026-07-14 — see `docs/edgar_backfill_shelved.md`]
     Fundamentals depth**: annual EPS growth, 3-quarter acceleration, surprise % — full
     working code + tests preserved in the (gitignored) shelf doc, held back from the
     working tree at the user's request ("save the edgar stuff for another commit").
   - **[DONE 2026-07-14 — §6.17] Volume bars colored by up/down day** in `charts.py`.
8. **Small bugs / polish (from the review):**
   - **[HISTORICAL — the freshness model described here no longer exists.]** As reviewed
     (2026-07-11), same-day prices were served from a rolling-24h mtime window (no top-up) and
     the Re-scan button force-refreshed only every OTHER click (`nonce % 2` bug). The parity
     bug was fixed in §6.10; the 24h window itself was then **replaced entirely** — §6.25
     (always-incremental top-up, Re-scan = top-up, Advanced ⟳ = full re-download), §6.29
     (30-min fresh window), §6.37 (settled-close serve: no session elapsed → cache current at
     any age). Read those for current behavior.
   - **[RESOLVED 2026-07-13 — §6.16] Two "pivots" can disagree**: table/levels pivot
     (`detect_breakout`/52-wk high, `scan._entry_levels`) vs the VCP detector's `pivot_price`
     (tier logic + tier_reason text). Resolved by picking ONE for display: tier_reason (the
     only UI surface of the detector pivot) was removed as clutter, so every user-facing
     pivot (table, Step 4, chart, watchlist freeze, EOD trigger) is now the app pivot; the
     detector's `pivot_price` remains internal to tier classification, by design.
   - **[DONE 2026-07-13 — §6.16] Target inconsistency**: locked to **+25%** (user call):
     `_entry_levels` targets pivot × 1.25; UI text updated to match.
   - **[DONE 2026-07-13 — §6.16] Scan progress**: `_cached_scan` now passes an
     underscore-prefixed `_progress` callback (excluded from the cache key) into `run_scan`
     → `st.progress` in a slot outside the cached function, throttled to ~1 repaint/20 names.
   - **Persistent watchlist** — *ticker persistence DONE 2026-07-11 (§6.12); frozen-pivot
     part DONE 2026-07-13 (§6.14).* Entries are now
     `{ticker, judged_pivot, date_added, pivot_source, note}` dicts; ⭐/📌 freeze the pivot
     YOU judged, the nightly EOD check auto-freezes the rest on first sight.

### 6.6 GTC stops + never-lower ratchet (2026-07-09)

Suite still **30/30 offline** (three ratchet cases folded into `test_submit_buy_plan_stop_logic`,
now A–F). Encodes Minervini's **"never lower a stop, only raise it"** rule. **Supersedes §6.3's
locked `TimeInForce.DAY` decision** — user reversed it: DAY stops expire each close and must be
re-armed, and a stateless re-arm recomputes the (current-price/recent-low-anchored) stop *down*
when the stock pulls back, which would lower the stop. GTC + a ratchet fixes both.

- **Held-position stop is now GTC + one-way ratchet** (`trade.submit_buy_plan`): read the
  ticker's open sell-stops via new `_open_sell_stops` + `_stop_price_of`; `cur = max(existing
  stop prices)`. Then: new stop invalid (not below price) → keep existing if any (`stop_kept`)
  else `skipped`; `new <= cur` → **keep** (`stop_kept`, no order, no cancel); `new > cur` (or no
  existing) → cancel the lower stop(s) and place a **GTC** `StopOrderRequest` (`stop_only`,
  detail "raised … (was X)"/"placed"). New result status **`stop_kept`** (🔒); counts as
  actioned in the UI success line.
- ~~**Fresh-buy OTO stop leg stays DAY** — Alpaca market entries can't be GTC~~ **[DISPROVEN +
  FIXED 2026-08-04 — §6.38.** A live paper probe showed a GTC market OTO IS accepted (the stop
  leg inherits GTC, held until fill). The DAY leg this section chose expired at the close
  minutes after an intraday fill — the PEBK incident — leaving the position unprotected until a
  manual re-arm. The OTO is now GTC end-to-end; the leg still inherits the parent TIF, which is
  the mechanism in both directions.]
- **Refactor:** `_cancel_open_sell_stops` split into `_open_sell_stops` (finder, returns order
  objects so the price is readable) + `_cancel_orders` (guarded cancel). `_stop_price_of` reads
  `Order.stop_price` (None → can't compare → falls back to replace, as in test case A).
- **UI (`app.py`):** toggle relabelled "…(sell-all, GTC)" with ratchet help; `stop_kept` → 🔒 in
  the per-line results and counted in "Actioned N/M".
- **Not fixed here (still §6.5 item 2):** the stop is still anchored to current price / recent
  low / 50-SMA, so the *computed* level can be loose vs the pivot — the ratchet only stops it
  going DOWN, it doesn't fix where it starts. Clamp-to-pivot is separate.

### 6.7 Risk-to-stop sizing mode (2026-07-09) — §6.5 item 1 DONE

Suite **30/30 offline** (risk cases folded into `test_build_buy_plan_sizing_modes`). Adds Minervini's
position sizer as a 4th trade-ticket mode and fixes the stale `trade.py` module docstring (it
claimed "sizing matches the Step-4 sizer exactly" — false since the pct/dollars/shares modes
landed). Closes §6.5 item 1.

- **`build_buy_plan` mode `"risk"`** (`SIZING_MODES` now 4): `shares = floor((equity × amount%) /
  (price − stop))`. Design decisions (locked with user):
  - **Anchor = current price, not pivot.** So the dollar risk on the order actually sent ≈ risk%
    of equity. Diverges from the Step-4 display (which uses the pivot) only for extended names,
    which converge in the buy zone anyway. Step-4 panel unchanged (still pivot-based, planned risk).
  - **Base = live Alpaca equity** (the same value the `pct` mode already fetches at Build), not the
    Step-4 "Account $" field (a fake $100k default).
  - **Cap = clamp, not skip.** Risk sizing gives position% ≈ risk% ÷ stop-distance%, which routinely
    exceeds the 10% single-order cap (1% risk / 8% stop = 12.5%; verified live: at 1% risk, 7–8%
    stops all clamp). So risk mode clamps shares to `floor(MAX_ORDER_PCT × equity / price)` and sets
    `capped=True` (realized risk then below target); the OTHER modes still skip an over-cap buy in
    `submit_buy_plan`. `MAX_ORDER_PCT = 0.10` mirrored into `trade.py` (like `MIN_TRADE_USD`) so the
    pure builder needn't import alpaca-py.
  - Skips: no equity / no stop / stop ≥ price (non-positive risk-per-share). $50 floor still applies
    (dollar-denominated). New plan key `capped` (always False for non-risk modes).
- **UI (`app.py`):** 4th selectbox option "Risk % to stop" → a risk-% `number_input` (default 1.0).
  Per-order line gains a `⚠︎ capped` flag + a footnote, and a live **"↳ risk to stop ≈ N% ($X)"**
  caption computed from the CURRENT shares and the (possibly edited) stop — so editing a stop after
  Build, which does NOT re-scale a risk-sized qty, stays visible instead of silently drifting.
- **Known wrinkle (documented, not fixed):** editing the stop in the preview doesn't recompute
  shares (Streamlit rebuild is fiddly); the live risk caption surfaces the resulting drift instead.
- **Not this change (still §6.5 item 2):** the stop LEVEL is still current-price/recent-low/50-SMA
  anchored — risk sizing takes the stop as given. Clamp-to-pivot remains the next stop-quality item.

### 6.8 Stop-to-pivot clamp (2026-07-09) — §6.5 item 2 DONE

Suite **31/31 offline** (`test_entry_levels_stop_clamped_to_pivot`). Enforces Minervini's stop
being measured from the BUY POINT (pivot): 7-8% ideal, **10% hard max**. Closes §6.5 item 2.

- **The bug:** `calculate_stop_loss` (`signal_engine.py`) anchors risk 3–10% below the *current
  price* and to swing-low/50-SMA support; `_entry_levels` accepted it as long as it was below the
  pivot. For a name still below its pivot, that support can sit well past 10% below the pivot, so
  the advisory stop (and everything sized off it) silently violated the max-loss rule.
- **Fix (`scan._entry_levels`):** floor the advisory stop at `MAX_STOP_FROM_PIVOT = 0.10` below
  the pivot: `stop = max(raw_stop, pivot × 0.90)`. A tighter engine stop is kept as-is (the clamp
  only bounds the loose side); the no-stop default stays 7.5% below pivot. New levels keys
  `stop_pct_from_pivot` and `stop_clamped`. Verified live over a full synthetic scan: no candidate
  ends with a stop > 10% below its pivot.
- **Why clamp up, not skip:** if logical support is > 10% below the pivot, Minervini's rule is that
  the base is too loose to risk more than 10% — you take the 10% hard stop or pass. Clamping to the
  10% line *is* that rule. It only ever TIGHTENS the advisory stop; it never moves a live order by
  itself (the GTC ratchet still governs real orders).
- **Flows downstream automatically:** the candidates `stop` column, the Step-4 sizer, the risk-mode
  sizing (§6.7), and the default GTC stop (§6.6) all read `levels["stop"]`, so they now inherit the
  clamped level. **UI:** Step-4 adds a "Risk pivot → stop: N% (7–8% ideal, 10% max)" caption, ✅/⚠️
  by whether ≤8%, with a note when the floor bound.
- **Note:** this clamps the stop *level* (the loose-vs-pivot problem). The *tight* side (an engine
  stop only 2-3% below pivot → oversized position) is left alone — it's the safe direction for
  account risk and `calculate_stop_loss` already enforces a 3%-from-price minimum.

### 6.9 Positions page — stop management (2026-07-09) — §6.5 item 3 DONE

Suite **36/36 offline** (5 new). The cockpit's missing "after the buy button" half: a dedicated
**Positions page** for the daily stop-management habit. Closes §6.5 item 3 (the biggest gap) and is
the payoff for the GTC ratchet (§6.6).

- **Placement (user-confirmed):** a dedicated page `cockpit/pages/2_Positions.py` (nav entry +
  sidebar cross-link `<a href="Positions" target="_blank">` in app.py), NOT an in-page tab — the
  sidebar hard-depends on the scan result, so a tab can't dodge the multi-minute scan; a separate
  page loads instantly (own script, own sidebar, never runs `run_scan`) at zero risk to the scanner.
- **Re-arm logic (user-confirmed): Auto per-position** stop basis. `trade.suggest_stop(...)` picks by
  the position's gain: fresh (`< BREAKEVEN_GAIN` 0.16) → initial 8% below entry; working → breakeven;
  well in profit (`>= TRAIL_GAIN` 0.20) with a 50-day → trail `sma_50×0.99`. Floored at the in-force
  stop (ratchet-safe) and — in AUTO only — at breakeven once working (an EXPLICIT basis is honored as
  chosen). Returns `(price|None, effective_basis)`; None = underwater (result not below price) → manual row.
- **Backend all in `trade.py`** (owns the connection + ratchet). The GTC ratchet was **extracted**
  from `submit_buy_plan`'s held branch into a shared `_rearm_gtc_stop(client, symbol, held_shares,
  desired_stop, price, existing, *, OrderSide, TimeInForce, StopOrderRequest)` — ONE source of truth,
  called by both `submit_buy_plan` (held branch collapsed to it — cases A–F unchanged) and the new
  `rearm_stops(targets)`. New `_open_sell_stops_by_symbol` = one `get_orders(status=OPEN, side=SELL)`
  query grouped by symbol (vs N per-ticker). `fetch_positions()` → `{account, positions[]}`: reads
  Alpaca `Position` P&L fields **defensively** (`_attr_float` getattr+coerce — this page is their
  first in-repo consumer), one batched `data_feed.get_many_prices` for `sma_50`/`below_sma50`/
  `volume_ratio` (lazy import). `position_advisories(pos)` (pure) = the 4 exit nudges (no stop / sell
  into strength ≥20% / closed below 50-SMA [+heavy vol] / raise to breakeven ≥2× initial risk — the
  ×risk uses the 8% default since entry-time stop isn't persisted).
- **Page:** account tiles (equity/cash/#/unrealized P&L), account-number confirmation, positions
  `st.dataframe`, a `STOP_BASES` radio (default `auto`), per-row editable stops (key
  `posstop_{sym}_{nonce}_{basis}`, seeded from `suggest_stop`, with a live "risk to stop ≈ N%"
  caption / red "set manually" when underwater), a **"Re-arm / raise all stops"** button via an
  `on_click` callback (reads edited stops from session_state, calls `rearm_stops`, bumps the nonce +
  clears the `@st.cache_data` fetch), and the shared status-icon results readout. Cached fetch behind
  a session nonce so it doesn't hit Alpaca every rerun; `TradeUnavailable` → `st.warning + st.stop`.
- **Tests:** `test_rearm_gtc_stop_ratchet` (raise/keep/equal/first-arm/not-held through the shared
  helper), `test_fetch_positions_offline` (fake client + patched `data_feed.get_many_prices`),
  `test_suggest_stop`, `test_position_advisories`, `test_positions_page_renders` (AppTest + patched
  `fetch_positions`). Regression gate green: `test_submit_buy_plan_stop_logic` A–F unchanged.
- **NOT yet done live** (user's step, mirrors §6.6): a real paper `fetch_positions`/`rearm_stops`
  against `…SZOE`. Risks carried: unverified Alpaca P&L attr names (read defensively), underwater
  invalid-stop rows (ratchet's `stop_is_valid` guards), `sma_50` None under 50 bars.

### 6.10 Re-scan force-refresh fix (2026-07-11) — §6.5 item 8 (parity bug) DONE

Suite **35/35 offline** (`test_streamlit_app_renders_offline` exercises the wiring; the note below
on the 36→35 count is unrelated). The "🔄 Re-scan (refresh prices)" button now ALWAYS forces a real
price refetch instead of only on every OTHER click.

- **Was:** `_cached_scan(…, nonce)` computed `force=bool(nonce and nonce % 2 == 0)`; the button
  bumped `nonce` 1→2→3…, so `force=True` only on EVEN nonces = every other click.
- **Now (`app.py`):** `_cached_scan(…, nonce, _force=False)` with `run_scan(force=_force)`. The
  `_force` arg is **underscore-prefixed so Streamlit EXCLUDES it from the cache key** — force=True
  and force=False share one memo entry, so a forced Re-scan doesn't fork the cache and a later
  filter change re-screens off the (force=False) 24h price cache instead of re-downloading. The
  function body only runs on a genuine miss (new nonce / cleared memo / changed filter), using
  whatever `_force` was passed then. The Re-scan handler sets a **per-run local `_force=True`** (True
  only on the click run), bumps nonce, and clears the memo → that run is a miss → `run_scan(force=
  True)` → full refetch; every other rerun leaves `_force=False` and hits the cache.
- **Deliberately unchanged:** the default 24h same-day REUSE in `data_feed.get_many_prices:383`
  (§6.5 item 8) — that's the cache working as intended; this only repairs the manual refresh hatch.

**Unrelated note (test count 36→35):** `test_build_buy_plan_attaches_stop` removed; its two
unique asserts folded into `test_build_buy_plan_sizing_modes` — no coverage lost.

### 6.11 Weekend Tier-A review workflow + trade cadence (2026-07-11)

A repeatable procedure for turning the Tier-A table into an actionable watchlist, plus the cadence
for acting on it. Faithful to Minervini: in a coiled tape the honest Step-4 output is "wait for the
break," not "buy the coil."

**Cadence — two jobs, two frequencies:**
- **Weekly (weekend) = HUNT.** Re-scan `full_us` for new Tier-A bases (they form over weeks, so daily
  re-scanning is noise). Output = an updated watchlist.
- **Daily at end of day = TRIGGER + STOPS.** (a) Check the *watchlist* for breakouts; (b) manage stops
  on open positions (raise/trail via the Positions page §6.9). EOD only — yfinance daily bars finalize
  after the close, midday reads are stale. Do NOT re-hunt or impulse-buy daily; that stays weekly.
  (GTC stops mean protection persists even if a day is missed — the daily task is *raising* them and
  catching a 50-SMA exit, not re-arming.)

**Step A — scan → Tier A.** Run the app (`full_us`) or programmatically
`run_scan(universe="full_us", cfg=ScanConfig(min_rs=70))` and filter `candidates.tier == "A"`. Read
the regime banner FIRST — only push in a risk-on / decent-breadth tape.

**Step B — judge each Tier-A name (its price history + fundamentals):**
- **RMV is THE base-tightness discriminator** (<~25 tight, >~45 loose). Tightening legs + `fund_score`
  can flatter a loose base — always confirm the base is genuinely quiet.
- **VCP legs:** progressively tighter drawdowns + volume drying up into the tight part.
- **Fuel:** read the RAW rev/eps YoY + operating margin, NOT just `fund_score` (it counts n/a as a
  fail and needs ≥20% on BOTH rev and eps). Ignore absurd EPS % (turnarounds off ~0 — cross-check rev).
- **Position vs pivot:** advise off the APP pivot (`levels["pivot"]`), not the detector pivot (sits
  higher → understates extension). `pct_to_pivot`: negative = price ABOVE pivot (in/past zone),
  positive = BELOW (approaching). Buy zone = pivot..pivot×1.05 (`pct_to_pivot` 0..−4.8%).
- **Earnings:** skip a fresh entry within ~21 days of a report (`earnings_in`, §6.4).

**Step C — classify for Step 4:**
- **Act now** = closes above the pivot on **≥1.5× volume**, price in the buy zone (not extended past
  pivot×1.05), tight base, decent fuel, no earnings ≤~21d.
- **Skip a trigger** that's extended (past the buy-zone top), loose-based (high RMV), or weak-fueled.
- **Coiled but not triggered** → watchlist; wait for the trigger (don't buy the coil pre-breakout).

**Step D — entry mechanic (EOD-confirm, the chosen default).** After the close, if a watchlist name
**closed above its pivot on ≥1.5× volume**, buy at/near the **next open**. Respects Minervini's
close-above-pivot-on-volume rule; costs ~1 day / a percent or two of entry. *Alternative:* a resting
**buy-stop-limit at the pivot** set over the weekend — fills without watching intraday, but can't
check the volume condition (may fill on a low-volume poke that fails). On fill: set the **7–8% GTC
stop immediately** (trade panel), size for ~**1% account risk**.

**Worked example (first run, 2026-07-11).** 3,585 scanned → 561 pass 8/8 → **Tier A = 161** in a
RISK-ON / 33%-breadth tape. Only **2 triggered** on volume, both skipped: **AMWL** (weak fuel — rev
−18%, op-margin −32%; loose RMV 39) and **ASIC** (extended +8% past the buy zone; loose RMV 50). Best
coiled bases to watch: **BAP** (cleanest — legs 7.5→5.5%, volume dry-up, at highs, rev +11%/eps +16%,
earnings 33d out), **PECO** (tightest — RMV 1, defensive REIT), **EBAY** (textbook 5-leg VCP but
earnings 18d + at top of zone). **Verdict: hold fire; watch BAP/PECO for the trigger.** (The scan +
per-name detail were produced by two throwaway scripts wiring `run_scan` + `data_feed.get_many_prices`
+ `detect_vcp`/RMV; reproducible, not committed — regenerate as needed.)

**Step E — the SELL procedure (added 2026-08-05, after the first live month).** A breakout buy's
thesis = **P1** resolution (closed above the pivot on ≥1.5× volume, and STAYS above) ∧ **P2**
Stage-2 structure (8/8 template, rising 50-SMA) ∧ **P3** risk-on tape ∧ **P4** no unpriced binary
(no report ≤21d without a cushion). Any pillar failing kills the trade — the stop is the disaster
floor for what happens *between* checks, never the sell signal. Decisions on **settled closes
only** (intraday prints mislead in both directions — see PEBK); execution at/near the next open.

- **Day 0:** entered intraday and it closed back below the pivot → the breakout never happened;
  sell next open (PEBK 2026-08-04).
- **Daily until cushioned (P1):** decisive close below the pivot (>1-2%, on volume, or a 2nd
  consecutive close) → sell; close below the breakout bar's low → sell, no grace day; highest-
  volume-since-breakout reversal closing weak → sell half.
- **Laggard clock:** no +3% cushion by ~day 10 → sell into strength; flat-to-red at day 15-20 →
  exit. Calibration, not scripture — the principle is that real breakouts pay quickly.
- **Earnings (P4):** a LOSS is never carried into a ≤21d report (gaps skip stops); a small gain
  is trimmed to hold-through size (`EARNINGS_CUSHION_MIN` 8%).
- **Tape (P3):** on a risk-off flip, demote every yellow flag to red — laggards go first.

First-month evidence: UNP sold −6% (correct call, made ~5% below the pivot — late by this
procedure), PEBK caught at −1.9% (Step Day-0), STRW exited before its report. Every dollar lost in
the month traced to *entry*-rule violations (§4's live-trading list); the sell side, when run,
worked.

### 6.12 Persistent watchlist across runs (2026-07-11) — §6.5 item 8 (ticker persistence) DONE

Suite **36/36 offline**. The watchlist now survives between app runs (was session-only + a manual
`.txt` download/upload round-trip). Supports the weekend-hunt → daily-trigger workflow (§6.11): build
the list Saturday, it's still there when you check for triggers during the week.

- **`export.py` (pure, tested):** `save_watchlist(path, tickers)` and `load_watchlist(path) -> List[str]`
  — JSON; upper-cased, de-duped, first-seen order; `load` returns `[]` on missing/corrupt/non-list so a
  bad file never breaks startup; `save` best-effort (swallows write errors, session list stays authoritative).
- **`cache.py`:** `WATCHLIST_JSON = data/cockpit/watchlist.json` (gitignored like the rest of the cache).
- **`app.py`:** `_wl()` loads from disk once per session (`if "watchlist" not in st.session_state`);
  new `_wl_persist()` saves after every mutation — wired into `_wl_add` / `_wl_remove` / `_wl_clear`, so
  the picker + `.txt` upload bulk-adders persist transitively (they call `_wl_add`). The path is referenced
  **module-qualified** (`cache.WATCHLIST_JSON`, read at call time) so tests can patch it.
- **Tests:** `test_watchlist_persistence` (pure round-trip + corrupt/missing/non-list → `[]`). The two
  watchlist AppTests (`test_watchlist_add_button_and_download`, `test_streamlit_app_renders_offline`)
  patch `cache.WATCHLIST_JSON` to a temp file so they neither read nor clobber the real one — verified the
  suite writes no real `watchlist.json`.
- **Still open (§6.5 item 8):** ~~the richer `{ticker, judged_pivot, date_added, note}` version (freeze the
  judged pivot) — a list[dict] refactor touching the picker + CSV export; deferred.~~ **Shipped 2026-07-13, §6.14.**

### 6.13 Trade journal page (2026-07-12) — §6.5 item 4 DONE

Suite **40/40 offline**. Minervini's "know your numbers" (*Think & Trade Like a Champion*):
batting average, avg win/loss, expectancy — the numbers that gate progressive exposure —
reconstructed from Alpaca order history with **zero new bookkeeping** (the client_order_id
tags carry everything). New page `pages/3_Journal.py` sits below Positions in the sidebar
nav (Streamlit's built-in page nav navigates in-tab; the filename number prefix orders it).

- **`trade.py` — pure, tested:**
  - `fetch_order_fills()` — pages backwards through `GetOrdersRequest(status=CLOSED)` via
    `until` = oldest `submitted_at` seen (exclusive, so pages don't overlap; deduped by order
    id anyway; `_MAX_ORDER_PAGES=40` ceiling). Drops never-filled orders; keeps partial fills
    at `filled_qty × filled_avg_price`. Returns `{account, fills}` oldest-first.
  - `build_trade_journal(fills)` — groups fills per symbol into **position episodes**
    (flat → long → flat = ONE closed trade), so scale-ins/partial exits aggregate into a
    single round trip (avg entry vs avg exit). Returns `{closed, open, unmatched_sells}`;
    open episodes are excluded from stats; a sell with no prior buy in the pulled history is
    recorded as unmatched, never guessed at. `tagged` = any fill's client_order_id starts
    with a `SEPA_TAG_PREFIXES` prefix — the ENTRY tag suffices, because a triggered OTO stop
    leg exits under an Alpaca-generated id, not the parent's.
  - `journal_stats(closed)` — n/wins/losses/scratches, batting avg (wins ÷ all closed),
    avg win/loss %, win/loss ratio, expectancy (mean per-trade P&L% = batting × avg win +
    (1 − batting) × avg loss), total realized P&L, avg hold days won/lost. All ratios
    degrade to `None` on empty inputs (fresh account renders "—", never crashes).
- **Page:** Refresh + cached fetch (same nonce pattern as Positions), 5 stat tiles, closed
  table (newest exit first) + CSV download, open-trades table, unmatched-sells warning.
  Checkbox **"Cockpit trades only"** (default ON) filters at the *episode* level — matching
  runs on all fills first so lot accounting stays correct — and keeps AWPmirror activity out
  if the page ever runs on the shared keys.
- **Tests (36→40):** `test_build_trade_journal` (episodes, re-entry, partial sell, orphan
  sell, tag propagation, junk fills), `test_journal_stats`, `test_fetch_order_fills_offline`
  (pagination forced via patched `_ORDERS_PAGE_LIMIT=2`, exclusive-until fake),
  `test_journal_page_renders` (AppTest, patched fetch). Verified live (read-only) against
  the real paper account: 4 SEPA-tagged holdings → 4 open episodes, 0 closed, stats all "—".
- **Known limits:** episode P&L ignores dividends/fees (paper account, none) and treats a
  same-timestamp sell-before-buy as unmatched (never seen in practice); shorts aren't
  modeled (long-only cockpit).

### 6.14 Frozen-pivot watchlist + nightly EOD trigger check (2026-07-13) — §6.5 items 5 & 8 DONE

Suite **49/49 offline**. Automates the DAILY half of the §6.11 cadence (weekly hunt → daily
EOD trigger + stops): a scheduled script answers "did any watchlist name close above its
pivot on ≥1.5× volume?" every weekday evening. Decision support only — it NEVER places
orders; a trigger means judge it, then buy at/near the next open via the trade panel.

**Frozen-pivot watchlist (the prerequisite).** `watchlist.json` entries are now
`{ticker, judged_pivot, date_added, pivot_source, note}` dicts (`export.py`: `make_entry`/
`_coerce_entry`/`watchlist_tickers` + save/load; legacy string-array files migrate at READ
time, element-wise, never raising — the file is rewritten on first mutation). The detected
pivot drifts each scan, so the trigger level is frozen once and stays put:
- **⭐ add from the chart** freezes the current app pivot (`pivot_source="judged"`).
- **📌 "Freeze pivot @ X"** (new button under the toggle) re-freezes an existing entry —
  shown when the entry is unfrozen, auto-frozen, or drifted from the current app pivot.
- **Nightly auto-freeze on first sight**: an entry still unfrozen at check time (picker/
  .txt adds) gets the scan-chain pivot computed and RECORDED (`pivot_source="auto"`,
  `triggers.freeze_missing_pivots` — pure; the CLI persists). Your 📌 overrides. Entries
  that can't compute (<200 rows / no data) stay unfrozen, report `no_pivot`, retry nightly.
  **`compute_scan_pivot` must pass `detect_vcp`'s result into `detect_breakout`** — the
  first cut passed `None`, got `breakout_level=None`, and `_entry_levels` silently fell
  back to the 52-week high: EBAY froze at 118.98 while the chart showed 111.86 (user-caught
  2026-07-13, fixed same day; regression test pins `compute_scan_pivot(payload["df"]) ==
  payload["levels"]["pivot"]` against `screen_universe`). A `breakout_level=None` fallback
  to the 52-wk high can still be LEGITIMATE (BAP: no VCP-peak level exists → 403.30 is the
  real app pivot).
- `date_added` = the date of the current pivot decision (add/re-freeze/auto-freeze stamp it).
- Sidebar renders `TICKER 34.12` / `(a)` auto / `\*` unfrozen; the List CSV always appends
  `judged_pivot,date_added,pivot_source,note`. `build_buy_plan`/OHLCV CSV keep their
  `Sequence[str]` contracts via the `_wl_tickers()` projection. No `st.data_editor`
  (AppTest can't reach it); `note` has no UI editor yet.
- **Rollback note:** pre-§6.14 code reading a new-schema file yields `[]` (its str-coercion
  rejects dicts) — harmless-but-empty; the Names .txt download is the portable backup.

**Trigger check (`triggers.py` pure + `eod_trigger.py` CLI).**
- Gate: `close > judged_pivot` AND `last volume / prior 50-DAY avg ≥ 1.5`
  (`TRIGGER_VOL_RATIO`, matches `trade.HEAVY_VOL_RATIO`; Minervini's standard) AND the bar
  is dated today (America/New_York) — a stale Friday bar can NOT re-fire on a holiday run
  (`all_stale` flags a no-new-bar day). NOTE: the app's "Vol OK" badge is a 20-day read
  (`detect_breakout`); the report carries that as `volume_ratio_20` context, never the gate.
- Status precedence (booleans stay authoritative; summary buckets BY STATUS):
  `no_data → no_pivot → stale → extended (close > pivot×1.05, don't-chase — beats
  triggered) → triggered → watch`. Earnings ⚠ when `0 ≤ earnings_in ≤ 21` (cached
  `get_fundamentals`). Market note = SPY-only `analyze_spy_trend` (full regime needs
  universe breadth a nightly check shouldn't pay for) — labeled `spy`, not `regime`.
- Prices via `get_many_prices(syms+["SPY"], max_age_days=0.0)` — NEW `max_age_days` param
  (default 1.0 = old hardcoded gate) routes warm names through the cheap incremental
  top-up for the finalized close; `force=True` would full-refetch 2y/name. Fetch failures
  degrade to the untouched cache (→ stale), cold no-cache names → `no_data`.
- Report: `data/cockpit/triggers/triggers_YYYY-MM-DD.json` (schema 1; same-day rerun
  overwrites; `load_latest_trigger_report` walks newest-first, first parseable wins). The
  app sidebar shows the latest report below the watchlist (read-only captions, `.get()`
  everywhere). Console output ASCII-only (the .bat logs to a cp1252 file — an emoji here
  crashed the first CLI test; keep icons in Streamlit only).
- CLI flags: `--date YYYY-MM-DD` (pin the run date), `--no-write` (print only — skips the
  report AND the watchlist write-back). ~~`--prewarm`~~ **[deleted §6.18 — refresh via the
  app's Re-scan instead].**
- **Concurrency note:** the script writes `watchlist.json` (auto-freeze). The app reads it
  once per browser session — refresh the app after a nightly run to see auto-frozen pivots.
  (Since §6.24 both writers merge with the file before saving — atomic, lost-update-safe —
  and any app mutation adopts disk pivots; the refresh is only needed to *see* them sooner.)

**Scheduling.** **[HISTORICAL — the original nightly task ("SEPA EOD Trigger", weekdays 17:30,
`--prewarm`) was UNREGISTERED and `--prewarm` deleted in §6.18, which replaced it with the
half-hourly "SEPA Intraday Trigger" task (weekdays 09:30–16:30). See §6.18 for the current
schedule; a manual in-app 🔔 button was added in §6.35. The old re-register command was removed
from this section so it can't resurrect the dead task.]**

**Tests (40 → 43 after the watchlist refactor → 49 with the trigger suite).** New:
`test_watchlist_entry_normalization`, `test_watchlist_tickers_projection`,
`test_watchlist_legacy_migration_roundtrip`, `test_check_triggers_pure`,
`test_freeze_missing_pivots`, `test_trigger_report_roundtrip`,
`test_get_many_prices_max_age_days`, `test_eod_trigger_cli_offline` (in-process, patched
feed; asserts the auto-freeze write-back + `--no-write` leaves disk untouched),
`test_trigger_report_sidebar_renders`. Updated: the four watchlist tests (dict schema +
end-to-end ⭐-freeze assert through the real app) + hermetic `TRIGGERS_DIR` patches.
Verified live 2026-07-13 (after the detect_vcp fix + re-freeze): 4 legacy names migrated +
auto-froze at the APP pivots (EBAY 111.86, BAP 403.30, PECO 40.96, EIX 75.01 — spot-checked
equal to the with-VCP chain per name), all `watch`, report written, task registered `Ready`.

### 6.15 Regime warning in the Build-plan flow (2026-07-13) — §6.5 item 6 DONE

Suite **49/49** (unchanged — app-only change, deliberately NO test assert: user preference,
it's a display-only caption and the general render path is already covered by the trade-panel
AppTest). One warning caption at the top of the sidebar paper-trade panel when
`res.regime["should_generate_buys"]` is False — the top-of-page CAUTION banner repeated at
the point where the finger is. No gate, no behavior change: managing stops in a weak tape
stays legitimate; the line just makes pressing a fresh buy a conscious act.
**Test gotcha discovered (and reverted) while trying to assert on it — KEEP for future
AppTests:** AppTests share one process, so `st.cache_data` on `_cached_scan` serves an
EARLIER test's ScanResult on a cache hit (the patched `run_scan` never runs). Any future
AppTest that mutates its ScanResult fixture must `st.cache_data.clear()` first.

### 6.16 Table declutter + target 25% + scan progress bar (2026-07-13) — §6.5 item 8 polish DONE

Suite **49/49** (benchmark distribution byte-identical: A=79/YES 53, B=114, C=7 — the tier
LOGIC is untouched). Three user calls in one pass, then a follow-up:
- **Tier reasons removed ENTIRELY** (user: clutter, then "useless overhead"). First pass
  dropped the column from `DISPLAY_ORDER`/`READABLE_COLS`/`COL_HELP` (and thus the List
  CSV); second pass removed the audit-string BUILDING in `vcp.py` and the `tier_reason`
  key from the result dict + scan frame. Each tier branch keeps the old reason wording as
  a code comment for revival. `pattern_details` still carries the `_empty()` exclusion
  reasons (dead tape / insufficient data), which is what the tests assert on now.
- **`pivot_price` export commented out, calculation kept LIVE** (user call): nothing
  consumed the exported key, but the internal value drives `in_buy_zone`/`below_pivot`/the
  A-vs-B extended split — it can NOT be commented out without breaking tiers and the
  never-miss benchmark. The return-dict line is commented in place, ready to re-enable.
  Side effect of losing tier_reason: the detector pivot no longer reaches the UI anywhere,
  which closes the "two pivots can disagree" display question — one pivot (the app pivot)
  everywhere the user looks.
- **Target = pivot × 1.25** (was ×1.225 with "20–25%" copy; guide said 15–20%). User locked 25%.
- **Cold-scan progress bar**: `_cached_scan(..., _progress=None)` → `run_scan(progress=...)`;
  `st.progress` in an `st.empty()` slot, throttled, cleared post-scan; memo hits never show
  it. Two landmines from shipping it (both fixed 2026-07-14):
  - **`st.cache_data` + in-function st elements = `CacheReplayClosureError` on the next
    rerun** — cache_data RECORDS elements emitted inside a cached fn and REPLAYS them into a
    vanished slot on every hit. Fix: `_cached_scan` is a hand-rolled **session_state memo**
    (`_scan_memo`; Re-scan pops it) — also retires the §6.15 cross-AppTest cache-leak for
    this fn (the gotcha stands for any future `@st.cache_data` use). Mocks hid it: a
    MagicMock `run_scan` never calls the callback — the renders test now drives `progress()`
    via side_effect and asserts the memoized second `at.run()` survives with one scan call.
    Tradeoff accepted: memo is per-browser-session, not process-global.
  - **Bar appeared only-full-at-end** — cache-served names never emitted AND the screening
    loop (the true time sink on a warm cache) had no progress. Now `get_many_prices` emits
    per cache-served name, `screen_universe` has a `progress` callback, and `run_scan`
    chains both phases through one `(done, total, label)` callback with phase-prefixed
    labels ("Prices · SYM" → "Screening · SYM").

**§6.5 backlog: FULLY CLOSED as of 2026-07-16** (RS §6.17, EDGAR §6.17, volume bars §6.17,
plus everything above). Only crumb left from item 7: the optional "RS line at new high
before price" chart flag — never built, low priority. New open items live in §6.19.

### 6.17 Item 7 fidelity: IBD-style RS + up/down volume bars (2026-07-14); EDGAR shelved

Suite **50/50** (new `test_rs_ratings_ibd_weighted`; per-bar color asserts folded into the
chart test). Two of item 7's three pieces shipped:
- **RS rating** (`scan._rs_ratings`): now the IBD blend — ``2·r(3mo) + r(6mo) + r(9mo) +
  r(12mo)`` percentiled 1-99 across the scanned set (was a plain 6-mo percentile). Ranked
  as the weighted MEAN over the legs a name actually has, so young listings (≥ the
  mandatory 6-mo leg — the old inclusion rule) compete on their 3+6-mo strength instead of
  dropping out; for full-history names the ranking is identical to the raw IBD sum.
  Horizons derive from ``cfg.rs_period`` (126 → 63/126/189/252). Recency now counts
  double: a +20% move inside 3 months outranks a +25% move from a year ago. UI help texts
  updated (sidebar slider, rs column tooltip, table info). NOTE: RS values shift vs the
  old formula — a name's rating can move a few points; min_rs=70 semantics unchanged.
- **Volume bars colored by up/down day** (`charts.py`): green/red by close-vs-prior-close
  (gray for the window's first bar) so heavy down-day volume — distribution, a VCP
  disqualifier — is visible at a glance. Vol SMA20 + 1.5× threshold lines unchanged.
  (Reverted 2026-07-14 during the one-item-per-commit resequencing; restored to the tree
  2026-07-16 as its own commit-ready item.)
- **EDGAR fundamentals depth: shelved 2026-07-14, REAPPLIED 2026-07-16** (suite 52/52;
  the shelf doc served its purpose and was deleted). WHY it was needed, measured against
  the real cache (968 names): `eps_yoy_prev`/`revenue_yoy_prev` were missing for **100%**
  of names (YoY-prev needs 6 quarters; yfinance caps at ~5), so the fund_score
  "EPS accelerating" check could NEVER pass — every score silently capped at 3/4; 12%
  lacked even current EPS YoY (n/a counts as fail). The backfill pulls SEC XBRL
  company-facts (primary source, keyless, 10 req/s w/ contact UA): revenue/EPS quarterly
  YoY (+prev, date-matched 330-400d so missing quarters can't misalign), FY EPS growth,
  3-quarter acceleration; plus `last_surprise_pct` from `yf.Ticker.earnings_dates`.
  Merge: **yfinance wins when present** (filings lag), EDGAR fills Nones + adds its own
  keys; per-ticker weekly JSON cache in `data/cockpit/edgar/`; cache schema-upgrade
  refetches once (first scan after this lands is slower — every passer pulls facts at
  ≤10/s, cached after). Step-2 box gains "FY EPS / 3q accel / surprise" line. Amended
  filings win by 'filed' date; revenue tag fallback chain as in Main.ipynb. Verified live
  vs real SEC payloads (EBAY, PECO — both returned full metrics incl. the previously
  impossible `*_prev`).

### 6.18 Half-hourly intraday trigger checks; nightly prewarm REMOVED (2026-07-16)

Suite **50/50**. The trigger check now runs **every 30 minutes during market hours**
instead of once nightly — near-real-time price/volume vs the frozen pivots without
watching the screen. Still watchlist-only (~1-2 batched yfinance calls per run — no
rate-limit exposure), still zero order placement.

- **Scheduling:** new task **"SEPA Intraday Trigger"** — weekdays 09:30, repeating every
  30 min for 7h (last run 16:30 = the day's final, settled-close report),
  `-StartWhenAvailable` (machine asleep across any number of marks → ONE catch-up run on
  wake, then the anchored schedule resumes — the user's requested wake behavior). The old
  17:30 **"SEPA EOD Trigger" task was UNREGISTERED** and the **`prewarm()` code + 
  `--prewarm` flag DELETED** from `eod_trigger.py` (user: too slow, no point refreshing
  accurate historical data daily; cache refreshes via the app's 🔄 Re-scan button — the
  weekend full_us scan rides the incremental top-up, full re-baseline only after a
  >10-day gap). `scripts/eod_trigger.bat` now forwards `%*`. 2026-08-12: the task's
  battery flags were cleared so unplugged ≠ silent (§6.49 — the one-liner to re-apply
  after any re-registration lives there).
- **Critical fix — `_merge_incremental` intraday false-split** (`data_feed.py`): the
  split detector compared Close on ALL overlapping dates; today's PROVISIONAL bar moves
  between intraday fetches, so every half-hourly run would have "diverged" → full 2-year
  re-baseline per name per run. Divergence is now measured on settled (pre-today) overlap
  only; the merge still takes the newest today-bar via `keep="last"`. Any future intraday
  consumer of the incremental cache needs this to exist.
- **Intraday awareness (`triggers.py`):** `check_one`/`check_triggers` gain `now=`
  (pinned in tests like `today=`). New per-name **`volume_pace`** = 50-day ratio ÷
  fraction of the 09:30-16:00 session elapsed (clipped ≥ 0.1 — the first ~40 min
  overstate pace) — "is volume running hot for this time of day?" (e.g. 0.5× actual at
  half-session = 1.0× pace). **Context only: the `triggered` gate is untouched** (actual
  ratio ≥ 1.5 on today's bar), so an early breakout reads `close_above_pivot` + hot pace
  but stays `watch` until real volume confirms. Top-level **`intraday`** flag = fresh bar
  + run before ~16:05 ET (provisional close/volume). yfinance's intraday daily-bar volume
  can lag real-time — pace is advisory. Schema stays 1 (additive; readers use `.get`).
- **Display:** console header "TRIGGER CHECK {date} {HH:MM}" + PACE column + intraday
  note (ASCII); sidebar header **"🔔 Trigger check — {date} {HH:MM}"** (from
  `generated_at`), ⏱ intraday caption, per-name `pace N.N×` (shown only on intraday
  reports). §6.14's "17:30"/"--prewarm" references are historical as of this section.
- **Sidebar auto-refresh:** the report block is a `@st.fragment(run_every="60s")` —
  an open browser tab picks up each half-hourly report within a minute WITHOUT user
  interaction. Fragment reruns are isolated (they re-read one small JSON; the memoized
  scan/table/chart never re-run) and the timer only ticks while a session is connected.
  AppTest executes fragments inline on the initial run (timer reruns don't fire in
  tests), so the existing sidebar asserts still hold.
- Same-day report file still overwritten per run (idempotent) — the sidebar always shows
  the latest check; ~15 log stanzas/day.

### 6.19 First-week operating notes: PECO case study, post-breakout freeze semantics, EDGAR precedence (2026-07-16)

**PECO — the "missed" +6% (worked through with actual bars, like §6.11's example):**
- Timeline: first close above the (later-frozen) 40.96 pivot on **2026-06-15**; the real
  Minervini trigger day was **2026-06-26 at 4.4× volume** (with 1.76×/2.02× days around
  the cross) — ALL before the frozen-pivot system existed (2026-07-13). By freeze time
  PECO was already in-zone; it then drifted to a 43.44 ATH on DRY volume (0.2-0.8×) into
  earnings (~2026-07-23). The half-hourly checks correctly said "above pivot, volume not
  confirming" the whole way — no system failure, a bootstrapping artifact.
- **Even with hindsight the playbook said skip**: on 7/13 price (42.04) was inside the
  buy zone, but earnings were 10 days out — inside §6.11's ~21-day no-fly window with
  zero cushion. The volume question was moot.
- **The plan now (don't chase +6% into a report):** (a) post-earnings retest of ~41 — the
  pivot (40.96) + 50-day SMA (40.84) cluster — on dry pullback volume = a legitimate
  second-chance entry with a stop under the shelf; or (b) if it runs, wait for a new
  3+ week base and 📌 re-freeze at the new top; or (c) let it go — skipped trades cost
  expectancy nothing, and the same discipline correctly skipped AMWL/ASIC on 7/11.
- Instrument note: PECO is a low-beta defensive REIT ("RMV 1" in the 7/11 hunt) —
  institutions accumulate these quietly; the 1.5× thrust signature is calibrated on
  growth leaders and may simply never print for this class of name.

**OPEN ITEM (proposed 2026-07-16 — BUILT 2026-08-04, §6.36) — post-breakout freeze
semantics:** the trigger's
"close above pivot on ≥1.5× volume" assumes the name is still BELOW its pivot; a pivot
frozen for a name already trading above it waits for an event that may already be behind
it. Proposed: (a) a distinct **"crossed"** status/icon (above pivot but never
volume-confirmed since the freeze) so quiet drifts are loud in the sidebar, and (b) a
freeze-time warning ("post-breakout name — plan a pullback/secondary entry, the trigger
won't re-fire"). Small change to `triggers.check_one` + the app freeze paths.

**DECISION — EDGAR does NOT take precedence over yfinance (user asked 2026-07-16; keep
the §6.17 yfinance-wins merge):**
- The backfill SUPPLEMENTS, never replaces: yfinance fetch runs first unchanged; EDGAR
  only fills None fields and adds its own keys (`eps_fy_yoy`, `eps_accel_3q`). EDGAR
  failure ⇒ exactly the old yfinance-only behavior. Margins / QoQ / inventory /
  earnings calendar / surprise % are yfinance-only regardless — "EDGAR instead of
  yfinance" is not even structurally possible; only precedence on the overlapping
  growth fields was in question.
- **Why yfinance stays first: the filing lag.** Yahoo carries a new quarter within hours
  of the press release; EDGAR only knows it when the 10-Q is FILED (typically days,
  legally up to 40). EDGAR-first would show the OLD quarter as "latest" exactly during
  earnings season — understating fuel in the week after a blowout report, the most
  expensive possible staleness for a breakout strategy.
- Known second-order risk accepted: the acceleration check can compare a yfinance-derived
  `eps_yoy` against an EDGAR-derived `eps_yoy_prev` (cross-source). Both are GAAP diluted
  EPS in the normal case, so mismatch-manufactured accel/decel should be rare.
- **Future refinement (only if evidence demands): freshness-aware merge** — prefer EDGAR
  unless its latest quarter-end is OLDER than yfinance's (i.e., detect the filing-lag
  window and only then fall back to Yahoo). Primary-source data ~90% of the time, Yahoo's
  speed during the lag.
- **Planned diagnostic before building anything:** both sources now cache side by side
  (`data/cockpit/fundamentals/` vs `data/cockpit/edgar/`) — after 1-2 weekend hunts, diff
  them and measure how often (and how big) `eps_yoy`/`revenue_yoy` disagreements actually
  are. Rare + small ⇒ current design stands; frequent or large ⇒ the freshness-aware
  merge earns its complexity.

### 6.20 Multi-agent code review + two HIGH fixes; remediation backlog (2026-07-17)

Ran a comprehensive multi-agent review of the cockpit + the reused `minervini_screener/screening`
rule functions (8 dimension reviewers → dedup → adversarial verify, with a 3-lens panel for HIGHs).
**34 findings survived verification** (2 high, 13 medium, 19 low). The full ranked backlog is
`cockpit/REVIEW_BACKLOG.md` (Phase 1 correctness → Phase 4 dead-code cleanup); it supersedes the
now-largely-DONE §6.5 list. Two findings claiming vendored modules are dead were **refuted** —
`notifications/`, the batch processors, `screener.py` etc. are intentionally-verbatim MIT code
(PROVENANCE.md), unused here by design.

Both HIGH findings fixed:
- **`full_us` universe filter silently dropped every W/R/U-ending symbol** (`data_feed.py`
  `_filter_us_symbols`). The unanchored `~sym.str.contains(r"(?:WS|WT|W|R|U)$")` matched symbols of
  ANY length, so ordinary 4-letter leaders (PLTR, SNOW, UBER, LULU, TROW, DOW, LOW, EMR, KR…) and
  single-letter `U` (Unity) were removed from the **only enabled discovery universe** — exactly the
  high-RS names a Minervini screen targets. Now anchored to the nasdaqtrader base+suffix shape
  `~str.match(r"^[A-Z]{4}[WRU]$")`, so those names are kept while genuine SPAC warrants/rights/units
  (`CVIIW`/`CVIIU`/`CVIIR`) still drop; a 3-char-base warrant (4 chars) can slip through but just
  fails the trend template. Regression: `test_get_universe_full_us_offline` now asserts SNOW/PLTR/
  LULU/`U` are KEPT and base+W/R/U are dropped. Commit `cb54058`. (See also the full_us note above.)
- **Trade plan sized/stopped on stale prices** (`build_buy_plan` + `app.py` Build flow). The scan is
  memoized in `session_state` with NO time-based invalidation and the 60s trigger fragment keeps the
  tab alive for days, so a weekend scan → Wednesday Build+Submit could size shares and validate the
  stop against a days-old close (market order then fills far from every displayed number; a
  stale-validated stop can trigger instantly on fill). Fix: **`freshen_prices`** re-pulls just the
  watchlist names' latest bars at Build (cheap `max_age_days=0` incremental top-up, not a universe
  re-download) and overlays them; **`build_buy_plan`** gained a `max_bar_age_days` staleness guard
  (`STALE_PLAN_BARS=2` trading days, tolerating a weekend + one holiday) that skips any name the
  refresh couldn't freshen, with reason `"stale price data … — Re-scan to refresh"`. Builder stays
  pure (network in `freshen_prices`; guard params default off). Regressions:
  `test_build_buy_plan_skips_stale_bars`, `test_freshen_prices_overlays_latest_bars`. Commit `8c4aa65`.

Offline suite **54/54**. Remaining 32 findings (13 med, 19 low) are the Phase-1..4 backlog — top of
Phase 1 is the frozen-`judged_pivot`-not-reaching-the-plan bug (§6.14 pivots drift; the plan uses the
current scan pivot for stop/extended/risk instead of the frozen trigger level).

### 6.21 Comment-trim pass — "moderate" density (2026-07-17)

Trimmed the verbose comment style across the **12 cockpit `.py` files** to a moderate level: cut
historical narration (superseded approaches, `"replaces the old …"`, commit hashes, dated change-log
asides like `"dropped 2026-07-13"`) and multi-line restatements of what a line obviously does; **kept
every load-bearing constraint** in 1-2 tightened lines (CacheReplayClosureError, `paper=True`,
"Alpaca market entries can't be GTC", the provisional-today-bar quirk, the RMV below/at-pivot veto,
why each VCP threshold has its value). Net **~121 comment lines removed** (256 ins / 377 del).
`cache.py`, `__init__.py`, `pages/1_SEPA_Guide.py` were already moderate → untouched. The vendored
`minervini_screener/` package was left entirely alone (verbatim MIT copy, PROVENANCE.md). **Verified
behavior-neutral:** a code-token diff (comments+strings stripped) is identical to HEAD on all 12
files, every file compiles, and the offline suite is 54/54.

### 6.22 Review-backlog Phase 1 mediums (items 1-5) fixed (2026-07-17)

Suite **60/60 offline** (+6 regression tests). Cleared all five Phase-1 mediums in
`cockpit/REVIEW_BACKLOG.md` (27 findings remain: 8 med, 19 low):

- **1 — frozen pivot now reaches the plan.** `build_buy_plan` gained a `pivots` map
  (ticker→frozen `judged_pivot`); for a name with a frozen pivot the buy zone, `extended` flag,
  default stop, and risk sizing key off it instead of the drifted scan pivot (mirrors
  `_entry_levels`: 7.5%-below default, 10% hard floor; a tighter engine stop below the frozen
  pivot is kept). `app.py` builds the map from watchlist entries and shows `📌 pivot NN.NN` in the
  preview; plan entries carry `pivot_frozen`. This is the §6.14-drift bug the trade panel had.
- **2 — margin quarter-alignment.** `_margin`/`_margin_trend` align num/den on their common
  quarter-end index (`_aligned`) so a newest quarter with revenue but not-yet-parsed
  GP/OI can't pair GP(Q-1) with Rev(Q0). `_yoy`/`_yoy_prev` now date-match the ~1-year-prior
  quarter (330-400 days, via `_yoy_at`) like `_edgar_yoy_series`, not a fixed 4-step lag.
- **3 — 50-SMA reclaim isn't a pivot.** `_entry_levels` nulls the pivot on
  `breakout_type == '50 SMA Breakout'`, falling through to the 52-wk-high fallback (fixes the
  frozen trigger level via `compute_scan_pivot` too). Vendored `phase_indicators.py` untouched.
- **4 — no double buy on re-submit.** `submit_buy_plan` queries open BUY orders
  (`_open_cockpit_buys`) and skips a not-held ticker with a pending cockpit buy
  (`client_order_id` 'SEPA…'), so the after-close cadence's queued DAY OTO buy isn't re-placed.
- **5 — VCP base can't span a collapse.** `vcp._detect_at`'s backward walk now breaks when a
  filtered-out leg lies chronologically between two kept legs, so a >35% crash-and-recover isn't
  stitched into one "base" (was reportable as is_vcp / tier A). **Scoped to depth only:** A/B
  benchmarking showed the leg-length half of the suggested fix over-fragments real bases on
  normal 1-2-bar shakeouts (−5 tier-A YES, recall 53→48) for a precision gain a recall-first
  detector doesn't want; depth-only reproduces the pre-fix benchmark exactly (zero recall cost).

Tests: `test_build_buy_plan_uses_frozen_pivot`, `test_margin_aligns_num_and_den_quarters`,
`test_yoy_is_date_matched`, `test_entry_levels_ignores_50sma_breakout_pivot`,
`test_submit_buy_plan_skips_pending_cockpit_buy`, `test_vcp_base_does_not_span_a_collapse`.

### 6.23 Review-backlog Phase 1 lows (items 6-10) fixed — Phase 1 COMPLETE (2026-07-18)

Suite **64/64 offline** (+4 tests + 1 assertion). All of Phase 1 done; 22 findings remain
(8 med, 14 low) → Phase 2 next.

- **6 — held names in the plan preview.** `build_buy_plan` is holdings-blind, but submit sends no
  buy for a held name. New `fetch_held_shares()` (best-effort) is fetched on Build → `trade_plan
  ["held"]`; the preview renders a held name as `already held (N sh) · stop re-arm only` (or
  `skipped (attach off)`), counts only `_buys` in the est-value total, and scopes the
  extended/capped/earnings footnotes to buys. Builder stays pure.
- **7 — watchlist CSV order.** `watchlist_list_csv` now reindexes over ALL tickers
  (`drop_duplicates("ticker").set_index("ticker").reindex(tickers)`) so a stale name stays in
  place instead of being concat-appended last.
- **8 — Minervini key spelling.** Kept the SINGLE canonical spelling `.env` uses
  (`ALPACA_API_KEY_MINERVINI` / `ALPACA_API_KEY_SECRET_MINERVINI`) and corrected the stale comment
  + §6.3's "accepts both spellings" claim to match — the phantom `ALPACA_MINERVINI_API_*` form was
  never in `.env`. (Chose the backlog's "correct the comment and HANDOFF" option over adding
  either/or logic — the user uses only the one spelling. Superseded my initial tuple change.)
- **9 — stop ratchet re-quantifies.** `_rearm_gtc_stop` (new `_order_qty`): a would-be-lower re-arm
  re-places the stop at the SAME level for the full held qty when the in-force stop under-covers
  a grown position (`0 < covered < held`); unreadable qty never churns.
- **10 — positions vol ratio.** `fetch_positions` uses `Volume.iloc[-51:-1].mean()` (prior 50,
  excl. today) to match `triggers._volume_ratio` and the shared 1.5× heavy-volume exit gate.

Tests: `test_trade_plan_preview_marks_held_names`, `test_watchlist_list_csv_keeps_stale_in_order`,
`test_minervini_key_envs_single_spelling`, `test_rearm_gtc_stop_requantifies_grown_position`,
+ a `volume_ratio == 3.0` assertion in `test_fetch_positions_offline`.

### 6.24 Review-backlog item 11: watchlist lost-update race + atomic save (2026-07-18)

Suite **67/67 offline** (+3 tests). First Phase 2 item; 21 findings remain (7 med, 14 low).

The app session and the half-hourly `eod_trigger` job both rewrote `watchlist.json` whole
from in-memory copies loaded long before the write — each could clobber the other's frozen
pivots/edits — and `save_watchlist`'s truncate-in-place write could leave a half-written
file that `load_watchlist` silently reads back as `[]`. Three-part fix:

- **Atomic save** (`export.save_watchlist`): serialize to a pid-suffixed sibling temp file,
  then `os.replace()` over the target. A crash mid-write leaves the old file intact; a
  failure is still swallowed (temp cleaned up best-effort).
- **Pure merge helper** (`export.merge_frozen_pivots(primary, donor)`): primary keeps
  membership/order/notes/its own pivots; an UNFROZEN primary entry adopts the donor's
  frozen pivot (`judged_pivot`/`date_added`/`pivot_source`); donor-only tickers are never
  resurrected.
- **Both writers merge just before saving.** App `_wl_persist()` saves
  `merge(session, disk)` — a stale session rewrite keeps pivots the trigger froze
  meanwhile, and the merged list becomes the session copy so the UI shows adopted pivots
  (this also retires §6.14's "refresh the app to see auto-frozen pivots" caveat for any
  session that mutates the list). `eod_trigger.build_report` saves
  `merge(disk_now, frozen_copies)` — re-reads the file AFTER the slow price fetch, so a
  removal / 📌 re-freeze / add landed mid-fetch survives, and auto pivots land only on
  entries still unfrozen on disk.

Remaining window: two writes inside the same few-ms load→replace span — last writer wins
whole-file, but always a valid file, and the next merge-save converges. Tests:
`test_save_watchlist_atomic`, `test_watchlist_merge_frozen_pivots`,
`test_eod_trigger_merges_concurrent_watchlist_edit` (simulates the app writing during the
trigger's fetch).

### 6.25 New-day 2y-refetch avalanche fixed; scans always top-up (2026-07-19) — item 12 DONE

Suite **68/68 offline** (+1 test, +1 case). User-reported: opening the app on a new day
(next morning / the weekend) re-downloaded ~2 years of history for much of the universe.

**Root cause — provisional bars poisoning the split check.** Any market-hours scan persists
each name's parquet with TODAY's PROVISIONAL close (the price at fetch time) — for the
whole ~3.8k universe. `_merge_incremental` protected that bar from its split/dividend
divergence check only *same-day* (`common < today`); on the NEXT day the bar counts as
settled, the delta fetch returns the true close, and any name that moved >`SPLIT_TOL`
(0.5%) after the scan tripped `needs_full` → full 2y refetch. Roughly half the universe
moves 0.5% in a few hours → the avalanche. Fix: the cache's FINAL bar is now also excluded
from the divergence comparison whenever older overlap days exist (a genuine re-adjustment
rescales those too, so real splits are still caught — regression case (d) +
`test_incremental_price_cache_appends_delta` (b) keeps the sole-overlap-bar split firing);
the merge overwrites the provisional bar with the settled fetch regardless (`keep="last"`).

**Scan fetch policy — always incremental (user's spec).** `run_scan` now passes
`max_age_days=0.0` for both the universe and SPY (`get_spy` gained the param): a cache
already holding today's bar re-fetches just the latest bars; a cache last written on an
earlier day fetches only the missing days; only cold names (or a true re-baseline) pay the
full 2y download. This replaces the mtime `max_age_days=1.0` fresh-serve for scans — which
also silently served STALE morning prices on same-day reopens. Cost: a fully-warm scan now
includes the ~38-chunk delta sweep (~1-2 min for full_us) instead of pure parquet reads;
the screening loop dominates scan time anyway, and prices are always current-bar.

**Re-scan button = top-up (backlog item 12); Advanced ⟳ = full re-download.** The Re-scan
button no longer sets `force=True` (which full-refetched 2y × ~3.8k per click); it pops the
scan memo + bumps the nonce and the always-on top-up refreshes the latest bars. The
explicit full-history escape hatch is a separate "⟳ Full re-download (2y, slow)" button
(key `full_refetch`) inside a sidebar "⚙ Advanced" expander (misclick friction on purpose)
→ `run_scan(force=True)` — for re-baselining suspect caches only; a NEWLY LISTED ticker
needs nothing special (no cache → automatic full fetch on any normal scan). Tests:
`test_run_scan_uses_topup_fetch`; case (d) in `test_incremental_price_cache_appends_delta`;
`test_full_redownload_button_forces_scan` (initial scan force=False, ⟳ click →
force=True). Suite 69/69.

### 6.26 Trade-plan per-buy checkboxes (2026-07-19)

Suite **70/70 offline** (+1 test). User request: with a 10-name watchlist mostly inside
earnings no-fly windows, the plan needed per-ticker include/exclude instead of all-or-nothing.

- Each BUY row in the plan preview is now a **checkbox** (label = the old order line:
  shares @ price, extended/capped/📌-pivot flags, earnings ⚠). Held rows keep their plain
  caption (no checkbox — they're stop re-arms, not buys).
- **Defaults encode the strategy:** an earnings-flagged buy (`_earnings_flag`, ≤21d) starts
  **UNCHECKED**; everything else starts checked. The footnote explains and the user can tick
  one back on to override. Checkbox keys carry the build nonce (`buy_<T>_<n>`, same pattern
  as the stop inputs) so a fresh Build re-seeds defaults instead of retaining stale picks.
- The caption shows `K/N buy(s) selected · ~$T est.` (total counts only checked buys);
  extended/capped footnotes describe checked buys; the earnings footnote scopes to all
  buyable rows (it explains why some start unchecked). An unchecked row's stop input is
  disabled and its risk-to-stop caption suppressed.
- **Submit sends only checked buys**; held names always pass through (stop re-arm). The
  Submit button disables when nothing would be sent. Test:
  `test_trade_plan_buy_checkboxes_filter_submit` (defaults, caption, submit filtering via a
  patched `submit_buy_plan`).

### 6.27 Positions page: earnings-aware sell advisories + manual selling (2026-07-20)

Suite **74/74 offline** (+4 tests, 2 extended). User request after the exit-rules discussion
(and the PGC/RSI audit): surface the cushion rule on the Positions page and allow selling
from it. The app still NEVER sells on its own — every sell is a two-step human confirm.

**Advisories/table (`trade.py` + `pages/2_Positions.py`).** `fetch_positions` now enriches
each position with `next_earnings`/`earnings_in` (best-effort `get_fundamentals` per symbol —
weekly JSON cache, serial is fine behind the page's cache_data — + lazy
`scan._days_to_earnings`) and `stage` (new pure `position_stage`: underwater / fresh <16% /
working <20% / well in profit — same constants as `suggest_stop`'s auto ladder).
`position_advisories` gained the two cushion rules, gated on a KNOWN upcoming report
(`0 ≤ earnings_in ≤ EARNINGS_SOON_DAYS`, new trade.py mirror constant = 21) and a KNOWN gain:
loss → "exit or reduce before the report"; gain < `EARNINGS_CUSHION_MIN` (0.08) → "consider
trimming". Just-reported (negative days) or unknown data stays silent. Table adds Stage +
Earnings columns (⚠ inside the window); old-shape dicts degrade via `.get`.

**Manual sell (`trade.submit_position_sell(symbol, qty)`).** Per-position expander: qty
input (seeded via session_state, NOT `value=` — presets write the same key and the double
default logs a warning) + ¼/½/All presets (`on_click` callbacks — the only legal way to set
another widget's key) + "Sell (market)" → freezes `{symbol, qty, held, stop}` into
`sell_pending` (a later qty edit can't change what the confirm submits) → warning summary
with the stop consequence + Confirm/Cancel. Keys `sellqty/sellq4/sellq2/sellqa/sell/sellgo/
sellno_{sym}_{nonce}`; results render like `rearm_result`.

Backend invariant (shares under a GTC stop are RESERVED at Alpaca): **cancel covering
stops → market SELL (DAY, `client_order_id="SEPAsell-…"`) → re-place the stop for any
remainder at the SAME level (GTC, `SEPAstop-`)**. If the sell fails AFTER the cancel, the
previous stop is RESTORED for the full held qty — review finding #14's cancel-before-place
gap done right in the new path (the old `_rearm_gtc_stop` gap remains backlog item 14).
Re-place/restore failures are loudly reported in `detail` and the no-stop advisory catches
them next refresh. No $50 floor / 10% cap / tradability gate — sells are risk-reducing, same
exemption as re-arm. `SEPAsell-` added to `SEPA_TAG_PREFIXES` so the journal tags manual
sells (episode-level `startswith` tuple — no other journal change).

Tests: `test_submit_position_sell`, `test_submit_position_sell_restores_stop_on_failure`,
`test_position_stage`, `test_positions_page_sell_flow` (AppTest: seed→preset→confirm/cancel);
extended `test_position_advisories` + `test_fetch_positions_offline` (now MUST patch
`data_feed.get_fundamentals` — module-attr, same try/finally).

### 6.28 Scan download-transparency log (2026-07-22)

Suite **75/75 offline** (+1 test). User request: the fetch phase showed only a ticker name on
the progress bar — no hint whether a name costs a 2-day top-up or a 2-year download.

- `get_many_prices`'s per-name `_emit` now carries a detail through the progress label
  (`"SYM: detail"`): `cached (fresh)` (fresh-serve), the **missing-days range**
  `M/D/YYYY - M/D/YYYY` (incremental; helpers `_fmt_us`/`_incr_detail` — a single date when
  only today's provisional bar needs refreshing; the 5-day split-detection overlap is
  deliberately not shown), or `full history (2y)` (cold/re-baseline). `run_scan` keeps its
  `"Prices · "` phase prefix unchanged.
- `app.py`: a scrolling download log (`st.empty` + 14-line `deque` tail, repainted ~every 5
  names; the bar keeps its ~25-name throttle) renders `Downloading SYM: detail` lines under
  the progress bar during the price phase only; both slots clear when the scan ends.
  Screening-phase labels are skipped (bar only, as before).

Tests: `test_get_many_prices_progress_labels` (all four label shapes, offline);
`test_streamlit_app_renders_offline`'s fake scan now emits phase-prefixed labels so the log
path runs under AppTest.

### 6.29 Scan freshness window: no re-download within 30 minutes (2026-07-22)

Suite **75/75**. Amends §6.25's always-top-up policy: `run_scan` now passes
`max_age_days = PRICE_FRESH_MINUTES/(24·60)` (new scan.py constant, **30 min**) for both the
universe and SPY, so a cache fetched within the last half hour is served as-is (the §6.28
log shows those as `cached (fresh)`) — reopening the app or clicking Re-scan minutes after a
scan costs zero network. Unchanged on purpose: `eod_trigger` keeps `max_age_days=0.0` (the
half-hourly volume gate needs the live bar), `freshen_prices` keeps `0.0` (order sizing wants
current prices; it's ~10 names), and the Advanced ⟳ `force=True` bypasses the window. The
gate is per-name file mtime, so a mixed cache tops up only its stale names. Pinned in
`test_run_scan_uses_topup_fetch` (window value + both call sites). **[Amended by §6.37: a
cache written with no market session since (post-close/weekend/pre-open) is served at ANY
wall-clock age — including for the `0.0` consumers, since a post-close fetch IS the finalized
close; negative `max_age_days` is the tests' bypass sentinel.]**

### 6.30 Review-backlog Phase 2 items 13-17 fixed — Phase 2 COMPLETE (2026-07-22)

Suite **80/80 offline** (+5 tests, 2 extended/updated). 15 findings remain (4 med, 11 low)
→ Phase 3 next. Order worked: 16 → 15 → 14 → 13 → 17.

- **16 — hidden credentials error.** The account-error warning renders for ANY built plan
  (hoisted above `if _plan:` — a missing-creds build produces exactly the empty plan that
  used to hide it); the target-account caption is guarded against the error case.
- **15 — dotted tickers.** `make_entry` applies the yfinance dash convention
  (`BRK.B → BRK-B`, mirroring `data_feed.normalize` without importing it). One choke point
  covers all adds and heals legacy dotted `watchlist.json` at load; `parse_ticker_list`
  stays dot-preserving (its pin test untouched).
- **14 — re-arm cancel-before-place gap.** `submit_position_sell`'s restore pattern ported
  into `_rearm_gtc_stop` (`_try_place` + `_failed`): a rejected replacement restores the
  previous stop at its OLD level for the full held qty (one attempt); failures return
  `status="failed"` with restored/arm-manually detail instead of raising. Success paths
  byte-identical (pinned by the two ratchet tests).
- **13 — full-fetch silent drops.** Extract-first restructure: partially-failed batches get
  ONE subset-retry batch (whole-batch-empty was already retried in `_download_batch`);
  still-failed names with a parquet serve the STALE cache (not re-persisted — mtime must
  not enter §6.29's fresh window) with honest log labels ("FAILED — stale cache served" /
  "FAILED (no data)"). The §6.28 pinned label test updated for the failed-cold case only.
- **17 — NYSE half days.** Pure `_early_close` hardcodes the three recurring early closes
  (Jul 3 Mon-Thu; day after 4th Thu of Nov; Dec 24 Mon-Thu — Friday instances are observed
  FULL holidays, excluded; one-off closes not modeled — user decision, no new dep).
  Session clock: `_session_len_min` 210 / `_intraday_cutoff_min` 13:05 on those days →
  correct `volume_pace` and settled-vs-intraday flag. Volume GATE scaled ×(390/210) on a
  fresh half-day bar (user decision — a heavy half session can confirm); RAW ratio still
  displayed, scaled value in new `volume_ratio_50_scaled`, per-name/report `early_close`
  flags, notes in `format_report` (ASCII) + the app sidebar. `volume_pace` deliberately
  NOT scaled (it previews the raw close ratio; the shorter divisor already fixes it).

Tests: `test_trade_account_error_shown_with_empty_plan`,
`test_rearm_gtc_stop_restores_stop_on_failure`,
`test_get_many_prices_full_fetch_serves_stale_cache`,
`test_get_many_prices_retries_failed_subset`, `test_early_close_calendar_and_gate`;
extended `test_watchlist_entry_normalization`, updated `test_get_many_prices_progress_labels`
case (d).

### 6.31 Review-backlog Phase 3 items 18-22 fixed — Phase 3 COMPLETE (2026-07-22)

Suite **85/85 offline** (+5 tests, 1 extended). 10 findings remain (1 med — item 23 — plus
9 low, all Phase 4 dead-code sweep). Order worked: 20 → 21 → 22 → 19 → 18 (21/22 cheapen
exactly the loop 18's loosest-gate scan runs for all 8/8 passers).

- **20 — guarded vendored import.** The `.screener` try/except block deleted from
  `screening/__init__.py` (import/guard-edit category; NO vendored files deleted, no
  function bodies touched); PROVENANCE.md change #2 rewritten honestly (the only consumer
  of those re-exports was the equally-dead vendored `notifications/scheduler.py`). The
  isolation test's subprocess now also imports `cockpit.scan` and asserts `.screener`
  never loads. **~1.7s off every process start.**
- **21 — BBWP fast path.** `bollinger_bandwidth_percentile_last` (series fn kept):
  `_pctrank` applied once to `tail(126)` — bit-identical to the rolling series' final
  row; scan uses it. None-on-NaN-tail replaces digging up a stale older percentile.
- **22 — VCP hot-loop hoist.** `rmv_now`/`week_52_high`/`breakout_volume_ratio` computed
  once in `detect_vcp` (after `_empty` early-returns), passed as kwargs to `_detect_at`
  (which lost its unused `price_data` param). **Benchmark line byte-identical** (A=79
  YES 53 / B=114 / C=7 / YES 72). scan.py's 5th RMV → `_rmv_display` (reuses the
  detector's 1-dp value when `zz_threshold` set; computes from `df` for `_empty` results
  so the 100.0 sentinel never surfaces).
- **19 — trade-plan invalidation.** `_invalidate_trade_plan()` (pops plan+result, never
  bumps `trade_build_n`) wired into Re-scan (button now `key="rescan"`), Advanced,
  `_wl_add/_wl_remove/_wl_clear/_wl_freeze` (bulk adders covered via `_wl_add`), and
  `on_change` on `trade_mode` + the four `trade_amt_*` widgets. A shown submit result
  clears on the next invalidating tweak (accepted).
- **18 — instant filter tweaks.** Memo key = (universe, min_criteria, nonce); the scan
  runs ONCE at the loosest gates; sliders apply via new pure `scan.filter_candidates`
  (gate-parity masks, new frame — the memoized ScanResult is never mutated). Filtered
  view → table/picker/search; watchlist CSV keeps the UNfiltered frame (export-stable
  under sliders — deliberate); caption gains "· N after filters" (n_passed now literally
  means the 8/8 count).

Tests: `test_filter_candidates_matches_scan_gates`, `test_filter_tweak_reuses_scan_memo`,
`test_trade_plan_invalidated_on_events`, `test_bbwp_last_matches_series`,
`test_scan_rmv_display_reuses_vcp`; extended `test_data_feed_isolated_from_vendored_data_layer`.
Perf felt: filter tweaks instant (was multi-minute each way, worse on toggle-back);
cold start ~1.7s faster; screening phase trimmed (RMV 5×→1×, BBWP's ~450 Python callbacks
per passer → one vectorized read).

### 6.32 Review-backlog Phase 4 items 23-32 — REVIEW CLOSED (2026-07-22)

Suite **86/86 offline** (+1 test, 1 extended). **The 2026-07-16 multi-agent review is fully
closed**: 34 verified findings → 30 fixed with tests, 2 closed kept-per-PROVENANCE, 2
refuted. `cockpit/REVIEW_BACKLOG.md` header now records the final state.

- **26/27/29 (trivial):** no-op `max_workers` dropped from `get_many_prices`/`run_scan`;
  dead `WATCHLIST_KEYS` deleted (`PIVOT_SOURCES` stays — load-bearing); `STATUSES` kept
  and made load-bearing (`test_check_triggers_pure` asserts every emitted status ∈ it —
  a future "crossed" status must register).
- **28 (TR/Bollinger dedup):** one `_true_range` (raw — the Keltner ATR keeps the
  unscaled variant) + one `true_range_pct` in `cockpit/indicators.py`; RMV + vcp's
  adaptive-threshold/dead-tape use the % variant (`vcp._true_range_pct` deleted);
  ttm_squeeze + both BBWP fns take their band legs from `bollinger_bands()`. Benchmark
  byte-identical.
- **32 (vendored calculate_sma):** phase_indicators' local copy → `from .indicators
  import calculate_sma` (the exported copy) — PROVENANCE change #3; identical returns.
- **25 (get_prices):** now a thin wrapper over `get_many_prices([sym])` (~50 dup lines
  gone; SPY inherits retry + log labels); the pinned incremental-cache test passes
  unchanged.
- **24 (universe UI, user decision):** the TESTING selectbox is GONE — `universe =
  "full_us"` constant + unconditional caption; sp500/tickers fetchers remain as offline
  fallbacks/programmatic options only.
- **30/31 (user decision):** vendored `analysis/` + `data/` KEPT, closed as unreachable —
  item 20 severed the only import path; PROVENANCE verbatim-copy property intact
  (matches the refuted notifications/batch-processor precedent).
- **23 (the MEDIUM — behavior change, A/B-gated KEEP):** new
  `scan.detect_breakout_prior_high` wrapper (vendored file untouched; backtest keeps old
  behavior): prior-bar 60/20-day highs make the Base/Pivot Breakout branches REACHABLE,
  preserving VCP precedence + the phase-1/2 gate; both cockpit call sites
  (screen_universe, compute_scan_pivot) share it, so app/frozen-pivot consistency holds.
  **A/B over the 200 fixtures: 12/200 pivots changed, ALL moved DOWN (median −5.3%,
  max −15%) from the 52-wk-high fallback to real base highs; VCP breakouts untouched;
  benchmark byte-identical** (tiers come from detect_vcp, independent of
  detect_breakout). Practical effect: `breakout_today` now honestly fires on fresh
  60/20-day-high closes, and affected names (EXEL among the fixtures) get actionable
  base-high pivots — future freezes/plans anchor to real structure; already-frozen 📌
  pivots are untouched. Test: `test_breakout_wrapper_fires_prior_bar_highs`.

### 6.33 Watchlist pills picker + the test that wiped the real watchlist (2026-07-22)

Suite **87/87**. Two parts, same day (the first cut — a separate "remove" expander — was
replaced hours later by the user's requested design):

**Pills UI (user spec).** The sidebar multiselect is now CONTROLLED: its selected pills
ARE the watchlist (the ×-to-dismiss chips in a box, like the uploader). The page re-seeds
`st.session_state["wl_picker"] = _wl_tickers()` every run BEFORE the widget, so changes
made anywhere (⭐ add, 📌, .txt upload, the EOD job's auto-freeze merge) always show —
including STALE names absent from the scan, which the old chart-toggle removal could never
reach. `_wl_sync_from_picker` (on_change) syncs back: new picks → unfrozen entries;
dismissed pills → entry + frozen pivot deleted, persisted, trade plan invalidated.
Help text warns that removal forgets the 📌 level (re-add auto-freezes at the CURRENT
pivot). `_wl_add_from_picker`/the remove expander are gone. Test:
`test_watchlist_picker_pills_sync` (seed incl. stale, dismiss, add, persistence, pivot
intact, plan invalidated).

**⚠ INCIDENT — the real watchlist.json was being wiped by the suite.** The user reported
"my watchlist keeps being reset": `test_trade_plan_invalidated_on_events` (added with item
19) ran app.py WITHOUT patching `cache.WATCHLIST_JSON` and its scenario (d) clicks 🗑
Clear-watchlist → `_wl_persist` saved an EMPTY list to the REAL
`data/cockpit/watchlist.json` on every suite run, destroying all frozen judged pivots.
Fixed: that test + the four other app AppTests missing the patch now redirect
WATCHLIST_JSON/TRIGGERS_DIR to a tempdir; the file was RESTORED from session records (all
10 entries with their judged pivots). **RULE: every AppTest that executes app.py MUST
patch `cache.WATCHLIST_JSON` (+ TRIGGERS_DIR) — even render-only tests; one later edit
that adds a mutation is all it takes.** Verified: a full suite run now leaves the real
file byte-intact.

### 6.34 UNTRACKED trigger status + sidebar chart-jump buttons (2026-07-23)

Suite **89/89** (+2 tests). Two user features, plus one adopted user edit.

- **UNTRACKED (the §6.19 "crossed"-adjacent idea, user-specified):** a watchlist name that
  no longer passes the 8/8 trend template has left the scan table — it STAYS on the list,
  but `check_one` no longer evaluates its trigger (a "breakout" on a broken-down base is
  noise). The headless job re-derives the template per name on its own frame
  (`classify_phase` + `validate_minervini_trend_template`; new `TEMPLATE_CRITERIA = 8`
  mirror constant); frames < 200 rows or template-chain errors FAIL OPEN (keep
  evaluating). New status `"untracked"` registered in STATUSES (the load-bearing
  vocabulary test forced it — as designed); precedence no_data → untracked → no_pivot →
  stale → …; summary bucket + ASCII console note + 🚫 sidebar caption. Test:
  `test_untracked_watchlist_names` (would-be trigger suppressed; outranks stale;
  template-passing control still fires — pivot must sit within the 5% zone or it files as
  extended).
- **📈 chart-jump buttons:** each trigger-report row in the sidebar carries a small 📈
  button that jumps the MAIN chart to that ticker. The panel is a fragment, so the click
  escalates via `st.rerun(scope="app")` (with a `StreamlitAPIException` → plain-rerun
  fallback for AppTest's inline-fragment execution — do NOT catch broad `Exception`
  there: `RerunException` IS the mechanism). The jump sets `chart_pick`, which the main
  body POPS after the table selection (one-run override; table regains control on the
  next interaction). Out-of-scan names (untracked/stale) render the button DISABLED with
  an explanatory tooltip — payloads are the authority. Test:
  `test_trigger_sidebar_chart_button` (target chosen at runtime as the non-default
  candidate — the displayed table's first row is sort-dependent).
- **Adopted user edit:** the 🗑 Clear-watchlist button + `_wl_clear` were removed from
  app.py by the user (post-wipe-incident; the pills give per-name removal). The
  invalidation test's scenario (d) now dismisses a pill instead.

### 6.35 Manual "Check triggers now" button (2026-07-27)

Suite **90/90** (+1 test). Motivated by the 7/23-24 outage: the Task Scheduler job is
"Interactive only" + "No Start On Batteries", so a closed/logged-out laptop silently skips
checks (Thursday's 1.7-2.5× breakouts went unalerted).

- `_trigger_report_panel` (the 60s sidebar fragment) now opens with a **🔔 Check triggers
  now** button (`key="trigger_check_now"`): it runs `eod_trigger.build_report()` in-process
  under a spinner — the SAME pipeline as scripts/eod_trigger.bat (top up watchlist bars at
  `max_age_days=0.0`, auto-freeze missing pivots with the lost-update-safe disk merge,
  evaluate) — then `save_trigger_report()` to `cache.TRIGGERS_DIR`.
- **Ordering is the trick:** the button handler sits BEFORE `load_latest_trigger_report`,
  so the fresh report renders in the same fragment pass — no rerun call, hence no
  RerunException-vs-broad-except hazard (the `except Exception` around the check is safe
  and deliberate: yfinance failures degrade to `st.warning`, panel stays alive, last good
  report stays up). Button click = fragment-scope rerun; the memoized scan never re-runs.
- The `---` divider moved above the button (drawn always now); the empty-state caption
  points at the button first, the .bat second. `build_report` is imported lazily at click
  time (`from ...eod_trigger import build_report`) — patch `eod_trigger.build_report` in
  tests and the from-import resolves the mock.
- Test: `test_trigger_check_now_button` — empty TRIGGERS_DIR shows the empty-state, click
  writes `triggers_<date>.json` + renders the canned name same-pass (build_report called
  exactly once), then a `side_effect=RuntimeError` click surfaces "Trigger check failed"
  with the last good report still rendered. (WATCHLIST_JSON + TRIGGERS_DIR tempdir-patched
  per the post-wipe hard rule.)

### 6.36 §6.19's post-breakout freeze semantics BUILT: "crossed" status + freeze-time warning (2026-08-04)

Suite **91/91 offline** (+1 test). The §6.19 open item, shipped after its scenario went live:
BOKF closed 2026-08-03 nine cents above its frozen 143.65 pivot on 0.65× volume —
`close_above_pivot: true, triggered: false`, rendered identically to a name still basing.
"It left without me" and "nothing happening" are now visually distinct states.

- **`triggers.py`:** new `crossed` boolean (`close_above_pivot and not volume_confirmed`)
  and status, registered in `STATUSES`. Precedence: stale → extended → triggered →
  **crossed** → watch — a stale bar or an extended (>+5%) name still files under those
  first, and an in-zone volume-confirmed close still wins as `triggered`. The summary gains
  a `crossed` bucket; `format_report` prints the count plus an ASCII explainer (quiet drift
  ≠ buy; wait for the ≥1.5× close or plan a pullback/secondary entry). Old reports without
  the key render unchanged (`.get` throughout).
- **`app.py` sidebar:** ↗ icon for crossed rows + a one-line explainer caption whenever the
  report's summary lists crossed names.
- **`app.py` freeze paths (⭐ add / 📌 freeze, one shared caption):** when the charted
  name's last close is ABOVE the pivot the button would freeze, a ⚠︎ caption warns that the
  armed trigger may never re-fire (the PECO bootstrap case) and points at the
  pullback/secondary-entry plan. Display only — nothing is gated; recall-first philosophy
  intact.
- **Tests:** `test_check_triggers_pure` case (b) + summary now pin `crossed`;
  `test_early_close_calendar_and_gate`'s normal-day unconfirmed cross likewise;
  `test_trigger_report_sidebar_renders` renders a crossed row + the explainer caption; NEW
  `test_freeze_warning_post_breakout` (AppTest) asserts the warning fires with price above
  the would-be-frozen pivot and stays silent below it.
- **Deferred (unchanged from §6.19):** re-arm-on-re-cross semantics, a retroactive
  freeze-time "already broke out on <date>" lookback, and a pullback/secondary-entry
  trigger type — each waits on a real case to force its design.
### 6.37 Settled-close cache serve — no top-up when no session has elapsed (2026-08-04)

Suite **93/93 offline** (+2). User-reported: an after-close scan with a 4:35 pm cache still
ran the 30-minute-window top-up sweep, though no new bar could exist. Now a price parquet
written with **no market session since** is served as-is at any wall-clock age.

- **`triggers.no_session_since(mtime_epoch, now=None)`** (triggers.py owns the session
  clock): True when the interval mtime→now contains no weekday 09:30→settled-cutoff time
  (16:05, or 13:05 on `_early_close` half days). Covers post-close evenings, weekends, and
  pre-open; the 16:00–16:05 settle window counts as session time, so a 16:02 cache (possibly
  provisional volume) correctly reads stale. Full-market holidays NOT modeled (same stance
  as `_early_close`) — they read as sessions, failing SAFE (a needless cheap top-up, never a
  stale serve). Epoch→ET via tz-aware conversion (machine-tz-proof).
- **`data_feed._cache_settled(path)`** (lazy triggers import — no module-level cycle;
  never raises) + the `get_many_prices` gate: `age <= max_age_days` OR (`max_age_days >= 0`
  AND settled) → serve, labeled **"cached (settled close)"** in the download log.
  `max_age_days=0` consumers (EOD trigger, `freshen_prices`) honor the settled gate — a
  post-close fetch IS the finalized close. **Negative max_age_days (the tests' sentinel)
  bypasses it** so the top-up/refetch tests stay deterministic after hours; `force=True`
  bypasses everything. `get_prices`/`get_spy` inherit via the wrapper.
- Intraday behavior unchanged: during the session ANY elapsed time is session time → the
  predicate is False → the 30-min window governs exactly as before.
- Side benefit: the weekend hunt no longer pays the ~38-chunk delta sweep (§6.29 noted
  ~1-2 min on full_us) when the cache was written after Friday's close.
- **Tests:** `test_no_session_since_calendar` (pinned-now truth table: evening, settle
  window, weekend, pre-open Monday, mid-session, early-close 13:05 boundary, future mtime)
  and `test_get_many_prices_settled_cache_served` (3-hour-old parquet + tight window:
  served with the predicate patched True, top-up fires with it patched False).

- **Addendum (same day, user request):** the Step-4 **Pivot metric got a `help` tooltip**
  distinguishing the DETECTED pivot (recomputed every scan, drifts; feeds the buy-zone/
  stop/target tiles) from the watchlist's FROZEN 📌 pivot (what triggers fire on and trade
  plans price off) — when the charted name has a frozen pivot the tooltip shows its value
  and points at the sidebar; otherwise it explains that ⭐/📌 freezes one. Display-only, no
  test assert per the §6.15 convention. (Prompted by PEBK reading 43.47 detected vs 43.66
  frozen on 2026-08-04 — the settled close 43.50 sat between the two, so the panels
  appeared to disagree.)

### 6.38 OTO buys are now GTC end-to-end — the expiring-DAY-stop-leg hole (2026-08-04)

Suite **93/93**. Live incident, user-reported as "the stop never placed": a risk-sized PEBK
buy at **15:58** filled as a DAY OTO; its stop leg (42.38) activated on the fill and
**expired at the 16:00 close two minutes later** — the position sat overnight with no stop
until a manual Positions-page re-arm at 20:27 (GTC 40.65). Order log evidence: the OTO
parent `SEPAoto-PEBK-…` (filled, tif day) with leg `sell stop 42.38 tif day status EXPIRED`.

- **Root cause:** §6.3/§6.6 locked the OTO to DAY on the belief "Alpaca market entries
  can't be GTC," with the leg "promoted to GTC on the next re-arm" — but the promotion only
  happens when the user takes another action, so any intraday fill's protection died at
  that day's close. §6.6 itself had flagged "GTC market OTO server-accepted?" as
  untested-live.
- **Probe (2026-08-04, paper):** a 1-share GTC OTO market buy + stop leg was ACCEPTED —
  parent `tif gtc class oto`, leg `sell stop tif gtc status held` — then canceled pre-open.
  The §6.6 belief was simply wrong.
- **Fix:** `submit_buy_plan`'s OTO request is now `TimeInForce.GTC` — the leg inherits GTC,
  rests as a persistent stop after the fill, and the held-name ratchet manages (only
  raises) it from the next re-arm on. The no-stop (`attach_stop=False`) naked buy stays
  DAY. Test: `test_submit_buy_plan_stop_logic` now pins `mreq.time_in_force == GTC` on the
  OTO request.

### 6.39 RS-line-at-new-high flag + the §6.3 held-name stop gap closed (2026-08-05)

Suite **95/95** (+2). The last two noted-but-unbuilt cockpit items, both user-requested.

**RS line at new high before price (§6.5 item 7's leftover, now built).**
- `scan.rs_line_at_high(df, spy_close, window=252, tol=0.002, min_days=126)`: the RS LINE
  (stock close ÷ SPY close, index-aligned) within 0.2% of its trailing-52-week max; None
  under ~6 months of overlap (unknown ≠ failed). The funnel combines it with "price still
  BELOW its own 52-week high" (from phase_info) → `rs_nh`: IBD/MarketSmith's blue dot —
  outperformance while still basing, the institutional-accumulation tell. A name breaking
  to a price new high today correctly reads False (no longer "before price").
- Surfaced: `rs_nh` column (Fuel group, after RS rating, with header tooltip) + payload
  key + a Step-4 "RS line:" row in the base section (✅ at-high-before-price / — no
  divergence / n/a). Advisory only, like every base read.
- Calibration knobs (not gospel): 0.2% at-high tolerance, 252-day window, 126-day minimum.

**Held-name stop gap at build (§6.3's tombstone, closed).**
- `build_buy_plan` gains optional `held: {ticker: shares}`; the app passes the
  `fetch_held_shares()` dict it already fetched for preview marking. A HELD name whose buy
  fails a sizing gate (rounds < 1 share / under the $50 floor) now emits a zero-share
  `stop_only=True` row (`_stop_only_entry`) instead of a skip — submit's held path sends
  no buy regardless and re-arms the GTC stop from `stop_price`, so stop maintenance no
  longer silently vanishes with the buy. Pre-level skips (not in scan / no price / stale)
  and no-computable-stop cases still skip — nothing to arm. `held` omitted = byte-for-byte
  old behavior (builder stays pure; all existing tests untouched).
- Preview: stop-only rows land in the existing held-row rendering ("already held (N sh) ·
  stop re-arm only, no buy") — excluded from buyable count/est-value total by the
  established `_held` check.
- Also fixed in passing: the attach-stop toggle's help text still described the §6.38-era
  DAY stop leg — now says GTC OTO.
- **Tests:** `test_rs_line_new_high_flag` (helper truth table + funnel column/payload) and
  `test_build_buy_plan_held_stop_only` (held vs un-held vs not-in-scan behavior).

### 6.40 Background scan worker — starts on any page, survives page switches (2026-08-06)

Suite **95/95** (no new tests; all existing AppTests pass unchanged). User pain, twofold:
(a) the scan only started when the scan page rendered — opening the app on
Positions/Journal left the multi-minute cold scan un-begun; (b) navigating to another page
mid-fetch KILLED the fetch, because Streamlit cancels the running script on navigation and
the scan ran inline in that script (the in-scan `st.progress` calls are where the
cancellation lands).

- **`cockpit/scan_worker.py` (new).** `ScanWorker` runs `scan.run_scan` in a daemon
  thread that never touches Streamlit APIs — page switches kill the script run, not the
  thread. One worker per browser session (`get_worker()` in `st.session_state`, so
  AppTest sessions stay isolated); the actual scan call is serialized process-wide
  (`_SCAN_SERIAL`) so laptop + phone sessions can't duplicate-download. State machine
  idle→running→done|error keyed by `(universe, min_criteria, generation)`;
  `request_rescan(force=)` bumps the generation (= the old nonce+memo-pop in one move) and
  carries the Advanced full-re-download flag. `run_scan` resolves at call time inside the
  thread, so `patch.object(scan, "run_scan", …)` keeps working — no test edits.
- **app.py** drops `_cached_scan`/nonce/spinner for: `ensure_started()` → `wait(grace=3.0)`
  → on None, render progress bar + download-log tail from `snapshot()` and poll via
  `time.sleep(1); st.rerun()`. The grace is anchored to the RUN's start (not the call), so
  a warm/memoized run renders in one pass (AppTest depends on this) while reruns during a
  cold scan fall straight through to the progress view. Error path renders the traceback +
  a Retry button (`request_rescan`) — `ensure_started` deliberately does NOT auto-retry a
  failed key, or a persistent failure would hammer yfinance in the rerun loop.
- **Pages** (Guide/Positions/Journal) call `scan_worker.autostart()` at import: whichever
  page the app opens on, the scan is already warming. `autostart` no-ops when
  `streamlit.testing.v1` is in `sys.modules` — page AppTests don't patch `run_scan`, and a
  real background scan under test would hit the network. (Verified both ways: the tell
  removed + `run_scan` faked, each page kicks the worker exactly once; suite untouched.)
- **Semantics kept:** filter sliders stay out of the scan key (instant post-filters);
  Re-scan = incremental top-up; Advanced ⟳ = force full 2y; a mid-flight run can't be
  cancelled (yfinance has no abort) — a Re-scan during one lands the old result stale and
  the fresh run starts the moment it finishes.
- **Known trade-off:** the watchlist/trigger sidebar and the candidate table still render
  only once the result lands (same as the old spinner behavior) — the progress view
  `st.stop()`s the page. If that grates, the next step is rendering the sidebar from the
  last stale result while the fresh scan runs.

### 6.41 Limit buys in the trade panel (2026-08-09)

Suite **97/97** (+2; also weekend-proofed `test_get_many_prices_progress_labels` — its
"cache ends today" fixture used `bdate_range(end=today)`, which on a weekend ends Friday,
so the suite failed every Saturday/Sunday; the fixture now appends a today-dated bar).
User-requested during the TXRH breakout call: the panel could only send market BUYs, and a
market order queued into the next open fills at the gap print — past the 5% no-chase line
if the name gaps. A limit at the buy-zone top makes the order itself enforce the rule.

- **`build_buy_plan(order_type="limit")`**: each entry gets `limit_price` defaulting to
  its buy-zone TOP (effective pivot × 1.05 — frozen 📌 pivot preferred via the existing
  `pivots` map; no pivot → last close as a marketable cap). Sizing, risk-per-share
  (`limit − stop`), the $50 floor, and the 10% cap all use the LIMIT as basis — the
  worst-case fill for a buy limit is the limit itself (fills lower, never higher), so
  `est_value` is the honest maximum. `"market"` (default) is byte-identical to before,
  `limit_price` None; unknown `order_type` raises.
- **`submit_buy_plan`** keys off the ENTRY (`limit_price` present and > 0 → limit path;
  no new signature): attach-on → **GTC OTO LimitOrderRequest** + stop leg — the §6.38
  end-to-end-GTC shape, so a fill on ANY later day arms its stop the moment it happens;
  attach-off → naked **DAY** limit (expires at the close, like the naked market path).
  Stop validity checks against the LIMIT, not the last close (a stop between close and
  limit is legal). A present-but-non-positive limit (widget edited to 0) is **skipped**,
  never silently downgraded to a market order. Tags unchanged (SEPAoto-/SEPAcockpit-) so
  the journal keeps matching.
- **UI:** "Order type" radio (Market / Limit (no-chase cap)) under the sizing mode,
  plan-invalidating like every sizing widget; `order_type` is stored in the built plan so
  rendering can't drift from what Build produced. Limit rows render ☑ name · limit · stop
  (three columns); per-name limit input seeded from the default, keyed on the build nonce
  like the stops. Live captions: risk-to-stop and stop validity use the edited limit as
  basis; a limit below the last close gets "fills only on a pullback into the zone" (the
  crossed-name secondary-entry mechanic, deliberate); GTC resting behavior is spelled out
  in the radio help (cancel via the Alpaca dashboard; the pending-buy guard blocks
  re-submits while one rests).
- **Semantics note:** an unfilled attach-on limit RESTS (GTC) — chosen over DAY because a
  DAY parent would re-open the §6.38 hole (an intraday fill's DAY stop leg dies at the
  close). The §6.22-item-4 pending-buy guard already prevents double-submits meanwhile.

### 6.42 Process-wide scan store, stale-while-refresh UI, warm-path speedups (2026-08-09)

Suite **107/107** (+7 new, 1 intentionally updated, 1 reverted-by-its-own-gate). Follows
§6.40: the user still hit "fetching again" walls — the result was SESSION-scoped (any
refresh/second device re-scanned), any in-flight scan hid the whole page behind a
progress view, and that view said "Downloading … cached (settled close)" for 100%-cache
passes. User-confirmed spec: serve the last result instantly everywhere; background
refresh at most every 30 min; **the 30-min price-freshness window is REMOVED** (the
throttle replaces it — a scan that runs must BE fresh).

- **`scan_worker.ResultStore`** — process-wide last-completed-scan store keyed
  `(universe, min_criteria)` (no generation: store = process identity), clock-injectable.
  Worker `ensure_started` ADOPTS a newer store entry (any page load/session = instant
  result), `_run` publishes on completion, and a run that queued behind `_SCAN_SERIAL`
  adopts a result that landed while it waited (two cold sessions = ONE scan).
  `latest()` = stale-while-refresh serve. Store holds ONLY the ScanResult — no price
  frames in RAM (user requirement). **Inert under the AppTest tell** (per-session
  isolation; the tell is STICKY process-wide, so worker unit tests inject
  `ScanWorker(store=ResultStore(clock=fake))` explicitly).
- **Throttle:** `try_claim_refresh` grants one background refresh per
  `REFRESH_TTL_SECONDS` (30 min), claim recorded ON START so a slow/failed refresh can't
  retry-loop; the claim branch also covers the ERROR state (a failed refresh retries when
  the window reopens; cold-start errors can't loop — no entry, claim denied). Manual
  Re-scan/full-re-download always run (`adopt_ok=False`) and stamp the claim.
- **Freshness window removed:** `scan.PRICE_FRESH_MINUTES` deleted; `run_scan` passes
  `max_age_days=0.0` (always-top-up, same as eod_trigger). §6.29 is retired; the
  settled-close serve (§6.37) still gives zero-network evenings/weekends.
  `test_run_scan_uses_topup_fetch` updated to pin 0.0.
- **UI (app.py):** `latest()`-first — the table renders instantly from the newest result;
  the full-page progress takeover is now only the true-cold path. A status-line fragment
  (2s while running / 30s idle) shows "data as of HH:MM · ⏳ Reading cache 2100/4198…",
  a failed-refresh caption + Retry, or an "⬆ Updated scan ready — load" button (no
  auto-swap mid-read; any interaction adopts via ensure_started). A changed `as_of`
  invalidates the trade plan (a background refresh is a re-scan for item-19 purposes).
- **Honest progress:** `_on_progress` classifies phases (cache/fetch/screen);
  cache serves never log "Downloading" (bar says "Reading cache"); data_feed label
  STRINGS untouched (pinned by tests).
- **data_feed:** cache-read pre-pass now runs in a `ThreadPoolExecutor(16)`
  (`_classify_cached` per name; emit under a lock, assembly in input order after the
  join — batches stay deterministic; yf.download stays strictly serial). Measured warm
  pass ~8 s → ~4.5 s. `_atomic_to_parquet` (tmp + `os.replace`) at both write sites —
  a torn parquet from the concurrent eod_trigger process used to silently become a full
  network refetch. `_cache_settled` caches the triggers MODULE (attribute lookup stays
  call-time — a test patches `no_session_since` on the module).
- **vcp.py:** `_zigzag_pivots` now feeds the SAME loop plain-Python floats
  (`_zigzag_pivots_ref` = the reference; per-element ndarray indexing boxed a np.float64
  every access). 2.1× on the hottest inner loop (101→49 µs/call);
  `test_zigzag_fast_parity` pins exact pivot equality across all 200 benchmark fixtures
  × 5 thresholds + seeded walks; the 200-chart benchmark line is byte-identical.
- **REVERTED by its own gate — don't re-attempt:** the tail-only SMA-200 stub
  (`Close.iloc[-220:]`) in screen_universe. pandas' rolling mean uses a sliding-sum
  kernel, so the tail slice differs from the full series by ~1 ulp at the consumed
  points (the parity test caught it) — enough to flip a knife-edge 8/8 template gate.
  Determinism beats the modest saving; a comment in scan.py marks the tombstone.
- **Deferred (recorded):** per-name screening memo (intraday provisional bars defeat it;
  the store+throttle make same-day re-screens rare); in-memory frame cache (staleness vs
  the cross-process eod_trigger writer + 120-160 MB).

### 6.43 Review round 2 + the HIGH fix: build-time intent is binding (2026-08-09/10)

Suite **108/108**. A scoped adversarial review (workflow `wf_be005f7b-07b`: 4 lens
finders → 12 deduped findings → 1 refuter each) of cb615cf..fc1aa1e + the §6.42 working
tree confirmed 10 findings (2 refuted). **Ranked open list = REVIEW_BACKLOG.md "Round 2"
section** (supersedes nothing — round 1 stays closed). The single HIGH is FIXED:

- **R2-1: a "stop re-arm only, no buy" row could become a full-size unconsented market
  BUY.** Build snapshots holdings; submit re-checks live; a position closed in the gap
  (its GTC stop firing suffices) dropped the held row into the buy branch — no checkbox
  consent, earnings no-fly never applied, limit stripped (held rows carry
  limit_price=None). Fix: the app stamps `rearm_only` on held-at-build rows in `_final`;
  `submit_buy_plan`'s buy branch skips `rearm_only`/`stop_only`/`shares<1` rows with
  "position closed since the plan was built — no buy sent (rebuild the plan to buy it)".
  Zero-share qty-0 API noise dies with the same guard. Test:
  `test_submit_buy_plan_rearm_only_never_buys` (closed-at-submit / still-held re-arm /
  zero-share / unaffected-sibling cases).
- Remaining: 3 MED code fixes (yf-download lock R2-2, limit-stop-vs-price R2-3,
  overlap-persist settled-gate R2-5), the no-chase honesty/caption work (R2-4), the
  edited-limit cap recompute (R2-9), and 4 LOWs — see the backlog for fixes already
  designed per item.

### 6.44 Review Round 2 CLOSED — all mediums + lows fixed (2026-08-10)

Suite **116/116** (+7 tests). The three risky designs (R2-2/R2-5/R2-6+8) were attacked
by a validation agent before implementation — it verified yfinance 0.2.65 internals, the
full lock-order graph, and refined each design (per-attempt locking; `index.max()`;
passing the resolved store into `_run`); it also ADDED R2-5b. Everything landed per
`REVIEW_BACKLOG.md` "Round 2" (now 🏁 CLOSED, all items ticked with fix + test notes):

- **Trading path:** limit-mode build skips a name whose stop ≥ current price (broken
  base — the marketable-limit instant-stop-out killer); submit validates stop <
  min(limit, price) and recomputes est = shares × limit for the 10% cap (edited limits
  re-enter the guardrail); marketable-limit ⚠ caption for below-pivot rows; radio help
  now states the TRUE limit contract; new `trade.cancel_pending_buys()` + 🗑 sidebar
  button (SEPA-tagged BUYs only) — the missing control for resting GTC limits.
- **Concurrency:** `_YF_LOCK` per-attempt inside `_download_batch` (in-process
  yf.download serialization — 0.2.65's shared._DFS reset + timeout-less spin-wait made
  overlaps wedge-capable; cross-process stays out of scope by design); incremental
  persist gated on the fetch reaching the cache's newest bar, AND settled-serve now
  requires the FRAME to be current (`triggers.frame_settled_current` — closes the
  full-fetch variant; a stale-ending delisted name now re-attempts a cheap top-up
  post-close instead of settle-serving, accepted); `ensure_started` runs an armed
  `_pending_force` before adoption (a Full-re-download can no longer be swallowed then
  detonate in a TTL refresh); `ResultStore.now()` + (store, run_started) passed as a
  pair into `_run` — the adopt check is single-clock by construction (was: fake test
  clock vs machine uptime, deterministic suite failure within ~17 min of boot).
- **Hygiene:** `save_trigger_report` writes tmp+os.replace; `crossed` boolean gains
  `not stale`; eod_trigger.log untracked by the user (`f58ebbb`).
- Meanwhile the user committed §6.42+§6.43 as `ab1ec43`; this section's work is the
  only uncommitted delta on top of `f58ebbb` at time of writing.

### 6.45 Last-scan persistence + progress-view removal (2026-08-11)

Suite **117/117** (+1). User restarted the server and hit the full-page progress view —
the §6.42 store was process-memory only, so a restart had nothing to serve (a known,
accepted trade-off now un-accepted). Two changes, both user-requested:

- **`ResultStore` persists the last completed ScanResult** to
  `data/cockpit/last_scan.pkl` (atomic tmp+os.replace, best-effort, written from the
  worker thread after `put`; `*.pkl` gitignored). A fresh store loads it lazily on the
  first `get` miss: same result, ORIGINAL `completed_wall` ("data as of yesterday
  18:30"), `completed_mono = -inf` — maximally stale, so the TTL throttle grants an
  immediate background refresh, and never adoptable by an in-flight run. Guards:
  `_PERSIST_VERSION` + key match + one load attempt per process; ANY failure fails open
  to a cold scan. `persist_path=None` (unit tests) = zero disk I/O. Net UX: restart →
  yesterday's table instantly + "data as of … ⏳ Reading cache n/m — refreshing in the
  background". Test: `test_scan_store_persists_and_reloads`.
- **The full-page progress bar + scrolling download log are REMOVED** (user: "remove
  everything related; maybe add back later"). The true-cold path (first scan ever on a
  machine, or a failed pickle) shows a plain `st.info` note + the poll loop. scan_worker
  lost the `_log` deque and snapshot's `"log"` key; the phase classification stays (it
  feeds the status line's "Reading cache / Downloading / Screening n/total"). Recover
  the old view from git history (`ab1ec43`-era app.py) if ever wanted.
- Deploy note: the pickle is written on scan COMPLETION — the first restart after
  deploying this shows the cold note once; every restart after that is instant.

### 6.46 Positions page decoupled from the bulk pipeline (2026-08-11)

Suite **119/119** (+2). Live incident: user couldn't sell NMM while a refresh was
mid-download. Root cause: not threading (the scan IS backgrounded) — R2-2's `_YF_LOCK`
correctly serializes every in-process yf.download, and `fetch_positions`' small
enrichment pull (SMA-50/volume advisories) queued behind the sweep's chunks, holding the
whole Positions page (and its sell controls) on the spinner. Selling itself never needed
the pipeline: `submit_position_sell` is pure Alpaca.

- `data_feed.network_busy()` (= `_YF_LOCK.locked()`) + `get_many_prices(...,
  allow_network=False)` cache-only mode: incremental candidates served from their
  pre-pass frames, too-stale names straight from parquet, no-cache names absent, ZERO
  yfinance calls (labels "cached (network busy)" — classified as the cache phase).
- `fetch_positions` passes `allow_network=not network_busy()`: pipeline busy → page
  renders instantly from cache (current_price is Alpaca's regardless; only the SMA-50/
  volume advisories read a bar older — tolerated); pipeline free → unchanged.
- Deliberately NOT applied to `freshen_prices` (trade-plan Build): order pricing should
  wait for fresh bars — bounded to ~one chunk attempt by the per-attempt lock, and the
  staleness guard covers failures.
- TOCTOU accepted: busy can flip right after the check — worst case the old ≤one-chunk
  wait, never wrong data.

### 6.47 Comment de-verbosing: incident references out of source (2026-08-11)

User request. All R2-N / §6.NN / review-date / incident-ticker references stripped from
the seven cockpit source files' comments and docstrings (plus two UI help strings);
each comment keeps its CONSTRAINT, loses its history — the incident ledger lives in the
test docstrings, TESTS.md, and this file. Verified behavior-neutral by AST comparison
(docstrings stripped; only the two intended help-string edits differ); suite 119/119.
Rule going forward: source comments say WHY the code must be this way, never which
bug/review/date produced it.

### 6.49 Trigger task un-blocked on battery (2026-08-12)

No code. "SEPA Intraday Trigger" was registered with `DisallowStartIfOnBatteries` +
`StopIfGoingOnBatteries` (Task Scheduler defaults): an unplugged laptop during market
hours silently skipped the half-hourly runs or killed one mid-run — no breakout
alerts, no error anywhere (the July 23-24 dark window, §6.35). Both flags are now
cleared; the schedule (weekdays 09:30 + 30-min repetition × 7h, `StartWhenAvailable`)
was preserved by the settings round-trip and verified via `Get-ScheduledTaskInfo`
(LastTaskResult 0, sane NextRunTime).

Re-apply after any re-registration (no committed script — §6.14 precedent):

```powershell
Get-ScheduledTask -TaskName "SEPA Intraday Trigger" | ForEach-Object {
  $_.Settings.DisallowStartIfOnBatteries = $false
  $_.Settings.StopIfGoingOnBatteries = $false
  Set-ScheduledTask -InputObject $_ }
# verify: (Get-ScheduledTask -TaskName "SEPA Intraday Trigger").Settings |
#   Select-Object DisallowStartIfOnBatteries, StopIfGoingOnBatteries   # both False
```

Deliberately kept: `LogonType Interactive` (runs still skip while logged out — the
user is logged in during market hours; switching to run-when-logged-off needs a
stored credential) and the laptop as host (no LAN-box migration).

### 6.50 `pullback` trigger status — the low-risk secondary entry (2026-08-12)

Suite **120/120**. The §6.36-deferred pullback trigger, prompted by the 8/9 hunt
(127/208 Tier-A names crossed without volume — dead ends under the old vocabulary).
New status in `triggers.py`, precedence `… triggered → pullback → crossed → watch`:

- **Definition:** some prior SETTLED close (a bar before today's), dated on/after the
  entry's `date_added`, beat the band top `pivot × (1 + PULLBACK_BAND)`; today's close
  is back within ±`PULLBACK_BAND` of the frozen pivot; today's volume is dry.
  `PULLBACK_BAND = 0.02` (the same ~2% that makes a close below the pivot "decisive"
  in Step E — beyond −2% the base is failing, not pulling back; band top rather than
  raw pivot so a name that only ever hovered at +0.5% never reads "retraced").
  `DRY_VOL_RATIO = 0.8` — the quiet-side mirror of the 1.5× confirmation gate.
- **Dry is measured on `volume_pace`, not the raw 50-day ratio** (user decision:
  intraday alerts wanted). The raw ratio is mechanically tiny all morning — every
  near-pivot crossed name would false-read "dry" at 10:00; pace projects the full-day
  ratio and equals it at the settle. On early closes pace is rescaled ×(390/210)
  (inverse of the volume-gate scaling) so a normal half day isn't misread as dry.
- **Stateless by design** — derived per-run from entry + frame; no watchlist schema
  change (no crossed-timestamp field to ripple through `_coerce_entry` /
  `merge_frozen_pivots` / CSV export). Trade-offs accepted: a cross of the same level
  BEFORE `date_added` is invisible (a 📌 re-freeze legitimately resets the clock —
  crossing the OLD level is not crossing this one); a triggered-then-retraced name
  still alerts (the code can't know whether the user bought; informational either way).
- **Surfaces:** new `crossed_earlier`/`pullback` booleans + summary bucket (schema
  stays 1, additive), ASCII `PULLBACK (...)` explainer in the console/log report, ↩
  icon + summary-gated caption in the sidebar. Raw booleans stay authoritative —
  a pullback row still carries `crossed=True` when above the pivot.
- Tests: `test_pullback_trigger` (the full matrix incl. the 11:00-vs-16:30 pace pin)
  + a pullback row in the canned sidebar report.

### 6.51 Progressive-exposure guidance in the trade panel (2026-08-12)

Suite **122/122**. Minervini's progressive exposure (*Think & Trade Like a
Champion*), previously a by-hand ritual, now lives at the point of sizing. User
decision: guidance + a one-click suggested risk % — the widget is never silently
changed.

- **`trade.suggest_risk_pct(closed, last_n=10)`** (pure): re-sorts the journal's
  closed trades by `exit_date` (`build_trade_journal` groups by SYMBOL — an unsorted
  "last 10" would read an alphabetical accident, pinned by test), slices the last 10,
  maps: <5 closed → 1.0% base ("sample too thin"); expectancy ≤ 0 **or** batting
  < .300 → 0.5% pilot; batting ≥ .500 with positive expectancy → 1.25% (one modest
  step — the 10% single-order cap still clamps); else 1.0%. Expectancy is checked
  FIRST, so a .500 hitter with small-win/big-loss churn still pilots. Rationale:
  ~.300 batting at the ~2:1 payoff the 7-8% stop discipline targets is roughly
  breakeven — below it the recent read is not working, halve the unit.
- **`journal_cache.py`** (new): the ONE `st.cache_data` wrapper around the expensive
  paginated `fetch_order_fills` pull (up to 40 API calls — full account history),
  shared by the trade panel, Journal page (its page-local `_cached_fills` deleted),
  and soon the Positions page. All callers key on the same `jr_nonce`, so the Journal
  Refresh busts every consumer; calls `trade.fetch_order_fills` via module attribute
  (patchable — app.py's by-name imports are not, so all NEW app.py trade calls go
  through the module import).
- **app.py**: `_risk_guidance()` memoizes the read per (session, jr_nonce) — a failed
  fetch is remembered as None so an Alpaca outage costs ONE attempt per session, and
  the panel renders "journal unavailable" instead of blocking. Guidance renders in
  risk mode only (tagged trades only — the cockpit's own record drives the cockpit's
  sizing). "Use suggested X%" applies via `on_click` callback (runs before widgets
  instantiate — mid-script assignment would raise) + plan invalidation; the
  `trade_amt_risk` widget converted to seed-if-absent so the callback write never
  collides with a `value=` default.
- Tests: `test_suggest_risk_pct`, `test_trade_panel_risk_guidance`; the risk-mode
  stop-controls AppTest + Journal-page AppTest gained the fetch patch/cache clear
  (suite must stay offline on machines whose env carries real Alpaca keys — see the
  new TESTS.md convention bullet).

### 6.52 Sell pillars P1-P4 on the Positions page (2026-08-12)

Suite **125/125**. The Step E doctrine (§6.11) as live per-position checkmarks —
four icon columns (✅/⚠️/❌/—) after Stage, per-row fail/warn details in the stops
loop, and a legend ("any ❌ kills the thesis; the stop is only the disaster floor
between checks"). This closed §6.48 — all four open items shipped 2026-08-12.

- **`trade.sell_pillars(pos, *, entry_date, pivot, regime, spy_note, today)`** (pure,
  like `position_advisories` — AppTest patches `fetch_positions` wholesale, so logic
  inside it would be page-test-invisible). Never raises; every missing input degrades
  its pillar to `unknown`, and a pre-§6.52 position dict yields four unknowns (pinned).
  - **P1 breakout holding** — entry date from the journal's OPEN episode (first buy,
    tz-aware UTC → ET date), day count via `_trading_days_since` (trading days, the
    laggard clock's unit). Fails: Day-0 settled close back below the pivot; decisive
    close (> `DECISIVE_BELOW_PIVOT_PCT` 2%) or a 2nd consecutive close below it;
    close below the breakout bar's low (bar at/just before entry — nothing persists
    the true breakout bar); day ≥ `P1_STALL_DAYS` 15 flat-to-red. Warn: day ≥
    `P1_CUSHION_DAYS` 10 without `P1_CUSHION_PCT` 3% — "sell into strength". No
    frozen pivot (name removed from the watchlist) → clock-only partial.
  - **P2 template** — `template_criteria` computed in `fetch_positions` off the
    position's daily frame. STRICT per user decision: anything under 8/8 fails;
    accepted cost = occasional one-day red flips on knife-edge SMA noise.
  - **P3 tape** — `scan_worker.get_worker().latest().regime` when a scan result
    exists; else the latest trigger report's SPY-only note (partial read: ok/warn);
    else unknown. Cold start / AppTest → unknown, never a raise.
  - **P4 earnings** — inside the 21-day window: loss or cushion < 8% fails, real
    cushion warns (trim to hold-through size); `earnings_in` None reads unknown ("no
    report scheduled" and "data missing" are indistinguishable upstream).
- **`scan.template_chain(df, close=None)`** extracted — the classify→full-frame
  SMA-200→validate chain was inlined in `screen_universe` and `check_one` and needed a
  third copy; the tail-slice ULP caveat now lives in ONE docstring. Error policy stays
  at the call sites (scan records, triggers fail open, positions degrade). Parity
  pinned by `test_template_chain_helper`.
- **`fetch_positions`** now carries `df` (the daily frame it previously discarded —
  P1's bar checks need it) and `template_criteria` per position.
- **Page composition** (2_Positions.py): journal open-episodes via the shared
  `journal_cache` (same `jr_nonce` — the Journal Refresh busts pillar reads too),
  pivots via `load_watchlist(cache.WATCHLIST_JSON)`, regime/SPY as above; every read
  is try/except — Alpaca down means degraded pillars, never a dead page. The page now
  READS `WATCHLIST_JSON`/`TRIGGERS_DIR`, so its AppTests patch both (TESTS.md
  convention updated).
- Tests: `test_sell_pillars` (the full matrix), `test_positions_page_sell_pillars`
  (composition + degradation), `test_template_chain_helper`; the two pre-existing
  Positions-page AppTests gained the offline patches and now double as the
  bare-dict-tolerance pin.

### 6.53 Execution audit + the operating rules (2026-08-16/17)

Weekend-review session (scan 2026-08-16: RISK-ON Moderate, 3,944 → 610 → 255 Tier A;
zero volume-confirmed triggers in the buy zone — hold-fire verdict again). Audited the
live journal (10 closed: **1W/9L, expectancy −1.7%, −$10.4k ≈ −1.1% of equity** over a
window where SPY made +3.2%). Loss control WORKED (avg loss −2.3%, worst −6%, avg win
+4.0%); what failed is follow-through, and the fills show why. The live result is
consistent with the research finding (no OOS selection alpha; risk rules real) — the
user is deliberately trading this for **execution practice**, judged on execution.

**Audit findings (from fill timestamps, not vibes):**
- **Decisions were being made intraday.** July 6 batch filled 11:10 ET; STRW 12:05 ET;
  PEBK 11:58 ET (the day-0 casualty). Contrast TXRH 2026-08-10: EOD-confirmed trigger,
  bought 9:34 ET next open, stop attached same minute — the one template-perfect entry.
- **Sell-signal latency of 2–5 days.** UNP sold ~5% below the pivot (late, §6.11);
  ARMK/ICCC held ~25 trading days vs the 15–20 laggard line; PKG/TMP hit day-10
  cushion warnings ~08-11 and were still held at day 15.
- **Batch entries** (6 on 2026-07-27 alone, 0-for-6) — repeat of the 8/3 coaching note.
- **Watchlist drift** — BRC still on the board inside its 21d earnings window; DAL 6%
  below pivot.
- **Dark windows** — the intraday task silently missed Aug 13–14 (laptop asleep/logged
  out; 28 missed slots), same shape as July 23–24. Motivated §6.54.

**THE RULES (the answer to "what am I missing execution-wise"):**
1. **Market hours execute yesterday's decisions — no new decisions intraday.** The one
   rule that covers chasing, impulse entries, and late sells at once.
2. **Daily 16:10 ET ritual (5 min):** read the trigger report (manual Check-triggers
   button §6.35 if the task missed), read the Positions pillars (§6.52), write down
   tomorrow's orders. A red/yellow pillar = decision TONIGHT, order at the NEXT OPEN —
   never "watch one more day."
3. **Max one new entry per day** — kills batching and forces each buy through the
   progressive-exposure gate (§6.51).
4. **Weekend = hunt + PRUNE:** drop names inside the 21d earnings window and >5%
   drifters, not just add.
5. **Pilot size (0.5% risk)** until the last-10 numbers improve (§6.51 guide).

### 6.54 Pi migration + pull-based deploy pipeline (BUILT 2026-08-18, Docker — Pi bring-up pending)

Motivation: §6.53's dark windows. Hardware (2026-08-18): the existing Pi-hole/unbound
**Pi 4, 4 GB / 32 GB / ethernet**. User chose **Docker** over the planned bare venv +
an EOD auto-pull — same pipeline shape, but the deploy unit became an **image tag**:
rollback is one retag, "deployed" is unambiguous. Shipped `d560197` (+`9078469`
exec-bit fix); runbook = `deploy/PI_SETUP.md` (tracked — setup/cutover steps live
THERE, not here).

**Architecture unchanged — the Pi is the cockpit's ONLY home**; laptop/phone = a
browser on `http://pi:8501`. Never run app/checks on two machines (§6.24 race).

**Built:**
- `deploy/requirements.txt` — pins mirror the INSTALLED env (pandas **2.2.3**, not
  environment.yml's drifted 2.3.1), all aarch64-wheel-clean. **64-bit OS is a hard
  requirement** (`uname -m` first; 32-bit box → reflash = DNS downtime).
- `Dockerfile` (3.12-slim-bookworm, PYTHONPATH=/app, requirements-first caching) +
  **`.dockerignore` whitelist** = the image closure: cockpit + 4 pure screening
  modules + `alpaca_trader.py` + guide md + the offline suite. Gate closure is wider
  than runtime: `backtest_daily/__init__` eager-imports the engine chain → whole
  package minus wrds_provider/ingest_wrds/run_backtest, + `momentum_lib.py` +
  `backtest_lib.py` (verified import-time numpy/pandas-only).
- `docker-compose.yml` — app (`cockpit:live`, restart policy, `./data:/app/data`,
  healthcheck) + trigger under `profiles:["manual"]`, fired per-run as
  `docker compose run --rm trigger` (fresh-process semantics kept; `docker exec`
  rejected). **Gotcha:** `compose run` extra args REPLACE the service command. No
  `build:` key anywhere — deploy.sh is the only builder, so the gate can't be skipped.
  **Pi-hole host gotcha:** Docker strips loopback nameservers from containers and
  falls back to 8.8.8.8 (bypasses Pi-hole; breaks if the router blocks outside DNS) —
  ready-to-uncomment `dns:` lines in compose, verify command in PI_SETUP §0.
- `deploy/deploy.sh` — SELF-LOCATING (resolves the repo from its own path — a
  hardcoded /home/pi cost the user a sudo-clone permission mess; never edit tracked
  files on the box, the dirty gate halts forever). Deployed SHA := `cockpit.sha`
  label on `cockpit:live`. flock →
  dirty-check (`-uno`) → fetch → ff-only → build → **gate**: suite INSIDE the image,
  `--network none`, NO volumes → live→prev retag → `up -d` → health-poll w/ auto
  rollback → prune (steady ≈3–4 GB of the 32 GB). A blocked deploy leaves the checkout
  AHEAD of the image — by design; the label is the truth, the next fire retries.
- Units: trigger Mon..Fri 09:30 + 10..16:00,30 ET (Persistent=false — missed checks
  skip, never replay late); deploy 17:30 ET + Sat 10:00 (Persistent=true) — pulls sit
  outside market hours by construction. Oneshot TimeoutStartSec 900/3600 (default
  90 s would kill a first build). No app unit — docker restart policy covers reboots.
- `.github/workflows/tests.yml` — push-time early warning; the Pi gate is the backstop.
- **Repo hygiene:** `.gitignore` `*.txt`/`*.json` blankets → `data/` + `src/data/` +
  `!deploy/**`; guide md + PROVENANCE.md now TRACKED (a clone renders the Guide page);
  `.gitattributes` LF-forces `*.sh`/units. **Windows gotcha:** a pathspec
  `git commit` rebuilds its index from the worktree and DROPS a staged 100755
  (core.filemode=false) — fixed via a temp-index plumbing commit.

**Verified without Docker** (laptop has none — fine, the real build is arm64-on-Pi):
whitelist mirrored to a temp tree (248 files) + clean venv from requirements.txt only
→ 125/125, proving closure + pins before anything reached GitHub.

**Rejected:** self-hosted runner / webhooks / push-to-Pi bare repo (as planned), plus
GHCR cross-builds (credentials on the DNS box; wheels make on-Pi builds compile-free)
and sparse-checkout (~6 MB clone). **Next:** user runs PI_SETUP top-to-bottom, then
CUTOVER (disable the Windows task; laptop never runs app/trigger again) and flips
this heading. Repo `github.com/liam-tolbert/ml-trading-pfopt`, branch `main`.

### 6.55 Auto-sell from the P1-P4 pillars (2026-08-20, suite 129/129, UNCOMMITTED)

Motivation: the §6.53 audit's dollar leak was sell-signal LATENCY (2-5 day delays
acting on red pillars); user asked to automate the sells. Design decisions (user's,
via Q&A): auto-act on **hard ❌ of P1/P2/P4 only** — P3 (tape) and every ⚠️ warn stay
report-only; **P2 needs two consecutive failing settled closes** (the strict template's
known one-day SMA noise flips must not churn exits); execution = **report-only evening
plan + separate morning submit** with an overnight veto window. As of this entry the
user also commits all code themselves — Claude leaves the tree dirty for review (their
push is the human gate in front of the Pi's auto-deploy).

**Shape (three new/edited code files + 4 units):**
- `cockpit/sells.py` — pure planner (`build_sell_plan`: full exits only; P2 streak fed
  by the PRIOR plan's per-position pillar `snapshot`, so every evening plan snapshots
  all holdings even with no orders), atomic dated persistence in `TRIGGERS_DIR`
  (`sell_plan_YYYY-MM-DD.json` — lives there so the suite's existing TRIGGERS_DIR
  patching keeps AppTests off real state), `veto_order`, `plan_is_current` (executable
  ONLY on the first trading day after evaluation; same-day and older both refused —
  bdate-based, holidays covered because the plan regenerates every weekday evening),
  `execute_sell_plan` (AUTOSELL env gate → freshness → per-order submit; qty clamped
  to CURRENT holdings; not-held skipped never shorted; FAILED stays failed — an
  ambiguous broker failure may have partially acted, blind retry could double-sell).
- `cockpit/sell_cli.py` — `plan` (16:40 ET: fetch_positions + journal entry dates +
  watchlist pivots + trigger-report SPY note = the Positions page's pillar wiring,
  headless; prior plan via `load_latest_sell_plan(before=today)`) and `execute`
  (09:25 ET: pre-open DAY market sells via the stop-aware `submit_position_sell` —
  they queue for the opening print; `--dry-run` submits nothing, saves nothing; exit 1
  on failed/partial/stale so systemd shows red). AUTOSELL unset → clean "disabled"
  exit 0, so the timer ships before the feature is armed.
- `pages/2_Positions.py` — "Planned auto-sells" section after the pillar table:
  orders + reasons + armed/disarmed state (the app container shares `.env`, so
  `autosell_enabled()` is truthful there), per-order **Veto** button whose callback
  re-reads the plan fresh from disk before flipping (the CLI may have rewritten it).
- `deploy/units/cockpit-sellplan.{service,timer}` (Mon..Fri 16:40 ET) +
  `cockpit-sellexec.{service,timer}` (Mon..Fri 09:25 ET), both `Persistent=false` — a
  missed morning fire must NOT sell mid-session at prices the plan never saw. PI_SETUP
  §7 gained the enable command + arming procedure (evening plan → page review →
  `execute --dry-run` → `AUTOSELL=1` in `.env`).

**Deliberate scope limits (don't "fix" without a reason):** full exits only (no auto
trims); no auto re-entry; no P3 action (a regime flip liquidating the whole book is a
human decision); no retry of failed orders; eod_trigger.py still NEVER places orders —
sells are a separate CLI so the trigger's no-orders doctrine stays true.

**Manual-sell execution doctrine (2026-08-20, for exits placed by hand):** a decided
exit is market-at-open — certainty beats price; the audit's leak was latency, not
slippage. A sell limit "protects" only in the gap-down case, exactly when you most
need out. Liquid names: market/OPG (submit the evening before, ≤~9:28 ET — fits the
ritual). Thin names (STRW-class ADV): marketable limit ~1-2% below prior close with a
standing "unfilled 5 min after the open → go market" rule. Mechanics trap: Alpaca
HOLDS shares against the GTC stop — cancel it first (or sell via the Positions page,
which does the cancel/sell/re-arm dance); canceling a stop the evening before a
next-open exit costs nothing, stops can't fire outside market hours.

**Tests (129/129):** `test_build_sell_plan_matrix`, `test_sell_plan_persistence_and_
veto`, `test_execute_sell_plan_gates_and_idempotency`, `test_positions_page_sell_plan_
veto` (TESTS.md §15). Next candidates when live data warrants: auto-trim on P4 warn,
P3-fail de-gross ritual, journal annotation of auto-sold episodes.

### 6.56 Scan refreshes: interaction-driven TTL → scheduled (2026-08-20, suite 129/129, UNCOMMITTED)

User-reported on the Pi: prices downloaded not just half-hourly (the trigger) but on
every app click/entry. VERIFIED, working as §6.42 designed: each script rerun called
`ensure_started()`, which claimed a background universe sweep whenever the served
result was ≥30 min old; a deploy restart added a guaranteed sweep via the pickle's
`completed_mono=-inf`. Right for an on-demand laptop; on an always-on server with
sparse interactions ~every visit kicked a multi-minute ~4,000-name sweep.

**Change (supersedes §6.42's throttle + §6.45's restart-immediate-refresh):**
- `REFRESH_TTL_SECONDS` + `try_claim_refresh`/`stamp_claim`/`_last_claim`/`ttl`
  DELETED. `ensure_started` now: adopt newer store entry → done/error → return.
  Page interaction NEVER downloads. No interaction-driven error retry either — the
  Retry button and the scheduler are the only recovery paths.
- New process-wide scheduler in scan_worker: `REFRESH_SCHEDULE_ET = (Mon-Fri 16:05,
  Sat 10:30)` ET — 16:05 gives the 16:10 ritual a near-settled regime/template read;
  Sat 10:30 feeds the weekend hunt (post-Sat-deploy window). `_next_fire` (pure),
  `_scheduled_refresh` (runs under `_SCAN_SERIAL`, publishes via `store.put`,
  deliberately never consults the adopt check — a refresh must ADVANCE data),
  `_scheduler_loop` (chunked ≤300 s sleeps — suspend/clock-step tolerant),
  `start_scheduler()` (once per process, armed from `get_worker()`, inert under the
  AppTest tell). Sessions adopt the fresh result on their next script run; the
  "Updated scan ready — load" button already covers the open-tab case.
- Restarts serve the pickle with NO immediate refresh (supersedes §6.45's). Scheduled
  refreshes run outside any session worker, so the status fragment shows no live
  progress — "data as of" just jumps on adoption (accepted trade-off).
- R2-8: direction 1 is structurally impossible now — the refresh path bypasses
  adoption entirely (`test_scheduled_refresh_always_rescans`); direction 2 unchanged.

Tests: throttle/claim tests replaced (`test_scan_refresh_schedule`,
`test_rescan_always_runs_and_no_interaction_refresh`), persist/reload test now
asserts NO thread starts on page entry; stale scan.py docstring reference to the
TTL updated. What still downloads on the Pi: half-hourly trigger (watchlist+SPY),
16:05/Sat-10:30 scheduled sweeps, 16:40 sell-plan + 09:25 executor (positions only),
manual Re-scan. Nothing else.

### 6.57 deploy/install-units.sh — one-command systemd unit sync (2026-08-20, UNCOMMITTED)

Replaces the manual cp + sed + daemon-reload + enable dance. `sudo
deploy/install-units.sh`: installs every `deploy/units/cockpit-*` unit, rewrites the
shipped pi user/home for `$SUDO_USER` in the INSTALLED copies only, removes installed
cockpit-* units the repo no longer ships (sync semantics, timers disabled first),
daemon-reloads, then (re)enables + RESTARTS every repo timer — restart matters, a
changed OnCalendar on a running timer is inert until then. `deploy.sh` diffs
`$DEPLOYED..$SHA -- deploy/units deploy/install-units.sh` after a successful deploy
and prints the exact sudo command when units changed. Deliberately NOT auto-run from
deploy.sh: that needs passwordless sudo for repo-sourced code = a root path onto the
DNS box (§6.54 posture) — root actions stay one explicit human command. It enables
the auto-sell timers too (safe: the §6.55 executor is dark until `AUTOSELL=1`).
PI_SETUP §7 + the auto-sell section rewritten around it.

### 6.58 GH #23/#24/#19 — exposure gate, armed entries, free-roll (2026-08-20, suite 139/139, UNCOMMITTED)

The three priority-high issues, built in one arc (shared trade-panel/executor
machinery); design adversarially validated pre-build — each validator correction is
flagged inline below.

**#23 gate (`trade.gate_status` + `fetch_gate_inputs`, panel wiring):**
- Scope = cockpit-TAGGED positions only (Alpaca positions ∩ tagged open journal
  episodes) — manual legacy names (ARMK-class) can never poison "flat". Open when
  flat, or when every position in the newest-DAY set has `unrealized_plpc >= 0` AND
  tagged net open P&L >= 0. Consecutive tagged losses (exit_date re-sort; pl >= 0
  incl. $0 scratch resets) >= `GATE_HALF_SIZE_AFTER` (2) → `probe_size_factor 0.5`,
  ADVISORY only. Sells/re-arms never gated.
- `fetch_gate_inputs` is a LIGHT read: `get_all_positions()` only (broker's own
  unrealized figures; no yfinance) + journal fills.
- app.py: gate fetched in the Build handler, stored as `trade_plan["gate"]` — plans
  without the key (older sessions, seeded tests) read unknown → no behavior change.
  Closed → red caption; buy rows stamped `gate_blocked` at Submit (server-side skip
  in submit_buy_plan's intent guard = the backstop); Submit disabled only when the
  payload is EXCLUSIVELY gate-blocked buys (held re-arms keep flowing).
- **Fail direction is asymmetric by design:** unknown → manual path OPEN (human
  judges), unattended executor CLOSED.

**#24 armed entries (`entries.py` + `entry_cli.py` + panel Arm/disarm + buyexec unit):**
- Arm = a 3rd button beside Submit, LIMIT+attach-stop plans only (a market row would
  buy the open blind); writes checked buy rows (session-edited limits/stops, held
  rows excluded, numpy coerced) to `entry_plan_YYYY-MM-DD.json` in TRIGGERS_DIR.
  Same-evening re-arm overwrites (last-arm-wins). Armed section renders with or
  without a built plan; per-row Disarm re-reads disk before flipping.
- Freshness is NOT sells': exactly one business day inside ``(plan_date, today]`` —
  a Sat/Sun-armed plan (the weekend hunt) executes Monday; sells' formula returns 0
  there and would never fire. Same-day + 2-session-old still refuse.
- Executor (`entry_cli.py execute`, cockpit-buyexec.timer Mon..Fri 09:26 ET,
  Persistent=false): AUTOBUY env gate (dark) → freshness → gate fresh (fails closed)
  → walk rows in panel order: held → skip to next; `skipped` result (pending buy /
  not tradable / cap) → next; only `submitted` consumes the ONE-PER-DAY bullet;
  `failed` → stop the walk, exit 1 (no blind retry; later rows stay armed and
  expire). No plan / stale plan = exit 0 (user simply didn't arm — H10). Submits
  through the REAL `submit_buy_plan([row])` → GTC OTO limit, all its guards re-run.
- Arming while the gate reads closed tonight is ALLOWED on purpose: the 16:40 sell
  plan may free the book at the open; the 09:26 gate re-check decides.

**#19 free-roll (`trade.r_multiple`, `remainder_stop=`, Positions page):**
- R column: risk reconstructed from the frozen pivot's derived stop
  (`pivot × (1 − DEFAULT_STOP_FROM_PIVOT)` — what the OTO actually attached; exact)
  else `INITIAL_STOP_PCT` off entry with a `~` prefix (also when pivot >= entry,
  where pivot-risk <= 0). Nothing persists the true entry stop — documented at the
  `position_advisories` note; persisting it at submit time is future work.
- At R >= `FREE_ROLL_R` (2.0) while `current_stop < avg_entry`: "Free-roll" in the
  sell expander seeds the existing two-step confirm with qty `held // 2` and
  `remainder_stop = avg_entry`; needs held >= 2 (half of 1 share would sell the whole
  position — a caption points at the re-arm instead).
- `submit_position_sell(remainder_stop=)`: effective level `max(old, requested)`
  (ratchet), places a NEW stop when none existed, and the failed-market-sell restore
  path restores at the OLD level — the sell never happened, so a raised stop has no
  business being in force. `stop_price` in the result reflects the PLACED level.

**Tests (139/139, +10):** gate matrix / R math / remainder_stop ratchet+restore /
gate AppTest (buys-only disable, held re-arm passes) / submit gate_blocked backstop /
entry-plan filters+coercion / weekend freshness+disarm / executor matrix / arm+disarm
AppTest / free-roll AppTest. One pre-existing fixture updated
(`test_positions_page_sell_flow`'s fake sell gained the remainder_stop kwarg).
TESTS.md §16.

**Remaining execution backlog = GitHub issues #18-25, priority-labeled** (this
section closed #23/#24/#19). Still open: #18 expectancy metrics / adaptive stop
distance (medium — next up), #20 portfolio heat cap (medium), #21 add-on pyramiding
(medium), #25 execution scorecard (low, user's call), #22 base count (low). The
issue bodies carry full context + acceptance criteria — don't re-derive here.

### 6.59 The units were installed at a path that did not exist — 3 days of silent no-deploys (2026-08-24, `58ddef8`)

**Every cockpit systemd unit had been failing since they were installed on 08-21: 29
fires, 0 successes.** `cockpit-deploy` died `203/EXEC`; the four container units died
`200/CHDIR`. Cause: `install-units.sh` rewrote only the shipped `/home/pi` PREFIX, so a
clone at `~/Documents/ml-trading-pfopt` became `/home/lct-raspi/ml-trading-pfopt` — a
path that does not exist. Fixed by rewriting the whole path to the self-located `$REPO`,
matching what `deploy.sh` already did. `PI_SETUP.md`'s "any home directory works" claim
is corrected in the same commit: true of `deploy.sh` alone, never of the units, which is
exactly what made this bite.

**Why it hid for three days:** `cockpit:live` kept serving, and the checkout looked
current because a hand-run deploy on 08-21 had advanced it — the "checkout ahead of the
deployed image" state `deploy.sh` documents as harmless. So `git log` on the Pi read fine
while nothing had deployed since.

**Diagnose this class from the deployed IMAGE LABEL, never the checkout:**

```bash
docker image inspect cockpit:live --format '{{index .Config.Labels "cockpit.sha"}}'
journalctl -u 'cockpit-*' --since -7d | grep -c 'Failed with result'
```

State at handoff: `cockpit:live` still `1983216` (7 commits behind); units re-installed
~17:35 with correct paths, all five armed; that day's 17:30 fire pre-dated the re-install
and failed on the old path, so the NEXT scheduled fire is the first real one. A manual
`./deploy/deploy.sh` on the Pi needs no sudo and closes the gap immediately. `AUTOBUY` /
`AUTOSELL` unset on both boxes — no automation armed.

### 6.60 Run logging — dated files under `data/cockpit/logs/` (2026-08-24, `ad19a7a`)

`cockpit/runlog.py`: one file per day (`cockpit_<iso-date>.log`), pruned at
`RETENTION_DAYS = 14`. Records also go to stdout, so `docker logs` and journald still see
everything — the file is the durable copy, not the only one.

- **Dated filenames, not `TimedRotatingFileHandler`:** the long-lived app, the one-shot
  trigger container and the morning CLIs all write this directory, and rename-on-rollover
  races between processes (two containers rolling the same file silently lose records).
  Appending to today's file needs no rollover at all.
- The pruner deletes only names matching `cockpit_<iso-date>.log` exactly — a
  hand-dropped file is never swept. Pruning runs as a side effect of logging and never
  raises upward.
- `get_many_prices` emits **one summary line per sweep, never per ticker** (a full-US
  sweep is ~4,200 names and `data/` is on the SD card): `requested / cached / topup /
  full / wrote / failed / elapsed`, plus a WARNING naming any no-data names. `cached` is
  the field parquet mtimes cannot give you — an all-cached line proves the box *chose*
  not to download, where an unchanged mtime is indistinguishable from a sweep that never
  ran.
- The dead `eod_trigger.log` was deleted on both boxes. `data/cockpit/logs/` appears on
  the Pi only at the first sweep after `ad19a7a` actually deploys.
- Coverage is prices only — fundamentals and EDGAR fetches are still silent;
  `runlog.get_logger()` is there when they're wanted.
- **The suite now aborts on WINDOWS** (verified 08-25, laptop, at `f3ef3c7`): it stops
  after 18 tests in `test_data_feed_logs_one_summary_line_per_sweep`, whose
  `TemporaryDirectory` cleanup hits `PermissionError [WinError 32]` because
  `DatedFileHandler` still holds `cockpit_<today>.log` open and Windows refuses to unlink
  an open file. POSIX allows it, so the Pi gate and CI stay green — this is laptop-only,
  but it means the local suite can no longer be run to completion. Fix is in the test:
  close the handler (or `ignore_cleanup_errors=True`) before leaving the `with`; the
  handler re-opens lazily on the next emit.

### 6.61 Deploy gate covers `tests/test_hunt.py` too (2026-08-24, `be006a3`)

`deploy.sh` loops over both suites and names the one that failed; CI runs the same list.
`test_hunt.py` pins the weekend hunt's rule boundaries (buy zone, RS floor, earnings
block) — the numbers that got mis-remembered when that review was done by hand.

Neither `src/stock_screener/hunt/` nor `tests/test_hunt.py` was in the image, so
`.dockerignore` un-ignores both and re-excludes `charts.py` / `__main__.py` (matplotlib
is not a runtime dep; chart rendering runs on the laptop). **Rule: a suite added to the
gate must be un-ignored in `.dockerignore` as well** — otherwise the run fails as "can't
open file" rather than as a red test.

Verified inside `cockpit:live` under the gate's own conditions (`--network none`, no
volumes): 146 cockpit tests + 30 hunt assertions green at `be006a3` (144 after `f3ef3c7`
dropped two — §6.62). Edits were staged in `/tmp` on
the Pi, never in the live checkout — modifying a tracked file there trips the
dirty-checkout halt permanently.

### 6.62 Deploy-timer cadence (2026-08-24, `f3ef3c7`) — three follow-ups open

Weekdays now fire at 16:30 ET then every 30 min through 23:30; weekends every 30 min
around the clock. All three expressions parse (`systemd-analyze calendar`), and the
cadence is cheap: `deploy.sh` exits early on "up to date", so a quiet fire costs one
`git fetch`. Three things were flagged and are still open:

1. **The timezone pin is gone.** The originals carried `America/New_York` on every
   `OnCalendar` so the schedule stayed correct regardless of host TZ and across DST; the
   new lines read system local time. Correct only while the Pi's TZ never changes — a
   reflash landing on UTC shifts every fire 4-5 hours, silently.
2. **16:30 collides with the evening ritual.** `cockpit-trigger` writes the settled-close
   report at 16:30 and `cockpit-sellplan` builds at 16:40, so on any day with a new commit
   a build + retag runs straight across that window. Not dangerous — the sellplan
   container resolves its image at start — but a first fire at 17:00 keeps the 30-min
   cadence and clears the overlap entirely.
3. No trailing newline on the file.

The same commit also deleted `test_runlog_timestamps_use_a_twelve_hour_clock` and
`test_trigger_report_header_uses_a_twelve_hour_clock`; the 12-hour formatting itself is
still live in `runlog._Formatter` and `triggers._clock12`, just no longer pinned.

**Chicken-and-egg:** this schedule cannot install itself. It takes effect only after
commit → push → a deploy pulls it → `sudo ./deploy/install-units.sh` is re-run by hand,
so the OLD schedule has to fire one more time first. `deploy.sh` prints the reminder
whenever a deploy touches `deploy/units/`.

### 6.63 Open items (2026-08-24)

- **Auto-arming entries — discussed, not built.** `triggers.py` already computes
  `triggered` (above the frozen pivot on ≥1.5× 50-day volume) and `entry_cli.py` already
  submits at most one capped, stopped limit order at 09:26. The missing link is the
  buy-side equivalent of `cockpit-sellplan`: turn settled-close `triggered` names into
  armed rows, leaving the overnight disarm window intact. **Never drive it from INTRADAY
  triggers** — the volume ratio is provisional and only clears 1.5× late in the session,
  so it would systematically buy near the close.
- `scripts/eod_trigger.bat` is dead: it targets `C:\Users\Unity\...` and appended to the
  `eod_trigger.log` that §6.60 deleted. Candidate for removal.
- **Handoff docs are becoming tracked** — `.gitignore` gains `!HANDOFF.md` and both
  `HANDOFF.md`s are staged. Consequence to remember: once tracked, an edit to this file
  ON THE PI shows in `git status --porcelain -uno` and halts every deploy. Edit it on the
  laptop. `CLAUDE.md`, `TESTS.md` and `REVIEW_BACKLOG.md` stay local-only.
- Pi access: `192.168.1.230`, user `lct-raspi`, repo at `~/Documents/ml-trading-pfopt`.
  No passwordless sudo (deliberate — `install-units.sh` explains why), so anything
  needing root is run by hand.

## Files (this venture)

- **Vendored rules:** `src/stock_screener/minervini_screener/` — pure rule logic in
  `screening/{phase_indicators,signal_engine,benchmark,indicators}.py`; `LICENSE`, `PROVENANCE.md`.
  (`data/`, `notifications/`, `analysis/`, batch processors, `quant_engine.py` = live-only, unused here.)
- **Harness:** `src/stock_screener/backtest_daily/` — `config.py`, `providers.py`,
  `synthetic_provider.py`, `ingest_wrds.py`, `wrds_provider.py`, `cache_io.py`, `fundamentals_adapter.py`,
  `indicators_cache.py`, `signals.py`, `regime.py`, `sizing.py`, `portfolio.py`, `metrics.py`,
  `engine.py`, `run_backtest.py` (`--wrds`).
- **Tests:** `tests/test_backtest_daily.py` (12 synthetic) + `tests/test_wrds_provider.py`
  (fixture-cache provider tests). Run `python tests/test_backtest_daily.py` / `..._wrds_provider.py`.
- **WRDS pull:** `ingest_wrds.py` → `data/wrds/*.parquet` (gitignored). **Overview:** `docs/MINERVINI_BACKTEST.md`.
- **Backtest outputs (saved, gitignored):** `data/wrds/_bt_*_daily.csv` + `_bt_*_blotter.csv` for the
  baseline, the A/B regime-fix runs, and the OOS train/test runs — start the delisting-avoidance work from these.
  The run/analysis scripts (full-pull backtest, A/B comparison, drawdown diagnostic, OOS validation) were
  one-off scratch in the job tmp, not committed; re-derivable from `run_backtest.py --wrds` + the config knobs.
- **Run on real data:** `python src/stock_screener/backtest_daily/run_backtest.py --wrds`  ·  synthetic demo: same without `--wrds`.
