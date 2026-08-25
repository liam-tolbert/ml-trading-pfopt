# HANDOFF — ML side PARKED (S9). Stock selection dead; the equal-weight "edge" also debunked. Repo cleaned; pivoting to a new idea.

**Status: S9, 2026-06-04 — ML PARKED.** Nine sessions. The cross-sectional ML model is
leak-free with a genuine *classification* hit-rate but yields NO tradable edge, and S9
closed the last open question. The user has **parked the ML side** and is starting a new
idea; this doc is the ML archive until that idea gets its own status. As part of parking:
Main.ipynb's diagnostic/eval/backtest cells were stripped (**42 → 23 cells**; the linear
data → features → label → train → `run_model` pipeline remains), the one-off diagnostic
scripts were deleted, and `scripts/` is now gitignored scratch. **Both stock selection AND
equal-weight are closed — do not re-open either.**

**S9 — the LEAD task (validate equal-weight standalone) is CLOSED, and it DEBUNKED the S8
"equal-weight is the edge" thesis.** Standalone validation (no model; reused the tested
engine via an all-buy degenerate model) confirmed the universe EW reproduces Sharpe
**0.864** on 2018–26 with ZERO model involvement — so the "edge" was never the XGBoost.
But the **survivorship-immune regime stress kills the smart-beta thesis**: the real
equal-weight S&P fund **RSP** has Sharpe 0.655 < SPY 0.720 over 2003–26 (CAPM α **−0.67%/yr,
t−0.63**) and **lagged SPY over the identical 2016–26 window** (12.56% vs 15.78% CAGR). The
EW premium was positive only pre-2015 and in the 2022 bear; negative through the mega-cap
decade. So our universe's apparent edge (18.5% CAGR, β0.93) was **survivorship + a defensive
composition tilt**, not the equal-weight factor. The only durable residue is the
**low-vol/defensive premium** + a bear-market diversification benefit (validate vs USMV/SPLV
the same survivorship-immune way RSP was used — only if the money track is ever resumed; see NEXT).

## TL;DR — the verdict (S8)

Walk-forward backtest, window 2018-01 → 2026-06-15 vs SPY, realistic price-dependent
spread cost (`spread_per_share=0.02`, ~2.7 bps one-way), equal-weight construction,
listing-seasoning ON. Four selection tests, all negative:

1. **CAPM α/β:** every config β<1, **no significant α** (all |t|<2). Long-only = no skill.
2. **Listing-seasoning (survivorship):** late-entrant inflation negligible (the ~0.84
   flat-EW bar is robust); de-biasing *widens* the strategy's loss → reinforces the verdict.
3. **Beta-neutral L/S:** market-neutral α **significantly NEGATIVE** (−9..−15%/yr, t≈−2.3) —
   the high-P_Buy names underperform the low-P_Buy ones even market-adjusted (ranking inverted).
4. **Residual cross-sectional label** (the last lever): α now ~zero (t≈−0.7) — the relative
   label *fixed* the perverse negative but produced **no positive α**; long-only just
   reproduces the flat-EW bar (0.86) and doesn't beat 1/N.

**Conclusion:** the ML cross-sectional signal has no extractable edge under any label,
construction, cost, or survivorship treatment. The model keeps rediscovering a low-β /
low-vol (BAB) tilt — a real anomaly, but it can't generate selection alpha in this regime.
The only thing that beats SPY risk-adjusted is **broad equal-weight itself**. *[S9 SUPERSEDED:
that EW edge was survivorship + a defensive tilt, NOT the equal-weight factor — RSP (real EW
S&P) has no long-run α and lagged SPY 2016–26. See the S9 block at top.]*

## NEXT — only if the money/ML track is RESUMED (parked by the user as of S9)

The user has pivoted to a new idea; the tracks below are the standing guidance *if* this
project is revisited. Goal split (user's own words): **primary = make money by any robust
means; secondary = the academic thrill of finding alpha.** They don't have to be the same system — run in
parallel. Selection on large-cap US is exhausted; don't re-open it. Build from what survived.

### Track A — MONEY (primary): harvest factor premia (smart beta)
The model never picked stocks; it kept rediscovering real, durable premia (low-vol/BAB,
equal-weight/breadth). Don't *predict* — *harvest*.
- **~~LEAD — equal-weight broad universe~~ — CLOSED (S9).** Validated standalone and
  regime-stressed vs RSP: the EW factor has no long-run α and our universe's edge was
  survivorship + a defensive tilt (see the S9 block at top). Do NOT pursue "equal-weight
  the universe" as alpha.
- **NEW LEAD if resumed — low-vol / defensive premium.** The actual durable residue. Validate
  it survivorship-immune vs real low-vol ETFs (USMV/SPLV) the same way RSP was used here;
  decompose how much of the universe's edge is the low-vol anomaly vs survivorship.
- **+ Low-vol / BAB tilt** — the persistent low-β signal; may add α over a full cycle.
- **+ Timing / regime overlay (drawdown control)** — scale market exposure with the
  aggregate signal / `Bull_Prob` (risk-on/off: SPY vs cash/bonds) to cut the −36% DD.
  Cheapest first test: does the aggregate signal predict forward SPY returns at all?
  (decide before building the overlay).
- Free, reuses existing infra. If productionizing, `predict.py` should drop pypfopt and
  just equal-weight a broad book.

### Track B — ALPHA HUNT (secondary / academic): a less-efficient arena
Real predictive alpha won't come from a new *feature* on large-caps (efficient market) —
it needs a new *arena*. Move to where inefficiency may still exist: **microcap / crypto /
a niche the user knows better than the crowd.** Higher risk, messier but free-er data;
low stakes, run as an experiment alongside Track A. NB: **sentiment / alt-data on large-caps
is NOT the path** — it's the most crowded signal there is, decays in seconds, and is priced
by weekly bars; only worth considering inside a less-efficient arena or at speeds an
individual can't reach.

### Cleanup / reminders
- **Easy win (anytime):** drop the 4 calendar features from `create_stock_features`
  (ablation cleared them: differential +0.382 without vs +0.377 with).
- `ETF_ML.ipynb` is the same story (timing P(Buy) yes, cross-sectional no); macro-timing
  path abandoned. Don't re-open selection.

## What was established — the four S8 selection tests (extended window, n=437 wk)

**1. CAPM α/β decomposition** (cell `9d5cdd26`; each config's weekly NET return on SPY,
quarterly, HAC t-stats):

| config | β | α (ann) | t(α) |
|--|--|--|--|
| top30 | 0.95 | −1.5% | −0.39 |
| top50 | 0.95 | −0.3% | −0.10 |
| top100 | 0.91 | +4.3% | 1.33 |
| buys>0.5 | 0.86 | +0.4% | 0.11 |
| universe_ew | 0.96 | +3.5% | 1.25 |

No config has significant α. The threshold rule (buys>0.5) lowers β, not the ranking;
even the `universe_ew` "bar to beat" is insignificant α (its Sharpe edge is a low-β/EW
artifact). More names → higher Sharpe monotonically (concentration penalty).

**2. Listing-seasoning / survivorship** (`bl.apply_listing_seasoning`, drops each ticker's
first 52 wk; cell `3f41d849` knob `BT_SEASONING_WEEKS=52`):

| arm | tickers | ew Sharpe | strat Sharpe | strat−ew |
|--|--|--|--|--|
| season_0 (full) | 240 | 0.859 | 0.561 | −0.298 |
| season_52 | 240 | 0.858 | 0.521 | −0.337 |
| fixed_univ (incumbents) | 210 | 0.841 | 0.475 | −0.367 |

Late-entrant inflation of `universe_ew` is negligible (0.859→0.841 even deleting all 30
late entrants). De-biasing CUTS the strategy and WIDENS its loss vs flat-EW → reinforces
the no-edge verdict. Dead-name (delisted) data abandoned — unobtainable free (see
operational notes); documented as a limitation (its direction inflates results, so the
negative verdict is conservative). Seasoning kept ON.

**3. Beta-neutral long/short** (`bl.walk_forward_long_short`, weekly, dollar-neutral,
ex-ante `Beta_26wk` hedge; cell `651e13b0`; absolute label):

| q | long β | short β | hedged Sharpe | CAPM α | t(α) |
|--|--|--|--|--|--|
| 0.1 | 0.72 | 1.24 | −0.56 | −14.9% | −2.24 |
| 0.2 | 0.75 | 1.14 | −0.62 | −11.5% | −2.43 |
| 0.3 | 0.76 | 1.08 | −0.54 | −8.6% | −2.31 |

Significantly negative α — the ranking is *inverted* risk-adjusted (not merely
uninformative). Robust to the imperfect hedge (residual +0.18–0.30 β was a bull tailwind,
yet it still lost). Reconciles with the +0.382 hit-rate: hit-rate ≠ return (buy = low-β
small movers; sell = high-β names that ripped in 2018–26). **Do NOT invert the signal** —
that's a regime bet (long high-β / short low-β) that a bear tape destroys.

**4. Residual cross-sectional label** (`bl.relative_residual_label`, per-week tercile of
fwd−β·mkt; cell `7ef381f9`) — the last selection test:

- *L/S (decisive):* α now ~zero, NOT significant (−4.9/−3.3/−3.6%, t≈−0.67/−0.58/−0.76 at
  q=0.1/0.2/0.3). The relative label fixed the negative-α but produced no positive α.
- *Long-only EW (qtr):* buys>0.5 Sharpe **0.86 / +244%**, top50 0.74 / +239%; CAPM α
  +4.6%/+5.3% but **t=1.41/1.13 (not sig)**, β 0.88. Matches the flat-EW bar and beats SPY
  — but it's reproducing equal-weight-of-universe, not adding selection skill (concentrate
  to 50 → 0.74). Model still picks low-β longs/high-β shorts (0.48 vs 1.59).

## Don't re-do (settled)

- **Don't re-open selection.** Exhausted across construction (EW vs max-Sharpe), breadth
  (top30→100→all), cost, survivorship, long-only & long-short, and two label designs
  (absolute vol-scaled, residual cross-sectional). No edge anywhere.
- **Don't invert the L/S signal** — regime-specific negative = leveraged bull bet.
- **Don't chase delisted data on the free stack** — impossible/corrupting (see ops notes).
- **Don't use max-Sharpe / sample mean-variance** — loses to 1/N at ~400-wk data size
  (DeMiguel-Garlappi-Uppal 2009); was the old vol problem. If productionizing EW, drop pypfopt.
- **Don't re-tune cost** — settled (~2.7 bps one-way; quarterly cost-immune, weekly cost-killed).
- **Don't tune XGBoost** — simple ≈ complex (Wolff-Echterling); edge was never model sophistication.
- **Don't worry about seasonality** (calendar ablation cleared); **don't tune to val**
  (val Bull_Prob 0.90 vs test 0.70, bullish-biased); **don't switch to 4-class label**;
  **don't re-litigate macro-vs-cross-sectional**.

## Model definition (notebook is authoritative; `overall_model`)

- **Label (default):** vol-scaled K=0.60 σ_12wk binary, 1-wk forward (`make_signal_column`).
  Alternative (S8): `bl.relative_residual_label` (residual cross-sectional tercile).
- **Features** (`create_stock_features`):
  - Technical (12): SMA_5v20, Volume, RSI, MACD, Bollinger_Bands, ATR, Stochastic, OBV,
    ADX, Aroon, Returns-3wk-1wklag, Returns-1wk-0wklag.
  - Macro/FRED (3): Inflation (real_pce_yoy +9wk), InterestRate (FEDFUNDS, unshifted),
    UnemploymentRate (UNRATE +6wk).
  - Fundamentals (17, filing-date indexed): HistPE, PE_Change_4wk, PB, FCFYield, PSR, PEG,
    ProfitMargin, ROE, ROA, DebtToEquity, RevenueGrowthYoY, EarningsGrowthYoY, Accruals,
    BuybackYield, OperatingMargin, AssetTurnover, CFO_to_NI.
  - Idio momentum (3): Idio_Momentum_{4,12,26}wk (residual vs market, `Beta_26wk.shift(1)`).
  - Earnings (2): Days_Since_Earnings, Post_Earnings_4wk (≤28d).
  - Vol/market (4): Volatility_Spike, SP500-Returns (6wk), SP500-Log-Returns, Beta_26wk.
  - Time (4): Week/Month/Quarter/Presidential-year — ablation shows they DON'T carry the
    edge; safe to drop.
  - Interaction (1): Idio_Mom_26wk*Bull_prob (causal).
- **Hyperparams:** XGBoost binary:logistic, max_depth=4, n_estimators=150, lr=0.05,
  subsample/colsample=0.8, reg_lambda=2, reg_alpha=1, gamma=0.5, min_child_weight=20.
- **What confidence sorts on:** low β (~0.85, even ~0.48 under the residual label), positive
  idio-momentum, ~6-7wk post-earnings — a **defensive / low-vol tilt**.

## Look-ahead landmines — all CLOSED in S5

1. `classify_regimes` smoothed → causal filtered, cached (`bull_prob_causal.csv`).
2. FRED no pub-lag → UNRATE+6wk, PCE+9wk (FEDFUNDS residual ~32d lag remains; see deferred).
3. Fundamentals `fp_end` → SEC `filed` date.

## Operational notes (durable)

- **Notebook editing:** `Main.ipynb` too large for Read. Edit via Python JSON mutation
  (`json.load`, locate cells by `id`, mutate `source`, back up to `Main.ipynb.bak{N}`,
  `ast.parse`-check). Inspect raw JSON with Grep. `*_output.ipynb` are papermill artifacts —
  never edit. `Main.ipynb` is git-tracked — recover via git, no `.bak` files kept.
- **Stale-variable gotcha:** notebook globals (`bt_panel`, `overall_df`, `probs_test`…) feed
  later cells — re-run assignment cells or diagnostics reflect stale kernel state. Re-run
  setup cell `3f41d849` before the S8 diagnostic cells. Byte-identical diagnostics across
  retrains → suspect this.
- **Delisted/survivorship data (S8, confirmed unobtainable free):** yfinance returns EMPTY
  for dead names or — worse — recycled-ticker data (BBBY/SBNY/HTZ post-bankruptcy re-IPOs)
  that CORRUPTS the panel; Stooq is now API-key gated; Kaggle dead-name archives (ARANDKEI,
  a TWTR set) are clean but don't overlap the large-cap universe and lack fundamentals.
  CRSP/WRDS (gold standard, was free at UVA) closed on graduation. Paid feeds (Sharadar,
  Norgate) declined. → dead-name bias is a documented limitation.
- **Working dir:** run `python` from project root; `src/` reads CWD-relative `data/…`.
- **numpy 2.x / sklearn 1.7.x:** `compute_class_weight` crashes; use
  `{c: n/(k*count) for c,count in zip(classes,counts)}`.
- **EDGAR submissions API:** 10 req/s, needs `User-Agent`; ~250 tickers ≈ 2-3 min.
- **FRED:** monthly series stamped at period-START, released ~14–35d after period-END →
  hence the 6/9-wk shifts. `get_fred_data` hardened S8 (429 retry/backoff; `observation_end`
  =2026-06-15).
- **Causal Bull_Prob recompute:** ~6.5 min; cache `data/bull_prob_causal.csv` (2002-01 → 2026-06-05).
- **Data freshness (S8):** `SP500.csv`, `bull_prob_causal.csv`, `training_set.csv` all end
  **2026-06-05**; `historical_fundamentals.csv` 2026-04-05 (FRED macro monthly → 2026-04);
  `BT_END`=2026-06-15. NOTE: SP500 has NO auto-refresh cell — `download_and_fix_sp500` is
  never called (load cell does a plain `pd.read_csv`) AND has a latent missing `return sp500`;
  on-disk file was refreshed manually. Wire the call + fix the return for hands-off refresh.
- **Intel Fortran kernel crash (Win):** set `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`.

## Deferred engineering (only if a direction shows promise)

- **Production rebuild for EQUAL-WEIGHT.** `src/lib.py`, `src/train_model.py`,
  `src/predict.py` are stale vs the notebook (no FRED shifts / filing-date join) AND use
  max-Sharpe. If pivoting to the equal-weight system, `predict.py` should **drop pypfopt
  and just equal-weight** a broad book (and `run_model`'s pypfopt default `frequency=252`
  on weekly data is a latent bug — backtest uses 52).
- **`FEDFUNDS` residual lag (~32 d).** Monthly-avg EFFR, unshifted; swap to a daily/announce
  series (`DFF`/`DFEDTARU`/`DFEDTARL`) to fully close landmine #2.
- **`fetch_*` cache validation bug** — validates date freshness, not ticker coverage; bites
  on the next universe change (work around with `force_refresh=True`).

## History — per session (compact)

- **S1–3 — diagnosis.** Pre-S4 model was macro-driven, not stock-picking (2024 differential
  −0.023, 2025 +0.245; in-sample 0.79 was 84% carried by 2025).
- **S4 — cross-sectional features.** 2024 differential −0.023 → +0.385 via idio-momentum
  (3 horizons) + earnings events + removing standalone `Bull_Probability`. Fixed
  `Days_Since_Earnings` bug via EDGAR submissions API.
- **S5 — closed all 3 look-ahead leaks** (differential held +0.377; landmines above).
- **S6 — built the walk-forward backtest** (`src/backtest_lib.py` pure loop; quarterly refit
  on `Date ≤ refit−2wk`, 52-wk trailing covariance, `Returns-future-1wk` as realized PnL).
- **S7 — realistic cost + construction overhaul.** Added price-dependent spread cost; under
  it the old max-Sharpe verdict flipped (no config beat SPY net). Dropped max-Sharpe for
  equal-weight (vol 26.5%→~19.5%, Sharpe 0.51→~0.68); added `buy_threshold`. Established the
  own-universe flat-EW benchmark (Sharpe 0.84 ≈ SPY) — strategy lost to it; diagnosed the
  defensive low-β tilt. Calendar ablation + literature review (`docs/literature_review.md`;
  closest match Wolff-Echterling 2024). Survivorship audit (`scripts/check_universe_coverage.py`:
  253 intended / 240 present / 29 late entrants).
- **S8 — data refresh + four selection tests → SELECTION CLOSED; equal-weight *looked* like
  the edge (DEBUNKED S9).** Extended all data to 2026-06-05 (n=437 wk); hardened FRED retrieval.
  Ran CAPM (no sig α), listing-seasoning (survivorship negligible, reinforces verdict),
  beta-neutral L/S (significantly negative α), residual cross-sectional label (fixes
  negative→zero α, still no edge). Surfaced the equal-weight smart-beta finding (Sharpe 0.86
  vs SPY 0.68).
- **S9 — equal-weight DEBUNKED; ML side PARKED + repo cleaned.** Standalone EW validation
  (no model) reproduced Sharpe 0.864 → the edge was never the model. Survivorship-immune
  regime stress vs RSP (real EW S&P ETF, 2003–26) showed the EW factor has no long-run α
  (t−0.63) and lagged SPY 2016–26 → our universe's edge was survivorship + a defensive tilt.
  User parked ML: stripped Main.ipynb diagnostics (42→23 cells), deleted one-off scripts,
  gitignored `scripts/`. Pivoting to a new idea. (The S9 validation script
  `validate_equal_weight.py` was a one-off, since deleted; its numbers are recorded above.)

## Files & key cells

- **Backtest engine:** `src/backtest_lib.py` — `walk_forward_backtest` (equal-weight,
  `spread_per_share`, `buy_threshold`, `rebalance_every`), `apply_listing_seasoning`,
  `walk_forward_long_short`, `relative_residual_label`. `tests/test_backtest_lib.py` —
  **11 tests** (`python tests/test_backtest_lib.py`).
- **Scripts:** none tracked — `scripts/` is gitignored scratch as of S9. The S8/S9 one-offs
  (`check_universe_coverage.py`, `validate_equal_weight.py`) were deleted (conclusions recorded
  here); `build_etf_ml_nb.py` was lost (untracked) but its output `ETF_ML.ipynb` survives.
- **Data:** `data/bull_prob_causal.csv`, `data/historical_fundamentals.csv` (fp_end-indexed),
  `data/earnings_filings_submissions.csv`, `data/training_set.csv` (240 tickers).
  `data/model.pkl` is **STALE — use notebook `overall_model`**.
- **Docs:** `docs/literature_review.md`; `docs/Indicators.xlsx` (stale — needs S4/S5 features).

**Main.ipynb cells (S9: diagnostic/eval/backtest cells STRIPPED → 23 surviving pipeline cells).**
The removed diagnostics' logic lives in `src/backtest_lib.py` + this doc — don't expect them in
the notebook. Surviving linear pipeline:

| Notebook cell (id) | Purpose |
|--|--|
| `a30b8c33…` | `stocks` / `overall_tickers` (current snapshot) |
| `24a7cf2f…` | `create_stock_features` (full feature set; calendar cols droppable) + `overall_df` |
| `548eac63…` | `make_signal_column` (K=0.60, absolute) + `train_val_test_split` |
| `1447ec4f…` | `train_model` + `predict` (XGBoost hyperparams) |
| `e55679e6…` | `overall_sets` split + `overall_model` train |
| `a9a9904c…` | `run_model` — predict + adjusted-μ + max-Sharpe → `recommendations` (surviving application) |

**REMOVED in S9** (logic preserved in `backtest_lib.py`/this doc): model-eval `719fa081` (Fig 2 /
`probs_test`), bucket differential `f35ddd42`, backtest setup `3f41d849`, run `d3ea0ea2`, sweep
`7182773c`, CAPM `9d5cdd26`, beta-neutral L/S `651e13b0`, residual-label `7ef381f9`, plus the
fundamentals-distribution exploration and two dead commented-out strategy cells.
