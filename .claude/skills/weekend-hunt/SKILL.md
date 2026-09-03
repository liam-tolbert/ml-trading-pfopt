---
name: weekend-hunt
description: Run the SEPA weekend hunt — chart-review every Tier A candidate from the latest cockpit scan and produce the gated report. Use when the user asks to run the weekend hunt / weekend workflow / Sunday review.
---

# Weekend Hunt

You are the Step-3 reviewer in a pipeline where everything else is deterministic.
The library (`src/stock_screener/hunt/`) computes; you judge charts. Do not
re-derive rules ad hoc — two of them were fumbled when this was improvised:

- **Buy zone = pivot to +5%** (`scan.py buy_zone`, "no chasing"). The +10% in
  `vcp.py BUY_ZONE_PCT` is the Tier-A *screening* tolerance, never an entry bound.
  Names below the pivot are NOT in the buy zone — they are approaching/watch.
- **RS floor = 70** (Step-1). The raw scan holds sub-70 Tier A rows; the hunt
  excludes them.
- "Buy now" exists only as a **volume-confirmed breakout**: close above pivot on
  ≥1.5× the prior 50-day average volume, today's bar excluded (`gates` reports this;
  do not infer it from price alone).

Every command runs from the repo root:

```
mamba run -n ml-trading python -m src.stock_screener.hunt <cmd>
```

## Procedure

1. **`status`** — confirm the scan is fresh (the tool refuses > 3 days old) and
   report the regime line to the user. If it errors, stop and relay the message.
2. **`candidates`** — writes `diagnostics.csv` + `meta.json` under
   `data/cockpit/hunt/<date>/`. Note the candidate count.
3. **`charts`** — renders review sheets (4 tickers per PNG) into
   `data/cockpit/hunt/<date>/charts/`.
4. **Review every sheet** (Read each PNG). Judge each ticker against Step 3:
   contractions tightening, volume drying up on pullbacks, higher lows, base
   depth sane, clear pivot, no distribution — plus liquidity (be suspicious
   under ~$2M ADV), penny/illiquid character, air pockets, stale or broken
   pivots. Verdict per ticker: `PASS` (chart confirms), `PASS-` (real setup,
   named caveat), `FAIL` (price action contradicts the label). Notes are one
   dense line naming the reason, e.g. `"12-9-9-4 tightening, vol dry-up, -2% to pivot"`.
5. **Record incrementally** (crash-safe): after each sheet batch (~16 tickers),
   append rows `ticker,verdict,notes` to `data/cockpit/hunt/<date>/verdicts.csv`
   (header on first write; `pipeline.append_verdicts` semantics — plain CSV append
   from a heredoc is fine).
6. **`validate-verdicts`** — must report `ok: true` (every candidate exactly
   once). Fix any problems it lists before proceeding.
7. **`gates --min-fund N`** — N is the user's choice (ask or default 0; report
   the F distribution rather than silently gating). Output has the buckets
   (buy_zone / approaching / below / past_entry), earnings-blocked names,
   volume-confirmed names, and the watchlist audit.
8. **`report --min-fund N`** — writes `report.html` in the hunt dir. Publish it
   as an artifact (keep the same artifact URL when re-running in one session)
   and summarize in chat: verdict counts, buy-zone list, volume-confirmation
   status (usually "none — waiting on Monday volume"), earnings blocks, and
   watchlist audit including any pins that failed review.

## Boundaries

- Never arm entries, modify `watchlist.json`, or place/cancel orders — the hunt
  ends at the report; Step 4 is the user's, in the cockpit.
- Read the scan pickle only through the CLI (it validates freshness/version).
- A partial run resumes: existing `verdicts.csv` rows stand; review only the
  tickers `validate-verdicts` lists as missing.
