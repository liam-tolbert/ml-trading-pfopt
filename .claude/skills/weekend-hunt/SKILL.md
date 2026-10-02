---
name: weekend-hunt
description: Run the SEPA weekend hunt — chart-review every Tier A candidate from the latest cockpit scan and produce the gated report. Use when the user asks to run the weekend hunt / weekend workflow / Sunday review.
---

# Weekend Hunt

You are the Step-3 reviewer in a pipeline where everything else is deterministic.
The library (`src/stock_screener/hunt/`) computes; you judge charts. The books' chart
rules (§4) and disqualifiers are in `src/stock_screener/SEPA_METHODOLOGY.md`; judge
against those. Do not re-derive rules ad hoc — two of them were fumbled when this was
improvised:

- **Buy zone = pivot to +5%** (`scan.py buy_zone`, "no chasing"). The +10% in
  `vcp.py BUY_ZONE_PCT` is the Tier-A *screening* tolerance, never an entry bound.
  Names below the pivot are NOT in the buy zone — they are approaching/watch.
- **RS floor = 70** (Step-1). The raw scan holds sub-70 Tier A rows; the hunt
  excludes them.
- "Buy now" exists only as a **volume-confirmed breakout**: close above pivot on
  ≥1.5× the prior 50-day average volume, today's bar excluded (`gates` reports this;
  do not infer it from price alone).

Every command runs from the repo root, through the wrapper that puts the `ml-trading`
env on PATH:

```
bash scripts/hunt/hunt.sh <cmd>
```

## Procedure

1. **`status`** — confirm the scan is fresh (the tool refuses > 3 days old) and
   report the regime line to the user. If it errors, stop and relay the message.
2. **`candidates`** — writes `diagnostics.csv` + `meta.json` under
   `data/cockpit/hunt/<date>/`. Note the candidate count.
3. **`charts`** — renders review sheets (4 tickers per PNG) into
   `data/cockpit/hunt/<date>/charts/`.
4. **Review every sheet** (Read each PNG). Judge each ticker against Step 3:
   contractions tightening, volume drying up (computed: the title's `DU r/n` is the
   final tight area's volume over its 50-day average and its near-silent days; read
   it, don't re-estimate it from the bars), higher lows, base
   depth sane, clear pivot, no distribution — plus liquidity (an order MUST stay
   within 2% of the 20-day dollar volume; `max_order_usd` in the diagnostics is the
   ceiling, and a ceiling under a full position is a `PASS-` at best),
   penny/illiquid character, air pockets, stale or broken
   pivots. Verdict per ticker: `PASS` (chart confirms), `PASS-` (real setup,
   named caveat), `FAIL` (price action contradicts the label). Notes are one
   dense line naming the reason, e.g. `"12-9-9-4 tightening, vol dry-up, -2% to pivot"`.
5. **Record incrementally** (crash-safe): after each sheet batch (~16 tickers),
   Write the batch to `data/cockpit/hunt/<date>/verdicts_batch_NN.csv` — header
   `ticker,verdict,notes`, one row per ticker, notes quoted when they hold a comma —
   then run **`append-verdicts --file <that path>`**. It refuses the whole batch if
   a ticker is not a candidate, is repeated, or already has a verdict, and prints
   how many candidates remain. Never edit `verdicts.csv` by hand.
6. **`validate-verdicts`** — must report `ok: true` (every candidate exactly
   once). Fix any problems it lists before proceeding.
7. **`gates --min-fund N`** — N is the user's choice (ask or default 0; report
   the F distribution rather than silently gating). F runs 0–8. A
   missing figure fails its check, so a low F can mean thin data. Output has the buckets
   (buy_zone / approaching / below / past_entry), earnings-blocked names,
   volume-confirmed names, and the watchlist audit.
8. **Write the narrative** to `data/cockpit/hunt/<date>/narrative.md`. This is
   your read of the week, in Markdown (headings, paragraphs, bullets, pipe
   tables; nothing fancier renders), 300–700 words:
   - the regime and what the breadth numbers mean for taking entries;
   - what stands out in this scan — groups, themes, the quality of the bases;
   - every buy-zone PASS name in one line each: the setup, the volume read,
     the F score, what would confirm it;
   - approaching names worth an alert, and earnings-blocked names to revisit;
   - the watchlist audit in words: pins that failed or drifted, names to prune;
   - caveats of the review (thin data, position-size assumptions, pivots you
     doubt).
   Verdicts and numbers come from steps 4–7; do not restate the tables.
9. **`report --min-fund N`** — writes `report.html` in the hunt dir, with the
   narrative under "Reviewer's read" and every chart sheet under its verdicts,
   and mirrors it with its `charts/` to **`docs/hunt/<date>/`**, the deliverable.
   Summarize in chat: verdict counts, buy-zone list, volume-confirmation status
   (usually "none — waiting on Monday volume"), earnings blocks, and watchlist
   audit including any pins that failed review.
10. **Publish the artifact** (in a session, never unattended): the Artifact tool
    with `file_path` = `docs/hunt/<date>/report.html`, `root` = that folder,
    `files` = every `charts/sheet_NNN.png` mapped to itself, icon `chart`.
    Re-publish to the same URL when re-running in one session. To publish a
    past unattended run, the same call on that date's folder is all it takes.
    Printing an artifact from the claude.ai viewer cuts it off; a browser's own
    print of `report.html` does not, so never suggest the former.

## Unattended run

The Friday scheduled task (`scripts/hunt/weekend_hunt.ps1`) starts this skill with
nobody at the keyboard. The task has already pulled the Pi's scan and checked it.
When the prompt says the run is unattended:

- Ask nothing. Use `--min-fund 0`.
- Run steps 1–9 and skip step 10 (no Artifact tool here): the report in
  `docs/hunt/<date>/` is the deliverable, and the narrative is where your
  judgment goes — write it with care.
- Only the hunt CLI, file reads, and writes inside `data/cockpit/hunt/` are
  permitted. Do not try ssh, scp or any other command; a refusal is not a
  reason to look for a workaround.
- If `verdicts.csv` already has rows, an earlier attempt was cut short: resume
  from the tickers `validate-verdicts` lists as missing.
- Your final message is saved as `summary.md` in the hunt dir. Make it the chat
  summary from step 9, in Markdown, and nothing else.
- If a step fails, stop and make the final message the command and its error.

## Boundaries

- Never arm entries, modify `watchlist.json`, or place/cancel orders — the hunt
  ends at the report; Step 4 is the user's, in the cockpit.
- Read the scan pickle only through the CLI (it validates freshness/version).
- A partial run resumes: existing `verdicts.csv` rows stand; review only the
  tickers `validate-verdicts` lists as missing.
