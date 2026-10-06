# SEPA methodology — the books' rules, and where the cockpit follows them

A reference for Claude Code sessions working in this repo. It summarises Mark Minervini's
method from *Trade Like a Stock Market Wizard* (2013) and *Think & Trade Like a Champion*
(2017), in paraphrase, and maps each rule to the code that implements it or the ledger entry
that explains why the cockpit departs from it.

Companion files:
- `src/stock_screener/HANDOFF.md`: project rules (§4), doctrine as traded (§7), conventions
  (§9) and open items (§12). The change ledger (§11) is in `HANDOFF_HISTORY.md`.
- `src/stock_screener/minervini_sepa_system.md`: the user-facing guide, rendered live by the
  app's SEPA Guide page. Write for the user there; write for sessions here.
- `src/stock_screener/cockpit/doctrine.py`: every number the cockpit trades on, defined once.

The key words MUST, MUST NOT, SHOULD and MAY are used as described in RFC 2119.

---

## 0. How a session uses this document

- **Numbers come from code, not from here.** When a rule has a cockpit constant, a session
  MUST read the value from `doctrine.py` (or the module named in §9), not from this page or
  from memory. This page gives the books' number and the constant's name.
- **Book versus cockpit.** Answers MUST say which one they are quoting. "The books say …"
  and "the cockpit does …" are different claims. Where they differ, §9 names the reason and
  its ledger entry.
- **No ad-hoc rules.** A session MUST NOT invent thresholds while judging charts, running
  the weekend hunt or answering "what would Minervini do". A missing rule is a question for
  the user, or a plan.
- **No tuning to the benchmark.** A new read's thresholds are fixed before it is measured on
  the 200 labelled charts (`tests/vcp_labels.py`) and MUST NOT be adjusted to the result.
  A failed pre-registered check ships the read without its verdict (HANDOFF §6.76, §6.91–6.93).
- **Advice.** The account is a paper account. A session MUST NOT give personalized
  investment advice. It explains what the books and the user's own rules say and leaves
  the decision to the user.
- **Orders.** A session MUST NOT arm, submit, cancel or veto orders unless the user asks
  in chat for that specific action.

---

## 1. The shape of the method

SEPA (Specific Entry Point Analysis) is a funnel. Each step removes most of what reached it.

1. **Trend:** trade only stocks already in a confirmed uptrend (Stage 2), led by relative
   strength.
2. **Fundamentals:** of those, favour companies whose earnings, sales and margins are
   growing and accelerating.
3. **The chart:** wait for a low-risk entry, a base whose swings contract and whose volume
   dries up, with a clear pivot.
4. **Entry and exit:** buy the pivot breakout on volume, cap the loss with a stop set
   before the trade, and sell by rules, into strength or on weakness.

**The books count five elements, not four:** trend, fundamentals, **catalyst**, entry points
and exit points. The four steps above are the cockpit's funnel, and its step numbers are used
throughout the code and the hunt, so they stay. The catalyst has no step of its own here and
no code (§3 "The catalyst", §9).

Around the four steps sit two more disciplines: **the market** (trade with the general tape,
never against it) and **keeping score** (let your own numbers set your stops and size).

The books' core claims, which the rest serves:
- Big winners share traits *before* their big moves; the method looks for those traits.
- Losses must stay small. The math of recovering from a large loss is the main risk.
- Timing matters as much as selection. A good stock bought at a bad point is a bad trade.

---

## 2. Step 1 — the trend

### Stage analysis
A stock's life has four stages: **1** basing (sideways, below a flat 200-day), **2**
advancing (higher highs, rising averages), **3** topping (volatile, stalling) and **4**
declining. The books buy only in Stage 2.

### The Trend Template (all eight must hold)
1. Price above the 150-day and 200-day simple moving averages.
2. 150-day above the 200-day.
3. 200-day rising for at least one month; the best names show four to five months or more.
4. 50-day above both the 150-day and the 200-day.
5. Price above the 50-day.
6. Price at least 30% above its 52-week low (big winners are often 100%+ above it).
7. Price within 25% of its 52-week high; the closer the better.
8. Relative-strength rating of at least 70, preferably in the 80s or 90s.

### Relative strength
- The RS rating ranks a stock's price performance against all others, 1–99, weighted toward
  recent months.
- The RS **line** (price ÷ index) SHOULD be rising, ideally for 13 weeks and at least 6. A
  line making a new high while price is still in its base is a strong sign.

### Supporting reads
- **Liquidity:** the stock must trade enough that your order doesn't move it.
- **Industry groups:** most big winners move with their group. A group with several leaders
  at new highs is a tailwind; a group leader breaking down is a warning for the rest.
- **Depth against the market:** a base's decline is judged against the index's over the same
  weeks. Avoid names that fell more than about 2.5–3× as much as the market.
- **Market breadth:** watch new 52-week highs against new lows, and whether the gap widens.

---

## 3. Step 2 — the company's numbers

What the books look for:
- **Quarterly EPS** growth of at least 20–25% year over year; higher is better.
- **Acceleration** matters more than the level: growth rates rising quarter after quarter.
- **Sales** growth of at least 20%, moving with earnings. Earnings rising on flat sales
  means cost cutting, which runs out.
- **Margins** stable or expanding. Sequential improvement is a strong sign.
- **"Code 33":** three quarters running in which EPS growth, sales growth and profit margin
  all rise together. Minervini's favourite pattern in the numbers.
- **Annual EPS** higher than the prior year, ideally for three years.
- **Earnings surprises:** beats cluster; one beat is often followed by more.
- **Estimate revisions:** analysts raising estimates (around 5% or more over three months)
  signals improving expectations.
- **Institutional sponsorship:** a rising number of quality funds holding the stock.
- **The reaction to the report:** strong numbers followed by a hard drop on heavy volume
  mean big holders sold the news. Stay away.

Red flags:
- EPS growth slowing for two quarters in a row.
- Inventory or receivables growing much faster than sales.
- Earnings flattered by one-time items.
- Estimates being cut; funds leaving.

The exception: the **power play** (§4) is the one setup the books buy without strong
earnings.

### The catalyst (the books' third element)

The books hold that nearly every big winner has a reason institutions want it, and that the
trader should know the reason before buying:
- a new product or service that is selling;
- an approval, a contract or a new market;
- new management, or a change in the industry that favours the company;
- sometimes only an earnings report that changes what the market expects.

The numbers above are where a catalyst shows up after the fact (acceleration, raised
estimates, a report the stock holds). The catalyst is the cause; the cockpit reads only
those traces (§9).

It is a story, and every losing stock has one too. A session asked for a stock's catalyst
MUST cite a dated source for it and MAY answer "none found". A catalyst is a label on a
candidate; it MUST NOT change a chart verdict or stand in for a failed check.

---

## 4. Step 3 — reading the chart

### The Volatility Contraction Pattern (VCP)
- A base forms after an advance. Inside it the stock pulls back two to six times, and each
  pullback is **smaller** than the one before, as a rule of thumb about half: for example
  25% → 12% → 6% → 3%. The books write this as a count of "T"s (contractions).
- **Volume dries up** as the swings shrink, and is at its lowest in the final, tightest
  area, often with a day or two of almost no trading. Sellers are exhausted.
- **Higher lows** through the base; price holds its 50-day.
- **Base length:** at least about three weeks; long bases can run more than a year.
- **Depth:** the first correction is usually 10–35% (deeper in a bear market), judged
  against the market's decline.
- **The pivot** is the high of the last tight area: the price the stock must close above to
  signal the breakout.
- **Tennis-ball action:** quick bounces off lows and closes in the upper half of the day's
  range, not grinding recoveries.

Disqualifiers:
- Pullbacks getting larger (volatility expanding).
- Heavy volume on down days (distribution).
- Wide, erratic daily swings. A stock whose normal day is larger than your stop is a
  "bucking bronco": pick another, don't widen the stop.
- A V-shaped recovery up the right side with no pause to digest.
- Breaking the 200-day during the base.

### Other features and setups
- **Shakeout:** a quick dip under an earlier low of the base that reverses within days,
  running stops before the breakout. A good sign. A dip that stays below is a lower low.
- **Cup with handle:** a rounded U followed by a short, shallow pullback on light volume
  near the old high; the pivot is the handle's high.
- **Cheat and low cheat:** a short tight pause in the middle or lower third of a cup's right
  side; an earlier, riskier entry below the full pivot.
- **Double bottom:** a W whose second low undercuts the first; the pivot is the middle peak.
- **Flat base:** five or more weeks sideways within about 10–15%, often after an earlier
  breakout.
- **Power play (high tight flag):** a stock that roughly doubles in eight weeks or less on
  heavy volume, then rests three to six weeks within about 20–25%. Earnings not required.

### Where the base sits in the run
The first and second bases of an advance work best. By the fifth or sixth, the stock is
widely owned and breakouts fail more often; the same chart behaviour that is healthy early
becomes a sell signal late.

---

## 5. Step 4 — entry

- **Buy the pivot breakout** on a closing basis above the pivot, on volume well above
  average (the books: at least 40–50% above; the cockpit uses 1.5×).
- **Buy zone:** as close to the pivot as possible. Chasing far above it widens the distance
  to the stop and raises the chance of being shaken out. The cockpit's line is +5%.
- **Earnings:** avoid opening a new position within about three weeks of a report without a
  profit cushion. A loss is never carried into a report.
- **Know the stop before you buy.** Size from the distance to it.
- **A breakout often dips back toward the pivot** in its first days; that is normal as long as
  the stop holds and the pullback is on light volume.

---

## 6. Risk: stops and sizing

- **Maximum loss:** never let a position fall more than 10% below the price paid, with no
  exceptions. The average loss SHOULD be much smaller, about 5–6% or less.
- **Set the stop from your own results:** no wider than about half your average gain. If
  your wins average 10%, cut losses around 5%.
- **Never lower a stop, never average down.**
- **Breakeven:** once a trade is up about three times the risk taken (or about twice your
  average gain), move the stop to at least breakeven. A good gain MUST NOT become a loss.
- **Back stop:** a line below which you refuse to give back profit, set around your average
  gain.
- **Position sizing:** risk about 1.25–2.5% of equity per trade. Concentrate in four to
  eight names, with the best at 20–25% of the account. Build size only after the account
  is winning; shrink it after losses.
- **Loss-adjustment exercise:** rerun your own closed trades as if every loss had been cut
  at X%. The result shows where your stop should be.

---

## 7. Selling

### Into weakness (the trade isn't working)
The books list "violations" after a buy. Several together are a reason to sell before the
stop:
- a close below the 20-day line soon after the breakout;
- a light-volume breakout followed by a heavy-volume down day;
- three or four lower lows in a row;
- more down days than up days, or more closes near the low than the high;
- a close below the 50-day on heavy volume;
- a good gain given all the way back.

### Into strength (the move is getting late)
- a 25–50% jump in one to three weeks;
- seven of ten days up late in a run;
- the largest up day, or the widest spread, of the whole advance;
- gaps late in the run (exhaustion);
- heavy volume without price progress;
- the largest down day since the run began;
- the P/E roughly doubling since the move started.

### Managing a winner
- Sell part into strength and keep part for a larger move when the market is strong.
- **Pyramid:** add to a proven winner at a second buy point, raising the stop so total risk
  stays the same. Never add to a loser.
- **Two-for-one:** sell half of two laggards to fund a full position in a leader.
- **Re-entry:** a stop-out is not a verdict on the stock; a second setup is often better
  than the first.

---

## 8. The market and keeping score

### The market
- Most stocks follow the general market. Buy aggressively only when the tape confirms: the
  index in an uptrend, leaders breaking out and working, new highs outnumbering new lows.
- When the market turns (the index breaks down, leaders fail, new lows expand), stop buying
  and **reduce** existing exposure. Build back only after new trades start working.
- In a hard market, tighten: stops about 5–6%, profits taken at about 10–12%, smaller size.

### Keeping score
- Track **batting average** (share of trades that win), **average gain**, **average loss**,
  and the gain-to-loss ratio. The three form a triangle; the weakest side is the one to work on.
- Read the largest win against the largest loss. Bigger losses than wins means holding
  losers and cutting winners.
- Before a sale, know how it will move your averages.

---

## 9. Where the cockpit implements each rule

Constants live in `src/stock_screener/cockpit/doctrine.py` unless another module is named.
"§" refers to `src/stock_screener/HANDOFF.md`.

| Rule | Cockpit | Deviation and why |
|---|---|---|
| Trend Template, 8 of 8 | `scan.book_template`: 7 vendored price criteria + `RS_FLOOR` (70) | Same as the book since §6.80 |
| RS line trend | `scan.rs_line_trend` (30/65 bars) | Shown, not gated |
| 200-day rising months | `scan.sma200_rising_months` | Shown, not gated |
| Liquidity | `MAX_ORDER_ADV_PCT` (2% of `ADV_DAYS` $-volume) caps every order | The books name the risk, no number (§6.81) |
| Industry groups | `sectors.py`, `scan.leading_groups`, `advisories.industry_concentration`, `group_leader_break` | Advisory (§6.84) |
| Depth vs market | `advisories.depth_vs_market`, flag at 3.0× | Advisory (§6.83) |
| NH/NL breadth | `breadth_store`, regime keys `new_highs`/`new_lows` | Advisory (§6.82) |
| Step-2 score F (0–8) | `scan._step2_summary`: rev ≥ 20%, EPS ≥ 20%, EPS accel, margin ↑, Code 33, annual EPS ↑, estimates ↑ ≥ 5%, last report held | Never a default gate: thin free data would drop names for missing data (§7, §6.89) |
| Earnings reaction | `advisories.earnings_reaction`, `EARNINGS_REACTION_PCT` (5%) | The books give no number (§6.87) |
| Catalyst | none | Not built. F reads its traces in the numbers; the reason behind them is not read. A news read on the hunt's PASS names is planned (§6.103, §12) |
| VCP detection | `cockpit/vcp.py` `detect_vcp`, tiers A/B/C | Looser tightening and 2-week bases on purpose, so it never misses a real base; benchmark-calibrated (§10) |
| Books' halving, base length | `advisories.book_tightening` | Numbers only; no verdict (§6.94, §6.100) |
| Volume dry-up | `advisories.volume_dryup` | Numbers only: doesn't separate good from bad on the benchmark (§6.91, §6.100) |
| Shakeouts, V recovery | — | Not built: removed after failing pre-registered checks (§6.92, §6.93, §6.100) |
| Other setups | none | Parked (`HANDOFF_HISTORY.md`, Ideas) |
| Breakout volume | `VOL_CONFIRM_RATIO` (1.5×) over `VOL_AVG_DAYS` (50), today excluded; `indicators.volume_ratio` | One rule everywhere |
| Buy zone | `NO_CHASE_PCT` (pivot to +5%) | `vcp.BUY_ZONE_PCT` (10%) is a tier tolerance, never an entry bound |
| Earnings window | `EARNINGS_SOON_DAYS` (21) | The app warns; the hunt blocks |
| Maximum loss | `MAX_LOSS_FROM_FILL` (10% below the fill) | Same as the book (§6.72) |
| Default stop | `DEFAULT_STOP_FROM_PIVOT` (7.5%) | Book midpoint of 7–8% |
| Stop from results | `DERIVED_STOP_*`: half the average win, floor `DERIVED_STOP_FLOOR` | Off until `DERIVED_STOP_MIN_WINS` wins (§6.73, §6.74) |
| Stop room (bucking bronco) | `advisories.stop_room`, `STOP_ROOM_MIN_DAYS` | Advisory (§6.75) |
| Risk per trade | `trade.RISK_PCT_PILOT/BASE/STRONG` (0.5/1.0/1.25%) | Below the books' 1.25–2.5% while the record is weak (on purpose) |
| Position cap | `trade.MAX_ORDER_PCT` (10% of equity) | Below the books' 20–25% on purpose |
| Breakeven / trail / back stop | `trade.book_stop_level`: the 50-day reaching the cost, `doctrine.BREAKEVEN_R` (3R), `BREAKEVEN_AVG_WIN_MULT` (2× average win, back stop at the average win); `FREE_ROLL_R` (2R) | Follows the books (§6.96). The average-win rules wait for 5 tagged wins |
| Sell pillars | `trade.sell_pillars` P1–P4; P1 uses `DECISIVE_BELOW_PIVOT_PCT`, `P1_CUSHION_*`, `P1_STALL_DAYS` | P1 sells a day-0 close back under the pivot, stricter than the books (§12). A close under the entry-day low only warns while the pivot holds (§6.95) |
| Violations after the buy | `advisories.post_breakout_read`; `VIOLATIONS_CAN_FAIL` (off), `VIOLATION_FAIL_COUNT` | Warnings only until switched on (§6.77) |
| Weak-tape advice | `WEAK_TAPE_STOP_PCT`, `WEAK_TAPE_TARGET_PCT` | Advice, not applied (§6.78) |
| Market turn | P3; `MARKET_TURN_CAN_TRADE` (off), `MARKET_TURN_REDUCE_FRACTION` | Notes only until switched on (§6.78) |
| Selling into strength | `advisories.strength_signs` from `STRENGTH_MIN_GAIN` (+20%); `STRENGTH_CAN_TRADE` (off), `STRENGTH_SIGNS_TO_SELL`, `STRENGTH_SELL_FRACTION` | The P/E sign is not read. A gain stands in for the base count. Notes only until switched on; then half is sold, the rest stopped at breakeven (§6.98) |
| Re-entry lag | `REGIME_CONFIRM_DAYS` (15) with `BREADTH_MIN_PHASE2` | From the backtest (§6.82) |
| Keeping score | Journal page: `trade.build_trade_journal`, `journal_stats`, `loss_adjustment_sweep` | — |

Decisions are made on the settled daily close and act at the next open. The user judges every
chart; the machine sorts and warns.

---

## 10. Vocabulary

| Term | Meaning |
|---|---|
| Base | A rest period after an advance, where the stock moves sideways while earlier buyers sell |
| Contraction / "T" | One peak-to-trough pullback inside a base |
| Pivot | The price that confirms the breakout when closed above |
| Breakout | A close above the pivot; confirmed on heavy volume |
| Buy zone | From the pivot to a small distance above it |
| Dry-up | Very low volume in the final tight area of a base |
| Distribution day | A down day on unusually heavy volume |
| R | Gain measured in units of the initial risk (entry minus stop) |
| Free roll | Selling part of a winner and stopping the rest at breakeven, so the trade can't lose |
| Batting average | Share of trades that make money |
| Tier A/B/C | The cockpit's VCP classes: A = valid base near its pivot, B = forming or extended, C = no base |
| F | The cockpit's Step-2 score, 0–8 |
