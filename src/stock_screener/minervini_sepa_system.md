# Minervini SEPA System — 4-Step Implementation Guide

> **SEPA** = Specific Entry Point Analysis. The goal is to filter from thousands
> of stocks down to 5–20 high-probability candidates where technical structure,
> fundamental momentum, and market timing all align simultaneously.

---

## Overview: The Funnel

```
ALL STOCKS (~5,000+)
        │
        ▼  Step 1 — Trend Template (8 technical conditions)
   ~50–200 stocks
        │
        ▼  Step 2 — Fundamental filters (earnings, sales, margins)
    ~20–50 stocks
        │
        ▼  Step 3 — Chart review (VCP quality check)
     ~5–20 stocks  ← your watchlist
        │
        ▼  Step 4 — Entry trigger (pivot breakout + volume)
       Active positions
```

Each step is a hard gate. A stock failing any single criterion in a step is
eliminated — there is no partial credit, and no rounding up.

---

## Step 1 — Trend Template (already implemented in Python)

All 8 conditions must be true simultaneously, using **simple moving averages only**
(not EMAs — they produce different results).

| # | Condition | Notes |
|---|-----------|-------|
| 1 | Price > 150-day SMA | Confirms positive long-term trend |
| 2 | Price > 200-day SMA | Confirms positive long-term trend |
| 3 | Price > 50-day SMA | Short-term momentum above all longer MAs |
| 4 | 50-day SMA > 150-day SMA | Short-term accelerating above medium-term |
| 5 | 150-day SMA > 200-day SMA | Medium-term trend above long-term baseline |
| 6 | 200-day SMA trending up ≥ 1 month | Compare today's 200d SMA vs value 22 trading days ago — it must be higher. Ideally rising 4–5 months for best setups |
| 7 | Price ≥ 30% above 52-week low | Eliminates bottoming stocks — you want proven strength, not a bounce |
| 8 | Price ≤ 25% below 52-week high | Stock must be near the top of its range, not far from highs |
| 9 | RS rating ≥ 70 (ideally 80s–90s) | IBD RS Rating or equivalent percentile vs all stocks |

**Market environment check (run before screening):**
Count how many stocks pass the template on any given day:

- 30+ passing → bullish environment, trade actively
- 10–20 passing → neutral, be selective
- 0–5 passing → bearish, preserve capital and wait

> When few stocks pass, the market is telling you something. Don't force trades.

---

## Step 2 — Fundamental Filters

Technical patterns provide timing. Fundamentals provide the fuel for sustained
moves. A beautiful chart with decelerating earnings is a trap.

### 2a. Earnings (EPS)

- Quarterly EPS growth ≥ 20–50% year-over-year (higher is better)
- **Acceleration is more important than the absolute number.** A stock going
  from 10% → 20% → 40% EPS growth QoQ is more compelling than flat 30% growth
- Last 3 quarters should show no deceleration — any slowdown is a yellow flag
- Earnings surprises (beating estimates) are a green light
- Upward analyst estimate revisions ≥ 5% over the past 3 months indicate
  improving institutional expectations

### 2b. Revenue / Sales

- Quarterly sales growth ≥ 20% year-over-year
- Sales should accelerate alongside earnings — divergence (earnings up, sales flat)
  can indicate cost-cutting rather than genuine growth
- Revenue estimate beats in recent quarters reinforce the trend

### 2c. Profit Margins

- Gross margin and operating margin should be stable or expanding
- Margin expansion confirms pricing power — the company can raise prices
  without losing customers
- Sequential margin improvement (quarter over quarter) is a strong signal

### 2d. Annual EPS

- Annual EPS must be higher in the most recent year vs the prior year
- Look for at least 3 years of earnings growth as a quality baseline

### 2e. Institutional Activity

- Look for increasing fund ownership over recent quarters (13-F filings, or
  via data providers like Finviz, Simply Wall St, MarketSmith)
- Rising number of funds holding the stock indicates accumulation
- Watch for high-quality fund sponsors (top-tier growth funds), not just quantity

### What to disqualify immediately

- EPS decelerating for 2+ consecutive quarters
- Revenue growing but earnings shrinking (margin compression)
- Negative surprises or downward estimate revisions
- Declining fund ownership (distribution, not accumulation)

---

## Step 3 — Chart Review: VCP Quality

The VCP (Volatility Contraction Pattern) is Minervini's primary entry setup.
After fundamentals pass, pull up the daily and weekly charts and look for this
structure.

### What a VCP looks like

A VCP forms **within** an existing Stage 2 uptrend. The stock pauses after a
prior advance and consolidates in a series of progressively tighter pullbacks:

```
     ←————————— Prior advance ——————————→
                                          ╲
           C1 (e.g. 18% pullback)          ╲   C2 (12%)   C3 (6%)
                                            ╲___/‾╲___/‾╲__/‾‾ ← PIVOT
                                                              ↑
                                                          breakout here
```

Each contraction (C1, C2, C3...) must be:
- Smaller than the previous one (if C1 is 18%, C2 must be less than 18%)
- Accompanied by declining volume on the down leg
- Followed by a recovery that does not need high volume — the stock just firms up

### Checklist for a high-quality VCP

**Price structure:**
- [ ] 2–6 contractions visible, each tighter than the last
- [ ] Higher lows during each contraction (buyers stepping in earlier each time)
- [ ] Total base depth typically 10–35% from peak to trough — avoid deep bases
      (>50% corrections create heavy overhead supply)
- [ ] Closes near the top of the daily range during the tightest part of the base
      ("tennis ball action" — quick bounces off lows, not grinding recoveries)

**Volume structure:**
- [ ] Volume declines on each successive down leg — selling pressure fading
- [ ] Volume is near its lowest point during the final, tightest contraction
- [ ] Up days in the base show higher volume than down days (accumulation signal)
- [ ] No heavy volume spikes on down days (which would signal distribution/selling)

**Moving average structure (per Trend Template):**
- [ ] All MAs remain in correct order throughout the base
- [ ] Price does not break below the 50-day SMA during the base (if it does,
      the pattern is weakened — wait for a new setup to form)

**The pivot point:**
- The pivot is the highest price in the consolidation — the line the stock
  must close above (on volume) to trigger entry
- It is usually the high of the most recent tight range, or the left-side high
  of the base

### What disqualifies a VCP

- Pullbacks getting **larger** instead of smaller (volatility expanding, not contracting)
- Heavy volume on down days — indicates distribution, not accumulation
- Wide, erratic day-to-day swings within the base
- Base depth > 50% (too much overhead supply to work through cleanly)
- Price breaking below the 200-day SMA during the base

### Weekly vs daily charts

Use the **weekly chart** first to assess the overall base structure and
identify the number of contractions. Then drop to the **daily chart** to
identify the exact pivot point, monitor volume on individual sessions, and time
the entry.

---

## Step 4 — Entry, Stop, and Exit

### Entry: the breakout

- Enter when price **closes above the pivot point** on volume that is at least
  40–50% above the stock's average daily volume
- Do not chase — if the stock is extended more than 5% above the pivot,
  wait for the next setup rather than buying extended
- The best entries happen on the day of the breakout, ideally within 1–2% of
  the pivot
- **Check the earnings calendar first** — do not open a fresh position within
  ~2–3 weeks of a scheduled earnings report. A new position has no profit
  cushion, and an earnings gap can move straight through a 7–8% stop. Hold
  through a report only when an existing gain already covers the risk

**Early entry (optional, advanced):** Some practitioners enter within the
final, tightest contraction before the pivot is broken, using a very tight
stop. This requires high conviction in the VCP quality and pattern-reading
experience.

### Stop-loss

- Place the stop **7–8% below the buy point** (the pivot price) — hard rule,
  no exceptions
- Never move the stop lower to "give it more room"
- If the stock triggers the stop, exit immediately — the setup has failed
- A 7–8% max loss with a 3:1 reward target means you only need to be right
  ~30% of the time to be profitable

### Position sizing

- Risk no more than **1–2.5% of total account equity** per trade
- Formula: `Position Size = (Account × Risk%) / (Entry Price × Stop%)`
- Example: $100,000 account, 1% risk, entry at $50, stop at 8% below ($46):
  `$100,000 × 0.01 / ($50 × 0.08) = $2,500 position` (~50 shares)
- This keeps any single loss manageable regardless of how wrong you are

### Profit targets and exits

- Initial target: 15–20% gain from the pivot breakout
- At 20% gain, evaluate: is the move happening on strong volume with no
  signs of distribution? If yes, consider holding for a larger move
- Trail the stop up as the stock advances — use the 50-day SMA as a guide
  once well into profit
- Exit immediately if the stock closes below the 50-day SMA on heavy volume
  (sign of institutional selling)
- Sell into strength on extended moves — do not wait for the stock to roll over

---

## Putting It Together: Workflow

```
1. SCREEN (automated, daily before market open)
   Run your Python Trend Template screener.
   Output: list of stocks passing all 8 conditions.

2. FUNDAMENTAL FILTER (semi-automated, ~30 min)
   For each passing stock, check:
   - Quarterly EPS growth ≥ 20%, accelerating
   - Quarterly revenue growth ≥ 20%
   - Margins stable or expanding
   - Upward estimate revisions
   Remove any that fail. Remaining list → watchlist candidates.

3. CHART REVIEW (manual, ~1–2 min per chart)
   Pull up weekly chart first, then daily.
   Ask: is there a VCP forming?
   - Contractions getting tighter? ✓
   - Volume drying up on pullbacks? ✓
   - Clean base, no violent swings? ✓
   - Clear pivot level visible? ✓
   Add qualifying stocks to an active watchlist with the pivot price noted.

4. ENTRY MONITORING (intraday or end-of-day)
   Watch for: price crossing the pivot on ≥40% above-average volume.
   When triggered:
   - Calculate position size
   - Place entry at or near pivot
   - Set stop 7–8% below pivot (immediately, not later)
   - Note initial profit target (15–20% above pivot)
```

---

## Quick Reference: Checklist Summary

### Step 1 — Trend Template (Python screener)
- [ ] Price > 50d, 150d, 200d SMA
- [ ] 50d > 150d > 200d (stacked)
- [ ] 200d SMA rising for ≥ 1 month
- [ ] Price ≥ 30% above 52-week low
- [ ] Price ≤ 25% below 52-week high
- [ ] RS rating ≥ 70

### Step 2 — Fundamentals
- [ ] EPS growth ≥ 20%, accelerating
- [ ] Revenue growth ≥ 20%
- [ ] Margins stable or expanding
- [ ] Upward estimate revisions
- [ ] Annual EPS up year-over-year
- [ ] Institutional ownership growing

### Step 3 — VCP Chart Quality
- [ ] 2–6 contractions, each tighter
- [ ] Volume declining on each pullback
- [ ] Higher lows across contractions
- [ ] Base depth 10–35%
- [ ] Clear pivot point identified
- [ ] No distribution signals (heavy down-day volume)

### Step 4 — Entry
- [ ] Price closes above pivot on ≥40–50% above-avg volume
- [ ] Entry within 5% of pivot (no chasing)
- [ ] No earnings report scheduled within ~3 weeks (or an existing cushion covers it)
- [ ] Stop set at 7–8% below pivot immediately
- [ ] Position size ≤ 1–2.5% account risk
- [ ] Initial target noted (15–20% above pivot)

---

## Common Mistakes to Avoid

| Mistake | Why it hurts |
|--------|-------------|
| Loosening template criteria to generate more candidates | Degrades the quality filter — 7/8 conditions is meaningfully worse than 8/8 |
| Buying on template pass alone (no VCP) | Template = eligible. VCP + volume = entry signal |
| Trading when only 0–5 stocks pass the template | Market is in a correction — most breakouts fail in bad market conditions |
| Buying extended from the pivot (>5% above) | Increases risk, reduces reward, stop gets further away |
| Moving stop lower to "give it room" | Turns a controlled loss into a large loss |
| Ignoring earnings deceleration on a good chart | Fundamentals are the fuel — a great chart without fuel stalls |
| Averaging down into a losing position | Minervini's rule: add to winners, never to losers |

---

*Based on Mark Minervini's SEPA methodology as described in*
*Trade Like a Stock Market Wizard (2013) and Think & Trade Like a Champion (2017).*
*This document is for educational reference only and does not constitute financial advice.*
