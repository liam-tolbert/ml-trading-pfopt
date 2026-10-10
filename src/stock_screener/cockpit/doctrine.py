"""Trading rules that more than one module reads. Constants only.

These numbers are the trading rules (HANDOFF §7). Each rule MUST be defined here only:
copies in several modules drift apart.

This module MUST NOT import anything, pandas or other cockpit modules included. ``trade.py``
and ``hunt/pipeline.py`` read it; they MUST stay quick to import and MUST NOT end up in a
circular import through ``scan``/``triggers``.

Paths and cache-format versions live in ``cache.py``.
"""
from __future__ import annotations

# The trend template's eighth criterion: an RS rating of at least 70. The books want the
# 80s-90s; 70 is the minimum. Step 1 of the scan, the P2 sell check, the app's slider
# default and the weekend hunt MUST all read this one number.
RS_FLOOR = 70

# Liquidity: one order MUST stay within this share of the stock's average daily dollar
# volume, measured over ADV_DAYS sessions. A bigger order moves the price against you.
# The plan builder cuts the share count to fit; submitting or arming (scheduling for the
# morning) refuses a row above it.
ADV_DAYS = 20
MAX_ORDER_ADV_PCT = 0.02

# No entry within ~3 weeks of a scheduled report: the price can jump on the report, and
# nobody knows which way. The app warns; the weekend hunt skips the stock.
EARNINGS_SOON_DAYS = 21

# The hard maximum loss: 10% below the price PAID, not the pivot. A stop based on the
# pivot, under a buy at the top of the buy zone, would risk 14.3%. The plan builder raises
# the stop to this level for the highest possible buy price, so paying more gives a tighter
# stop, never a wider loss.
MAX_LOSS_FROM_FILL = 0.10

# The default stop: 7.5% below the pivot, the middle of Minervini's 7-8%. Every module that
# uses it MUST read it here, so the stop the Scan page shows is the stop the order carries.
DEFAULT_STOP_FROM_PIVOT = 0.075

# The buy zone runs from the pivot to 5% above it. Above that the stock is too far past
# the pivot and SHOULD NOT be bought.
NO_CHASE_PCT = 0.05

# Minervini's stop once there are numbers: half the average win, measured from the buy
# price. Under DERIVED_STOP_MIN_WINS wins the average is too unreliable, and the default
# stop applies. The floor keeps the stop out of ordinary daily price swings. It comes from
# the rule written down in advance in HANDOFF §6.73, applied to one winner. That rule MUST
# be applied again when the 5th win turns this on. The floor MUST NOT be the stop that
# gives the best return when past trades are replayed.
DERIVED_STOP_MIN_WINS = 5
DERIVED_STOP_WIN_FRACTION = 0.5
DERIVED_STOP_FLOOR = 0.04

# The books' triggers for raising the stop to breakeven, in place of fixed gains: the
# rising 50-day average reaching the buy price, a gain of BREAKEVEN_R times the initial
# risk, or BREAKEVEN_AVG_WIN_MULT times the average win. The last also sets a second stop
# (the "back stop") at the average win. The average-win rules wait for
# DERIVED_STOP_MIN_WINS wins.
BREAKEVEN_R = 3.0
BREAKEVEN_AVG_WIN_MULT = 2.0

# A stop fewer than this many typical days (median daily true range) below the buy price
# is within ordinary daily movement: one normal day can hit it. Advisory only.
STOP_ROOM_MIN_DAYS = 2.0

# Post-breakout violations warn in the P1 sell check and in the evening plan's notes. With
# the switch on, VIOLATION_FAIL_COUNT of them fail P1, and a P1 fail sells the whole
# position. The switch SHOULD stay off until the warnings have been checked against real
# positions.
VIOLATIONS_CAN_FAIL = False
VIOLATION_FAIL_COUNT = 3

# Minervini's weak-market numbers: 5-6% stops, profits taken at 10-12%. Shown as advice
# beside the plan's own numbers. They MUST NOT be applied automatically: replaying past
# trades in HANDOFF §6.73 shows tighter stops lost money on this trade record.
WEAK_TAPE_STOP_PCT = (0.05, 0.06)
WEAK_TAPE_TARGET_PCT = (0.10, 0.12)

# The market-timing rule: SPY in Stage 4 means cut risk. After SPY recovers, wait this
# many sessions before buying again. The value is fixed (HANDOFF_HISTORY §1 says how) and
# MUST NOT be tuned.
REGIME_CONFIRM_DAYS = 15
# Each session counted toward that wait also needs this percent of the scanned stocks in
# Stage 2 (the market's breadth). MUST equal the min_phase2_pct default of
# `minervini_screener`'s should_generate_signals, or the banner and the wait would disagree.
BREADTH_MIN_PHASE2 = 15.0
# Off: the market turn is advice only. On: the evening SPY first closes in Stage 4, the
# plan sells this fraction of each position, not all of it: Minervini's "reduce exposure".
MARKET_TURN_CAN_TRADE = False
MARKET_TURN_REDUCE_FRACTION = 0.5

# Selling part of a position while it is still rising. The books' signs of a climax run
# (a late, very steep rise) are read only once a position is up STRENGTH_MIN_GAIN: early in
# a rise the same behaviour is healthy, and the cockpit doesn't count bases. Off: the signs
# are notes. On: STRENGTH_SIGNS_TO_SELL signs on one close plan a sale of
# STRENGTH_SELL_FRACTION, with the stop on the rest raised to the buy price, once per
# position. The switch SHOULD stay off until the signs have been checked against real
# rises.
STRENGTH_MIN_GAIN = 0.20
STRENGTH_CAN_TRADE = False
STRENGTH_SIGNS_TO_SELL = 3
STRENGTH_SELL_FRACTION = 0.5

# Breakout confirmation: volume >= 1.5x the average of the PRIOR VOL_AVG_DAYS bars. The
# current bar MUST be excluded; it would dilute the spike being tested.
VOL_CONFIRM_RATIO = 1.5
VOL_AVG_DAYS = 50

# The reaction to the last earnings report: a session that moved this many percent on
# VOL_CONFIRM_RATIO volume. The books say a stock that "drops hard on huge volume" after
# its report should be left alone, but give no number; 5% is this project's own choice.
EARNINGS_REACTION_PCT = 5.0
