"""Trading rules that more than one module reads. Constants only.

These numbers are the doctrine (HANDOFF §7). Each rule MUST be defined here once:
copies in several modules drift apart.

This module MUST NOT import anything, pandas or siblings included. ``trade.py`` and
``hunt/pipeline.py`` read it and must stay light, with no import cycle through
``scan``/``triggers``.

Paths and cache-format versions live in ``cache.py``.
"""
from __future__ import annotations

# The trend template's eighth criterion: an RS rating of at least 70. The books want the
# 80s-90s; 70 is the floor. The scan gate, the P2 pillar, the app's slider default and the
# hunt MUST all read this one number.
RS_FLOOR = 70

# No entry within ~3 weeks of a scheduled report: an earnings gap is an unpriced risk.
# The app warns; the hunt blocks.
EARNINGS_SOON_DAYS = 21

# The hard maximum loss: 10% below the price PAID, not the pivot. A pivot-based stop under
# a zone-top fill would risk 14.3%. The plan builder raises the stop to this floor for the
# worst-case fill, so paying more buys a tighter stop, never a wider loss.
MAX_LOSS_FROM_FILL = 0.10

# The default stop: 7.5% below the pivot, the middle of the book's 7-8%. Consumers MUST
# read it here, so the stop the Scan page quotes is the stop the order carries.
DEFAULT_STOP_FROM_PIVOT = 0.075

# The buy zone runs from the pivot to 5% above it. Above that the name is extended and
# SHOULD NOT be bought.
NO_CHASE_PCT = 0.05

# The book's stop once there are numbers: half the average win, measured from the fill.
# Under DERIVED_STOP_MIN_WINS wins the average is noise, and the default stop applies.
# The floor keeps the stop out of ordinary daily noise. It comes from the rule
# pre-registered in HANDOFF §6.73, applied to one winner. That rule MUST be re-applied
# when the 5th win activates this. The floor MUST NOT be the stop that maximises the
# replayed return.
DERIVED_STOP_MIN_WINS = 5
DERIVED_STOP_WIN_FRACTION = 0.5
DERIVED_STOP_FLOOR = 0.04

# A stop fewer than this many typical days (median daily true range) below the fill sits
# inside ordinary noise: an ordinary day shakes the trade out. Advisory only.
STOP_ROOM_MIN_DAYS = 2.0

# Post-breakout violations warn in P1 and in the evening plan's notes. With the switch
# on, VIOLATION_FAIL_COUNT of them fail P1, and a P1 fail orders a full exit. The switch
# SHOULD stay off until the warnings have been checked against live positions.
VIOLATIONS_CAN_FAIL = False
VIOLATION_FAIL_COUNT = 3

# The book's weak-market numbers: 5-6% stops, profits taken at 10-12%. Shown as advice
# beside the plan's own numbers. They MUST NOT be applied automatically: the replay in
# HANDOFF §6.73 says tighter stops cost money on this record.
WEAK_TAPE_STOP_PCT = (0.05, 0.06)
WEAK_TAPE_TARGET_PCT = (0.10, 0.12)

# The one market-timing rule validated out of sample: SPY in Stage 4 means get defensive.
# After SPY recovers, wait this many sessions before adding. 15 was chosen on 2003-13 and
# tested on 2014-24 (HANDOFF §1); 25 was fitted on the full history. It MUST NOT be tuned.
REGIME_CONFIRM_DAYS = 15
# Off: the market turn is advice only. On: the evening SPY first closes in Stage 4, the
# plan sells this fraction of each position without a full exit. The backtest exited
# fully; half is the book's "reduce exposure".
MARKET_TURN_CAN_TRADE = False
MARKET_TURN_REDUCE_FRACTION = 0.5

# Breakout confirmation: volume >= 1.5x the average of the PRIOR VOL_AVG_DAYS bars. The
# current bar MUST be excluded; it would dilute the spike being tested.
VOL_CONFIRM_RATIO = 1.5
VOL_AVG_DAYS = 50
