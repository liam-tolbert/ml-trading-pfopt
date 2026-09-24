"""The trading rules that more than one module has to agree on. Constants only.

These numbers ARE the doctrine (HANDOFF §7). Each was previously written out in two to
four modules, and the copies had already drifted: the weekend hunt confirmed breakouts on
a looser volume ratio, over a different averaging window, than the daily trigger job that
is supposed to enforce the same rule. A number the user reasons about as one rule must
exist once.

**This module imports nothing** — not pandas, not any sibling. That is what lets
``trade.py`` (which must stay import-light) and ``hunt/pipeline.py`` (which must not drag
in the Streamlit stack) both read it, with no cycle through ``scan``/``triggers``.

Paths and cache-format versions live in ``cache.py``, not here.
"""
from __future__ import annotations

# Entry is refused inside ~3 weeks of a scheduled report: an unpriced binary is not a
# setup, however good the base looks. Advisory in the app, a hard block in the hunt.
EARNINGS_SOON_DAYS = 21

# Minervini's hard maximum loss, measured from the PRICE PAID ("never more than 10% below
# your purchase price"), not from the pivot: a buy near the top of the zone with a
# pivot-based stop would otherwise risk up to 14.3%. The plan builder raises a stop to
# this floor for the worst-case fill; paying more therefore buys a tighter stop, never a
# wider loss. A stop further away is not a looser risk budget, it is a different trade.
MAX_LOSS_FROM_FILL = 0.10

# The default stop when no tighter support exists: 7.5% below the pivot, the middle of the
# book's 7-8%. Every consumer (scan levels, frozen-pivot plans, R reconstruction) reads it
# here so a stop quoted on the Scan page is the stop the order carries.
DEFAULT_STOP_FROM_PIVOT = 0.075

# No chasing: the buy zone is the pivot to 5% above it. Past that the name is "extended" —
# the stop from a fill up there is too far below the base for the risk to be the setup's.
NO_CHASE_PCT = 0.05

# The book's stop sizing once you have numbers: no more than HALF your average gain, from
# the fill. Below DERIVED_STOP_MIN_WINS winning trades the average is noise, so the 7.5%
# default stands. The floor keeps a tiny average win from producing a stop inside ordinary
# daily noise: set by the rule pre-registered in HANDOFF §6.73 (the tightest stop >= 3% that
# turned none of the closed winners into a loss), on ONE winner — re-apply the same rule
# when the 5th win activates this, never pick the stop that maximises the replayed return.
DERIVED_STOP_MIN_WINS = 5
DERIVED_STOP_WIN_FRACTION = 0.5
DERIVED_STOP_FLOOR = 0.04

# A stop fewer than this many ORDINARY days of movement (median daily true range) below the
# fill is inside normal noise — the book's "bucking bronco": the stock shakes you out on an
# ordinary day before the trade can work. Advisory: the caption says so, the stop is yours.
STOP_ROOM_MIN_DAYS = 2.0

# Post-breakout violations (a close under the 20-day line, heavy selling, lower lows, a
# gain given back...) are warnings in the P1 pillar and the evening plan's notes. With the
# switch ON, VIOLATION_FAIL_COUNT of them at once fail P1 — and a P1 fail is an automatic
# full-exit order in the evening sell plan. Off until the warnings have been watched on
# live positions for a while: the counters are new and have never been wrong in public.
VIOLATIONS_CAN_FAIL = False
VIOLATION_FAIL_COUNT = 3

# In a weak or choppy market the book tightens up: stops around 5-6% instead of 7-8%, and
# profits taken sooner, at 10-12%. Shown beside the plan's own numbers as advice — never
# applied, because this record's replay (HANDOFF §6.73) says tighter stops cost money here.
WEAK_TAPE_STOP_PCT = (0.05, 0.06)
WEAK_TAPE_TARGET_PCT = (0.10, 0.12)

# The one market-timing rule the backtest validated out of sample: when SPY enters Stage 4,
# get defensive; after it recovers, wait REGIME_CONFIRM_DAYS sessions of a buy-able tape
# before adding again. 15 is the value chosen on 2003-13 and TESTED on 2014-24 (HANDOFF
# §1); 25 was picked by looking at the full history and is tainted. Do not tune it.
REGIME_CONFIRM_DAYS = 15
# ADVISORY until switched on: the evening plan then also orders a partial sell of every
# position without a full exit, on the evening SPY first closes in Stage 4 (the backtest
# exited fully; half is the book's "reduce exposure").
MARKET_TURN_CAN_TRADE = False
MARKET_TURN_REDUCE_FRACTION = 0.5

# Breakout confirmation: the close must come on >=1.5x the average of the PRIOR
# VOL_AVG_DAYS bars. Excluding the current bar is the point — including it dilutes the
# very spike being tested, and the dilution grows with the size of the spike.
VOL_CONFIRM_RATIO = 1.5
VOL_AVG_DAYS = 50
