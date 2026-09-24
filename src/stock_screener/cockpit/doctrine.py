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

# Breakout confirmation: the close must come on >=1.5x the average of the PRIOR
# VOL_AVG_DAYS bars. Excluding the current bar is the point — including it dilutes the
# very spike being tested, and the dilution grows with the size of the spike.
VOL_CONFIRM_RATIO = 1.5
VOL_AVG_DAYS = 50
