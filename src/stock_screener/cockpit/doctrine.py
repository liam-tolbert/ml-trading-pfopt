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

# Minervini's hard maximum, measured from the PIVOT, not from the fill: a stop further
# away than this is not a tighter risk budget, it is a different trade.
MAX_STOP_FROM_PIVOT = 0.10

# Breakout confirmation: the close must come on >=1.5x the average of the PRIOR
# VOL_AVG_DAYS bars. Excluding the current bar is the point — including it dilutes the
# very spike being tested, and the dilution grows with the size of the spike.
VOL_CONFIRM_RATIO = 1.5
VOL_AVG_DAYS = 50
