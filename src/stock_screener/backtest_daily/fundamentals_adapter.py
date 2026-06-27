"""Map quarterly fundamentals -> the dict the vendored buy scorer consumes.

``score_buy_signal`` reads these keys off the ``fundamentals`` dict:
  - ``quarterly_revenue``    : {datadate -> revenue}  (it sorts by index, needs >=4 for the 3Q path)
  - ``revenue_qoq_change``   : % vs previous quarter
  - ``revenue_yoy_change``   : % vs the t-4 quarter
  - ``eps_yoy_change``       : % vs the t-4 quarter
  - ``inventory_qoq_change`` : % vs previous quarter

POINT-IN-TIME: only quarters whose report date (``rdq``) is <= the as-of date are
visible. This single ``rdq <= asof`` filter is the entire leak guarantee on
fundamentals. Returns ``None`` when no quarter is known yet (the scorer then falls
back to its neutral score).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Columns expected on the per-name quarterly frame (Compustat fundq names kept
# generic so the WRDS adapter and the synthetic provider share this code).
REQUIRED_COLS = ("datadate", "rdq", "revtq", "eps", "invtq")


def _pct(curr, prev):
    if prev is None or curr is None or pd.isna(prev) or pd.isna(curr) or prev == 0:
        return None
    return (curr - prev) / abs(prev) * 100.0


def compustat_to_scorer_dict(quarters: pd.DataFrame, asof) -> Optional[dict]:
    """Build the scorer's fundamentals dict from a per-name quarterly frame, as of
    ``asof`` (a Timestamp). ``quarters`` must have the columns in ``REQUIRED_COLS``
    with ``datadate``/``rdq`` parseable as datetimes.
    """
    if quarters is None or len(quarters) == 0:
        return None
    asof = pd.Timestamp(asof)
    q = quarters.copy()
    q["rdq"] = pd.to_datetime(q["rdq"])
    q["datadate"] = pd.to_datetime(q["datadate"])
    known = q[q["rdq"] <= asof].sort_values("datadate")
    if known.empty:
        return None

    rev = known["revtq"].to_numpy(dtype=float)
    eps = known["eps"].to_numpy(dtype=float)
    inv = known["invtq"].to_numpy(dtype=float)

    out = {
        "quarterly_revenue": {d: float(v) for d, v in zip(known["datadate"], rev)
                              if not pd.isna(v)},
        "revenue_qoq_change": _pct(rev[-1], rev[-2]) if len(rev) >= 2 else None,
        "revenue_yoy_change": _pct(rev[-1], rev[-5]) if len(rev) >= 5 else None,
        "eps_yoy_change": _pct(eps[-1], eps[-5]) if len(eps) >= 5 else None,
        "inventory_qoq_change": _pct(inv[-1], inv[-2]) if len(inv) >= 2 else None,
    }
    return out
