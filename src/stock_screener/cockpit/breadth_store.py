"""Daily market-breadth history: one row per settled session.

The books read the market by the count of new 52-week highs against new lows, and by
whether that spread widens. That needs history the scan pickle doesn't keep. ``append``
MUST be called only by the scheduled screen, so each row is a settled close; an in-app
scan mid-session would write a provisional one.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

from . import cache

COLUMNS = ("date", "n_scanned", "phase2_pct", "new_highs", "new_lows")
SPREAD_LOOKBACK = 10        # sessions; "widening" compares with the spread this far back


def _path(path) -> Path:
    return Path(path) if path is not None else cache.BREADTH_CSV


def _parse(row: dict) -> Optional[dict]:
    try:
        return {"date": str(row["date"]), "n_scanned": int(float(row["n_scanned"])),
                "phase2_pct": float(row["phase2_pct"]),
                "new_highs": int(float(row["new_highs"])),
                "new_lows": int(float(row["new_lows"]))}
    except (KeyError, TypeError, ValueError):
        return None


def load(path=None) -> List[dict]:
    """Rows sorted by date, numbers parsed. ``[]`` when the file is missing or unreadable;
    a malformed row is skipped."""
    p = _path(path)
    try:
        with open(p, newline="", encoding="utf-8") as f:
            rows = [r for r in (_parse(x) for x in csv.DictReader(f)) if r]
    except (OSError, csv.Error):
        return []
    return sorted(rows, key=lambda r: r["date"])


def append(row: dict, path=None) -> Path:
    """Add the row for ``row["date"]``, replacing any row with that date, and rewrite the
    file atomically (tmp + ``os.replace``). Returns the path. Raises on a malformed row."""
    new = _parse(row)
    if new is None:
        raise ValueError(f"malformed breadth row: {row!r}")
    p = _path(path)
    rows = [r for r in load(p) if r["date"] != new["date"]] + [new]
    rows.sort(key=lambda r: r["date"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, p)
    return p


def phase2_by_date(rows: List[dict]) -> Dict[str, float]:
    return {r["date"]: r["phase2_pct"] for r in rows}


def spread_expanding(rows: List[dict], session: str, spread_today: float,
                     lookback: int = SPREAD_LOOKBACK) -> Optional[bool]:
    """Whether today's new-high minus new-low spread exceeds the spread ``lookback``
    settled sessions before ``session``. None with too little history."""
    prior = [r for r in rows if r["date"] < session]
    if len(prior) < lookback:
        return None
    ref = prior[-lookback]
    return bool(spread_today > ref["new_highs"] - ref["new_lows"])
