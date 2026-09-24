"""Display-only SEPA reads that sit beside the numbers the cockpit already acts on.

Nothing here places, sizes, or blocks an order. Each function turns a price frame or a
regime dict into a small, testable verdict the pages (and the evening sell plan's notes)
render as a caption. Where one of these is promoted to act, the promotion is a doctrine
switch read at call time, never a change here.

Pure and light: pandas and the indicator helpers are imported inside the functions that
need them, so ``trade.py`` can import this module without pulling in the scan stack.
"""
from __future__ import annotations

from typing import Optional

from . import doctrine

TYPICAL_DAY_BARS = 42       # ~2 months — the same window as the VCP detector's dead-tape
                            # read, so the Scan table and the Positions page quote one number


def typical_day_range(df, bars: int = TYPICAL_DAY_BARS) -> Optional[float]:
    """The median daily true range over the last ``bars`` sessions, as a FRACTION of price
    — how far this stock ordinarily moves in a day. The median, not the mean: one gap day
    must not make a quiet stock look wild. None without enough bars."""
    if df is None or len(df) < bars + 1:
        return None
    import numpy as np
    from .indicators import true_range_pct
    tr = true_range_pct(df.tail(bars + 1)).tail(bars).to_numpy(dtype=float)
    med = float(np.nanmedian(tr)) if np.isfinite(tr).any() else float("nan")
    return med if np.isfinite(med) and med > 0 else None


def stop_room(day_range, stop, fill) -> Optional[dict]:
    """How many ordinary days of movement sit between ``fill`` and ``stop``.

    ``day_range`` is :func:`typical_day_range` (a fraction). A stop inside about two
    ordinary days is the book's "bucking bronco" problem: normal noise takes you out before
    the trade can work. Returns ``{loss_pct, room_days, warn}`` (``loss_pct`` a fraction), or
    None when any input is missing or the stop isn't below the fill."""
    try:
        d, s, f = float(day_range), float(stop), float(fill)
    except (TypeError, ValueError):
        return None
    if not (d > 0 and 0 < s < f):
        return None
    loss = (f - s) / f
    room = loss / d
    return {"loss_pct": loss, "room_days": room,
            "warn": room < doctrine.STOP_ROOM_MIN_DAYS}


def stop_room_text(room: Optional[dict], day_range) -> str:
    """One caption fragment for :func:`stop_room`'s result: ``'stop 3.1 typical days away
    (2.4%/day)'``, ⚠-prefixed inside ``STOP_ROOM_MIN_DAYS``. Empty when unknown."""
    if not room or not day_range:
        return ""
    head = "⚠ " if room["warn"] else ""
    tail = " — inside ordinary daily noise" if room["warn"] else ""
    return (f"{head}stop {room['room_days']:.1f} typical days away "
            f"({float(day_range) * 100:.1f}%/day){tail}")
