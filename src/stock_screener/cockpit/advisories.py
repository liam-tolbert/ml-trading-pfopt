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


POST_BREAKOUT_DAYS = 20     # the 20-day line is the breakout's test for its first ~month
SMA_SHORT = 20
GIVEBACK_GAIN = 0.05        # a post-entry gain this big, fully given back, is a violation
LOWER_LOWS_RUN = 3          # this many consecutive lower lows without buying support


def post_breakout_read(df, entry_date, *, avg_entry=None, below_sma50=False,
                       volume_ratio=None, today=None, now=None) -> Optional[dict]:
    """How a breakout has behaved since the buy: Minervini's post-breakout VIOLATIONS
    (reasons to doubt it) and FOLLOW-THROUGH (signs it is working), over the settled bars
    from the entry day on.

    Violations: a close under the 20-day line inside the first ``POST_BREAKOUT_DAYS``; a
    light-volume breakout day followed by a heavy down day; ``LOWER_LOWS_RUN`` lower lows in
    a row with no above-average-volume up close among them; more down closes than up (3+
    sessions); more closes in the lower half of the day's range than the upper; a close
    under the 50-day on heavy volume; a +``GIVEBACK_GAIN`` gain fully given back.
    Follow-through: up closes on rising volume, 3 of the first 4 / 6 of the first 8
    sessions up, more upper-half closes, and every dip recovered within two sessions.

    A live read (``today`` None) drops today's bar while its session is still open — its
    "close" is the latest print (:func:`triggers.bar_is_provisional`). Returns ``{day_n,
    violations, follow_through, sma20, provisional_dropped}`` (lists of short strings), or
    None without a frame or an entry date."""
    if df is None or not len(df) or entry_date is None:
        return None
    import numpy as np
    import pandas as pd

    from .indicators import prior_volume_average

    try:
        e = pd.Timestamp(entry_date)
        if e.tzinfo is not None:
            e = e.tz_convert("America/New_York").tz_localize(None)
        e = e.normalize()
    except Exception:
        return None
    dropped = False
    if today is None:
        from .triggers import bar_is_provisional
        if bar_is_provisional(df.index[-1], now):
            df, dropped = df.iloc[:-1], True
    if not len(df):
        return None

    close = df["Close"].astype(float)
    sma20 = close.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean()
    vol = df["Volume"].astype(float) if "Volume" in df.columns else None
    vavg = (prior_volume_average(vol, doctrine.VOL_AVG_DAYS)
            if vol is not None else None)
    idx = pd.DatetimeIndex(df.index).normalize()
    post_pos = np.flatnonzero(idx >= e)
    if not len(post_pos):
        return None
    first = int(post_pos[0])
    post = df.iloc[first:]
    n = len(post) - 1                                   # day 0 = the entry day
    c = post["Close"].to_numpy(dtype=float)
    hi = post["High"].to_numpy(dtype=float)
    lo = post["Low"].to_numpy(dtype=float)
    vr = ((vol / vavg).iloc[first:].to_numpy(dtype=float)
          if vol is not None else np.full(len(post), np.nan))
    s20 = sma20.iloc[first:].to_numpy(dtype=float)

    violations, follow = [], []

    # 20-day line: the breakout's first test. Strict "<" so a flat tape never trips it.
    under = [d for d in range(1, min(n, POST_BREAKOUT_DAYS) + 1)
             if np.isfinite(s20[d]) and c[d] < s20[d]]
    sma20_note = None
    if under:
        sma20_note = (f"closed below the 20-day line on day {under[-1]}"
                      + (f" ({len(under)}×)" if len(under) > 1 else ""))
        violations.append(sma20_note)

    ups = [d for d in range(1, n + 1) if c[d] > c[d - 1]]
    downs = [d for d in range(1, n + 1) if c[d] < c[d - 1]]

    # A breakout on light volume, then institutions selling into it.
    heavy = doctrine.VOL_CONFIRM_RATIO
    if np.isfinite(vr[0]) and vr[0] < heavy:
        hd = [d for d in downs if np.isfinite(vr[d]) and vr[d] >= heavy]
        if hd:
            violations.append(f"light-volume breakout ({vr[0]:.1f}×) then a heavy down "
                              f"day (day {hd[0]}, {vr[hd[0]]:.1f}×)")

    # Lower lows in a row with nobody stepping in (no above-average-volume up close).
    run = 0
    for d in range(1, n + 1):
        support = d in ups and np.isfinite(vr[d]) and vr[d] >= 1.0
        run = run + 1 if (lo[d] < lo[d - 1] and not support) else 0
        if run >= LOWER_LOWS_RUN:
            violations.append(f"{run} lower lows in a row (to day {d})")
            break

    if n >= 3 and len(downs) > len(ups):
        violations.append(f"more down closes than up ({len(downs)} vs {len(ups)})")

    rng = hi[1:] - lo[1:]
    pos_in_range = np.where(rng > 0, (c[1:] - lo[1:]) / np.where(rng > 0, rng, 1.0), np.nan)
    lower_half = int(np.sum(pos_in_range < 0.5))
    upper_half = int(np.sum(pos_in_range > 0.5))
    if n >= 3 and lower_half > upper_half:
        violations.append(f"more closes in the lower half of the day's range "
                          f"({lower_half} vs {upper_half})")

    if below_sma50 and volume_ratio is not None and volume_ratio >= heavy:
        violations.append("closed below the 50-day on heavy volume")

    if avg_entry and n >= 1:
        best = float(np.max(hi[1:])) / float(avg_entry) - 1.0
        if best >= GIVEBACK_GAIN and c[-1] <= float(avg_entry):
            violations.append(f"gave back a +{best * 100:.0f}% gain")

    # Follow-through.
    rising = [d for d in ups if vol is not None
              and float(post["Volume"].iloc[d]) > float(post["Volume"].iloc[d - 1])]
    if rising:
        follow.append(f"{len(rising)} up close(s) on rising volume")
    if n >= 4 and sum(1 for d in ups if d <= 4) >= 3:
        follow.append("3 of the first 4 sessions up")
    if n >= 8 and sum(1 for d in ups if d <= 8) >= 6:
        follow.append("6 of the first 8 sessions up")
    if n >= 3 and upper_half > lower_half:
        follow.append(f"more upper-half closes ({upper_half} vs {lower_half})")
    judged = [d for d in downs if d + 2 <= n]
    if judged and all(max(c[d + 1], c[d + 2]) > c[d - 1] for d in judged):
        follow.append("every dip recovered within 2 sessions")

    return {"day_n": n, "violations": violations, "follow_through": follow,
            "sma20": sma20_note, "provisional_dropped": dropped}


def stop_room_text(room: Optional[dict], day_range) -> str:
    """One caption fragment for :func:`stop_room`'s result: ``'stop 3.1 typical days away
    (2.4%/day)'``, ⚠-prefixed inside ``STOP_ROOM_MIN_DAYS``. Empty when unknown."""
    if not room or not day_range:
        return ""
    head = "⚠ " if room["warn"] else ""
    tail = " — inside ordinary daily noise" if room["warn"] else ""
    return (f"{head}stop {room['room_days']:.1f} typical days away "
            f"({float(day_range) * 100:.1f}%/day){tail}")
