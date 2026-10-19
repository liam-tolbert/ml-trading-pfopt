"""Swing readers for the VCP detector: plain numpy over one price window.

Two readers look at the same bars. ``extremes_walk`` finds the base's legs: a swing
high is a high that price pulls back at least the leg floor from before trading above
it, and the leg's trough is the lowest low before price trades above that high again. It
needs no reversal size beyond the floor every leg must clear. ``zigzag_pivots`` marks a
swing once price reverses by a fixed fraction; ``vcp.detect_vcp`` runs it once over the
base for the answers that compare one swing with the next.

A segment runs from one breakout close (a close above every earlier high in the window)
to the next. A base sits inside one segment, or across several whose tops stay close.

Every function takes plain float arrays and integer bar positions, never dates.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def segments(high: np.ndarray, close: np.ndarray) -> List[Tuple[int, int]]:
    """``[(start, end)]`` bar ranges, end exclusive, covering the whole window.

    A new segment starts at every bar whose close is strictly above the highest high
    before it. A close equal to the old high does not count: a test of the top is not a
    breakout. Empty for an empty window."""
    n = len(high)
    if n == 0:
        return []
    prev_max = np.empty(n)
    prev_max[0] = np.inf
    if n > 1:
        prev_max[1:] = np.maximum.accumulate(high[:-1])
    starts = [0] + [int(i) for i in np.flatnonzero(close > prev_max) if i > 0]
    return list(zip(starts, starts[1:] + [n]))


def next_peak(high: np.ndarray, low: np.ndarray, start: int, end: int,
              min_depth: float) -> Optional[int]:
    """The first swing high at or after ``start``: the running highest high at the first
    bar, later than that high, whose low sits at least ``min_depth`` (a fraction) below
    it. On equal highs the first counts. None when price never pulls back that far
    before ``end``."""
    if end - start < 2:
        return None
    h, lo = high[start:end], low[start:end]
    run = np.maximum.accumulate(h)
    pos = np.arange(len(h))
    new = np.ones(len(h), dtype=bool)
    new[1:] = h[1:] > run[:-1]
    at = np.maximum.accumulate(np.where(new, pos, 0))
    hit = np.flatnonzero((lo <= run * (1.0 - min_depth)) & (pos > at))
    return start + int(at[hit[0]]) if len(hit) else None


def walk_start(high: np.ndarray, low: np.ndarray, start: int, end: int,
               min_depth: float) -> Optional[int]:
    """Where the walk starts in a segment: the highest high before the segment's lowest
    low. When the lowest low is the segment's first bar, the first swing high
    (:func:`next_peak`).

    Starting at the first swing high everywhere would let a small dip just after the
    breakout bar stand in for the base's top."""
    t = start + int(np.argmin(low[start:end]))
    if t > start:
        return start + int(np.argmax(high[start:t]))
    return next_peak(high, low, start, end, min_depth)


def extremes_walk(high: np.ndarray, low: np.ndarray, peak: Optional[int], end: int, *,
                  min_leg_bars: int, min_depth: float) -> List[Tuple[int, int]]:
    """``[(peak_i, trough_i)]`` legs from ``peak`` up to ``end`` (exclusive).

    A leg's trough is the lowest low after its peak and before price trades above that
    peak. The next peak is the next swing high after the trough (:func:`next_peak`), so a
    climb back to the top at the end of the window adds no leg and hides none. A trough
    fewer than ``min_leg_bars`` bars after its peak is a one-bar plunge: it is not a leg,
    and the walk goes on from the next swing high. A leg shallower than ``min_depth`` is
    dropped the same way. A trough on the last bar is a leg still forming."""
    legs: List[Tuple[int, int]] = []
    h = peak
    while h is not None and h + 1 < end and high[h] > 0:
        above = np.flatnonzero(high[h + 1:end] > high[h])
        stop = h + 1 + int(above[0]) if len(above) else end
        if stop <= h + 1:
            h = next_peak(high, low, h + 1, end, min_depth)
            continue
        t = h + 1 + int(np.argmin(low[h + 1:stop]))
        if t - h >= min_leg_bars and 1.0 - low[t] / high[h] >= min_depth:
            legs.append((h, t))
        h = next_peak(high, low, t + 1, end, min_depth)
    return legs


def zigzag_pivots(high, low, thr: float) -> List[tuple]:
    """Fast percentage ZigZag: :func:`zigzag_pivots_ref` fed plain-Python floats.
    Indexing an ndarray one element at a time creates a new np.float64 on every access.
    ``test_zigzag_fast_parity`` checks it matches the reference version."""
    return zigzag_pivots_ref(np.asarray(high).tolist(), np.asarray(low).tolist(), thr)


def zigzag_pivots_ref(high, low, thr: float) -> List[tuple]:
    """Percentage ZigZag -> strictly alternating ``(index, price, kind)`` pivots, kind
    'H' or 'L'.

    A swing high is confirmed once price falls ``thr`` below the running high; a swing low
    once it rises ``thr`` above the running low. Every 'H' is followed by an 'L', so every
    down-leg has its trough below its peak. The final running extreme is appended as a
    pivot not yet confirmed. Accepts any indexable sequence. Empty for fewer than 2 bars.
    """
    n = len(high)
    piv: List[tuple] = []
    if n < 2:
        return piv
    trend = 0                                  # 0 unknown, +1 up, -1 down
    hi_i, hi_p = 0, high[0]
    lo_i, lo_p = 0, low[0]
    for i in range(1, n):
        if trend >= 0 and high[i] > hi_p:
            hi_i, hi_p = i, high[i]
        if trend <= 0 and low[i] < lo_p:
            lo_i, lo_p = i, low[i]
        if trend == 0:
            if low[i] <= hi_p * (1 - thr):
                piv.append((hi_i, hi_p, 'H')); trend = -1; lo_i, lo_p = i, low[i]
            elif high[i] >= lo_p * (1 + thr):
                piv.append((lo_i, lo_p, 'L')); trend = 1; hi_i, hi_p = i, high[i]
        elif trend > 0:
            if low[i] <= hi_p * (1 - thr):
                piv.append((hi_i, hi_p, 'H')); trend = -1; lo_i, lo_p = i, low[i]
        else:
            if high[i] >= lo_p * (1 + thr):
                piv.append((lo_i, lo_p, 'L')); trend = 1; hi_i, hi_p = i, high[i]
    if trend > 0:
        piv.append((hi_i, hi_p, 'H'))
    elif trend < 0:
        piv.append((lo_i, lo_p, 'L'))
    return piv


def zigzag_legs(pivots: List[tuple]) -> List[Tuple[int, int]]:
    """``[(peak_i, trough_i)]`` for each 'H' pivot directly followed by an 'L' pivot."""
    return [(a[0], b[0]) for a, b in zip(pivots, pivots[1:]) if a[2] == 'H' and b[2] == 'L']


def tight_area(high: np.ndarray, low: np.ndarray, close: np.ndarray, end: int,
               last_trough: Optional[int], *, max_range: float, min_bars: int,
               tight_bars: int, max_bars: int) -> Tuple[str, int, Optional[float]]:
    """The final tight area before ``end`` (exclusive): ``(state, bars, area_high)``.

    ``bars`` is the longest trailing stretch of up to ``max_bars`` bars whose high-to-low
    range is at most ``max_range`` of the close at ``end - 1``. ``state`` is
    ``not_formed`` when fewer than ``min_bars`` bars, the trough included, follow
    ``last_trough`` (or there is no trough); ``tight`` when ``bars >= tight_bars``;
    otherwise ``loose``. ``area_high`` is the highest high over the stretch, or over the
    last ``min_bars`` bars when the stretch is shorter; None when not formed."""
    if last_trough is None or end - last_trough < min_bars:
        return "not_formed", 0, None
    k = min(max_bars, end)
    ref = close[end - 1]
    if not np.isfinite(ref) or ref <= 0:
        return "loose", 0, None
    hi = np.maximum.accumulate(high[end - k:end][::-1])
    lo = np.minimum.accumulate(low[end - k:end][::-1])
    bars = int(np.sum((hi - lo) / ref <= max_range))
    state = "tight" if bars >= tight_bars else "loose"
    span = max(bars, min_bars)
    return state, bars, float(np.max(high[end - span:end]))
