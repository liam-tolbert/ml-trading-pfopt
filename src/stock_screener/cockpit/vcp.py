"""The cockpit's VCP / contraction detector. It replaces, with the same inputs and
outputs, the one in the copy of ``minervini_screener`` kept in this repo.

How it reads a chart (the readers are in ``swings.py``):

1. The window is cut into segments at every breakout close. The newest segments that
   hold pullbacks are joined into one base while their tops stay within
   ``PEAK_FLAT_BAND``. A segment with no pullback after the base is the breakout run.
2. Inside each segment a walk finds the legs: from the top, the lowest low before price
   trades above that top again, then the next high that price pulls back at least
   ``THR_MIN_TIGHT`` from, and so on. These legs are the ``contractions``.
3. One ZigZag pass at ``THR_FIXED_TIGHT`` over the base gives every swing. It is the
   only reader that sees a pullback deeper than the one before, which the walk merges.
4. The final tight area is read on its own from the last bars before any breakout,
   because a swing reader set wide enough for the early base cannot see a 3% squeeze.

Design goal: a first-pass filter that lets borderline cases through rather than miss real ones. It MUST
NOT hide a setup that could still be bought. So instead of one yes/no it assigns a tier:

  A — review: a valid tightening base with price within ~10% of its top.
  B — watch: a base still forming, or a valid one already extended. Never hidden.
  C — skipped, with the reason recorded: only safe exclusions (a stock that has barely
      moved, no pullbacks, a base whose newest pullback is too old).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import swings
from .indicators import relative_measured_volatility, true_range_pct

# --- Settings. Change one only with a reason fixed before it is measured on the user's
# hand-judged hunt charts (HANDOFF §2, §12). --- #
THR_FIXED_TIGHT = 0.035    # the ZigZag reader's reversal size: still sees quiet bases' 4-6% legs
THR_MIN_TIGHT = 0.02       # a pullback shallower than this is not a leg, and no swing high
TIGHTEN_MIN_PCT = 60.0     # is_vcp minimum on the 0-100 tightening-quality score
MIN_CONTRACTIONS = 2
MAX_CONTRACTIONS = 6
MAX_BASE_WEEKS = 65        # a leg that started earlier than this isn't the current base
MAX_DEPTH_PCT = 35.0       # a leg deeper than this is a decline, not a contraction, and it
                           # bounds the base: nothing older is joined across it
PEAK_FLAT_BAND = 1.15      # max ratio between the base's tops; a bigger gap means price
                           # advanced between the legs, so they belong to different bases
OLDER_LEG_MIN_RATIO = 0.9  # an older leg joins the base when at least this deep as the next
MAX_VIOLATIONS = 1         # older legs shallower than that, joined anyway; the next one ends
                           # the base. The books' halving is a rule of thumb (HANDOFF §6.94).
FINAL_TIGHT_PCT = 12.0     # max depth of the last contraction
UNIFORM_TIGHT_PCT = 6.5    # …or at/below this absolute size: in a flat, quiet stretch every
                           # leg is small, so the last can't shrink to 0.8× the first, yet it is
                           # tight
NEAR_HIGH_PCT = 25.0       # max % below the 52-week high
LOOKBACK_BARS = 325        # ~65 weeks of trading days
RMV_TIGHT_MAX = 30.0       # below the pivot the base passes only if RMV (0-100) is this low.
                           # At/above the pivot a breakout is a burst of movement, so RMV stops
                           # rejecting. The below-pivot rejection MUST stay: without it, jumpy
                           # low-quality stocks pass.
MIN_LEG_BARS = 2           # a trough at least this many bars after its peak. A 2-day shakeout
                           # can be the real final contraction of a quiet stock.
MIN_BASE_WEEKS = 2.0       # shorter than this isn't a base
MAX_LEG_AGE_WEEKS = 13.0   # newest pullback older than this = stale base
DEAD_TAPE_BARS = 42        # window (~60 calendar days) for the barely-moving check …
DEAD_TAPE_MEDIAN_TR = 0.010  # … median daily true range below 1% = a stock that barely moves
                           # (held near a buyout price, or nearly inactive): no setup.
BUY_ZONE_PCT = 0.10        # tier A allows price up to this far above the base top. A tier
                           # tolerance, never an entry bound (the buy zone is doctrine's).
NEAR_PIVOT_BAND = 0.10     # …and no more than this far below it
EXPAND_RATIO = 1.10        # a pullback is wider only when deeper than this × the one before
VOLUME_BEFORE_BARS = 20    # a leg's volume is compared with this many bars before its peak

# The final tight area (swings.tight_area). The books put the last contraction at a few
# percent; 8% over up to two weeks is the loose end of the published readers' range.
AREA_RANGE_PCT = 0.08
AREA_MIN_BARS = 3
AREA_TIGHT_BARS = 5
AREA_MAX_BARS = 10

# A V-shaped recovery, fixed in HANDOFF_HISTORY §6.109 before it was measured: the base's
# deepest leg at least V_MIN_DEPTH_PCT, then a climb back to within V_REACH_PCT of the top
# with no pullback of THR_MIN_TIGHT lasting MIN_LEG_BARS, in at most V_SPEED × the bars of
# the fall.
V_MIN_DEPTH_PCT = 15.0
V_REACH_PCT = 0.03
V_SPEED = 0.5


# Kept under their old names: tests and older callers import them from here.
_zigzag_pivots = swings.zigzag_pivots
_zigzag_pivots_ref = swings.zigzag_pivots_ref


def _empty(reason: str) -> Dict[str, any]:
    return {
        'is_vcp': False, 'vcp_quality': 0.0, 'contractions': [], 'contraction_count': 0,
        'contraction_quality': 0.0, 'volume_quality': 0.0, 'base_length_weeks': 0.0,
        'rmv': 100.0, 'pattern_details': reason,
        'tier': 'C', 'zz_threshold': None, 'median_tr_pct': None,
        'pivot_price': None, 'first_leg_pct': None, 'final_leg_pct': None, 'violations': 0,
        **_no_reads(),
    }


def _no_reads() -> dict:
    return {'area_high': None, 'final_area': 'not_formed', 'area_bars': 0,
            'breakout_bar_date': None, 'swings_disagree': False, 'zz_legs': [],
            'recoveries': [], 'v_shape': None}


def _walk_legs(high, low, close) -> List[tuple]:
    """Every walk leg in the window, oldest first: ``(peak_i, trough_i, depth_pct,
    peak_price)``."""
    out = []
    for s, e in swings.segments(high, close):
        if e - s <= MIN_LEG_BARS:
            continue
        a = swings.walk_start(high, low, s, e, THR_MIN_TIGHT)
        for p, t in swings.extremes_walk(high, low, a, e, min_leg_bars=MIN_LEG_BARS,
                                         min_depth=THR_MIN_TIGHT):
            out.append((p, t, (high[p] - low[t]) / high[p] * 100.0, float(high[p])))
    return out


def _select_base(legs: List[tuple], idx, max_contractions: int):
    """The current base from ``legs``: ``(selected legs, violations)``.

    Anchor on the newest leg and walk backward. An older leg joins while no leg deeper
    than ``MAX_DEPTH_PCT`` lies between it and the base, the base's tops stay within
    ``PEAK_FLAT_BAND``, and it is at least ``OLDER_LEG_MIN_RATIO`` as deep as the leg after
    it, with ``MAX_VIOLATIONS`` exceptions."""
    cutoff = idx[-1] - pd.Timedelta(weeks=MAX_BASE_WEEKS)
    recent = [g for g in legs if g[2] <= MAX_DEPTH_PCT and idx[g[0]] >= cutoff]
    deep = [g for g in legs if g[2] > MAX_DEPTH_PCT]
    if not recent:
        return [], 0
    sel, violations = [recent[-1]], 0
    tops = [recent[-1][3]]
    for g in reversed(recent[:-1]):
        if len(sel) >= max_contractions:
            break
        if any(g[0] < d[0] < sel[0][0] for d in deep):
            break
        if max(tops + [g[3]]) / min(tops + [g[3]]) > PEAK_FLAT_BAND:
            break
        if g[2] < sel[0][2] * OLDER_LEG_MIN_RATIO:
            if violations >= MAX_VIOLATIONS:
                break
            violations += 1
        sel.insert(0, g)
        tops.append(g[3])
    return sel, violations


def _v_shape(high, low, sel, pivot: float) -> Optional[dict]:
    """The V read on the base's deepest leg: ``{depth_pct, fall_bars, rise_bars, paused,
    flag}``. ``rise_bars`` and ``paused`` are None when price never got back to within
    ``V_REACH_PCT`` of the top. None without legs."""
    if not sel:
        return None
    p, t, depth = max(sel, key=lambda g: g[2])[:3]
    fall = t - p
    reach = np.flatnonzero(high[t + 1:] >= pivot * (1.0 - V_REACH_PCT))
    if not len(reach):
        return {'depth_pct': round(depth, 1), 'fall_bars': int(fall), 'rise_bars': None,
                'paused': None, 'flag': False}
    j = t + 1 + int(reach[0])
    paused = False
    if j - t >= MIN_LEG_BARS:
        run = np.maximum.accumulate(high[t:j + 1])
        dd = 1.0 - low[t + MIN_LEG_BARS:j + 1] / run[:j + 1 - t - MIN_LEG_BARS]
        paused = bool(np.any(dd >= THR_MIN_TIGHT))
    rise = j - t
    flag = bool(depth >= V_MIN_DEPTH_PCT and not paused and rise <= V_SPEED * fall)
    return {'depth_pct': round(depth, 1), 'fall_bars': int(fall), 'rise_bars': int(rise),
            'paused': paused, 'flag': flag}


def _leg_volume_ratio(vol, p: int, t: int) -> float:
    """Mean volume over the leg over the mean of the ``VOLUME_BEFORE_BARS`` bars before its
    peak; 1.0 when either is unknown."""
    before = vol[max(0, p - VOLUME_BEFORE_BARS):p]
    during = vol[p:t + 1]
    if not np.isfinite(before).any() or not np.isfinite(during).any():
        return 1.0
    b, d = np.nanmean(before), np.nanmean(during)
    return float(d / b) if b > 0 else 1.0


def detect_vcp(price_data: pd.DataFrame, current_price: float, phase_info: Dict,
               thr: Optional[float] = None, min_contractions: int = MIN_CONTRACTIONS,
               max_contractions: int = MAX_CONTRACTIONS) -> Dict[str, any]:
    """Detect a Volatility Contraction Pattern. Replaces ``minervini_screener``'s
    ``detect_vcp_pattern``, with the same inputs and return keys (``is_vcp``,
    ``vcp_quality``, ``contractions`` oldest first, each with number/peak_date/trough_date/
    peak_price/trough_price/drawdown_pct/volume_ratio/duration_days, ``contraction_count``,
    ``contraction_quality``, ``volume_quality``, ``base_length_weeks``, ``rmv``,
    ``pattern_details``, ``tier``, ``zz_threshold``, ``median_tr_pct``) plus:

    - ``pivot_price``: the base's highest high, the level the tiers are measured from;
    - ``area_high``, ``final_area`` ('tight' | 'loose' | 'not_formed'), ``area_bars``: the
      final tight area before any breakout. The books' pivot is its high;
    - ``breakout_bar_date``: the first close above ``pivot_price`` after the last leg, or None;
    - ``zz_legs``: the ZigZag reader's legs over the base, each {peak_date, trough_date,
      peak_price, trough_price, drawdown_pct};
    - ``swings_disagree``: True when the ZigZag sees a pullback, before the newest two,
      more than ``EXPAND_RATIO`` × the one before it (the walk merges such a leg);
    - ``recoveries``: ``[(fall_bars, rise_bars)]`` for each ZigZag low between two highs;
    - ``first_leg_pct``, ``final_leg_pct``, ``violations`` and ``v_shape`` (``_v_shape``).

    ``thr`` is the ZigZag reader's reversal size (default ``THR_FIXED_TIGHT``). An
    explicit value also skips the barely-moving check, so made-up test frames with
    High = Low = Close stay usable. No frame or under 40 bars returns an empty tier-C
    result."""
    if price_data is None or len(price_data) < 40:
        return _empty('Insufficient data')

    base = price_data.tail(min(len(price_data), LOOKBACK_BARS))

    _tr = true_range_pct(base).tail(DEAD_TAPE_BARS).to_numpy()
    med_tr = float(np.nanmedian(_tr)) if np.isfinite(_tr).any() else float('nan')
    med_tr_pct = round(med_tr * 100.0, 2) if np.isfinite(med_tr) else None

    if thr is None:
        if np.isfinite(med_tr) and med_tr < DEAD_TAPE_MEDIAN_TR:
            return {**_empty(f'Barely moving: median daily range {med_tr * 100:.2f}% '
                             f'over the last {DEAD_TAPE_BARS} trading days'),
                    'median_tr_pct': med_tr_pct}
        zz_thr = THR_FIXED_TIGHT
    else:
        zz_thr = float(thr)

    high = base['High'].to_numpy(dtype=float)
    low = base['Low'].to_numpy(dtype=float)
    close = base['Close'].to_numpy(dtype=float)
    vol = (base['Volume'].to_numpy(dtype=float) if 'Volume' in base.columns
           else np.full(len(base), np.nan))
    idx = base.index

    week_52_high = phase_info.get('week_52_high') or float(price_data['High'].tail(252).max())
    rmv_series = relative_measured_volatility(base).dropna()
    rmv_now = float(rmv_series.iloc[-1]) if len(rmv_series) else 100.0

    sel, violations = _select_base(_walk_legs(high, low, close), idx, max_contractions)

    contractions: List[dict] = []
    for k, (p, t, d, top) in enumerate(sel):
        try:
            dur = int((idx[t] - idx[p]).days)
        except (AttributeError, TypeError):
            dur = 0
        contractions.append({
            'number': k + 1, 'peak_date': idx[p], 'trough_date': idx[t],
            'peak_price': top, 'trough_price': float(low[t]),
            'drawdown_pct': round(d, 2),
            'volume_ratio': round(_leg_volume_ratio(vol, p, t), 2), 'duration_days': dur,
        })

    cnt = len(sel)
    depths = [c['drawdown_pct'] for c in contractions]

    # Tightening quality (0-100): a slope on log-depths, so one leg that grows
    # (25→12→14→6) doesn't sink an obviously tightening base, blended with the share of
    # legs that shrink and the first→last shrink.
    if cnt >= 3:
        x = np.arange(cnt, dtype=float)
        y = np.log(np.clip(np.asarray(depths, dtype=float), 1e-6, None))
        slope = float(np.polyfit(x, y, 1)[0])
        trend = 1.0 if slope < 0 else 0.0
        monotone = sum(1 for i in range(1, cnt) if depths[i] < depths[i - 1]) / (cnt - 1)
        shrink = max(0.0, 1.0 - depths[-1] / depths[0]) if depths[0] > 0 else 0.0
        contraction_quality = 100.0 * (0.4 * trend + 0.3 * monotone
                                       + 0.3 * min(1.0, shrink / 0.5))
    elif cnt == 2:
        contraction_quality = 100.0 if depths[1] < depths[0] else 0.0
    else:
        contraction_quality = 0.0

    volume_quality = (sum(1 for c in contractions if c['volume_ratio'] < 1.0) / cnt * 100.0
                      if cnt else 0.0)

    last_date = idx[-1]
    base_length_weeks = ((last_date - contractions[0]['peak_date']).days / 7.0) if cnt else 0.0
    leg_age_weeks = ((last_date - contractions[-1]['trough_date']).days / 7.0) if cnt else np.inf

    dist_high = ((week_52_high - current_price) / week_52_high * 100.0) if week_52_high > 0 else 100.0
    near_high = dist_high <= NEAR_HIGH_PCT

    pivot_price = max((c['peak_price'] for c in contractions), default=None)
    below_pivot = bool(pivot_price and current_price < pivot_price)
    in_buy_zone = bool(pivot_price
                       and pivot_price * (1 - NEAR_PIVOT_BAND)
                       <= current_price
                       <= pivot_price * (1 + BUY_ZONE_PCT))
    rmv_ok = rmv_now <= RMV_TIGHT_MAX or not below_pivot

    fresh = leg_age_weeks <= MAX_LEG_AGE_WEEKS
    long_enough = base_length_weeks >= MIN_BASE_WEEKS
    final_tight = bool(depths and depths[-1] <= FINAL_TIGHT_PCT
                       and (depths[-1] <= depths[0] * 0.8
                            or depths[-1] <= UNIFORM_TIGHT_PCT))

    structure_ok = bool(
        min_contractions <= cnt <= max_contractions
        and contraction_quality >= TIGHTEN_MIN_PCT
        and final_tight
        and near_high
        and long_enough
        and fresh
    )
    is_vcp = structure_ok and rmv_ok

    # Tier C MUST be only a safe exclusion, one that can't be a setup.
    if cnt == 0 or not fresh:
        tier = 'C'
    elif is_vcp and in_buy_zone:
        tier = 'A'
    else:
        tier = 'B'

    # Quality 0-100. The app's help text for the score column describes these weights.
    q = 0.0
    if min_contractions <= cnt <= max_contractions:
        q += min(20.0, cnt / max_contractions * 20.0)
    q += contraction_quality / 100.0 * 30.0
    q += volume_quality / 100.0 * 20.0
    if 3 <= base_length_weeks <= 65:
        q += 10.0
    if near_high:
        q += max(0.0, 20.0 - dist_high / NEAR_HIGH_PCT * 20.0)

    if cnt >= min_contractions:
        sizes = ' → '.join(f"{d:.1f}%" for d in depths[-4:])
        pattern_details = f"{cnt} contractions: {sizes}"
    elif cnt == 0:
        pattern_details = "No pullbacks found"
    else:
        pattern_details = f"Only {cnt} contraction(s) detected (need {min_contractions}+)"
    if cnt and not fresh:
        pattern_details += f"; newest pullback {leg_age_weeks:.0f} weeks old"

    return {
        'is_vcp': is_vcp,
        'vcp_quality': round(q, 1),
        'contractions': contractions,
        'contraction_count': cnt,
        'contraction_quality': round(contraction_quality, 1),
        'volume_quality': round(volume_quality, 1),
        'base_length_weeks': round(base_length_weeks, 1),
        'rmv': round(rmv_now, 1),
        'pattern_details': pattern_details,
        'tier': tier,
        'zz_threshold': round(zz_thr, 4),
        'median_tr_pct': med_tr_pct,
        'pivot_price': round(pivot_price, 2) if pivot_price else None,
        'first_leg_pct': depths[0] if depths else None,
        'final_leg_pct': depths[-1] if depths else None,
        'violations': violations,
        **_reads(high, low, close, idx, sel, pivot_price, zz_thr),
    }


def _reads(high, low, close, idx, sel, pivot_price, zz_thr) -> dict:
    """The reads that describe the base without deciding its tier: the final area, the
    breakout bar, the ZigZag reader and the V. Empty values without legs."""
    out = _no_reads()
    if not sel:
        return out
    n = len(high)
    start, last_trough = sel[0][0], sel[-1][1]
    above = np.flatnonzero(close[last_trough + 1:] > pivot_price)
    end = last_trough + 1 + int(above[0]) if len(above) else n
    if len(above):
        out['breakout_bar_date'] = idx[end]

    state, bars, area_high = swings.tight_area(
        high, low, close, end, last_trough, max_range=AREA_RANGE_PCT,
        min_bars=AREA_MIN_BARS, tight_bars=AREA_TIGHT_BARS, max_bars=AREA_MAX_BARS)
    out.update(final_area=state, area_bars=bars,
               area_high=round(area_high, 2) if area_high is not None else None)

    piv = [(i + start, p, k) for i, p, k in swings.zigzag_pivots(high[start:end],
                                                                 low[start:end], zz_thr)]
    zz = swings.zigzag_legs(piv)
    zz_depths = [(high[p] - low[t]) / high[p] * 100.0 for p, t in zz]
    out['zz_legs'] = [{'peak_date': idx[p], 'trough_date': idx[t],
                       'peak_price': float(high[p]), 'trough_price': float(low[t]),
                       'drawdown_pct': round(d, 2)} for (p, t), d in zip(zz, zz_depths)]
    out['swings_disagree'] = any(zz_depths[k] > EXPAND_RATIO * zz_depths[k - 1]
                                 for k in range(1, len(zz_depths) - 1))
    out['recoveries'] = [(int(b[0] - a[0]), int(c[0] - b[0]))
                         for a, b, c in zip(piv, piv[1:], piv[2:])
                         if a[2] == 'H' and b[2] == 'L' and c[2] == 'H']
    out['v_shape'] = _v_shape(high, low, sel, pivot_price)
    return out
