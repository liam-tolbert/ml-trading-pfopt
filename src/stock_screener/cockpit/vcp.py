"""The cockpit's own VCP / contraction detector. It replaces, with the same inputs and
outputs, the one in the copy of ``minervini_screener`` kept in this repo.

That copy's ``detect_vcp_pattern`` finds zero contractions for most stocks in a broad
list: the points it measures a base from are set too tightly for the window it uses to
find swing highs and lows. This detector uses a **ZigZag** instead (it marks swing highs
and lows, ignoring moves smaller than a threshold), with the threshold scaled to the
stock's own volatility. Its turning points (``_zigzag_pivots``) strictly alternate
high/low, so every contraction is well-formed (depth ≥ 0) and shallow tight bases are
found. It returns the same dict schema, so ``scan.py`` uses it without changes.

Design goal, checked against the 200 hand-labeled charts in ``tests/vcp_labels.py``: a
first-pass filter that lets borderline cases through rather than miss real ones. It MUST
NOT hide a setup that could still be bought. Misses are unacceptable; false alarms only
cost a glance. So instead of one yes/no it assigns a review **tier**:

  A — review: a valid tightening base with price below / at / within ~5% above the pivot.
  B — watch: a plausible base still forming, or a valid pattern already extended past the
      buy zone. Never hidden.
  C — skipped, with the reason recorded: only *safe* exclusions (a stock that has barely
      moved, no pullbacks at any threshold, a base whose newest pullback is too old) that
      cannot be a usable setup.

Detection runs at several ZigZag thresholds (long-history, recent-window, and extra-tight
variants) and keeps the best result. A VCP by definition ends *quieter* than the stock's
own history, so a single threshold set from the full history misses the small final swings.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import relative_measured_volatility, true_range_pct

# --- Settings (tuned on the full_us scan's chain of filters + the 200 labeled charts) --- #
ZZ_THRESHOLD = 0.04        # fallback ZigZag reversal size (used when the volatility-scaled
                           # estimate can't be computed, and as the reference point for ADAPT_K).
ATR_PERIOD = 10            # smoothing for the true-range% used by the scaled threshold
ADAPT_K = 2.75             # scaled ZigZag threshold = ADAPT_K × median(true-range%) over the
                           # base, clipped to [THR_MIN, THR_MAX]. K is set so a typical stock
                           # (~1.5%/day true range) lands near ZZ_THRESHOLD.
THR_MIN = 0.03             # floor: tiny wiggles of very quiet large companies aren't swings
THR_MAX = 0.10             # cap: don't merge a jumpy small company's swings into one leg
RECENT_WINDOW_BARS = 42    # ~2 months: "how quiet is the stock NOW" (the contraction itself)
RECENT_THR_SHRINK = 0.7    # extra-tight candidate = 0.7 × the recent-window threshold …
THR_MIN_TIGHT = 0.02       # … floored here (deliberately below THR_MIN, to see tight endings)
THR_FIXED_TIGHT = 0.035    # tight candidate always tried: the breakout's own big moves can
                           # inflate the recent window, so volatility-scaled thresholds alone
                           # miss short, narrow pauses. 3.5% still sees quiet bases' 4-6% legs.
TIGHTEN_MIN_PCT = 60.0     # is_vcp minimum on the 0-100 tightening-quality score (below)
MIN_CONTRACTIONS = 2
MAX_CONTRACTIONS = 6
MAX_BASE_WEEKS = 65        # a base older than this isn't the current setup
MAX_DEPTH_PCT = 35.0       # a leg (one peak-to-trough pullback) deeper than this is a decline,
                           # not a base contraction
PEAK_FLAT_BAND = 1.15      # max ratio between base peaks (a flat-ish top); a bigger gap
                           # means price advanced/broke out between the legs -> a different base
FINAL_TIGHT_PCT = 12.0     # max depth of the last contraction
UNIFORM_TIGHT_PCT = 6.5    # …or at/below this absolute size. In a flat, quiet stretch ALL legs
                           # are already small, so the final one may not shrink to 0.8× the
                           # first, yet it is tight.
NEAR_HIGH_PCT = 25.0       # max % below the 52-week high
LOOKBACK_BARS = 325        # ~65 weeks of trading days
RMV_TIGHT_MAX = 30.0       # RMV (relative measured volatility, 0-100) check: below the pivot the
                           # base passes only if quiet (near the bottom of its own volatility
                           # range). At/above the pivot a breakout is a burst of movement, so RMV
                           # stops rejecting and structure alone decides. The below-pivot
                           # rejection MUST stay: without it, jumpy low-quality stocks pass.
# Sanity rules (HANDOFF §6):
MIN_LEG_BARS = 2           # same-day / 1-bar "legs" are not real pullbacks. A 2-day shakeout
                           # (a quick dip) can be the real final contraction of a quiet stock
                           # rising in steps, so the floor sits at 2.
MIN_BASE_WEEKS = 2.0       # shorter than this isn't a base. Kept low (2.0) because base length is
                           # measured over the SELECTED legs only, which under-reads the true base.
MAX_LEG_AGE_WEEKS = 13.0   # newest pullback older than this = stale base
                           # (e.g. TWO and MNST in tests/vcp_labels.py)
DEAD_TAPE_BARS = 42        # window (~60 calendar days) for the barely-moving check …
DEAD_TAPE_MEDIAN_TR = 0.010  # … median daily true-range% below 1% = a stock that barely moves
                           # (e.g. held near a buyout price, or nearly inactive): no swings to
                           # contract, so it can't be a setup.
BUY_ZONE_PCT = 0.10        # tier A allows price up to this far above the DETECTED pivot. That
                           # pivot (top of the selected legs) usually sits below the real pivot
                           # you would buy at, so +10% here ≈ "≤~5-8% past the real pivot"; a
                           # stock further above stays out.
NEAR_PIVOT_BAND = 0.10     # …and no more than this far BELOW it: price falling away from the
                           # base top is a failing pattern, not a tightening base.

_TIER_RANK = {'A': 0, 'B': 1, 'C': 2}


def _zigzag_pivots(high: np.ndarray, low: np.ndarray, thr: float) -> List[tuple]:
    """Fast percentage ZigZag: :func:`_zigzag_pivots_ref` fed plain-Python floats.
    Indexing an ndarray one element at a time creates a new np.float64 on every access, and
    the loop runs ~325 iterations × up to 4 thresholds × every VCP candidate.
    ``test_zigzag_fast_parity`` checks it matches the reference version on the 200
    labeled charts."""
    return _zigzag_pivots_ref(np.asarray(high).tolist(), np.asarray(low).tolist(), thr)


def _zigzag_pivots_ref(high, low, thr: float) -> List[tuple]:
    """Percentage ZigZag -> strictly-alternating (index, price, kind) pivots ('H'/'L').

    A swing high is confirmed once price falls ``thr`` below the running high; a swing low
    once it rises ``thr`` above the running low. The loop only ever looks for the opposite
    kind next, so every 'H' is followed by an 'L', and every down-leg is a real
    peak->trough with the trough below the peak. The final running extreme is appended as
    a not-yet-confirmed pivot (the latest swing).
    The reference implementation; accepts any indexable sequence (ndarray or list). Empty
    for fewer than 2 bars.
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
            if low[i] <= hi_p * (1 - thr):     # reversal down -> confirm the swing high
                piv.append((hi_i, hi_p, 'H')); trend = -1; lo_i, lo_p = i, low[i]
        else:
            if high[i] >= lo_p * (1 + thr):    # reversal up -> confirm the swing low
                piv.append((lo_i, lo_p, 'L')); trend = 1; hi_i, hi_p = i, high[i]
    if trend > 0:
        piv.append((hi_i, hi_p, 'H'))
    elif trend < 0:
        piv.append((lo_i, lo_p, 'L'))
    return piv


def _adaptive_threshold(base: pd.DataFrame) -> float:
    """Scale the ZigZag reversal size to the stock's *own* volatility.

    A jumpy small company needs a wider swing filter, or one pullback splits into three.
    A very quiet large company needs a tighter one, or its tight final contraction is
    missed. The scale is the median of the ``ATR_PERIOD``-bar mean true-range% over the
    base, the same percentage volatility (comparable across share prices) RMV is built on.
    Returns ``ADAPT_K`` × that, clipped to
    [THR_MIN, THR_MAX]; ``ZZ_THRESHOLD`` when the median is missing or non-positive.
    """
    trp = true_range_pct(base).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    med = float(np.nanmedian(trp.to_numpy())) if len(trp) else np.nan
    if not np.isfinite(med) or med <= 0:
        return ZZ_THRESHOLD
    return float(np.clip(ADAPT_K * med, THR_MIN, THR_MAX))


def _empty(reason: str) -> Dict[str, any]:
    return {
        'is_vcp': False, 'vcp_quality': 0.0, 'contractions': [], 'contraction_count': 0,
        'contraction_quality': 0.0, 'volume_quality': 0.0, 'base_length_weeks': 0.0,
        'rmv': 100.0, 'pattern_details': reason,
        'tier': 'C', 'zz_threshold': None, 'median_tr_pct': None,
    }


def _detect_at(base: pd.DataFrame, current_price: float,
               phase_info: Dict, thr: float, min_contractions: int,
               max_contractions: int, *, rmv_now: float,
               week_52_high: float) -> Dict[str, any]:
    """One detection pass at a fixed ZigZag threshold; returns the full result dict.

    ``rmv_now`` and ``week_52_high`` don't depend on the threshold, so :func:`detect_vcp`
    computes them once for every pass."""
    high = base['High'].to_numpy(dtype=float)
    low = base['Low'].to_numpy(dtype=float)
    vol = (base['Volume'].to_numpy(dtype=float) if 'Volume' in base.columns
           else np.full(len(base), np.nan))
    idx = base.index

    piv = _zigzag_pivots(high, low, thr)

    # Down-legs = consecutive High -> Low pivots (well-formed, depth >= 0 by construction).
    contractions: List[dict] = []
    for a, b in zip(piv, piv[1:]):
        if a[2] != 'H' or b[2] != 'L':
            continue
        pi, pp, _ = a
        ti, tp, _ = b
        if pp <= 0:
            continue
        before = vol[max(0, pi - 20):pi]
        during = vol[pi:ti + 1]
        avg_before = np.nanmean(before) if len(before) else np.nan
        avg_during = np.nanmean(during) if len(during) else np.nan
        vratio = (float(avg_during / avg_before)
                  if np.isfinite(avg_before) and np.isfinite(avg_during) and avg_before > 0
                  else 1.0)
        try:
            dur = int((idx[ti] - idx[pi]).days)
        except (AttributeError, TypeError):
            dur = 0
        contractions.append({
            'peak_index': pi, 'trough_index': ti,
            'peak_date': idx[pi], 'trough_date': idx[ti],
            'peak_price': float(pp), 'trough_price': float(tp),
            'drawdown_pct': round((pp - tp) / pp * 100.0, 2),
            'volume_ratio': round(vratio, 2), 'duration_days': dur,
        })

    # Select the CURRENT base. A VCP is a single consolidation under a flat-ish top that
    # tightens toward the pivot. So anchor on the most recent contraction and walk BACKWARD,
    # adding an older leg only while (a) the peaks stay within PEAK_FLAT_BAND, since a bigger
    # gap means price broke out between the legs, a *different* base; and (b) the older leg
    # is at least 0.9× as deep as the one after it (widest first). Legs spanning fewer than
    # MIN_LEG_BARS bars are dropped first: a same-bar or next-bar dip is not a real pullback
    # and makes a fake single-leg base.
    last_date = idx[-1]
    cutoff = last_date - pd.Timedelta(weeks=MAX_BASE_WEEKS)
    recent = [c for c in contractions
              if c['drawdown_pct'] <= MAX_DEPTH_PCT and c['peak_date'] >= cutoff
              and (c['trough_index'] - c['peak_index']) >= MIN_LEG_BARS]
    # A leg REMOVED for depth (> MAX_DEPTH_PCT) that sits between two kept legs is a base
    # BOUNDARY, not a leg to skip over. PEAK_FLAT_BAND only rejects an ADVANCE between legs,
    # so without this a crash-and-recover is stitched into one base (tier A over a > 35%
    # drop). A short (< MIN_LEG_BARS) leg MUST NOT bound the base: it is a normal quick
    # dip, and bounding on those splits real bases into pieces (5 fewer tier-A YES on the
    # 200 labeled charts). That trades misses for fewer false alarms, which this tool
    # doesn't want.
    disq = [c for c in contractions if c['drawdown_pct'] > MAX_DEPTH_PCT]
    sel: List[dict] = []
    if recent:
        sel = [recent[-1]]
        for c in reversed(recent[:-1]):
            if len(sel) >= max_contractions:
                break
            if any(c['peak_index'] < b['peak_index'] < sel[0]['peak_index'] for b in disq):
                break                                              # a removed leg intervenes -> boundary
            peaks = [x['peak_price'] for x in sel] + [c['peak_price']]
            if max(peaks) / min(peaks) > PEAK_FLAT_BAND:            # top not flat -> different base
                break
            if c['drawdown_pct'] >= sel[0]['drawdown_pct'] * 0.9:   # older leg >= 0.9× as deep
                sel.insert(0, c)
            else:
                break
    for k, c in enumerate(sel):
        c['number'] = k + 1
        c.pop('peak_index', None)
        c.pop('trough_index', None)

    n = len(sel)
    depths = [c['drawdown_pct'] for c in sel]

    # Tightening quality (0-100): is each pullback shrinking? A slope on log-depths, so one
    # leg that grows (25→12→14→6) doesn't sink an obviously-tightening base. The score
    # blends the log-depth downtrend, the strictly-shrinking fraction and the first→last
    # shrink.
    if n >= 3:
        x = np.arange(n, dtype=float)
        y = np.log(np.clip(np.asarray(depths, dtype=float), 1e-6, None))
        slope = float(np.polyfit(x, y, 1)[0])                       # < 0 => tightening
        trend = 1.0 if slope < 0 else 0.0
        monotone = sum(1 for i in range(1, n) if depths[i] < depths[i - 1]) / (n - 1)
        shrink = max(0.0, 1.0 - depths[-1] / depths[0]) if depths[0] > 0 else 0.0
        contraction_quality = 100.0 * (0.4 * trend + 0.3 * monotone
                                       + 0.3 * min(1.0, shrink / 0.5))
    elif n == 2:
        contraction_quality = 100.0 if depths[1] < depths[0] else 0.0
    else:
        contraction_quality = 0.0

    volume_quality = (sum(1 for c in sel if c['volume_ratio'] < 1.0) / n * 100.0) if n else 0.0

    base_length_weeks = ((last_date - sel[0]['peak_date']).days / 7.0) if n else 0.0
    leg_age_weeks = ((last_date - sel[-1]['trough_date']).days / 7.0) if n else np.inf

    dist_high = ((week_52_high - current_price) / week_52_high * 100.0) if week_52_high > 0 else 100.0
    near_high = dist_high <= NEAR_HIGH_PCT

    pivot_price = max((c['peak_price'] for c in sel), default=None)
    below_pivot = bool(pivot_price and current_price < pivot_price)
    in_buy_zone = bool(pivot_price
                       and pivot_price * (1 - NEAR_PIVOT_BAND)
                       <= current_price
                       <= pivot_price * (1 + BUY_ZONE_PCT))
    # RMV is scaled 0-100 between its lowest and highest over its lookback, so rmv_now <=
    # RMV_TIGHT_MAX means volatility sits at the tight end of the base's own range. It
    # rejects only below the pivot (see RMV_TIGHT_MAX).
    vol_confirms = rmv_now <= RMV_TIGHT_MAX
    rmv_ok = vol_confirms or not below_pivot

    fresh = leg_age_weeks <= MAX_LEG_AGE_WEEKS
    long_enough = base_length_weeks >= MIN_BASE_WEEKS
    # Tight ending: clearly tighter than the first leg, or small in absolute terms. A flat,
    # quiet stretch (4.4% → 3.8%) can't shrink 20% further but IS tight.
    final_tight = bool(depths and depths[-1] <= FINAL_TIGHT_PCT
                       and (depths[-1] <= depths[0] * 0.8
                            or depths[-1] <= UNIFORM_TIGHT_PCT))

    # A VCP: min_contractions to max_contractions progressively-tighter pullbacks (each
    # spanning >= MIN_LEG_BARS bars), a tight final leg, a base at least MIN_BASE_WEEKS
    # long with its newest leg fresh, price near its high, and, below the pivot, volatility
    # confirming.
    structure_ok = bool(
        min_contractions <= n <= max_contractions
        and contraction_quality >= TIGHTEN_MIN_PCT
        and final_tight
        and near_high
        and long_enough
        and fresh
    )
    is_vcp = structure_ok and rmv_ok

    # ---- Review tier (C MUST be only a safe exclusion, one that can't be a setup) ---- #
    if n == 0:
        tier = 'C'          # no pullbacks found: nothing resembling a base
    elif not fresh:
        tier = 'C'          # stale base: newest pullback older than MAX_LEG_AGE_WEEKS
    elif is_vcp and in_buy_zone:
        tier = 'A'          # valid tightening base in/near the buy zone
    elif is_vcp:
        tier = 'B'          # valid pattern but extended past / fallen away from the pivot
    else:
        tier = 'B'          # base still forming (few legs / far from the 52-wk high /
        #                     too short / final leg not tight / not tightening / too volatile)

    # Quality 0-100. The weighting MUST match the detector in the repo's copy of
    # ``minervini_screener``: the app help text describes it.
    q = 0.0
    if min_contractions <= n <= max_contractions:
        q += min(20.0, n / max_contractions * 20.0)
    q += contraction_quality / 100.0 * 30.0
    q += volume_quality / 100.0 * 20.0
    if 3 <= base_length_weeks <= 65:
        q += 10.0
    if near_high:
        q += max(0.0, 20.0 - dist_high / NEAR_HIGH_PCT * 20.0)

    if n >= min_contractions:
        sizes = ' → '.join(f"{d:.1f}%" for d in depths[-4:])
        pattern_details = f"{n} contractions: {sizes}"
    else:
        pattern_details = f"Only {n} contraction(s) detected (need {min_contractions}+)"

    return {
        'is_vcp': is_vcp,
        'vcp_quality': round(q, 1),
        'contractions': sel,
        'contraction_count': n,
        'contraction_quality': round(contraction_quality, 1),
        'volume_quality': round(volume_quality, 1),
        'base_length_weeks': round(base_length_weeks, 1),
        'rmv': round(rmv_now, 1),
        'pattern_details': pattern_details,
        'tier': tier,
        'zz_threshold': round(float(thr), 4),
        # pivot_price is not exported, but in_buy_zone, below_pivot and the A-vs-B tier split
        # use it. Un-comment to expose it:
        # 'pivot_price': round(pivot_price, 2) if pivot_price else None,
    }


def detect_vcp(price_data: pd.DataFrame, current_price: float, phase_info: Dict,
               thr: Optional[float] = None, min_contractions: int = MIN_CONTRACTIONS,
               max_contractions: int = MAX_CONTRACTIONS) -> Dict[str, any]:
    """Detect a Volatility Contraction Pattern. Replaces ``minervini_screener``'s
    ``detect_vcp_pattern``, with the same inputs and return schema (``is_vcp``,
    ``vcp_quality``, ``contractions`` with number/peak_date/trough_date/peak_price/trough_price/
    drawdown_pct/volume_ratio/duration_days, ``contraction_count`` …) plus the review
    ``tier`` ('A'/'B'/'C'), ``zz_threshold`` and ``median_tr_pct`` (the median daily
    true range over the last ``DEAD_TAPE_BARS``, in percent; None with too little data).
    (The pivot is computed internally for the buy-zone/extended tier split but not
    exported; see ``_detect_at``.)

    ``thr`` is the ZigZag reversal size. Leave it ``None`` (default) to run at up to four
    thresholds: long-history, recent-window (~2 months), an extra-tight 0.7× recent, and
    a fixed 3.5%. The best result wins: one with ``is_vcp`` True, else the best
    tier/quality. An explicit value uses that one threshold and skips the barely-moving
    check; tests set it to get the same pivot counts every run. No frame or under 40 bars
    returns an empty tier-C result."""
    if price_data is None or len(price_data) < 40:
        return _empty('Insufficient data')

    base = price_data.tail(min(len(price_data), LOOKBACK_BARS))

    # Returned on every path as the typical daily range. Only ``thr=None`` rejects on it.
    _tr = true_range_pct(base).tail(DEAD_TAPE_BARS).to_numpy()
    med_tr = float(np.nanmedian(_tr)) if np.isfinite(_tr).any() else float('nan')
    med_tr_pct = round(med_tr * 100.0, 2) if np.isfinite(med_tr) else None

    if thr is not None:
        # One fixed threshold. The barely-moving check is skipped so made-up test frames
        # with High = Low = Close stay usable: with no range inside each bar they
        # under-read true range.
        candidates = [float(thr)]
    else:
        # A stock that has barely moved for months has no swings to contract, so it can't
        # be a setup worth buying.
        if np.isfinite(med_tr) and med_tr < DEAD_TAPE_MEDIAN_TR:
            return {**_empty(f'Barely moving: median daily range {med_tr * 100:.2f}% '
                             f'over the last {DEAD_TAPE_BARS} trading days'),
                    'median_tr_pct': med_tr_pct}
        thr_long = _adaptive_threshold(base)
        thr_recent = _adaptive_threshold(base.tail(RECENT_WINDOW_BARS))
        thr_tight = max(RECENT_THR_SHRINK * thr_recent, THR_MIN_TIGHT)
        candidates = sorted({round(t, 4) for t in (thr_long, thr_recent, thr_tight,
                                                   THR_FIXED_TIGHT)},
                            reverse=True)

    # Threshold-independent, so computed once for every pass.
    week_52_high = phase_info.get('week_52_high') or float(price_data['High'].tail(252).max())
    rmv_series = relative_measured_volatility(base).dropna()
    rmv_now = float(rmv_series.iloc[-1]) if len(rmv_series) else 100.0

    results = [_detect_at(base, current_price, phase_info, t,
                          min_contractions, max_contractions, rmv_now=rmv_now,
                          week_52_high=week_52_high)
               for t in candidates]
    best = min(results, key=lambda r: (0 if r['is_vcp'] else 1,
                                       _TIER_RANK[r['tier']], -r['vcp_quality']))
    return {**best, 'median_tr_pct': med_tr_pct}
