"""Cockpit-local VCP / contraction detector (a drop-in for the vendored one).

The vendored ``minervini_screener`` ``detect_vcp_pattern`` collapses on a broad universe:
on ~3.6k US common stocks it reports **zero contractions for 72% of names** and starves
exactly the strong uptrends you care about (its base-start anchors to the *most recent*
20%-recovery low, giving a ~20-bar base with no room for its 10-day centered swing window
→ 0 swings). It also drops the most recent contraction (``peaks[:-1]``) and mis-matches
peaks to troughs (producing negative "contractions").

This module re-implements detection with a **volatility-adaptive ZigZag** that yields a
strictly-alternating high/low pivot sequence — so contractions are always well-formed
(depth ≥ 0), shallow tight bases are found, and the most recent contraction is kept. The
vendored detector is edit-restricted (PROVENANCE), so this lives in the cockpit and
returns the SAME dict schema, making it a drop-in for ``scan.py`` (chart hover,
``detect_breakout`` and the results table are unchanged).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import relative_measured_volatility

# --- Tunables (calibrated against the live full_us funnel) ------------------ #
ZZ_THRESHOLD = 0.04        # fallback ZigZag reversal size (used when the adaptive estimate
                           # can't be computed, and as the calibration anchor for ADAPT_K).
ATR_PERIOD = 10            # smoothing for the true-range% used by the adaptive threshold
ADAPT_K = 2.75             # adaptive ZigZag threshold = ADAPT_K × median(true-range%) over the
                           # base, clipped to [THR_MIN, THR_MAX]. K is set so a typical mid-vol
                           # name (~1.5%/day true range) lands near the old fixed 0.04.
THR_MIN = 0.03             # floor: don't chase noise on ultra-quiet mega-caps
THR_MAX = 0.10             # cap: don't blur a high-vol small-cap's swings into one leg
TIGHTEN_MIN_PCT = 60.0     # is_vcp gate on the 0-100 tightening-quality score (below)
MIN_CONTRACTIONS = 2
MAX_CONTRACTIONS = 6
MAX_BASE_WEEKS = 65        # a base older than this isn't the current setup
MAX_DEPTH_PCT = 35.0       # a leg deeper than this is a decline, not a base contraction
PEAK_FLAT_BAND = 1.15      # base peaks must sit within this ratio (a flat-ish top); a bigger
                           # gap means price advanced/broke out between the legs -> a different base
FINAL_TIGHT_PCT = 12.0     # the last contraction must be at least this tight
NEAR_HIGH_PCT = 25.0       # price must be within this % of the 52-week high
LOOKBACK_BARS = 325        # ~65 weeks of trading days
RMV_TIGHT_MAX = 25.0       # strict gate: current RMV must be <= this (bottom quartile of the
                           # base's own volatility range) — the volatility half of the VCP


def _zigzag_pivots(high: np.ndarray, low: np.ndarray, thr: float) -> List[tuple]:
    """Percentage ZigZag -> strictly-alternating (index, price, kind) pivots ('H'/'L').

    A swing high is confirmed once price falls ``thr`` below the running high; a swing low
    once it rises ``thr`` above the running low. Alternation is structural, so every 'H' is
    followed by an 'L' — every down-leg is a genuine peak->trough with the trough below the
    peak. The final running extreme is appended as a tentative pivot (the live edge).
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

    A high-vol small-cap needs a wider swing filter (or one pullback fragments into three);
    an ultra-quiet mega-cap needs a tighter one (or its tight final contraction is invisible).
    We use median true-range% over the base — the same scale-free volatility RMV is built on —
    so ``thr`` floats with the stock instead of being a magic 0.04. Clipped to [THR_MIN, THR_MAX].
    """
    high, low, close = base['High'], base['Low'], base['Close']
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    trp = (tr / close.replace(0, np.nan)).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    med = float(np.nanmedian(trp.to_numpy())) if len(trp) else np.nan
    if not np.isfinite(med) or med <= 0:
        return ZZ_THRESHOLD
    return float(np.clip(ADAPT_K * med, THR_MIN, THR_MAX))


def _empty(reason: str) -> Dict[str, any]:
    return {
        'is_vcp': False, 'vcp_quality': 0.0, 'contractions': [], 'contraction_count': 0,
        'contraction_quality': 0.0, 'volume_quality': 0.0, 'base_length_weeks': 0.0,
        'breakout_volume_ratio': 1.0, 'near_52w_high': False,
        'distance_from_52w_high_pct': 100.0, 'rmv': 100.0, 'pattern_details': reason,
    }


def detect_vcp(price_data: pd.DataFrame, current_price: float, phase_info: Dict,
               thr: Optional[float] = None, min_contractions: int = MIN_CONTRACTIONS,
               max_contractions: int = MAX_CONTRACTIONS) -> Dict[str, any]:
    """Detect a Volatility Contraction Pattern. Drop-in for the vendored
    ``detect_vcp_pattern`` — same return schema (``is_vcp``, ``vcp_quality``,
    ``contractions`` with number/peak_date/trough_date/peak_price/trough_price/
    drawdown_pct/volume_ratio/duration_days, ``contraction_count`` …).

    ``thr`` is the ZigZag reversal size. Leave it ``None`` (default) to scale it to the
    stock's own volatility (see ``_adaptive_threshold``); pass an explicit value to pin it
    (tests do this for deterministic pivot counts)."""
    if price_data is None or len(price_data) < 40:
        return _empty('Insufficient data')

    base = price_data.tail(min(len(price_data), LOOKBACK_BARS))
    high = base['High'].to_numpy(dtype=float)
    low = base['Low'].to_numpy(dtype=float)
    vol = (base['Volume'].to_numpy(dtype=float) if 'Volume' in base.columns
           else np.full(len(base), np.nan))
    idx = base.index

    if thr is None:
        thr = _adaptive_threshold(base)
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
    # tightens toward the pivot, so anchor on the most recent contraction and walk BACKWARD
    # including an older leg only while (a) the base's peaks stay within a flat band — a
    # bigger gap means price advanced/broke out between the legs, i.e. a *different* base —
    # and (b) the older leg is wider (the widest-first → tighter shape). This keeps the most
    # recent contraction, refuses to stitch a base across a breakout (the DVA case), and
    # lets the count vary instead of pinning at the cap.
    last_date = idx[-1]
    cutoff = last_date - pd.Timedelta(weeks=MAX_BASE_WEEKS)
    recent = [c for c in contractions
              if c['drawdown_pct'] <= MAX_DEPTH_PCT and c['peak_date'] >= cutoff]
    sel: List[dict] = []
    if recent:
        sel = [recent[-1]]
        for c in reversed(recent[:-1]):
            if len(sel) >= max_contractions:
                break
            peaks = [x['peak_price'] for x in sel] + [c['peak_price']]
            if max(peaks) / min(peaks) > PEAK_FLAT_BAND:            # top not flat -> different base
                break
            if c['drawdown_pct'] >= sel[0]['drawdown_pct'] * 0.9:   # older leg is wider-or-equal
                sel.insert(0, c)
            else:
                break
    for k, c in enumerate(sel):
        c['number'] = k + 1
        c.pop('peak_index', None)
        c.pop('trough_index', None)

    n = len(sel)
    depths = [c['drawdown_pct'] for c in sel]

    # Tightening quality (0-100): is each pullback shrinking? A slope on log-depths, so a
    # single non-monotone leg (25→12→14→6) no longer tanks an obviously-tightening base the
    # way the old strict-decrease count did. Blend the log-depth downtrend, the strict-
    # monotone fraction, and the overall first→last shrink; n==2 keeps the simple check.
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

    week_52_high = phase_info.get('week_52_high') or float(price_data['High'].tail(252).max())
    dist_high = ((week_52_high - current_price) / week_52_high * 100.0) if week_52_high > 0 else 100.0
    near_high = dist_high <= NEAR_HIGH_PCT

    v = price_data['Volume'] if 'Volume' in price_data.columns else pd.Series([], dtype=float)
    if len(v) > 20:
        avg20 = v.iloc[-21:-1].mean()
        breakout_volume_ratio = float(v.iloc[-1] / avg20) if avg20 > 0 else 1.0
    else:
        breakout_volume_ratio = 1.0

    # Volatility confirmation (strict AND-gate): swing STRUCTURE tightening alone isn't enough
    # — the base's own volatility must actually be compressed. RMV is min-max normalized 0-100
    # over its lookback, so rmv_now <= RMV_TIGHT_MAX == "current volatility in the bottom
    # quartile of the base's range" (i.e. it has contracted to the tight end). Structure is the
    # pattern's other half; we require both to agree. (We gate on the tightness *level* rather
    # than an RMV slope: at a completed base RMV has already bottomed and often ticks up as
    # price recovers toward the pivot, so a "still-falling" rule would reject mature setups.)
    rmv_series = relative_measured_volatility(base).dropna()
    rmv_now = float(rmv_series.iloc[-1]) if len(rmv_series) else 100.0
    vol_confirms = rmv_now <= RMV_TIGHT_MAX

    # A VCP: 2-6 progressively-tighter pullbacks, the last one tight, price near its high, and
    # volatility confirming the contraction (RMV in the bottom quartile of the base's range).
    is_vcp = bool(
        min_contractions <= n <= max_contractions
        and contraction_quality >= TIGHTEN_MIN_PCT
        and depths[-1] <= depths[0] * 0.8
        and depths[-1] <= FINAL_TIGHT_PCT
        and near_high
        and vol_confirms
    )

    # Quality 0-100 (same weighting as the vendored detector, so the app help text holds).
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
        'breakout_volume_ratio': round(breakout_volume_ratio, 2),
        'near_52w_high': near_high,
        'distance_from_52w_high_pct': round(dist_high, 1),
        'rmv': round(rmv_now, 1),
        'pattern_details': pattern_details,
    }
