"""Cockpit-local VCP / contraction detector (a drop-in for the vendored one).

The vendored ``minervini_screener`` ``detect_vcp_pattern`` finds zero contractions on most
of a broad universe (its base anchors too tight for its swing window), so this re-implements
detection with a **volatility-adaptive ZigZag** yielding a strictly-alternating high/low
pivot sequence — contractions are always well-formed (depth ≥ 0) and shallow tight bases
are found. Returns the SAME dict schema, so it drops into ``scan.py`` unchanged.

Design goal (validated against the 200-chart hand-labeled benchmark in
``tests/vcp_labels.py``): a **recall-first pre-filter** that must never hide a live setup —
misses are unacceptable, false alarms only cost a glance. So instead of one yes/no it
assigns a review **tier**:

  A — review: a valid tightening base with price below / at / within ~5% above the pivot.
  B — watch: a plausible base still forming, or a valid pattern already extended past the
      buy zone. Never hidden.
  C — skipped, with the reason recorded: only *safe* exclusions (dead tape, no pullbacks
      at any threshold, stale base) that cannot be a usable setup.

Detection runs at several ZigZag thresholds (long-history, recent-window, and extra-tight
variants) and keeps the best read — a VCP by definition ends *quieter* than the stock's own
history, so a single history-calibrated threshold goes blind exactly at the tight ending.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import relative_measured_volatility

# --- Tunables (calibrated against the live full_us funnel + the 200-chart benchmark) --- #
ZZ_THRESHOLD = 0.04        # fallback ZigZag reversal size (used when the adaptive estimate
                           # can't be computed, and as the calibration anchor for ADAPT_K).
ATR_PERIOD = 10            # smoothing for the true-range% used by the adaptive threshold
ADAPT_K = 2.75             # adaptive ZigZag threshold = ADAPT_K × median(true-range%) over the
                           # base, clipped to [THR_MIN, THR_MAX]. K is set so a typical mid-vol
                           # name (~1.5%/day true range) lands near the old fixed 0.04.
THR_MIN = 0.03             # floor: don't chase noise on ultra-quiet mega-caps
THR_MAX = 0.10             # cap: don't blur a high-vol small-cap's swings into one leg
RECENT_WINDOW_BARS = 42    # ~2 months: "how quiet is the stock NOW" (the contraction itself)
RECENT_THR_SHRINK = 0.7    # extra-tight candidate = 0.7 × the recent-window threshold …
THR_MIN_TIGHT = 0.02       # … floored here (deliberately below THR_MIN, to see tight endings)
THR_FIXED_TIGHT = 0.035    # always-on tight candidate: the recent window can be polluted by the
                           # breakout burst itself, so a volatility-scaled ladder alone goes blind
                           # on tight flags. 3.5% still sees the 4-6% legs of quiet bases.
TIGHTEN_MIN_PCT = 60.0     # is_vcp gate on the 0-100 tightening-quality score (below)
MIN_CONTRACTIONS = 2
MAX_CONTRACTIONS = 6
MAX_BASE_WEEKS = 65        # a base older than this isn't the current setup
MAX_DEPTH_PCT = 35.0       # a leg deeper than this is a decline, not a base contraction
PEAK_FLAT_BAND = 1.15      # base peaks must sit within this ratio (a flat-ish top); a bigger
                           # gap means price advanced/broke out between the legs -> a different base
FINAL_TIGHT_PCT = 12.0     # the last contraction must be at least this tight
UNIFORM_TIGHT_PCT = 6.5    # …or, if the "tighter than 0.8× the first leg" ratio fails because ALL
                           # legs are already small (a uniform quiet shelf), a final leg at/below
                           # this absolute size still counts as tight
NEAR_HIGH_PCT = 25.0       # price must be within this % of the 52-week high
LOOKBACK_BARS = 325        # ~65 weeks of trading days
RMV_TIGHT_MAX = 30.0       # RMV gate: below the pivot the base must be quiet (near the bottom of
                           # its own volatility range); at/above the pivot a breakout is a burst of
                           # movement, so RMV stops vetoing and structure alone decides. The
                           # below-pivot veto stays — removing it admits loud/junk names.
# Sanity rules (HANDOFF §6):
MIN_LEG_BARS = 2           # same-day / 1-bar "legs" are junk anchors; 2-day shakeouts are real
                           # final contractions in quiet staged climbers, so the floor sits at 2.
MIN_BASE_WEEKS = 2.0       # shorter than this isn't a base. Kept low (2.0) because base length is
                           # measured over the SELECTED legs only, which under-reads the true base.
MAX_LEG_AGE_WEEKS = 13.0   # newest pullback older than this = stale base (the TWO/MNST class)
DEAD_TAPE_BARS = 42        # dead-tape window (~60 calendar days) …
DEAD_TAPE_MEDIAN_TR = 0.010  # … median daily true-range% below 1% = pinned/zombie tape (deal
                           # arbs, zombie listings): no swings to contract, so it can't be a setup.
BUY_ZONE_PCT = 0.10        # tier A allows price up to this far above the DETECTED pivot. That
                           # pivot (top of the selected legs) usually sits below the true
                           # actionable pivot, so +10% detector-relative ≈ "≤~5-8% past the real
                           # pivot"; a further-extended name stays out.
NEAR_PIVOT_BAND = 0.10     # …and no more than this far BELOW it: price collapsing away from the
                           # base top is a failing pattern, not a coil.

_TIER_RANK = {'A': 0, 'B': 1, 'C': 2}


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


def _true_range_pct(df: pd.DataFrame) -> pd.Series:
    """Per-bar true range as a fraction of the close (scale-free volatility)."""
    high, low, close = df['High'], df['Low'], df['Close']
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr / close.replace(0, np.nan)


def _adaptive_threshold(base: pd.DataFrame) -> float:
    """Scale the ZigZag reversal size to the stock's *own* volatility.

    A high-vol small-cap needs a wider swing filter (or one pullback fragments into three);
    an ultra-quiet mega-cap needs a tighter one (or its tight final contraction is invisible).
    We use median true-range% over the base — the same scale-free volatility RMV is built on —
    so ``thr`` floats with the stock instead of being a magic 0.04. Clipped to [THR_MIN, THR_MAX].
    """
    trp = _true_range_pct(base).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
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
        'tier': 'C', 'zz_threshold': None,
    }


def _detect_at(price_data: pd.DataFrame, base: pd.DataFrame, current_price: float,
               phase_info: Dict, thr: float, min_contractions: int,
               max_contractions: int) -> Dict[str, any]:
    """One detection pass at a fixed ZigZag threshold; returns the full result dict."""
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
    # tightens toward the pivot, so anchor on the most recent contraction and walk BACKWARD,
    # including an older leg only while (a) the base's peaks stay within a flat band (a bigger
    # gap means price broke out between the legs — a *different* base) and (b) the older leg is
    # wider (widest-first → tighter shape). Legs spanning < MIN_LEG_BARS trading days are
    # dropped up front — a same-day/2-day dip is noise that produced fake single-leg "bases".
    last_date = idx[-1]
    cutoff = last_date - pd.Timedelta(weeks=MAX_BASE_WEEKS)
    recent = [c for c in contractions
              if c['drawdown_pct'] <= MAX_DEPTH_PCT and c['peak_date'] >= cutoff
              and (c['trough_index'] - c['peak_index']) >= MIN_LEG_BARS]
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

    # Tightening quality (0-100): is each pullback shrinking? A slope on log-depths, so a single
    # non-monotone leg (25→12→14→6) doesn't tank an obviously-tightening base. Blend the log-depth
    # downtrend, the strict-monotone fraction, and the overall first→last shrink; n==2 stays simple.
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

    week_52_high = phase_info.get('week_52_high') or float(price_data['High'].tail(252).max())
    dist_high = ((week_52_high - current_price) / week_52_high * 100.0) if week_52_high > 0 else 100.0
    near_high = dist_high <= NEAR_HIGH_PCT

    v = price_data['Volume'] if 'Volume' in price_data.columns else pd.Series([], dtype=float)
    if len(v) > 20:
        avg20 = v.iloc[-21:-1].mean()
        breakout_volume_ratio = float(v.iloc[-1] / avg20) if avg20 > 0 else 1.0
    else:
        breakout_volume_ratio = 1.0

    # Volatility read. RMV is min-max normalized 0-100 over its lookback, so rmv_now <=
    # RMV_TIGHT_MAX means volatility has contracted to the tight end of the base's own range.
    # Below the pivot the base should be quiet, so RMV vetoes there; at/above the pivot a
    # breakout is a burst of movement, so it stops vetoing and structure alone decides.
    rmv_series = relative_measured_volatility(base).dropna()
    rmv_now = float(rmv_series.iloc[-1]) if len(rmv_series) else 100.0

    pivot_price = max((c['peak_price'] for c in sel), default=None)
    below_pivot = bool(pivot_price and current_price < pivot_price)
    in_buy_zone = bool(pivot_price
                       and pivot_price * (1 - NEAR_PIVOT_BAND)
                       <= current_price
                       <= pivot_price * (1 + BUY_ZONE_PCT))
    vol_confirms = rmv_now <= RMV_TIGHT_MAX
    rmv_ok = vol_confirms or not below_pivot

    fresh = leg_age_weeks <= MAX_LEG_AGE_WEEKS
    long_enough = base_length_weeks >= MIN_BASE_WEEKS
    # Tight ending: clearly tighter than the first leg, or absolutely small — a uniform
    # quiet shelf (4.4% → 3.8%) can't shrink 20% further but IS tight.
    final_tight = bool(depths and depths[-1] <= FINAL_TIGHT_PCT
                       and (depths[-1] <= depths[0] * 0.8
                            or depths[-1] <= UNIFORM_TIGHT_PCT))

    # A VCP: 2-6 progressively-tighter pullbacks (each spanning >= MIN_LEG_BARS days), a
    # tight final leg, a base at least 3 weeks long with its newest leg recent, price near
    # its high, and — while still below the pivot — volatility confirming the contraction.
    structure_ok = bool(
        min_contractions <= n <= max_contractions
        and contraction_quality >= TIGHTEN_MIN_PCT
        and final_tight
        and near_high
        and long_enough
        and fresh
    )
    is_vcp = structure_ok and rmv_ok

    # ---- Review tier (recall-first: C is ONLY for safe, can't-be-a-setup exclusions) ---- #
    # Each branch's comment records why that tier was assigned.
    if n == 0:
        tier = 'C'          # no pullbacks found — nothing resembling a base
    elif not fresh:
        tier = 'C'          # stale base: newest pullback {leg_age_weeks:.0f} weeks old
    elif is_vcp and in_buy_zone:
        tier = 'A'          # valid tightening base in/near the buy zone
    elif is_vcp:
        tier = 'B'          # valid pattern but extended past / fallen away from the pivot
    else:
        tier = 'B'          # base still forming (few legs / far from the 52-wk high /
        #                     too short / final leg not tight / not tightening / loud tape)

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
        'tier': tier,
        'zz_threshold': round(float(thr), 4),
        # pivot_price is not exported, but the calculation above stays LIVE: in_buy_zone /
        # below_pivot / the A-vs-B tier split all hang off it. Un-comment to expose it:
        # 'pivot_price': round(pivot_price, 2) if pivot_price else None,
    }


def detect_vcp(price_data: pd.DataFrame, current_price: float, phase_info: Dict,
               thr: Optional[float] = None, min_contractions: int = MIN_CONTRACTIONS,
               max_contractions: int = MAX_CONTRACTIONS) -> Dict[str, any]:
    """Detect a Volatility Contraction Pattern. Drop-in for the vendored
    ``detect_vcp_pattern`` — same return schema (``is_vcp``, ``vcp_quality``,
    ``contractions`` with number/peak_date/trough_date/peak_price/trough_price/
    drawdown_pct/volume_ratio/duration_days, ``contraction_count`` …) plus the review
    ``tier`` ('A'/'B'/'C') and ``zz_threshold``. (The pivot is computed internally for the
    buy-zone/extended tier split but not exported; see ``_detect_at``.)

    ``thr`` is the ZigZag reversal size. Leave it ``None`` (default) to run at up to four
    thresholds — long-history, recent-window (~2 months), an extra-tight 0.7× recent, and
    a fixed 3.5% — keeping the best read (a strict pass wins; otherwise the strongest
    tier/quality). Pass an explicit value to pin a single threshold (tests do this for
    deterministic pivot counts; pinning also skips the dead-tape guard)."""
    if price_data is None or len(price_data) < 40:
        return _empty('Insufficient data')

    base = price_data.tail(min(len(price_data), LOOKBACK_BARS))

    if thr is not None:
        # Pinned threshold = a raw single-threshold read (tests pin this for deterministic
        # pivot counts); the dead-tape guard is skipped so synthetic H=L=C frames (which
        # have no intrabar range and so under-read true range) stay usable.
        candidates = [float(thr)]
    else:
        # Dead tape (threshold-independent): a stock pinned flat for months has no swings to
        # contract. Median daily true-range% below DEAD_TAPE_MEDIAN_TR can't be a live setup.
        med_tr = float(np.nanmedian(_true_range_pct(base).tail(DEAD_TAPE_BARS).to_numpy()))
        if np.isfinite(med_tr) and med_tr < DEAD_TAPE_MEDIAN_TR:
            return _empty(f'Dead tape: median daily range {med_tr * 100:.2f}% '
                          f'over the last {DEAD_TAPE_BARS} sessions')
        thr_long = _adaptive_threshold(base)
        thr_recent = _adaptive_threshold(base.tail(RECENT_WINDOW_BARS))
        thr_tight = max(RECENT_THR_SHRINK * thr_recent, THR_MIN_TIGHT)
        candidates = sorted({round(t, 4) for t in (thr_long, thr_recent, thr_tight,
                                                   THR_FIXED_TIGHT)},
                            reverse=True)

    results = [_detect_at(price_data, base, current_price, phase_info, t,
                          min_contractions, max_contractions) for t in candidates]
    return min(results, key=lambda r: (0 if r['is_vcp'] else 1,
                                       _TIER_RANK[r['tier']], -r['vcp_quality']))
