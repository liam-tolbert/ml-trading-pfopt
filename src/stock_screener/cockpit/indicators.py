"""Cockpit-local technical indicators (kept out of the vendored screening package).

RMV (Relative Measured Volatility) is a decision-support metric for the SEPA cockpit, not
part of the upstream Minervini screener — so it lives here rather than in the vendored
``minervini_screener/screening`` tree (see that package's PROVENANCE.md).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _true_range(df: pd.DataFrame) -> pd.Series:
    """Raw true range per bar: max of (H-L, |H-prev C|, |L-prev C|). PRICE units — the
    Keltner ATR in :func:`ttm_squeeze` needs it unscaled; use :func:`true_range_pct` for
    the scale-free variant."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift()
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                     axis=1).max(axis=1)


def true_range_pct(df: pd.DataFrame) -> pd.Series:
    """True range as a FRACTION of price (scale-free across a $10 and a $500 stock).
    Shared by RMV here and vcp.py's adaptive-threshold / dead-tape reads — one source of
    truth for the TR% convention (zero closes -> NaN, never a division blow-up)."""
    return _true_range(df) / df["Close"].replace(0, np.nan)


def relative_measured_volatility(df: pd.DataFrame, atr_period: int = 10,
                                 lookback: int = 50) -> pd.Series:
    """Deepvue-style RMV: current volatility vs its own recent range, 0-100 (low = tight).

    True range is taken as a fraction of price (so the score is scale-free across a $10 and
    a $500 stock), smoothed over ``atr_period`` bars, then min-max normalized over the
    trailing ``lookback`` bars. RMV → 0 means the stock is as quiet as it has been all
    window (a tight VCP contraction — a low-risk, high-quality base); RMV → 100 means it's
    at the loud end of its recent range. Traders treat < ~25 as an ideal tight base.

    RMV is *self-referential* (a stock vs its own recent volatility) — a different axis from
    the breakout's volume surge, so the two don't conflict: you want a quiet base (low RMV)
    that then breaks out on heavy volume.

    Returns a Series aligned to ``df`` (NaN until enough history exists).
    """
    vol = true_range_pct(df).rolling(atr_period, min_periods=atr_period).mean()
    lo = vol.rolling(lookback, min_periods=atr_period).min()
    hi = vol.rolling(lookback, min_periods=atr_period).max()
    span = hi - lo
    return (100.0 * (vol - lo) / span.where(span > 0)).clip(0, 100)


def bollinger_bandwidth_percentile(df: pd.DataFrame, period: int = 20,
                                   num_std: float = 2.0,
                                   lookback: int = 126) -> pd.Series:
    """Bollinger Band-Width Percentile (BBWP): today's band width vs its own recent range.

    BandWidth = (upper - lower) / middle = ``2 * num_std * sigma / sma`` — the classic
    Bollinger squeeze measure. We then percentile-rank the current width within the trailing
    ``lookback`` bars, so the output is 0-100: **low = a squeeze** (bands as tight as they've
    been all window), high = expanded. Close-based, so it cross-checks RMV (true-range based,
    sees gaps/wicks); the two agreeing is a stronger tight-base signal than either alone.

    Returns a Series aligned to ``df`` (NaN until enough history exists).
    """
    upper, mid, lower = bollinger_bands(df, period, num_std)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)

    def _pctrank(w: np.ndarray) -> float:
        x = w[~np.isnan(w)]
        if len(x) < period or np.isnan(w[-1]):
            return np.nan
        return float((x <= w[-1]).mean() * 100.0)

    return bandwidth.rolling(lookback, min_periods=period).apply(_pctrank, raw=True)


def bollinger_bandwidth_percentile_last(df: pd.DataFrame, period: int = 20,
                                        num_std: float = 2.0,
                                        lookback: int = 126) -> Optional[float]:
    """The FINAL value of :func:`bollinger_bandwidth_percentile`, without the per-row
    Python rolling-apply over full history (the scan reads only ``.iloc[-1]``, so the
    hundreds of `_pctrank` callbacks per ticker were pure waste). Bit-identical to the
    series' last row — ``rolling(lookback, min_periods=period)`` at the final row sees
    exactly ``tail(min(len, lookback))``. Returns ``None`` when the tail holds fewer than
    ``period`` band values or the final one is NaN (no digging up a stale older read)."""
    upper, mid, lower = bollinger_bands(df, period, num_std)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    w = bandwidth.tail(lookback).to_numpy()
    x = w[~np.isnan(w)] if len(w) else w
    if len(w) == 0 or len(x) < period or np.isnan(w[-1]):
        return None
    return float((x <= w[-1]).mean() * 100.0)


def bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0):
    """Classic Bollinger Bands: middle = SMA(``period``), upper/lower = middle ± ``num_std``·σ.

    Population std (``ddof=0``) to match ``ttm_squeeze`` and ``bollinger_bandwidth_percentile``,
    so the band the chart draws is the exact same volatility envelope those squeeze reads are
    computed from. Returns ``(upper, middle, lower)`` Series aligned to ``df`` (NaN during warm-up).
    """
    close = df["Close"]
    mid = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return mid + num_std * sd, mid, mid - num_std * sd


def ttm_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                kc_period: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """TTM Squeeze: True where the Bollinger Bands sit *inside* the Keltner Channel.

    This is the clean combination of the two volatility proxies — Bollinger Bands are
    ``sigma``-based (close dispersion) and the Keltner Channel is ATR-based (true range) —
    so a squeeze means volatility is compressed on *both* measures at once (the coiled
    spring). Bands = ``sma ± bb_std * sigma``; Keltner = ``ema ± kc_mult * ATR``.

    Returns a boolean Series aligned to ``df`` (False during warm-up).
    """
    close = df["Close"]
    bb_upper, _mid, bb_lower = bollinger_bands(df, bb_period, bb_std)

    # RAW true range (price units) — the Keltner band is an absolute envelope, unlike the
    # scale-free TR% that RMV/vcp use.
    atr = _true_range(df).rolling(kc_period, min_periods=kc_period).mean()
    ema = close.ewm(span=kc_period, adjust=False, min_periods=kc_period).mean()
    kc_upper, kc_lower = ema + kc_mult * atr, ema - kc_mult * atr

    # NaN comparisons yield False, so the warm-up region is already un-squeezed.
    return ((bb_lower > kc_lower) & (bb_upper < kc_upper)).fillna(False).astype(bool)
