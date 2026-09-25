"""Cockpit-local technical indicators, kept out of the vendored screening package.

RMV (Relative Measured Volatility) is a decision-support metric for the SEPA cockpit, not
part of the upstream Minervini screener. So it lives here, not in the vendored
``minervini_screener/screening`` tree (see that package's PROVENANCE.md).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def prior_volume_average(vol: pd.Series, window: int) -> pd.Series:
    """Rolling mean of the ``window`` bars before each bar: the series form of the
    breakout-confirmation denominator.

    The ``shift(1)`` MUST stay. A breakout bar in its own average dilutes the spike it is
    measured against, and the bigger the spike, the more it dilutes. NaN until a full
    window exists, so a short frame reads as unknown, not as a confident ratio off three
    bars."""
    return vol.shift(1).rolling(window, min_periods=window).mean()


def volume_ratio(df: pd.DataFrame, window: int) -> Optional[float]:
    """Last bar's volume over the mean of the ``window`` bars before it.
    None when there's no Volume column, too little history, or a non-positive mean.

    The one breakout-confirmation read. The trigger job, the Positions heavy-volume flag
    and the weekend hunt all call it, because they enforce one doctrine rule."""
    try:
        v = df["Volume"]
        if len(v) < window + 1:
            return None
        avg = float(v.iloc[-(window + 1):-1].mean())
        return float(v.iloc[-1]) / avg if avg > 0 else None
    except Exception:
        return None


def dollar_adv(df: Optional[pd.DataFrame], days: int) -> Optional[float]:
    """Average daily dollar volume: the mean of ``Close × Volume`` over the last ``days``
    bars. None without Close/Volume, under ``days`` bars, or when the mean isn't a
    positive number."""
    try:
        if df is None or len(df) < days:
            return None
        v = float((df["Close"] * df["Volume"]).tail(days).mean())
        return v if np.isfinite(v) and v > 0 else None
    except Exception:
        return None


def _true_range(df: pd.DataFrame) -> pd.Series:
    """Raw true range per bar: max of (H-L, |H-prev C|, |L-prev C|), in price units. The
    Keltner ATR in :func:`ttm_squeeze` needs it unscaled; :func:`true_range_pct` is the
    scale-free form."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift()
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                     axis=1).max(axis=1)


def true_range_pct(df: pd.DataFrame) -> pd.Series:
    """True range as a fraction of the close, so a $10 and a $500 stock compare. RMV and
    vcp.py's adaptive-threshold and dead-tape reads share it, so the TR% convention is
    defined once. A zero close gives NaN, not inf."""
    return _true_range(df) / df["Close"].replace(0, np.nan)


def relative_measured_volatility(df: pd.DataFrame, atr_period: int = 10,
                                 lookback: int = 50) -> pd.Series:
    """Deepvue-style RMV: current volatility vs its own recent range, 0-100 (low = tight).

    True range as a fraction of price is smoothed over ``atr_period`` bars, then min-max
    normalized over the trailing ``lookback`` bars. RMV near 0 means the stock is as quiet
    as it has been all window: a tight VCP contraction. RMV near 100 is the loud end of its
    recent range. Traders treat < ~25 as an ideal tight base.

    RMV compares a stock with its own recent volatility. That is a different axis from the
    breakout's volume surge, so the two don't conflict: you want a quiet base (low RMV)
    that then breaks out on heavy volume.

    Returns a Series aligned to ``df``: NaN until enough history exists, and where the
    window's volatility is flat.
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

    BandWidth = (upper - lower) / middle = ``2 * num_std * sigma / sma``, the classic
    Bollinger squeeze measure. The current width is percentile-ranked within the trailing
    ``lookback`` bars, so the output is 0-100: **low = a squeeze** (bands as tight as they've
    been all window), high = expanded. It is close-based, so it cross-checks RMV, which is
    true-range based and sees gaps and wicks. The two agreeing is a stronger tight-base
    signal than either alone.

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
    """The final value of :func:`bollinger_bandwidth_percentile`, without its per-row
    Python rolling apply over the full history. The scan reads only the last value.

    Bit-identical to the series' last row: ``rolling(lookback, min_periods=period)`` at the
    final row sees exactly ``tail(min(len, lookback))``. Returns ``None`` when the tail holds
    fewer than ``period`` band values or the final one is NaN; no older value stands in."""
    upper, mid, lower = bollinger_bands(df, period, num_std)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    w = bandwidth.tail(lookback).to_numpy()
    x = w[~np.isnan(w)] if len(w) else w
    if len(w) == 0 or len(x) < period or np.isnan(w[-1]):
        return None
    return float((x <= w[-1]).mean() * 100.0)


def bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0):
    """Classic Bollinger Bands: middle = SMA(``period``), upper/lower = middle ± ``num_std``·σ.

    Population std (``ddof=0``). ``ttm_squeeze`` and ``bollinger_bandwidth_percentile``
    build on it, so the band the chart draws is the envelope those squeeze reads use.
    Returns ``(upper, middle, lower)`` Series aligned to ``df`` (NaN during warm-up).
    """
    close = df["Close"]
    mid = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return mid + num_std * sd, mid, mid - num_std * sd


def ttm_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                kc_period: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """TTM Squeeze: True where the Bollinger Bands sit *inside* the Keltner Channel.

    Bollinger Bands are ``sigma``-based (close dispersion); the Keltner Channel is ATR-based
    (true range). A squeeze means volatility is compressed on *both* measures at once.
    Bands = ``sma ± bb_std * sigma``; Keltner = ``ema ± kc_mult * ATR``.

    Returns a boolean Series aligned to ``df`` (False during warm-up).
    """
    close = df["Close"]
    bb_upper, _mid, bb_lower = bollinger_bands(df, bb_period, bb_std)

    # Price units, not TR%: the Keltner band is an absolute envelope.
    atr = _true_range(df).rolling(kc_period, min_periods=kc_period).mean()
    ema = close.ewm(span=kc_period, adjust=False, min_periods=kc_period).mean()
    kc_upper, kc_lower = ema + kc_mult * atr, ema - kc_mult * atr

    # NaN comparisons yield False, so the warm-up region is already un-squeezed.
    return ((bb_lower > kc_lower) & (bb_upper < kc_upper)).fillna(False).astype(bool)
