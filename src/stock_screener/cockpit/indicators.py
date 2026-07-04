"""Cockpit-local technical indicators (kept out of the vendored screening package).

RMV (Relative Measured Volatility) is a decision-support metric for the SEPA cockpit,
not part of the upstream Minervini screener — so it lives here rather than in the
vendored ``minervini_screener/screening`` tree (whose edits are limited to import
rewrites; see that package's PROVENANCE.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def relative_measured_volatility(df: pd.DataFrame, atr_period: int = 10,
                                 lookback: int = 50) -> pd.Series:
    """Deepvue-style RMV: current volatility vs its own recent range, 0-100 (low = tight).

    True range is taken as a fraction of price (so the score is scale-free across a $10 and
    a $500 stock), smoothed over ``atr_period`` bars, then min-max normalized over the
    trailing ``lookback`` bars. RMV → 0 means the stock is as quiet as it has been all
    window (a tight VCP contraction — a low-risk, high-quality base); RMV → 100 means it's
    at the loud end of its recent range. Traders treat < ~25 as an ideal tight base.

    Note RMV is *self-referential* (a stock vs its own recent volatility), which is a
    different axis from the breakout's volume surge (participation) — the two don't
    conflict: you want a quiet base (low RMV) that then breaks out on heavy volume.

    Returns a Series aligned to ``df`` (NaN until enough history exists).
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    # True range as a fraction of price, so drift in the price level over the lookback
    # window doesn't masquerade as changing volatility.
    trp = tr / close.replace(0, np.nan)
    vol = trp.rolling(atr_period, min_periods=atr_period).mean()
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
    been all window), high = expanded. This is the *close-based* volatility read — it cross-
    checks RMV (which uses true range, so it also sees gaps/wicks) from the Bollinger side;
    the two agreeing is a stronger tight-base signal than either alone.

    Returns a Series aligned to ``df`` (NaN until enough history exists).
    """
    close = df["Close"]
    sma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    bandwidth = (2.0 * num_std * std) / sma.replace(0, np.nan)

    def _pctrank(w: np.ndarray) -> float:
        x = w[~np.isnan(w)]
        if len(x) < period or np.isnan(w[-1]):
            return np.nan
        return float((x <= w[-1]).mean() * 100.0)

    return bandwidth.rolling(lookback, min_periods=period).apply(_pctrank, raw=True)


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
    high, low, close = df["High"], df["Low"], df["Close"]
    sma = close.rolling(bb_period, min_periods=bb_period).mean()
    std = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
    bb_upper, bb_lower = sma + bb_std * std, sma - bb_std * std

    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(kc_period, min_periods=kc_period).mean()
    ema = close.ewm(span=kc_period, adjust=False, min_periods=kc_period).mean()
    kc_upper, kc_lower = ema + kc_mult * atr, ema - kc_mult * atr

    # NaN comparisons yield False, so the warm-up region is already un-squeezed.
    return ((bb_lower > kc_lower) & (bb_upper < kc_upper)).fillna(False).astype(bool)
