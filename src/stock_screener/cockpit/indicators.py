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
