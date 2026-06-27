"""Deterministic synthetic data so the harness RUNS and is testable today.

Generates a market (SPY with bull then bear blocks) and a cast of PERMNO-keyed
names with scripted life-cycles — "winners" (base -> Stage-2 uptrend that passes the
Minervini template -> decline), "losers" (never qualify), and a couple of late
listers — plus point-in-time eligibility windows and one mid/late delisting. Quarterly
fundamentals carry an ``rdq`` report-date lag so the PIT fundamentals path is exercised.

All paths are seeded; the same seed reproduces the same data exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .providers import (
    DictFundamentalsProvider,
    InMemoryPriceProvider,
    StaticUniverseProvider,
)


@dataclass
class SyntheticData:
    price: InMemoryPriceProvider
    universe: StaticUniverseProvider
    fundamentals: DictFundamentalsProvider
    winners: List[int]
    losers: List[int]
    late: List[int]
    delistings: Dict[int, tuple]


def _ohlcv(close, dates, band=0.008, vol=1_000_000.0) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    openp = np.empty_like(close)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    hi = np.maximum(openp, close) * (1.0 + band)
    lo = np.minimum(openp, close) * (1.0 - band)
    volume = np.full(len(close), float(vol))
    return pd.DataFrame(
        {"Open": openp, "High": hi, "Low": lo, "Close": close, "Volume": volume},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _winner_close(periods, base_len, up_len, dec_len, rng,
                  base=100.0, up=0.004, dec=-0.004) -> np.ndarray:
    n_flat = max(0, periods - base_len - up_len - dec_len)
    rates = np.concatenate([
        np.zeros(base_len),
        np.full(up_len, up),
        np.full(dec_len, dec),
        np.zeros(n_flat),
    ])[:periods]
    noise = rng.normal(0.0, 0.002, size=len(rates))
    return base * np.cumprod(1.0 + rates + noise)


def _loser_close(periods, rng, base=80.0, drift=-0.0005) -> np.ndarray:
    noise = rng.normal(0.0, 0.012, size=periods)
    return np.maximum(base * np.cumprod(1.0 + drift + noise), 1.0)


def _spy_close(periods, rng, base=1000.0, base_len=150, up_len=410,
               up=0.0007, dec=-0.0018) -> np.ndarray:
    dec_len = max(0, periods - base_len - up_len)
    rates = np.concatenate([np.zeros(base_len), np.full(up_len, up),
                            np.full(dec_len, dec)])[:periods]
    noise = rng.normal(0.0, 0.005, size=len(rates))
    return base * np.cumprod(1.0 + rates + noise)


def _quarters(dates, growth) -> pd.DataFrame:
    """Quarterly fundamentals along a name's dates: a datadate every ~63 business
    days, rdq lagged 45 calendar days, revenue/eps compounding at ``growth``."""
    recs = []
    rev, eps, inv = 100.0, 1.0, 50.0
    for k in range(0, len(dates), 63):
        datadate = dates[k]
        recs.append({"datadate": datadate, "rdq": datadate + pd.Timedelta(days=45),
                     "revtq": rev, "eps": eps, "invtq": inv})
        rev *= (1.0 + growth)
        eps *= (1.0 + growth)
    return pd.DataFrame(recs)


def make_synthetic(seed=0, n_winners=10, n_losers=6, n_late=2,
                   periods=760, start="2015-01-02") -> SyntheticData:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=periods)
    spy = _ohlcv(_spy_close(periods, rng), cal, band=0.004, vol=5_000_000.0)

    frames, windows, quarters, tickers = {}, {}, {}, {}
    delistings: Dict[int, tuple] = {}
    winners, losers, late = [], [], []
    pid = 1001

    for i in range(n_winners):
        base_len = 200 + i * 6                          # staggered breakouts; warmup satisfied
        close = _winner_close(periods, base_len, 170, 120, rng)
        frames[pid] = _ohlcv(close, cal)
        windows[pid] = (cal[0], None)
        quarters[pid] = _quarters(cal, growth=0.08)
        tickers[pid] = f"WIN{i}"
        winners.append(pid)
        pid += 1

    for i in range(n_losers):
        close = _loser_close(periods, rng)
        frames[pid] = _ohlcv(close, cal)
        windows[pid] = (cal[0], None)
        quarters[pid] = _quarters(cal, growth=-0.02)
        tickers[pid] = f"LOS{i}"
        losers.append(pid)
        pid += 1

    for i in range(n_late):
        list_pos = 100 + i * 20
        sub_cal = cal[list_pos:]
        m = len(sub_cal)
        close = _winner_close(m, 200, 150, 100, rng)
        frames[pid] = _ohlcv(close, sub_cal)
        windows[pid] = (sub_cal[0], None)
        quarters[pid] = _quarters(sub_cal, growth=0.07)
        tickers[pid] = f"LATE{i}"
        late.append(pid)
        pid += 1

    # one loser goes bankrupt near the end (delisting return -85%)
    if losers:
        dl = losers[-1]
        ddate = cal[periods - 60]
        frames[dl] = frames[dl].loc[:ddate]
        windows[dl] = (cal[0], ddate)
        delistings[dl] = (ddate, -0.85)

    price = InMemoryPriceProvider(frames, spy, delistings=delistings,
                                  calendar=cal, tickers=tickers)
    universe = StaticUniverseProvider(windows)
    fundamentals = DictFundamentalsProvider(quarters)
    return SyntheticData(price, universe, fundamentals, winners, losers, late, delistings)
