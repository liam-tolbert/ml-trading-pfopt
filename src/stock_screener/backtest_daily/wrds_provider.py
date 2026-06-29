"""Real WRDS-backed providers — read the local cache written by ingest_wrds.py.

No live WRDS connection here (that is the ingester's job). These just load the
parquet/CSV cache and satisfy the same provider contract the synthetic providers do,
so the engine is unchanged. All ids are PERMNO.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .cache_io import read_table
from .fundamentals_adapter import compustat_to_scorer_dict
from .providers import FundamentalsProvider, PriceProvider, UniverseProvider

_OHLCV = ["open", "high", "low", "close", "volume"]
_OHLCV_CAP = ["Open", "High", "Low", "Close", "Volume"]


def _to_ohlcv_frame(g: pd.DataFrame) -> pd.DataFrame:
    has_raw = "raw_close" in g.columns
    cols = _OHLCV + (["raw_close"] if has_raw else [])
    f = g.sort_values("date").set_index("date")[cols].copy()
    f.columns = _OHLCV_CAP + (["RawClose"] if has_raw else [])
    f.index = pd.DatetimeIndex(f.index, name="Date")
    return f


class WrdsPriceProvider(PriceProvider):
    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        px = read_table(cache_dir, "prices", parse_dates=["date"])
        self._frames = {int(p): _to_ohlcv_frame(g) for p, g in px.groupby("permno")}

        spy = read_table(cache_dir, "spy", parse_dates=["date"])
        self._spy = _to_ohlcv_frame(spy)
        self._calendar = self._spy.index.unique().sort_values()

        self._delistings = {}
        try:
            dl = read_table(cache_dir, "delist", parse_dates=["delist_date"])
            for r in dl.itertuples(index=False):
                if pd.notna(r.dlret) and pd.notna(r.delist_date):
                    self._delistings[int(r.permno)] = (pd.Timestamp(r.delist_date), float(r.dlret))
        except FileNotFoundError:
            pass

    def calendar(self) -> pd.DatetimeIndex:
        return self._calendar

    def permnos(self) -> List[int]:
        return list(self._frames.keys())

    def prices(self, permno: int) -> pd.DataFrame:
        return self._frames[int(permno)]

    def spy(self) -> pd.DataFrame:
        return self._spy

    def delisting(self, permno: int) -> Optional[Tuple[pd.Timestamp, float]]:
        return self._delistings.get(int(permno))


class WrdsUniverseProvider(UniverseProvider):
    """Point-in-time membership = the permnos at the latest rebalance <= date."""

    def __init__(self, cache_dir):
        u = read_table(Path(cache_dir), "universe", parse_dates=["rebalance_date"])
        self._rebals = pd.DatetimeIndex(u["rebalance_date"].unique()).sort_values()
        self._by_rebal = {pd.Timestamp(d): set(g["permno"].astype(int))
                          for d, g in u.groupby("rebalance_date")}

    def members_asof(self, date) -> set:
        d = pd.Timestamp(date)
        idx = self._rebals.searchsorted(d, side="right") - 1
        if idx < 0:
            return set()
        return self._by_rebal[pd.Timestamp(self._rebals[idx])]


class WrdsFundamentalsProvider(FundamentalsProvider):
    def __init__(self, cache_dir):
        q = read_table(Path(cache_dir), "fundamentals", parse_dates=["datadate", "rdq"])
        self._q = {int(p): g.sort_values("datadate") for p, g in q.groupby("permno")}

    def fundamentals_asof(self, permno: int, date) -> Optional[dict]:
        g = self._q.get(int(permno))
        if g is None:
            return None
        return compustat_to_scorer_dict(g, date)


def load_wrds_providers(cache_dir="data/wrds"):
    """Return (price, universe, fundamentals) providers reading the cache — the
    real-data analogue of synthetic_provider.make_synthetic()."""
    return (WrdsPriceProvider(cache_dir),
            WrdsUniverseProvider(cache_dir),
            WrdsFundamentalsProvider(cache_dir))
