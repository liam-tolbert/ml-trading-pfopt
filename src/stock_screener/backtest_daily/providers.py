"""Data-provider contracts + simple in-memory implementations.

The engine talks ONLY to these interfaces, so it never imports WRDS. Everything is
keyed on a stable integer id (``permno``), never a ticker — tickers change and get
recycled, which is exactly the survivorship trap we are trying to avoid.

Concrete in-memory implementations (``InMemoryPriceProvider`` etc.) are used by the
synthetic provider and by tests; the live ``WrdsProvider`` (stub) lives in
``wrds_provider.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .fundamentals_adapter import compustat_to_scorer_dict

OHLCV = ("Open", "High", "Low", "Close", "Volume")


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
class PriceProvider(ABC):
    @abstractmethod
    def calendar(self) -> pd.DatetimeIndex:
        """Master trading calendar (sorted, unique)."""

    @abstractmethod
    def permnos(self) -> List[int]:
        """Every permno that ever existed (the cache iterates this)."""

    @abstractmethod
    def prices(self, permno: int) -> pd.DataFrame:
        """Full daily OHLCV for one name, DatetimeIndex over its listed window."""

    @abstractmethod
    def spy(self) -> pd.DataFrame:
        """Daily OHLCV for the market benchmark (regime gate)."""

    @abstractmethod
    def delisting(self, permno: int) -> Optional[Tuple[pd.Timestamp, float]]:
        """(`delist_date`, `delisting_return`) or None if the name still trades."""

    def ticker(self, permno: int) -> str:  # display only; default = str(permno)
        return str(permno)


class UniverseProvider(ABC):
    @abstractmethod
    def members_asof(self, date) -> set:
        """Point-in-time eligible permnos on ``date`` (survivorship-free)."""


class FundamentalsProvider(ABC):
    @abstractmethod
    def fundamentals_asof(self, permno: int, date) -> Optional[dict]:
        """The scorer's fundamentals dict using only quarters known by ``date``."""


# --------------------------------------------------------------------------- #
# In-memory implementations (synthetic + tests)
# --------------------------------------------------------------------------- #
class InMemoryPriceProvider(PriceProvider):
    def __init__(self, frames: Dict[int, pd.DataFrame], spy: pd.DataFrame,
                 delistings: Optional[Dict[int, Tuple[pd.Timestamp, float]]] = None,
                 calendar: Optional[pd.DatetimeIndex] = None,
                 tickers: Optional[Dict[int, str]] = None):
        self._frames = {int(k): v.sort_index() for k, v in frames.items()}
        self._spy = spy.sort_index()
        self._delistings = {int(k): (pd.Timestamp(d), float(r))
                            for k, (d, r) in (delistings or {}).items()}
        self._tickers = dict(tickers or {})
        if calendar is not None:
            self._calendar = pd.DatetimeIndex(calendar).unique().sort_values()
        else:
            idx = self._spy.index
            for f in self._frames.values():
                idx = idx.union(f.index)
            self._calendar = pd.DatetimeIndex(idx).unique().sort_values()

    def calendar(self) -> pd.DatetimeIndex:
        return self._calendar

    def permnos(self) -> List[int]:
        return list(self._frames.keys())

    def prices(self, permno: int) -> pd.DataFrame:
        return self._frames[int(permno)]

    def spy(self) -> pd.DataFrame:
        return self._spy

    def delisting(self, permno: int):
        return self._delistings.get(int(permno))

    def ticker(self, permno: int) -> str:
        return self._tickers.get(int(permno), str(permno))

    def truncate(self, cutoff) -> "InMemoryPriceProvider":
        """A copy with all data clipped to index <= cutoff (for the leak test)."""
        cutoff = pd.Timestamp(cutoff)
        frames = {p: f.loc[:cutoff] for p, f in self._frames.items()}
        frames = {p: f for p, f in frames.items() if len(f)}
        dels = {p: (d, r) for p, (d, r) in self._delistings.items() if d <= cutoff}
        cal = self._calendar[self._calendar <= cutoff]
        return InMemoryPriceProvider(frames, self._spy.loc[:cutoff],
                                     delistings=dels, calendar=cal, tickers=self._tickers)


class StaticUniverseProvider(UniverseProvider):
    """Eligibility from per-name [start, end] windows (inclusive). A name with no
    explicit end is eligible to the far future."""

    _FAR = pd.Timestamp("2200-01-01")

    def __init__(self, windows: Dict[int, Tuple[pd.Timestamp, Optional[pd.Timestamp]]]):
        self._windows = {
            int(p): (pd.Timestamp(s), self._FAR if e is None else pd.Timestamp(e))
            for p, (s, e) in windows.items()
        }

    @classmethod
    def always(cls, permnos) -> "StaticUniverseProvider":
        return cls({int(p): (pd.Timestamp("1900-01-01"), None) for p in permnos})

    def members_asof(self, date) -> set:
        d = pd.Timestamp(date)
        return {p for p, (s, e) in self._windows.items() if s <= d <= e}


class DictFundamentalsProvider(FundamentalsProvider):
    """Per-name quarterly frames -> scorer dict via the point-in-time adapter."""

    def __init__(self, quarters_by_permno: Dict[int, pd.DataFrame]):
        self._q = {int(k): v for k, v in quarters_by_permno.items()}

    def fundamentals_asof(self, permno: int, date) -> Optional[dict]:
        q = self._q.get(int(permno))
        if q is None:
            return None
        return compustat_to_scorer_dict(q, date)


class NullFundamentalsProvider(FundamentalsProvider):
    """Always returns None (scorer falls back to its neutral fundamentals score)."""

    def fundamentals_asof(self, permno: int, date) -> Optional[dict]:
        return None
