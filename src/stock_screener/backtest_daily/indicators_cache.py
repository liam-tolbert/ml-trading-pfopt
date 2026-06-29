"""One-time vectorized precompute + leak-safe slice access.

Two jobs:
  1. Performance — precompute, per name, trailing rolling indicators and a CHEAP
     ``buy_candidate`` superset mask + a vectorized phase, so the engine only makes
     the expensive vendored calls on a small candidate set on scan days.
  2. Leak-safety — ``ohlcv_upto(permno, t)`` (= ``frame.loc[:t]``) is the single
     primitive every vendored call slices through; nothing downstream ever sees a
     bar dated after ``t``.

The vectorized phase / candidate mask are LOOSER than the real vendored gate (a
strict superset), so they never drop a name ``score_buy_signal`` would accept; every
PnL-affecting decision is still a vendored call on the ``<= t`` slice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class IndicatorsCache:
    def __init__(self, calendar, pos, frames, close_arr, open_arr, low_arr,
                 cand_arr, phase_arr, raw_arr, spy):
        self.calendar = calendar
        self.pos = pos
        self.frames = frames
        self.close_arr = close_arr
        self.open_arr = open_arr
        self.low_arr = low_arr
        self.cand_arr = cand_arr
        self.phase_arr = phase_arr
        self.raw_arr = raw_arr          # UNADJUSTED close, for the price floor
        self.spy = spy

    # ---- build ----------------------------------------------------------- #
    @staticmethod
    def _vec_phase(close, sma50, sma200, slope50_proxy):
        c = close.to_numpy(dtype=float)
        s50 = sma50.to_numpy(dtype=float)
        s200 = sma200.to_numpy(dtype=float)
        sl = slope50_proxy.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist50 = (c / s50 - 1.0) * 100.0
        p4 = (c < s50) & (c < s200) & (s50 < s200)
        p2 = (c > s50) & (s50 > s200) & (sl > 0)
        p3 = (c > s50) & (dist50 > 25)
        phase = np.select([p4, p2, p3], [4, 2, 3], default=1).astype(float)
        phase[np.isnan(s200)] = 0.0                 # warmup
        return pd.Series(phase, index=close.index)

    @classmethod
    def build(cls, price_provider, cfg):
        cal = pd.DatetimeIndex(price_provider.calendar()).unique().sort_values()
        pos = {ts: i for i, ts in enumerate(cal)}
        spy = price_provider.spy().sort_index()

        min_price = getattr(cfg, "min_price", 0.0) or 0.0
        frames, close_arr, open_arr, low_arr, cand_arr, phase_arr, raw_arr = \
            {}, {}, {}, {}, {}, {}, {}
        for permno in price_provider.permnos():
            f = price_provider.prices(permno).sort_index()
            frames[int(permno)] = f
            close = f["Close"].astype(float)
            high = f["High"].astype(float)
            low = f["Low"].astype(float)
            openp = f["Open"].astype(float)
            # UNADJUSTED close for the price floor; fall back to Close when absent
            raw = (f["RawClose"].astype(float) if "RawClose" in f.columns else close)

            sma50 = close.rolling(50, min_periods=50).mean()
            sma200 = close.rolling(200, min_periods=200).mean()
            hi52 = high.rolling(252, min_periods=200).max()
            slope50_proxy = sma50 - sma50.shift(20)         # rising if > 0

            phase = cls._vec_phase(close, sma50, sma200, slope50_proxy)
            cand = (sma200.notna()
                    & (close > sma50 * 0.98)
                    & (sma50 > sma200 * 0.98)
                    & (close >= hi52 * 0.73)                # within ~27% of 52w high
                    & (raw >= min_price))                   # Minervini price floor
            cand = cand.fillna(False)

            close_arr[int(permno)] = close.reindex(cal).to_numpy(dtype=float)
            open_arr[int(permno)] = openp.reindex(cal).to_numpy(dtype=float)
            low_arr[int(permno)] = low.reindex(cal).to_numpy(dtype=float)
            raw_arr[int(permno)] = raw.reindex(cal).to_numpy(dtype=float)
            cand_arr[int(permno)] = cand.reindex(cal, fill_value=False).to_numpy(dtype=bool)
            phase_arr[int(permno)] = phase.reindex(cal).fillna(0).to_numpy(dtype=int)

        return cls(cal, pos, frames, close_arr, open_arr, low_arr, cand_arr, phase_arr,
                   raw_arr, spy)

    # ---- access ---------------------------------------------------------- #
    def ohlcv_upto(self, permno, t):
        """The leak primitive: this name's bars with index <= t (or None)."""
        f = self.frames.get(int(permno))
        if f is None:
            return None
        return f.loc[:pd.Timestamp(t)]

    def spy_upto(self, t):
        return self.spy.loc[:pd.Timestamp(t)]

    def _scalar(self, arr_dict, permno, t):
        a = arr_dict.get(int(permno))
        if a is None:
            return np.nan
        i = self.pos.get(pd.Timestamp(t))
        return a[i] if i is not None else np.nan

    def close(self, permno, t):
        return self._scalar(self.close_arr, permno, t)

    def open(self, permno, t):
        return self._scalar(self.open_arr, permno, t)

    def low(self, permno, t):
        return self._scalar(self.low_arr, permno, t)

    def raw_price(self, permno, t):
        return self._scalar(self.raw_arr, permno, t)

    def buy_candidate_mask(self, permno, t):
        a = self.cand_arr.get(int(permno))
        if a is None:
            return False
        i = self.pos.get(pd.Timestamp(t))
        return bool(a[i]) if i is not None else False

    def vectorized_phase(self, permno, t):
        a = self.phase_arr.get(int(permno))
        if a is None:
            return 0
        i = self.pos.get(pd.Timestamp(t))
        return int(a[i]) if i is not None else 0
