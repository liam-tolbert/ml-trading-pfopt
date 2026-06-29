"""Thin yfinance data layer for the cockpit.

Deliberately does NOT import the vendored ``minervini_screener.data`` package — that
package's ``__init__`` eager-loads SQLAlchemy (absent from the ``ml-trading`` env), so
importing even its yfinance fetcher would crash. We re-implement the small amount of
fetching we need on yfinance + requests directly.

Public surface:
- ``get_universe(name)``         -> list[str] of yfinance-normalized symbols
- ``get_prices(ticker)``         -> daily OHLCV DataFrame (cached parquet, age-refreshed)
- ``get_many_prices(tickers)``   -> {ticker: DataFrame}, threaded
- ``get_spy()``                  -> SPY daily OHLCV
- ``get_fundamentals(ticker)``   -> dict of growth/margin metrics (or None)
"""
from __future__ import annotations

import io
import json
from typing import Callable, Dict, List, Optional

import pandas as pd

from .cache import (CACHE_DIR, FUNDAMENTALS_DIR, PRICES_DIR, TICKERS_TXT,
                    age_days, ensure_dirs)

# A stable, maintained constituents CSV (no API key); Wikipedia is the fallback.
SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/"
    "data/constituents.csv"
)
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- #
# Symbols / universe
# --------------------------------------------------------------------------- #
def normalize(ticker: str) -> str:
    """yfinance uses '-' where exchanges use '.' (BRK.B -> BRK-B)."""
    return str(ticker).strip().upper().replace(".", "-")


def get_universe(name: str = "sp500", force: bool = False,
                 max_age_days: float = 7.0) -> List[str]:
    if name == "tickers":
        return _read_tickers_txt()
    if name == "sp500":
        return _get_sp500(force=force, max_age_days=max_age_days)
    if name == "full_us":
        raise NotImplementedError(
            "full_us universe not wired yet; design point is get_universe()")
    raise ValueError(f"unknown universe: {name!r}")


def _read_tickers_txt() -> List[str]:
    if not TICKERS_TXT.exists():
        return []
    syms = [normalize(line) for line in TICKERS_TXT.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    return sorted(set(syms))


def _get_sp500(force: bool, max_age_days: float) -> List[str]:
    path = CACHE_DIR / "sp500_constituents.csv"
    if not force and path.exists() and age_days(path) <= max_age_days:
        cached = _syms_from_csv(path)
        if cached:
            return cached
    syms = _fetch_sp500_datahub() or _fetch_sp500_wikipedia()
    if syms:
        ensure_dirs()
        pd.Series(sorted(set(syms)), name="Symbol").to_csv(path, index=False)
        return [normalize(s) for s in sorted(set(syms))]
    if path.exists():                       # stale-but-present beats nothing
        cached = _syms_from_csv(path)
        if cached:
            return cached
    return _read_tickers_txt()              # last-ditch offline fallback


def _syms_from_csv(path) -> List[str]:
    try:
        df = pd.read_csv(path)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        return [normalize(s) for s in df[col].dropna().astype(str)]
    except Exception:
        return []


def _fetch_sp500_datahub() -> Optional[List[str]]:
    try:
        import requests
        r = requests.get(SP500_CSV_URL, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df["Symbol"].dropna().astype(str).tolist()
    except Exception:
        return None


def _fetch_sp500_wikipedia() -> Optional[List[str]]:
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"id": "constituents"}) or soup.find("table")
        syms = []
        for row in table.find_all("tr")[1:]:
            cell = row.find("td")
            if cell:
                syms.append(cell.get_text(strip=True))
        return syms or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def _clean_prices(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):          # single-ticker download
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]            # yfinance can double columns under threads
    cols = [c for c in _OHLCV if c in df.columns]
    if "Close" not in cols:
        return None
    df = df[cols]
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df.index.name = "Date"
    return df.dropna(subset=["Close"])


def get_prices(ticker: str, lookback: str = "2y", force: bool = False,
               max_age_days: float = 1.0) -> Optional[pd.DataFrame]:
    """Daily OHLCV (auto-adjusted) for one name, parquet-cached and age-refreshed."""
    ensure_dirs()
    sym = normalize(ticker)
    path = PRICES_DIR / f"{sym}.parquet"
    if not force and age_days(path) <= max_age_days:
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    try:
        import yfinance as yf
        raw = yf.download(sym, period=lookback, interval="1d",
                          auto_adjust=True, progress=False, threads=False)
    except Exception:
        raw = None
    df = _clean_prices(raw)
    if df is None:                                      # network/empty -> stale cache
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                return None
        return None
    try:
        df.to_parquet(path)
    except Exception:
        pass
    return df


def _extract_ticker(raw: pd.DataFrame, sym: str) -> Optional[pd.DataFrame]:
    """Pull one ticker's sub-frame out of a multi-ticker yf.download result, tolerant
    of either column orientation; for a flat single-ticker frame, return it as-is."""
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        if sym in cols.get_level_values(0):
            return raw[sym]
        if sym in cols.get_level_values(1):
            return raw.xs(sym, axis=1, level=1)
        return None
    return raw


def get_many_prices(tickers: List[str], lookback: str = "2y", force: bool = False,
                    chunk: int = 100, max_workers: int = 6,  # max_workers kept for API compat
                    progress: Optional[Callable[[int, int, str], None]] = None
                    ) -> Dict[str, pd.DataFrame]:
    """Fetch many tickers SAFELY. Concurrent single-ticker ``yf.download`` calls race on
    yfinance's shared global state (returning the wrong ticker's data), so we instead
    use yfinance's own batch download (``group_by='ticker'``, internal threading) in
    chunks, splitting per ticker. Cache hits are read first; only misses are fetched.
    """
    ensure_dirs()
    syms = [normalize(t) for t in tickers]
    out: Dict[str, pd.DataFrame] = {}
    to_fetch: List[str] = []
    for sym in syms:
        path = PRICES_DIR / f"{sym}.parquet"
        if not force and age_days(path) <= 1.0:
            try:
                out[sym] = pd.read_parquet(path)
                continue
            except Exception:
                pass
        to_fetch.append(sym)

    total = len(syms)
    done = len(out)
    if to_fetch:
        import yfinance as yf
        for i in range(0, len(to_fetch), chunk):
            part = to_fetch[i:i + chunk]
            try:
                raw = yf.download(part, period=lookback, interval="1d",
                                  auto_adjust=True, group_by="ticker",
                                  threads=True, progress=False)
            except Exception:
                raw = None
            for sym in part:
                df = None
                if raw is not None and len(raw):
                    sub = _extract_ticker(raw, sym)
                    df = _clean_prices(sub) if sub is not None else None
                if df is not None and len(df):
                    out[sym] = df
                    try:
                        df.to_parquet(PRICES_DIR / f"{sym}.parquet")
                    except Exception:
                        pass
                done += 1
                if progress:
                    progress(done, total, sym)
    return out


def get_spy(force: bool = False) -> Optional[pd.DataFrame]:
    return get_prices("SPY", force=force)


# --------------------------------------------------------------------------- #
# Fundamentals (current quarters, from yfinance — no API key)
# --------------------------------------------------------------------------- #
def _row(df: Optional[pd.DataFrame], *names: str) -> Optional[pd.Series]:
    """First matching row from a yfinance statement frame, as an ascending-by-date
    Series (statement columns are quarter-end dates, newest first)."""
    if df is None or getattr(df, "empty", True):
        return None
    for name in names:
        if name in df.index:
            s = df.loc[name].dropna()
            if not s.empty:
                s = s.copy()
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return None


def _pct(curr, prev) -> Optional[float]:
    if curr is None or prev is None or pd.isna(curr) or pd.isna(prev) or prev == 0:
        return None
    return (curr - prev) / abs(prev) * 100.0


def _yoy(s: Optional[pd.Series], lag: int = 4) -> Optional[float]:
    if s is None or len(s) < lag + 1:
        return None
    return _pct(s.iloc[-1], s.iloc[-1 - lag])


def _yoy_prev(s: Optional[pd.Series], lag: int = 4) -> Optional[float]:
    if s is None or len(s) < lag + 2:
        return None
    return _pct(s.iloc[-2], s.iloc[-2 - lag])


def _qoq(s: Optional[pd.Series]) -> Optional[float]:
    if s is None or len(s) < 2:
        return None
    return _pct(s.iloc[-1], s.iloc[-2])


def _margin(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[float]:
    if num is None or den is None or len(num) == 0 or len(den) == 0:
        return None
    d = den.iloc[-1]
    if d == 0 or pd.isna(d) or pd.isna(num.iloc[-1]):
        return None
    return num.iloc[-1] / d * 100.0


def _margin_trend(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[float]:
    """Change in operating margin vs the prior quarter (pp); + = expanding."""
    if num is None or den is None or len(num) < 2 or len(den) < 2:
        return None
    if den.iloc[-1] == 0 or den.iloc[-2] == 0:
        return None
    cur = num.iloc[-1] / den.iloc[-1] * 100.0
    prev = num.iloc[-2] / den.iloc[-2] * 100.0
    if pd.isna(cur) or pd.isna(prev):
        return None
    return cur - prev


def _jsonable(v) -> Optional[float]:
    """Coerce numpy/pandas scalars to plain float (or None) so the dict is JSON-safe."""
    if v is None or (isinstance(v, float) and v != v):   # None or NaN
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fetch_fundamentals(sym: str) -> Optional[dict]:
    """Quarterly growth/margin metrics from yfinance (no cache). Keys are None when a
    metric can't be computed (yfinance often exposes only ~4 quarters, so YoY may be
    absent; QoQ is the reliable fallback). Returns None if nothing could be fetched."""
    try:
        import yfinance as yf
        tk = yf.Ticker(sym)
        fin = tk.quarterly_financials
        bs = tk.quarterly_balance_sheet
    except Exception:
        return None
    if (fin is None or getattr(fin, "empty", True)) and \
       (bs is None or getattr(bs, "empty", True)):
        return None

    rev = _row(fin, "Total Revenue", "TotalRevenue")
    gp = _row(fin, "Gross Profit", "GrossProfit")
    oi = _row(fin, "Operating Income", "OperatingIncome", "EBIT")
    eps = _row(fin, "Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS")
    inv = _row(bs, "Inventory")

    out = {
        "revenue_yoy": _yoy(rev), "revenue_qoq": _qoq(rev),
        "revenue_yoy_prev": _yoy_prev(rev),
        "eps_yoy": _yoy(eps), "eps_qoq": _qoq(eps),
        "eps_yoy_prev": _yoy_prev(eps),
        "gross_margin": _margin(gp, rev),
        "operating_margin": _margin(oi, rev),
        "margin_trend": _margin_trend(oi, rev),
        "inventory_qoq": _qoq(inv),
    }
    return {k: _jsonable(v) for k, v in out.items()}


def get_fundamentals(ticker: str, force: bool = False,
                     max_age_days: float = 7.0) -> Optional[dict]:
    """Cached quarterly fundamentals. Fundamentals change only quarterly, so a JSON
    cache per ticker (weekly staleness by default) keeps repeat scans near-instant.
    Falls back to a stale cache if a live fetch fails."""
    ensure_dirs()
    sym = normalize(ticker)
    path = FUNDAMENTALS_DIR / f"{sym}.json"
    if not force and age_days(path) <= max_age_days:
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    out = _fetch_fundamentals(sym)
    if out is not None:
        try:
            path.write_text(json.dumps(out))
        except Exception:
            pass
        return out
    if path.exists():                                  # fetch failed -> stale cache
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None
