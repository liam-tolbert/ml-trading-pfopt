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
import time
from typing import Callable, Dict, List, Optional

import pandas as pd

from .cache import (CACHE_DIR, FUNDAMENTALS_DIR, PRICES_DIR, TICKERS_TXT,
                    age_days, ensure_dirs)

# A stable, maintained constituents CSV (no API key); Wikipedia is the fallback.
SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/"
    "data/constituents.csv"
)
# NASDAQ Trader symbol directories (HTTPS mirror of the upstream ftp:// endpoints, which
# are commonly blocked). Together these list every NASDAQ + NYSE/AMEX security; we filter
# them down to clean common stock — the ~3-4.5k "full US" universe. Re-implements
# minervini_screener.data.universe_fetcher (which we can't import: its package __init__
# eager-loads SQLAlchemy, absent from this env — see data_feed's module docstring).
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
US_COMMON_CSV = "us_common_universe.csv"
# Names that betray a fund/ETF/note rather than an operating company.
_ETF_NAME_RE = r"ETF|FUND|TRUST|INDEX|PORTFOLIO|SHARES|NOTES|BOND|TREASURY"
# Relative Close divergence over the overlap window that signals yfinance re-adjusted the
# whole history (a split/dividend) — i.e. an incremental append would splice two different
# adjustment bases, so we re-baseline with a full refetch instead.
SPLIT_TOL = 0.005
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
        return _get_us_common(force=force, max_age_days=max_age_days)
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


def _get_us_common(force: bool, max_age_days: float) -> List[str]:
    """Broad US common-stock universe (~3-4.5k), cached like sp500. Listings churn, so the
    cache is capped at 1 day (matching upstream). Fallbacks never trigger a hidden network
    call: stale full_us cache -> the sp500 cache (offline read) -> tickers.txt."""
    path = CACHE_DIR / US_COMMON_CSV
    max_age_days = min(max_age_days, 1.0)
    if not force and path.exists() and age_days(path) <= max_age_days:
        cached = _syms_from_csv(path)
        if cached:
            return cached
    syms = _fetch_us_common_nasdaqtrader()
    if syms:
        ensure_dirs()
        pd.Series(sorted(set(syms)), name="Symbol").to_csv(path, index=False)
        return [normalize(s) for s in sorted(set(syms))]
    if path.exists():                              # stale-but-present beats nothing
        cached = _syms_from_csv(path)
        if cached:
            return cached
    sp = CACHE_DIR / "sp500_constituents.csv"      # narrower offline fallback (no network)
    if sp.exists():
        cached = _syms_from_csv(sp)
        if cached:
            return cached
    return _read_tickers_txt()                      # last-ditch offline fallback


def _fetch_us_common_nasdaqtrader() -> Optional[List[str]]:
    """NASDAQ + NYSE/AMEX common stocks from the nasdaqtrader SymDir over HTTPS.
    Returns raw (pre-normalize) symbols, or None if nothing could be fetched."""
    try:
        import requests
        frames = []
        for url, sym_col in ((NASDAQ_LISTED_URL, "Symbol"),
                             (OTHER_LISTED_URL, "ACT Symbol")):
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            # Test Issue == 'N' drops test issues AND the trailing "File Creation Time…"
            # footer row (its Test Issue field is NaN).
            df = df[df["Test Issue"] == "N"]
            if "ETF" in df.columns:                 # exchange ETF flag: stronger than a name guess
                df = df[df["ETF"] != "Y"]
            df = df.rename(columns={sym_col: "symbol", "Security Name": "name"})
            frames.append(df[["symbol", "name"]])
        allsym = (pd.concat(frames, ignore_index=True)
                    .dropna(subset=["symbol"])
                    .drop_duplicates(subset=["symbol"]))
        syms = sorted(_filter_us_symbols(allsym)["symbol"].astype(str).tolist())
        return syms or None
    except Exception:
        return None


def _filter_us_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only clean common-stock tickers. Faithful to the upstream Minervini filter (a
    single pass avoids its boolean-alignment bug): drop symbols with $ ^ . - ; drop
    warrant/right/unit suffixes; keep 1-5 uppercase letters; drop fund/ETF/note names.
    Note this also omits dotted class shares (BRK.B) and 1-letter-suffix names (U) — the
    same known limitation as upstream."""
    if df.empty:
        return df
    sym = df["symbol"].astype(str)
    df = df[~sym.str.contains(r"[\$\^\.\-]", regex=True, na=False)]
    df = df[~df["symbol"].astype(str).str.contains(r"(?:WS|WT|W|R|U)$", regex=True, na=False)]
    df = df[df["symbol"].astype(str).str.match(r"^[A-Z]{1,5}$", na=False)]
    name_u = df["name"].astype(str).str.upper()
    df = df[~name_u.str.contains(_ETF_NAME_RE, regex=True, na=False)]
    return df


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


def _download_batch(yf, part, retries: int, pause: float, **dl):
    """yf.download with bounded retry + exponential backoff on an empty/failed result.
    ``**dl`` carries either ``period=`` (full history) or ``start=/end=`` (incremental), so
    the one helper serves both the cold and delta fetch paths."""
    raw = None
    for attempt in range(retries + 1):
        try:
            raw = yf.download(part, **dl)
        except Exception:
            raw = None
        if raw is not None and len(raw):
            return raw
        if attempt < retries:
            time.sleep(pause * (2 ** attempt))
    return raw


def _lookback_to_offset(lookback: str):
    """'2y'/'18mo'/'90d' -> a pandas offset for trimming the incremental cache window."""
    try:
        if lookback.endswith("mo"):
            return pd.DateOffset(months=int(lookback[:-2]))
        if lookback.endswith("y"):
            return pd.DateOffset(years=int(lookback[:-1]))
        if lookback.endswith("wk"):
            return pd.Timedelta(weeks=int(lookback[:-2]))
        if lookback.endswith("d"):
            return pd.Timedelta(days=int(lookback[:-1]))
    except Exception:
        pass
    return pd.DateOffset(years=2)


def _merge_incremental(cached: pd.DataFrame, new: Optional[pd.DataFrame],
                       lookback: str) -> tuple:
    """Append only genuinely-new bars to ``cached``. Returns ``(df, needs_full)``.

    ``needs_full=True`` means the overlapping days diverged beyond ``SPLIT_TOL`` — yfinance
    re-adjusted the history (split/dividend), so appending would splice two adjustment
    bases; the caller should re-baseline with a full refetch. When ``new`` is empty/None
    (nothing new, or a failed fetch) we return the cache untouched."""
    if new is None or not len(new):
        return cached, False
    common = cached.index.intersection(new.index)
    if len(common):
        c = cached.loc[common, "Close"].astype(float)
        n = new.loc[common, "Close"].astype(float)
        rel = ((c - n).abs() / c.abs().replace(0, pd.NA)).dropna()
        if len(rel) and float(rel.max()) > SPLIT_TOL:
            return cached, True
    merged = pd.concat([cached, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    cutoff = merged.index[-1] - _lookback_to_offset(lookback)
    return merged.loc[merged.index >= cutoff], False


def get_prices(ticker: str, lookback: str = "2y", force: bool = False,
               max_age_days: float = 1.0, incremental: bool = True,
               overlap_days: int = 5, max_gap_days: int = 10) -> Optional[pd.DataFrame]:
    """Daily OHLCV (auto-adjusted) for one name, parquet-cached and age-refreshed. When a
    recent cache exists, only the bars since its last date are fetched and appended."""
    ensure_dirs()
    sym = normalize(ticker)
    path = PRICES_DIR / f"{sym}.parquet"
    if not force and age_days(path) <= max_age_days:
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    import yfinance as yf

    # Incremental: fetch only bars since the last cached date (small recent gap only).
    if not force and incremental and path.exists():
        try:
            cached = pd.read_parquet(path)
        except Exception:
            cached = None
        if cached is not None and len(cached):
            last = pd.Timestamp(cached.index[-1]).normalize()
            gap = (pd.Timestamp.today().normalize() - last).days
            if 0 <= gap <= max_gap_days:
                start = (last - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
                try:
                    raw = yf.download(sym, start=start, interval="1d", auto_adjust=True,
                                      progress=False, threads=False)
                except Exception:
                    raw = None
                new = _clean_prices(raw)
                merged, needs_full = _merge_incremental(cached, new, lookback)
                if not needs_full:
                    if new is not None and len(new):    # only persist when we got data
                        try:
                            merged.to_parquet(path)
                        except Exception:
                            pass
                    return merged
                # needs_full -> fall through to a full re-baseline fetch below

    try:
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
                    max_age_days: float = 1.0,
                    chunk: int = 100, max_workers: int = 6,  # max_workers kept for API compat
                    pause: float = 0.5, retries: int = 2, incremental: bool = True,
                    overlap_days: int = 5, max_gap_days: int = 10,
                    progress: Optional[Callable[[int, int, str], None]] = None
                    ) -> Dict[str, pd.DataFrame]:
    """Fetch many tickers SAFELY. Concurrent single-ticker ``yf.download`` calls race on
    yfinance's shared global state (returning the wrong ticker's data), so we instead
    use yfinance's own batch download (``group_by='ticker'``, internal threading) in
    chunks, with inter-batch pauses + retry/backoff so a large universe doesn't get
    rate-limited into silently-dropped batches.

    Caching is incremental: a fresh (< ``max_age_days``) parquet is used as-is; a cache
    with a small recent gap is topped up with only the bars since its last date (one
    shared ``start`` across the batch); everything else (no cache, or a gap >
    ``max_gap_days``) gets a full ``lookback`` refetch, which also re-baselines
    auto-adjusted history. ``max_age_days=0`` sends every cached name through the cheap
    incremental top-up (the nightly EOD path — gets the finalized close without the full
    2y refetch that ``force=True`` would do); the 1.0 default preserves the old behavior.
    """
    ensure_dirs()
    syms = [normalize(t) for t in tickers]
    out: Dict[str, pd.DataFrame] = {}
    full_fetch: List[str] = []                 # need full period=lookback (cold / re-baseline)
    incr: Dict[str, tuple] = {}                # sym -> (cached_df, last_date)
    today = pd.Timestamp.today().normalize()
    for sym in syms:
        path = PRICES_DIR / f"{sym}.parquet"
        if not force and age_days(path) <= max_age_days:
            try:
                out[sym] = pd.read_parquet(path)
                continue
            except Exception:
                pass
        cached = None
        if not force and incremental and path.exists():
            try:
                cached = pd.read_parquet(path)
            except Exception:
                cached = None
        if cached is not None and len(cached):
            last = pd.Timestamp(cached.index[-1]).normalize()
            gap = (today - last).days
            if 0 <= gap <= max_gap_days:
                incr[sym] = (cached, last)
            else:
                full_fetch.append(sym)         # too-stale -> full refetch (re-baseline)
        else:
            full_fetch.append(sym)

    total = len(syms)
    done = len(out)

    def _emit(sym: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, total, sym)

    if full_fetch or incr:
        import yfinance as yf

        # ---- Incremental: only bars since each cache's last date (shared start) ----
        if incr:
            start = (min(last for _c, last in incr.values())
                     - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
            incr_syms = list(incr)
            for i in range(0, len(incr_syms), chunk):
                part = incr_syms[i:i + chunk]
                raw = _download_batch(yf, part, retries, pause, start=start,
                                      interval="1d", auto_adjust=True,
                                      group_by="ticker", threads=True, progress=False)
                for sym in part:
                    cached, _last = incr[sym]
                    new = None
                    if raw is not None and len(raw):
                        sub = _extract_ticker(raw, sym)
                        new = _clean_prices(sub) if sub is not None else None
                    merged, needs_full = _merge_incremental(cached, new, lookback)
                    if needs_full:
                        full_fetch.append(sym)        # re-baseline in the full pass below
                        continue
                    out[sym] = merged
                    if new is not None and len(new):  # only persist when we got new data
                        try:
                            merged.to_parquet(PRICES_DIR / f"{sym}.parquet")
                        except Exception:
                            pass
                    _emit(sym)
                if i + chunk < len(incr_syms):
                    time.sleep(pause)

        # ---- Full: cold caches + incremental re-baselines ----
        if full_fetch:
            for i in range(0, len(full_fetch), chunk):
                part = full_fetch[i:i + chunk]
                raw = _download_batch(yf, part, retries, pause, period=lookback,
                                      interval="1d", auto_adjust=True,
                                      group_by="ticker", threads=True, progress=False)
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
                    _emit(sym)
                if i + chunk < len(full_fetch):
                    time.sleep(pause)
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


def _next_earnings_date(tk) -> Optional[str]:
    """Next scheduled earnings date as ``'YYYY-MM-DD'``, or None if unknown.

    Reads ``yf.Ticker.calendar`` — a dict on modern yfinance
    (``{'Earnings Date': [date, ...]}``, often a 2-day window; we take the earliest)
    and a DataFrame with an ``'Earnings Date'`` row on older versions. Yahoo sometimes
    lists only the LAST report until the next one is scheduled, so the returned date
    can be in the past — callers surface that as "just reported" rather than hiding it.
    """
    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date")
        elif cal is not None and not getattr(cal, "empty", True) \
                and "Earnings Date" in getattr(cal, "index", []):
            dates = list(cal.loc["Earnings Date"])
        else:
            dates = None
        if dates is None or isinstance(dates, str):
            dates = [dates] if dates else []
        elif not isinstance(dates, (list, tuple)):
            try:
                dates = list(dates)          # a Series/array of dates
            except TypeError:
                dates = [dates]              # a single scalar date
        parsed = sorted(pd.Timestamp(d) for d in dates if d is not None and not pd.isna(d))
        return parsed[0].strftime("%Y-%m-%d") if parsed else None
    except Exception:
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
    out = {k: _jsonable(v) for k, v in out.items()}
    # A date string, so added AFTER the float coercion (which would None it out).
    out["next_earnings"] = _next_earnings_date(tk)
    return out


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
            cached = json.loads(path.read_text())
            # Schema upgrade: caches written before the earnings-date field lack the
            # key entirely — refetch those once so the column fills without waiting
            # out the weekly staleness. (A present-but-None value stays cached.)
            if "next_earnings" in cached:
                return cached
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
