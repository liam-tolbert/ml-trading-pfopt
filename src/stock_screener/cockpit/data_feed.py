"""Market data for the cockpit: universes, daily prices and fundamentals.

Built on yfinance, requests and the SEC EDGAR API. The vendored screener has no data
layer of its own (see ``minervini_screener/PROVENANCE.md``).

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
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from .cache import (CACHE_DIR, EDGAR_DIR, FUNDAMENTALS_DIR, PRICES_DIR,
                    age_days, ensure_dirs)
from .runlog import get_logger

# get_many_prices MUST log one summary line per call, never one per ticker. A full-US
# sweep touches ~4,200 names, and per-name records would cost more writes than the price
# cache itself. Failed symbols are named, up to _FAILED_SAMPLE: they are what to act on.
_LOG = get_logger("prices")
_FAILED_SAMPLE = 12

# A maintained constituents CSV with no API key. Wikipedia is the fallback.
SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/"
    "data/constituents.csv"
)
# NASDAQ Trader symbol directories over HTTPS; the ftp:// endpoints are often blocked.
# Together they list every NASDAQ and NYSE/AMEX security. Filtered to common stock, they
# give the ~3-4.5k "full US" universe.
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
US_COMMON_CSV = "us_common_universe.csv"
# Names that betray a fund/ETF/note rather than an operating company.
_ETF_NAME_RE = r"ETF|FUND|TRUST|INDEX|PORTFOLIO|SHARES|NOTES|BOND|TREASURY"
# Max relative Close divergence on overlapping settled bars. Above it, yfinance has
# re-adjusted the history for a split or dividend. Appending would splice two adjustment
# bases, so the name is refetched in full.
SPLIT_TOL = 0.005
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]
# Pool size for the cache-read pre-pass. It does local parquet reads only, and pyarrow
# releases the GIL: ~1.7x over serial on a 4,200-name warm scan. yf.download stays serial.
_CACHE_READ_WORKERS = 16
# Serializes yf.download across the process. yfinance resets a module-global result dict
# at the top of every download() and spin-waits on its length with no timeout. Two
# concurrent calls wipe each other's frames, and the loser can hang forever.
# scan_worker's _SCAN_SERIAL orders scans only against each other. Every fetch in the
# process MUST take this lock: freshen_prices, the Check-triggers button and the Positions
# page run beside the background scan thread. It is held per attempt and released during
# retry backoff, so a waiting fetch blocks for about one attempt, not a retry cycle.
# Another process, such as the scheduled refresh job, has its own yfinance globals; shared
# cache files are protected by _atomic_to_parquet.
_YF_LOCK = threading.Lock()


def network_busy() -> bool:
    """True while a thread holds ``_YF_LOCK`` inside ``yf.download``.

    Interactive pages check it and read cache-only (``allow_network=False``) rather than
    queue behind a multi-minute bulk sweep. Selling a position MUST NOT wait on a
    4,000-name download."""
    return _YF_LOCK.locked()


# --------------------------------------------------------------------------- #
# Symbols / universe
# --------------------------------------------------------------------------- #
def normalize(ticker: str) -> str:
    """yfinance uses '-' where exchanges use '.' (BRK.B -> BRK-B)."""
    return str(ticker).strip().upper().replace(".", "-")


def get_universe(name: str, force: bool = False,
                 max_age_days: float = 7.0) -> List[str]:
    """Normalized symbols for ``name``, ``'full_us'`` or ``'sp500'``. Empty when neither a
    fetch nor a cache yields any. Raises ValueError for any other name.

    full_us is the universe the cockpit screens; sp500 is its offline fallback. ``name``
    MUST NOT get a default: a bare call would silently screen a different universe from
    the scheduled jobs."""
    if name == "sp500":
        return _get_sp500(force=force, max_age_days=max_age_days)
    if name == "full_us":
        return _get_us_common(force=force, max_age_days=max_age_days)
    raise ValueError(f"unknown universe: {name!r}")


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
    return []                               # empty is LOUD: callers report "no tickers"
                                            # rather than screening a silent subset


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
    """US common stocks (~3-4.5k), cached like sp500. Listings churn, so the cache age is
    capped at 1 day. The fallbacks MUST NOT touch the network: the stale full_us cache,
    then the sp500 cache, then empty."""
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
    return []                                      # empty is LOUD; see _get_sp500


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
            # Test Issue == 'N' also drops the trailing "File Creation Time" footer row:
            # its Test Issue is NaN.
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
    """The rows of ``df`` that look like common stock: 1-5 uppercase letters, none of
    $ ^ . -, not a warrant/right/unit, not a fund/ETF/note name.

    Warrants, rights and units are matched by the SymDir shape: a 4-letter base plus W, R
    or U. A plain ``(?:W|R|U)$`` would drop ordinary names such as PLTR, SNOW, UBER and U.
    A 3-letter base plus W still passes; such a warrant just fails the trend template.
    Dotted class shares (BRK.B, BF.B) are dropped."""
    if df.empty:
        return df
    sym = df["symbol"].astype(str)
    df = df[~sym.str.contains(r"[\$\^\.\-]", regex=True, na=False)]
    df = df[~df["symbol"].astype(str).str.match(r"^[A-Z]{4}[WRU]$", na=False)]
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
    """``yf.download(part, **dl)``, retried up to ``retries`` times on an empty or failed
    result, with backoff doubling from ``pause`` seconds. Returns the first non-empty
    frame, else the last attempt's result: an empty frame, or None if it raised.

    ``**dl`` carries ``period=`` for a full fetch or ``start=`` for a top-up."""
    raw = None
    for attempt in range(retries + 1):
        try:
            with _YF_LOCK:              # per-attempt: released during backoff sleeps
                raw = yf.download(part, **dl)
        except Exception:
            raw = None
        if raw is not None and len(raw):
            return raw
        if attempt < retries:
            time.sleep(pause * (2 ** attempt))
    return raw


def _lookback_to_offset(lookback: str):
    """'2y', '18mo', '6wk' or '90d' as a pandas offset, for trimming the merged cache.
    Anything unparseable reads as 2 years."""
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
    """Merge ``new`` bars into ``cached``, trimmed to ``lookback``. Returns
    ``(df, needs_full)``.

    Where the two overlap, ``new`` wins. ``needs_full`` is True, with ``cached`` returned
    as is, when settled overlap diverges beyond ``SPLIT_TOL``: yfinance has re-adjusted
    the history, and the caller SHOULD refetch it in full. An empty or None ``new`` returns
    ``(cached, False)``."""
    if new is None or not len(new):
        return cached, False
    common = cached.index.intersection(new.index)
    # Today's bar is provisional: its close moves between intraday fetches. Only settled
    # overlap is compared, or every intraday refresh would read as a re-adjustment.
    common = common[common < pd.Timestamp.today().normalize()]
    # The cache's last bar can be provisional too: an intraday scan persists a mid-session
    # close. Compared next day with the settled close, it would read as a split and force a
    # full refetch of every name that moved. The merge overwrites that bar anyway, so it is
    # left out when older overlap exists; a real re-adjustment rescales those bars too. As
    # the only overlap it stays in: some evidence beats none, and a rare false full refetch
    # on a near-empty cache is cheap.
    if len(common) > 1:
        common = common[common < cached.index[-1]]
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
    """Daily auto-adjusted OHLCV for one ticker, or None when there is neither data nor a
    cache. A wrapper over :func:`get_many_prices`, so it shares that function's cache,
    top-up, re-baseline, retry and stale-cache fallback."""
    sym = normalize(ticker)
    return get_many_prices([sym], lookback=lookback, force=force,
                           max_age_days=max_age_days, incremental=incremental,
                           overlap_days=overlap_days, max_gap_days=max_gap_days).get(sym)


def _fmt_us(ts) -> str:
    """M/D/YYYY (no leading zeros) — the per-ticker download-log date format."""
    ts = pd.Timestamp(ts)
    return f"{ts.month}/{ts.day}/{ts.year}"


def _incr_detail(last, today) -> str:
    """Progress text for a top-up: the missing-day range, e.g. ``'7/20/2026 - 7/22/2026'``.
    Just today's date when ``last`` is today or yesterday. The overlap days refetched for
    split detection are left out on purpose."""
    start = pd.Timestamp(last).normalize() + pd.Timedelta(days=1)
    today = pd.Timestamp(today).normalize()
    if start >= today:
        return _fmt_us(today)
    return f"{_fmt_us(start)} - {_fmt_us(today)}"


def _extract_ticker(raw: pd.DataFrame, sym: str) -> Optional[pd.DataFrame]:
    """``sym``'s sub-frame from a ``yf.download`` result, with the ticker on either column
    level. A flat frame is returned as is. None when a MultiIndex frame lacks ``sym``."""
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        if sym in cols.get_level_values(0):
            return raw[sym]
        if sym in cols.get_level_values(1):
            return raw.xs(sym, axis=1, level=1)
        return None
    return raw


_TRIGGERS = None                    # lazily-cached triggers MODULE (see _cache_settled)


def _cache_settled(path) -> bool:
    """True when no market session has run since ``path`` was written: evenings, weekends,
    pre-open. Such a cache holds the settled close, whatever its wall-clock age. The
    calendar is ``triggers.no_session_since``. Never raises; a missing file or any error
    reads False.

    The triggers module is cached, since this runs once per name in the pre-pass.
    ``no_session_since`` MUST be looked up per call: tests patch it on the module."""
    global _TRIGGERS
    try:
        if _TRIGGERS is None:
            from . import triggers as _t
            _TRIGGERS = _t
        return _TRIGGERS.no_session_since(path.stat().st_mtime)
    except Exception:
        return False


def _frame_settled_current(last_bar_date) -> bool:
    """``triggers.frame_settled_current(last_bar_date)``, with the module cached as in
    :func:`_cache_settled`. Never raises; errors read False."""
    global _TRIGGERS
    try:
        if _TRIGGERS is None:
            from . import triggers as _t
            _TRIGGERS = _t
        return _TRIGGERS.frame_settled_current(last_bar_date)
    except Exception:
        return False


def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` through a temp file and ``os.replace``, so a reader never
    sees a torn file. The app and the refresh jobs are separate processes that share these
    files, and the pre-pass treats a torn parquet as missing: a full refetch.

    Raises on failure and leaves the old file intact. On Windows the replace can raise
    PermissionError while another process holds the file open."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _log_fetch(kind: str, requested: int, cached: int, topup: int, full: int,
               wrote: int, failed: List[str], t0: float) -> None:
    """Log one summary line for a sweep, plus a warning naming up to ``_FAILED_SAMPLE``
    failed symbols. The line MUST stay ASCII, for journald and grep.

    ``cached`` names never touched the network. An all-cached line records that the box
    chose not to download. Parquet mtimes can't show that: an unchanged mtime looks the
    same as a sweep that never ran."""
    _LOG.info("%s: %d requested  cached %d  topup %d  full %d  wrote %d  failed %d  %.1fs",
              kind, requested, cached, topup, full, wrote, len(failed), time.time() - t0)
    if failed:
        more = (f" (+{len(failed) - _FAILED_SAMPLE} more)"
                if len(failed) > _FAILED_SAMPLE else "")
        _LOG.warning("%s: no data for %d name(s): %s%s", kind, len(failed),
                     ", ".join(failed[:_FAILED_SAMPLE]), more)


def get_many_prices(tickers: List[str], lookback: str = "2y", force: bool = False,
                    max_age_days: float = 1.0,
                    chunk: int = 100,
                    pause: float = 0.5, retries: int = 2, incremental: bool = True,
                    overlap_days: int = 5, max_gap_days: int = 10,
                    progress: Optional[Callable[[int, int, str], None]] = None,
                    allow_network: bool = True
                    ) -> Dict[str, pd.DataFrame]:
    """Daily auto-adjusted OHLCV for ``tickers``, as ``{normalized symbol: DataFrame}``. A
    name with neither data nor a cache is absent.

    Downloads use yfinance's batch call (``group_by='ticker'``) in chunks of ``chunk``,
    with ``pause`` seconds between batches and ``retries`` with backoff, so a large
    universe isn't rate-limited into dropped batches. Concurrent single-ticker
    ``yf.download`` calls race on yfinance's global state and return the wrong ticker's
    data.

    Each name's parquet cache decides its fetch:

    * age within ``max_age_days``: served as is;
    * written with no market session since, and its last bar current: served as is, even
      at ``max_age_days=0``, since no new bar can exist;
    * a gap of at most ``max_gap_days``: topped up from one shared ``start``,
      ``overlap_days`` before the oldest last bar;
    * no cache, a longer gap, ``incremental=False`` or a split/dividend re-adjustment: a
      full ``lookback`` refetch, which re-baselines the adjusted history.

    With ``max_age_days=0`` only the settled-close rule serves a cache as is. The EOD
    sweep uses it to take the settled close with a top-up, not a full refetch. A
    negative ``max_age_days`` also skips the settled-close serve; it is the tests'
    sentinel. ``force=True`` refetches everything. A failed full fetch serves the stale
    parquet when one exists.

    ``allow_network=False`` is cache-only: every name comes from its parquet as is, with
    no fetch and no wait on ``_YF_LOCK``. Interactive pages pair it with
    :func:`network_busy`.

    ``progress(done, total, text)`` is called once per name. Logs one summary line.
    """
    ensure_dirs()
    syms = [normalize(t) for t in tickers]
    out: Dict[str, pd.DataFrame] = {}
    full_fetch: List[str] = []                 # need full period=lookback (cold / re-baseline)
    incr: Dict[str, tuple] = {}                # sym -> (cached_df, last_date)
    today = pd.Timestamp.today().normalize()
    total = len(syms)
    done = 0

    _t0 = time.time()
    _wrote = 0                                 # parquets actually rewritten this sweep
    _rebaselined = 0                           # top-ups that fell back to a full refetch
    _failed: List[str] = []                    # names the provider returned nothing for

    _emit_lock = threading.Lock()

    def _emit(sym: str, detail: str = "") -> None:
        # ``detail`` says what was fetched for this name, for the UI's per-ticker line.
        # Locked: pool workers in the cache pre-pass emit concurrently.
        nonlocal done
        with _emit_lock:
            done += 1
            if progress:
                progress(done, total, f"{sym}: {detail}" if detail else sym)

    def _classify_cached(sym: str):
        """One name's cache verdict: ``('served', df, detail)``, ``('incr', cached_df,
        last_date)`` or ``('full', None, None)``. It runs on pool workers and MUST NOT
        write."""
        path = PRICES_DIR / f"{sym}.parquet"
        if not force and age_days(path) <= max_age_days:
            try:
                return "served", pd.read_parquet(path), "cached (fresh)"
            except Exception:
                pass
        elif not force and max_age_days >= 0 and _cache_settled(path):
            # No session since the write, so the cache is current whatever its age. A
            # negative max_age_days, the tests' sentinel, skips this serve too, so the
            # top-up paths stay reachable.
            try:
                df = pd.read_parquet(path)
                # The mtime dates the file, not its bars. A lagging provider response,
                # persisted by the ungated full fetch, can end before the latest settled
                # session. Such a frame falls through to the top-up.
                if len(df) and _frame_settled_current(df.index[-1]):
                    return "served", df, "cached (settled close)"
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
                return "incr", cached, last
        return "full", None, None              # cold, or too-stale -> full re-baseline

    def _classify_and_emit(sym: str):
        kind, frame, extra = _classify_cached(sym)
        if kind == "served":
            _emit(sym, extra)                  # progress moves while reads stream in
        return kind, frame, extra

    # Reading thousands of warm parquets takes real time; the pool roughly halves it.
    # Served names emit from the workers, so progress order is interleaved; consumers treat
    # the log as an unordered tail. out, incr and full_fetch MUST be built in syms order
    # after the join, so dict order and batch chunking stay deterministic. yf.download
    # below MUST stay serial: concurrent calls race yfinance's shared state.
    if syms:
        with ThreadPoolExecutor(max_workers=min(_CACHE_READ_WORKERS, len(syms))) as _pool:
            verdicts = list(_pool.map(_classify_and_emit, syms))
        for sym, (kind, frame, extra) in zip(syms, verdicts):
            if kind == "served":
                out[sym] = frame
            elif kind == "incr":
                incr[sym] = (frame, extra)
            else:
                full_fetch.append(sym)

    _served0, _incr0, _full0 = len(out), len(incr), len(full_fetch)

    if not allow_network and (full_fetch or incr):
        # Cache-only. Top-up names already hold their frames from the pre-pass. Other
        # names get their parquet when one exists and are absent otherwise; callers
        # degrade per name.
        for sym in syms:
            if sym in out:
                continue
            if sym in incr:
                out[sym] = incr[sym][0]
                _emit(sym, "cached (network busy)")
                continue
            try:
                out[sym] = pd.read_parquet(PRICES_DIR / f"{sym}.parquet")
                _emit(sym, "cached (network busy)")
            except Exception:
                _emit(sym, "no cache (network busy)")
        _log_fetch("cache-only", total, len(out), 0, 0, 0, [], _t0)
        return out

    if full_fetch or incr:
        import yfinance as yf

        # ---- Incremental top-up, one shared start ----
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
                        _rebaselined += 1
                        continue
                    out[sym] = merged
                    # Persist only when the fetch reached the cache's newest bar. An
                    # overlap-only response (provider lag) MUST NOT re-stamp the mtime.
                    # A rewrite after the close would arm the settled-close serve on a
                    # frame missing the settled bar, and nothing would refetch it all
                    # weekend. ``>=``, not ``>``: the ~16:30 settle of today's bar has
                    # max == last and MUST persist. ``index.max()``, not ``[-1]``:
                    # _clean_prices does not sort.
                    if (new is not None and len(new)
                            and pd.Timestamp(new.index.max()).normalize() >= _last):
                        try:
                            _atomic_to_parquet(merged, PRICES_DIR / f"{sym}.parquet")
                            _wrote += 1
                        except Exception:
                            pass
                    _emit(sym, _incr_detail(_last, today))
                if i + chunk < len(incr_syms):
                    time.sleep(pause)

        # ---- Full: cold caches + incremental re-baselines ----
        if full_fetch:
            for i in range(0, len(full_fetch), chunk):
                part = full_fetch[i:i + chunk]
                raw = _download_batch(yf, part, retries, pause, period=lookback,
                                      interval="1d", auto_adjust=True,
                                      group_by="ticker", threads=True, progress=False)
                got: Dict[str, pd.DataFrame] = {}
                for sym in part:
                    if raw is not None and len(raw):
                        sub = _extract_ticker(raw, sym)
                        df = _clean_prices(sub) if sub is not None else None
                        if df is not None and len(df):
                            got[sym] = df
                # One retry for the names the batch missed. _download_batch returns once
                # any rows exist, so its own retry never covers a partial failure. An
                # all-empty batch was already retried there.
                missing = [s for s in part if s not in got]
                if missing and retries > 0 and raw is not None and len(raw):
                    raw2 = _download_batch(yf, missing, retries, pause, period=lookback,
                                           interval="1d", auto_adjust=True,
                                           group_by="ticker", threads=True, progress=False)
                    if raw2 is not None and len(raw2):
                        for sym in missing:
                            sub = _extract_ticker(raw2, sym)
                            df = _clean_prices(sub) if sub is not None else None
                            if df is not None and len(df):
                                got[sym] = df
                for sym in part:
                    df = got.get(sym)
                    if df is not None:
                        out[sym] = df
                        try:
                            _atomic_to_parquet(df, PRICES_DIR / f"{sym}.parquet")
                            _wrote += 1
                        except Exception:
                            pass
                        _emit(sym, f"full history ({lookback})")
                        continue
                    _failed.append(sym)       # counted even when a stale cache is served
                                              # below: the provider returned nothing
                    # Serve the stale parquet when one exists (long-gap and re-baseline
                    # names have one) rather than drop the name. It MUST NOT be
                    # re-persisted: that would stamp known-stale data as fresh.
                    path = PRICES_DIR / f"{sym}.parquet"
                    if path.exists():
                        try:
                            out[sym] = pd.read_parquet(path)
                            _emit(sym, f"full history ({lookback}) FAILED — "
                                       "stale cache served")
                            continue
                        except Exception:
                            pass
                    _emit(sym, f"full history ({lookback}) FAILED (no data)")
                if i + chunk < len(full_fetch):
                    time.sleep(pause)
    _log_fetch("sweep", total, _served0, _incr0 - _rebaselined, _full0 + _rebaselined,
               _wrote, _failed, _t0)
    return out


def get_spy(force: bool = False, max_age_days: float = 1.0) -> Optional[pd.DataFrame]:
    return get_prices("SPY", force=force, max_age_days=max_age_days)


# --------------------------------------------------------------------------- #
# Fundamentals (current quarters, from yfinance — no API key)
# --------------------------------------------------------------------------- #
def _row(df: Optional[pd.DataFrame], *names: str) -> Optional[pd.Series]:
    """The first of ``names`` with data in a yfinance statement frame, as a Series
    ascending by date. Statement columns are quarter-end dates, newest first. None when
    none has data."""
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


def _yoy_at(s: Optional[pd.Series], back: int = 0) -> Optional[float]:
    """YoY % for the quarter ``back`` steps before the latest (0 = latest), against the
    most recent entry 330-400 days earlier. Matching by date, as ``_edgar_yoy_series``
    does, keeps a missing or extra quarter from shifting a fixed 4-step lag. None when
    ``s`` is too short or no such entry exists."""
    if s is None or back < 0 or len(s) < back + 2:
        return None
    i = len(s) - 1 - back                        # absolute position of the anchor quarter
    end = s.index[i]
    for k in range(i - 1, -1, -1):
        if 330 <= (end - s.index[k]).days <= 400:
            return _pct(s.iloc[i], s.iloc[k])
    return None


def _yoy(s: Optional[pd.Series], lag: int = 4) -> Optional[float]:
    return _yoy_at(s, 0)


def _yoy_prev(s: Optional[pd.Series], lag: int = 4) -> Optional[float]:
    return _yoy_at(s, 1)


def _qoq(s: Optional[pd.Series]) -> Optional[float]:
    if s is None or len(s) < 2:
        return None
    return _pct(s.iloc[-1], s.iloc[-2])


def _aligned(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[pd.DataFrame]:
    """``num`` and ``den`` on their common quarters, as columns ``n`` and ``d``. None when
    either is missing or no quarter is common.

    A margin MUST NOT divide figures from different quarters. ``_row`` drops NaNs per line
    item, and yfinance can fill Total Revenue for the newest quarter before Gross Profit or
    Operating Income; unaligned, that pairs GP(Q-1) with Rev(Q0)."""
    if num is None or den is None:
        return None
    both = pd.concat([num.rename("n"), den.rename("d")], axis=1).dropna()
    return both if not both.empty else None


def _margin(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[float]:
    a = _aligned(num, den)
    if a is None or a["d"].iloc[-1] == 0:
        return None
    return a["n"].iloc[-1] / a["d"].iloc[-1] * 100.0


def _margin_trend(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[float]:
    """Change in operating margin vs the prior quarter (pp); + = expanding."""
    a = _aligned(num, den)
    if a is None or len(a) < 2 or a["d"].iloc[-1] == 0 or a["d"].iloc[-2] == 0:
        return None
    cur = a["n"].iloc[-1] / a["d"].iloc[-1] * 100.0
    prev = a["n"].iloc[-2] / a["d"].iloc[-2] * 100.0
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
    """Next scheduled earnings date as ``'YYYY-MM-DD'``, or None if unknown. Never raises.

    ``yf.Ticker.calendar`` is a dict on current yfinance, ``{'Earnings Date': [date,
    ...]}``, often a 2-day window; the earliest date is taken. Older versions return a
    DataFrame with an ``'Earnings Date'`` row. Yahoo can list only the last report until
    the next is scheduled, so the date can be in the past. Callers show that as "just
    reported".
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


def _last_earnings_surprise(tk) -> Optional[float]:
    """Latest reported EPS surprise %, from the ``'Surprise(%)'`` column of
    ``yf.Ticker.earnings_dates``. Unreported rows are NaN and drop out. None on any miss;
    never raises."""
    try:
        ed = tk.earnings_dates
        if ed is None or getattr(ed, "empty", True):
            return None
        col = next((c for c in ed.columns if "surprise" in str(c).lower()), None)
        if col is None:
            return None
        s = ed[col].dropna()
        if not len(s):
            return None
        return float(s.sort_index().iloc[-1])          # newest REPORTED quarter
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# SEC EDGAR XBRL backfill
# --------------------------------------------------------------------------- #
# YoY history past yfinance's ~4 quarters, plus annual EPS growth and 3-quarter
# acceleration, which yfinance lacks. The company-facts API needs no key. SEC fair use
# is ~10 req/s with a contact User-Agent.
EDGAR_UA = {"User-Agent": "ml-trading-pfopt cockpit (treblotmail@gmail.com)"}
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# Tags tried in order, as in the repo's Main.ipynb EDGAR pipeline.
EDGAR_REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                      "SalesRevenueNet")
EDGAR_EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")


def _edgar_get_json(url: str) -> Optional[dict]:
    try:
        import requests
        time.sleep(0.12)                               # stay politely under 10 req/s
        r = requests.get(url, headers=EDGAR_UA, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _edgar_cik(sym: str) -> Optional[int]:
    """SEC CIK for ``sym`` from company_tickers.json, cached 30 days. None when the map
    lacks it or can't be fetched. Matches both the normalized dash form and the SEC's
    dot form (BRK-B, BRK.B)."""
    path = EDGAR_DIR / "company_tickers.json"
    data = None
    if age_days(path) <= 30.0:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = None
    if data is None:
        data = _edgar_get_json(EDGAR_TICKERS_URL)
        if data is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass
    wanted = {sym.upper(), sym.upper().replace("-", ".")}
    try:
        for row in data.values():
            if str(row.get("ticker", "")).upper() in wanted:
                return int(row["cik_str"])
    except Exception:
        return None
    return None


def _edgar_series(facts: dict, tags, unit_keys) -> tuple:
    """``(quarterly, annual)`` lists of ``(end_Timestamp, value)``, ascending by period end,
    from the first us-gaap tag with usable data. ``([], [])`` when no tag has any.

    Quarterly means a 60-120 day period, annual 300-400 days; other durations, such as
    10-K YTD, are dropped. A repeated period end keeps the latest ``filed``, so amended
    figures win."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    for tag in tags:
        units = (gaap.get(tag) or {}).get("units") or {}
        entries = None
        for uk in unit_keys:
            if units.get(uk):
                entries = units[uk]
                break
        if not entries:
            continue
        q: dict = {}
        a: dict = {}
        for e in entries:
            try:
                dur = (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days
                end = pd.Timestamp(e["end"])
                val = float(e["val"])
            except Exception:
                continue
            bucket = q if 60 <= dur <= 120 else (a if 300 <= dur <= 400 else None)
            if bucket is None:
                continue
            filed = str(e.get("filed") or "")
            if end not in bucket or filed >= bucket[end][0]:
                bucket[end] = (filed, val)
        if q or a:
            return (sorted((k, v[1]) for k, v in q.items()),
                    sorted((k, v[1]) for k, v in a.items()))
    return [], []


def _edgar_yoy_series(quarterly) -> list:
    """``[(end, yoy_pct)]``: each quarter against the most recent one 330-400 days earlier.
    Matching by date, not a fixed 4-step lag, keeps a missing quarter from shifting the
    comparison. A quarter with no usable match is skipped."""
    out = []
    for i, (end, val) in enumerate(quarterly):
        prior = next((v for e2, v in reversed(quarterly[:i])
                      if 330 <= (end - e2).days <= 400), None)
        g = _pct(val, prior)
        if g is not None:
            out.append((end, g))
    return out


def _edgar_backfill(sym: str) -> Optional[dict]:
    """EDGAR growth metrics for ``sym``, cached 7 days as JSON in ``EDGAR_DIR``.

    Keys: ``revenue_yoy``, ``revenue_yoy_prev``, ``eps_yoy``, ``eps_yoy_prev``,
    ``eps_fy_yoy`` and ``eps_accel_3q`` (EPS YoY rising across the last 3 quarters). A
    key is None when the facts can't support it. With no CIK or a failed fetch, returns
    the stale cache if any, else None; foreign listings and funds have no CIK."""
    ensure_dirs()
    path = EDGAR_DIR / f"{sym}.json"
    if age_days(path) <= 7.0:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    cik = _edgar_cik(sym)
    facts = _edgar_get_json(EDGAR_FACTS_URL.format(cik=cik)) if cik is not None else None
    if facts is None:
        if path.exists():                              # fetch failed -> stale cache
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None
    eps_q, eps_fy = _edgar_series(facts, EDGAR_EPS_TAGS, ("USD/shares",))
    rev_q, _rev_fy = _edgar_series(facts, EDGAR_REVENUE_TAGS, ("USD",))
    eps_g = _edgar_yoy_series(eps_q)
    rev_g = _edgar_yoy_series(rev_q)
    out = {
        "revenue_yoy": _jsonable(rev_g[-1][1]) if rev_g else None,
        "revenue_yoy_prev": _jsonable(rev_g[-2][1]) if len(rev_g) >= 2 else None,
        "eps_yoy": _jsonable(eps_g[-1][1]) if eps_g else None,
        "eps_yoy_prev": _jsonable(eps_g[-2][1]) if len(eps_g) >= 2 else None,
        "eps_fy_yoy": (_jsonable(_pct(eps_fy[-1][1], eps_fy[-2][1]))
                       if len(eps_fy) >= 2 else None),
        "eps_accel_3q": (bool(eps_g[-1][1] > eps_g[-2][1] > eps_g[-3][1])
                         if len(eps_g) >= 3 else None),
    }
    try:
        path.write_text(json.dumps(out), encoding="utf-8")
    except Exception:
        pass
    return out


def _fetch_fundamentals(sym: str) -> Optional[dict]:
    """Quarterly growth and margin metrics from yfinance, uncached, plus ``next_earnings``
    and ``last_surprise_pct``. A key is None when its metric can't be computed. yfinance
    often has only ~4 quarters, so YoY may be missing; QoQ is the reliable fallback. None
    when nothing could be fetched."""
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
    # Added after the float coercion, which would turn the date string into None.
    out["next_earnings"] = _next_earnings_date(tk)
    out["last_surprise_pct"] = _last_earnings_surprise(tk)
    return out


def get_fundamentals(ticker: str, force: bool = False,
                     max_age_days: float = 7.0) -> Optional[dict]:
    """Quarterly fundamentals for ``ticker`` as a dict, from a per-ticker JSON cache up to
    ``max_age_days`` old. A live fetch is backfilled from EDGAR and written to the cache.
    Falls back to a stale cache when the fetch fails; None when there is none."""
    ensure_dirs()
    sym = normalize(ticker)
    path = FUNDAMENTALS_DIR / f"{sym}.json"
    if not force and age_days(path) <= max_age_days:
        try:
            cached = json.loads(path.read_text())
            # A cache without these keys has an older schema and is refetched at once,
            # not after max_age_days. A key present as None is a valid cache.
            if "next_earnings" in cached and "last_surprise_pct" in cached:
                return cached
        except Exception:
            pass
    out = _fetch_fundamentals(sym)
    if out is not None:
        # yfinance values win. EDGAR fills the Nones and adds its own keys (FY EPS
        # growth, 3q accel). Any EDGAR failure leaves the yfinance dict as is.
        try:
            ed = _edgar_backfill(sym)
        except Exception:
            ed = None
        for k, v in (ed or {}).items():
            if out.get(k) is None and v is not None:
                out[k] = v
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
