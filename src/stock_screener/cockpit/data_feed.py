"""Thin yfinance data layer for the cockpit.

All fetching is re-implemented here on yfinance + requests directly. The vendored
package shipped its own data layer, but it was SQLAlchemy-backed, unused, and has
since been deleted (see PROVENANCE.md); only the pure rule modules remain.

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

# One summary line per get_many_prices call — never per ticker. A full-US sweep touches
# ~4,200 names and data/cockpit/ is the box's hot write path; per-name records would cost
# more writes than the price cache they describe. Failures name their symbols (capped),
# because that is the part worth acting on.
_LOG = get_logger("prices")
_FAILED_SAMPLE = 12

# A stable, maintained constituents CSV (no API key); Wikipedia is the fallback.
SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/"
    "data/constituents.csv"
)
# NASDAQ Trader symbol directories (HTTPS mirror of the commonly-blocked ftp:// endpoints).
# Together these list every NASDAQ + NYSE/AMEX security; we filter them down to clean
# common stock — the ~3-4.5k "full US" universe.
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
US_COMMON_CSV = "us_common_universe.csv"
# Names that betray a fund/ETF/note rather than an operating company.
_ETF_NAME_RE = r"ETF|FUND|TRUST|INDEX|PORTFOLIO|SHARES|NOTES|BOND|TREASURY"
# Relative Close divergence over the overlap window signaling yfinance re-adjusted the whole
# history (split/dividend); appending would splice two adjustment bases, so we refetch fully.
SPLIT_TOL = 0.005
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]
# Pool size for the cache-read pre-pass (local parquet reads only — pyarrow releases the
# GIL; measured ~1.7x over serial on a 4,200-name warm scan). yf.download stays serial.
_CACHE_READ_WORKERS = 16
# Process-wide serialization of yf.download itself. yfinance resets a module-global
# result dict at the top of EVERY download() and spin-waits on its length with no
# timeout — two concurrent calls wipe each other's frames, and the loser can hang
# forever. scan_worker's _SCAN_SERIAL only serializes scan-vs-scan; script-thread
# fetches (freshen_prices, the Check-triggers button, the Positions page) run in the
# SAME process as the background scan thread and must share this lock. Held per ATTEMPT
# (inside _download_batch, released during retry backoff) so a waiting fetch is blocked
# for ~one attempt, not a whole retry cycle. Cross-PROCESS overlap (the scheduled
# refresh job) is inherently out of an in-process lock's reach — yfinance globals
# are per-process, and cache-file contention is handled by _atomic_to_parquet.
_YF_LOCK = threading.Lock()


def network_busy() -> bool:
    """True while some thread is inside ``yf.download`` (holding ``_YF_LOCK``).
    Interactive pages check this to serve cache-only (``allow_network=False``) instead
    of queueing a small fetch behind a multi-minute bulk sweep — selling a position
    must never wait on a 4,000-name delta download."""
    return _YF_LOCK.locked()


# --------------------------------------------------------------------------- #
# Symbols / universe
# --------------------------------------------------------------------------- #
def normalize(ticker: str) -> str:
    """yfinance uses '-' where exchanges use '.' (BRK.B -> BRK-B)."""
    return str(ticker).strip().upper().replace(".", "-")


def get_universe(name: str, force: bool = False,
                 max_age_days: float = 7.0) -> List[str]:
    """``name`` is REQUIRED. full_us is the only universe the cockpit screens; sp500
    survives as the offline fallback below. A default here would let a bare call screen
    a different universe than every scheduled job does, silently."""
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
    """Broad US common-stock universe (~3-4.5k), cached like sp500. Listings churn, so the
    cache is capped at 1 day (matching upstream). Fallbacks never trigger a hidden network
    call: stale full_us cache -> the sp500 cache (offline read) -> empty."""
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
    """Keep only clean common-stock tickers: drop symbols with $ ^ . - ; drop
    warrant/right/unit issues; keep 1-5 uppercase letters; drop fund/ETF/note names.

    Warrant/right/unit drop is anchored to the nasdaqtrader SymDir shape: a genuine
    derivative is a 5-char symbol (up-to-4-char base + trailing W/R/U). Matching that shape
    rather than a plain ``(?:W|R|U)$`` avoids dropping ordinary 4-letter names that merely
    end in those letters (PLTR, SNOW, UBER, single-letter U). A short-base warrant (3-char
    base + W) can still slip through — an errant warrant just fails the trend template.
    Dotted class shares (BRK.B/BF.B) remain omitted, same as upstream."""
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
    """yf.download with bounded retry + exponential backoff on an empty/failed result.
    ``**dl`` carries either ``period=`` (full history) or ``start=/end=`` (incremental), so
    the one helper serves both the cold and delta fetch paths."""
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
    # TODAY's bar is PROVISIONAL while the session runs — its close moves between intraday
    # fetches, which is not a split/dividend re-adjustment. Compare only settled (pre-today)
    # overlap, else every intraday refresh re-baselines because the live bar "diverged".
    common = common[common < pd.Timestamp.today().normalize()]
    # The cache's FINAL bar can be provisional too: an intraday scan persists the
    # mid-session close, and on the NEXT day this comparison would read the settled close
    # as a "split" and full-refetch every name that moved after that scan (the 2y avalanche
    # on each new-day open). The merge overwrites that bar with the fetched one anyway, so
    # drop it whenever older overlap days exist — a real re-adjustment rescales those too.
    # With no other overlap it stays in: sole evidence beats none, and a rare false full
    # refetch on a near-empty cache is cheap.
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
    """Daily OHLCV (auto-adjusted) for one name — a thin wrapper over
    :func:`get_many_prices`, which owns the whole cache/incremental/re-baseline pipeline
    (this used to duplicate ~50 lines of it for the one SPY caller; the wrapper also
    inherits the batch path's retry and stale-cache fallback)."""
    sym = normalize(ticker)
    return get_many_prices([sym], lookback=lookback, force=force,
                           max_age_days=max_age_days, incremental=incremental,
                           overlap_days=overlap_days, max_gap_days=max_gap_days).get(sym)


def _fmt_us(ts) -> str:
    """M/D/YYYY (no leading zeros) — the per-ticker download-log date format."""
    ts = pd.Timestamp(ts)
    return f"{ts.month}/{ts.day}/{ts.year}"


def _incr_detail(last, today) -> str:
    """Human description of an incremental top-up: the missing-days range ('7/20/2026 -
    7/22/2026'), or just today's date when the cache already holds today's (provisional)
    bar / is only one day behind. The overlap days re-fetched for split detection are an
    implementation detail and deliberately not shown."""
    start = pd.Timestamp(last).normalize() + pd.Timedelta(days=1)
    today = pd.Timestamp(today).normalize()
    if start >= today:
        return _fmt_us(today)
    return f"{_fmt_us(start)} - {_fmt_us(today)}"


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


_TRIGGERS = None                    # lazily-cached triggers MODULE (see _cache_settled)


def _cache_settled(path) -> bool:
    """True when ``path`` was written with NO market session since (post-close evenings,
    weekends, pre-open) — the cache holds the settled close and is current regardless of
    the wall-clock ``max_age_days`` window. Calendar logic lives in
    ``triggers.no_session_since`` (session clock + early-close single source). Never
    raises; missing file / any error reads as not-settled (the normal gates decide).

    The triggers MODULE is cached after the first call (this runs once per name in the
    cache pre-pass), but ``no_session_since`` is looked up per call — tests patch it as
    a module attribute, and call-time lookup is what honors the patch."""
    global _TRIGGERS
    try:
        if _TRIGGERS is None:
            from . import triggers as _t
            _TRIGGERS = _t
        return _TRIGGERS.no_session_since(path.stat().st_mtime)
    except Exception:
        return False


def _frame_settled_current(last_bar_date) -> bool:
    """``triggers.frame_settled_current`` via the same cached-module pattern as
    ``_cache_settled`` (attribute lookup stays call-time → test patches on the triggers
    module keep working). Errors read as not-current."""
    global _TRIGGERS
    try:
        if _TRIGGERS is None:
            from . import triggers as _t
            _TRIGGERS = _t
        return _TRIGGERS.frame_settled_current(last_bar_date)
    except Exception:
        return False


def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write-then-``os.replace`` so a concurrent reader never sees a torn file. The app
    and the half-hourly refresh job run in SEPARATE processes but share these cache
    files, and a torn parquet reads as corruption — which the pre-pass silently converts
    into a full network refetch. On Windows, replacing a file another process holds open
    can raise PermissionError; callers swallow it, leaving the OLD intact cache."""
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
    """One compact ASCII line per sweep — journald-safe, greppable, no unicode.

    ``cached`` names never touched the network, so an all-cached line is the positive
    record that the box deliberately did NOT download. That is precisely what parquet
    mtimes cannot tell you: an unchanged mtime is indistinguishable from a sweep that
    never ran."""
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
    """Fetch many tickers SAFELY. Concurrent single-ticker ``yf.download`` calls race on
    yfinance's shared global state (returning the wrong ticker's data), so we use yfinance's
    own batch download (``group_by='ticker'``, internal threading) in chunks, with inter-batch
    pauses + retry/backoff so a large universe isn't rate-limited into silently-dropped batches.

    Caching is incremental: a fresh (< ``max_age_days``) parquet is used as-is; a cache with a
    small recent gap is topped up with only the bars since its last date (one shared ``start``
    across the batch); everything else (no cache, or a gap > ``max_gap_days``) gets a full
    ``lookback`` refetch, which also re-baselines auto-adjusted history. ``max_age_days=0`` sends
    every cached name through the cheap incremental top-up (the nightly EOD path — finalized
    close without a full 2y refetch).

    Independent of the age window, a cache written with NO market session since (after the
    settled close → evenings, weekends, pre-open; see ``triggers.no_session_since``) is served
    as-is — no new bars can exist, so wall-clock age is irrelevant. Even ``max_age_days=0``
    honors this (a post-close fetch IS the finalized close). A NEGATIVE ``max_age_days`` (the
    tests' sentinel for "never serve from cache freshness") bypasses the settled gate too;
    ``force=True`` bypasses everything.

    ``allow_network=False`` = CACHE-ONLY: every name is served from its parquet as-is
    (no top-up, no full fetch, no ``_YF_LOCK`` contention); names with no cache at all
    come back absent. For interactive pages that must not queue behind a bulk sweep —
    pair with :func:`network_busy`.
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
        # ``detail`` says WHAT was fetched for this name (missing-days range / full history /
        # cache-served) so the UI can log one transparent line per ticker. Locked: the cache
        # pre-pass emits from pool workers, so the counter increment + callback serialize.
        nonlocal done
        with _emit_lock:
            done += 1
            if progress:
                progress(done, total, f"{sym}: {detail}" if detail else sym)

    def _classify_cached(sym: str):
        """Read-only per-name decision (thread-safe — stats + parquet reads, no writes):
        ``('served', df, label_detail)`` | ``('incr', cached_df, last_date)`` |
        ``('full', None, None)``."""
        path = PRICES_DIR / f"{sym}.parquet"
        if not force and age_days(path) <= max_age_days:
            try:
                return "served", pd.read_parquet(path), "cached (fresh)"
            except Exception:
                pass
        elif not force and max_age_days >= 0 and _cache_settled(path):
            # Written after the settled close with no session since — current no matter
            # how old the wall clock says it is (evening/weekend/pre-open scans). A
            # NEGATIVE max_age_days (the tests' skip-every-fresh-serve sentinel) bypasses
            # this gate too, keeping the top-up paths deterministically reachable.
            try:
                df = pd.read_parquet(path)
                # Content-side companion check: the mtime says the FILE was written
                # post-settle, but a lagging provider response (via the ungated
                # full-fetch persist) can leave a frame whose last bar predates the
                # latest settled session. Serve as settled only when the FRAME is
                # current too; else fall through to the cheap incremental top-up.
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
            _emit(sym, extra)                  # live bar while reads stream in
        return kind, frame, extra

    # Emit progress for cache-served names too — on a warm cache reading thousands of
    # parquets is real wall-clock time; pyarrow releases the GIL, so a small thread pool
    # roughly halves it. Served names emit from the workers (bar count monotonic under
    # _emit's lock; ORDER is interleaved — consumers treat the log as an unordered tail),
    # while out/incr/full_fetch are assembled strictly in syms order AFTER the join, so
    # dict order, batch composition, and chunking stay byte-deterministic. yf.download
    # below stays strictly serial (concurrent calls race yfinance's shared state).
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
        # CACHE-ONLY mode: serve whatever the disk has AS-IS instead of fetching.
        # Incremental names already hold their cached frames from the pre-pass;
        # too-stale/cold names get their parquet when one exists (absent otherwise —
        # callers degrade per name, e.g. fetch_positions' advisories read as None).
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
                        _rebaselined += 1
                        continue
                    out[sym] = merged
                    # Persist ONLY when the fetch reached the cache's newest bar: an
                    # overlap-only response (provider lag) must not re-stamp the mtime,
                    # or a post-cutoff rewrite would arm the settled-close gate on a
                    # frame LACKING the settled bar — served as "settled" all weekend
                    # with no healing fetch. ``>=`` is load-bearing (the ~16:30 settle
                    # of today's bar lands with max == last and MUST persist);
                    # ``index.max()`` not ``[-1]`` (_clean_prices never sorts).
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
                # One subset retry for names the batch missed: _download_batch returns as
                # soon as ANY rows exist, so its whole-batch retry never fires on the common
                # PARTIAL failure. Skipped when the whole batch came back empty (that case
                # was already retried inside _download_batch).
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
                    _failed.append(sym)       # counted once; a stale-cache serve below
                                              # still means the provider gave us nothing
                    # Failed fetch: serve the stale parquet when one exists (gap>max_gap and
                    # re-baseline names HAVE one) instead of silently dropping the name —
                    # the same fallback get_prices uses. NOT re-persisted: don't bump the
                    # mtime into the fresh window with data we know is stale.
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


def _yoy_at(s: Optional[pd.Series], back: int = 0) -> Optional[float]:
    """YoY % for the quarter ``back`` steps from the latest (0 = latest, 1 = prior): its value
    vs the entry ~a year earlier (330-400 days back), DATE-matched like ``_edgar_yoy_series``
    so a missing/extra quarter can't misalign a fixed 4-step lag. None if ``s`` is too short or
    no ~1-year-prior quarter exists."""
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
    """``num``/``den`` aligned on their common (quarter-end) index with NaNs dropped, so a margin
    never divides a numerator and denominator from DIFFERENT quarters. ``_row`` dropna's each line
    item independently, so yfinance populating Total Revenue for the newest quarter before Gross
    Profit / Operating Income would otherwise pair GP(Q-1) with Rev(Q0). Columns ``n``/``d``;
    None when either input is missing or no common quarter remains."""
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


def _last_earnings_surprise(tk) -> Optional[float]:
    """Most recent reported EPS surprise %, from ``yf.Ticker.earnings_dates`` (the
    'Surprise(%)' column; future/unreported rows are NaN and drop out). None on any
    miss — never raises."""
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
# SEC EDGAR XBRL backfill — real YoY history where yfinance's ~4 quarters run out,
# plus annual EPS growth and 3-quarter acceleration that yfinance can never provide.
# Company-facts API (no key; SEC fair-use ~10 req/s with a contact User-Agent).
# --------------------------------------------------------------------------- #
EDGAR_UA = {"User-Agent": "ml-trading-pfopt cockpit (treblotmail@gmail.com)"}
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# Fallback tag chains (same as the repo's Main.ipynb EDGAR pipeline).
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
    """Ticker -> SEC CIK via the company_tickers.json map (one cached download, ~monthly).
    Tries the dash form we normalize to AND the SEC's dot form (BRK-B vs BRK.B)."""
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
    """``(quarterly, annual)`` lists of ``(end_Timestamp, value)`` from the first us-gaap
    tag with usable data. Quarterly = filing duration 60-120 days, annual = 300-400 days
    (10-K YTD/other durations are dropped). Duplicate period-ends keep the LATEST 'filed'
    (amended figures win). Ascending by period end."""
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
    """``[(end, yoy_pct)]`` — each quarter vs the one ~a year earlier (330-400 days back;
    date-matched rather than a fixed 4-step lag, so a missing quarter can't misalign the
    comparison)."""
    out = []
    for i, (end, val) in enumerate(quarterly):
        prior = next((v for e2, v in reversed(quarterly[:i])
                      if 330 <= (end - e2).days <= 400), None)
        g = _pct(val, prior)
        if g is not None:
            out.append((end, g))
    return out


def _edgar_backfill(sym: str) -> Optional[dict]:
    """EDGAR-derived growth metrics for one ticker, cached weekly (like fundamentals):
    quarterly revenue/EPS YoY (+prev), annual FY EPS growth, and a 3-quarter EPS
    acceleration flag. None when the ticker has no CIK / no usable facts (foreign
    listings, funds) — the caller just keeps its yfinance numbers."""
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
    out["last_surprise_pct"] = _last_earnings_surprise(tk)
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
            # Schema upgrade: caches predating the earnings-date/surprise/EDGAR fields lack
            # those keys — refetch once so the new columns fill without waiting out weekly
            # staleness. (A present-but-None value stays cached.)
            if "next_earnings" in cached and "last_surprise_pct" in cached:
                return cached
        except Exception:
            pass
    out = _fetch_fundamentals(sym)
    if out is not None:
        # SEC EDGAR backfill: yfinance values WIN when present; EDGAR fills the Nones
        # (deep YoY history) and contributes its own keys (FY EPS growth, 3q accel).
        # Any failure leaves the yfinance dict untouched.
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
