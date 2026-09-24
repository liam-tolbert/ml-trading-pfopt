"""Scheduled data refresh: the CLI for both refresh scopes.

    python src/stock_screener/cockpit/refresh_job.py [--scope watchlist|universe]
                                                     [--max-age-days N]
                                                     [--date YYYY-MM-DD] [--no-write]

Two scopes, two schedules, because their costs are three orders of magnitude apart:

* ``watchlist`` (default): ``cockpit-refresh.timer``, 09:30, then :00/:30 through 15:30,
  then 16:10 ET (the settled-close run).
  Tops up the watchlist plus any symbol held on the paper account that is not on it. A
  held name MUST stay priceable after it leaves the watchlist, or the Positions page and
  the sell pillars go blind on it. Tens of names; seconds per run.
* ``universe``: step 1 of ``cockpit-eod.timer``, 16:20 ET weekdays. The screen is step 2
  of the same unit, so it can't start until this exits 0. Tops up all ~4,100
  scan-universe tickers once, after the settled close. This is the run that matters: it
  writes post-settle, which arms ``data_feed._cache_settled``, so every later read
  (evening, overnight, pre-open) is served from cache with zero network.

Both scopes then run the watchlist trigger check, reusing the frames just fetched.

The whole universe MUST NOT be refreshed intraday. During a live session
``_cache_settled`` is false by definition and every name is older than the max age, so
all ~4,100 names re-download on every fire. That is ~12 minutes of continuous yfinance
traffic per run, ~2.8 hours a day, and draws ``YFRateLimitError``. Intraday only the
watchlist and positions are read; the rest of the universe is read only by a screen.

The full scan (template chain, VCP, RS) is never here: ``screen_job.py`` and the app's
Re-scan own it.

Never places orders: a trigger means you judge it and buy via the trade panel.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:                       # so `from src.X import ...` resolves
    sys.path.insert(0, str(ROOT))

from src.stock_screener.cockpit import (  # noqa: E402
    cache, data_feed, export, runlog, trade, triggers)
from src.stock_screener.cockpit.scan_worker import DEFAULT_UNIVERSE  # noqa: E402

_LOG = runlog.get_logger("refresh")

SCOPE_WATCHLIST = "watchlist"
SCOPE_UNIVERSE = "universe"
SCOPES = (SCOPE_WATCHLIST, SCOPE_UNIVERSE)

# A safety valve against duplicate work, not a freshness policy: a re-run inside this
# window (a hand invocation, a fire on a slow predecessor's heels) is served from cache.
# It MUST be above 0: no existing file is ever <= 0 days old, so 0 makes the freshness
# branch in _classify_cached unreachable.
#
# It MUST stay well under half the 30-minute cadence. Age is measured from when a file
# was written, not when its run started, so the real gap to the next fire is the interval
# minus the run's duration minus AccuracySec. Near 29 minutes, one slow run makes the next
# fire serve the last sweep from cache and silently skip the refresh.
REFRESH_MAX_AGE_DAYS = 0.01         # ~14.4 minutes


def refresh_targets(scope: str) -> List[str]:
    """The tickers this run tops up. Never raises.

    ``universe`` is the scan universe, or empty when it is unavailable. ``watchlist`` is
    the watchlist plus the paper account's open positions. The two drift apart: a name
    leaves the watchlist but is still held, or is held but was never watchlisted. An
    unreachable broker (no credentials, network down) degrades to watchlist-only rather
    than failing the refresh; the watchlist is the half with a deadline."""
    if scope == SCOPE_UNIVERSE:
        try:
            return list(data_feed.get_universe(DEFAULT_UNIVERSE))
        except Exception as e:
            _LOG.warning("universe %s unavailable (%s) — trigger check only",
                         DEFAULT_UNIVERSE, e)
            return []

    try:
        names = list(export.watchlist_tickers(
            export.load_watchlist(cache.WATCHLIST_JSON)))
    except Exception as e:
        _LOG.warning("watchlist unreadable (%s)", e)
        names = []
    have = {data_feed.normalize(t) for t in names}
    try:
        held = [s for s in trade.position_symbols()
                if data_feed.normalize(s) not in have]
        if held:
            _LOG.info("including %d held name(s) not on the watchlist: %s",
                      len(held), ", ".join(held))
        names += held
    except Exception as e:
        _LOG.warning("positions unavailable (%s) — watchlist only", e)
    return names


def refresh_prices(tickers: Sequence[str], *, scope: str,
                   max_age_days: float = REFRESH_MAX_AGE_DAYS) -> dict:
    """Top up ``tickers``; return ``{ticker: DataFrame}``, empty for no tickers.

    Failures degrade per name inside ``get_many_prices``: a name with no data is absent.
    The sweep's summary (requested, cached, topup, full, wrote, failed) lands in the run
    log, so a zero-network sweep is distinguishable from one that never ran."""
    if not tickers:
        return {}
    _LOG.info("%s top-up starting: %d tickers", scope, len(tickers))
    t0 = time.time()
    frames = data_feed.get_many_prices(list(tickers), max_age_days=max_age_days)
    _LOG.info("%s top-up done: %d/%d frames in %.1fs",
              scope, len(frames), len(tickers), time.time() - t0)
    return frames


def build_report(today=None, write_watchlist: bool = True,
                 prefetched: Optional[dict] = None,
                 max_age_days: float = REFRESH_MAX_AGE_DAYS) -> dict:
    """Fetch, auto-freeze pivots, evaluate. Returns the report dict (see triggers.py).

    ``prefetched`` is the sweep's result; a watchlisted name already in it is reused.
    Other names, and always SPY (never in the scan universe), are fetched here.
    Auto-frozen pivots are persisted before the evaluation, unless ``write_watchlist`` is
    False, so tomorrow's run checks the same level. The write-back merges into a fresh
    read of the file, so a concurrent app-session save is never clobbered. A per-name
    data problem degrades to that name's row, never a crash."""
    entries = export.load_watchlist(cache.WATCHLIST_JSON)
    syms = export.watchlist_tickers(entries)

    have = prefetched or {}
    missing = [t for t in syms if data_feed.normalize(t) not in have] + ["SPY"]
    fetched = dict(have)
    if missing:
        fetched.update(data_feed.get_many_prices(missing, max_age_days=max_age_days))
    spy = fetched.get("SPY")
    # get_many_prices keys by normalize() (e.g. BRK.B -> BRK-B); re-key by entry ticker.
    frames = {t: fetched.get(data_feed.normalize(t), fetched.get(t)) for t in syms}

    entries, frozen = triggers.freeze_missing_pivots(entries, frames, today=today)
    if frozen and write_watchlist:
        # MUST merge into the file's current state, not the copy loaded before the slow
        # price fetch: an app-session save in that window (remove, 📌 re-freeze, add)
        # would be clobbered. Disk wins membership and any pivot it has; auto pivots land
        # only on entries still unfrozen on disk.
        disk = export.load_watchlist(cache.WATCHLIST_JSON)
        export.save_watchlist(cache.WATCHLIST_JSON,
                              export.merge_frozen_pivots(disk, entries))

    def _fund(t):
        try:
            return data_feed.get_fundamentals(t)     # weekly JSON cache per ticker
        except Exception:
            return None

    report = triggers.check_triggers(entries, frames, fundamentals=_fund, spy=spy,
                                     today=today)
    report["summary"]["auto_frozen"] = frozen
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Scheduled data refresh: price top-up + "
                                             "watchlist trigger check (decision support "
                                             "only — never places orders).")
    ap.add_argument("--scope", choices=SCOPES, default=SCOPE_WATCHLIST,
                    help="watchlist (default; + held names not on it) or universe")
    ap.add_argument("--max-age-days", type=float, default=REFRESH_MAX_AGE_DAYS,
                    metavar="N", help="serve a cache younger than this instead of "
                                      f"re-fetching (default {REFRESH_MAX_AGE_DAYS})")
    ap.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                    help="pin the run date (tests/backfill); default = today in New York")
    ap.add_argument("--no-write", action="store_true",
                    help="print only — skip the report file AND the watchlist write-back")
    args = ap.parse_args(argv)

    try:
        targets = refresh_targets(args.scope)
        prefetched = refresh_prices(targets, scope=args.scope,
                                    max_age_days=args.max_age_days)
        report = build_report(today=args.date, write_watchlist=not args.no_write,
                              prefetched=prefetched, max_age_days=args.max_age_days)
        print(triggers.format_report(report))
        if not args.no_write:
            path = triggers.save_trigger_report(report)
            print(f"report: {path}")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
