"""Scheduled data refresh: the CLI for both refresh scopes.

    python src/stock_screener/cockpit/refresh_job.py [--scope watchlist|universe]
                                                     [--max-age-days N]
                                                     [--date YYYY-MM-DD] [--no-write]

Each run downloads only the missing recent days for its tickers. Two scopes, two
schedules, because one costs about a thousand times more than the other:

* ``watchlist`` (default): ``cockpit-refresh.timer``, 09:30, then :00/:30 through 15:30,
  then 16:10 ET (the run on the settled close).
  Covers the watchlist plus any symbol held on the paper account that is not on it. A
  held stock MUST keep getting prices after it leaves the watchlist, or the Positions
  page and the sell pillars have no data for it. Tens of tickers; seconds per run.
* ``universe``: step 1 of ``cockpit-eod.timer``, 16:20 ET weekdays. The screen is step 2
  of the same unit, so it can't start until this exits 0. Covers all ~4,100 tickers the
  scan screens (the universe) once, after the settled close. This is the run that
  matters: its files are written after the close, so ``data_feed._cache_settled`` reads
  True for them and every later read (evening, overnight, before the open) comes from
  the saved files with no download.

Both scopes then run the watchlist trigger check, reusing the prices just downloaded.

The whole universe MUST NOT be refreshed during market hours. While the market is open
``_cache_settled`` is always false and every ticker is older than the max age, so all
~4,100 re-download on every timer run. That is ~12 minutes of continuous yfinance
traffic per run, ~2.8 hours a day, and triggers ``YFRateLimitError``. During market
hours only the watchlist and positions are read; the rest of the universe is read only
by a screen.

The full scan (the Trend Template filters, VCP, RS) never runs here: ``screen_job.py``
and the app's Re-scan do that.

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

# A guard against downloading the same data twice, not a rule for how new data must be:
# a re-run inside this window (a run by hand, a timer run right after a slow earlier run)
# reads the saved files. It MUST be above 0: no existing file is ever <= 0 days old, so 0
# means the "new enough" branch in _classify_cached never runs.
#
# It MUST stay well under half the 30-minute schedule. Age is measured from when a file
# was written, not when its run started, so the real gap to the next run is the interval
# minus the run's duration minus the timer's AccuracySec. Near 29 minutes, one slow run
# makes the next run reuse the last run's files and silently skip the refresh.
REFRESH_MAX_AGE_DAYS = 0.01         # ~14.4 minutes


def refresh_targets(scope: str) -> List[str]:
    """The tickers this run updates. Never raises.

    ``universe`` is the scan universe, or empty when it is unavailable. ``watchlist`` is
    the watchlist plus the paper account's open positions. The two lists differ: a stock
    leaves the watchlist but is still held, or is held but was never on the watchlist. An
    unreachable broker (no credentials, network down) falls back to watchlist-only rather
    than failing the refresh, since the trigger check needs the watchlist on time."""
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
    """Download only the missing recent days for ``tickers``; return ``{ticker:
    DataFrame}``, empty for no tickers.

    ``get_many_prices`` handles failures per ticker: one with no data is left out. The
    run's download summary (requested, cached, topup, full, wrote, failed) lands in the
    run log, so a run that downloaded nothing can be told apart from one that never
    ran."""
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
    """Download prices, freeze missing pivots, check triggers. Returns the report dict
    (see triggers.py).

    ``prefetched`` is the result of :func:`refresh_prices`; a watchlist ticker already in
    it is reused. Other tickers, and always SPY (never in the scan universe), are
    downloaded here. Auto-frozen pivots are saved before the trigger check, unless
    ``write_watchlist`` is False, so tomorrow's run checks the same level. The save
    merges into a fresh read of the file, so an app-session save made meanwhile is never
    overwritten. A data problem for one ticker shows up in that ticker's row; it never
    crashes the run."""
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
        # price download: an app-session save in that window (remove, 📌 judge again,
        # add) would be lost. The file on disk decides which tickers are listed and
        # keeps any pivot it has; auto pivots land only on entries still unfrozen on
        # disk.
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
    ap = argparse.ArgumentParser(description="Scheduled data refresh: download the missing "
                                             "recent prices + watchlist trigger check (it "
                                             "only informs you — never places orders).")
    ap.add_argument("--scope", choices=SCOPES, default=SCOPE_WATCHLIST,
                    help="watchlist (default; + held stocks not on it) or universe "
                         "(every ticker the scan screens)")
    ap.add_argument("--max-age-days", type=float, default=REFRESH_MAX_AGE_DAYS,
                    metavar="N", help="use saved prices newer than this many days instead "
                                      f"of downloading again (default {REFRESH_MAX_AGE_DAYS})")
    ap.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                    help="set the run date (for tests or past dates); default = today in New York")
    ap.add_argument("--no-write", action="store_true",
                    help="print only — don't save the report file OR the watchlist")
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
