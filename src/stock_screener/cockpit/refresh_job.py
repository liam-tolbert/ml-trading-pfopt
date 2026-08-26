"""Scheduled data refresh — CLI wrapper (``cockpit-refresh.timer``, every 30 minutes
09:30-16:30 ET weekdays).

Run from the project root:

    python src/stock_screener/cockpit/refresh_job.py [--date YYYY-MM-DD] [--no-write]
                                                     [--no-universe]

Two jobs, in order:

1. **Universe price top-up** — incremental delta fetch for every ticker in the scan
   universe (``max_age_days=0.0``: a cache holding today's bar re-fetches just the latest
   bar; only cold names or a split re-baseline pay a full 2y download). This is the price
   layer ONLY — no screening. It used to run inside the Streamlit container on a private
   thread, which meant a container restart could silently skip a day; owning it here makes
   it survive restarts and show up in ``systemctl list-timers`` like everything else.
2. **Watchlist trigger check** — freezes a pivot for any entry lacking one (recorded back
   into watchlist.json, ``pivot_source="auto"``; the app's 📌 overrides), evaluates
   Minervini's trigger — above the frozen pivot on >=1.5x the 50-day average volume — and
   prints + saves the dated JSON report the app's sidebar surfaces. Reuses the frames the
   universe pass just fetched, so a watchlisted name is not downloaded twice.

The FULL SCAN (screening: template chain, VCP, RS) is deliberately NOT here — it runs only
from the app's explicit Re-scan button. Screening every half hour would cost far more than
it buys, and the scan table is a thing you read deliberately, not a live tape.

Intraday runs see the live provisional bar (report flag ``intraday``; ``volume_pace`` =
volume so far vs expected by this time of day); the ~16:30 run sees the settled close.
Outside market hours the top-up costs no network at all — ``data_feed._cache_settled``
serves a cache written after the settled close as-is, because no new bar can exist.

NEVER places orders: a trigger means YOU judge it and buy via the trade panel
(HANDOFF §6.11/§6.14/§6.18).
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:                       # so `from src.X import ...` resolves
    sys.path.insert(0, str(ROOT))

from src.stock_screener.cockpit import (  # noqa: E402
    cache, data_feed, export, runlog, triggers)
from src.stock_screener.cockpit.scan_worker import DEFAULT_UNIVERSE  # noqa: E402

_LOG = runlog.get_logger("refresh")


def refresh_universe(universe: str = DEFAULT_UNIVERSE) -> dict:
    """Top up daily bars for the whole scan universe. Returns ``{ticker: DataFrame}``.

    Failures degrade per name inside ``get_many_prices`` (a name with no data is simply
    absent), and the sweep's own summary line — requested / cached / topup / full / wrote
    / failed — lands in the run log, so a quiet no-network sweep is distinguishable from
    one that never ran. A universe that cannot be resolved at all is NOT fatal: the
    trigger check still runs on the watchlist, which is the half that has a deadline."""
    try:
        tickers = data_feed.get_universe(universe)
    except Exception as e:
        _LOG.warning("universe %s unavailable (%s) — trigger check only", universe, e)
        return {}
    _LOG.info("universe top-up starting: %s, %d tickers", universe, len(tickers))
    t0 = time.time()
    frames = data_feed.get_many_prices(tickers, max_age_days=0.0)
    _LOG.info("universe top-up done: %d/%d frames in %.1fs",
              len(frames), len(tickers), time.time() - t0)
    return frames


def build_report(today=None, write_watchlist: bool = True,
                 prefetched: Optional[dict] = None) -> dict:
    """Fetch -> auto-freeze -> evaluate. Returns the report dict (see triggers.py).

    ``prefetched`` is the universe sweep's result: any watchlisted name already in it is
    reused rather than re-fetched (``max_age_days=0.0`` would otherwise force a second
    round-trip for names the sweep just refreshed). Names outside the universe — a hand-added
    watchlist entry — plus SPY are still fetched here. Auto-frozen pivots are persisted
    BEFORE the evaluation (skipped under ``--no-write``) so tomorrow's run checks the same
    level; the write-back merges into a fresh read of the file so a concurrent app-session
    save is never clobbered. Every per-name data problem degrades to that name's row,
    never a crash."""
    entries = export.load_watchlist(cache.WATCHLIST_JSON)
    syms = export.watchlist_tickers(entries)

    have = prefetched or {}
    # SPY rides along even when the list is empty (the report's market note). SPY is not
    # in the screening universe, so it is always part of this second fetch.
    missing = [t for t in syms if data_feed.normalize(t) not in have] + ["SPY"]
    fetched = dict(have)
    if missing:
        fetched.update(data_feed.get_many_prices(missing, max_age_days=0.0))
    spy = fetched.get("SPY")
    # get_many_prices keys by normalize() (e.g. BRK.B -> BRK-B); re-key by entry ticker.
    frames = {t: fetched.get(data_feed.normalize(t), fetched.get(t)) for t in syms}

    entries, frozen = triggers.freeze_missing_pivots(entries, frames, today=today)
    if frozen and write_watchlist:
        # Merge into the file's CURRENT state, not the copy loaded before the slow price
        # fetch: an app-session save during that window (remove / 📌 re-freeze / add)
        # would otherwise be clobbered. Disk wins membership and any pivot it has; our
        # auto pivots land only on entries still unfrozen on disk.
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
    ap = argparse.ArgumentParser(description="Scheduled data refresh: universe price "
                                             "top-up + watchlist trigger check "
                                             "(decision support only — never places "
                                             "orders).")
    ap.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                    help="pin the run date (tests/backfill); default = today in New York")
    ap.add_argument("--no-write", action="store_true",
                    help="print only — skip the report file AND the watchlist write-back")
    ap.add_argument("--no-universe", action="store_true",
                    help="skip the universe top-up; check the watchlist only (a fast "
                         "trigger-only run for a hand check)")
    args = ap.parse_args(argv)

    try:
        prefetched = {} if args.no_universe else refresh_universe()
        report = build_report(today=args.date, write_watchlist=not args.no_write,
                              prefetched=prefetched)
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
