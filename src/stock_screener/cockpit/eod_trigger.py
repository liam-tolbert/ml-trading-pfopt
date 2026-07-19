"""Watchlist trigger check — CLI wrapper (scheduled via Windows Task Scheduler, every 30
minutes during market hours: task "SEPA Intraday Trigger", weekdays 09:30-16:30).

Run from the project root (the .bat wrapper does this):

    python src/stock_screener/cockpit/eod_trigger.py [--date YYYY-MM-DD] [--no-write]

Loads the persisted watchlist, tops up ONLY those names' daily bars (incremental delta
fetch — a run costs ~1-2 batched yfinance calls), freezes a pivot for any entry that lacks
one (recorded back into watchlist.json, ``pivot_source="auto"``; your 📌 in the app
overrides), evaluates Minervini's trigger — above the frozen pivot on >=1.5x the 50-day
average volume — and prints + saves a dated JSON report the app's sidebar surfaces.
Intraday runs see the live provisional bar (report flag ``intraday``; ``volume_pace`` =
volume so far vs expected by this time of day); the ~16:30 run sees the settled close.
NEVER places orders: a trigger means YOU judge it and buy via the trade panel
(HANDOFF §6.11/§6.14/§6.18).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:                       # so `from src.X import ...` resolves
    sys.path.insert(0, str(ROOT))

from src.stock_screener.cockpit import cache, data_feed, export, triggers  # noqa: E402


def build_report(today=None, write_watchlist: bool = True) -> dict:
    """Fetch -> auto-freeze -> evaluate. Returns the report dict (see triggers.py).

    The price refresh uses ``max_age_days=0.0`` (incremental top-up with split detection) —
    cheap and correct for the latest bar; ``force=True`` would re-download full 2y histories.
    Auto-frozen pivots are persisted BEFORE the evaluation (skipped under ``--no-write``) so
    tomorrow's run checks the same level; the write-back merges into a fresh read of the
    file so a concurrent app-session save is never clobbered. Every per-name data problem
    degrades to that name's row, never a crash."""
    entries = export.load_watchlist(cache.WATCHLIST_JSON)
    syms = export.watchlist_tickers(entries)

    # SPY rides along even when the list is empty (the report's market note).
    fetched = data_feed.get_many_prices(syms + ["SPY"], max_age_days=0.0)
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
    ap = argparse.ArgumentParser(description="Watchlist trigger check (decision support "
                                             "only — never places orders).")
    ap.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                    help="pin the run date (tests/backfill); default = today in New York")
    ap.add_argument("--no-write", action="store_true",
                    help="print only — skip the report file AND the watchlist write-back")
    args = ap.parse_args(argv)

    try:
        report = build_report(today=args.date, write_watchlist=not args.no_write)
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
