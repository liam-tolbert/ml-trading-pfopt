"""Nightly EOD trigger check — CLI wrapper (scheduled via Windows Task Scheduler).

Run from the project root (the .bat wrapper does this):

    python src/stock_screener/cockpit/eod_trigger.py [--prewarm] [--date YYYY-MM-DD]
                                                     [--no-write]

Loads the persisted watchlist, tops up those names' daily bars (incremental — the
finalized close, NOT a full refetch), freezes a pivot for any entry that doesn't have one
yet (recorded back into watchlist.json, ``pivot_source="auto"``; your 📌 in the app
overrides), evaluates Minervini's trigger — closed above the frozen pivot on >=1.5x the
50-day average volume — and prints + saves a dated JSON report the app's sidebar surfaces.
NEVER places orders: a trigger tonight means YOU buy at/near the next open via the trade
panel (EOD-confirm cadence, HANDOFF §6.11/§6.14).

``--prewarm`` afterwards tops up the whole full_us price cache so the next interactive
scan is instant; it runs strictly AFTER the report so a prewarm failure can't cost it.
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

    The price refresh uses ``max_age_days=0.0`` (incremental top-up with split detection)
    — cheap and correct for the just-finalized close; ``force=True`` would re-download
    full 2y histories. Auto-frozen pivots are persisted to the watchlist BEFORE the
    evaluation (skipped under ``--no-write``), so tomorrow's run checks the same level.
    Every per-name data problem degrades to that name's row, never a crash."""
    entries = export.load_watchlist(cache.WATCHLIST_JSON)
    syms = export.watchlist_tickers(entries)

    # SPY rides along even when the list is empty (the report's market note).
    fetched = data_feed.get_many_prices(syms + ["SPY"], max_age_days=0.0)
    spy = fetched.get("SPY")
    # get_many_prices keys by normalize() (e.g. BRK.B -> BRK-B); re-key by entry ticker.
    frames = {t: fetched.get(data_feed.normalize(t), fetched.get(t)) for t in syms}

    entries, frozen = triggers.freeze_missing_pivots(entries, frames, today=today)
    if frozen and write_watchlist:
        export.save_watchlist(cache.WATCHLIST_JSON, entries)

    def _fund(t):
        try:
            return data_feed.get_fundamentals(t)     # weekly JSON cache per ticker
        except Exception:
            return None

    report = triggers.check_triggers(entries, frames, fundamentals=_fund, spy=spy,
                                     today=today)
    report["summary"]["auto_frozen"] = frozen
    return report


def prewarm(universe: str = "full_us") -> None:
    """Top up the whole universe's price cache (incremental — new bars only for warm
    names, full history only for cold ones) so the first interactive scan of the next
    session is instant. Batch semantics live in get_many_prices (chunks of 100, backoff,
    inter-chunk pauses — never per-ticker threads)."""
    tickers = data_feed.get_universe(universe)
    print(f"prewarm: topping up {len(tickers)} {universe} price histories...")

    def _progress(done, total, sym):
        if done % 500 == 0 or done == total:
            print(f"prewarm: {done}/{total} ({sym})")

    data_feed.get_many_prices(tickers, max_age_days=0.0, progress=_progress)
    print("prewarm: done.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Nightly EOD trigger check (decision support "
                                             "only — never places orders).")
    ap.add_argument("--prewarm", action="store_true",
                    help="afterwards, top up the full_us price cache for tomorrow's scan")
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

    if args.prewarm:                                 # strictly after the report is safe
        try:
            prewarm()
        except Exception:
            traceback.print_exc()                    # logged, but the report succeeded
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
