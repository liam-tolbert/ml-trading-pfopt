"""Auto-sell CLI: the two scheduled halves of the P1-P4 sell automation.

    python src/stock_screener/cockpit/sell_job.py plan    [--date YYYY-MM-DD] [--no-write]
    python src/stock_screener/cockpit/sell_job.py execute [--date YYYY-MM-DD] [--dry-run]

``plan`` runs after the settled close (16:15 ET). It reads the paper account, evaluates
the sell pillars per holding as the Positions page does (journal entry dates, watchlist
frozen pivots, trigger-report SPY note) and writes the dated sell plan the page renders
for the overnight veto.

``execute`` runs pre-open (~09:25 ET). It submits a market SELL for every still-planned
order via the stop-aware flow; pre-open orders queue for the opening print. Execution
requires ``AUTOSELL=1`` in the environment (.env). Without it the run leaves the plan
untouched and exits 0, so the timer can ship before the feature is armed. Paper account
only.
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

from src.stock_screener.cockpit import (cache, export, plan_store, sells,  # noqa: E402
                                        trade, triggers)


def _rs_ratings() -> dict:
    """The last scan's RS ratings from the persisted pickle, cache only. ``{}`` without
    a scan or on any error."""
    try:
        from src.stock_screener.cockpit import scan_worker
        ent = scan_worker._STORE.get((scan_worker.DEFAULT_UNIVERSE,
                                      scan_worker.DEFAULT_MIN_CRITERIA))
        return dict(getattr(getattr(ent, "result", None), "rs_ratings", None) or {})
    except Exception:
        return {}


def _positions_and_pillars(today=None):
    """The Positions page's pillar wiring, headless. Returns ``(data, positions, pillars,
    spy_note)``; a failed positions read raises.

    Every side input is best effort: a missing journal, watchlist or report degrades
    pillars to unknown, and unknown never trades. P2's RS rating comes from the persisted
    last scan. The scan-store regime isn't available in a fresh process, so P3 falls back
    to the trigger report's SPY note. P3 is report-only in the plan."""
    data = trade.fetch_positions()
    positions = data["positions"]
    rs_map = _rs_ratings()
    try:
        fills = trade.fetch_order_fills()["fills"]
        open_by_sym = {r["symbol"]: r for r in trade.build_trade_journal(fills)["open"]}
    except Exception:
        open_by_sym = {}
    try:
        wl_pivots = {e["ticker"]: e.get("judged_pivot")
                     for e in export.load_watchlist(cache.WATCHLIST_JSON)}
    except Exception:
        wl_pivots = {}
    try:
        spy = (triggers.load_latest_trigger_report() or {}).get("spy")
    except Exception:
        spy = None
    pillars = {p["symbol"]: trade.sell_pillars(
                   p, entry_date=(open_by_sym.get(p["symbol"]) or {}).get("entry_date"),
                   pivot=wl_pivots.get(p["symbol"]), regime=None, spy_note=spy,
                   today=today, rs=rs_map.get(p["symbol"]))
               for p in positions}
    return data, positions, pillars, spy


def _market(spy_note) -> dict:
    """The plan's market read: the trigger report's SPY note and the re-entry streak.
    The note is the 16:10 settled close, fresher than the scan. The streak MUST read
    cached bars only: this job never downloads."""
    streak = None
    try:
        from src.stock_screener.cockpit import advisories, breadth_store, data_feed
        spy_df = data_feed.get_many_prices(["SPY"], allow_network=False).get("SPY")
        hist = breadth_store.load()
        streak = advisories.spy_confirm_streak(
            spy_df, phase2_by_date=breadth_store.phase2_by_date(hist) if hist else None)
    except Exception:
        streak = None
    return {"spy_note": spy_note, "streak": streak}


def cmd_plan(date: Optional[str], write: bool) -> int:
    data, positions, pillars, spy = _positions_and_pillars(today=date)
    prior = sells.load_latest_sell_plan(before=plan_store.today_iso(date))
    plan = sells.build_sell_plan(positions, pillars, prior_plan=prior, today=date,
                                 market=_market(spy))
    acct = data["account"]
    print(f"account ...{str(acct.get('account_number'))[-4:]}  "
          f"equity ${acct.get('equity', 0):,.0f}  positions {len(positions)}")
    print(sells.format_plan(plan))
    if write:
        path = sells.save_sell_plan(plan)
        print(f"plan: {path}")
    else:
        print("(--no-write: plan not saved)")
    return 0


def cmd_execute(date: Optional[str], dry_run: bool) -> int:
    plan = sells.load_latest_sell_plan()
    if plan is None:
        print("no sell plan found - nothing to execute")
        return 0
    if not sells.autosell_enabled() and not dry_run:
        print(f"AUTOSELL not enabled - plan {plan.get('date')} left untouched "
              "(set AUTOSELL=1 in .env to arm)")
        return 0
    if dry_run:
        submit = lambda sym, qty, **kw: {"status": "submitted",  # noqa: E731
                                         "detail": "DRY RUN"}
    else:
        submit = trade.submit_position_sell
    held = {p["symbol"]: int(p["qty"] or 0)
            for p in trade.fetch_positions()["positions"]}
    summary = sells.execute_sell_plan(plan, submit=submit, held_by_symbol=held,
                                      today=date, enabled=True)
    print(f"execute[{'dry-run' if dry_run else 'live'}] plan {plan.get('date')}: "
          f"{summary['status']}  submitted={summary['submitted']}  "
          f"vetoed={summary['vetoed']}  skipped={summary['skipped']}  "
          f"failed={summary['failed']}")
    print(sells.format_plan(plan))
    # A disabled or stale run returns before touching the plan; saving it would only
    # move the file's mtime.
    if not dry_run and summary["status"] not in ("disabled", "stale"):
        sells.save_sell_plan(plan)
    # "stale" is normal: a holiday, or a morning with no plan from the night before.
    # Exit 1 MUST be reserved for real failures. A red systemd unit on an ordinary skip
    # trains you to ignore the one that matters.
    return 1 if summary["status"] in ("failed", "partial") else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    # AUTOSELL lives in .env. The compose service passes it, but a hand run on the
    # laptop needs it loaded here.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="P1-P4 auto-sell: evening plan / morning "
                                             "execute (paper account).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="evaluate pillars, write tonight's sell plan")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--no-write", action="store_true")
    e = sub.add_parser("execute", help="submit still-planned sells for the open")
    e.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    e.add_argument("--dry-run", action="store_true",
                   help="print what would be submitted; no orders, no plan update")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "plan":
            return cmd_plan(args.date, write=not args.no_write)
        return cmd_execute(args.date, dry_run=args.dry_run)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
