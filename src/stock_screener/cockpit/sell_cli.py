"""Auto-sell CLI — the two scheduled halves of the P1-P4 sell automation.

    python src/stock_screener/cockpit/sell_cli.py plan    [--date YYYY-MM-DD] [--no-write]
    python src/stock_screener/cockpit/sell_cli.py execute [--date YYYY-MM-DD] [--dry-run]

``plan`` runs after the settled close (~16:40 ET): reads the paper account, evaluates
the sell pillars per holding exactly as the Positions page does (journal entry dates,
watchlist frozen pivots, trigger-report SPY note), and writes the dated sell plan the
page renders for overnight veto. ``execute`` runs pre-open (~09:25 ET): submits a
market SELL for every still-planned order via the stop-aware flow — pre-open orders
queue for the opening print. Execution requires ``AUTOSELL=1`` in the environment
(.env); without it the run reports "disabled" and exits clean, so the timer can ship
before the feature is armed. Paper account only.
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

from src.stock_screener.cockpit import cache, export, sells, trade, triggers  # noqa: E402


def _positions_and_pillars(today=None):
    """The Positions page's pillar wiring, headless. Every side input is best-effort —
    a missing journal/watchlist/report degrades pillars to unknown, and unknown never
    trades. The scan-store regime isn't available in a fresh process; P3 falls back to
    the trigger report's SPY note (and P3 is report-only in the plan anyway)."""
    data = trade.fetch_positions()
    positions = data["positions"]
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
                   today=today)
               for p in positions}
    return data, positions, pillars


def cmd_plan(date: Optional[str], write: bool) -> int:
    data, positions, pillars = _positions_and_pillars(today=date)
    prior = sells.load_latest_sell_plan(before=sells._today_iso(date))
    plan = sells.build_sell_plan(positions, pillars, prior_plan=prior, today=date)
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
        submit = lambda sym, qty: {"status": "submitted", "detail": "DRY RUN"}  # noqa: E731
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
    if not dry_run:
        sells.save_sell_plan(plan)
    return 1 if summary["status"] in ("failed", "partial", "stale") else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
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
