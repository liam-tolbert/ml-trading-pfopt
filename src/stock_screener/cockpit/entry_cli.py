"""Armed-entries CLI — the morning half of the buy automation.

    python src/stock_screener/cockpit/entry_cli.py execute [--date YYYY-MM-DD] [--dry-run]

Runs pre-open (~09:26 ET): loads the latest armed entry plan (written by the app's
"Arm for next open" button at the evening ritual) and submits AT MOST ONE still-armed
buy — limit at the buy-zone top + GTC OTO stop — through the same plan-submit path the
panel uses. Requires ``AUTOBUY=1`` (ships dark); the progressive-exposure gate FAILS
CLOSED on this unattended path. No plan / a stale plan is NORMAL (the user simply
didn't arm anything) — exit 0; exit 1 only on failed/partial submission.
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

from src.stock_screener.cockpit import entries, trade  # noqa: E402


def _real_submit(row: dict) -> dict:
    """One armed row through the REAL plan-submit path (pending-buy guard, tradability,
    10% cap, stop validity, GTC OTO limit — all re-checked there)."""
    out = trade.submit_buy_plan([row], attach_stop=True)
    res = (out.get("results") or [{}])[0]
    return {"status": res.get("status"), "detail": res.get("detail", "")}


def cmd_execute(date: Optional[str], dry_run: bool) -> int:
    plan = entries.load_latest_entry_plan()
    if plan is None:
        print("no armed entry plan - nothing to do")
        return 0
    if not entries.autobuy_enabled() and not dry_run:
        print(f"AUTOBUY not enabled - plan {plan.get('date')} left untouched "
              "(set AUTOBUY=1 in .env to arm)")
        return 0

    gate = None
    held = {}
    try:
        gi = trade.fetch_gate_inputs()
        gate = trade.gate_status(gi["positions"], gi["open_episodes"],
                                 gi["closed_episodes"])
        held = {p["symbol"]: int(p.get("qty") or 0) for p in gi["positions"]}
    except Exception:
        gate = None                       # executor fails closed on unknown state

    if dry_run:
        submit = lambda row: {"status": "submitted", "detail": "DRY RUN"}  # noqa: E731
    else:
        submit = _real_submit
    summary = entries.execute_entry_plan(plan, submit=submit, gate=gate,
                                         held_by_symbol=held, today=date,
                                         enabled=True)
    print(f"execute[{'dry-run' if dry_run else 'live'}] plan {plan.get('date')}: "
          f"{summary['status']}  submitted={summary['submitted']}  "
          f"skipped={summary['skipped']}  disarmed={summary['disarmed']}  "
          f"failed={summary['failed']}")
    if gate is not None:
        print(f"gate: {'open' if gate.get('open') else 'closed'} - "
              f"{gate.get('reason', '')}")
    print(entries.format_plan(plan))
    if not dry_run and summary["status"] not in ("disabled", "stale"):
        entries.save_entry_plan(plan)
    return 1 if summary["status"] in ("failed", "partial") else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Armed-entries morning executor "
                                             "(paper account; at most one buy).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("execute", help="submit at most one still-armed entry")
    e.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    e.add_argument("--dry-run", action="store_true",
                   help="print the walk; no orders, no plan update")
    args = ap.parse_args(argv)
    try:
        return cmd_execute(args.date, dry_run=args.dry_run)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
