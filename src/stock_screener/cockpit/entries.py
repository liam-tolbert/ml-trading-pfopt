"""Armed entries — pre-authorized next-open buys, the buy-side mirror of sells.py.

The judgment stays human and happens at the evening ritual: the user builds a LIMIT
trade plan in the panel exactly as always, then "arms" it instead of (or after)
submitting. That writes tonight's ``entry_plan_YYYY-MM-DD.json`` beside the trigger
reports. A pre-open CLI (``entry_job.py execute``, ~09:26 ET) then submits AT MOST ONE
still-armed row — limit at the buy-zone top (no-chase, self-enforcing) with the GTC
OTO stop leg — after clearing, in order: the ``AUTOBUY`` env gate (the feature ships
dark), plan freshness (next-trading-day-only), and the progressive-exposure gate,
which for this unattended path FAILS CLOSED (unknown = no buys). Per-row disarm in
the app is the overnight veto; rows never executed simply expire with the plan.

Plan files live in ``cache.TRIGGERS_DIR`` so the test suite's existing patching keeps
AppTests away from real state. Paper account only, like every trade path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.stock_screener.cockpit import plan_store

AUTOBUY_ENV = "AUTOBUY"
_PREFIX = "entry_plan"

ROW_ARMED = "armed"
ROW_DISARMED = "disarmed"
ROW_SUBMITTED = "submitted"
ROW_FAILED = "failed"
ROW_SKIPPED = "skipped"


def _row(o: dict) -> dict:
    """One armed row from a built plan entry — coerced to plain JSON types (the scan
    payload leaks numpy scalars into plan rows)."""
    def f(v):
        return None if v is None else float(v)
    return {"ticker": str(o["ticker"]), "shares": int(o["shares"]),
            "price": f(o.get("price")), "pivot": f(o.get("pivot")),
            "limit_price": f(o.get("limit_price")), "stop_price": f(o.get("stop_price")),
            "est_value": f(o.get("est_value")),
            "earnings_in": None if o.get("earnings_in") is None
            else int(o["earnings_in"]),
            "status": ROW_ARMED, "detail": ""}


def build_entry_plan(final_rows: List[dict], today=None) -> dict:
    """Tonight's armed-entry plan from the panel's final rows. Pure.

    Only genuine BUY rows arm: ``shares >= 1``, not ``rearm_only``/``stop_only``, and
    both a positive ``limit_price`` (the no-chase cap IS the entry mechanic — a market
    row must never arm) and a positive ``stop_price`` below it (the OTO leg). Order is
    preserved — the executor walks rows top-down, so the panel's ordering is the
    ranking."""
    rows = []
    skipped = []
    for o in final_rows or []:
        t = o.get("ticker")
        if (o.get("rearm_only") or o.get("stop_only")
                or int(o.get("shares") or 0) < 1):
            continue                      # not a buy — nothing to arm
        lim = o.get("limit_price")
        stop = o.get("stop_price")
        if not lim or float(lim) <= 0:
            skipped.append(f"{t}: no limit price — market rows never arm")
            continue
        if not stop or not (0 < float(stop) < float(lim)):
            skipped.append(f"{t}: stop must sit below the limit")
            continue
        rows.append(_row(o))

    import pandas as pd
    return {"date": plan_store.today_iso(today),
            "generated_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "rows": rows, "notes": skipped, "executed_at": None}


# Storage is shared with sells.py (see plan_store); these keep the entry-side names the
# app, the pages and the CLI already import.
def entry_plan_path(date_iso: str, dir_path=None) -> Path:
    return plan_store.plan_path(_PREFIX, date_iso, dir_path)


def save_entry_plan(plan: dict, dir_path=None) -> Path:
    """Atomic write — the arming click, a disarm click and the morning executor run in
    separate processes against the same day-file."""
    return plan_store.save_plan(_PREFIX, plan, dir_path)


def load_latest_entry_plan(dir_path=None, *, before: Optional[str] = None
                           ) -> Optional[dict]:
    """Newest parseable ``entry_plan_*.json``, or None. ``before`` is carried from the
    shared loader; only the sell planner needs it in production. Never raises."""
    return plan_store.load_latest_plan(_PREFIX, dir_path, before=before)


def disarm_row(plan: dict, ticker: str) -> bool:
    """Mark ``ticker``'s armed row disarmed (in place). True if a row changed."""
    return plan_store.flip_status(plan, items_key="rows", id_key="ticker",
                                  ident=ticker, from_status=ROW_ARMED,
                                  to_status=ROW_DISARMED)


def plan_is_current(plan: dict, today=None) -> bool:
    """Executable only on the FIRST trading day after the plan's date. NOT the sells
    version of this check: entries can be armed on a WEEKEND (the Sunday hunt), so the
    rule is "exactly one business day inside ``(plan_date, today]``" — a Saturday or
    Sunday plan executes Monday; a Friday plan executes Monday; anything older is
    stale, and same-day execution is refused (orders are for the NEXT open)."""
    import pandas as pd
    try:
        d = pd.Timestamp(str(plan.get("date"))).normalize()
        if today is None:
            t = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        else:
            t = pd.Timestamp(today).normalize()
        if t <= d:
            return False
        return len(pd.bdate_range(d + pd.Timedelta(days=1), t)) == 1
    except Exception:
        return False


def autobuy_enabled(env: Optional[dict] = None) -> bool:
    return plan_store.env_enabled(AUTOBUY_ENV, env)


def execute_entry_plan(plan: dict, *, submit: Callable[[dict], dict],
                       gate: Optional[dict], held_by_symbol: Dict[str, int],
                       today=None, enabled: Optional[bool] = None) -> dict:
    """Submit AT MOST ONE still-armed row. Mutates ``plan`` in place; the caller
    persists it.

    ``submit(row)`` sends one buy through the real plan-submit path (limit + GTC OTO
    stop; its own pending-buy/tradability/cap guards all re-check) and returns
    ``{status, detail}``. Guard order: ``AUTOBUY`` env (ships dark) → freshness →
    progressive-exposure gate, which here FAILS CLOSED (``gate`` None or not open ⇒
    every armed row is skipped — the unattended path never buys on unknown state).
    Bullet semantics: only a ``submitted`` result consumes the one-per-day bullet; an
    already-held row or a ``skipped`` result moves on to the NEXT armed row; a
    ``failed`` result stops the walk (no blind retry — mirror of the sells doctrine).
    Idempotent: a plan already carrying a submitted row submits nothing more."""
    if enabled is None:
        enabled = autobuy_enabled()
    summary = {"status": "ok", "submitted": [], "skipped": [], "disarmed": [],
               "failed": []}
    if not enabled:
        summary["status"] = "disabled"
        return summary
    if not plan_is_current(plan, today=today):
        summary["status"] = "stale"
        return summary

    import pandas as pd
    gate_open = bool(gate) and gate.get("open") is True
    bullet_used = any(r.get("status") == ROW_SUBMITTED for r in plan.get("rows", []))

    for r in plan.get("rows", []):
        t, status = r.get("ticker"), r.get("status")
        if status == ROW_DISARMED:
            summary["disarmed"].append(t)
            continue
        if status != ROW_ARMED:
            continue
        if not gate_open:
            r["status"] = ROW_SKIPPED
            r["detail"] = ("progressive-exposure gate closed"
                           if gate else "gate unknown — unattended path fails closed")
            summary["skipped"].append(t)
            continue
        if bullet_used:
            r["status"] = ROW_SKIPPED
            r["detail"] = "one entry per day — bullet already used"
            summary["skipped"].append(t)
            continue
        if int(held_by_symbol.get(t, 0)) > 0:
            r["status"] = ROW_SKIPPED
            r["detail"] = "already held at execution"
            summary["skipped"].append(t)
            continue
        try:
            res = submit(dict(r))
        except Exception as e:
            res = {"status": "failed", "detail": str(e)}
        if res.get("status") == "submitted":
            r["status"] = ROW_SUBMITTED
            r["detail"] = res.get("detail", "")
            summary["submitted"].append(t)
            bullet_used = True
        elif res.get("status") == "failed":
            r["status"] = ROW_FAILED
            r["detail"] = res.get("detail", "")
            summary["failed"].append(t)
            break                       # no blind retry; later rows stay armed/expire
        else:
            r["status"] = ROW_SKIPPED
            r["detail"] = res.get("detail", str(res.get("status")))
            summary["skipped"].append(t)

    plan["executed_at"] = pd.Timestamp.now(tz="America/New_York").isoformat()
    if not gate_open:
        summary["status"] = "gate_closed"
    elif summary["failed"]:
        summary["status"] = "partial" if summary["submitted"] else "failed"
    return summary


def format_plan(plan: dict) -> str:
    """ASCII-only console rendering (journald-safe, same convention as triggers)."""
    lines = [f"ENTRY PLAN  {plan.get('date', '?')}  rows={len(plan.get('rows', []))}"]
    for r in plan.get("rows", []):
        lines.append(f"  {r.get('status', '?').upper():>9}  {r.get('ticker')} "
                     f"x{r.get('shares')}  limit {r.get('limit_price')}  "
                     f"stop {r.get('stop_price')}"
                     + (f"  [{r['detail']}]" if r.get("detail") else ""))
    for n in plan.get("notes", []):
        lines.append(f"  note: {n}")
    if not plan.get("rows"):
        lines.append("  nothing armed")
    return "\n".join(lines)
