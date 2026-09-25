"""Armed entries: pre-authorized next-open buys, the buy-side mirror of sells.py.

The judgment stays human, at the evening ritual. The user builds a LIMIT trade plan in
the panel as usual, then arms it instead of (or after) submitting. That writes tonight's
``entry_plan_YYYY-MM-DD.json`` beside the trigger reports. A pre-open CLI
(``entry_job.py execute``, ~09:26 ET) then submits at most one still-armed row: a limit
at the buy-zone top, which enforces no-chase, with the GTC OTO stop leg. It first clears,
in order, the ``AUTOBUY`` env gate (the feature ships dark), plan freshness (next
business day only) and the progressive-exposure gate. That gate fails closed on this
unattended path: unknown state means no buys. Per-row disarm in the app is the overnight
veto; rows never executed expire with the plan.

Plan files live in ``cache.TRIGGERS_DIR``, so the test suite's patching keeps AppTests
away from real state. Paper account only, like every trade path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.stock_screener.cockpit import plan_store
from src.stock_screener.cockpit.doctrine import MAX_LOSS_FROM_FILL, MAX_ORDER_ADV_PCT
from src.stock_screener.cockpit.trade import fill_floor, stop_within_max_loss

AUTOBUY_ENV = "AUTOBUY"
_PREFIX = "entry_plan"

ROW_ARMED = "armed"
ROW_DISARMED = "disarmed"
ROW_SUBMITTED = "submitted"
ROW_FAILED = "failed"
ROW_SKIPPED = "skipped"


def _row(o: dict) -> dict:
    """One armed row from a built plan entry, coerced to plain JSON types: the scan
    payload leaks numpy scalars into plan rows."""
    def f(v):
        return None if v is None else float(v)
    return {"ticker": str(o["ticker"]), "shares": int(o["shares"]),
            "price": f(o.get("price")), "pivot": f(o.get("pivot")),
            "limit_price": f(o.get("limit_price")), "stop_price": f(o.get("stop_price")),
            "est_value": f(o.get("est_value")), "adv_usd": f(o.get("adv_usd")),
            "earnings_in": None if o.get("earnings_in") is None
            else int(o["earnings_in"]),
            "status": ROW_ARMED, "detail": ""}


def build_entry_plan(final_rows: List[dict], today=None) -> dict:
    """Tonight's armed-entry plan from the panel's final rows. Pure apart from the
    wall-clock ``generated_at``. Returns ``{date, generated_at, rows, notes,
    executed_at}``; ``notes`` gives each refused row's reason.

    Only genuine buy rows arm: ``shares >= 1``, not ``rearm_only``/``stop_only``, and
    both a positive ``limit_price`` (the no-chase cap is the entry mechanic; a market
    row MUST NOT arm) and a positive ``stop_price`` below it (the OTO leg), at most
    ``MAX_LOSS_FROM_FILL`` below the limit, and a notional within ``MAX_ORDER_ADV_PCT`` of
    ``adv_usd`` when that is known. The executor runs unattended, so a bad row MUST be
    refused at arming, not at 09:26. Order is preserved — the executor walks rows
    top-down, so the panel's ordering is the ranking."""
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
        if not stop_within_max_loss(stop, lim):
            skipped.append(f"{t}: stop {float(stop):,.2f} is more than "
                           f"{MAX_LOSS_FROM_FILL * 100:.0f}% below the limit "
                           f"{float(lim):,.2f} — raise it to ≥ {fill_floor(lim):,.2f}")
            continue
        adv = o.get("adv_usd")
        notional = int(o["shares"]) * float(lim)
        if adv and notional > MAX_ORDER_ADV_PCT * float(adv):
            skipped.append(f"{t}: {int(o['shares'])} sh at {float(lim):,.2f} is "
                           f"{notional / float(adv) * 100:.1f}% of a day's $ volume — max "
                           f"{MAX_ORDER_ADV_PCT * 100:.0f}%")
            continue
        rows.append(_row(o))

    import pandas as pd
    return {"date": plan_store.today_iso(today),
            "generated_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "rows": rows, "notes": skipped, "executed_at": None}


# Storage is shared with sells.py (see plan_store). These wrappers give the app, the
# pages and the CLI entry-side names.
def entry_plan_path(date_iso: str, dir_path=None) -> Path:
    return plan_store.plan_path(_PREFIX, date_iso, dir_path)


def save_entry_plan(plan: dict, dir_path=None) -> Path:
    """Atomic write; returns the path. The arming click, a disarm click and the morning
    executor write the same day-file from separate processes."""
    return plan_store.save_plan(_PREFIX, plan, dir_path)


def load_latest_entry_plan(dir_path=None, *, before: Optional[str] = None
                           ) -> Optional[dict]:
    """Newest parseable ``entry_plan_*.json``, or None. ``before`` passes through to the
    shared loader; only the sell planner uses it in production. Never raises."""
    return plan_store.load_latest_plan(_PREFIX, dir_path, before=before)


def disarm_row(plan: dict, ticker: str) -> bool:
    """Mark ``ticker``'s armed row disarmed (in place). True if a row changed."""
    return plan_store.flip_status(plan, items_key="rows", id_key="ticker",
                                  ident=ticker, from_status=ROW_ARMED,
                                  to_status=ROW_DISARMED)


def plan_is_current(plan: dict, today=None) -> bool:
    """True only on the first business day after the plan's date; False on any error.

    This differs from the sells check: entries can be armed on a weekend (the Sunday
    hunt). The rule is exactly one business day in ``(plan_date, today]``; holidays
    count as business days. A Friday, Saturday or Sunday plan executes Monday. Anything
    older is stale. Same-day execution is refused: orders are for the next open."""
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
    """Submit at most one still-armed row. Mutates ``plan`` in place; the caller
    persists it.

    ``submit(row)`` sends one buy through the real plan-submit path (limit plus GTC OTO
    stop, with its own pending-buy, tradability and cap guards) and returns ``{status,
    detail}``. An exception from it counts as ``failed``.

    Guards, in order: ``enabled`` (default: the ``AUTOBUY`` env), plan freshness, then
    the progressive-exposure gate. The gate fails closed: with ``gate`` None or not open,
    every armed row is skipped. The unattended path MUST NOT buy on unknown state.

    Only a ``submitted`` result uses the day's one bullet. An already-held row or a
    ``skipped`` result moves on to the next armed row. A ``failed`` result stops the
    walk: no blind retry, as on the sell side. A plan that already has a submitted row
    submits nothing more.

    Returns ``{status, submitted, skipped, disarmed, failed}``. ``status`` is ``ok``,
    ``disabled``, ``stale``, ``gate_closed``, ``partial`` or ``failed``."""
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
