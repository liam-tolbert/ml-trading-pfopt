"""Automated sell planning + execution from the P1-P4 sell pillars.

Two-phase, matching the operating rules (decision at the settled close, order at the
next open):

* **Evening** (``sell_job.py plan``, after the ~16:30 settled trigger run):
  :func:`build_sell_plan` turns each held position's pillars into a *plan* —
  full-exit orders for name-specific hard fails (P1, P2, P4). P3 (the tape) and every
  warn are recorded but never traded automatically. P2 must fail on TWO consecutive
  settled closes before it plans a sell (the strict template is known to flip for a
  day on knife-edge SMA noise); the streak is read from the prior day's plan snapshot.
* **Overnight veto**: the plan is a JSON file the Positions page renders with per-order
  Veto buttons; a vetoed order is kept in the file (audit trail) but never submitted.
* **Morning** (``sell_job.py execute``, ~09:25 ET): :func:`execute_sell_plan` submits a
  market SELL for every still-planned order via the stop-aware sell flow (cancel the
  covering GTC stop, sell, re-arm any remainder); placed pre-open, the order queues for
  the opening print. Refuses stale plans and requires the ``AUTOSELL`` env var — the
  feature ships dark until armed in ``.env``.

Plan files live beside the trigger reports (``sell_plan_YYYY-MM-DD.json`` in
``cache.TRIGGERS_DIR``) so the test suite's existing TRIGGERS_DIR patching keeps
AppTests away from real state. Paper account only, like every trade path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.stock_screener.cockpit import cache

AUTOSELL_ENV = "AUTOSELL"
_TRUTHY = {"1", "true", "yes", "on"}

# Pillars whose hard fail plans an automatic full exit. P3 is market-wide (a regime
# flip would liquidate the whole book) — deliberately excluded, report-only.
ACTIONABLE_PILLARS = ("P1", "P2", "P4")
# Pillars that act on their FIRST failing settled close. P2 is absent on purpose: the
# strict template flips for a day on knife-edge SMA noise, so it needs a 2-close streak.
IMMEDIATE_PILLARS = ("P1", "P4")

ORDER_PLANNED = "planned"
ORDER_VETOED = "vetoed"
ORDER_SUBMITTED = "submitted"
ORDER_FAILED = "failed"
ORDER_SKIPPED = "skipped"


def _today_iso(today=None) -> str:
    import pandas as pd
    if today is None:
        t = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    else:
        t = pd.Timestamp(today).normalize()
    return t.date().isoformat()


def build_sell_plan(positions: List[dict], pillars: Dict[str, dict], *,
                    prior_plan: Optional[dict] = None, today=None) -> dict:
    """Turn per-position pillar reads into the evening sell plan. Pure.

    ``positions``: :func:`trade.fetch_positions`-shaped dicts (``symbol``/``qty`` used).
    ``pillars``: ``{symbol: sell_pillars(...) result}``. ``prior_plan``: the previous
    trading day's plan (its pillar snapshot supplies the P2 streak); pass None when
    there is none — a first P2 fail then only starts the streak, never sells.

    Every position gets a snapshot row (tomorrow's streak needs today's statuses even
    for names with no order). Unknown pillars never trade — missing data is not a
    signal. Orders are always FULL exits; qty is re-read at execution time anyway."""
    prior_snap = (prior_plan or {}).get("snapshot", {})
    snapshot: Dict[str, dict] = {}
    orders: List[dict] = []
    notes: List[str] = []

    for pos in positions:
        sym = pos.get("symbol")
        held = int(pos.get("qty") or 0)
        pil = pillars.get(sym) or {}
        snapshot[sym] = {k: {"status": (pil.get(k) or {}).get("status", "unknown"),
                             "detail": (pil.get(k) or {}).get("detail", "")}
                         for k in ("P1", "P2", "P3", "P4")}
        if held < 1:
            continue

        reasons: List[str] = []
        for k in ACTIONABLE_PILLARS:
            p = snapshot[sym][k]
            if p["status"] != "fail":
                continue
            if k in IMMEDIATE_PILLARS:
                reasons.append(f"{k} fail: {p['detail']}")
            else:
                prev = ((prior_snap.get(sym) or {}).get(k) or {}).get("status")
                if prev == "fail":
                    reasons.append(f"{k} fail (2nd consecutive close): {p['detail']}")
                else:
                    notes.append(f"{sym}: {k} first failing close - streak started, "
                                 "no order yet")
        warn_only = [f"{k} warn: {snapshot[sym][k]['detail']}"
                     for k in ("P1", "P2", "P3", "P4")
                     if snapshot[sym][k]["status"] == "warn"]
        p3 = snapshot[sym]["P3"]
        if p3["status"] == "fail":
            notes.append(f"{sym}: P3 fail ({p3['detail']}) - tape is report-only, "
                         "de-grossing stays a human call")
        if warn_only and not reasons:
            notes.append(f"{sym}: " + "; ".join(warn_only))

        if reasons:
            orders.append({"symbol": sym, "qty": held, "reasons": reasons,
                           "status": ORDER_PLANNED, "detail": ""})

    import pandas as pd
    return {"date": _today_iso(today),
            "generated_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "orders": orders, "snapshot": snapshot, "notes": notes,
            "executed_at": None}


def sell_plan_path(date_iso: str, dir_path=None) -> Path:
    d = Path(dir_path if dir_path is not None else cache.TRIGGERS_DIR)
    return d / f"sell_plan_{date_iso}.json"


def save_sell_plan(plan: dict, dir_path=None) -> Path:
    """Atomic write (tmp + ``os.replace``) — the evening CLI, the morning executor, and
    a page Veto click run in separate processes against the same day-file."""
    path = sell_plan_path(plan["date"], dir_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_latest_sell_plan(dir_path=None, *, before: Optional[str] = None
                          ) -> Optional[dict]:
    """Newest parseable ``sell_plan_*.json``, or None. ``before`` (ISO date) skips plans
    dated >= it — the evening planner uses ``before=today`` so a same-day rerun reads
    yesterday's snapshot for the P2 streak, not its own earlier output. Never raises."""
    try:
        d = Path(dir_path if dir_path is not None else cache.TRIGGERS_DIR)
        for path in sorted(d.glob("sell_plan_*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                if before is not None and str(data.get("date", "")) >= before:
                    continue
                return data
            except Exception:
                continue
    except Exception:
        pass
    return None


def veto_order(plan: dict, symbol: str) -> bool:
    """Mark ``symbol``'s planned order vetoed (in place). True if an order changed."""
    changed = False
    for o in plan.get("orders", []):
        if o.get("symbol") == symbol and o.get("status") == ORDER_PLANNED:
            o["status"] = ORDER_VETOED
            changed = True
    return changed


def plan_is_current(plan: dict, today=None) -> bool:
    """A plan is executable only on the FIRST trading day after its evaluation date —
    Monday morning executes Friday evening's plan; anything older is stale (the pillars
    were true of a market two sessions gone) and same-day execution is refused (orders
    are for the NEXT open by design)."""
    import pandas as pd
    try:
        d = pd.Timestamp(str(plan.get("date"))).normalize()
        if today is None:
            t = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        else:
            t = pd.Timestamp(today).normalize()
        if t <= d:
            return False
        return len(pd.bdate_range(d, t)) - 1 == 1
    except Exception:
        return False


def autosell_enabled(env: Optional[dict] = None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(AUTOSELL_ENV, "")).strip().lower() in _TRUTHY


def execute_sell_plan(plan: dict, *, submit: Callable[[str, int], dict],
                      held_by_symbol: Dict[str, int], today=None,
                      enabled: Optional[bool] = None) -> dict:
    """Submit every still-planned order. Mutates ``plan`` in place and returns a result
    summary; the caller persists the updated plan.

    ``submit(symbol, qty)`` is the stop-aware sell (``trade.submit_position_sell``) —
    injected so the logic tests offline. ``held_by_symbol`` is the account's CURRENT
    holdings: qty is clamped to it (the plan's count may be a day old) and a no-longer-
    held name is skipped, never shorted. Guards, in order: the ``AUTOSELL`` env gate
    (ships dark), then plan freshness (:func:`plan_is_current`). Idempotent — only
    ``planned`` orders act; a double-fire submits nothing twice, and a FAILED order
    stays failed for a human (an ambiguous broker failure may have partially acted —
    blind auto-retry could double-sell)."""
    if enabled is None:
        enabled = autosell_enabled()
    summary = {"status": "ok", "submitted": [], "vetoed": [], "skipped": [],
               "failed": []}
    if not enabled:
        summary["status"] = "disabled"
        return summary
    if not plan_is_current(plan, today=today):
        summary["status"] = "stale"
        return summary

    import pandas as pd
    for o in plan.get("orders", []):
        sym, status = o.get("symbol"), o.get("status")
        if status == ORDER_VETOED:
            summary["vetoed"].append(sym)
            continue
        if status != ORDER_PLANNED:
            summary["skipped"].append(sym)
            continue
        held = int(held_by_symbol.get(sym, 0))
        if held < 1:
            o["status"] = ORDER_SKIPPED
            o["detail"] = "no longer held"
            summary["skipped"].append(sym)
            continue
        qty = min(int(o.get("qty") or 0), held) or held
        try:
            res = submit(sym, qty)
        except Exception as e:            # a raise mid-loop must not strand the rest
            res = {"status": "failed", "detail": str(e)}
        if res.get("status") == "submitted":
            o["status"] = ORDER_SUBMITTED
            o["detail"] = res.get("detail", "")
            summary["submitted"].append(sym)
        else:
            o["status"] = ORDER_FAILED
            o["detail"] = res.get("detail", str(res.get("status")))
            summary["failed"].append(sym)
    plan["executed_at"] = pd.Timestamp.now(tz="America/New_York").isoformat()
    if summary["failed"]:
        summary["status"] = "partial" if summary["submitted"] else "failed"
    return summary


def format_plan(plan: dict) -> str:
    """ASCII-only console rendering (journald/log-safe, same convention as triggers)."""
    lines = [f"SELL PLAN  {plan.get('date', '?')}  "
             f"orders={len(plan.get('orders', []))}"]
    for o in plan.get("orders", []):
        lines.append(f"  {o.get('status', '?').upper():>9}  {o.get('symbol')} "
                     f"x{o.get('qty')}  - " + "; ".join(o.get("reasons", []))
                     + (f"  [{o['detail']}]" if o.get("detail") else ""))
    for n in plan.get("notes", []):
        lines.append(f"  note: {n}")
    if not plan.get("orders") and not plan.get("notes"):
        lines.append("  all pillars green - nothing to do")
    return "\n".join(lines)
