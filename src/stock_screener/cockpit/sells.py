"""Automated sell planning and execution from the P1-P4 sell pillars.

Two phases, matching the operating rules: decide at the settled close, order at the
next open.

* **Evening** (``sell_job.py plan`` at 16:15 ET, after the 16:10 settled trigger run):
  :func:`build_sell_plan` turns each held position's pillars into a *plan* of
  full-exit orders for name-specific hard fails (P1, P2, P4). P3 (the tape) and every
  warn are recorded but never traded automatically. P2 MUST fail on two consecutive
  settled closes before it plans a sell: the strict template flips for a day on
  knife-edge SMA noise. The streak is read from the prior day's plan snapshot.
* **Overnight veto**: the Positions page renders the plan's JSON file with a Veto button
  per order. A vetoed order stays in the file as an audit trail but is never submitted.
* **Morning** (``sell_job.py execute``, ~09:25 ET): :func:`execute_sell_plan` submits a
  market SELL for every still-planned order via the stop-aware sell flow (cancel the
  covering GTC stop, sell, re-arm any remainder). Placed pre-open, the order queues for
  the opening print. It refuses stale plans and requires the ``AUTOSELL`` env var; the
  feature ships dark until armed in ``.env``.

Plan files live beside the trigger reports (``sell_plan_YYYY-MM-DD.json`` in
``cache.TRIGGERS_DIR``), so the test suite's TRIGGERS_DIR patching keeps AppTests away
from real state. Paper account only, like every trade path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.stock_screener.cockpit import plan_store

AUTOSELL_ENV = "AUTOSELL"
_PREFIX = "sell_plan"

# Pillars whose hard fail plans an automatic full exit. P3 is market-wide and MUST stay
# report-only: a regime flip would liquidate the whole book.
ACTIONABLE_PILLARS = ("P1", "P2", "P4")
# Pillars that act on their first failing settled close. P2 needs a 2-close streak: the
# strict template flips for a day on knife-edge SMA noise.
IMMEDIATE_PILLARS = ("P1", "P4")

ORDER_PLANNED = "planned"
ORDER_VETOED = "vetoed"
ORDER_SUBMITTED = "submitted"
ORDER_FAILED = "failed"
ORDER_SKIPPED = "skipped"


def build_sell_plan(positions: List[dict], pillars: Dict[str, dict], *,
                    prior_plan: Optional[dict] = None, today=None,
                    market: Optional[dict] = None) -> dict:
    """Turn per-position pillar reads into the evening sell plan. Pure apart from the
    wall-clock ``generated_at``. Returns ``{date, generated_at, orders, snapshot, notes,
    executed_at}``, plus ``market`` when ``market`` is given.

    ``positions``: :func:`trade.fetch_positions`-shaped dicts (``symbol``/``qty`` used).
    ``pillars``: ``{symbol: sell_pillars(...) result}``. ``prior_plan``: the previous
    trading day's plan; its pillar snapshot supplies the P2 streak. With None, a first
    P2 fail only starts the streak and never sells.

    Every position gets a snapshot row: tomorrow's streak needs today's statuses even
    for names with no order. Unknown pillars never trade; missing data is not a
    signal. Pillar orders are full exits (``exit: "full"``); qty is re-read at execution.

    ``market`` is ``{spy_note, streak}``: the trigger report's SPY note and
    :func:`advisories.spy_confirm_streak`. It adds a ``plan["market"]`` record, which the
    next plan dates the turn from, and a note when the tape needs one. On the evening SPY
    first closes in Stage 4 (:func:`advisories.market_turn`), the note says to reduce.
    With ``doctrine.MARKET_TURN_CAN_TRADE`` on, the plan also sells
    ``MARKET_TURN_REDUCE_FRACTION`` of each position without a full exit. A single share
    gets a note only. A plan MUST hold at most one order per symbol, and a full exit wins,
    so a Veto still means "don't sell this name tomorrow"."""
    from src.stock_screener.cockpit import advisories, doctrine
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
            orders.append({"symbol": sym, "qty": held, "exit": "full", "reasons": reasons,
                           "status": ORDER_PLANNED, "detail": ""})

    market_rec = None
    if market is not None:
        turn = advisories.market_turn(market.get("spy_note"),
                                      (prior_plan or {}).get("market"),
                                      has_prior_plan=prior_plan is not None)
        stk = market.get("streak") or {}
        market_rec = {"spy_phase": turn["spy_phase"], "turn": turn["turn"],
                      "streak": stk.get("streak")}
        frac = doctrine.MARKET_TURN_REDUCE_FRACTION
        if turn["turn"]:
            notes.append(f"MARKET: SPY closed in Stage 4 today — the backtest's one validated "
                         f"exit. Reduce exposure (sell {frac:.0%} of each position; the "
                         "backtest exited fully) and add nothing until SPY recovers for "
                         f"{doctrine.REGIME_CONFIRM_DAYS} sessions.")
            if doctrine.MARKET_TURN_CAN_TRADE:
                full = {o["symbol"] for o in orders}
                for pos in positions:
                    sym, held = pos.get("symbol"), int(pos.get("qty") or 0)
                    if sym in full or held < 1:
                        continue
                    qty = int(held * frac)
                    if qty < 1:
                        notes.append(f"{sym}: market turn — a single share, no partial "
                                     "sell (decide by hand)")
                        continue
                    orders.append({"symbol": sym, "qty": qty, "exit": "partial",
                                   "reasons": [f"market turn: SPY entered Stage 4 — sell "
                                               f"{qty}/{held}"],
                                   "status": ORDER_PLANNED, "detail": ""})
        elif turn["unconfirmed"]:
            notes.append("MARKET: SPY is in Stage 4, but with no earlier plan on file the "
                         "turn can't be dated — the book may already have been reduced; "
                         "decide by hand.")
        elif turn["stage4"]:
            notes.append("MARKET: SPY still in Stage 4 — stay defensive, no new buys.")
        elif stk.get("streak") is not None and not stk.get("satisfied"):
            notes.append(f"MARKET: SPY {stk['streak']}/{doctrine.REGIME_CONFIRM_DAYS} "
                         "sessions back in Stage 1-2 — the re-entry lag isn't met; don't "
                         "add yet"
                         + (" (with breadth)" if stk.get("breadth") else " (SPY only)")
                         + ".")

    import pandas as pd
    plan = {"date": plan_store.today_iso(today),
            "generated_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
            "orders": orders, "snapshot": snapshot, "notes": notes,
            "executed_at": None}
    if market_rec is not None:
        plan["market"] = market_rec
    return plan


# Storage is shared with entries.py (see plan_store). These wrappers give the pages and
# the CLI sell-side names.
def sell_plan_path(date_iso: str, dir_path=None) -> Path:
    return plan_store.plan_path(_PREFIX, date_iso, dir_path)


def save_sell_plan(plan: dict, dir_path=None) -> Path:
    """Atomic write; returns the path. The evening CLI, the morning executor and a page
    Veto click write the same day-file from separate processes."""
    return plan_store.save_plan(_PREFIX, plan, dir_path)


def load_latest_sell_plan(dir_path=None, *, before: Optional[str] = None
                          ) -> Optional[dict]:
    """Newest parseable ``sell_plan_*.json``, or None. Never raises. ``before`` (ISO date)
    skips plans dated on or after it. The evening planner passes ``before=today``, so a
    same-day rerun reads the previous plan's snapshot for the P2 streak, not its own
    output."""
    return plan_store.load_latest_plan(_PREFIX, dir_path, before=before)


def veto_order(plan: dict, symbol: str) -> bool:
    """Mark ``symbol``'s planned order vetoed (in place). True if an order changed."""
    return plan_store.flip_status(plan, items_key="orders", id_key="symbol",
                                  ident=symbol, from_status=ORDER_PLANNED,
                                  to_status=ORDER_VETOED)


def plan_is_current(plan: dict, today=None) -> bool:
    """True when ``today`` is the next business day after a weekday plan's date; False
    on any error. Monday morning executes Friday evening's plan. Anything older is
    stale: its pillars describe a market two sessions gone. Same-day execution is
    refused: orders are for the next open. Holidays count as business days."""
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
    return plan_store.env_enabled(AUTOSELL_ENV, env)


def execute_sell_plan(plan: dict, *, submit: Callable[[str, int], dict],
                      held_by_symbol: Dict[str, int], today=None,
                      enabled: Optional[bool] = None) -> dict:
    """Submit every still-planned order. Mutates ``plan`` in place; the caller persists
    it. Returns ``{status, submitted, vetoed, skipped, failed}``; ``status`` is ``ok``,
    ``disabled``, ``stale``, ``partial`` or ``failed``.

    ``submit(symbol, qty)`` is the stop-aware sell (``trade.submit_position_sell``),
    injected so the logic tests offline. An order's ``remainder_stop`` is passed as a
    keyword. ``held_by_symbol`` is the account's CURRENT holdings: qty is clamped to it
    (the plan's count may be a day old) and a no-longer-held name is skipped, never
    shorted. A ``partial`` order sells its clamped quantity, or is skipped at 0; it MUST
    NOT widen to a full exit. Guards, in order: the ``AUTOSELL`` env gate
    (ships dark), then plan freshness (:func:`plan_is_current`). Idempotent: only
    ``planned`` orders act, so a double-fire submits nothing twice. A failed order MUST
    stay failed for a human: an ambiguous broker failure may have partly acted, and a
    blind retry could double-sell."""
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
        plan_qty = int(o.get("qty") or 0)
        if o.get("exit") == "partial":
            # MUST NOT reach the full-exit branch: its `or held` turns 0 into "sell all".
            qty = min(plan_qty, held)
            if qty < 1:
                o["status"] = ORDER_SKIPPED
                o["detail"] = "nothing to sell"
                summary["skipped"].append(sym)
                continue
        else:                             # a full exit; a plan without "exit" is full
            qty = min(plan_qty, held) or held
        kw = ({"remainder_stop": o["remainder_stop"]}
              if o.get("remainder_stop") is not None else {})
        try:
            res = submit(sym, qty, **kw)
        except Exception as e:            # a raise mid-loop MUST NOT strand the rest
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
                     f"x{o.get('qty')}{' (partial)' if o.get('exit') == 'partial' else ''}"
                     "  - " + "; ".join(o.get("reasons", []))
                     + (f"  [{o['detail']}]" if o.get("detail") else ""))
    for n in plan.get("notes", []):
        lines.append(f"  note: {n}")
    if not plan.get("orders") and not plan.get("notes"):
        lines.append("  all pillars green - nothing to do")
    return "\n".join(lines)
