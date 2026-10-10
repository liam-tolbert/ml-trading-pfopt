"""Automated sell planning and execution from the P1-P4 sell pillars.

The pillars are the four sell checks :func:`trade.sell_pillars` runs on each held
position. Two phases, matching the operating rules: decide at the settled close, order
at the next open.

* **Evening** (``sell_job.py plan`` at 16:15 ET, after the 16:10 trigger run on the
  settled close): :func:`build_sell_plan` turns each held position's pillars into a
  *plan* of full-exit orders for fails specific to that stock (P1, P2, P4). P3 (the
  whole market) and every warn are recorded but never traded automatically. P2 MUST
  fail on two settled closes in a row before it plans a sell: the strict Trend Template
  can fail for a single day when price sits right on an SMA. The streak is read from
  the prior day's plan snapshot.
* **Overnight veto**: the Positions page shows the plan's JSON file with a Veto button
  per order. A vetoed order stays in the file as a record but is never submitted.
* **Morning** (``sell_job.py execute``, ~09:25 ET): :func:`execute_sell_plan` submits a
  market SELL for every still-planned order through the stop-aware sell (cancel the GTC
  stop on those shares, sell, place a new stop on any shares left). Placed before the
  open, the order waits for the opening trade. It refuses out-of-date plans and does
  nothing unless ``AUTOSELL`` is set in ``.env``; the feature is off by default.

Plan files live beside the trigger reports (``sell_plan_YYYY-MM-DD.json`` in
``cache.TRIGGERS_DIR``), so when the test suite patches TRIGGERS_DIR, AppTests never
touch the real files. Paper account only, like every trade path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.stock_screener.cockpit import plan_store

AUTOSELL_ENV = "AUTOSELL"
_PREFIX = "sell_plan"

# Pillars whose fail plans an automatic full exit. P3 reads the whole market and MUST
# stay report-only: one market-wide turn would sell every position at once.
ACTIONABLE_PILLARS = ("P1", "P2", "P4")
# Pillars that act on their first failing settled close. P2 needs two failing closes in a
# row: the strict Trend Template can fail for a single day when price sits right on an
# SMA.
IMMEDIATE_PILLARS = ("P1", "P4")

ORDER_PLANNED = "planned"
ORDER_VETOED = "vetoed"
ORDER_SUBMITTED = "submitted"
ORDER_FAILED = "failed"
ORDER_SKIPPED = "skipped"


def build_sell_plan(positions: List[dict], pillars: Dict[str, dict], *,
                    prior_plan: Optional[dict] = None, today=None,
                    market: Optional[dict] = None,
                    strength: Optional[Dict[str, dict]] = None) -> dict:
    """Turn per-position pillar reads into the evening sell plan. No file or network
    access; only ``generated_at`` reads the clock. Returns ``{date, generated_at, orders,
    snapshot, notes, executed_at}``, plus ``market`` when ``market`` is given.

    ``positions``: dicts shaped like :func:`trade.fetch_positions` returns
    (``symbol``/``qty`` used). ``pillars``: ``{symbol: sell_pillars(...) result}``.
    ``prior_plan``: the previous trading day's plan; its pillar snapshot supplies the P2
    streak. With None, a first P2 fail only starts the streak and never sells.

    Every position gets a snapshot row: tomorrow's streak needs today's statuses even
    for stocks with no order. Unknown pillars never trade; missing data never causes a
    sale. Pillar orders sell the whole position (``exit: "full"``); qty is re-read at
    execution.

    ``market`` is ``{spy_note, streak}``: the trigger report's SPY note and
    :func:`advisories.spy_confirm_streak`. It adds a ``plan["market"]`` record, which the
    next plan uses to tell which evening the market turned, and a note when the market
    calls for one. On the evening SPY first closes in Stage 4
    (:func:`advisories.market_turn`), the note says to reduce. With
    ``doctrine.MARKET_TURN_CAN_TRADE`` on, the plan also sells
    ``MARKET_TURN_REDUCE_FRACTION`` of each position without a full exit. A single share
    gets a note only.

    ``strength`` is ``{symbol: {signs, avg_entry, trimmed}}`` from
    :func:`advisories.strength_signs`; ``trimmed`` means part of the position was already
    sold. Every holding with signs gets a note. With ``doctrine.STRENGTH_CAN_TRADE`` on,
    ``STRENGTH_SIGNS_TO_SELL`` signs plan a partial sale of ``STRENGTH_SELL_FRACTION``.
    Its ``remainder_stop``, the stop for the shares kept, is ``avg_entry``. A position
    already trimmed gets no second sale.

    A plan MUST hold at most one order per symbol, and a full exit wins, so a Veto still
    means "don't sell this stock tomorrow"."""
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
            notes.append(f"MARKET: SPY closed in Stage 4 today. Reduce exposure (sell "
                         f"{frac:.0%} of each position) and add nothing until SPY recovers "
                         f"for {doctrine.REGIME_CONFIRM_DAYS} sessions.")
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

    for pos in positions:
        sym, held = pos.get("symbol"), int(pos.get("qty") or 0)
        rd = (strength or {}).get(sym) or {}
        signs = rd.get("signs") or []
        if not signs or held < 1:
            continue
        notes.append(f"{sym}: selling-into-strength signs ({len(signs)}): "
                     + "; ".join(signs))
        if (not doctrine.STRENGTH_CAN_TRADE
                or len(signs) < doctrine.STRENGTH_SIGNS_TO_SELL
                or sym in {o["symbol"] for o in orders}):
            continue
        # Once per position: a second strength sale every evening would empty it.
        if rd.get("trimmed"):
            notes.append(f"{sym}: already trimmed - no second strength sale")
            continue
        qty = int(held * doctrine.STRENGTH_SELL_FRACTION)
        if qty < 1:
            notes.append(f"{sym}: selling into strength - a single share, no partial "
                         "sell (decide by hand)")
            continue
        order = {"symbol": sym, "qty": qty, "exit": "partial",
                 "reasons": [f"selling into strength: {len(signs)} signs - sell "
                             f"{qty}/{held}, rest stopped at breakeven"],
                 "status": ORDER_PLANNED, "detail": ""}
        if rd.get("avg_entry"):
            order["remainder_stop"] = round(float(rd["avg_entry"]), 2)
        orders.append(order)

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
    """Write the plan to a temp file, then rename it over the dated file, so a reader
    never sees a half-written file. Returns the path. The evening CLI, the morning
    execute run and a page Veto click write the same file from separate processes."""
    return plan_store.save_plan(_PREFIX, plan, dir_path)


def load_latest_sell_plan(dir_path=None, *, before: Optional[str] = None
                          ) -> Optional[dict]:
    """Newest readable ``sell_plan_*.json``, or None. Never raises. ``before`` (ISO date)
    skips plans dated on or after it. The evening planner passes ``before=today``, so a
    second run on the same day reads the previous plan's snapshot for the P2 streak, not
    its own output."""
    return plan_store.load_latest_plan(_PREFIX, dir_path, before=before)


def veto_order(plan: dict, symbol: str) -> bool:
    """Mark ``symbol``'s planned order vetoed (in place). True if an order changed."""
    return plan_store.flip_status(plan, items_key="orders", id_key="symbol",
                                  ident=symbol, from_status=ORDER_PLANNED,
                                  to_status=ORDER_VETOED)


def plan_is_current(plan: dict, today=None) -> bool:
    """True when ``today`` is the next business day after a weekday plan's date; False
    on any error. Monday morning executes Friday evening's plan. Anything older is out
    of date: its pillars describe the market two trading days back. Same-day execution
    is refused: orders are for the next open. Holidays count as business days."""
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
    passed in so tests run offline. An order's ``remainder_stop`` is passed as a
    keyword. ``held_by_symbol`` is the account's CURRENT holdings: qty is capped at it
    (the plan's count may be a day old) and a stock no longer held is skipped, never
    sold short. A ``partial`` order sells its capped quantity, or is skipped at 0; it
    MUST NOT become a full exit. Checks, in order: the ``AUTOSELL`` setting (off by
    default), then whether the plan is current (:func:`plan_is_current`). Running it
    twice is safe: only ``planned`` orders act, so a second run submits nothing again. A
    failed order MUST stay failed until a person looks at it: an unclear broker failure
    may have partly gone through, and an automatic retry could sell twice."""
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
        except Exception as e:            # one order's error MUST NOT stop the rest
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
    """The plan as ASCII-only console text, safe for the systemd journal (same
    convention as triggers)."""
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
