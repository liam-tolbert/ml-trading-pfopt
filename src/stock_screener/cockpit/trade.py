"""Paper-trade the cockpit watchlist through Alpaca.

Sizing matches the Step-4 position sizer exactly: for each name,
``shares = floor((account $ × risk %) / (pivot − stop))`` — the same share count the
sizer shows at the bottom of the page, using each name's own pivot and stop. Orders are
plain market BUYs on the **paper** account (``alpaca_trader.connect()`` forces
``paper=True``); the existing guardrails apply — a per-order floor of $50 and a cap of 10%
of equity (an over-cap name is skipped, not fatal).

The plan builder (:func:`build_buy_plan`) is pure and network-free so it's unit-tested;
:func:`submit_buy_plan` is the thin side-effecting wrapper that talks to Alpaca and is
imported lazily (so the cockpit still loads when ``alpaca-py`` isn't installed).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

MIN_TRADE_USD = 50.0        # mirrors alpaca_trader.MIN_TRADE_USD (kept here so the pure
                            # plan builder needn't import alpaca-py)

# The cockpit trades a SEPARATE Alpaca paper account from the All-Weather mirror (which
# owns the shared ALPACA_API_KEY/SECRET pair). Each Alpaca paper account has its own key
# pair, so we select the "Minervini Trader" account by preferring its dedicated keys,
# falling back to the shared ones only if the dedicated pair isn't set.
#
# We accept TWO spellings of each name — the originally-documented ``ALPACA_MINERVINI_API_*``
# form and the ``ALPACA_API_KEY_*_MINERVINI`` / ``*_PAPER1`` form actually used in this repo's
# .env — because a silent drift between the .env names and the names the code reads routes
# every order to nowhere (both resolve to None -> TradeUnavailable) with no obvious cause.
# First non-empty match in each tuple wins.
MINERVINI_KEY_ENVS = "ALPACA_API_KEY_MINERVINI"
MINERVINI_SECRET_ENVS = "ALPACA_API_KEY_SECRET_MINERVINI"
SHARED_KEY_ENVS = ("ALPACA_API_KEY", "ALPACA_API_KEY_PAPER1")
SHARED_SECRET_ENVS = ("ALPACA_API_SECRET", "ALPACA_API_SECRET_PAPER1")


def _first_env(names: "Sequence[str] | str") -> Optional[str]:
    """Return the value of the first env var named in ``names`` that is set and non-empty.

    Accepts a single name (``str``) or a sequence of candidate names. A bare string is
    treated as ONE name — never iterated character-by-character, which would silently
    resolve a stray one-char env var (e.g. ``$_``) instead of the intended key.
    """
    if isinstance(names, str):
        names = (names,)
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


class TradeUnavailable(RuntimeError):
    """Alpaca can't be reached — package missing, or credentials absent from .env."""


def _connect_paper():
    """Return ``(paper TradingClient, using_dedicated)`` for the cockpit's account.

    Prefers the dedicated Minervini keys; falls back to the shared ALPACA_* pair. Always
    ``paper=True``. Raises :class:`TradeUnavailable` if alpaca-py or credentials are missing.
    """
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as e:
        raise TradeUnavailable(
            "alpaca-py is not installed — run `pip install alpaca-py` "
            "(or `conda env update -f environment.yml`).") from e
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ded_key = _first_env(MINERVINI_KEY_ENVS)
    ded_secret = _first_env(MINERVINI_SECRET_ENVS)
    key = ded_key or _first_env(SHARED_KEY_ENVS)
    secret = ded_secret or _first_env(SHARED_SECRET_ENVS)
    if not key or not secret:
        raise TradeUnavailable(
            "No Alpaca credentials in .env. Add the Minervini Trader paper account's keys as "
            "ALPACA_API_KEY_MINERVINI / ALPACA_API_KEY_SECRET_MINERVINI (each Alpaca paper "
            "account has its own key pair), or a shared ALPACA_API_KEY / ALPACA_API_SECRET pair.")
    return TradingClient(key, secret, paper=True), bool(ded_key and ded_secret)


def fetch_account_summary() -> dict:
    """Connect and read the target account so the UI can confirm *which* account will be
    traded before any order is sent. Returns ``{account_number, equity, cash,
    using_dedicated}``; raises :class:`TradeUnavailable` on missing package/credentials."""
    client, using_dedicated = _connect_paper()
    acct = client.get_account()
    return {
        "account_number": getattr(acct, "account_number", "?"),
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "using_dedicated": using_dedicated,
    }


SIZING_MODES = ("pct", "dollars", "shares")


def stop_is_valid(stop_price, price) -> bool:
    """A protective sell-stop is valid only strictly BELOW the reference price.

    Alpaca rejects a sell stop at/above the market (it would trigger instantly) and an OTO
    stop-loss leg that isn't below the entry. Used by the UI (live per-keystroke check) and
    re-checked in :func:`submit_buy_plan` against the last close.
    """
    return bool(stop_price and price and stop_price > 0 and stop_price < price)


def build_buy_plan(tickers: Sequence[str], payloads: Dict[str, dict], *,
                   mode: str, amount: float,
                   equity: Optional[float] = None) -> Tuple[List[dict], List[dict]]:
    """Size a market-BUY for EACH watchlisted name by the chosen ``mode``:

    * ``"pct"``     — ``amount`` % of the account ``equity`` per name (needs ``equity``);
    * ``"dollars"`` — ``amount`` dollars per name;
    * ``"shares"``  — exactly ``amount`` (whole) shares per name.

    Returns ``(plan, skipped)``. Each plan entry has ticker / shares / price / pivot /
    est_value / extended / stop_price; each skipped entry is ``{ticker, reason}``. A name is
    skipped when it isn't in the current scan, has no current price, sizes to < 1 share, or
    (for the two dollar-denominated modes) rounds to a notional under the $50 floor — the
    ``"shares"`` mode is exempt from that floor since the count is explicit. ``extended`` flags
    a price already above the no-chase buy zone (> pivot × 1.05); the caller surfaces it as a
    warning rather than skipping. ``stop_price`` is the app-computed protective stop
    (``levels["stop"]``, ~7-8% below pivot) or ``None`` if unavailable — the caller may edit it
    before submit; it is never used for sizing here. ``earnings_in`` (calendar days to the next
    scheduled report, from the scan payload; None = unknown) is carried through untouched so the
    caller can warn about buying into an imminent report — advisory only, never a skip.
    """
    if mode not in SIZING_MODES:
        raise ValueError(f"mode must be one of {SIZING_MODES}, got {mode!r}")
    plan: List[dict] = []
    skipped: List[dict] = []
    for t in dict.fromkeys(tickers):
        payload = payloads.get(t)
        if not payload:
            skipped.append({"ticker": t, "reason": "not in the current scan"})
            continue
        lv = payload.get("levels", {}) or {}
        df = payload.get("df")
        price = (float(df["Close"].iloc[-1])
                 if df is not None and len(df) else None)
        if not price or price <= 0:
            skipped.append({"ticker": t, "reason": "no current price"})
            continue

        if mode == "pct":
            if not equity or equity <= 0:
                skipped.append({"ticker": t, "reason": "account equity unavailable"})
                continue
            shares = int((equity * amount / 100.0) / price)          # floor
        elif mode == "dollars":
            shares = int(amount / price)                             # floor
        else:                                                        # "shares"
            shares = int(amount)

        if shares < 1:
            skipped.append({"ticker": t, "reason": "sizing rounds to < 1 share"})
            continue
        est_value = shares * price
        if mode != "shares" and est_value < MIN_TRADE_USD:
            skipped.append({"ticker": t,
                            "reason": f"under the ${MIN_TRADE_USD:.0f} order minimum"})
            continue

        bz = lv.get("buy_zone") or (None, None)
        pivot = bz[0]
        stop = lv.get("stop")
        plan.append({
            "ticker": t, "shares": shares, "price": round(price, 2),
            "pivot": round(float(pivot), 2) if pivot else None,
            "est_value": round(est_value, 2),
            "extended": bool(bz[1] and price > bz[1]),
            "stop_price": round(float(stop), 2) if stop and stop > 0 else None,
            "earnings_in": payload.get("earnings_in"),
        })
    return plan, skipped


def _open_sell_stops(client, ticker: str, *, GetOrdersRequest, QueryOrderStatus,
                     OrderSide, OrderType) -> List:
    """This ticker's OPEN sell **stop** orders (STOP / STOP_LIMIT / TRAILING_STOP).

    A manual limit sell isn't a stop, so it's excluded. Alpaca surfaces a triggered OTO stop
    leg as its own top-level SELL order too, so this flat query catches both standalone stops
    and prior OTO legs. Returns the order objects (each carries a ``stop_price``) so the caller
    can read the current stop level for the ratchet AND cancel them when raising.
    """
    stop_types = {OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP}
    try:
        opens = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, side=OrderSide.SELL, symbols=[ticker]))
    except Exception:
        return []
    out = []
    for od in opens or []:
        if getattr(od, "symbol", None) != ticker:
            continue
        otype = getattr(od, "type", None) or getattr(od, "order_type", None)
        if otype in stop_types:
            out.append(od)
    return out


def _stop_price_of(order) -> Optional[float]:
    """Best-effort read of an order's stop trigger price (alpaca-py ``Order.stop_price``) as a
    float, or None if absent/unparseable. Real stops always carry one; None just means we
    can't compare, so the caller falls back to replacing rather than ratcheting."""
    v = getattr(order, "stop_price", None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cancel_orders(client, orders) -> List[str]:
    """Cancel each order by id, independently guarded so one stuck order never blocks the rest.
    Returns the cancelled ids."""
    cancelled: List[str] = []
    for od in orders or []:
        try:
            client.cancel_order_by_id(od.id)
            cancelled.append(str(getattr(od, "id", "?")))
        except Exception:
            pass
    return cancelled


def submit_buy_plan(plan: List[dict], *, attach_stop: bool = True) -> dict:
    """Submit each planned order on the cockpit's Alpaca **paper** account (Minervini Trader
    keys preferred — see :func:`_connect_paper`), attaching a protective stop when
    ``attach_stop`` is set.

    Per name:

    * **already held** in the account — no buy is sent; a **GTC** sell-stop protects the WHOLE
      held position, managed as Minervini's one-way ratchet (never lower a stop, only raise it):
      if no stop is open it's placed at ``stop_price``; if one already is, it's replaced only to
      RAISE it — a would-be lower-or-equal stop is left untouched (result ``"stop_kept"``). GTC
      so it persists across sessions instead of expiring each close. Exempt from the $50 floor /
      10%-cap since a protective stop is risk-reducing.
    * **not held** — a market BUY. With ``attach_stop`` it's an OTO order carrying a stop-loss
      leg (the stop activates only after the buy fills, so it works even when the buy is queued
      to the next open). That OTO stop leg rides the market entry as a DAY order (Alpaca market
      entries can't be GTC); it's promoted to a GTC ratcheted stop by the held-name path on the
      next re-arm. Without ``attach_stop``, a plain market BUY.

    Reuses ``alpaca_trader``'s tradability check and 10%-of-equity order cap (buys only).
    Returns ``{equity, cash, account_number, using_dedicated, results}`` where each result is
    the plan entry plus a ``status`` ("submitted" / "stop_only" / "stop_kept" / "skipped" /
    "failed") and a ``detail`` string. Raises :class:`TradeUnavailable` if alpaca-py or
    credentials are missing.
    """
    client, using_dedicated = _connect_paper()          # paper=True enforced inside
    try:
        from alpaca.trading.requests import (
            MarketOrderRequest, StopOrderRequest, StopLossRequest, GetOrdersRequest)
        from alpaca.trading.enums import (
            OrderSide, TimeInForce, OrderClass, QueryOrderStatus, OrderType)
        from src.portfolio_experimentation.alpaca_trader import (
            validate_tradable, MAX_ORDER_PCT)
    except ImportError as e:                             # alpaca-py present for _connect but
        raise TradeUnavailable(str(e)) from e            # a submodule/name somehow isn't

    acct = client.get_account()
    equity, cash = float(acct.equity), float(acct.cash)
    account_number = getattr(acct, "account_number", "?")
    # {ticker: whole shares held} — mirrors get_account_state's int(float(qty)) convention.
    held = {p.symbol: int(float(p.qty)) for p in client.get_all_positions()}
    tradable = validate_tradable(client, [o["ticker"] for o in plan])
    max_allowed = MAX_ORDER_PCT * equity

    results: List[dict] = []
    for o in plan:
        t = o["ticker"]
        stop = o.get("stop_price")
        held_shares = held.get(t, 0)
        if t not in tradable:
            results.append({**o, "status": "skipped",
                            "detail": "not tradable on Alpaca"})
            continue
        try:
            if held_shares > 0:
                # Already invested — manage a GTC protective stop for the whole position, no
                # buy. Minervini's rule is a ONE-WAY RATCHET: never lower a stop, only raise it.
                # A GTC stop persists across sessions, so we place it once and thereafter replace
                # it only to move it UP. A re-arm that would compute a LOWER stop (e.g. the stock
                # pulled back, dragging the recent-low/price-based stop down with it) is ignored,
                # leaving the existing higher stop in force.
                if not attach_stop:
                    results.append({**o, "status": "skipped",
                                    "detail": f"already held ({held_shares} sh); "
                                              "stop attach disabled"})
                    continue
                existing = _open_sell_stops(
                    client, t, GetOrdersRequest=GetOrdersRequest,
                    QueryOrderStatus=QueryOrderStatus, OrderSide=OrderSide, OrderType=OrderType)
                prices = [p for p in (_stop_price_of(od) for od in existing) if p is not None]
                cur = max(prices) if prices else None            # current stop level, if any
                new_stop = round(float(stop), 2) if stop else None

                if not stop_is_valid(new_stop, o["price"]):
                    # New stop isn't below the price. If a valid stop is already in force the
                    # position stays protected — keep it; otherwise there's nothing to place.
                    if cur is not None:
                        results.append({**o, "status": "stop_kept", "stop_price": cur,
                                        "detail": f"kept existing stop @ {cur:.2f} "
                                                  "(new stop not below price)"})
                    else:
                        results.append({**o, "status": "skipped",
                                        "detail": "no valid stop (must be > 0 and < current price)"})
                    continue

                # Ratchet: only replace to RAISE the stop; a lower-or-equal one is kept.
                if cur is not None and new_stop <= cur:
                    results.append({**o, "status": "stop_kept", "stop_price": cur,
                                    "detail": f"kept existing stop @ {cur:.2f} — "
                                              f"not lowering to {new_stop:.2f}"})
                    continue

                _cancel_orders(client, existing)                 # replace the lower stop(s)
                req = StopOrderRequest(
                    symbol=t, qty=held_shares, side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC, stop_price=new_stop,
                    client_order_id=f"SEPAstop-{t}-{int(time.time() * 1000)}")
                resp = client.submit_order(order_data=req)
                verb = "raised" if cur is not None else "placed"
                detail = (f"GTC stop {verb}: SELL {held_shares} @ {new_stop:.2f} "
                          f"(id {getattr(resp, 'id', '?')})")
                if cur is not None:
                    detail += f" (was {cur:.2f})"
                results.append({**o, "status": "stop_only",
                                "stop_price": new_stop, "detail": detail})
                continue

            # Not held — a market BUY, with an OTO protective stop when attach_stop is on.
            if o["est_value"] > max_allowed:
                results.append({**o, "status": "skipped",
                                "detail": f"exceeds 10% of equity (${max_allowed:,.0f} cap)"})
                continue
            if attach_stop:
                if not stop_is_valid(stop, o["price"]):
                    results.append({**o, "status": "skipped",
                                    "detail": "stop not below entry — fix stop or turn off "
                                              "Attach stop"})
                    continue
                # OTO = DAY (Alpaca market entries can't be GTC), so this initial stop leg is a
                # DAY order; the held-name path promotes it to a persistent GTC ratcheted stop on
                # the next re-arm once the fill shows up as a position.
                req = MarketOrderRequest(
                    symbol=t, qty=o["shares"], side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY, order_class=OrderClass.OTO,
                    stop_loss=StopLossRequest(stop_price=round(float(stop), 2)),
                    client_order_id=f"SEPAoto-{t}-{int(time.time() * 1000)}")
                detail_head = f"buy {o['shares']} + stop @ {stop:.2f}"
            else:
                req = MarketOrderRequest(
                    symbol=t, qty=o["shares"], side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=f"SEPAcockpit-{t}-{int(time.time() * 1000)}")
                detail_head = f"buy {o['shares']} (no stop)"
            resp = client.submit_order(order_data=req)
            results.append({**o, "status": "submitted",
                            "detail": f"{detail_head} (id {getattr(resp, 'id', '?')})"})
        except Exception as e:                          # one bad symbol shouldn't abort the rest
            results.append({**o, "status": "failed", "detail": str(e)})

    return {"equity": equity, "cash": cash, "account_number": account_number,
            "using_dedicated": using_dedicated, "results": results}
