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
# owns ALPACA_API_KEY/SECRET). Each Alpaca paper account has its own key pair, so we select
# the "Minervini Trader" account by preferring its dedicated keys, falling back to the
# shared ones only if the dedicated pair isn't set.
MINERVINI_KEY_ENV = "ALPACA_MINERVINI_API_KEY"
MINERVINI_SECRET_ENV = "ALPACA_MINERVINI_API_SECRET"


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

    ded_key = os.environ.get(MINERVINI_KEY_ENV)
    ded_secret = os.environ.get(MINERVINI_SECRET_ENV)
    key = ded_key or os.environ.get("ALPACA_API_KEY")
    secret = ded_secret or os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise TradeUnavailable(
            f"No Alpaca credentials in .env. Add the Minervini Trader paper account's keys "
            f"as {MINERVINI_KEY_ENV} / {MINERVINI_SECRET_ENV} (each Alpaca paper account has "
            f"its own key pair), or the shared ALPACA_API_KEY / ALPACA_API_SECRET.")
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


def build_buy_plan(tickers: Sequence[str], payloads: Dict[str, dict], *,
                   mode: str, amount: float,
                   equity: Optional[float] = None) -> Tuple[List[dict], List[dict]]:
    """Size a market-BUY for EACH watchlisted name by the chosen ``mode``:

    * ``"pct"``     — ``amount`` % of the account ``equity`` per name (needs ``equity``);
    * ``"dollars"`` — ``amount`` dollars per name;
    * ``"shares"``  — exactly ``amount`` (whole) shares per name.

    Returns ``(plan, skipped)``. Each plan entry has ticker / shares / price / pivot /
    est_value / extended; each skipped entry is ``{ticker, reason}``. A name is skipped when
    it isn't in the current scan, has no current price, sizes to < 1 share, or (for the two
    dollar-denominated modes) rounds to a notional under the $50 floor — the ``"shares"``
    mode is exempt from that floor since the count is explicit. ``extended`` flags a price
    already above the no-chase buy zone (> pivot × 1.05); the caller surfaces it as a
    warning rather than skipping.
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
        plan.append({
            "ticker": t, "shares": shares, "price": round(price, 2),
            "pivot": round(float(pivot), 2) if pivot else None,
            "est_value": round(est_value, 2),
            "extended": bool(bz[1] and price > bz[1]),
        })
    return plan, skipped


def submit_buy_plan(plan: List[dict]) -> dict:
    """Submit each planned BUY as a market/DAY order on the cockpit's Alpaca **paper**
    account (Minervini Trader keys preferred — see :func:`_connect_paper`).

    Reuses ``alpaca_trader``'s tradability check and 10%-of-equity order cap. Returns
    ``{equity, cash, account_number, using_dedicated, results}`` where each result is the
    plan entry plus a ``status`` ("submitted" / "skipped" / "failed") and a ``detail``
    string. Raises :class:`TradeUnavailable` if alpaca-py or credentials are missing.
    """
    client, using_dedicated = _connect_paper()          # paper=True enforced inside
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        from src.portfolio_experimentation.alpaca_trader import (
            validate_tradable, MAX_ORDER_PCT)
    except ImportError as e:                             # alpaca-py present for _connect but
        raise TradeUnavailable(str(e)) from e            # the requests submodule somehow isn't

    acct = client.get_account()
    equity, cash = float(acct.equity), float(acct.cash)
    account_number = getattr(acct, "account_number", "?")
    tradable = validate_tradable(client, [o["ticker"] for o in plan])
    max_allowed = MAX_ORDER_PCT * equity

    results: List[dict] = []
    for o in plan:
        t = o["ticker"]
        if t not in tradable:
            results.append({**o, "status": "skipped",
                            "detail": "not tradable on Alpaca"})
            continue
        if o["est_value"] > max_allowed:
            results.append({**o, "status": "skipped",
                            "detail": f"exceeds 10% of equity (${max_allowed:,.0f} cap)"})
            continue
        try:
            req = MarketOrderRequest(
                symbol=t, qty=o["shares"], side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                client_order_id=f"SEPAcockpit-{t}-{int(time.time())}")
            resp = client.submit_order(order_data=req)
            results.append({**o, "status": "submitted",
                            "detail": f"order id {getattr(resp, 'id', '?')}"})
        except Exception as e:                          # one bad symbol shouldn't abort the rest
            results.append({**o, "status": "failed", "detail": str(e)})

    return {"equity": equity, "cash": cash, "account_number": account_number,
            "using_dedicated": using_dedicated, "results": results}
