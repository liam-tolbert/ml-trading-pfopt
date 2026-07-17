"""Paper-trade the cockpit watchlist through Alpaca.

Each name is sized by a chosen mode (see :func:`build_buy_plan`): a % of equity, a flat $
amount, an explicit share count, or **risk-to-stop** —
``shares = floor((equity × risk%) / (price − stop))``, Minervini's position sizer, so a
stop-out costs ≈ risk% of the account (the idea the Step-4 panel shows, sized on the live
fill price rather than the pivot). Orders are plain market BUYs on the **paper** account
(``alpaca_trader.connect()`` forces ``paper=True``); the guardrails apply — a per-order
floor of $50 (dollar-denominated modes) and a 10%-of-equity single-order cap (the risk mode
clamps to it and flags ``capped``; the other modes skip an over-cap name — never fatal).

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
MAX_ORDER_PCT = 0.10        # mirrors alpaca_trader.MAX_ORDER_PCT — single-order cap as a
                            # fraction of equity; the risk mode clamps to it (submit re-checks)
STALE_PLAN_BARS = 2         # skip a plan name whose freshest daily bar is more than this many
                            # *trading* days old rather than size on stale data (the scan memo
                            # has no time-based invalidation). 2 absorbs a weekend + a holiday.

# --- Positions-page stop management (Minervini exit rules) ---------------------------------- #
INITIAL_STOP_PCT = 0.08     # ~8% initial stop below the entry (buy point)
BREAKEVEN_GAIN = 0.16       # gain past which the stop should be at least breakeven (~2x initial risk)
TRAIL_GAIN = 0.20           # gain past which, "well in profit", trail the 50-day SMA
SELL_STRENGTH_GAIN = 0.20   # gain past which to consider selling part into strength
HEAVY_VOL_RATIO = 1.5       # latest volume vs its 50-day average = a heavy-volume day
# Suggested-stop bases for the re-arm action; "auto" picks per position by its gain.
STOP_BASES = ("auto", "initial", "breakeven", "sma50")

# The cockpit trades a SEPARATE Alpaca paper account from the All-Weather mirror (which owns
# the shared ALPACA_API_KEY/SECRET pair). Each paper account has its own key pair, so prefer
# the dedicated "Minervini Trader" keys, falling back to the shared pair only if unset. Name
# drift between .env and these constants silently resolves to None -> TradeUnavailable, so keep
# them in sync. First non-empty match in each tuple wins.
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


SIZING_MODES = ("pct", "dollars", "shares", "risk")


def stop_is_valid(stop_price, price) -> bool:
    """A protective sell-stop is valid only strictly BELOW the reference price.

    Alpaca rejects a sell stop at/above the market (it would trigger instantly) and an OTO
    stop-loss leg that isn't below the entry. Used by the UI (live per-keystroke check) and
    re-checked in :func:`submit_buy_plan` against the last close.
    """
    return bool(stop_price and price and stop_price > 0 and stop_price < price)


def suggest_stop(*, avg_entry: Optional[float], current_price: Optional[float],
                 sma_50: Optional[float], current_stop: Optional[float],
                 gain_pct: Optional[float], basis: str = "auto"
                 ) -> Tuple[Optional[float], str]:
    """Minervini stop suggestion for a held position under a chosen ``basis``.

    Basis levels: ``initial`` = ``avg_entry × (1 - INITIAL_STOP_PCT)`` (~8% below entry);
    ``breakeven`` = ``avg_entry``; ``sma50`` = ``sma_50 × 0.99`` (just under the 50-day). ``auto``
    picks by gain (the position's stage): well in profit (``gain_pct >= TRAIL_GAIN``) with a 50-day
    available → trail the SMA; working (``gain_pct >= BREAKEVEN_GAIN``) → at least breakeven; else
    the initial 8% stop.

    Returns ``(suggested_price_or_None, effective_basis_label)``. The suggestion is floored at the
    current in-force stop — and, once the trade is working, at breakeven — so it is ratchet-safe
    (never proposes LOWER than what's in force, never gives back a working trade below breakeven).
    Returns ``None`` when no basis input is available or the result isn't strictly below
    ``current_price`` (underwater / already stopped-out territory → leave for a manual edit)."""
    initial_val = avg_entry * (1.0 - INITIAL_STOP_PCT) if avg_entry else None
    breakeven_val = float(avg_entry) if avg_entry else None
    sma_val = sma_50 * 0.99 if sma_50 else None

    if basis == "auto":
        if gain_pct is not None and gain_pct >= TRAIL_GAIN and sma_val is not None:
            eff = "sma50"
        elif gain_pct is not None and gain_pct >= BREAKEVEN_GAIN and breakeven_val is not None:
            eff = "breakeven"
        else:
            eff = "initial"
    else:
        eff = basis

    base_val = {"initial": initial_val, "breakeven": breakeven_val, "sma50": sma_val}.get(eff)
    # Ratchet-safe floor: never below the in-force stop. In AUTO mode also never below breakeven
    # once the trade is working (so trailing the 50-day can't give back a won trade) — an EXPLICIT
    # basis is honored as chosen (still floored at the in-force stop).
    floors = [v for v in (base_val, current_stop) if v is not None]
    if (basis == "auto" and gain_pct is not None and gain_pct >= BREAKEVEN_GAIN
            and breakeven_val is not None):
        floors.append(breakeven_val)
    if not floors:
        return None, eff
    cand = round(max(floors), 2)
    return (cand, eff) if stop_is_valid(cand, current_price) else (None, eff)


def position_advisories(pos: dict) -> List[str]:
    """Display-only Minervini exit advisories derived from a :func:`fetch_positions` dict. Pure.

    Note the "×initial-risk" rule (#4) approximates the initial risk at ``INITIAL_STOP_PCT`` (8%),
    because the entry-time stop distance isn't persisted anywhere — so it's a nudge, not exact."""
    out: List[str] = []
    gain = pos.get("gain_pct")
    avg_entry = pos.get("avg_entry")
    cur_stop = pos.get("current_stop")

    if not pos.get("has_stop"):
        out.append("⚠ No protective stop armed — arm one.")
    if gain is not None and gain >= SELL_STRENGTH_GAIN:
        out.append(f"Up {gain * 100:.0f}% — consider selling part into strength.")
    if pos.get("below_sma50"):
        vr = pos.get("volume_ratio")
        heavy = " on heavy volume" if (vr is not None and vr >= HEAVY_VOL_RATIO) else ""
        out.append(f"Closed below the 50-day SMA{heavy} — exit signal.")
    if (gain is not None and gain >= BREAKEVEN_GAIN and avg_entry
            and (cur_stop is None or cur_stop < avg_entry)):
        out.append("Up ≥ 2× initial risk — raise stop to at least breakeven.")
    return out


def _trading_days_since(last, today) -> Optional[int]:
    """Number of *trading* days between a bar date ``last`` and ``today`` — 0 when ``last`` is
    the freshest possible bar (e.g. Friday's bar read on the weekend, or today's own bar).
    Business-day based; holidays aren't modelled, so a holiday reads as one extra day and the
    caller's tolerance absorbs it. Returns None if either date can't be parsed, so the caller
    skips the check rather than blocking a trade on a parse error."""
    try:
        import pandas as pd
        a = pd.Timestamp(last).normalize()
        b = pd.Timestamp(today).normalize()
        if b <= a:
            return 0
        return len(pd.bdate_range(a, b)) - 1
    except Exception:
        return None


def build_buy_plan(tickers: Sequence[str], payloads: Dict[str, dict], *,
                   mode: str, amount: float,
                   equity: Optional[float] = None, asof=None,
                   max_bar_age_days: Optional[float] = None) -> Tuple[List[dict], List[dict]]:
    """Size a market-BUY for EACH watchlisted name by the chosen ``mode``:

    * ``"pct"``     — ``amount`` % of the account ``equity`` per name (needs ``equity``);
    * ``"dollars"`` — ``amount`` dollars per name;
    * ``"shares"``  — exactly ``amount`` (whole) shares per name;
    * ``"risk"``    — ``amount`` % of ``equity`` risked to the stop (needs ``equity`` + a stop):
      ``shares = (equity × amount%) / (price − stop)`` — Minervini's position sizer, so a
      stop-out costs ≈ ``amount``% of the account. Sized on the current price (the real fill),
      not the pivot, so the risk figure is honest for the order actually sent.

    Returns ``(plan, skipped)``. Each plan entry has ticker / shares / price / pivot /
    est_value / extended / stop_price / capped; each skipped entry is ``{ticker, reason}``. A
    name is skipped when it isn't in the current scan, has no current price, sizes to < 1 share,
    or (for every dollar-denominated mode — pct/dollars/risk) rounds to a notional under the $50
    floor; the ``"shares"`` mode is exempt from that floor since the count is explicit. The
    ``"risk"`` mode additionally skips a name with no stop, or a stop not below the price (a
    non-positive risk-per-share). ``extended`` flags a price already above the no-chase buy zone
    (> pivot × 1.05); the caller surfaces it as a warning rather than skipping. ``capped`` is
    True only in ``"risk"`` mode when the risk-sized quantity would exceed the 10%-of-equity
    single-order cap and was clamped down to it (so the realized risk falls BELOW the target —
    the caller labels it); it's always False for the other modes. ``stop_price`` is the
    app-computed protective stop (``levels["stop"]``, ~7-8% below pivot) or ``None`` if
    unavailable — the caller may edit it before submit; note that editing it after build does
    NOT re-scale a risk-sized quantity. ``earnings_in`` (calendar days to the next scheduled
    report, from the scan payload; None = unknown) is carried through untouched so the caller
    can warn about buying into an imminent report — advisory only, never a skip.

    When ``max_bar_age_days`` is set, a name whose freshest bar is more than that many
    *trading* days old (relative to ``asof`` or today) is skipped as stale rather than sized
    on days-old data — the app pairs this with :func:`freshen_prices` at Build so ordinary
    names read fresh. Both default to off, keeping the builder pure and unchanged for callers
    (and unit tests) that omit them.
    """
    if mode not in SIZING_MODES:
        raise ValueError(f"mode must be one of {SIZING_MODES}, got {mode!r}")
    _stale_ref = None
    if max_bar_age_days is not None:
        import pandas as pd
        _stale_ref = (pd.Timestamp(asof).normalize() if asof is not None
                      else pd.Timestamp.today().normalize())
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
        if _stale_ref is not None:
            _age = _trading_days_since(df.index[-1], _stale_ref)
            if _age is not None and _age > max_bar_age_days:
                skipped.append({"ticker": t, "reason":
                                f"stale price data (last bar "
                                f"{pd.Timestamp(df.index[-1]).date()}, ~{_age} trading days "
                                f"old) — Re-scan to refresh"})
                continue

        stop = lv.get("stop")            # read early — the risk mode sizes against it
        capped = False

        if mode == "pct":
            if not equity or equity <= 0:
                skipped.append({"ticker": t, "reason": "account equity unavailable"})
                continue
            shares = int((equity * amount / 100.0) / price)          # floor
        elif mode == "dollars":
            shares = int(amount / price)                             # floor
        elif mode == "risk":
            if not equity or equity <= 0:
                skipped.append({"ticker": t, "reason": "account equity unavailable"})
                continue
            if not stop or stop <= 0:
                skipped.append({"ticker": t, "reason": "no stop to risk-size against"})
                continue
            if stop >= price:
                skipped.append({"ticker": t, "reason": "stop not below price — can't risk-size"})
                continue
            shares = int((equity * amount / 100.0) / (price - stop))  # floor
            # Risk sizing yields position% ≈ risk% / stop-distance%, which routinely exceeds the
            # 10% single-order cap (1% risk / 8% stop = 12.5%). Clamp to the cap rather than skip;
            # the realized risk then sits below target and the caller flags it via ``capped``.
            cap_shares = int(MAX_ORDER_PCT * equity / price)
            if shares > cap_shares:
                shares, capped = cap_shares, True
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
            "capped": capped,
            "stop_price": round(float(stop), 2) if stop and stop > 0 else None,
            "earnings_in": payload.get("earnings_in"),
        })
    return plan, skipped


def freshen_prices(tickers: Sequence[str], payloads: Dict[str, dict]) -> Dict[str, dict]:
    """Re-pull the latest daily bars for the given watchlist ``tickers`` and return a small
    payloads dict with each name's ``df`` replaced by the freshest available frame, so
    :func:`build_buy_plan` sizes on current prices instead of the possibly days-old closes
    frozen in the scan memo (the scan result lives in session_state with no time-based
    invalidation and the trigger fragment keeps the tab alive for days).

    Uses the cheap incremental top-up (``max_age_days=0`` — the same path the EOD trigger
    uses to fetch the finalized close without a full 2y refetch); only the handful of
    watchlist names are fetched, never the universe. Any name the refresh can't reach keeps
    its existing frame — :func:`build_buy_plan`'s ``max_bar_age_days`` guard then skips a
    genuinely stale one. Tickers absent from ``payloads`` are dropped (the builder already
    reports those as 'not in the current scan'). ``levels`` and ``earnings_in`` are carried
    through untouched; only the price frame is refreshed. ``data_feed`` is imported lazily so
    the cockpit still loads without its optional deps, and a total fetch failure degrades to
    the original frames rather than raising."""
    want = [t for t in dict.fromkeys(tickers) if payloads.get(t)]
    if not want:
        return {}
    try:
        from . import data_feed
        fresh = data_feed.get_many_prices(want, max_age_days=0.0) or {}
    except Exception:
        fresh = {}
    out: Dict[str, dict] = {}
    for t in want:
        f = fresh.get(t)
        out[t] = {**payloads[t], "df": f} if (f is not None and len(f)) else payloads[t]
    return out


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


def _open_sell_stops_by_symbol(client, *, GetOrdersRequest, QueryOrderStatus,
                               OrderSide, OrderType) -> Dict[str, List]:
    """ALL open sell STOP/STOP_LIMIT/TRAILING_STOP orders in ONE query, grouped by symbol.

    Same type filter as :func:`_open_sell_stops` but omits the ``symbols=`` filter, so the whole
    account's protective stops come back in a single round-trip (the positions page needs every
    symbol's stop at once). Returns ``{symbol: [order, ...]}``; empty dict on any error."""
    stop_types = {OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP}
    try:
        opens = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, side=OrderSide.SELL))
    except Exception:
        return {}
    out: Dict[str, List] = {}
    for od in opens or []:
        sym = getattr(od, "symbol", None)
        if not sym:
            continue
        otype = getattr(od, "type", None) or getattr(od, "order_type", None)
        if otype in stop_types:
            out.setdefault(sym, []).append(od)
    return out


def _rearm_gtc_stop(client, symbol: str, held_shares: int, desired_stop, price, existing, *,
                    OrderSide, TimeInForce, StopOrderRequest) -> dict:
    """Minervini's one-way GTC stop ratchet for a held position — the single source of truth,
    called by both :func:`submit_buy_plan` (held-name branch) and :func:`rearm_stops`.

    ``existing`` is that symbol's open sell-stop orders (the caller fetches them, per-ticker or
    batched). The current in-force stop is ``max(_stop_price_of(existing))``. Returns a PARTIAL
    result dict — ``{status, detail}`` plus ``stop_price`` when a level is set — where status is
    ``"stop_only"`` (placed/raised), ``"stop_kept"`` (existing kept — a would-be lower/equal or
    an invalid new stop) or ``"skipped"`` (no valid stop and none in force). NEVER lowers a stop:
    it only cancels + replaces to RAISE. GTC so the stop persists across sessions."""
    prices = [p for p in (_stop_price_of(od) for od in existing) if p is not None]
    cur = max(prices) if prices else None                # current stop level, if any
    new_stop = round(float(desired_stop), 2) if desired_stop else None

    if not stop_is_valid(new_stop, price):
        # New stop isn't below the price. If a valid stop is already in force the position stays
        # protected — keep it; otherwise there's nothing to place.
        if cur is not None:
            return {"status": "stop_kept", "stop_price": cur,
                    "detail": f"kept existing stop @ {cur:.2f} (new stop not below price)"}
        return {"status": "skipped",
                "detail": "no valid stop (must be > 0 and < current price)"}

    # Ratchet: only replace to RAISE the stop; a lower-or-equal one is kept.
    if cur is not None and new_stop <= cur:
        return {"status": "stop_kept", "stop_price": cur,
                "detail": f"kept existing stop @ {cur:.2f} — not lowering to {new_stop:.2f}"}

    _cancel_orders(client, existing)                     # replace the lower stop(s)
    req = StopOrderRequest(
        symbol=symbol, qty=held_shares, side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC, stop_price=new_stop,
        client_order_id=f"SEPAstop-{symbol}-{int(time.time() * 1000)}")
    resp = client.submit_order(order_data=req)
    verb = "raised" if cur is not None else "placed"
    detail = (f"GTC stop {verb}: SELL {held_shares} @ {new_stop:.2f} "
              f"(id {getattr(resp, 'id', '?')})")
    if cur is not None:
        detail += f" (was {cur:.2f})"
    return {"status": "stop_only", "stop_price": new_stop, "detail": detail}


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
                # Already invested — no buy; manage a GTC protective stop for the whole position
                # via the shared one-way ratchet (never lower, only raise — see _rearm_gtc_stop).
                if not attach_stop:
                    results.append({**o, "status": "skipped",
                                    "detail": f"already held ({held_shares} sh); "
                                              "stop attach disabled"})
                    continue
                existing = _open_sell_stops(
                    client, t, GetOrdersRequest=GetOrdersRequest,
                    QueryOrderStatus=QueryOrderStatus, OrderSide=OrderSide, OrderType=OrderType)
                res = _rearm_gtc_stop(client, t, held_shares, stop, o["price"], existing,
                                      OrderSide=OrderSide, TimeInForce=TimeInForce,
                                      StopOrderRequest=StopOrderRequest)
                results.append({**o, **res})
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


def _attr_float(obj, attr: str) -> Optional[float]:
    """``getattr`` + float-coerce-or-None. alpaca-py ``Position`` P&L fields are read
    defensively — any absent/odd-typed field reads as None and the view renders without it."""
    v = getattr(obj, attr, None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_positions() -> dict:
    """Read the cockpit's Alpaca **paper** account (Minervini keys preferred) and return holdings
    enriched with P&L, in-force stop level, 50-day SMA, and Minervini exit advisories.

    Returns ``{"account": {account_number, equity, cash, using_dedicated, positions_count,
    total_unrealized_pl}, "positions": [ per-position dict ]}``. Raises :class:`TradeUnavailable`
    on missing package/credentials (the page catches it). Every per-position numeric field degrades
    to ``None`` (never raises) when a Position attribute is absent or price history is < 50 bars."""
    client, using_dedicated = _connect_paper()
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, OrderType
    except ImportError as e:
        raise TradeUnavailable(str(e)) from e

    acct = client.get_account()
    equity, cash = float(acct.equity), float(acct.cash)
    account_number = getattr(acct, "account_number", "?")
    raw = list(client.get_all_positions())
    stops_by_sym = _open_sell_stops_by_symbol(
        client, GetOrdersRequest=GetOrdersRequest, QueryOrderStatus=QueryOrderStatus,
        OrderSide=OrderSide, OrderType=OrderType)

    symbols = [s for s in (getattr(p, "symbol", None) for p in raw) if s]
    # One batched price pull for the 50-day SMA + SMA-cross advisory; lazy imports so trade.py
    # still loads without yfinance / the vendored screening package present.
    frames: dict = {}
    data_feed = None
    if symbols:
        try:
            from . import data_feed as _df_mod
            data_feed = _df_mod
            frames = data_feed.get_many_prices(symbols)
        except Exception:
            frames = {}
    try:
        from src.stock_screener.minervini_screener.screening import calculate_sma
    except Exception:
        calculate_sma = None
    import pandas as pd

    positions: List[dict] = []
    for p in raw:
        sym = getattr(p, "symbol", None)
        if not sym:
            continue
        try:
            qty = int(float(getattr(p, "qty", 0) or 0))
        except (TypeError, ValueError):
            qty = 0
        avg_entry = _attr_float(p, "avg_entry_price")
        price = _attr_float(p, "current_price")

        stop_prices = [q for q in (_stop_price_of(od) for od in stops_by_sym.get(sym, []))
                       if q is not None]
        current_stop = max(stop_prices) if stop_prices else None

        sma_50 = last_close = volume_ratio = None
        df = frames.get(sym)
        if df is None and data_feed is not None:
            df = frames.get(data_feed.normalize(sym))
        if df is not None and len(df):
            last_close = float(df["Close"].iloc[-1])
            if price is None:
                price = last_close
            if calculate_sma is not None and len(df) >= 50:
                s = calculate_sma(df["Close"], 50)
                if len(s) and pd.notna(s.iloc[-1]):
                    sma_50 = float(s.iloc[-1])
            if "Volume" in df.columns and len(df) >= 50:
                avg_vol = float(df["Volume"].iloc[-50:].mean())
                if avg_vol > 0:
                    volume_ratio = float(df["Volume"].iloc[-1]) / avg_vol

        gain_pct = _attr_float(p, "unrealized_plpc")
        if gain_pct is None and avg_entry and price:
            gain_pct = (price - avg_entry) / avg_entry
        below_sma50 = bool(sma_50 is not None and last_close is not None and last_close < sma_50)

        pos = {
            "symbol": sym, "qty": qty, "avg_entry": avg_entry, "current_price": price,
            "market_value": _attr_float(p, "market_value"),
            "cost_basis": _attr_float(p, "cost_basis"),
            "unrealized_pl": _attr_float(p, "unrealized_pl"),
            "unrealized_plpc": _attr_float(p, "unrealized_plpc"),
            "lastday_price": _attr_float(p, "lastday_price"),
            "current_stop": current_stop, "has_stop": current_stop is not None,
            "sma_50": sma_50, "last_close": last_close, "volume_ratio": volume_ratio,
            "gain_pct": gain_pct, "below_sma50": below_sma50,
        }
        pos["advisories"] = position_advisories(pos)
        positions.append(pos)

    total_pl = sum(p["unrealized_pl"] for p in positions if p["unrealized_pl"] is not None)
    account = {
        "account_number": account_number, "equity": equity, "cash": cash,
        "using_dedicated": using_dedicated, "positions_count": len(positions),
        "total_unrealized_pl": total_pl,
    }
    return {"account": account, "positions": positions}


def rearm_stops(targets: List[dict]) -> dict:
    """Raise/place GTC protective stops for already-held names via the shared one-way ratchet
    (:func:`_rearm_gtc_stop`) — never lowering a stop. Each target: ``{ticker, stop_price, price}``
    (``price`` = the reference the stop must sit below). A ticker not held in the account is
    skipped. Returns ``{equity, cash, account_number, using_dedicated, results}`` — the same shape
    and status vocabulary as :func:`submit_buy_plan`. Raises :class:`TradeUnavailable`."""
    client, using_dedicated = _connect_paper()
    try:
        from alpaca.trading.requests import StopOrderRequest, GetOrdersRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderType
    except ImportError as e:
        raise TradeUnavailable(str(e)) from e

    acct = client.get_account()
    equity, cash = float(acct.equity), float(acct.cash)
    account_number = getattr(acct, "account_number", "?")
    held = {p.symbol: int(float(p.qty)) for p in client.get_all_positions()}
    by_sym = _open_sell_stops_by_symbol(
        client, GetOrdersRequest=GetOrdersRequest, QueryOrderStatus=QueryOrderStatus,
        OrderSide=OrderSide, OrderType=OrderType)

    results: List[dict] = []
    for tgt in targets:
        t = tgt.get("ticker")
        held_shares = held.get(t, 0)
        if held_shares <= 0:
            results.append({**tgt, "status": "skipped", "detail": "not held in this account"})
            continue
        try:
            res = _rearm_gtc_stop(client, t, held_shares, tgt.get("stop_price"),
                                  tgt.get("price"), by_sym.get(t, []),
                                  OrderSide=OrderSide, TimeInForce=TimeInForce,
                                  StopOrderRequest=StopOrderRequest)
            results.append({**tgt, **res})
        except Exception as e:                          # one bad symbol shouldn't abort the rest
            results.append({**tgt, "status": "failed", "detail": str(e)})
    return {"equity": equity, "cash": cash, "account_number": account_number,
            "using_dedicated": using_dedicated, "results": results}


# --------------------------------------------------------------------------- #
# Trade journal — "know your numbers" (Think & Trade Like a Champion)
# --------------------------------------------------------------------------- #
SEPA_TAG_PREFIXES = ("SEPAoto-", "SEPAstop-", "SEPAcockpit-")
_ORDERS_PAGE_LIMIT = 500        # Alpaca's max order-history page size
_MAX_ORDER_PAGES = 40           # pagination ceiling (40 × 500 = 20k orders), not a real limit


def _fill_time(fill: dict):
    """A fill's timestamp as a tz-aware UTC ``pd.Timestamp``. Missing/unparseable times map
    to epoch 0 so undated fills sort first deterministically instead of raising."""
    import pandas as pd
    t = pd.to_datetime(fill.get("time"), utc=True, errors="coerce")
    return t if pd.notna(t) else pd.Timestamp(0, tz="UTC")


def build_trade_journal(fills: List[dict]) -> dict:
    """Reconstruct round-trip trades from raw order fills. Pure — no network.

    ``fills``: ``{symbol, side ("buy"/"sell"), qty, price, time, client_order_id}`` dicts
    (:func:`fetch_order_fills` produces them; any time parseable by pandas works). Fills are
    sorted by time and grouped per symbol into POSITION EPISODES — flat → long → flat closes
    one trade — so scale-ins and partial sells aggregate into a single round trip (avg entry
    vs avg exit), which is what the "know your numbers" stats count as ONE trade.

    Returns ``{"closed": [...], "open": [...], "unmatched_sells": [...]}``:

    * ``closed`` — fully-exited episodes: ``{symbol, entry_date, exit_date, hold_days,
      shares, avg_entry, avg_exit, cost, proceeds, pl, pl_pct, n_fills, tagged}``
      (``pl_pct`` is a fraction of cost, like the positions page's ``gain_pct``);
    * ``open`` — episodes still holding shares: ``{symbol, entry_date, shares_open,
      avg_entry, realized_pl, n_fills, tagged}`` — ``realized_pl`` books partial sells at
      the episode's average cost; open episodes are EXCLUDED from the closed-trade stats;
    * ``unmatched_sells`` — a sell fill (or the excess part of one) with no prior buy in
      the supplied history (pre-history / transferred shares): recorded, never guessed at.

    ``tagged`` is True when ANY fill in the episode carries a cockpit client_order_id
    (``SEPA_TAG_PREFIXES``) — the entry tag alone is enough, because a triggered OTO stop
    leg exits under an Alpaca-generated id, not the parent's SEPA one.
    """
    eps = 1e-9
    by_sym: Dict[str, List[dict]] = {}
    for f in sorted(fills, key=_fill_time):
        sym = f.get("symbol")
        if sym:
            by_sym.setdefault(sym, []).append(f)

    closed: List[dict] = []
    open_eps: List[dict] = []
    unmatched: List[dict] = []
    for sym in sorted(by_sym):
        ep = None                                       # the in-flight episode, if any
        for f in by_sym[sym]:
            try:
                qty = float(f.get("qty") or 0.0)
                price = float(f.get("price") or 0.0)
            except (TypeError, ValueError):
                continue
            if qty <= 0 or price <= 0:
                continue
            side = str(f.get("side", "")).lower()
            tagged = str(f.get("client_order_id") or "").startswith(SEPA_TAG_PREFIXES)
            t = _fill_time(f)

            if side == "buy":
                if ep is None:
                    ep = {"entry": t, "buy_qty": 0.0, "buy_cost": 0.0, "sell_qty": 0.0,
                          "proceeds": 0.0, "open": 0.0, "n": 0, "tagged": False}
                ep["buy_qty"] += qty
                ep["buy_cost"] += qty * price
                ep["open"] += qty
                ep["n"] += 1
                ep["tagged"] = ep["tagged"] or tagged
            elif side == "sell":
                if ep is None or ep["open"] <= eps:
                    unmatched.append({"symbol": sym, "qty": qty, "price": price, "time": t,
                                      "client_order_id": f.get("client_order_id", "")})
                    continue
                matched = min(qty, ep["open"])
                if qty - matched > eps:                 # the excess has nothing to close
                    unmatched.append({"symbol": sym, "qty": qty - matched, "price": price,
                                      "time": t,
                                      "client_order_id": f.get("client_order_id", "")})
                ep["sell_qty"] += matched
                ep["proceeds"] += matched * price
                ep["open"] -= matched
                ep["n"] += 1
                ep["tagged"] = ep["tagged"] or tagged
                if ep["open"] <= eps:                   # flat again -> one closed round trip
                    pl = ep["proceeds"] - ep["buy_cost"]
                    closed.append({
                        "symbol": sym, "entry_date": ep["entry"], "exit_date": t,
                        "hold_days": max((t - ep["entry"]).days, 0),
                        "shares": ep["buy_qty"],
                        "avg_entry": ep["buy_cost"] / ep["buy_qty"],
                        "avg_exit": ep["proceeds"] / ep["sell_qty"],
                        "cost": ep["buy_cost"], "proceeds": ep["proceeds"],
                        "pl": pl, "pl_pct": pl / ep["buy_cost"],
                        "n_fills": ep["n"], "tagged": ep["tagged"],
                    })
                    ep = None
        if ep is not None and ep["open"] > eps:
            avg_cost = ep["buy_cost"] / ep["buy_qty"]
            open_eps.append({
                "symbol": sym, "entry_date": ep["entry"], "shares_open": ep["open"],
                "avg_entry": avg_cost,
                "realized_pl": ep["proceeds"] - avg_cost * ep["sell_qty"],
                "n_fills": ep["n"], "tagged": ep["tagged"],
            })
    return {"closed": closed, "open": open_eps, "unmatched_sells": unmatched}


def journal_stats(closed: List[dict]) -> dict:
    """Minervini's "know your numbers" from :func:`build_trade_journal` closed trades. Pure.

    Returns ``{n, wins, losses, scratches, batting_avg, avg_win_pct, avg_loss_pct,
    win_loss_ratio, expectancy_pct, total_pl, avg_hold_days_win, avg_hold_days_loss}``.
    Percent fields are FRACTIONS (0.15 = +15%). ``batting_avg`` = wins / all closed
    (a $0 scratch counts against the average but is neither win nor loss);
    ``expectancy_pct`` = mean ``pl_pct`` across ALL closed trades, i.e. batting × avg win
    + (1 − batting) × avg loss with scratches at 0 — the per-trade edge that gates
    progressive exposure. Every ratio degrades to ``None`` (never raises) when its inputs
    are empty, so a fresh account renders as "—" rather than a crash."""
    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    n = len(closed)
    wins = [t for t in closed if t["pl"] > 0]
    losses = [t for t in closed if t["pl"] < 0]
    avg_win = _mean([t["pl_pct"] for t in wins])
    avg_loss = _mean([t["pl_pct"] for t in losses])
    return {
        "n": n, "wins": len(wins), "losses": len(losses),
        "scratches": n - len(wins) - len(losses),
        "batting_avg": (len(wins) / n) if n else None,
        "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
        "win_loss_ratio": (avg_win / abs(avg_loss)) if (avg_win is not None and avg_loss)
                          else None,
        "expectancy_pct": _mean([t["pl_pct"] for t in closed]),
        "total_pl": sum(t["pl"] for t in closed),
        "avg_hold_days_win": _mean([t["hold_days"] for t in wins]),
        "avg_hold_days_loss": _mean([t["hold_days"] for t in losses]),
    }


def fetch_order_fills() -> dict:
    """Pull the cockpit account's FULL closed-order history from Alpaca and normalize the
    filled ones into the fill dicts :func:`build_trade_journal` consumes.

    Pages backwards through ``GetOrdersRequest(status=CLOSED)`` using ``until`` = the
    oldest ``submitted_at`` seen (Alpaca's ``until`` is exclusive, so pages don't overlap;
    orders are deduped by id anyway as cheap insurance). Orders that never filled
    (cancelled/expired, ``filled_qty`` 0) are dropped; partial fills are kept at their
    ``filled_qty`` × ``filled_avg_price``. Returns ``{"account": {account_number, equity,
    cash, using_dedicated}, "fills": [...]}`` with fills sorted oldest-first. Raises
    :class:`TradeUnavailable` on missing package/credentials."""
    client, using_dedicated = _connect_paper()
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
    except ImportError as e:
        raise TradeUnavailable(str(e)) from e
    try:
        from alpaca.common.enums import Sort
        direction = Sort.DESC                          # newest-first pages; until walks back
    except Exception:
        direction = None                               # older alpaca-py: server default (desc)

    acct = client.get_account()
    account = {
        "account_number": getattr(acct, "account_number", "?"),
        "equity": float(acct.equity), "cash": float(acct.cash),
        "using_dedicated": using_dedicated,
    }

    orders: List = []
    until = None
    for _ in range(_MAX_ORDER_PAGES):
        kw = {"status": QueryOrderStatus.CLOSED, "limit": _ORDERS_PAGE_LIMIT}
        if direction is not None:
            kw["direction"] = direction
        if until is not None:
            kw["until"] = until
        page = list(client.get_orders(filter=GetOrdersRequest(**kw)) or [])
        orders.extend(page)
        if len(page) < _ORDERS_PAGE_LIMIT:
            break
        stamps = [s for s in (getattr(o, "submitted_at", None) for o in page)
                  if s is not None]
        nxt = min(stamps) if stamps else None
        if nxt is None or (until is not None and nxt >= until):
            break                                      # no progress -> stop, don't spin
        until = nxt

    fills: List[dict] = []
    seen = set()
    for o in orders:
        oid = str(getattr(o, "id", "") or "")
        if oid and oid in seen:
            continue
        seen.add(oid)
        sym = getattr(o, "symbol", None)
        qty = _attr_float(o, "filled_qty")
        price = _attr_float(o, "filled_avg_price")
        if not sym or not qty or qty <= 0 or not price or price <= 0:
            continue
        raw_side = getattr(o, "side", "")
        side = str(getattr(raw_side, "value", raw_side)).lower()
        side = "buy" if "buy" in side else ("sell" if "sell" in side else None)
        if side is None:
            continue
        fills.append({
            "symbol": sym, "side": side, "qty": qty, "price": price,
            "time": getattr(o, "filled_at", None) or getattr(o, "submitted_at", None),
            "order_id": oid,
            "client_order_id": str(getattr(o, "client_order_id", "") or ""),
        })
    fills.sort(key=_fill_time)
    return {"account": account, "fills": fills}
