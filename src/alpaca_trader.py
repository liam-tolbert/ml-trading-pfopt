"""
alpaca_trader.py — Alpaca paper-account auto-trader for the All-Weather mirror.

Reconciles a target portfolio (parsed from the All-Weather Google Sheet via
track_portfolio.aggregate_positions) against current Alpaca holdings and submits
market orders to align share counts. Sizing matches the sheet's
percentage-of-portfolio allocation, scaled to the user's Alpaca equity.

Requires `alpaca-py` and an Alpaca paper account.

Environment variables (read from .env):
    ALPACA_API_KEY=...
    ALPACA_API_SECRET=...
    ALPACA_BASE_URL=https://paper-api.alpaca.markets   # informational; paper=True is enforced

Live trading is intentionally disabled in this version — the TradingClient is
always constructed with paper=True.
"""

import math
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except ImportError as e:
    raise ImportError(
        "alpaca-py is required for Alpaca trading. "
        "Install with: pip install alpaca-py"
    ) from e


MIN_TRADE_USD = 50.0
MAX_ORDER_PCT = 0.10  # of account equity
ORDER_FILL_TIMEOUT_S = 30


@dataclass
class PlannedOrder:
    ticker: str
    action: str       # "BUY" or "SELL"
    qty: int          # positive shares
    price: float      # estimated price (from sheet)
    est_value: float  # qty * price (always positive)
    target_pct: float # target % of portfolio
    current_shares: int
    target_shares: int


def connect() -> TradingClient:
    """Construct an Alpaca paper TradingClient from env credentials."""
    load_dotenv()
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env"
        )
    return TradingClient(api_key, api_secret, paper=True)


def get_account_state(client: TradingClient) -> tuple[float, dict[str, int]]:
    """Return (equity_usd, {ticker: current_shares}) from Alpaca."""
    account = client.get_account()
    equity = float(account.equity)

    current: dict[str, int] = {}
    for pos in client.get_all_positions():
        current[pos.symbol] = int(float(pos.qty))
    return equity, current


def validate_tradable(client: TradingClient, tickers: list[str]) -> set[str]:
    """Return the subset of tickers that exist on Alpaca and are tradable."""
    ok: set[str] = set()
    for t in tickers:
        try:
            asset = client.get_asset(t)
            if asset.tradable:
                ok.add(t)
            else:
                print(f"  ! {t}: asset exists but not tradable")
        except Exception as e:
            print(f"  ! {t}: lookup failed ({e})")
    return ok


def build_target_shares(positions: dict, equity: float) -> dict[str, int]:
    """
    Compute integer target share counts, sized so each ticker takes the same
    fraction of `equity` as it has of the sheet's total active value.
    """
    total_value = sum(
        p["total_current_value"]
        for p in positions.values()
        if p["total_shares"] > 0
    )
    if total_value <= 0 or equity <= 0:
        return {}

    target: dict[str, int] = {}
    for ticker, pos in positions.items():
        if pos["total_shares"] <= 0 or pos["total_current_value"] <= 0:
            continue
        price = pos.get("current_price")
        if not price or price <= 0:
            continue
        target_pct = pos["total_current_value"] / total_value
        target_dollars = target_pct * equity
        target[ticker] = math.floor(target_dollars / price)
    return target


def build_order_plan(
    positions: dict,
    target_shares: dict[str, int],
    current_positions: dict[str, int],
    tradable: set[str],
) -> tuple[list[PlannedOrder], list[str]]:
    """
    Build a list of orders. Orphan policy: tickers held in Alpaca but absent
    from the sheet are flagged for warning and never auto-sold.
    """
    total_value = sum(
        p["total_current_value"]
        for p in positions.values()
        if p["total_shares"] > 0
    )

    plan: list[PlannedOrder] = []
    sheet_tickers = set(target_shares)
    orphans = [
        t for t in current_positions
        if t not in sheet_tickers and current_positions[t] > 0
    ]

    for ticker in sorted(sheet_tickers):
        if ticker not in tradable:
            continue
        target = target_shares[ticker]
        current = current_positions.get(ticker, 0)
        delta = target - current
        if delta == 0:
            continue

        price = positions[ticker]["current_price"]
        qty = abs(delta)
        est_value = qty * price
        if est_value < MIN_TRADE_USD:
            continue

        action = "BUY" if delta > 0 else "SELL"
        target_pct = (
            positions[ticker]["total_current_value"] / total_value
            if total_value > 0 else 0.0
        )

        plan.append(PlannedOrder(
            ticker=ticker,
            action=action,
            qty=qty,
            price=price,
            est_value=est_value,
            target_pct=target_pct,
            current_shares=current,
            target_shares=target,
        ))

    return plan, orphans


def print_order_plan(plan: list[PlannedOrder], equity: float, orphans: list[str]) -> None:
    """Pretty-print the order plan as a table."""
    w = 100
    print("\n" + "=" * w)
    print(f"  ORDER PLAN  |  Equity: ${equity:,.2f}  |  {len(plan)} orders")
    print("=" * w)

    if orphans:
        print(f"  ! Held in Alpaca but not in sheet (ignored, not sold): {', '.join(orphans)}")
        print("-" * w)

    if not plan:
        print("  (no orders — portfolio already matches target)")
        print("=" * w)
        return

    print(
        f"  {'Ticker':<8} {'Action':<6} {'Curr':>8}    {'Target':>8} {'Qty':>9} "
        f"{'Price':>10} {'Est $':>14} {'Target%':>9}"
    )
    print("-" * w)
    total_buy = 0.0
    total_sell = 0.0
    for o in plan:
        sign = "+" if o.action == "BUY" else "-"
        print(
            f"  {o.ticker:<8} {o.action:<6} {o.current_shares:>8d}    {o.target_shares:>8d} "
            f"{sign}{o.qty:>8d} ${o.price:>8,.2f} ${o.est_value:>12,.2f} {o.target_pct*100:>8.2f}%"
        )
        if o.action == "BUY":
            total_buy += o.est_value
        else:
            total_sell += o.est_value
    print("-" * w)
    print(f"  Total BUY:  ${total_buy:,.2f}")
    print(f"  Total SELL: ${total_sell:,.2f}")
    print(f"  Net cash:   ${total_sell - total_buy:,.2f}")
    print("=" * w)


def submit_orders(
    client: TradingClient,
    plan: list[PlannedOrder],
    equity: float,
) -> list:
    """Submit market orders for each plan entry, poll for fills, print summary."""
    max_allowed = MAX_ORDER_PCT * equity
    for o in plan:
        if o.est_value > max_allowed:
            raise RuntimeError(
                f"Order for {o.ticker} (${o.est_value:,.2f}) exceeds "
                f"{MAX_ORDER_PCT*100:.0f}% of equity (${max_allowed:,.2f}). "
                "Aborting entire plan."
            )

    submitted = []
    for o in plan:
        req = MarketOrderRequest(
            symbol=o.ticker,
            qty=o.qty,
            side=OrderSide.BUY if o.action == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"AWPmirror-{o.ticker}-{int(time.time())}",
        )
        try:
            resp = client.submit_order(order_data=req)
            submitted.append((o, resp))
            print(f"  -> {o.action} {o.qty} {o.ticker} submitted (id={resp.id})")
        except Exception as e:
            print(f"  ! {o.action} {o.qty} {o.ticker} FAILED: {e}")

    # Poll for fills
    deadline = time.time() + ORDER_FILL_TIMEOUT_S
    pending = {str(r.id): (o, r) for o, r in submitted}
    while pending and time.time() < deadline:
        for oid in list(pending):
            updated = client.get_order_by_id(oid)
            if updated.status in ("filled", "canceled", "rejected", "expired"):
                pending[oid] = (pending[oid][0], updated)
                del pending[oid]
        if pending:
            time.sleep(1)

    print("\nFill summary:")
    for o, resp in submitted:
        final = client.get_order_by_id(str(resp.id))
        avg = float(final.filled_avg_price) if final.filled_avg_price else 0.0
        filled = int(float(final.filled_qty)) if final.filled_qty else 0
        print(
            f"  {o.action:<4} {o.qty:>6} {o.ticker:<8} "
            f"status={final.status:<12} filled={filled} avgPx=${avg:,.2f}"
        )
    return submitted


def run_reconcile(positions: dict, dry_run: bool = False) -> None:
    """Top-level entry: connect, build plan, prompt, submit."""
    print("\nConnecting to Alpaca paper...")
    client = connect()

    equity, current_positions = get_account_state(client)
    print(f"Connected. Equity: ${equity:,.2f}  Positions: {len(current_positions)}")

    active_tickers = [
        t for t, p in positions.items()
        if p["total_shares"] > 0 and p["total_current_value"] > 0
    ]
    print(f"Validating {len(active_tickers)} sheet tickers with Alpaca...")
    tradable = validate_tradable(client, active_tickers)

    not_tradable = [t for t in active_tickers if t not in tradable]
    if not_tradable:
        print(f"  ! Not tradable on Alpaca: {', '.join(not_tradable)}")

    target_shares = build_target_shares(positions, equity)
    plan, orphans = build_order_plan(positions, target_shares, current_positions, tradable)
    print_order_plan(plan, equity, orphans)

    if not plan:
        return

    if dry_run:
        print("\n--dry-run set; no orders submitted.")
        return

    resp = input("\nProceed? [y/N]: ").strip().lower()
    if resp != "y":
        print("Aborted. No orders submitted.")
        return

    submit_orders(client, plan, equity)
