"""Cockpit tests — the trade journal — fills to episodes, stats, risk suggestion.

Runs standalone (`python tests/cockpit/test_journal.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_build_trade_journal():
    """Fills group into position episodes (flat → long → flat = one closed trade): scale-ins
    average, partial sells stay open, a re-entry starts a NEW episode, an orphan sell is
    recorded as unmatched, and the SEPA tag on any fill marks the whole episode."""
    from src.stock_screener.cockpit.trade import build_trade_journal

    def F(sym, side, qty, price, t, coid=""):
        return {"symbol": sym, "side": side, "qty": qty, "price": price, "time": t,
                "client_order_id": coid}

    fills = [
        # WIN: two-lot entry (only the first is tagged), one full exit -> closed episode
        F("WIN", "buy", 10, 100.0, "2026-06-01", "SEPAoto-WIN-1"),
        F("WIN", "buy", 10, 110.0, "2026-06-03"),
        F("WIN", "sell", 20, 120.0, "2026-06-11", "SEPAstop-WIN-2"),
        # ...then a re-entry weeks later -> a SECOND, separate episode, still open
        F("WIN", "buy", 5, 130.0, "2026-06-20"),
        # LOSS: untagged (manual) round trip
        F("LOSS", "buy", 5, 50.0, "2026-06-02"),
        F("LOSS", "sell", 5, 45.0, "2026-06-05"),
        # OPEN: partial sell -> episode stays open with realized-so-far P&L
        F("OPEN", "buy", 10, 20.0, "2026-06-04", "SEPAcockpit-OPEN-1"),
        F("OPEN", "sell", 4, 30.0, "2026-06-09"),
        # ORPH: sell with no prior buy in the history -> unmatched, never guessed at
        F("ORPH", "sell", 3, 10.0, "2026-06-05"),
    ]
    # input order must not matter — the builder sorts by fill time
    j = build_trade_journal(list(reversed(fills)))

    closed = {t["symbol"]: t for t in j["closed"]}
    assert set(closed) == {"WIN", "LOSS"}, sorted(closed)
    w = closed["WIN"]
    assert w["shares"] == 20 and abs(w["avg_entry"] - 105.0) < 1e-9
    assert abs(w["avg_exit"] - 120.0) < 1e-9
    assert abs(w["pl"] - 300.0) < 1e-9                       # 2400 - 2100
    assert abs(w["pl_pct"] - 300.0 / 2100.0) < 1e-9
    assert w["hold_days"] == 10 and w["n_fills"] == 3
    assert w["tagged"] is True                               # entry tag marks the episode
    lo = closed["LOSS"]
    assert abs(lo["pl"] - (-25.0)) < 1e-9 and lo["hold_days"] == 3
    assert lo["tagged"] is False

    opens = {t["symbol"]: t for t in j["open"]}
    assert set(opens) == {"WIN", "OPEN"}, sorted(opens)      # the re-entry is its own episode
    assert opens["WIN"]["shares_open"] == 5 and abs(opens["WIN"]["realized_pl"]) < 1e-9
    o = opens["OPEN"]
    assert o["shares_open"] == 6 and o["tagged"] is True
    assert abs(o["realized_pl"] - 40.0) < 1e-9               # 4 sold at 30 vs avg cost 20

    assert len(j["unmatched_sells"]) == 1
    assert j["unmatched_sells"][0]["symbol"] == "ORPH"

    # junk fills (qty/price <= 0, unknown side) are ignored, never a raise
    junk = [F("AAA", "buy", 0, 100.0, "2026-06-01"), F("AAA", "hold", 5, 100.0, "2026-06-02"),
            F("AAA", "buy", 5, 0.0, "2026-06-03")]
    j2 = build_trade_journal(junk)
    assert j2["closed"] == [] and j2["open"] == [] and j2["unmatched_sells"] == []


def test_journal_stats():
    """journal_stats: batting counts wins over ALL closed (scratches included), expectancy is
    the mean per-trade P&L%, and every ratio degrades to None on an empty side."""
    from src.stock_screener.cockpit.trade import journal_stats

    def T(pl, pl_pct, hold):
        return {"pl": pl, "pl_pct": pl_pct, "hold_days": hold}

    s = journal_stats([T(100.0, 0.10, 10), T(200.0, 0.20, 20),
                       T(-50.0, -0.10, 5), T(0.0, 0.0, 2)])
    assert s["n"] == 4 and s["wins"] == 2 and s["losses"] == 1 and s["scratches"] == 1
    assert abs(s["batting_avg"] - 0.5) < 1e-9
    assert abs(s["avg_win_pct"] - 0.15) < 1e-9
    assert abs(s["avg_loss_pct"] - (-0.10)) < 1e-9
    assert abs(s["win_loss_ratio"] - 1.5) < 1e-9
    assert abs(s["expectancy_pct"] - 0.05) < 1e-9            # (0.10+0.20-0.10+0)/4
    assert abs(s["total_pl"] - 250.0) < 1e-9
    assert abs(s["avg_hold_days_win"] - 15.0) < 1e-9
    assert abs(s["avg_hold_days_loss"] - 5.0) < 1e-9

    empty = journal_stats([])
    assert empty["n"] == 0 and empty["total_pl"] == 0.0
    for k in ("batting_avg", "avg_win_pct", "avg_loss_pct", "win_loss_ratio",
              "expectancy_pct", "avg_hold_days_win", "avg_hold_days_loss"):
        assert empty[k] is None, k

    # all-winner history: the loss side is None, ratio undefined, batting 1.0
    allwin = journal_stats([T(10.0, 0.05, 3)])
    assert allwin["batting_avg"] == 1.0
    assert allwin["avg_loss_pct"] is None and allwin["win_loss_ratio"] is None


def test_suggest_risk_pct():
    """§6.51 progressive exposure: last-10 form maps to pilot 0.5% (negative expectancy
    or batting < .300), base 1.0% (in between, or a thin/empty sample), strong 1.25%
    (batting ≥ .500 with positive expectancy — expectancy is checked FIRST, so a .500
    hitter with negative expectancy still pilots). The exit-date re-sort is pinned:
    build_trade_journal emits closed trades grouped by SYMBOL, so an unsorted "last 10"
    would read an alphabetical accident, not recent form."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import suggest_risk_pct

    def T(sym, exit_day, pl_pct):
        return {"symbol": sym, "exit_date": pd.Timestamp(f"2026-07-{exit_day:02d}"),
                "pl": pl_pct * 1000.0, "pl_pct": pl_pct, "hold_days": 5}

    # pilot: 2W/8L, negative expectancy
    pilot = suggest_risk_pct([T(f"S{i}", i + 1, 0.10 if i < 2 else -0.06)
                              for i in range(10)])
    assert pilot["risk_pct"] == 0.5 and "pilot" in pilot["reason"]
    assert "2W/8L" in pilot["reason"]

    # strong: 6W/4L, positive expectancy
    strong = suggest_risk_pct([T(f"S{i}", i + 1, 0.10 if i < 6 else -0.05)
                               for i in range(10)])
    assert strong["risk_pct"] == 1.25 and "press" in strong["reason"]

    # base: batting .400 with positive expectancy sits between the gates
    base = suggest_risk_pct([T(f"S{i}", i + 1, 0.15 if i < 4 else -0.05)
                             for i in range(10)])
    assert base["risk_pct"] == 1.0 and "normal" in base["reason"]

    # .500 batting but negative expectancy (small wins, big losses) -> still pilot
    churn = suggest_risk_pct([T(f"S{i}", i + 1, 0.01 if i < 5 else -0.10)
                              for i in range(10)])
    assert churn["risk_pct"] == 0.5

    # thin sample and empty history -> base, flagged as such
    thin = suggest_risk_pct([T(f"S{i}", i + 1, 0.10) for i in range(3)])
    assert thin["risk_pct"] == 1.0 and "sample" in thin["reason"]
    empty = suggest_risk_pct([])
    assert empty["risk_pct"] == 1.0 and empty["n"] == 0

    # exit-date sorting: 5 OLD winners under late-alphabet symbols, 7 RECENT losers under
    # early-alphabet symbols. Symbol-grouped order puts the losers first, so an unsorted
    # tail would read 5W/5L (strong); the true last 10 by exit date read 3W/7L -> pilot.
    trades = ([T(f"ZZ{i}", i + 1, 0.10) for i in range(5)]          # exits Jul 1-5
              + [T(f"AA{i}", i + 10, -0.05) for i in range(7)])     # exits Jul 10-16
    grouped = sorted(trades, key=lambda t: t["symbol"])             # journal emit order
    sorted_read = suggest_risk_pct(grouped)
    assert sorted_read["risk_pct"] == 0.5, \
        f"last-10 must be by exit date, not symbol order: {sorted_read['reason']}"


def test_fetch_order_fills_offline():
    """fetch_order_fills pages through the closed-order history with until= (exclusive),
    drops never-filled orders, normalizes sides/qty/price, and returns fills oldest-first —
    against a fake client with the page size patched down to force pagination."""
    import datetime as _dt
    from src.stock_screener.cockpit import trade
    from alpaca.trading.enums import OrderSide

    def _ts(day):
        return _dt.datetime(2026, 6, day, 15, 0, tzinfo=_dt.timezone.utc)

    class _O:
        def __init__(self, oid, symbol, side, fqty, fprice, day, coid=""):
            self.id, self.symbol, self.side = oid, symbol, side
            self.filled_qty, self.filled_avg_price = fqty, fprice
            self.submitted_at = self.filled_at = _ts(day)
            self.client_order_id = coid

    class FakeClient:
        def __init__(self, orders):
            self._orders = sorted(orders, key=lambda o: o.submitted_at, reverse=True)

        def get_account(self):
            class _A:
                equity = "50000"; cash = "10000"; account_number = "PA00SZOE"
            return _A()

        def get_orders(self, filter=None):
            until = getattr(filter, "until", None)
            limit = getattr(filter, "limit", None) or 500
            out = [o for o in self._orders
                   if until is None or o.submitted_at < until]     # Alpaca until = exclusive
            return out[:limit]

    orders = [
        _O("1", "AAA", OrderSide.BUY, "10", "100.0", 1, "SEPAoto-AAA-1"),
        _O("2", "AAA", OrderSide.SELL, "10", "111.0", 8, "SEPAstop-AAA-2"),
        _O("3", "BBB", OrderSide.BUY, "5", "20.0", 3, "SEPAcockpit-BBB-3"),
        _O("4", "CCC", OrderSide.BUY, "0", None, 4),               # cancelled, no fill -> drop
        _O("5", "DDD", "sell", "2", "30.0", 5),                    # plain-string side works too
    ]
    fake = FakeClient(orders)
    orig_conn, orig_lim = trade._connect_paper, trade._ORDERS_PAGE_LIMIT
    trade._connect_paper = lambda: (fake, True)
    trade._ORDERS_PAGE_LIMIT = 2                                    # force several pages
    try:
        out = trade.fetch_order_fills()
    finally:
        trade._connect_paper, trade._ORDERS_PAGE_LIMIT = orig_conn, orig_lim

    assert out["account"]["account_number"] == "PA00SZOE"
    assert out["account"]["using_dedicated"] is True
    fills = out["fills"]
    assert [f["order_id"] for f in fills] == ["1", "3", "5", "2"], \
        [f["order_id"] for f in fills]                              # oldest-first, "4" dropped
    assert all(f["side"] in ("buy", "sell") for f in fills)
    assert fills[0]["side"] == "buy" and fills[-1]["side"] == "sell"
    assert isinstance(fills[0]["qty"], float) and fills[0]["qty"] == 10.0
    assert fills[0]["price"] == 100.0
    assert fills[0]["client_order_id"] == "SEPAoto-AAA-1"

    # the round trip these fills describe closes cleanly through the journal builder
    j = trade.build_trade_journal(fills)
    assert {t["symbol"] for t in j["closed"]} == {"AAA"}
    assert abs(j["closed"][0]["pl"] - 110.0) < 1e-9



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "journal"))
