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


def _lae_trade(sym, entry, exit_, avg_entry, pl_pct):
    import pandas as pd
    return {"symbol": sym, "entry_date": pd.Timestamp(entry, tz="UTC"),
            "exit_date": pd.Timestamp(exit_, tz="UTC"), "avg_entry": avg_entry,
            "pl_pct": pl_pct, "pl": pl_pct * 1000.0, "hold_days": 5}


def test_loss_adjustment_book_variant():
    """§6.73 (audit #3): the book's Loss Adjustment Exercise — every loss bigger than X is
    cut to −X, wins and smaller losses untouched, compounded in exit order. Without price
    frames every trade is judged on the book value."""
    from src.stock_screener.cockpit.trade import loss_adjustment_sweep

    closed = [_lae_trade("W", "2026-06-01 15:00", "2026-06-10 15:00", 100.0, 0.20),
              _lae_trade("L1", "2026-06-02 15:00", "2026-06-05 15:00", 50.0, -0.08),
              _lae_trade("L2", "2026-06-03 15:00", "2026-06-06 15:00", 20.0, -0.03)]
    sw = loss_adjustment_sweep(closed, xs=(5.0, 10.0))
    assert sw["n"] == 3 and sw["wins"] == 1 and sw["n_price_aware"] == 0
    assert abs(sw["actual_total"] - (1.2 * 0.92 * 0.97 - 1)) < 1e-12
    r5, r10 = sw["rows"]
    assert abs(r5["book_total"] - (1.2 * 0.95 * 0.97 - 1)) < 1e-12     # -8% cut to -5%
    assert abs(r10["book_total"] - sw["actual_total"]) < 1e-12          # nothing past -10%
    assert abs(r5["book_expectancy"] - (0.20 - 0.05 - 0.03) / 3) < 1e-12
    assert r5["aware_total"] == r5["book_total"]                        # no frames -> book
    assert r5["winners_stopped"] == 0


def test_loss_adjustment_price_aware():
    """§6.73: the price-aware variant replays the stop on the trade's own bars. A winner
    that dipped X first becomes a −X loss (the cost the book variant hides); a gap below
    the stop fills at the open, worse than −X; the entry day's low is ignored (it may
    predate the fill); a frame ending before the exit falls back to the book value; and
    stop_floor_from_sweep picks the tightest grid stop (>= 3%) that keeps every winner."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import loss_adjustment_sweep, stop_floor_from_sweep

    idx = pd.bdate_range("2026-06-01", "2026-06-12")
    # W: bought 100 on Jun 1 (entry-day low 90 must NOT count), dipped to 96 on Jun 3,
    # sold +20% on Jun 12.
    w = pd.DataFrame({"Open": 100.0, "High": 121.0, "Low": 99.0, "Close": 110.0},
                     index=idx)
    w.loc["2026-06-01", "Low"] = 90.0
    w.loc["2026-06-03", "Low"] = 96.0
    # G: bought 50, gapped to open at 44 (-12%) on Jun 4, sold -15% that day.
    g = pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.5, "Close": 50.0}, index=idx)
    g.loc["2026-06-04", ["Open", "Low"]] = [44.0, 42.0]
    closed = [_lae_trade("W", "2026-06-01 19:00", "2026-06-12 19:00", 100.0, 0.20),
              _lae_trade("G", "2026-06-02 19:00", "2026-06-04 19:00", 50.0, -0.15),
              _lae_trade("OLD", "2026-06-02 19:00", "2026-07-20 19:00", 10.0, -0.09)]
    frames = {"W": w, "G": g, "OLD": w.iloc[:3]}                 # OLD's frame ends early
    sw = loss_adjustment_sweep(closed, xs=(3.0, 4.0, 5.0, 10.0), frames=frames)
    assert sw["n_price_aware"] == 2 and sw["wins_price_aware"] == 1
    by = {r["x"]: r for r in sw["rows"]}
    # 3%: W's Jun-3 dip to 96 (-4%) stops it at -3%; G gaps to -12% at the open; OLD is
    # book-capped at -3%.
    assert by[3.0]["winners_stopped"] == 1
    assert abs(by[3.0]["aware_total"] - (0.97 * 0.88 * 0.97 - 1)) < 1e-12
    assert abs(by[3.0]["book_total"] - (1.20 * 0.97 * 0.97 - 1)) < 1e-12
    # 4%: the dip reaches exactly -4% -> still stopped; 5%: the winner survives
    assert by[4.0]["winners_stopped"] == 1 and by[5.0]["winners_stopped"] == 0
    # 10%: nothing but the gap trade (-12% open) hits the stop
    assert abs(by[10.0]["aware_total"] - (1.20 * 0.88 * 0.91 - 1)) < 1e-12
    assert stop_floor_from_sweep(sw) == 0.05
    # no winner with bars to judge -> no floor (the caller keeps the default)
    assert stop_floor_from_sweep(loss_adjustment_sweep(closed, xs=(3.0, 5.0))) is None


def test_derived_stop_pct():
    """§6.74 (audit #2): the book sizes the stop at no more than half the average gain.
    Below 5 cockpit wins it stays off (the reason says how many there are); then ½ × the
    average win, floored at DERIVED_STOP_FLOOR (a small average win must not put the stop
    inside daily noise) and capped at the 10% max loss. Manual (untagged) trades are a
    different trader and never count."""
    from src.stock_screener.cockpit.doctrine import DERIVED_STOP_FLOOR
    from src.stock_screener.cockpit.trade import derived_stop_pct

    def T(pl_pct, tagged=True):
        return {"pl": pl_pct * 1000.0, "pl_pct": pl_pct, "hold_days": 5, "tagged": tagged}

    few = derived_stop_pct([T(0.12)] * 4 + [T(-0.05)] * 6 + [T(0.30, tagged=False)] * 3)
    assert few["stop_pct"] is None and few["wins"] == 4 and "you have 4" in few["reason"]

    ok = derived_stop_pct([T(0.12)] * 5 + [T(-0.05)] * 5)
    assert abs(ok["stop_pct"] - 0.06) < 1e-12 and "6.0%" in ok["reason"]

    cap = derived_stop_pct([T(0.30)] * 5)
    assert cap["stop_pct"] == 0.10 and "capped at 10%" in cap["reason"]

    floor = derived_stop_pct([T(0.05)] * 6)                  # half of 5% = 2.5%
    assert floor["stop_pct"] == DERIVED_STOP_FLOOR and "floored" in floor["reason"]
    assert derived_stop_pct([])["stop_pct"] is None


def test_build_buy_plan_stop_pct():
    """§6.74: with the derived stop active, a buy's stop sits stop_pct below its
    worst-case fill; a tighter stop already on the name is kept; the same "stop above the
    market" skip applies to a limit far above the price; held names and stop_pct=None are
    untouched."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import build_buy_plan

    def _pl(price, stop):
        idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=3)
        df = pd.DataFrame({"Open": price, "High": price, "Low": price,
                           "Close": price, "Volume": 1000}, index=idx)
        return {"df": df, "levels": {"pivot": 100.0, "buy_zone": (100.0, 105.0),
                                     "stop": stop}}

    base = {"A": _pl(100.0, 92.5)}
    p, _ = build_buy_plan(["A"], base, mode="shares", amount=5, stop_pct=0.06)
    assert p[0]["stop_price"] == 94.0 and p[0]["stop_derived"] is True
    p, _ = build_buy_plan(["A"], base, mode="shares", amount=5)                  # off
    assert p[0]["stop_price"] == 92.5 and p[0]["stop_derived"] is False
    p, _ = build_buy_plan(["A"], {"A": _pl(100.0, 96.0)}, mode="shares", amount=5,
                          stop_pct=0.06)                           # support is tighter
    assert p[0]["stop_price"] == 96.0 and p[0]["stop_derived"] is False
    p, _ = build_buy_plan(["A"], base, mode="shares", amount=5, stop_pct=0.10)
    assert p[0]["stop_price"] == 92.5, "a looser derived stop never widens the default"
    p, _ = build_buy_plan(["A"], base, mode="shares", amount=5, stop_pct=0.06,
                          held={"A": 10})
    assert p[0]["stop_price"] == 92.5, "held names keep their re-arm stop"

    # limit 105 with a 4% derived stop -> 100.80; the name at 101 plans, at 100 it skips
    p, _ = build_buy_plan(["A"], {"A": _pl(101.0, 92.5)}, mode="shares", amount=5,
                          order_type="limit", stop_pct=0.04)
    assert p[0]["stop_price"] == 100.8
    p, s = build_buy_plan(["A"], {"A": _pl(100.0, 92.5)}, mode="shares", amount=5,
                          order_type="limit", stop_pct=0.04)
    assert not p and "average win" in s[0]["reason"], s


def test_trade_path_tz_dates():
    """§6.73: fill times are UTC, bars are exchange dates. An after-hours sell at 20:30 ET
    is 00:30 UTC the NEXT day — the path must end on the ET date, and the entry day is
    excluded."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import trade_path

    idx = pd.bdate_range("2026-06-01", "2026-06-10")
    df = pd.DataFrame({"Open": 1.0, "Low": 1.0}, index=idx)
    t = {"entry_date": pd.Timestamp("2026-06-01 13:35", tz="UTC"),      # 09:35 ET Jun 1
         "exit_date": pd.Timestamp("2026-06-05 00:30", tz="UTC")}       # 20:30 ET Jun 4
    p = trade_path(t, df)
    assert [d.day for d in p.index] == [2, 3, 4], list(p.index)
    assert trade_path(t, None) is None
    assert trade_path(t, df.iloc[:2]) is None                            # ends before exit
    same = {"entry_date": t["entry_date"], "exit_date": pd.Timestamp("2026-06-01 19:00",
                                                                     tz="UTC")}
    assert len(trade_path(same, df)) == 0


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



def _check_fills_read_per_session(page):
    """The per-session order-history contract, driven through the Journal page at ``page``."""
    from streamlit.testing.v1 import AppTest
    from unittest.mock import patch
    from src.stock_screener.cockpit import trade

    calls = []

    def _fake_fills():
        calls.append(1)
        return {"account": {"account_number": "PA00SZOE", "cash": 10000.0,
                            "equity": 50000.0 + 1000.0 * len(calls),   # every read is distinct
                            "using_dedicated": True},
                "fills": [
                    {"symbol": "AAA", "side": "buy", "qty": 10.0, "price": 100.0,
                     "time": "2026-06-01T14:30:00Z", "order_id": "1",
                     "client_order_id": "SEPAoto-AAA-1"},
                    {"symbol": "AAA", "side": "sell", "qty": 10.0, "price": 111.0,
                     "time": "2026-06-10T14:30:00Z", "order_id": "2",
                     "client_order_id": "SEPAstop-AAA-2"}]}

    with patch.object(trade, "fetch_order_fills", side_effect=_fake_fills):
        a = AppTest.from_file(page, default_timeout=60)
        a.run()
        assert not a.exception, f"session A raised: {a.exception}"
        assert len(calls) == 1 and "equity $51,000" in _rendered_text(a), len(calls)

        b = AppTest.from_file(page, default_timeout=60)      # a second visitor, same server
        b.run()
        assert not b.exception, f"session B raised: {b.exception}"
        assert len(calls) == 2, (f"session B must read the order history itself, not reuse "
                                 f"A's (fetches={len(calls)})")
        assert "equity $52,000" in _rendered_text(b), "session B shows another session's read"

        a.checkbox[0].uncheck().run()                        # an ordinary widget rerun
        assert len(calls) == 2, "a widget rerun inside FILLS_MAX_AGE_S must reuse the read"

        [r for r in a.button if "Refresh" in str(r.label)][0].click().run()
        assert len(calls) == 3 and "equity $53,000" in _rendered_text(a), len(calls)

        memo = dict(a.session_state["fills_memo"])
        memo["mono"] -= 3600                    # age the read past FILLS_MAX_AGE_S
        a.session_state["fills_memo"] = memo
        a.checkbox[0].check().run()
        assert len(calls) == 4, "a read older than FILLS_MAX_AGE_S must be re-fetched"
        assert not a.exception, f"session A raised: {a.exception}"


def test_order_history_read_per_session():
    """journal_cache.cached_fills reads per browser session; reruns reuse it; age and the
    Journal's Refresh expire it.

    It was @st.cache_data keyed on jr_nonce, which every session starts at 1: ONE entry per
    server with no expiry, so (found 2026-09-23 beside the Positions-page staleness) a new buy
    stayed out of the Positions page's P1 entry dates, the Journal, and the trade panel's
    risk sizing until someone pressed Refresh on the Journal. Against the pre-fix
    journal_cache this fails at session B (fetches stays 1)."""
    try:
        from streamlit.testing.v1 import AppTest  # noqa: F401
    except Exception as e:
        print(f"  SKIP test_order_history_read_per_session (AppTest unavailable: {e})")
        return
    _check_fills_read_per_session(
        str(ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "3_Journal.py"))


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "journal"))
