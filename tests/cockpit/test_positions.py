"""Cockpit tests — the Positions page — stop management, P1-P4 sell pillars, the sell plan.

Runs standalone (`python tests/cockpit/test_positions.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_fetch_positions_offline():
    """fetch_positions enriches Alpaca holdings with P&L, the in-force stop, 50-day SMA and
    advisories — against a fake client + an offline price feed (no network)."""
    import pandas as pd
    from src.stock_screener.cockpit import trade, data_feed
    Client, _Pos, _Order = _pos_fakes()

    def _frame(closes, vols=None):
        idx = pd.bdate_range(end=pd.Timestamp("2026-07-08"), periods=len(closes))
        vols = vols if vols is not None else [1000] * len(closes)
        return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                             "Close": closes, "Volume": vols}, index=idx)

    rising = [100 + i * 0.5 for i in range(60)]                 # ends 129.5, SMA below -> not below
    falling = [60 - i * 0.25 for i in range(60)]               # ends 45.25, SMA above -> below_sma50
    frames = {"AAA": _frame(rising), "BBB": _frame(falling, vols=[1000] * 59 + [3000])}

    positions = [
        _Pos("AAA", 10, avg_entry_price=100.0, current_price=130.0, market_value=1300.0,
             cost_basis=1000.0, unrealized_pl=300.0, unrealized_plpc=0.30, lastday_price=128.0),
        _Pos("BBB", 5, avg_entry_price=50.0, current_price=45.0, market_value=225.0,
             cost_basis=250.0, unrealized_pl=-25.0, unrealized_plpc=-0.10, lastday_price=46.0),
    ]
    client = Client(positions, {"AAA": [_Order("s1", "AAA", 120.0)]})   # AAA has a stop, BBB none

    future = (pd.Timestamp.today().normalize() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    orig_conn, orig_gmp = trade._connect_paper, data_feed.get_many_prices
    orig_fund = data_feed.get_fundamentals
    trade._connect_paper = lambda: (client, True)
    data_feed.get_many_prices = lambda syms, **kw: frames
    data_feed.get_fundamentals = lambda t, **kw: {"next_earnings": future}
    try:
        out = trade.fetch_positions()
    finally:
        trade._connect_paper, data_feed.get_many_prices = orig_conn, orig_gmp
        data_feed.get_fundamentals = orig_fund

    acct = out["account"]
    assert acct["positions_count"] == 2
    assert abs(acct["total_unrealized_pl"] - 275.0) < 1e-9      # 300 + (-25)
    by = {p["symbol"]: p for p in out["positions"]}
    assert by["AAA"]["current_stop"] == 120.0 and by["AAA"]["has_stop"] is True
    assert abs(by["AAA"]["gain_pct"] - 0.30) < 1e-9
    assert by["AAA"]["sma_50"] is not None and by["AAA"]["below_sma50"] is False
    assert by["BBB"]["has_stop"] is False
    assert by["BBB"]["below_sma50"] is True                     # last close under its 50-day SMA
    assert any("No protective stop" in a for a in by["BBB"]["advisories"])
    assert any("50-day SMA" in a for a in by["BBB"]["advisories"])
    # Issue 10: the ratio divides by the PRIOR 50 bars (all 1000), EXCLUDING today's 3000 spike,
    # so it reads exactly 3.0 — matching triggers._volume_ratio, not 2.88 (today-in-average).
    assert abs(by["BBB"]["volume_ratio"] - 3.0) < 1e-9
    # Earnings enrichment + stage: BBB is a LOSER 10 days from its report -> cushion advisory;
    # AAA (+30%) is cushioned, so the same imminent report stays silent.
    assert by["AAA"]["next_earnings"] == future and by["AAA"]["earnings_in"] == 10
    assert by["AAA"]["stage"] == "well in profit" and by["BBB"]["stage"] == "underwater"
    assert any("Earnings in 10d with a loss" in a for a in by["BBB"]["advisories"])
    assert not any("Earnings" in a for a in by["AAA"]["advisories"])


def test_suggest_stop():
    """suggest_stop: each basis + auto selection by gain; floors at the in-force stop and (once
    working) at breakeven; returns None when the result isn't below price (underwater)."""
    from src.stock_screener.cockpit.trade import suggest_stop, INITIAL_STOP_PCT

    # explicit bases
    assert suggest_stop(avg_entry=100, current_price=130, sma_50=115, current_stop=None,
                        gain_pct=0.30, basis="initial")[0] == round(100 * (1 - INITIAL_STOP_PCT), 2)
    assert suggest_stop(avg_entry=100, current_price=130, sma_50=115, current_stop=None,
                        gain_pct=0.30, basis="sma50")[0] == round(115 * 0.99, 2)

    # auto picks by stage: fresh -> initial, working -> breakeven, well-in-profit -> sma50
    assert suggest_stop(avg_entry=100, current_price=103, sma_50=98, current_stop=None,
                        gain_pct=0.03, basis="auto")[1] == "initial"
    assert suggest_stop(avg_entry=100, current_price=118, sma_50=110, current_stop=None,
                        gain_pct=0.18, basis="auto")[1] == "breakeven"
    assert suggest_stop(avg_entry=100, current_price=125, sma_50=115, current_stop=None,
                        gain_pct=0.25, basis="auto")[1] == "sma50"

    # never below the in-force stop
    val, _ = suggest_stop(avg_entry=100, current_price=125, sma_50=90, current_stop=118,
                          gain_pct=0.25, basis="sma50")
    assert val == 118.0

    # underwater / result not below price -> None (manual row)
    val2, _ = suggest_stop(avg_entry=100, current_price=90, sma_50=None, current_stop=None,
                           gain_pct=-0.10, basis="initial")
    assert val2 is None


def test_position_advisories():
    """position_advisories emits exactly the four Minervini exit strings when applicable, and
    nothing for a healthy, protected, sub-target position."""
    from src.stock_screener.cockpit.trade import position_advisories

    flagged = position_advisories({"has_stop": False, "gain_pct": 0.22, "below_sma50": True,
                                   "volume_ratio": 1.8, "avg_entry": 100.0, "current_stop": None})
    joined = " | ".join(flagged)
    assert "No protective stop" in joined
    assert "selling part into strength" in joined
    assert "50-day SMA on heavy volume" in joined
    assert "breakeven" in joined

    clean = position_advisories({"has_stop": True, "gain_pct": 0.05, "below_sma50": False,
                                 "volume_ratio": 1.0, "avg_entry": 100.0, "current_stop": 96.0})
    assert clean == []

    # Earnings-cushion rules: a loser into a report, then a thin cushion, both warn; a real
    # cushion, an unknown/negative days-to-earnings, or an unknown gain stay silent.
    base = {"has_stop": True, "below_sma50": False, "volume_ratio": 1.0,
            "avg_entry": 100.0, "current_stop": 96.0}
    lose = position_advisories({**base, "gain_pct": -0.05, "earnings_in": 7})
    assert any("Earnings in 7d with a loss" in a for a in lose)
    thin = position_advisories({**base, "gain_pct": 0.03, "earnings_in": 9})
    assert any("only a 3% cushion" in a for a in thin)
    for quiet in ({**base, "gain_pct": 0.12, "earnings_in": 9},      # cushioned
                  {**base, "gain_pct": -0.05, "earnings_in": None},  # date unknown
                  {**base, "gain_pct": -0.05, "earnings_in": -3},    # just reported
                  {**base, "gain_pct": None, "earnings_in": 7}):     # gain unknown
        assert not any("Earnings" in a for a in position_advisories(quiet)), quiet


def test_sell_pillars():
    """§6.52: the Step-E doctrine as per-position P1-P4 statuses (pure, pinned today).
    P1 fails on Day-0 close below pivot / decisive (>2%) close / 2nd consecutive close /
    close below the breakout bar's low / flat-to-red at day 15+, warns on no-3%-cushion
    at day 10+, degrades without an entry date or pivot. P2 is STRICT (user decision:
    7/8 fails). P3 reads the scan regime, falls back to the trigger report's SPY note.
    P4 fails on loss-or-thin-cushion inside the 21-day window, warns with a real
    cushion. A bare pos dict (no new keys) yields four unknowns, never a raise."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import sell_pillars

    TODAY = "2026-08-12"                                   # a Wednesday

    def P(**kw):
        base = {"last_close": None, "gain_pct": None, "df": None,
                "template_criteria": None, "earnings_in": None}
        base.update(kw)
        return base

    def S(r):                                              # status shorthand
        return {k: v["status"] for k, v in r.items()}

    # all-ok: day 5, cushioned, above pivot, 8/8, risk-on, report far out. The tz-aware
    # UTC entry (the journal's native form) must land on the ET trading date.
    ok = sell_pillars(
        P(last_close=108.0, gain_pct=0.05, template_criteria=8, earnings_in=40,
          df=_trigger_frame(TODAY, [108.0] * 30)),
        entry_date=pd.Timestamp("2026-08-05 14:30", tz="UTC"), pivot=100.0,
        regime={"regime": "RISK-ON (Strong)", "should_generate_buys": True},
        today=TODAY)
    assert S(ok) == {"P1": "ok", "P2": "ok", "P3": "ok", "P4": "ok"}, ok
    assert "day 5" in ok["P1"]["detail"]

    # P1 Day-0: entered today, settled back below the pivot -> the breakout never happened
    d0 = sell_pillars(P(last_close=99.0, gain_pct=-0.01,
                        df=_trigger_frame(TODAY, [101.0] * 29 + [99.0])),
                      entry_date=TODAY, pivot=100.0, today=TODAY)
    assert d0["P1"]["status"] == "fail" and "Day-0" in d0["P1"]["detail"]

    # P1 decisive: closed >2% below the pivot
    dec = sell_pillars(P(last_close=97.4, gain_pct=-0.03), entry_date="2026-08-05",
                       pivot=100.0, today=TODAY)
    assert dec["P1"]["status"] == "fail" and "decisive" in dec["P1"]["detail"]

    # P1 second consecutive close below the pivot (neither decisive on its own)
    two = sell_pillars(P(last_close=99.0, gain_pct=-0.01,
                         df=_trigger_frame(TODAY, [101.0] * 28 + [99.5, 99.0])),
                       entry_date="2026-08-05", pivot=100.0, today=TODAY)
    assert two["P1"]["status"] == "fail" and "consecutive" in two["P1"]["detail"]

    # P1 breakout-bar low: still above the pivot, but under the entry bar's low
    bo = sell_pillars(P(last_close=98.5, gain_pct=-0.015,
                        df=_trigger_frame(TODAY, [100.0] * 25 + [98.5] * 5)),
                      entry_date="2026-08-05", pivot=95.0, today=TODAY)
    assert bo["P1"]["status"] == "fail" and "breakout bar" in bo["P1"]["detail"]

    # P1 laggard clock: day 12 with +1% -> warn; day 18 flat-to-red -> fail
    lag = sell_pillars(P(last_close=101.0, gain_pct=0.01), entry_date="2026-07-27",
                       pivot=100.0, today=TODAY)
    assert lag["P1"]["status"] == "warn" and "cushion" in lag["P1"]["detail"]
    stall = sell_pillars(P(last_close=100.5, gain_pct=-0.01), entry_date="2026-07-17",
                         pivot=100.0, today=TODAY)
    assert stall["P1"]["status"] == "fail" and "flat-to-red" in stall["P1"]["detail"]

    # P1 degradation: no journal episode -> unknown; no pivot -> clock-only partial
    assert sell_pillars(P(), today=TODAY)["P1"]["status"] == "unknown"
    nopiv = sell_pillars(P(last_close=108.0, gain_pct=0.05), entry_date="2026-08-05",
                         today=TODAY)
    assert nopiv["P1"]["status"] == "ok" and "clock only" in nopiv["P1"]["detail"]

    # P2 strict: 8 ok, 7 FAILS (user decision), 5 fails, None unknown
    assert sell_pillars(P(template_criteria=8), today=TODAY)["P2"]["status"] == "ok"
    p27 = sell_pillars(P(template_criteria=7), today=TODAY)["P2"]
    assert p27["status"] == "fail" and "7/8" in p27["detail"]
    assert sell_pillars(P(template_criteria=5), today=TODAY)["P2"]["status"] == "fail"
    assert sell_pillars(P(), today=TODAY)["P2"]["status"] == "unknown"

    # P3: scan regime beats everything; SPY note is the partial fallback
    off = sell_pillars(P(), regime={"regime": "RISK-OFF", "should_generate_buys": False},
                       today=TODAY)
    assert off["P3"]["status"] == "fail" and "risk-off" in off["P3"]["detail"].lower()
    assert sell_pillars(P(), spy_note={"trend": "Bullish"},
                        today=TODAY)["P3"]["status"] == "ok"
    assert sell_pillars(P(), spy_note={"trend": "Bearish"},
                        today=TODAY)["P3"]["status"] == "warn"
    assert sell_pillars(P(), today=TODAY)["P3"]["status"] == "unknown"

    # P4: the earnings window vs the cushion
    assert sell_pillars(P(earnings_in=10, gain_pct=-0.05),
                        today=TODAY)["P4"]["status"] == "fail"          # loss into report
    thin = sell_pillars(P(earnings_in=10, gain_pct=0.02), today=TODAY)["P4"]
    assert thin["status"] == "fail" and "trim/exit" in thin["detail"]   # thin cushion
    assert sell_pillars(P(earnings_in=10, gain_pct=0.10),
                        today=TODAY)["P4"]["status"] == "warn"          # real cushion
    assert sell_pillars(P(earnings_in=40, gain_pct=0.10),
                        today=TODAY)["P4"]["status"] == "ok"
    assert sell_pillars(P(earnings_in=-5), today=TODAY)["P4"]["status"] == "ok"
    assert sell_pillars(P(), today=TODAY)["P4"]["status"] == "unknown"

    # a bare dict with NONE of the new keys: four unknowns, no raise
    bare = sell_pillars({}, today=TODAY)
    assert S(bare) == {"P1": "unknown", "P2": "unknown", "P3": "unknown",
                       "P4": "unknown"}, bare


def test_position_stage():
    """The stop-ladder stage label mirrors suggest_stop's auto thresholds exactly."""
    from src.stock_screener.cockpit.trade import position_stage

    assert position_stage(None) is None
    assert position_stage(-0.02) == "underwater"
    assert position_stage(0.0) == "fresh"
    assert position_stage(0.15) == "fresh"
    assert position_stage(0.16) == "working"
    assert position_stage(0.19) == "working"
    assert position_stage(0.20) == "well in profit"


def test_submit_position_sell():
    """Manual sell: cancels the covering stop, submits a SEPAsell- market SELL (DAY), and on a
    partial sell re-places the GTC stop for the remainder at the SAME level. Skips unheld
    names, clamps an over-ask, and notes an unprotected remainder when no stop existed."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
    from src.stock_screener.cockpit import trade

    Client, _Pos, _Order = _pos_fakes()

    def _run(client, symbol, qty):
        orig = trade._connect_paper
        trade._connect_paper = lambda: (client, True)
        try:
            return trade.submit_position_sell(symbol, qty)
        finally:
            trade._connect_paper = orig

    # (a) FULL sell: stop cancelled, one market SELL DAY for all 40, NO stop re-placed.
    c = Client([_Pos("HELD", 40)], {"HELD": [_Order("s1", "HELD", 95.0)]})
    r = _run(c, "HELD", 40)
    assert r["status"] == "submitted" and r["sold_qty"] == 40 and r["remaining"] == 0
    assert "s1" in c.cancelled
    assert len(c.submitted) == 1
    mkt = c.submitted[0]
    assert isinstance(mkt, MarketOrderRequest) and mkt.side == OrderSide.SELL
    assert int(float(mkt.qty)) == 40 and mkt.time_in_force == TimeInForce.DAY
    assert str(mkt.client_order_id).startswith("SEPAsell-HELD-")
    assert "no shares remain" in r["detail"]

    # (b) PARTIAL sell: market SELL 15, then a GTC stop re-placed for the remaining 25 at 95.0.
    c2 = Client([_Pos("HELD", 40)], {"HELD": [_Order("s1", "HELD", 95.0)]})
    r2 = _run(c2, "HELD", 15)
    assert r2["status"] == "submitted" and r2["remaining"] == 25
    assert len(c2.submitted) == 2
    stop = c2.submitted[1]
    assert isinstance(stop, StopOrderRequest) and int(float(stop.qty)) == 25
    assert float(stop.stop_price) == 95.0 and stop.time_in_force == TimeInForce.GTC
    assert str(stop.client_order_id).startswith("SEPAstop-HELD-")
    assert "re-placed @ 95.00" in r2["detail"]

    # (c) over-ask clamps to held; (d) unheld/zero-qty -> skipped, nothing touched.
    c3 = Client([_Pos("HELD", 40)], {"HELD": [_Order("s1", "HELD", 95.0)]})
    r3 = _run(c3, "HELD", 100)
    assert r3["sold_qty"] == 40 and "clamped" in r3["detail"]
    c4 = Client([_Pos("HELD", 40)], {})
    assert _run(c4, "NOPE", 10)["status"] == "skipped"
    assert _run(c4, "HELD", 0)["status"] == "skipped"
    assert not c4.cancelled and not c4.submitted

    # (e) partial with NO existing stop: no cancel, no re-place, loud unprotected note.
    c5 = Client([_Pos("HELD", 40)], {})
    r5 = _run(c5, "HELD", 10)
    assert r5["status"] == "submitted" and len(c5.submitted) == 1 and not c5.cancelled
    assert "have no stop" in r5["detail"]

    assert "SEPAsell-" in trade.SEPA_TAG_PREFIXES     # journal tags manual sells


def test_submit_position_sell_restores_stop_on_failure():
    """The cancel-before-sell gap done right: if the market SELL fails AFTER the covering stop
    was cancelled, the previous stop is restored for the FULL held qty at the old level; if
    even the restore fails, the detail says to arm one manually."""
    from alpaca.trading.enums import TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
    from src.stock_screener.cockpit import trade

    Client, _Pos, _Order = _pos_fakes()

    class FailSellClient(Client):
        def submit_order(self, order_data=None):
            if isinstance(order_data, MarketOrderRequest):
                raise RuntimeError("rejected")
            return super().submit_order(order_data=order_data)

    c = FailSellClient([_Pos("HELD", 40)], {"HELD": [_Order("s1", "HELD", 95.0)]})
    orig = trade._connect_paper
    trade._connect_paper = lambda: (c, True)
    try:
        r = trade.submit_position_sell("HELD", 15)
    finally:
        trade._connect_paper = orig
    assert r["status"] == "failed" and "s1" in c.cancelled
    restored = [o for o in c.submitted if isinstance(o, StopOrderRequest)]
    assert len(restored) == 1 and int(float(restored[0].qty)) == 40
    assert float(restored[0].stop_price) == 95.0
    assert restored[0].time_in_force == TimeInForce.GTC
    assert "previous stop restored @ 95.00" in r["detail"]

    class FailAllClient(Client):
        def submit_order(self, order_data=None):
            raise RuntimeError("rejected")

    c2 = FailAllClient([_Pos("HELD", 40)], {"HELD": [_Order("s1", "HELD", 95.0)]})
    trade._connect_paper = lambda: (c2, True)
    try:
        r2 = trade.submit_position_sell("HELD", 15)
    finally:
        trade._connect_paper = orig
    assert r2["status"] == "failed" and "arm a stop manually" in r2["detail"]


def test_positions_page_renders():
    """The Positions page loads and renders with an offline (patched) fetch_positions — no
    network, no re-arm click (covered by the rearm unit test)."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_positions_page_renders (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch
    from src.stock_screener.cockpit import cache, journal_cache, trade

    offline = {
        "account": {"account_number": "PA00SZOE", "equity": 50000.0, "cash": 10000.0,
                    "using_dedicated": True, "positions_count": 1, "total_unrealized_pl": 300.0},
        # Deliberately NO template_criteria/df keys — the sell-pillar read must tolerate a
        # pre-§6.52 position dict end-to-end (all-unknown pillars, no raise).
        "positions": [{
            "symbol": "AAA", "qty": 10, "avg_entry": 100.0, "current_price": 130.0,
            "market_value": 1300.0, "cost_basis": 1000.0, "unrealized_pl": 300.0,
            "unrealized_plpc": 0.30, "lastday_price": 128.0, "current_stop": 120.0,
            "has_stop": True, "sma_50": 115.0, "last_close": 130.0, "volume_ratio": 1.1,
            "gain_pct": 0.30, "below_sma50": False,
            "advisories": ["Up 30% — consider selling part into strength."],
        }],
    }
    page = str(ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "2_Positions.py")
    journal_cache.cached_fills.clear()          # process-global cache; keep this run offline
    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(trade, "fetch_positions", return_value=offline), \
            patch.object(trade, "fetch_order_fills",
                         side_effect=trade.TradeUnavailable("offline")), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
        at = AppTest.from_file(page, default_timeout=60)
        at.run()
    assert not at.exception, f"positions page raised: {at.exception}"
    # The re-arm button only renders after the positions loop, so its presence proves the page
    # rendered end-to-end with the offline holding (and didn't st.stop() early).
    assert any("Re-arm" in str(getattr(b, "label", "")) for b in at.button), \
        "positions page did not render the re-arm control"


def test_positions_page_sell_flow():
    """The manual-sell UI end-to-end (offline): qty seeds to the full position, the ½ preset
    halves it, Sell opens the two-step confirm, Cancel clears it without submitting, and
    Confirm calls submit_position_sell with the FROZEN quantity and renders the result."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_positions_page_sell_flow (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch
    from src.stock_screener.cockpit import cache, journal_cache, trade

    offline = {
        "account": {"account_number": "PA00SZOE", "equity": 50000.0, "cash": 10000.0,
                    "using_dedicated": True, "positions_count": 1, "total_unrealized_pl": -50.0},
        "positions": [{
            "symbol": "AAA", "qty": 10, "avg_entry": 100.0, "current_price": 95.0,
            "market_value": 950.0, "cost_basis": 1000.0, "unrealized_pl": -50.0,
            "unrealized_plpc": -0.05, "lastday_price": 96.0, "current_stop": 92.0,
            "has_stop": True, "sma_50": 90.0, "last_close": 95.0, "volume_ratio": 1.0,
            "gain_pct": -0.05, "below_sma50": False, "next_earnings": "2026-07-27",
            "earnings_in": 7, "stage": "underwater",
            "advisories": ["⚠ Earnings in 7d with a loss — no cushion; exit or reduce "
                           "before the report."],
        }],
    }
    calls = []

    def _fake_sell(symbol, qty, remainder_stop=None):
        # remainder_stop rides the free-roll path; a plain manual sell passes None.
        calls.append((symbol, qty))
        return {"status": "submitted", "detail": "market SELL 5/10 sh (DAY); stop re-placed "
                "@ 92.00 for the remaining 5 sh", "symbol": symbol, "sold_qty": qty,
                "remaining": 5, "stop_price": 92.0, "account_number": "PA00SZOE",
                "equity": 50000.0}

    def _btn(at, key):
        hits = [b for b in at.button if b.key == key]
        assert hits, f"button {key} not rendered"
        return hits[0]

    page = str(ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "2_Positions.py")
    journal_cache.cached_fills.clear()          # process-global cache; keep this run offline
    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(trade, "fetch_positions", return_value=offline), \
            patch.object(trade, "fetch_order_fills",
                         side_effect=trade.TradeUnavailable("offline")), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"), \
            patch.object(trade, "submit_position_sell", side_effect=_fake_sell):
        at = AppTest.from_file(page, default_timeout=60)
        at.run()
        assert not at.exception, f"page raised: {at.exception}"
        qty = [n for n in at.number_input if n.key == "sellqty_AAA_1"]
        assert qty and qty[0].value == 10, "sell qty must seed to the full position"

        _btn(at, "sellq2_AAA_1").click().run()               # ½ preset -> 5
        qty = [n for n in at.number_input if n.key == "sellqty_AAA_1"]
        assert qty[0].value == 5, f"½ preset should set qty to 5, got {qty[0].value}"

        _btn(at, "sell_AAA_1").click().run()                 # step 1 -> confirm renders
        assert [b for b in at.button if b.key == "sellgo_AAA_1"], "confirm button missing"

        _btn(at, "sellno_AAA_1").click().run()               # cancel -> pending cleared
        assert not [b for b in at.button if b.key == "sellgo_AAA_1"]
        assert not calls, "cancel must not submit"

        _btn(at, "sell_AAA_1").click().run()                 # step 1 again
        _btn(at, "sellgo_AAA_1").click().run()               # step 2 -> submit
        assert not at.exception, f"page raised on confirm: {at.exception}"
    assert calls == [("AAA", 5)], f"confirm must sell the frozen qty, got {calls}"
    rendered = " ".join(str(getattr(m, "value", ""))
                        for m in list(at.markdown) + list(getattr(at, "caption", [])))
    assert "submitted" in rendered and "re-placed" in rendered, rendered


def test_positions_page_sell_pillars():
    """§6.52 composition on the page: journal open episode -> P1 entry date, watchlist ->
    frozen pivot, canned trigger report -> P3 SPY fallback; the legend renders and a
    laggard (day ~20, +1%) shows its per-row P1 warn detail. With the journal down every
    journal-fed pillar degrades to unknown and the page still renders."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_positions_page_sell_pillars (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch
    import pandas as pd
    from src.stock_screener.cockpit import cache, journal_cache, trade
    from src.stock_screener.cockpit.export import make_entry, save_watchlist
    from src.stock_screener.cockpit.triggers import save_trigger_report

    offline = _positions_offline()
    # Open episode entered ~20 trading days ago (relative to the REAL clock — the page
    # doesn't pin today): day_n >= 10 with +1% stays a cushion warn at any later date.
    entry_iso = (pd.Timestamp.now(tz="UTC") - pd.offsets.BDay(20)).isoformat()
    fills = {"account": offline["account"],
             "fills": [{"symbol": "AAA", "side": "buy", "qty": 10, "price": 100.0,
                        "time": entry_iso, "order_id": "1",
                        "client_order_id": "SEPAcockpit-AAA-1"}]}
    canned_report = {"schema": 1, "date": "2026-08-11", "generated_at": "2026-08-11T16:31:00-04:00",
                     "spy": {"phase": 2, "phase_name": "Stage 2 - Advancing", "trend": "Bullish"},
                     "names": [], "summary": {"n": 0}}

    page = str(ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "2_Positions.py")

    # 1) full composition: pivot from the watchlist, entry from the journal, SPY fallback
    journal_cache.cached_fills.clear()
    with tempfile.TemporaryDirectory() as _tmp:
        wl = Path(_tmp) / "watchlist.json"
        trg = Path(_tmp) / "triggers"
        save_watchlist(wl, [make_entry("AAA", 100.0, date_added="2026-07-01",
                                       pivot_source="judged")])
        save_trigger_report(canned_report, trg)
        with patch.object(trade, "fetch_positions", return_value=offline), \
                patch.object(trade, "fetch_order_fills", return_value=fills), \
                patch.object(cache, "WATCHLIST_JSON", wl), \
                patch.object(cache, "TRIGGERS_DIR", trg):
            at = AppTest.from_file(page, default_timeout=60)
            at.run()
    assert not at.exception, f"positions page raised: {at.exception}"
    rendered = _rendered_text(at)
    assert "Sell pillars" in rendered, "pillar legend missing"
    assert "cushion" in rendered and "sell into strength" in rendered, \
        f"laggard P1 warn detail missing: {rendered[-500:]}"

    # 2) journal down: every journal-fed pillar unknown, page alive, no flagged captions
    journal_cache.cached_fills.clear()
    with tempfile.TemporaryDirectory() as _tmp:
        with patch.object(trade, "fetch_positions", return_value=offline), \
                patch.object(trade, "fetch_order_fills",
                             side_effect=trade.TradeUnavailable("down")), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
            at = AppTest.from_file(page, default_timeout=60)
            at.run()
    assert not at.exception, f"positions page raised: {at.exception}"
    rendered = _rendered_text(at)
    assert "Sell pillars" in rendered
    assert "sell into strength" not in rendered, "no journal -> no laggard read"


def test_build_sell_plan_matrix():
    """§6.55 auto-sell planner: name-specific hard fails (P1/P4) plan a FULL exit on
    the first failing settled close; P2 (strict template, known one-day SMA noise
    flips) needs TWO consecutive failing closes fed by the prior plan's snapshot; P3
    (tape) and every warn are report-only notes; unknowns never trade; a zero-qty row
    snapshots but cannot order."""
    from src.stock_screener.cockpit import sells

    def pil(**kw):
        base = {k: {"status": "ok", "detail": ""} for k in ("P1", "P2", "P3", "P4")}
        for k, v in kw.items():
            base[k] = {"status": v, "detail": f"{k} {v}"}
        return base

    poss = [{"symbol": s, "qty": 10} for s in
            ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")] + [{"symbol": "GGG", "qty": 0}]
    pillars = {"AAA": pil(P1="fail"),          # immediate
               "BBB": pil(P4="fail"),          # immediate
               "CCC": pil(P2="fail"),          # first close -> streak only
               "DDD": pil(P3="fail"),          # report-only
               "EEE": pil(P1="warn", P4="warn"),
               "FFF": pil(P2="unknown"),
               "GGG": pil(P1="fail")}          # qty 0 -> no order
    plan = sells.build_sell_plan(poss, pillars, prior_plan=None, today="2026-08-18")
    by = {o["symbol"]: o for o in plan["orders"]}
    assert set(by) == {"AAA", "BBB"}, f"unexpected orders: {sorted(by)}"
    assert by["AAA"]["qty"] == 10 and by["AAA"]["status"] == "planned"
    assert any("P1 fail" in r for r in by["AAA"]["reasons"])
    assert any("CCC" in n and "streak" in n for n in plan["notes"])
    assert any("DDD" in n and "P3" in n for n in plan["notes"])
    assert plan["snapshot"]["CCC"]["P2"]["status"] == "fail"
    assert plan["snapshot"]["GGG"]["P1"]["status"] == "fail"
    assert plan["date"] == "2026-08-18"

    # Second consecutive P2 failing close, fed by the prior plan's snapshot -> order.
    plan2 = sells.build_sell_plan([{"symbol": "CCC", "qty": 7}],
                                  {"CCC": pil(P2="fail")},
                                  prior_plan=plan, today="2026-08-19")
    assert [o["symbol"] for o in plan2["orders"]] == ["CCC"]
    assert plan2["orders"][0]["qty"] == 7
    assert any("2nd consecutive" in r for r in plan2["orders"][0]["reasons"])


def test_sell_plan_persistence_and_veto():
    """§6.55 plan files: atomic dated save, newest-parseable load, ``before=`` excludes
    same/later dates (the evening planner must read YESTERDAY's snapshot, never its own
    same-day rerun), corrupt files skipped, and veto flips only still-planned orders."""
    import tempfile
    from src.stock_screener.cockpit import sells

    with tempfile.TemporaryDirectory() as tmp:
        p1 = {"date": "2026-08-18", "generated_at": "x", "orders": [
              {"symbol": "AAA", "qty": 10, "reasons": ["P1 fail: x"],
               "status": "planned", "detail": ""}],
              "snapshot": {}, "notes": [], "executed_at": None}
        p2 = {"date": "2026-08-19", "generated_at": "x", "orders": [],
              "snapshot": {"AAA": {"P2": {"status": "fail", "detail": ""}}},
              "notes": [], "executed_at": None}
        sells.save_sell_plan(p1, tmp)
        sells.save_sell_plan(p2, tmp)
        (Path(tmp) / "sell_plan_2026-08-20.json").write_text("{corrupt",
                                                             encoding="utf-8")
        assert sells.load_latest_sell_plan(tmp)["date"] == "2026-08-19"
        assert sells.load_latest_sell_plan(tmp, before="2026-08-19")["date"] == \
            "2026-08-18"
        assert sells.load_latest_sell_plan(tmp, before="2026-08-18") is None

        plan = sells.load_latest_sell_plan(tmp, before="2026-08-19")
        assert sells.veto_order(plan, "AAA") is True
        assert plan["orders"][0]["status"] == "vetoed"
        assert sells.veto_order(plan, "AAA") is False       # already vetoed
        assert sells.veto_order(plan, "ZZZ") is False       # no such order


def test_execute_sell_plan_gates_and_idempotency():
    """§6.55 morning executor: AUTOSELL gate (ships dark), stale-plan refusal (same-day
    and two-sessions-old both refused), veto respected, a no-longer-held name skipped
    (never shorted), qty clamped to CURRENT holdings, per-order failure recording, and
    a rerun resubmits nothing (failed stays failed for a human — blind retry after an
    ambiguous broker failure could double-sell)."""
    from src.stock_screener.cockpit import sells

    def mkplan():
        return {"date": "2026-08-18", "generated_at": "x", "orders": [
                {"symbol": "AAA", "qty": 10, "reasons": ["P1"], "status": "planned",
                 "detail": ""},
                {"symbol": "BBB", "qty": 5, "reasons": ["P4"], "status": "vetoed",
                 "detail": ""},
                {"symbol": "CCC", "qty": 5, "reasons": ["P4"], "status": "planned",
                 "detail": ""},
                {"symbol": "DDD", "qty": 8, "reasons": ["P1"], "status": "planned",
                 "detail": ""}],
                "snapshot": {}, "notes": [], "executed_at": None}

    calls = []

    def submit(sym, qty):
        calls.append((sym, qty))
        if sym == "DDD":
            return {"status": "failed", "detail": "boom"}
        return {"status": "submitted", "detail": "ok"}

    held = {"AAA": 6, "DDD": 8}              # AAA shrank; CCC gone; BBB vetoed
    s = sells.execute_sell_plan(mkplan(), submit=submit, held_by_symbol=held,
                                today="2026-08-19", enabled=False)
    assert s["status"] == "disabled" and not calls
    s = sells.execute_sell_plan(mkplan(), submit=submit, held_by_symbol=held,
                                today="2026-08-20", enabled=True)
    assert s["status"] == "stale" and not calls
    s = sells.execute_sell_plan(mkplan(), submit=submit, held_by_symbol=held,
                                today="2026-08-18", enabled=True)
    assert s["status"] == "stale" and not calls

    plan = mkplan()
    s = sells.execute_sell_plan(plan, submit=submit, held_by_symbol=held,
                                today="2026-08-19", enabled=True)
    assert calls == [("AAA", 6), ("DDD", 8)], f"unexpected submits: {calls}"
    assert s["status"] == "partial"
    assert s["submitted"] == ["AAA"] and s["failed"] == ["DDD"]
    assert s["vetoed"] == ["BBB"] and s["skipped"] == ["CCC"]
    by = {o["symbol"]: o for o in plan["orders"]}
    assert by["AAA"]["status"] == "submitted" and by["DDD"]["status"] == "failed"
    assert by["CCC"]["status"] == "skipped" and "no longer held" in by["CCC"]["detail"]
    assert plan["executed_at"]

    calls.clear()
    s2 = sells.execute_sell_plan(plan, submit=submit, held_by_symbol=held,
                                 today="2026-08-19", enabled=True)
    assert calls == [] and s2["submitted"] == []


def test_positions_page_sell_plan_veto():
    """§6.55 page surface: the evening plan renders with its reasons and the disarmed
    note (AUTOSELL unset), and the Veto button rewrites the plan file on disk
    (planned -> vetoed) so the morning executor will skip the order."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_positions_page_sell_plan_veto (AppTest unavailable: {e})")
        return
    import os
    import tempfile
    from unittest.mock import patch
    from src.stock_screener.cockpit import cache, journal_cache, sells, trade

    offline = _positions_offline()
    plan = {"date": "2026-08-18", "generated_at": "x",
            "orders": [{"symbol": "AAA", "qty": 10,
                        "reasons": ["P1 fail: day-0 close below the pivot"],
                        "status": "planned", "detail": ""}],
            "snapshot": {}, "notes": [], "executed_at": None}

    page = str(ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "2_Positions.py")
    journal_cache.cached_fills.clear()
    with tempfile.TemporaryDirectory() as _tmp:
        wl = Path(_tmp) / "watchlist.json"
        trg = Path(_tmp) / "triggers"
        sells.save_sell_plan(plan, trg)
        with patch.object(trade, "fetch_positions", return_value=offline), \
                patch.object(trade, "fetch_order_fills",
                             side_effect=trade.TradeUnavailable("down")), \
                patch.object(cache, "WATCHLIST_JSON", wl), \
                patch.object(cache, "TRIGGERS_DIR", trg), \
                patch.dict(os.environ, {"AUTOSELL": ""}):
            at = AppTest.from_file(page, default_timeout=60)
            at.run()
            assert not at.exception, f"positions page raised: {at.exception}"
            rendered = _rendered_text(at)
            assert "Planned auto-sells" in rendered, "plan section missing"
            assert "P1 fail" in rendered, "order reason missing"
            assert "not armed" in rendered, "disarmed note missing"
            btns = [b for b in at.button if b.key == "veto_AAA_2026-08-18"]
            assert btns, "veto button missing"
            btns[0].click()
            at.run()
            assert not at.exception, f"veto rerun raised: {at.exception}"
        saved = sells.load_latest_sell_plan(trg)
    assert saved["orders"][0]["status"] == "vetoed", saved["orders"][0]


def test_submit_position_sell_remainder_stop():
    """#19 free-roll mechanics: remainder_stop raises the remainder's stop under the
    ratchet (never lowers), places one when no prior stop existed, and the failed-sell
    restore path restores at the OLD level (the sell never happened, so a raised stop
    has no business being in force)."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from src.stock_screener.cockpit import trade

    class FakeClient:
        def __init__(self, fail_market=False):
            self.orders = []
            self.fail_market = fail_market

        def get_account(self):
            return SimpleNamespace(account_number="PA00TEST", equity=50000.0)

        def get_all_positions(self):
            return [SimpleNamespace(symbol="AAA", qty="10")]

        def submit_order(self, order_data):
            if self.fail_market and type(order_data).__name__ == "MarketOrderRequest":
                raise RuntimeError("rejected")
            self.orders.append(order_data)

    def run(*, fail_market=False, stops=(90.0,), **kw):
        client = FakeClient(fail_market=fail_market)
        existing = [SimpleNamespace(stop_price=s) for s in stops]
        with patch.object(trade, "_connect_paper", return_value=(client, True)), \
                patch.object(trade, "_open_sell_stops", return_value=existing), \
                patch.object(trade, "_cancel_orders", return_value=bool(existing)):
            res = trade.submit_position_sell("AAA", 5, **kw)
        placed = [o for o in client.orders
                  if type(o).__name__ == "StopOrderRequest"]
        return res, placed

    res, placed = run()                                    # baseline: old level kept
    assert res["status"] == "submitted" and res["stop_price"] == 90.0
    assert len(placed) == 1 and float(placed[0].stop_price) == 90.0
    assert int(placed[0].qty) == 5

    res, placed = run(remainder_stop=100.0)                # raise to breakeven
    assert res["stop_price"] == 100.0 and float(placed[0].stop_price) == 100.0
    assert "raised from 90.00" in res["detail"]

    res, placed = run(remainder_stop=85.0)                 # ratchet: never lower
    assert res["stop_price"] == 90.0 and float(placed[0].stop_price) == 90.0
    assert "raised" not in res["detail"]

    res, placed = run(stops=(), remainder_stop=100.0)      # no prior stop -> place new
    assert res["stop_price"] == 100.0 and float(placed[0].stop_price) == 100.0

    res, placed = run(fail_market=True, remainder_stop=100.0)   # restore at OLD level
    assert res["status"] == "failed"
    assert len(placed) == 1 and float(placed[0].stop_price) == 90.0
    assert int(placed[0].qty) == 10                        # full held qty re-protected


def test_positions_page_free_roll():
    """#19 page surface: the R column renders pivot-derived (no '~') for a watchlisted
    name, the free-roll button seeds a HALF-size pending sell carrying
    remainder_stop=avg_entry, the banner explains the breakeven ratchet, and confirm
    passes remainder_stop through to submit_position_sell."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_positions_page_free_roll (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch
    from src.stock_screener.cockpit import cache, journal_cache, trade
    from src.stock_screener.cockpit.export import make_entry, save_watchlist

    offline = _positions_offline(current_price=116.0, market_value=1160.0,
                                 unrealized_pl=160.0, unrealized_plpc=0.16,
                                 lastday_price=115.0, sma_50=105.0, last_close=116.0,
                                 gain_pct=0.16, stage="working")
    calls = {}

    def _fake_sell(symbol, qty, remainder_stop=None):
        calls["args"] = (symbol, qty, remainder_stop)
        return {"status": "submitted", "detail": "ok", "symbol": symbol,
                "sold_qty": qty, "remaining": 10 - qty,
                "stop_price": remainder_stop, "account_number": "PA00SZOE",
                "equity": 50000.0}

    page = str(ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "2_Positions.py")
    journal_cache.cached_fills.clear()
    with tempfile.TemporaryDirectory() as _tmp:
        wl = Path(_tmp) / "watchlist.json"
        save_watchlist(wl, [make_entry("AAA", 100.0, date_added="2026-07-01",
                                       pivot_source="judged")])
        with patch.object(trade, "fetch_positions", return_value=offline), \
                patch.object(trade, "fetch_order_fills",
                             side_effect=trade.TradeUnavailable("down")), \
                patch.object(trade, "submit_position_sell", _fake_sell), \
                patch.object(cache, "WATCHLIST_JSON", wl), \
                patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
            at = AppTest.from_file(page, default_timeout=60)
            at.run()
            assert not at.exception, f"positions page raised: {at.exception}"
            _df = at.dataframe[0].value
            assert "R" in _df.columns and _df["R"].iloc[0] == "2.1", \
                f"pivot-derived R expected '2.1', got {_df['R'].iloc[0]!r}"

            froll = [b for b in at.button
                     if str(b.key or "").startswith("froll_AAA")]
            assert froll, "free-roll button missing at >=2R with stop below entry"
            froll[0].click()
            at.run()
            assert not at.exception, f"page raised on free-roll: {at.exception}"
            rendered = " ".join(str(getattr(m, "value", ""))
                                for m in list(at.markdown)
                                + list(getattr(at, "caption", []))
                                + list(getattr(at, "warning", [])))
            assert "breakeven @ 100.00" in rendered, \
                f"free-roll banner missing: {rendered[-400:]}"

            conf = [b for b in at.button
                    if str(b.key or "").startswith("sellgo_AAA")]
            assert conf, "confirm button missing"
            conf[0].click()
            at.run()
            assert not at.exception, f"page raised on confirm: {at.exception}"
    assert calls.get("args") == ("AAA", 5, 100.0), \
        f"expected half-size sell with breakeven remainder_stop, got {calls.get('args')}"



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "positions"))
