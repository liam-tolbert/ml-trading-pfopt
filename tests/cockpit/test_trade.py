"""Cockpit tests — buy-plan sizing and the Alpaca submit path (stops, limits, OTO, exposure gate).

Runs standalone (`python tests/cockpit/test_trade.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_build_buy_plan_held_stop_only():
    """§6.39 (closes §6.3's build-time stop gap): a HELD name whose buy fails a sizing
    gate becomes a zero-share `stop_only` row — its stop still reaches submit's re-arm
    path — instead of silently vanishing. Un-held, the same names skip exactly as before;
    a pre-level skip (not in the scan) stays a skip even when held."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import build_buy_plan

    def _payload(price, pivot):
        idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=3)
        df = pd.DataFrame({"Open": price, "High": price, "Low": price,
                           "Close": price, "Volume": 1000}, index=idx)
        return {"df": df, "levels": {"pivot": pivot, "buy_zone": (pivot, pivot * 1.05),
                                     "stop": round(pivot * 0.925, 2)}}

    pl = {"BIG": _payload(500.0, 500.0),     # $300/name -> rounds to 0 shares
          "TIN": _payload(40.0, 40.0)}       # $45/name -> 1 share = $40, under the floor

    # un-held: both skip (behavior unchanged when `held` is omitted)
    plan0, skip0 = build_buy_plan(["BIG"], pl, mode="dollars", amount=300.0)
    assert not plan0 and skip0[0]["reason"] == "sizing rounds to < 1 share"
    plan1, skip1 = build_buy_plan(["TIN"], pl, mode="dollars", amount=45.0)
    assert not plan1 and "order minimum" in skip1[0]["reason"]

    # held: zero-share stop-only rows carrying the stop, nothing skipped
    held = {"BIG": 10, "TIN": 100}
    plan2, skip2 = build_buy_plan(["BIG"], pl, mode="dollars", amount=300.0, held=held)
    assert not skip2 and plan2[0]["stop_only"] is True and plan2[0]["shares"] == 0
    assert plan2[0]["stop_price"] == round(500.0 * 0.925, 2)
    assert plan2[0]["est_value"] == 0.0
    plan3, skip3 = build_buy_plan(["TIN"], pl, mode="dollars", amount=45.0, held=held)
    assert not skip3 and plan3[0]["stop_only"] is True

    # pre-level skips unchanged: a held name absent from the scan has no level to arm
    plan4, skip4 = build_buy_plan(["GONE"], pl, mode="dollars", amount=300.0,
                                  held={"GONE": 5})
    assert not plan4 and skip4[0]["reason"] == "not in the current scan"


def test_build_buy_plan_sizing_modes():
    """The paper-trade plan builder sizes each name by the chosen mode — % of equity,
    $ per name, an explicit share count, or risk-to-stop — flags extended names, and skips
    ones that round below 1 share, fall under the $50 floor (dollar modes only), lack equity
    for the %/risk modes, or aren't in the scan."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import build_buy_plan, MIN_TRADE_USD, MAX_ORDER_PCT

    def _payload(price, pivot, stop=None):
        idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=3)
        df = pd.DataFrame({"Open": price, "High": price, "Low": price,
                           "Close": price, "Volume": 1000}, index=idx)
        lv = {"pivot": pivot, "buy_zone": (pivot, pivot * 1.05),
              "stop": round(pivot * 0.925, 2) if stop is None else stop}
        return {"df": df, "levels": lv}

    pl = {"AAA": _payload(100.0, 100.0),        # in zone
          "BBB": _payload(110.0, 100.0)}        # extended (110 > 100*1.05)

    # % of portfolio: 5% of $100k equity = $5,000 per name -> floor($5,000 / price)
    plan, _ = build_buy_plan(["AAA", "BBB"], pl, mode="pct", amount=5.0, equity=100_000.0)
    by = {o["ticker"]: o for o in plan}
    assert by["AAA"]["shares"] == int(5000 / 100)              # 50
    assert by["BBB"]["shares"] == int(5000 / 110)              # 45
    assert by["AAA"]["extended"] is False and by["BBB"]["extended"] is True
    assert by["AAA"]["stop_price"] == round(100.0 * 0.925, 2)  # computed stop carried through

    # % mode with no equity available -> every name skipped with a clear reason
    p_noeq, s_noeq = build_buy_plan(["AAA"], pl, mode="pct", amount=5.0, equity=None)
    assert not p_noeq and "equity" in s_noeq[0]["reason"]

    # $ per name: floor($ / price)
    p_dol, _ = build_buy_plan(["AAA"], pl, mode="dollars", amount=1000.0)
    assert p_dol[0]["shares"] == int(1000 / 100)               # 10

    # # shares per name: exact count, and exempt from the $50 floor
    p_sh, _ = build_buy_plan(["AAA"], pl, mode="shares", amount=3)
    assert p_sh[0]["shares"] == 3
    cheap = {"CHEAP": _payload(10.0, 10.0)}
    p_one, _ = build_buy_plan(["CHEAP"], cheap, mode="shares", amount=1)   # $10 order OK
    assert p_one and p_one[0]["shares"] == 1
    # the same $10 notional IS skipped in a dollar-denominated mode (< $50 floor)
    p_tiny, s_tiny = build_buy_plan(["CHEAP"], cheap, mode="dollars", amount=10.0)
    assert not p_tiny and s_tiny[0]["ticker"] == "CHEAP"
    assert MIN_TRADE_USD == 50.0

    # not-in-scan is always skipped; an unknown mode is a hard error
    _, s_zzz = build_buy_plan(["ZZZ"], pl, mode="dollars", amount=1000.0)
    assert "scan" in s_zzz[0]["reason"]
    try:
        build_buy_plan(["AAA"], pl, mode="bogus", amount=1.0)
        raise AssertionError("expected ValueError for unknown mode")
    except ValueError:
        pass

    # --- risk mode: shares = (equity × risk%) / (price − stop), Minervini's sizer ----------
    # price 100, stop 90 (10% away = $10/sh risk); 0.5% of $100k = $500 budget -> 50 sh.
    # Notional $5,000 = 5% of equity, under the 10% cap -> not capped.
    risk_ok = {"AAA": _payload(100.0, 100.0, stop=90.0)}
    p_risk, _ = build_buy_plan(["AAA"], risk_ok, mode="risk", amount=0.5, equity=100_000.0)
    assert p_risk[0]["shares"] == int(500 / 10)                # 50
    assert p_risk[0]["capped"] is False
    # real dollar risk to the stop is ~0.5% of equity, the whole point of the mode
    assert abs(p_risk[0]["shares"] * (100.0 - 90.0) - 500.0) <= 100.0

    # cap clamp: 1% risk with a 7.5% stop wants ~13.3% of equity -> clamped to the 10% cap.
    risk_cap = {"AAA": _payload(100.0, 100.0)}                 # default stop 92.5 (7.5% away)
    p_cap, _ = build_buy_plan(["AAA"], risk_cap, mode="risk", amount=1.0, equity=100_000.0)
    assert p_cap[0]["capped"] is True
    assert p_cap[0]["shares"] == int(MAX_ORDER_PCT * 100_000 / 100)   # 100 (the cap)
    assert p_cap[0]["est_value"] <= MAX_ORDER_PCT * 100_000 + 1e-6

    # risk mode skips: no equity, no stop, and a stop not below price
    _, s_ne = build_buy_plan(["AAA"], risk_ok, mode="risk", amount=1.0, equity=None)
    assert "equity" in s_ne[0]["reason"]
    _, s_nostop = build_buy_plan(["AAA"], {"AAA": _payload(100.0, 100.0, stop=0.0)},
                                 mode="risk", amount=1.0, equity=100_000.0)
    assert "stop" in s_nostop[0]["reason"]
    _, s_above = build_buy_plan(["AAA"], {"AAA": _payload(100.0, 100.0, stop=105.0)},
                                mode="risk", amount=1.0, equity=100_000.0)
    assert "below price" in s_above[0]["reason"]

    # stop_price carry-through (folded in from the former test_build_buy_plan_attaches_stop):
    # a computed positive stop is carried (asserted above); a levels dict with no 'stop' key, or a
    # non-positive stop, yields stop_price None.
    _df = _payload(100.0, 100.0)["df"]
    edge = {"NOSTOP": {"df": _df, "levels": {"pivot": 100.0, "buy_zone": (100.0, 105.0)}},
            "ZERO": _payload(100.0, 100.0, stop=0.0)}
    p_edge, _ = build_buy_plan(["NOSTOP", "ZERO"], edge, mode="shares", amount=5)
    by_edge = {o["ticker"]: o for o in p_edge}
    assert by_edge["NOSTOP"]["stop_price"] is None            # no stop key -> None
    assert by_edge["ZERO"]["stop_price"] is None              # stop 0 -> None


def test_build_buy_plan_skips_stale_bars():
    """With ``max_bar_age_days`` set, a name whose freshest daily bar is older than the
    tolerance (relative to ``asof``) is skipped as stale instead of sized on days-old data;
    a name whose bar is current is planned normally. Omitting ``max_bar_age_days`` keeps the
    builder's old behavior (no time check) so existing callers/tests are unaffected."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import build_buy_plan, STALE_PLAN_BARS

    asof = pd.Timestamp("2026-07-15")

    def mk(last_ts):
        idx = pd.bdate_range(end=pd.Timestamp(last_ts), periods=3)
        df = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                           "Close": 100.0, "Volume": 1000}, index=idx)
        return {"df": df, "levels": {"pivot": 100.0, "buy_zone": (100.0, 105.0), "stop": 92.0}}

    fresh = mk(asof)                                        # last bar == asof -> 0 days old
    stale = mk(pd.bdate_range(end=asof, periods=11)[0])     # 10 trading days back -> stale
    edge = mk(pd.bdate_range(end=asof, periods=STALE_PLAN_BARS + 1)[0])  # exactly tolerated

    plan, skipped = build_buy_plan(
        ["F", "S", "E"], {"F": fresh, "S": stale, "E": edge},
        mode="shares", amount=5, asof=asof, max_bar_age_days=STALE_PLAN_BARS)
    planned = {o["ticker"] for o in plan}
    assert "F" in planned, "current-bar name must be planned"
    assert "E" in planned, "a bar exactly at the tolerance must NOT be skipped"
    assert "S" not in planned, "a clearly-stale name must be skipped"
    _s = {s["ticker"]: s["reason"] for s in skipped}
    assert "S" in _s and "stale" in _s["S"].lower(), _s

    # Back-compat: without max_bar_age_days the stale name is planned (no time check).
    p2, s2 = build_buy_plan(["S"], {"S": stale}, mode="shares", amount=5)
    assert p2 and p2[0]["ticker"] == "S" and not s2


def test_build_buy_plan_uses_frozen_pivot():
    """A watchlist entry's FROZEN judged_pivot (the level its trigger fired on) overrides the
    drifted scan pivot: the buy zone / extended flag / default stop / risk sizing all key off the
    frozen level, not the current payload's levels. Names absent from ``pivots`` are unchanged."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import (
        build_buy_plan, DEFAULT_STOP_FROM_PIVOT, MAX_STOP_FROM_PIVOT)

    def _pl(price, scan_pivot, stop=None):
        idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=3)
        df = pd.DataFrame({"Open": price, "High": price, "Low": price,
                           "Close": price, "Volume": 1000}, index=idx)
        lv = {"pivot": scan_pivot, "buy_zone": (scan_pivot, scan_pivot * 1.05)}
        if stop is not None:
            lv["stop"] = stop
        return {"df": df, "levels": lv}

    # scan pivot drifted DOWN to 95 (price sits above it); the frozen trigger level is 120, and
    # the payload carries no engine stop -> the frozen-pivot default 7.5%-below stop applies,
    # floored at 10% below (here 111.0 = 120×0.925, above the 108.0 floor).
    pl = {"AAA": _pl(100.0, 95.0)}
    expect_stop = round(max(120.0 * (1.0 - DEFAULT_STOP_FROM_PIVOT),
                            120.0 * (1.0 - MAX_STOP_FROM_PIVOT)), 2)   # 111.0

    frozen, _ = build_buy_plan(["AAA"], pl, mode="shares", amount=5, pivots={"AAA": 120.0})
    o = frozen[0]
    assert o["pivot"] == 120.0 and o["pivot_frozen"] is True
    assert o["stop_price"] == expect_stop                  # stop off the frozen pivot, not 95
    assert o["extended"] is False                          # 100 < 120×1.05, unlike 100 > 95×1.05

    # same name WITHOUT the frozen pivot: scan pivot 95 -> extended (100 > 99.75), no stop.
    plain, _ = build_buy_plan(["AAA"], pl, mode="shares", amount=5)
    assert plain[0]["pivot"] == 95.0 and plain[0]["pivot_frozen"] is False
    assert plain[0]["extended"] is True and plain[0]["stop_price"] is None

    # price above the frozen buy zone -> extended flags off the FROZEN pivot (126 = 120×1.05).
    ext, _ = build_buy_plan(["AAA"], {"AAA": _pl(130.0, 95.0)},
                            mode="shares", amount=5, pivots={"AAA": 120.0})
    assert ext[0]["extended"] is True

    # risk sizing keys off the frozen-pivot stop: the payload has NO engine stop, so WITHOUT a
    # frozen pivot the risk mode can't size (skipped 'no stop'); WITH it, shares = budget/(price−stop).
    risk_pl = {"AAA": _pl(118.0, 95.0)}
    _, s_norisk = build_buy_plan(["AAA"], risk_pl, mode="risk", amount=0.5, equity=100_000.0)
    assert s_norisk and "stop" in s_norisk[0]["reason"]    # no stop to risk-size against
    p_risk, _ = build_buy_plan(["AAA"], risk_pl, mode="risk", amount=0.5,
                               equity=100_000.0, pivots={"AAA": 120.0})
    assert p_risk[0]["stop_price"] == expect_stop          # 111.0
    assert p_risk[0]["shares"] == int((100_000.0 * 0.5 / 100.0) / (118.0 - expect_stop))
    assert p_risk[0]["capped"] is False

    # a tighter engine stop that sits BELOW the frozen pivot and within 10% is kept as-is.
    kept, _ = build_buy_plan(["AAA"], {"AAA": _pl(118.0, 95.0, stop=115.0)},
                             mode="shares", amount=5, pivots={"AAA": 120.0})
    assert kept[0]["stop_price"] == 115.0                  # 115 < 120 and within 10% -> kept


def test_freshen_prices_overlays_latest_bars():
    """freshen_prices re-pulls the watchlist names' latest bars (incremental top-up) and
    overlays them onto the payload frames, carrying levels/earnings through; a name the
    refresh can't reach keeps its original frame, and non-scan tickers are dropped."""
    import pandas as pd
    from src.stock_screener.cockpit import trade, data_feed

    def _old(price):
        idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=3)
        return pd.DataFrame({"Open": price, "High": price, "Low": price,
                             "Close": price, "Volume": 1}, index=idx)

    payloads = {
        "AAA": {"df": _old(10.0), "levels": {"pivot": 10.0}, "earnings_in": 5},
        "BBB": {"df": _old(20.0), "levels": {"pivot": 20.0}},
    }

    captured = {}

    def fake_gmp(syms, **kw):
        captured["syms"], captured["max_age_days"] = list(syms), kw.get("max_age_days")
        idx = pd.bdate_range(end=pd.Timestamp("2026-07-15"), periods=4)
        # Fresh data for AAA only; BBB absent -> caller must fall back to its old frame.
        return {"AAA": pd.DataFrame({"Open": 12, "High": 12, "Low": 12,
                                     "Close": [12, 13, 14, 15.5], "Volume": 1}, index=idx)}

    _orig = data_feed.get_many_prices
    data_feed.get_many_prices = fake_gmp
    try:
        out = trade.freshen_prices(["AAA", "BBB", "ZZZ"], payloads)
    finally:
        data_feed.get_many_prices = _orig

    assert set(out) == {"AAA", "BBB"}, "ZZZ isn't in payloads -> dropped"
    assert captured["max_age_days"] == 0.0, "must use the cheap incremental top-up path"
    assert "ZZZ" not in captured["syms"], "only refresh names present in the scan"
    assert float(out["AAA"]["df"]["Close"].iloc[-1]) == 15.5, "AAA price refreshed"
    assert out["AAA"]["levels"] == {"pivot": 10.0} and out["AAA"]["earnings_in"] == 5, \
        "levels/earnings carried through untouched"
    assert out["BBB"]["df"] is payloads["BBB"]["df"], "unrefreshable name keeps its old frame"


def test_stop_is_valid():
    """A protective sell-stop is valid only strictly below the reference price."""
    from src.stock_screener.cockpit.trade import stop_is_valid
    assert stop_is_valid(92.0, 100.0) is True
    for bad in [(100.0, 100.0), (101.0, 100.0), (0.0, 100.0), (None, 100.0), (50.0, None)]:
        assert stop_is_valid(*bad) is False, bad


def test_minervini_key_envs_single_spelling():
    """Issue 8: the dedicated Minervini paper keys use the SINGLE canonical spelling .env actually
    holds (ALPACA_API_KEY_MINERVINI / ALPACA_API_KEY_SECRET_MINERVINI) — no either/or logic, and
    the comment/HANDOFF are corrected to match. `_first_env` resolves the set value; an unset key
    returns None so `_connect_paper` falls back to the shared pair."""
    from unittest.mock import patch
    from src.stock_screener.cockpit.trade import (
        _first_env, MINERVINI_KEY_ENVS, MINERVINI_SECRET_ENVS)

    assert MINERVINI_KEY_ENVS == "ALPACA_API_KEY_MINERVINI"
    assert MINERVINI_SECRET_ENVS == "ALPACA_API_KEY_SECRET_MINERVINI"

    with patch.dict("os.environ", {"ALPACA_API_KEY_MINERVINI": "k-live",
                                   "ALPACA_API_KEY_SECRET_MINERVINI": "s-live"}, clear=True):
        assert _first_env(MINERVINI_KEY_ENVS) == "k-live"
        assert _first_env(MINERVINI_SECRET_ENVS) == "s-live"

    with patch.dict("os.environ", {}, clear=True):          # unset -> None -> shared-pair fallback
        assert _first_env(MINERVINI_KEY_ENVS) is None


def test_submit_buy_plan_stop_logic():
    """submit_buy_plan against a fake Alpaca client: an already-held name becomes a GTC
    stop-only order that REPLACES its open stop; a fresh name becomes an OTO buy+stop;
    attach_stop=False yields a naked buy (and skips held names); an invalid stop (>= price)
    is skipped, no order. Cases D/E cover Minervini's never-lower-a-stop ratchet: an existing
    HIGHER stop is kept (no order), a LOWER one is replaced upward."""
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
    from alpaca.trading.enums import OrderSide, OrderClass, OrderType, TimeInForce

    FakeClient, _Order = _submit_fakes()
    _entry, _run = _submit_entry, (lambda plan, attach, fake: _run_submit(plan, fake, attach))

    # --- A: held name (40 sh, open stop of UNKNOWN price) + a fresh name, attach on -------
    # No readable stop_price -> can't ratchet -> replace with the new GTC stop.
    fa = FakeClient(positions={"HELD": "40"},
                    open_orders=[_Order("old-1", "HELD", OrderType.STOP)])
    outA = {r["ticker"]: r for r in
            _run([_entry("HELD", 10, 100.0, 95.0), _entry("NEW", 5, 50.0, 46.0)], True, fa)["results"]}
    assert outA["HELD"]["status"] == "stop_only"
    assert "old-1" in fa.cancelled                       # replaced the existing stop
    sreq = [r for r in fa.submitted if isinstance(r, StopOrderRequest)][0]
    assert sreq.symbol == "HELD" and int(sreq.qty) == 40 and sreq.side == OrderSide.SELL
    assert float(sreq.stop_price) == 95.0                # full held qty, at the edited stop
    assert sreq.time_in_force == TimeInForce.GTC         # persistent, not DAY
    assert outA["NEW"]["status"] == "submitted"
    mreq = [r for r in fa.submitted if isinstance(r, MarketOrderRequest)][0]
    assert mreq.order_class == OrderClass.OTO and mreq.side == OrderSide.BUY
    assert int(mreq.qty) == 5 and float(mreq.stop_loss.stop_price) == 46.0
    # §6.38: the OTO must be GTC end-to-end — a DAY parent made the stop leg a DAY order
    # that EXPIRED at that day's close (PEBK 2026-08-04: bought 15:58, stop dead 16:00).
    assert mreq.time_in_force == TimeInForce.GTC

    # --- B: attach OFF -> naked buy for a fresh name; held name skipped -------------------
    fb = FakeClient()
    outB = _run([_entry("NEW", 5, 50.0, 46.0)], False, fb)["results"][0]
    assert outB["status"] == "submitted"
    mb = [r for r in fb.submitted if isinstance(r, MarketOrderRequest)][0]
    assert mb.order_class is None and mb.client_order_id.startswith("SEPAcockpit-")

    fb2 = FakeClient(positions={"HELD": "40"})
    outB2 = _run([_entry("HELD", 10, 100.0, 95.0)], False, fb2)["results"][0]
    assert outB2["status"] == "skipped" and not fb2.submitted

    # --- C: invalid stop (>= price), fresh name, attach on -> skipped, nothing submitted --
    fc = FakeClient()
    outC = _run([_entry("NEW", 5, 50.0, 55.0)], True, fc)["results"][0]
    assert outC["status"] == "skipped" and not fc.submitted

    # --- D: held name with an existing HIGHER stop -> ratchet HOLDS (kept, no order) -------
    fd = FakeClient(positions={"HELD": "40"},
                    open_orders=[_Order("hi-1", "HELD", OrderType.STOP, stop_price=98.0)])
    outD = _run([_entry("HELD", 10, 100.0, 95.0)], True, fd)["results"][0]
    assert outD["status"] == "stop_kept"
    assert outD["stop_price"] == 98.0                    # kept the higher existing stop
    assert not fd.submitted and not fd.cancelled         # nothing placed, nothing cancelled

    # --- E: held name with an existing LOWER stop -> RAISE (cancel old, place GTC at new) --
    fe = FakeClient(positions={"HELD": "40"},
                    open_orders=[_Order("lo-1", "HELD", OrderType.STOP, stop_price=90.0)])
    outE = _run([_entry("HELD", 10, 100.0, 95.0)], True, fe)["results"][0]
    assert outE["status"] == "stop_only" and outE["stop_price"] == 95.0
    assert "lo-1" in fe.cancelled                        # replaced the lower stop
    ereq = [r for r in fe.submitted if isinstance(r, StopOrderRequest)][0]
    assert float(ereq.stop_price) == 95.0 and ereq.time_in_force == TimeInForce.GTC

    # --- F: held name, existing stop EQUAL to the new stop -> kept (no churn) --------------
    ff = FakeClient(positions={"HELD": "40"},
                    open_orders=[_Order("eq-1", "HELD", OrderType.STOP, stop_price=95.0)])
    outF = _run([_entry("HELD", 10, 100.0, 95.0)], True, ff)["results"][0]
    assert outF["status"] == "stop_kept" and not ff.submitted and not ff.cancelled


def test_build_buy_plan_limit_orders():
    """order_type="limit": each entry carries limit_price = its buy-zone TOP (the no-chase
    cap; frozen 📌 pivot preferred), and sizing / risk-per-share / est_value / the 10% cap
    all use the LIMIT as the worst-case-fill basis. Market mode stays byte-identical with
    limit_price None; an unknown order_type is a hard error."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import build_buy_plan, MAX_ORDER_PCT

    def _payload(price, pivot, stop=None):
        idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=3)
        df = pd.DataFrame({"Open": price, "High": price, "Low": price,
                           "Close": price, "Volume": 1000}, index=idx)
        lv = {"pivot": pivot, "buy_zone": (pivot, pivot * 1.05),
              "stop": round(pivot * 0.925, 2) if stop is None else stop}
        return {"df": df, "levels": lv}

    pl = {"AAA": _payload(100.0, 100.0, stop=90.0)}          # zone top = 105

    # dollars: $1,000 at limit 105 -> 9 sh (a market basis would give 10); est = 9 × 105
    p_dol, _ = build_buy_plan(["AAA"], pl, mode="dollars", amount=1000.0, order_type="limit")
    o = p_dol[0]
    assert o["limit_price"] == 105.0
    assert o["shares"] == int(1000 / 105.0)                  # 9
    assert o["est_value"] == round(9 * 105.0, 2)
    assert o["price"] == 100.0                               # last close still reported

    # market mode: unchanged sizing, limit_price None
    p_mkt, _ = build_buy_plan(["AAA"], pl, mode="dollars", amount=1000.0)
    assert p_mkt[0]["shares"] == 10 and p_mkt[0]["limit_price"] is None

    # risk: $500 budget / (limit 105 − stop 90) = 33 sh (a market basis: 500/10 = 50)
    p_risk, _ = build_buy_plan(["AAA"], pl, mode="risk", amount=0.5, equity=100_000.0,
                               order_type="limit")
    assert p_risk[0]["shares"] == int(500 / 15.0)            # 33

    # stop 106 ≥ price 100 -> the R2-3 broken-base gate fires first (it outranks the
    # risk gate: a stop above the market is wrong in EVERY sizing mode)
    _, s_bad = build_buy_plan(["AAA"], {"AAA": _payload(100.0, 100.0, stop=106.0)},
                              mode="risk", amount=0.5, equity=100_000.0, order_type="limit")
    assert "current price" in s_bad[0]["reason"]
    # the risk gate's own limit check is still reachable for an EXTENDED name: price 110
    # above the zone, limit 105 < price, stop 106 between them -> "stop not below limit"
    _, s_ext = build_buy_plan(["EEE"], {"EEE": _payload(110.0, 100.0, stop=106.0)},
                              mode="risk", amount=0.5, equity=100_000.0, order_type="limit")
    assert "limit" in s_ext[0]["reason"]

    # frozen pivot overrides: limit = frozen × 1.05, not the scan zone. (Pivot must sit
    # NEAR the price — a far-above frozen pivot now trips the R2-3 broken-base skip,
    # since its derived stop would sit above the market.)
    p_fz, _ = build_buy_plan(["AAA"], pl, mode="shares", amount=5, order_type="limit",
                             pivots={"AAA": 102.0})
    assert p_fz[0]["limit_price"] == round(102.0 * 1.05, 2)
    # ...and the far-above frozen pivot IS skipped by the R2-3 gate (stop 91.8 < price
    # would pass, but pivot 200 -> stop 180 >= price 100 -> broken base)
    _, s_far = build_buy_plan(["AAA"], pl, mode="shares", amount=5, order_type="limit",
                              pivots={"AAA": 200.0})
    assert s_far and "current price" in s_far[0]["reason"]

    # extended name (price above the zone): planned + flagged, limit sits BELOW the price
    ext = {"BBB": _payload(110.0, 100.0, stop=90.0)}
    p_ext, _ = build_buy_plan(["BBB"], ext, mode="shares", amount=5, order_type="limit")
    assert p_ext[0]["extended"] is True and p_ext[0]["limit_price"] == 105.0 < 110.0

    # no pivot/zone -> the limit falls back to the last close (a marketable cap)
    nz = {"CCC": {"df": _payload(100.0, 100.0)["df"], "levels": {"stop": 90.0}}}
    p_nz, _ = build_buy_plan(["CCC"], nz, mode="shares", amount=5, order_type="limit")
    assert p_nz[0]["limit_price"] == 100.0

    # R2-3: broken-down name — stop (92.5) at/above the CURRENT price (88) means the
    # zone-top limit is marketable and the stop would arm above the market. Build skips
    # it in limit mode, in EVERY sizing mode (no stop-gate existed outside risk mode).
    broke = {"DDD": _payload(88.0, 100.0, stop=92.5)}
    for _mode, _amt in (("shares", 5), ("dollars", 1000.0)):
        _, s_brk = build_buy_plan(["DDD"], broke, mode=_mode, amount=_amt,
                                  order_type="limit")
        assert s_brk and "current price" in s_brk[0]["reason"], (_mode, s_brk)
    # ...and market mode is untouched at build (submit's stop_is_valid catches it there)
    p_mkt_brk, _ = build_buy_plan(["DDD"], broke, mode="shares", amount=5)
    assert p_mkt_brk and p_mkt_brk[0]["ticker"] == "DDD"

    # the 10% single-order cap clamps on the limit basis too
    p_cap, _ = build_buy_plan(["AAA"], pl, mode="risk", amount=5.0, equity=100_000.0,
                              order_type="limit")
    assert p_cap[0]["capped"] is True
    assert p_cap[0]["shares"] == int(MAX_ORDER_PCT * 100_000 / 105.0)   # 95

    try:
        build_buy_plan(["AAA"], pl, mode="shares", amount=5, order_type="bogus")
        raise AssertionError("expected ValueError for unknown order_type")
    except ValueError:
        pass


def test_submit_buy_plan_rearm_only_never_buys():
    """Review 2026-08-09 HIGH: a plan row shown as 'already held — stop re-arm only, no
    buy' (rearm_only, stamped from BUILD-time holdings) must never become a buy when the
    position closes between Build and Submit (its GTC stop firing). Live-held rearm rows
    keep working; zero-share stop_only rows are skipped instead of reaching the API as
    qty-0 orders; ordinary buy rows alongside are unaffected."""
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
    from alpaca.trading.enums import OrderType

    FakeClient, _Order = _submit_fakes()
    _entry, _run = _submit_entry, _run_submit

    # --- A: held at build, position CLOSED before submit -> skipped, NEVER bought -----
    fa = FakeClient()                                    # nothing held anymore
    outA = _run([_entry("GONE", 50, 100.0, 95.0, rearm_only=True)], fa)["results"][0]
    assert outA["status"] == "skipped" and not fa.submitted
    assert "position closed" in outA["detail"]

    # --- B: rearm_only but STILL held -> normal stop re-arm path, no buy --------------
    fb = FakeClient(positions={"HELD": "40"},
                    open_orders=[_Order("lo-1", "HELD", OrderType.STOP, stop_price=90.0)])
    outB = _run([_entry("HELD", 10, 100.0, 95.0, rearm_only=True)], fb)["results"][0]
    assert outB["status"] == "stop_only"                 # raised the stop, sent no buy
    assert all(isinstance(r, StopOrderRequest) for r in fb.submitted)

    # --- C: zero-share stop_only row, position closed -> skipped, no qty-0 order ------
    fc = FakeClient()
    outC = _run([_entry("ZERO", 0, 100.0, 95.0, stop_only=True)], fc)["results"][0]
    assert outC["status"] == "skipped" and not fc.submitted

    # --- D: an ordinary buy row alongside is unaffected by the guard ------------------
    fd = FakeClient()
    outD = {r["ticker"]: r for r in _run(
        [_entry("GONE", 50, 100.0, 95.0, rearm_only=True),
         _entry("NEW", 5, 50.0, 46.0)], fd)["results"]}
    assert outD["GONE"]["status"] == "skipped"
    assert outD["NEW"]["status"] == "submitted"
    assert len([r for r in fd.submitted if isinstance(r, MarketOrderRequest)]) == 1


def test_submit_buy_plan_limit_orders():
    """An entry carrying limit_price becomes a LIMIT buy: a GTC OTO + stop leg when attach
    is on (the §6.38 end-to-end-GTC shape — the stop arms whenever the fill happens, even
    days later), a naked DAY limit when off. Stop validity checks against the LIMIT (the
    worst-case fill), not the last close; a present-but-invalid limit is SKIPPED rather
    than silently downgraded to a market order; entries without the key keep the market
    path untouched."""
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce

    FakeClient, _ = _submit_fakes()
    _entry, _run = _submit_entry, (lambda plan, attach, fake: _run_submit(plan, fake, attach))

    # --- A: attach on -> GTC OTO LIMIT buy with the stop leg, SEPAoto- tag ----------------
    fa = FakeClient()
    outA = _run([_entry("NEW", 5, 100.0, 95.0, limit=105.0)], True, fa)["results"][0]
    assert outA["status"] == "submitted"
    lr = [r for r in fa.submitted if isinstance(r, LimitOrderRequest)][0]
    assert float(lr.limit_price) == 105.0 and lr.side == OrderSide.BUY and int(lr.qty) == 5
    assert lr.order_class == OrderClass.OTO and lr.time_in_force == TimeInForce.GTC
    assert float(lr.stop_loss.stop_price) == 95.0
    assert lr.client_order_id.startswith("SEPAoto-")

    # --- B (R2-3): stop validity is vs min(limit, price) — the worst fill of a limit BUY
    #        is ~the current price when the limit is marketable. A stop BETWEEN price and
    #        limit (102 with price 100 / limit 105) would arm above the market -> skipped;
    #        a stop AT the limit is likewise skipped; nothing is sent either way. --------
    fb = FakeClient()
    outB = _run([_entry("NEW", 5, 100.0, 102.0, limit=105.0)], True, fb)["results"][0]
    assert outB["status"] == "skipped" and not fb.submitted
    fb2 = FakeClient()
    outB2 = _run([_entry("NEW", 5, 100.0, 105.0, limit=105.0)], True, fb2)["results"][0]
    assert outB2["status"] == "skipped" and not fb2.submitted

    # --- B3 (R2-3): the review's exact scenario — broken-down name, stop 92.5 ABOVE the
    #        88 price but below the 105 limit -> skipped, no instant stop-out order -----
    fb3 = FakeClient()
    outB3 = _run([_entry("BRK", 5, 88.0, 92.5, limit=105.0)], True, fb3)["results"][0]
    assert outB3["status"] == "skipped" and not fb3.submitted

    # --- B4 (R2-9): an upward-EDITED limit re-enters the 10% cap — est_value is stale
    #        build-time (under the cap) but shares x limit exceeds it -> skipped --------
    fb4 = FakeClient()
    e4 = _entry("CAP", 95, 100.0, 95.0, limit=150.0)     # 95 x 150 = $14,250 worst case
    e4["est_value"] = 9500.0                             # stale, under the $10k cap
    outB4 = _run([e4], True, fb4)["results"][0]
    assert outB4["status"] == "skipped" and "10%" in outB4["detail"] and not fb4.submitted

    # --- C: attach off -> naked DAY limit, SEPAcockpit- tag -------------------------------
    fc = FakeClient()
    outC = _run([_entry("NEW", 5, 100.0, 95.0, limit=105.0)], False, fc)["results"][0]
    assert outC["status"] == "submitted"
    lc = [r for r in fc.submitted if isinstance(r, LimitOrderRequest)][0]
    assert lc.order_class is None and lc.time_in_force == TimeInForce.DAY
    assert lc.client_order_id.startswith("SEPAcockpit-")

    # --- D: a present-but-invalid limit (0) is skipped, never a market fallback -----------
    fd = FakeClient()
    outD = _run([_entry("NEW", 5, 100.0, 95.0, limit=0.0)], True, fd)["results"][0]
    assert outD["status"] == "skipped" and not fd.submitted and "limit" in outD["detail"]

    # --- E: no limit key at all -> the market path, untouched -----------------------------
    fe = FakeClient()
    assert _run([_entry("NEW", 5, 100.0, 95.0)], True, fe)["results"][0]["status"] == "submitted"
    assert isinstance(fe.submitted[0], MarketOrderRequest)


def test_cancel_pending_buys_cockpit_only():
    """cancel_pending_buys cancels ONLY open SEPA-tagged BUY orders — never sells/stops,
    never other tools' buys — and one failed cancel doesn't abort the rest (R2-4c)."""
    from src.stock_screener.cockpit import trade
    from alpaca.trading.enums import OrderSide

    FakeClient, _Order = _submit_fakes()

    def _run_cancel(fake):
        orig = trade._connect_paper
        trade._connect_paper = lambda: (fake, True)
        try:
            return trade.cancel_pending_buys()
        finally:
            trade._connect_paper = orig

    fa = FakeClient(open_orders=[
        _Order("b1", "LIMA", side=OrderSide.BUY, coid="SEPAoto-LIMA-1"),
        _Order("b2", "MKT", side=OrderSide.BUY, coid="SEPAcockpit-MKT-2"),
        _Order("b3", "OTHER", side=OrderSide.BUY, coid="someBroker-OTHER-3"),
        _Order("s1", "HELD", stop_price=95.0),               # SELL stop — untouchable
    ])
    out = _run_cancel(fa)
    assert {c["ticker"] for c in out["cancelled"]} == {"LIMA", "MKT"}
    assert not out["errors"]
    assert set(fa.cancelled) == {"b1", "b2"}                 # never s1 (stop) / b3 (foreign)

    # one stuck cancel -> recorded as an error, the rest still cancel
    fb = FakeClient(open_orders=[
        _Order("b1", "STUCK", side=OrderSide.BUY, coid="SEPAoto-STUCK-1"),
        _Order("b2", "OK", side=OrderSide.BUY, coid="SEPAoto-OK-2"),
    ])
    _orig = fb.cancel_order_by_id

    def _flaky(oid):
        if str(oid) == "b1":
            raise RuntimeError("stuck order")
        _orig(oid)

    fb.cancel_order_by_id = _flaky
    out2 = _run_cancel(fb)
    assert [c["ticker"] for c in out2["cancelled"]] == ["OK"]
    assert out2["errors"] and out2["errors"][0]["ticker"] == "STUCK"
    assert "stuck" in out2["errors"][0]["error"]


def test_submit_buy_plan_skips_pending_cockpit_buy():
    """A cockpit BUY submitted after the close sits QUEUED (no position yet) until the next
    open. A re-submit must not place a second BUY: submit_buy_plan queries open BUY orders and
    skips a not-held ticker that already has a pending cockpit buy (client_order_id 'SEPA…').
    A pending BUY from some OTHER tool (no SEPA prefix) does NOT block, and a ticker with no
    pending order is bought normally."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide

    FakeClient, _Order = _submit_fakes()
    _entry, _run = _submit_entry, _run_submit

    # QUEUED covers a cockpit SEPAoto buy; FREE has an unrelated (non-SEPA) buy that must NOT
    # block; NEW has nothing pending and should submit.
    fake = FakeClient(open_orders=[
        _Order("q1", "QUEUED", side=OrderSide.BUY, coid="SEPAoto-QUEUED-123"),
        _Order("q2", "FREE", side=OrderSide.BUY, coid="someBroker-FREE-1"),
    ])
    out = {r["ticker"]: r for r in _run(
        [_entry("QUEUED", 10, 50.0, 46.0),
         _entry("FREE", 10, 50.0, 46.0),
         _entry("NEW", 10, 50.0, 46.0)], fake)["results"]}

    assert out["QUEUED"]["status"] == "skipped" and "queued" in out["QUEUED"]["detail"]
    assert out["FREE"]["status"] == "submitted"       # a non-cockpit buy doesn't block
    assert out["NEW"]["status"] == "submitted"
    bought = {r.symbol for r in fake.submitted if isinstance(r, MarketOrderRequest)}
    assert bought == {"FREE", "NEW"} and "QUEUED" not in bought


def test_rearm_gtc_stop_ratchet():
    """rearm_stops drives the SHARED _rearm_gtc_stop helper: an existing higher stop is kept
    (no order), a lower one is raised (cancel old + GTC at the new level), an equal one is kept,
    a first-arm places a GTC stop, and a not-held ticker is skipped. Proves the extracted helper
    preserves the ratchet semantics for the new caller."""
    from src.stock_screener.cockpit import trade
    from alpaca.trading.requests import StopOrderRequest
    from alpaca.trading.enums import TimeInForce
    Client, _Pos, _Order = _pos_fakes()

    def run(client):
        orig = trade._connect_paper
        trade._connect_paper = lambda: (client, True)
        try:
            return trade.rearm_stops([
                {"ticker": "HELD", "stop_price": 95.0, "price": 100.0},
                {"ticker": "NOPE", "stop_price": 40.0, "price": 50.0},   # not in the account
            ])
        finally:
            trade._connect_paper = orig

    # existing HIGHER stop (98) -> kept; NOPE not held -> skipped
    c1 = Client([_Pos("HELD", 40)], {"HELD": [_Order("hi", "HELD", 98.0)]})
    r1 = {x["ticker"]: x for x in run(c1)["results"]}
    assert r1["HELD"]["status"] == "stop_kept" and r1["HELD"]["stop_price"] == 98.0
    assert not c1.submitted and not c1.cancelled
    assert r1["NOPE"]["status"] == "skipped"

    # existing LOWER stop (90) -> raise: cancel old, place GTC at 95
    c2 = Client([_Pos("HELD", 40)], {"HELD": [_Order("lo", "HELD", 90.0)]})
    r2 = {x["ticker"]: x for x in run(c2)["results"]}
    assert r2["HELD"]["status"] == "stop_only" and r2["HELD"]["stop_price"] == 95.0
    assert "lo" in c2.cancelled
    sreq = [o for o in c2.submitted if isinstance(o, StopOrderRequest)][0]
    assert int(sreq.qty) == 40 and float(sreq.stop_price) == 95.0
    assert sreq.time_in_force == TimeInForce.GTC

    # existing EQUAL stop (95) -> kept, no churn
    c3 = Client([_Pos("HELD", 40)], {"HELD": [_Order("eq", "HELD", 95.0)]})
    r3 = {x["ticker"]: x for x in run(c3)["results"]}
    assert r3["HELD"]["status"] == "stop_kept" and not c3.submitted and not c3.cancelled

    # NO existing stop -> first-arm a GTC stop at 95
    c4 = Client([_Pos("HELD", 40)], {})
    r4 = {x["ticker"]: x for x in run(c4)["results"]}
    assert r4["HELD"]["status"] == "stop_only" and r4["HELD"]["stop_price"] == 95.0


def test_rearm_gtc_stop_requantifies_grown_position():
    """Issue 9: a would-be-lower re-arm normally KEEPS the higher in-force stop — but if that stop
    covers fewer shares than are held (the position grew via a manual pyramid buy Alpaca-side), the
    ratchet re-places it at the SAME (never-lower) level for the FULL held qty so the added shares
    aren't left unprotected. A stop that already covers the whole position is kept untouched, and an
    order with an unreadable qty never triggers needless churn."""
    from src.stock_screener.cockpit import trade
    from alpaca.trading.requests import StopOrderRequest
    from alpaca.trading.enums import TimeInForce
    Client, _Pos, _Order = _pos_fakes()

    def run(client):                                          # new stop 95 < the in-force 110 stop
        orig = trade._connect_paper
        trade._connect_paper = lambda: (client, True)
        try:
            return trade.rearm_stops([{"ticker": "HELD", "stop_price": 95.0, "price": 130.0}])
        finally:
            trade._connect_paper = orig

    # in-force stop @110 covers only 20 of 40 held -> re-place at 110 (NOT lowered to 95) for 40
    c1 = Client([_Pos("HELD", 40)], {"HELD": [_Order("u", "HELD", 110.0, qty=20)]})
    r1 = {x["ticker"]: x for x in run(c1)["results"]}
    assert r1["HELD"]["status"] == "stop_only" and r1["HELD"]["stop_price"] == 110.0
    assert "u" in c1.cancelled
    sreq = [o for o in c1.submitted if isinstance(o, StopOrderRequest)][0]
    assert int(sreq.qty) == 40 and float(sreq.stop_price) == 110.0
    assert sreq.time_in_force == TimeInForce.GTC

    # in-force stop already covers the whole 40 -> kept, no churn
    c2 = Client([_Pos("HELD", 40)], {"HELD": [_Order("f", "HELD", 110.0, qty=40)]})
    r2 = {x["ticker"]: x for x in run(c2)["results"]}
    assert r2["HELD"]["status"] == "stop_kept" and not c2.submitted and not c2.cancelled

    # unreadable qty (None) -> can't prove under-coverage -> kept, no churn (fail-safe)
    c3 = Client([_Pos("HELD", 40)], {"HELD": [_Order("n", "HELD", 110.0)]})
    r3 = {x["ticker"]: x for x in run(c3)["results"]}
    assert r3["HELD"]["status"] == "stop_kept" and not c3.submitted and not c3.cancelled


def test_rearm_gtc_stop_restores_stop_on_failure():
    """Item 14 (the cancel-before-place gap, re-arm side): if the replacement GTC stop is
    rejected AFTER the old stop was cancelled, the previous stop is restored at its old
    level for the full held qty; a failed restore says to arm one manually. Success-path
    details/statuses are pinned by the two ratchet tests above and must not change."""
    from src.stock_screener.cockpit import trade
    from alpaca.trading.requests import StopOrderRequest
    from alpaca.trading.enums import TimeInForce
    Client, _Pos, _Order = _pos_fakes()

    class FailFirstN(Client):
        def __init__(self, *a, fail_n=1, **kw):
            super().__init__(*a, **kw)
            self._fail_n = fail_n

        def submit_order(self, order_data=None):
            if self._fail_n > 0:
                self._fail_n -= 1
                raise RuntimeError("rejected")
            return super().submit_order(order_data=order_data)

    def run(client, stop_price):
        orig = trade._connect_paper
        trade._connect_paper = lambda: (client, True)
        try:
            return trade.rearm_stops([{"ticker": "HELD", "stop_price": stop_price,
                                       "price": 130.0}])["results"][0]
        finally:
            trade._connect_paper = orig

    # (A) RAISE path: replacement @95 rejected -> old stop @90 restored for the full 40.
    cA = FailFirstN([_Pos("HELD", 40)], {"HELD": [_Order("lo", "HELD", 90.0)]}, fail_n=1)
    rA = run(cA, 95.0)
    assert rA["status"] == "failed" and "lo" in cA.cancelled
    rest = [o for o in cA.submitted if isinstance(o, StopOrderRequest)]
    assert len(rest) == 1 and int(rest[0].qty) == 40 and float(rest[0].stop_price) == 90.0
    assert rest[0].time_in_force == TimeInForce.GTC
    assert "previous stop restored @ 90.00" in rA["detail"] and rA["stop_price"] == 90.0

    # (B) everything rejected -> loud manual-arm message, nothing successfully placed.
    cB = FailFirstN([_Pos("HELD", 40)], {"HELD": [_Order("lo", "HELD", 90.0)]}, fail_n=99)
    rB = run(cB, 95.0)
    assert rB["status"] == "failed" and "arm a stop manually" in rB["detail"]
    assert not cB.submitted

    # (C) UNDER-COVER path (re-place at cur for the grown qty) rejected -> same restore.
    cC = FailFirstN([_Pos("HELD", 40)], {"HELD": [_Order("u", "HELD", 110.0, qty=20)]},
                    fail_n=1)
    rC = run(cC, 95.0)                                   # 95 < 110 -> under-cover branch
    assert rC["status"] == "failed" and "u" in cC.cancelled
    restC = [o for o in cC.submitted if isinstance(o, StopOrderRequest)]
    assert len(restC) == 1 and int(restC[0].qty) == 40 and float(restC[0].stop_price) == 110.0
    assert "previous stop restored @ 110.00" in rC["detail"]

    # first-arm failure (nothing cancelled) -> plain failed, no restore attempted
    cD = FailFirstN([_Pos("HELD", 40)], {}, fail_n=99)
    rD = run(cD, 95.0)
    assert rD["status"] == "failed" and "restore" not in rD["detail"]
    assert not cD.cancelled and not cD.submitted


def test_gate_status_matrix():
    """#23 progressive-exposure gate: tagged-only scope (manual holdings read as flat),
    first pilot always allowed, newest-DAY set must all be at breakeven+ AND net open
    P&L >= 0, unknown entry dates count as newest (conservative), and the consecutive-
    loss streak (exit_date-sorted, scratch resets) drives the advisory half-size
    factor."""
    from src.stock_screener.cockpit import trade

    def pos(sym, plpc, pl, qty=10):
        return {"symbol": sym, "qty": qty, "avg_entry": 100.0, "current_price": 100.0,
                "unrealized_plpc": plpc, "unrealized_pl": pl}

    def oep(sym, date, tagged=True):
        return {"symbol": sym, "entry_date": date, "shares_open": 10.0,
                "avg_entry": 100.0, "tagged": tagged}

    def cep(sym, exit_date, pl, tagged=True):
        return {"symbol": sym, "exit_date": exit_date, "pl": pl, "tagged": tagged}

    g = trade.gate_status([], [], [])
    assert g["open"] is True and g["probe_size_factor"] == 1.0
    assert g["consecutive_losses"] == 0

    # Manual/untagged holdings never poison the gate — reads as flat.
    g = trade.gate_status([pos("ARMK", -0.05, -100.0)],
                          [oep("ARMK", "2026-08-10", tagged=False)], [])
    assert g["open"] is True and "flat" in g["reason"]

    # Newest (by entry DAY) below breakeven -> closed, even with an older winner.
    g = trade.gate_status(
        [pos("AAA", 0.05, 50.0), pos("BBB", -0.02, -20.0)],
        [oep("AAA", "2026-08-10"), oep("BBB", "2026-08-18")], [])
    assert g["open"] is False and "BBB" in g["reason"]

    # Newest green but the tagged book net-red -> closed.
    g = trade.gate_status(
        [pos("AAA", -0.06, -60.0), pos("BBB", 0.01, 10.0)],
        [oep("AAA", "2026-08-10"), oep("BBB", "2026-08-18")], [])
    assert g["open"] is False and "net open" in g["reason"]

    # Newest green, net green -> open.
    g = trade.gate_status(
        [pos("AAA", 0.02, 20.0), pos("BBB", 0.01, 10.0)],
        [oep("AAA", "2026-08-10"), oep("BBB", "2026-08-18")], [])
    assert g["open"] is True

    # Two same-day pilots: BOTH must be at breakeven+ (timestamps differ by ms only).
    g = trade.gate_status(
        [pos("AAA", 0.02, 20.0), pos("BBB", -0.01, -10.0)],
        [oep("AAA", "2026-08-18 09:30:01"), oep("BBB", "2026-08-18 09:30:02")], [])
    assert g["open"] is False and "BBB" in g["reason"]

    # Unreadable entry date counts as newest (conservative).
    g = trade.gate_status(
        [pos("AAA", 0.05, 50.0), pos("CCC", -0.03, -10.0, qty=5)],
        [oep("AAA", "2026-08-10"), oep("CCC", "not-a-date")], [])
    assert g["open"] is False and "CCC" in g["reason"]

    # Streak: two most-recent tagged closed trades red -> half-size advisory; a newer
    # scratch (pl=0) resets; untagged losses are invisible.
    closed = [cep("X", "2026-08-10", -50.0), cep("Y", "2026-08-12", 80.0),
              cep("Z", "2026-08-14", -30.0), cep("W", "2026-08-15", -20.0)]
    g = trade.gate_status([], [], closed)
    assert g["consecutive_losses"] == 2 and g["probe_size_factor"] == 0.5
    assert "half-size" in g["reason"]
    g = trade.gate_status([], [], closed + [cep("S", "2026-08-17", 0.0)])
    assert g["consecutive_losses"] == 0 and g["probe_size_factor"] == 1.0
    g = trade.gate_status([], [], [cep("U1", "2026-08-14", -30.0, tagged=False),
                                   cep("U2", "2026-08-15", -20.0, tagged=False)])
    assert g["consecutive_losses"] == 0


def test_r_multiple_reconstruction():
    """#19: R = gain / reconstructed initial risk. A frozen pivot below the entry
    reproduces the OTO's actual stop (exact); no pivot, or a pivot at/above the entry
    (derived risk <= 0), falls back to INITIAL_STOP_PCT off the entry (approximate);
    degenerate inputs -> (None, True)."""
    from src.stock_screener.cockpit import trade

    r, approx = trade.r_multiple(100.0, 115.0, pivot=100.0)
    assert approx is False and abs(r - 2.0) < 1e-9        # risk = 100 - 92.5 = 7.5

    r, approx = trade.r_multiple(100.0, 116.0, pivot=None)
    assert approx is True and abs(r - 2.0) < 1e-9         # risk = 8.0

    r, approx = trade.r_multiple(100.0, 116.0, pivot=110.0)
    assert approx is True and abs(r - 2.0) < 1e-9         # pivot-stop 101.75 >= entry

    assert trade.r_multiple(None, 100.0) == (None, True)
    assert trade.r_multiple(0.0, 100.0) == (None, True)


def test_submit_buy_plan_skips_gate_blocked():
    """#23 server-side backstop: a gate_blocked row is skipped before any order is
    sent — a stale client (gate closed at Build, checkbox still on) can't slip a buy
    through."""
    from unittest.mock import patch
    from src.stock_screener.cockpit import trade

    FakeClient, _Order = _submit_fakes()
    client = FakeClient()
    plan = [{"ticker": "AAA", "shares": 10, "price": 50.0, "pivot": 50.0,
             "est_value": 500.0, "extended": False, "capped": False,
             "stop_price": 46.0, "limit_price": None, "earnings_in": None,
             "gate_blocked": True}]
    with patch.object(trade, "_connect_paper", return_value=(client, True)):
        out = trade.submit_buy_plan(plan, attach_stop=True)
    assert out["results"][0]["status"] == "skipped"
    assert "gate" in out["results"][0]["detail"]
    assert client.submitted == [], "no order may reach the API for a gate-blocked row"



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "trade"))
