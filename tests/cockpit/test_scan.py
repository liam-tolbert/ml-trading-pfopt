"""Cockpit tests — the SEPA screening funnel — trend template, RS, Step-2, entry levels, charts.

Runs standalone (`python tests/cockpit/test_scan.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_screen_universe_finds_winners_rejects_losers():
    prices, spy, _ = _synthetic_slice()
    res = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                          cfg=ScanConfig(min_rs=0.0))
    assert res.n_scanned > 0
    cands = set(res.candidates["ticker"]) if len(res.candidates) else set()
    assert any(t.startswith("WIN") for t in cands), f"no winners in candidates: {cands}"
    assert not any(t.startswith("LOS") for t in cands), f"a loser slipped through: {cands}"
    # every candidate genuinely cleared the Step-1 gate
    if len(res.candidates):
        assert (res.candidates["criteria"] >= 7).all()
    # regime banner is populated
    assert "regime" in res.regime and "phase2_pct" in res.regime


def test_strict_gate_and_fundamental_filter():
    prices, spy, _ = _synthetic_slice()
    base = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0))  # default 8/8

    # the default 8/8 gate is a subset of a looser 7/8 gate -> never more candidates
    loose = screen_universe(list(prices), prices, spy,
                            cfg=ScanConfig(min_rs=0.0, min_criteria=7))
    assert len(base.candidates) <= len(loose.candidates)
    if len(base.candidates):
        assert (base.candidates["criteria"] >= 8).all()

    # a fundamentals callable that passes everything raises fund_score; requiring
    # >=3 checks must not increase the candidate set vs no requirement
    def good_fund(_t):
        return {"revenue_yoy": 40.0, "revenue_yoy_prev": 30.0, "eps_yoy": 60.0,
                "eps_yoy_prev": 50.0, "margin_trend": 1.0, "operating_margin": 25.0}

    gated = screen_universe(list(prices), prices, spy, get_fundamentals=good_fund,
                            cfg=ScanConfig(min_rs=0.0, min_fundamental_score=3))
    assert len(gated.candidates) <= len(base.candidates)
    if len(gated.candidates):
        assert (gated.candidates["fund_score"] >= 3).all()


def test_build_chart_returns_figure_with_expected_traces():
    prices, _, _ = _synthetic_slice()
    ticker, df = next(iter(prices.items()))
    levels = {"pivot": float(df["Close"].iloc[-1]) * 1.02,
              "buy_zone": (float(df["Close"].iloc[-1]) * 1.02,
                           float(df["Close"].iloc[-1]) * 1.07),
              "stop": float(df["Close"].iloc[-1]) * 0.94,
              "target": float(df["Close"].iloc[-1]) * 1.25}
    fig = build_chart(ticker, df, vcp={"contractions": []}, levels=levels)
    assert isinstance(fig, go.Figure)
    assert any(isinstance(tr, go.Candlestick) for tr in fig.data), "no candlestick"
    n_sma = sum(isinstance(tr, go.Scatter) for tr in fig.data)
    assert n_sma >= 3, f"expected 3 SMA overlays, got {n_sma}"
    vol_bar = next(tr for tr in fig.data if isinstance(tr, go.Bar))
    assert not isinstance(vol_bar.marker.color, str), "volume bars must be colored per-bar"
    assert len(vol_bar.marker.color) == len(vol_bar.x), "one up/down color per bar"
    # weekly view should also build
    assert isinstance(build_chart(ticker, df, weekly=True), go.Figure)

    # lookback_days zooms the VIEW: fewer candles than the full series, but SMAs intact
    def _candle_len(f):
        return len(next(tr for tr in f.data if isinstance(tr, go.Candlestick)).x)
    full_n = _candle_len(build_chart(ticker, df))
    zoom_n = _candle_len(build_chart(ticker, df, lookback_days=90))
    assert zoom_n < full_n, f"lookback did not slice the view ({zoom_n} vs {full_n})"
    assert sum(isinstance(tr, go.Scatter) for tr in
               build_chart(ticker, df, lookback_days=90).data) >= 3


def test_rs_ratings_ibd_weighted():
    """The RS rating is an IBD-style weighted multi-horizon percentile (2×3mo + 6mo + 9mo
    + 12mo): a smaller move concentrated in the last 3 months outranks a bigger move that
    happened a year ago (recency counts double); a young name (≥6mo history) is rated on
    the legs it has; a <6mo name is excluded."""
    import pandas as pd
    from src.stock_screener.cockpit.scan import _rs_ratings

    def frame(closes):
        idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=len(closes))
        return pd.DataFrame({"Close": closes}, index=idx)

    n = 260                                              # > 12mo of bars
    # RECENT: +20% entirely inside the last ~3 months -> EVERY window sees it (they nest),
    # so the blend is the full +20%
    recent = [100.0] * (n - 60) + [100.0 + 20.0 * (i + 1) / 60 for i in range(60)]
    # STALE: +25% a year ago, dead flat since -> only the 12-mo leg still sees (part of)
    # it -> blend ≈ +4%
    stale = [100.0 + 25.0 * (i + 1) / 55 for i in range(55)] + [125.0] * (n - 55)
    # YOUNG: 140 bars (6-mo leg exists, 9/12-mo don't), +10% in the last 3 months
    young = [100.0] * 80 + [100.0 + 10.0 * (i + 1) / 60 for i in range(60)]
    prices = {"RECENT": frame(recent), "STALE": frame(stale),
              "YOUNG": frame(young), "FLAT": frame([100.0] * n),
              "SHORT": frame([100.0] * 100)}             # < 6mo -> no rating

    rs = _rs_ratings(prices, 126)
    assert "SHORT" not in rs and set(rs) == {"RECENT", "STALE", "YOUNG", "FLAT"}
    # recency weighting: RECENT's +20% (all in 3mo) beats STALE's bigger-but-old +25%
    assert rs["RECENT"] > rs["YOUNG"] > rs["STALE"] > rs["FLAT"], rs
    assert all(0 <= v <= 99 for v in rs.values())


def test_step2_summary_logic():
    """§6.89: F counts eight checks. The four original ones, then Code 33, annual EPS up,
    estimates raised >= 5% over 90 days, and the last report held (no hard drop on heavy
    volume). A missing figure fails its check; a missing reaction fails only the last."""
    s = scan_mod._step2_summary(None)
    assert s["score"] == 0 and s["available"] is False
    old = {"revenue_yoy": 35.0, "eps_yoy": 50.0, "eps_yoy_prev": 40.0, "margin_trend": 2.0}
    strong = scan_mod._step2_summary(old)
    assert strong["score"] == 4 and len(strong["checks"]) == 8
    full = {**old, "code33": {"eps": True, "sales": True, "margin": True, "all": True},
            "eps_fy_up": True, "est_rev_90d": 5.0}
    held = {"flag": None, "day_pct": 1.0}
    assert scan_mod._step2_summary(full, held)["score"] == 8
    no_react = scan_mod._step2_summary(full)
    assert no_react["score"] == 7 and no_react["checks"]["report_held"] is False
    dropped = scan_mod._step2_summary(full, {"flag": "hard_drop"})
    assert dropped["checks"]["report_held"] is False
    assert scan_mod._step2_summary(full, {"flag": "strong"})["checks"]["report_held"]
    part = {**full, "code33": {"eps": True, "sales": True, "margin": False, "all": False},
            "est_rev_90d": 4.9, "eps_fy_up": None}
    assert scan_mod._step2_summary(part, held)["score"] == 5
    weak = scan_mod._step2_summary({"revenue_qoq": 1.0, "eps_qoq": -5.0})
    assert weak["score"] == 0


def test_code33_inventory_and_step2_lines():
    """§6.86: the scan's Code 33 count, the inventory warning and the Step-2 panel lines,
    from a fundamentals dict; an older cache without the keys yields None and no lines."""
    f = {"code33": {"eps": True, "sales": False, "margin": True, "all": False},
         "eps_g3": [40.0, 30.0, 20.0], "rev_g3": [10.0, 12.0, 11.0], "margin3": [8.0, 9.0, 9.5],
         "eps_decel_2q": True, "eps_fy_up": False, "eps_fy_up_3y": None,
         "inventory_qoq": 15.0, "revenue_qoq": 2.0}
    assert scan_mod.code33_parts(f) == 2
    assert scan_mod.inventory_flag(f) is True
    assert scan_mod.inventory_flag({**f, "inventory_qoq": 11.9}) is False
    assert scan_mod.inventory_flag({"revenue_qoq": 2.0}) is None
    lines = scan_mod.step2_lines(f)
    assert lines[0].startswith("**Code 33** 2/3 · EPS +40%→+30%→+20% ✅ · sales"), lines
    assert "sales +10%→+12%→+11% —" in lines[0]
    assert lines[1] == "**Annual EPS** ↓ —"
    assert lines[2].startswith("⚠️ EPS growth slowed two quarters running")
    assert lines[3].startswith("⚠️ Inventory +15.0% vs sales +2.0%")
    assert scan_mod.code33_parts({"revenue_yoy": 1.0}) is None
    # §6.88: estimates and funds
    est = scan_mod.step2_lines({"est_rev_90d": 84.6, "est_rev_30d": -14.4, "inst_count": 235,
                                "inst_pct": 96.0,
                                "inst_history": [["2026-07-02", 221], ["2026-09-25", 235]]})
    assert est == ["**Estimates** this year's EPS +84.6% over 90 days (-14.4% over 30) ✅",
                   "**Funds** 235 holding 96% (↑ from 221 on 2026-07-02)"], est
    cut = scan_mod.step2_lines({"est_rev_90d": -6.0, "inst_count": 10})
    assert cut == ["**Estimates** this year's EPS -6.0% over 90 days ⚠️ analysts are cutting",
                   "**Funds** 10"], cut
    assert scan_mod.step2_lines({"revenue_yoy": 1.0}) == [] and scan_mod.step2_lines(None) == []


def test_entry_levels_stop_clamped_to_pivot():
    """The advisory stop is floored at 10% below the pivot (Minervini's hard max): a looser
    engine stop is clamped up and flagged, a tighter one is kept, and the no-stop default is
    7.5% below the pivot."""
    bo = {"breakout_level": 100.0, "is_breakout": False, "volume_ratio": 1.0,
          "volume_confirmed": False}
    ph = {"week_52_high": 100.0}

    # loose engine stop (20% below the pivot) -> clamped up to 90.0 (10% below), flagged
    loose = scan_mod._entry_levels(95.0, bo, 80.0, ph)
    assert loose["stop"] == 90.0 and loose["stop_clamped"] is True
    assert abs(loose["stop_pct_from_pivot"] - 10.0) < 1e-9

    # in-range stop (7% below) -> kept, not flagged
    ok = scan_mod._entry_levels(99.0, bo, 93.0, ph)
    assert ok["stop"] == 93.0 and ok["stop_clamped"] is False
    assert abs(ok["stop_pct_from_pivot"] - 7.0) < 1e-9

    # tighter stop (3% below) -> kept as-is; the clamp only bounds the loose side
    tight = scan_mod._entry_levels(99.0, bo, 97.0, ph)
    assert tight["stop"] == 97.0 and tight["stop_clamped"] is False

    # no/invalid engine stop -> 7.5%-below-pivot default, within the max, not flagged
    default = scan_mod._entry_levels(99.0, bo, None, ph)
    assert default["stop"] == 92.5 and default["stop_clamped"] is False
    assert default["stop"] < default["pivot"]               # never at/above the pivot

    # §6.72: the max loss is from the price PAID — 92.5 covers fills up to 102.78; the
    # zone runs to 105, so a plan filling higher raises the stop
    assert abs(default["max_fill_for_stop"] - 92.5 / 0.9) < 1e-9
    assert default["max_fill_for_stop"] < default["buy_zone"][1]


def test_entry_levels_ignores_50sma_breakout_pivot():
    """A '50 SMA Breakout' level is the 50-day SMA (a routine pullback recovery), not a base
    pivot — _entry_levels must ignore it and fall through to the 52-week-high fallback so the
    buy zone/stop/target (and the frozen trigger level) anchor to a real pivot."""
    ph = {"week_52_high": 120.0}
    # 50-SMA reclaim: the engine reports a breakout AT the 50-day SMA (90.0), below the market.
    sma_bo = {"breakout_level": 90.0, "breakout_type": "50 SMA Breakout",
              "is_breakout": True, "volume_ratio": 1.0, "volume_confirmed": False}
    lv = scan_mod._entry_levels(100.0, sma_bo, None, ph)
    assert lv["pivot"] == 120.0                             # the 52-wk high, NOT the 90.0 SMA
    assert lv["buy_zone"][0] == 120.0
    assert lv["target"] == 120.0 * 1.25

    # a genuine VCP/base breakout level IS still adopted as the pivot
    vcp_bo = {"breakout_level": 110.0, "breakout_type": "VCP Breakout (3 contractions)",
              "is_breakout": True, "volume_ratio": 1.6, "volume_confirmed": True}
    lv2 = scan_mod._entry_levels(112.0, vcp_bo, None, ph)
    assert lv2["pivot"] == 110.0


def test_filter_candidates_matches_scan_gates():
    """Item 18 parity: filtering a LOOSEST-gates scan with filter_candidates yields exactly
    the tickers (order included) that screen_universe produces when the same gates run
    inside the funnel — so the post-filter memo design changes nothing but speed."""
    from src.stock_screener.cockpit.scan import filter_candidates

    prices, spy, _ = _synthetic_slice()

    def _fund(t):
        # Varied fundamentals so min_fund actually splits the fixture: even-digit names
        # pass 7 of 8 checks (no release date, so no reaction), the rest have no data (0).
        if t and t[-1] in "02468":
            return {"revenue_yoy": 40.0, "eps_yoy": 60.0, "eps_yoy_prev": 50.0,
                    "margin_trend": 1.0, "operating_margin": 25.0,
                    "code33": {"eps": True, "sales": True, "margin": True, "all": True},
                    "eps_fy_up": True, "est_rev_90d": 12.0}
        return None

    loosest = screen_universe(list(prices), prices, spy, get_fundamentals=_fund,
                              cfg=ScanConfig(min_rs=0.0))
    assert len(loosest.candidates) >= 5, "fixture too small for a meaningful parity test"

    cases = [dict(min_rs=r) for r in (0.0, 60.0, 70.0, 90.0, 99.0)]
    cases += [dict(require_vcp=True), dict(min_fundamental_score=1),
              dict(min_fundamental_score=6), dict(min_fundamental_score=8),
              dict(min_rs=70.0, min_fundamental_score=1)]
    for kw in cases:
        gated = screen_universe(list(prices), prices, spy, get_fundamentals=_fund,
                                cfg=ScanConfig(min_rs=kw.get("min_rs", 0.0),
                                               require_vcp=kw.get("require_vcp", False),
                                               min_fundamental_score=kw.get(
                                                   "min_fundamental_score", 0)))
        post = filter_candidates(loosest.candidates, **kw)
        g = (gated.candidates["ticker"].tolist()
             if len(gated.candidates) else [])
        assert (post["ticker"].tolist() if len(post) else []) == g, (kw, g)
    # Columnless-empty edge: the memoized empty ScanResult frame has no columns.
    import pandas as pd
    assert len(filter_candidates(pd.DataFrame(), min_rs=70)) == 0
    assert len(filter_candidates(None, min_rs=70)) == 0


def test_rs_line_new_high_flag():
    """§6.39 RS line at new high before price (IBD blue dot): the ÷SPY line at its 52-wk
    high while the stock still bases -> True; underperformance -> False; too little
    overlapping history -> None (unknown, never a failed check). Funnel: `rs_nh` lands in
    the candidates frame and every payload."""
    import pandas as pd
    from src.stock_screener.cockpit.scan import rs_line_at_high

    idx = pd.bdate_range(end="2026-06-30", periods=300)
    # SPY drifts down 5% while the stock holds flat -> RS line rises to its high
    spy_dn = pd.Series([300.0 * (1 - 0.05 * i / 299) for i in range(300)], index=idx)
    flat = pd.DataFrame({"Close": [100.0] * 300}, index=idx)
    assert rs_line_at_high(flat, spy_dn) is True

    # stock falls 10% while SPY holds flat -> RS line at its LOWS
    spy_flat = pd.Series([300.0] * 300, index=idx)
    falling = pd.DataFrame(
        {"Close": [100.0 * (1 - 0.10 * i / 299) for i in range(300)]}, index=idx)
    assert rs_line_at_high(falling, spy_flat) is False

    # under min_days of overlap -> None
    short = pd.DataFrame({"Close": [100.0] * 50}, index=idx[-50:])
    assert rs_line_at_high(short, spy_flat) is None

    # funnel integration: the flag reaches rows + payloads (value bool or None)
    prices, spy, _ = _synthetic_slice()
    res = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                          cfg=ScanConfig(min_rs=0.0))
    assert "rs_nh" in res.candidates.columns
    assert all("rs_nh" in p for p in res.payloads.values())
    for v in res.candidates["rs_nh"]:
        assert v is None or v is True or v is False or pd.isna(v)


def test_book_template():
    """§6.80 (audit Step-1 #1): the gate is the book's eight criteria — the vendored
    template's seven price criteria plus an RS rating >= RS_FLOOR (70). The vendored
    eighth, a Stage-2 slope check, is not one of the book's and no longer counts: a name
    with the seven and RS 84 passes even when that slope reads negative; RS 64 or an
    unknown rating fails the eighth."""
    from src.stock_screener.cockpit.scan import book_template, price_criteria_passed

    seven = {"price_above_150_200": True, "sma_150_above_200": True,
             "sma_200_rising": True, "sma_50_above_150": True, "price_above_50": True,
             "price_30pct_above_52w_low": True, "price_near_52w_high": True,
             "distance_from_52w_low_pct": 45.0, "distance_from_52w_high_pct": 3.0}
    tmpl = {"criteria_passed": 7, "criteria_details": {**seven, "confirmed_stage_2": False}}
    b = book_template(tmpl, 84)
    assert b["passes"] and b["criteria_passed"] == 8 and b["rs_ok"] is True
    assert set(b["details"]) == set(seven) - {"distance_from_52w_low_pct",
                                              "distance_from_52w_high_pct"} | {"rs_rating"}
    assert price_criteria_passed(tmpl) == 7

    low = book_template(tmpl, 64)
    assert not low["passes"] and low["criteria_passed"] == 7 and low["rs_ok"] is False
    unknown = book_template(tmpl, None)
    assert not unknown["passes"] and unknown["criteria_passed"] == 7
    assert unknown["rs_ok"] is None
    assert book_template(tmpl, float("nan"))["rs_ok"] is None

    # a price criterion failing counts whatever the RS
    broken = {**tmpl, "criteria_details": {**tmpl["criteria_details"],
                                           "price_above_50": False}}
    assert book_template(broken, 99)["criteria_passed"] == 7
    assert price_criteria_passed(broken) == 6
    assert price_criteria_passed(None) is None
    assert book_template(None, 90)["criteria_passed"] == 1


def test_screen_universe_gates_on_rs():
    """§6.80: RS >= 70 is the eighth criterion of the scan gate. A name that passes the
    seven price criteria with RS 64 is not a candidate; at 70 it is. The result carries
    every name's rating (``rs_ratings``), not only the passers', so a held name that
    dropped out of the list still has one for P2."""
    from unittest.mock import patch

    prices, spy, _ = _synthetic_slice()
    real = scan_mod._rs_ratings(prices, 126)
    base = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0))
    assert len(base.candidates) >= 3
    assert (base.candidates["rs"] >= 70).all(), base.candidates[["ticker", "rs"]]
    assert (base.candidates["criteria"] == 8).all()
    assert set(base.rs_ratings) == set(real) and len(base.rs_ratings) > len(base.candidates)

    top = base.candidates["ticker"].iloc[0]
    with patch.object(scan_mod, "_rs_ratings", lambda p, n: {**real, top: 64}):
        low = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0))
    assert top not in set(low.candidates["ticker"])
    assert low.rs_ratings[top] == 64                      # rated, just not a candidate
    with patch.object(scan_mod, "_rs_ratings", lambda p, n: {**real, top: 70}):
        edge = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0))
    assert top in set(edge.candidates["ticker"])

    # an older pickle's ScanResult has no rs_ratings attribute: readers use getattr
    import pickle
    old = pickle.loads(pickle.dumps(base))
    del old.__dict__["rs_ratings"]
    assert getattr(pickle.loads(pickle.dumps(old)), "rs_ratings", {}) == {}


def test_rs_line_trend_labels():
    """§6.80 (audit Step-1 #2): the RS line's direction over ~6 and ~13 weeks, the 2017
    template's addition. One synthetic ratio per label; too little overlap -> None."""
    import pandas as pd
    from src.stock_screener.cockpit.scan import rs_line_trend

    idx = pd.bdate_range(end="2026-06-30", periods=120)
    spy = pd.Series(300.0, index=idx)

    def trend(closes):
        return rs_line_trend(pd.DataFrame({"Close": closes}, index=idx), spy)

    up = [100.0 + 0.3 * i for i in range(120)]
    assert trend(up)["label"] == "rising 13w"
    assert trend(up)["slope_6w"] > 0 and trend(up)["slope_13w"] > 0
    # down 8% over the first 90 bars, then up 4% over the last 30: short up, long down
    late = [100.0 - 8.0 * i / 89 for i in range(90)] + [92.0 + 4.0 * i / 29 for i in range(30)]
    assert trend(late)["label"] == "rising 6w"
    # up 12% over the first 90 bars, then down 3% over the last 30: long up, short down
    roll = [100.0 + 12.0 * i / 89 for i in range(90)] + [112.0 - 3.0 * i / 29 for i in range(30)]
    assert trend(roll)["label"] == "rolling over"
    down = [100.0 - 0.2 * i for i in range(120)]
    assert trend(down)["label"] == "falling"
    assert trend([100.0] * 120)["label"] == "flat"
    assert trend([100.0] * 120)["slope_13w"] == 0.0

    short = pd.DataFrame({"Close": [100.0] * 40}, index=idx[-40:])
    assert rs_line_trend(short, spy) is None


def test_sma200_rising_months():
    """§6.80 (audit Step-1 #3): how long the 200-day has risen, in 21-session months,
    by the template's own test (above its value 19 sessions earlier). A series flat for
    250 bars then rising 60 has an SMA rising for exactly those 60 sessions -> 2.9; a
    falling series reads 0.0; under 219 rows -> None."""
    import pandas as pd
    from src.stock_screener.cockpit.scan import sma200_rising_months

    def frame(closes):
        idx = pd.bdate_range(end="2026-06-30", periods=len(closes))
        return pd.DataFrame({"Close": closes}, index=idx)

    flat_then_up = [100.0] * 250 + [100.0 + 0.5 * i for i in range(1, 61)]
    assert sma200_rising_months(frame(flat_then_up)) == 2.9
    assert sma200_rising_months(frame([200.0 - 0.1 * i for i in range(300)])) == 0.0
    # rising throughout: every testable session counts (the first 218 can't be tested)
    assert sma200_rising_months(frame([100.0 + 0.3 * i for i in range(400)])) == \
        round((400 - 218) / 21.0, 1)
    assert sma200_rising_months(frame([100.0] * 218)) is None
    assert sma200_rising_months(None) is None


def test_dollar_adv():
    """§6.81: average daily dollar volume over the last N bars; None without Volume,
    under N bars, or when the mean isn't positive. The scan row carries it in $M."""
    import pandas as pd
    from src.stock_screener.cockpit.indicators import dollar_adv

    idx = pd.bdate_range(end="2026-06-30", periods=30)
    df = pd.DataFrame({"Close": 10.0, "Volume": 100_000}, index=idx)
    assert dollar_adv(df, 20) == 1_000_000.0
    assert dollar_adv(df.iloc[-19:], 20) is None
    assert dollar_adv(df.drop(columns=["Volume"]), 20) is None
    assert dollar_adv(df.assign(Volume=0), 20) is None
    assert dollar_adv(None, 20) is None

    prices, spy, _ = _synthetic_slice()
    res = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0))
    assert "adv_musd" in res.candidates.columns
    assert all("adv_usd" in p for p in res.payloads.values())


def test_breadth_counts():
    """§6.82 (audit Step-1 #8): the scan counts names at a new 52-week high and at a new
    low over every name it phases, before the gate. The regime carries the counts, the
    spread, the session date and, with history, whether the spread is wider than ten
    sessions ago; the breadth-aware re-entry streak says whether it had breadth."""
    from src.stock_screener.cockpit.scan import _new_high_low

    END = "2026-06-30"
    up = [100.0 + 0.3 * i for i in range(260)]                 # last High is the max
    down = [180.0 - 0.3 * i for i in range(260)]               # last Low is the min
    mid = [100.0 + 0.3 * i for i in range(250)] + [175.0 - 0.5 * i for i in range(10)]
    prices = {"NH": _trigger_frame(END, up), "NL": _trigger_frame(END, down),
              "MID": _trigger_frame(END, mid)}
    spy = _trigger_frame(END, [300.0 + 0.2 * i for i in range(260)])

    res = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0))
    reg = res.regime
    assert reg["session"] == END
    assert reg["new_highs"] == 1 and reg["new_lows"] == 1 and reg["nh_nl_spread"] == 0
    assert reg["nh_nl_pct"] == 0.0 and reg["nh_nl_expanding"] is None
    assert reg["spy_ok_breadth"] is False
    if len(res.candidates):
        by = res.candidates.set_index("ticker")
        assert bool(by.loc["NH", "new_high"]) is True
        assert "NL" not in by.index

    # ten settled sessions of history with a -5 spread: today's 0 is wider; the streak
    # now has breadth, partial because the history doesn't cover every counted session
    hist = [{"date": d.strftime("%Y-%m-%d"), "n_scanned": 3, "phase2_pct": 40.0,
             "new_highs": 5, "new_lows": 10}
            for d in __import__("pandas").bdate_range("2026-06-01", periods=12)]
    res2 = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0),
                           breadth_history=hist)
    assert res2.regime["nh_nl_expanding"] is True
    assert res2.regime["spy_ok_breadth"] is True and res2.regime["spy_ok_partial"] is True

    # the helper: Close-only frames fall back to Close; the 2-dp window gets half-cent slack
    import pandas as pd
    idx = pd.bdate_range(end=END, periods=5)
    close_only = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    assert _new_high_low(close_only, {"week_52_high": 5.0, "week_52_low": 1.0}) == (True, False)
    assert _new_high_low(close_only, {"week_52_high": 5.004, "week_52_low": 0.0}) == (True, False)
    assert _new_high_low(close_only, {}) == (False, False)


def test_breadth_store_append_replace_load():
    """§6.82: the breadth history is one CSV row per session, replaced on a same-day
    rerun, sorted on load, robust to a missing file and a malformed line; the spread read
    compares with ten settled sessions back and reads None without them."""
    import tempfile
    from src.stock_screener.cockpit import breadth_store as bs

    def row(d, nh, nl, p2=20.0):
        return {"date": d, "n_scanned": 3900, "phase2_pct": p2, "new_highs": nh,
                "new_lows": nl}

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "b" / "breadth.csv"
        assert bs.load(p) == []
        bs.append(row("2026-09-23", 150, 60), p)
        bs.append(row("2026-09-22", 120, 80), p)
        bs.append(row("2026-09-23", 155, 58), p)               # same day: replaced
        rows = bs.load(p)
        assert [r["date"] for r in rows] == ["2026-09-22", "2026-09-23"]
        assert rows[1]["new_highs"] == 155 and rows[1]["n_scanned"] == 3900
        assert isinstance(rows[0]["phase2_pct"], float)
        assert not list(p.parent.glob("*.tmp"))
        with open(p, "a", encoding="utf-8") as f:
            f.write("garbage,line\n")
        assert len(bs.load(p)) == 2
        assert bs.phase2_by_date(rows) == {"2026-09-22": 20.0, "2026-09-23": 20.0}
        try:
            bs.append({"date": "2026-09-24"}, p)
            raise AssertionError("a malformed row must raise")
        except ValueError:
            pass

    import pandas as pd
    hist = [row(d.strftime("%Y-%m-%d"), 100 + i, 50) for i, d in
            enumerate(pd.bdate_range("2026-09-01", periods=12))]
    today = "2026-09-30"
    assert bs.spread_expanding(hist, today, 60) is True         # ref = 12-10 -> 102-50
    assert bs.spread_expanding(hist, today, 52) is False
    assert bs.spread_expanding(hist[:9], today, 999) is None
    assert bs.spread_expanding(hist, "2026-09-05", 999) is None  # only 4 prior sessions


def test_screen_job_appends_breadth():
    """§6.82: the scheduled screen appends one breadth row per session after publishing
    the scan, and a same-evening rerun replaces it rather than doubling it. A result with
    no session (an older scan shape) writes nothing and the job still succeeds."""
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import breadth_store, cache, screen_job
    from src.stock_screener.cockpit.scan_worker import ResultStore

    class _Res:
        n_scanned, n_passed, errors = 3927, 417, []
        candidates = [1, 2, 3]
        regime = {"session": "2026-09-24", "phase2_pct": 20.6, "new_highs": 212,
                  "new_lows": 48}

    class _Old:
        n_scanned, n_passed, errors, candidates = 10, 5, [], []
        regime = {"regime": "RISK-ON"}

    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(cache, "BREADTH_CSV", Path(tmp) / "breadth.csv"):
        with patch.object(screen_job.scan, "run_scan", lambda **kw: _Res()):
            screen_job.run_screen(store=ResultStore())
            screen_job.run_screen(store=ResultStore())
        rows = breadth_store.load()
        assert len(rows) == 1 and rows[0]["new_highs"] == 212 and rows[0]["n_scanned"] == 3927
        with patch.object(screen_job.scan, "run_scan", lambda **kw: _Old()):
            out = screen_job.run_screen(store=ResultStore())
        assert out["passed"] == 5 and len(breadth_store.load()) == 1


def test_screen_universe_rows_carry_step1_reads():
    """§6.80: the three Step-1 reads reach the candidate rows and payloads: the book's
    count in ``criteria``, ``rs_trend``/``rs_slope_13w`` and ``sma200_rising_m``."""
    prices, spy, _ = _synthetic_slice()
    res = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0))
    for col in ("rs_trend", "rs_slope_13w", "sma200_rising_m", "criteria",
                "depth_vs_spy", "depth_flag"):
        assert col in res.candidates.columns, col
    for v in res.candidates["depth_vs_spy"].dropna():
        assert v > 0
    assert all("depth" in p for p in res.payloads.values())
    labels = {"rising 13w", "rising 6w", "rolling over", "falling", "flat"}
    assert set(res.candidates["rs_trend"].dropna()) <= labels
    assert (res.candidates["sma200_rising_m"].dropna() >= 0).all()
    for p in res.payloads.values():
        assert p["book_template"]["passes"] and p["book_template"]["rs_ok"] is True
        assert p["rs_trend"] is None or p["rs_trend"]["label"] in labels
        assert "sma200_rising_m" in p


def test_run_scan_uses_topup_fetch():
    """run_scan routes ALL price fetches (universe + SPY) through the ALWAYS-top-up path
    (max_age_days=0.0 — same semantics as the EOD trigger): the old 30-minute freshness
    window is gone (user decision 2026-08-09; scan_worker's process-wide refresh throttle
    is the only fetch-rate limiter now, so a scan that runs must BE fresh). Older caches
    fetch only their missing days; only cold names pay the full 2y download; the
    settled-close serve still short-circuits network after hours. force is not passed by
    the app's Re-scan (top-up instead)."""
    from unittest.mock import patch

    from src.stock_screener.cockpit import data_feed as dfeed
    from src.stock_screener.cockpit.scan import run_scan

    TODAY = "2026-07-10"
    frame = _trigger_frame(TODAY, [100.0 + i * 0.3 for i in range(260)])
    spy = _trigger_frame(TODAY, [300 + i * 0.5 for i in range(260)])
    seen = {}

    def fake_many(tickers, **kw):
        seen["many"] = kw
        return {t: frame for t in tickers}

    def fake_spy(**kw):
        seen["spy"] = kw
        return spy

    from src.stock_screener.cockpit import breadth_store, sectors
    with patch.object(dfeed, "get_universe", lambda u, **kw: ["UPUP"]), \
            patch.object(dfeed, "get_spy", fake_spy), \
            patch.object(dfeed, "get_many_prices", fake_many), \
            patch.object(dfeed, "get_fundamentals", lambda t, **kw: None), \
            patch.object(sectors, "get_sector", lambda t, **kw: {}), \
            patch.object(breadth_store, "load", lambda path=None: []):
        run_scan(universe="full_us")
    assert seen["many"].get("max_age_days") == 0.0, \
        "universe fetch must always top up (no freshness window)"
    assert seen["spy"].get("max_age_days") == 0.0, "SPY fetch must match"
    assert not seen["many"].get("force") and not seen["spy"].get("force")



def test_screen_job_publishes_under_the_key_the_app_reads():
    """cockpit-eod step 2 rebuilds the scan table. It MUST publish under the same
    ``(universe, min_criteria)`` key the app reads, or the result lands in the store and the
    app never sees it — and the failure would be silent, since the job exits 0 either way."""
    from unittest.mock import patch

    from src.stock_screener.cockpit import screen_job
    from src.stock_screener.cockpit.scan_worker import (
        DEFAULT_MIN_CRITERIA, DEFAULT_UNIVERSE, ResultStore)

    class _Res:
        n_scanned, n_passed, errors = 4120, 610, []
        candidates = [1, 2, 3]

    seen = {}

    def _fake_run_scan(universe=None, cfg=None, **kw):
        seen["universe"] = universe
        seen["min_criteria"] = getattr(cfg, "min_criteria", None)
        return _Res()

    store = ResultStore()                      # persist_path=None -> no disk I/O
    with patch.object(screen_job.scan, "run_scan", _fake_run_scan):
        out = screen_job.run_screen(store=store)

    assert seen["universe"] == DEFAULT_UNIVERSE, seen
    assert seen["min_criteria"] == DEFAULT_MIN_CRITERIA, seen
    ent = store.get((DEFAULT_UNIVERSE, DEFAULT_MIN_CRITERIA))
    assert ent is not None, "published under a key the app does not read"
    assert ent.result is not None and out["passed"] == 610, out


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "scan"))
