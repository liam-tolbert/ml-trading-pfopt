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
    s = scan_mod._step2_summary(None)
    assert s["score"] == 0 and s["available"] is False
    strong = scan_mod._step2_summary(
        {"revenue_yoy": 35.0, "eps_yoy": 50.0, "eps_yoy_prev": 40.0, "margin_trend": 2.0})
    assert strong["score"] == 4
    weak = scan_mod._step2_summary({"revenue_qoq": 1.0, "eps_qoq": -5.0})
    assert weak["score"] == 0


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
        # score 4/4 checks, the rest have no data (score 0).
        if t and t[-1] in "02468":
            return {"revenue_yoy": 40.0, "eps_yoy": 60.0, "eps_yoy_prev": 50.0,
                    "margin_trend": 1.0, "operating_margin": 25.0}
        return None

    loosest = screen_universe(list(prices), prices, spy, get_fundamentals=_fund,
                              cfg=ScanConfig(min_rs=0.0))
    assert len(loosest.candidates) >= 5, "fixture too small for a meaningful parity test"

    cases = [dict(min_rs=r) for r in (0.0, 60.0, 70.0, 90.0, 99.0)]
    cases += [dict(require_vcp=True), dict(min_fundamental_score=1),
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

    with patch.object(dfeed, "get_universe", lambda u, **kw: ["UPUP"]), \
            patch.object(dfeed, "get_spy", fake_spy), \
            patch.object(dfeed, "get_many_prices", fake_many), \
            patch.object(dfeed, "get_fundamentals", lambda t, **kw: None):
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
