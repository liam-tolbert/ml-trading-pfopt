"""Deterministic, offline tests for the SEPA cockpit.

Run as a plain script (matching the repo's test style):

    python tests/test_cockpit.py

Covers: (1) the cockpit's data layer is independent of the vendored, SQLAlchemy-
trapped ``minervini_screener.data`` package; (2) the SEPA funnel finds the synthetic
"winner" uptrends and rejects the "losers"; (3) the chart builder returns a proper
Plotly figure. No network is used — prices come from the backtest synthetic provider.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go  # noqa: E402

from src.stock_screener.backtest_daily.synthetic_provider import make_synthetic  # noqa: E402
from src.stock_screener.cockpit import scan as scan_mod  # noqa: E402
from src.stock_screener.cockpit.charts import build_chart  # noqa: E402
from src.stock_screener.cockpit.scan import ScanConfig, screen_universe  # noqa: E402


def test_data_feed_isolated_from_vendored_data_layer():
    """A fresh interpreter importing only cockpit.data_feed must NOT pull the vendored
    ``minervini_screener.data`` package (whose __init__ needs SQLAlchemy)."""
    code = (
        "import sys\n"
        "import src.stock_screener.cockpit.data_feed\n"
        "bad = [m for m in sys.modules if m == 'src.stock_screener.minervini_screener.data'"
        " or m.startswith('src.stock_screener.minervini_screener.data.')]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, f"isolated import failed:\n{out.stdout}\n{out.stderr}"
    assert "OK" in out.stdout


def _synthetic_slice(idx=350):
    data = make_synthetic(seed=7)
    cal = data.price.calendar()
    t = cal[idx]
    prices = {}
    for p in data.winners + data.losers + data.late:
        df = data.price.prices(p).loc[:t]
        if len(df) >= 200:
            prices[data.price.ticker(p)] = df
    spy = data.price.spy().loc[:t]
    return prices, spy, data


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
    assert any(isinstance(tr, go.Bar) for tr in fig.data), "no volume bars"
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


def test_step2_summary_logic():
    s = scan_mod._step2_summary(None)
    assert s["score"] == 0 and s["available"] is False
    strong = scan_mod._step2_summary(
        {"revenue_yoy": 35.0, "eps_yoy": 50.0, "eps_yoy_prev": 40.0, "margin_trend": 2.0})
    assert strong["score"] == 4
    weak = scan_mod._step2_summary({"revenue_qoq": 1.0, "eps_qoq": -5.0})
    assert weak["score"] == 0


def test_streamlit_app_renders_offline():
    """Execute app.py through Streamlit's AppTest with run_scan patched to a real,
    offline ScanResult (synthetic fixture). Verifies the whole UI render path —
    regime banner, table, selectbox, chart, Step-2/Step-4 panels — raises nothing."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_streamlit_app_renders_offline (AppTest unavailable: {e})")
        return
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    assert len(result.candidates) >= 1, "fixture must yield >=1 candidate for the app path"

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    # app.py does `from ...scan import run_scan`, resolved at script-exec time, so
    # patching the source attribute before .run() propagates into the app's namespace.
    with patch.object(scanmod, "run_scan", return_value=result):
        at = AppTest.from_file(app_path, default_timeout=60)
        at.run()
    assert not at.exception, f"app raised: {at.exception}"


def test_sepa_guide_page_renders():
    """The SEPA Guide page must load and render the method markdown without error."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_sepa_guide_page_renders (AppTest unavailable: {e})")
        return
    page = ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "1_SEPA_Guide.py"
    at = AppTest.from_file(str(page), default_timeout=30)
    at.run()
    assert not at.exception, f"guide page raised: {at.exception}"
    assert any("SEPA" in str(getattr(m, "value", "")) for m in at.markdown), \
        "guide page rendered no SEPA markdown"


def test_get_universe_full_us_offline():
    """full_us branch: mocked nasdaqtrader payloads -> many normalized common-stock
    symbols with ETFs / test issues / warrants / dotted class shares filtered out, and
    NO NotImplementedError. Fully offline (mocked requests + temp CACHE_DIR)."""
    import tempfile
    from unittest.mock import patch, MagicMock

    from src.stock_screener.cockpit import data_feed as dfeed

    nasdaq = ("Symbol|Security Name|Market Category|Test Issue|Financial Status|"
              "Round Lot Size|ETF|NextShares\n"
              "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
              "MSFT|Microsoft Corp - Common Stock|Q|N|N|100|N|N\n"
              "QQQ|Invesco QQQ Trust|Q|N|N|100|Y|N\n"            # ETF flag -> drop
              "TSTZ|Test Issue Co|Q|Y|N|100|N|N\n"              # Test Issue -> drop
              "XYZW|Some Warrant|Q|N|N|100|N|N\n"               # W suffix -> drop
              "File Creation Time: 07/02/2026 05:30|||||||\n")   # footer -> drop
    other = ("ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
             "Test Issue|NASDAQ Symbol\n"
             "IBM|International Business Machines|N|IBM|N|100|N|IBM\n"
             "BRK.B|Berkshire Hathaway Class B|N|BRK.B|N|100|N|BRK.B\n"  # dot -> drop (known limit)
             "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\n"           # ETF -> drop
             "File Creation Time: 07/02/2026 05:30|||||||\n")

    def fake_get(url, timeout=0):
        r = MagicMock()
        r.text = nasdaq if "nasdaqlisted" in url else other
        r.raise_for_status.return_value = None
        return r

    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(dfeed, "CACHE_DIR", Path(tmp)), patch("requests.get", fake_get):
            syms = dfeed.get_universe("full_us", force=True)

    assert "AAPL" in syms and "MSFT" in syms and "IBM" in syms
    assert all(s == s.upper() and "." not in s for s in syms), "symbols must be normalized"
    assert "QQQ" not in syms and "SPY" not in syms, "ETFs must be dropped"
    assert "TSTZ" not in syms and "XYZW" not in syms, "test/warrant must be dropped"
    assert not any(s.startswith("BRK") for s in syms), "dotted class share must be dropped"


def test_incremental_price_cache_appends_delta():
    """Incremental cache: a recent parquet is topped up with only the bars since its last
    date (start=, not period=); a split that re-adjusts the overlap forces a full
    re-baseline instead. Offline (mocked yfinance.download + temp PRICES_DIR)."""
    import tempfile
    from unittest.mock import patch
    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    def _ohlcv(idx, close):
        return pd.DataFrame({"Open": close, "High": [c + 1 for c in close],
                             "Low": [c - 1 for c in close], "Close": close,
                             "Volume": [1000] * len(idx)}, index=pd.DatetimeIndex(idx))

    today = pd.Timestamp.today().normalize()
    cidx = pd.bdate_range(end=today - pd.Timedelta(days=4), periods=15)
    cached = _ohlcv(cidx, [100.0 + i for i in range(15)])
    last = cidx[-1]
    n1, n2 = last + pd.Timedelta(days=1), last + pd.Timedelta(days=2)
    incr_idx = [last, n1, n2]                                   # 1-day overlap + 2 new bars
    incr_a = _ohlcv(incr_idx, [cached.loc[last, "Close"], 200.0, 201.0])   # overlap matches
    incr_b = _ohlcv(incr_idx, [cached.loc[last, "Close"] / 2, 200.0, 201.0])  # halved -> split
    full = _ohlcv(pd.bdate_range(end=today, periods=20), [50.0 + i for i in range(20)])

    calls = []

    def fake_a(sym, **kw):
        calls.append(kw)
        return incr_a

    def fake_b(sym, **kw):
        calls.append(kw)
        return incr_b if "start" in kw else full

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        with patch.object(dfeed, "PRICES_DIR", pdir):
            # (a) small gap, overlap agrees -> delta append, no full refetch
            cached.to_parquet(pdir / "AAPL.parquet")
            calls.clear()
            with patch("yfinance.download", fake_a):
                got = dfeed.get_prices("AAPL", max_age_days=-1)      # -1 skips fresh short-circuit
            assert any("start" in c for c in calls), "should delta-fetch with start="
            assert not any("period" in c for c in calls), "should NOT full-refetch"
            assert len(got) == 17 and n2 in got.index and got.loc[n2, "Close"] == 201.0

            # (b) overlap diverged (split) -> re-baseline via a full period= fetch
            cached.to_parquet(pdir / "AAPL.parquet")               # reset cache
            calls.clear()
            with patch("yfinance.download", fake_b):
                got2 = dfeed.get_prices("AAPL", max_age_days=-1)
            assert any("start" in c for c in calls), "must try incremental first"
            assert any("period" in c for c in calls), "divergence must trigger a full refetch"
            assert len(got2) == 20 and got2["Close"].iloc[0] == 50.0


def _lin_series(segments, start=100.0):
    """Build an OHLCV frame whose Close walks piecewise-linearly through `segments`
    (list of (n_days, end_price)); High=Low=Open=Close so swings are exact/deterministic."""
    import pandas as pd
    closes = [start]
    for days, end in segments:
        s = closes[-1]
        closes += [s + (end - s) * k / days for k in range(1, days + 1)]
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=len(closes))
    c = pd.Series(closes, index=idx)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                         "Volume": 1_000_000}, index=idx)


def test_vcp_detects_tightening_bases():
    """The cockpit detector flags genuine widest-first tightening bases — including the
    shallow ones the vendored detector missed (cc=0)."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    # Textbook VCP: rise, then pullbacks ~16→11→7→5% into a tight base near the high.
    # thr pinned to 0.04 so the adaptive default doesn't shift the deterministic pivot count.
    df = _lin_series([(100, 150), (10, 126), (8, 148), (10, 131.7), (8, 146),
                      (10, 135.8), (8, 144), (10, 136.8), (8, 143)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {}, thr=0.04)
    assert v["is_vcp"] is True, v["pattern_details"]
    assert v["contraction_count"] == 4, v["contraction_count"]
    depths = [c["drawdown_pct"] for c in v["contractions"]]
    assert depths == sorted(depths, reverse=True) and depths[0] > 12 and depths[-1] < 8, depths
    assert v["contractions"][-1]["trough_date"] == max(c["trough_date"] for c in v["contractions"])

    # Shallow tight base (~8→5%) that the vendored detector reported as cc=0. The final leg is
    # longer/calmer than the advance into it, so RMV bottoms in the base (the strict vol gate).
    df2 = _lin_series([(60, 130), (8, 119.6), (8, 128), (12, 121.6), (16, 127)])
    v2 = detect_vcp(df2, float(df2["Close"].iloc[-1]), {}, thr=0.04)
    assert v2["is_vcp"] is True and v2["contraction_count"] >= 2, v2["pattern_details"]


def test_vcp_rejects_noise_and_nobase():
    """Choppy names far from their high and straight-up no-base movers are NOT VCPs, and
    the contraction count stays bounded (no 22-swing noise tail)."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    # Chop that fades ~30% off its high -> not near the high -> not a VCP.
    choppy = _lin_series([(40, 150), (8, 138), (6, 146), (8, 128), (6, 138),
                          (8, 118), (6, 128), (8, 108), (6, 116), (8, 105)])
    v = detect_vcp(choppy, float(choppy["Close"].iloc[-1]), {})
    assert v["is_vcp"] is False and v["contraction_count"] <= 6, v["contraction_count"]

    # Straight-up, no pullback -> no completed peak->trough -> cc 0, not a VCP.
    up = _lin_series([(150, 250)])
    v2 = detect_vcp(up, float(up["Close"].iloc[-1]), {})
    assert v2["is_vcp"] is False and v2["contraction_count"] <= 1, v2["contraction_count"]


def test_vcp_base_does_not_span_a_breakout():
    """The base must be a single consolidation under a flat top: contractions from an OLD
    base (before a big advance) must not be stitched into the current one (the DVA bug)."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    # Base A near ~100 (8%, 5.6%), a +22% advance to ~120, then base B near ~120 (8%, 5%).
    df = _lin_series([(40, 100), (8, 92), (8, 99), (8, 93.5), (8, 98),
                      (10, 120), (8, 110.4), (8, 118), (8, 112), (8, 117)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {}, thr=0.04)
    # Only base B is the current base — base A's ~100 peaks are excluded by the flat-top rule.
    assert v["contraction_count"] == 2, v["contraction_count"]
    assert all(c["peak_price"] > 108 for c in v["contractions"]), \
        [round(c["peak_price"]) for c in v["contractions"]]
    assert v["is_vcp"] is True, v["pattern_details"]


def _range_frame(wide=150, tight=60):
    """OHLCV with real intrabar range (unlike ``_lin_series`` where H=L=C): a volatile body
    (closes oscillate ±2.5) then a near-flat tail (closes ±0.1) that still has intraday range.
    Needed for the TTM squeeze, which fires only when close dispersion falls below the range."""
    import pandas as pd
    close, high, low = [], [], []
    for i in range(wide):
        c = 100.0 + (2.5 if i % 2 else -2.5)
        close.append(c); high.append(c + 1.0); low.append(c - 1.0)
    for i in range(tight):
        c = 100.0 + (0.1 if i % 2 else -0.1)
        close.append(c); high.append(c + 1.5); low.append(c - 1.5)
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=len(close))
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": 1_000_000}, index=idx)


def test_adaptive_threshold_scales_with_volatility():
    """The ZigZag threshold floats with the stock's own volatility (clamped to
    [THR_MIN, THR_MAX]) instead of a fixed 0.04 — a quiet name gets a tighter swing filter,
    a wild one a wider one."""
    from src.stock_screener.cockpit.vcp import _adaptive_threshold, THR_MIN, THR_MAX

    quiet = _lin_series([(200, 108)])                       # ~0.04%/bar drift -> very quiet
    wild = _lin_series([(10, 150), (10, 100), (10, 150), (10, 100),
                        (10, 150), (10, 100), (10, 150), (10, 100)])   # ±40% whipsaws
    t_quiet, t_wild = _adaptive_threshold(quiet), _adaptive_threshold(wild)
    assert THR_MIN <= t_quiet <= THR_MAX, t_quiet
    assert THR_MIN <= t_wild <= THR_MAX, t_wild
    assert t_quiet < t_wild, (t_quiet, t_wild)


def test_bbwp_and_squeeze_indicators():
    """BBWP stays in 0-100 and reads low (a squeeze) as a wide base coils tight; TTM squeeze
    returns a bool Series that is on in the tight tail."""
    from src.stock_screener.cockpit.indicators import (
        bollinger_bandwidth_percentile, ttm_squeeze)

    df = _range_frame()
    bbwp = bollinger_bandwidth_percentile(df).dropna()
    assert len(bbwp), "BBWP produced no values"
    assert float(bbwp.min()) >= 0.0 and float(bbwp.max()) <= 100.0, (bbwp.min(), bbwp.max())
    assert float(bbwp.iloc[-1]) < 40.0, float(bbwp.iloc[-1])   # coiled tail = low percentile

    sq = ttm_squeeze(df)
    assert sq.dtype == bool and len(sq) == len(df)
    assert bool(sq.iloc[-1]) is True, "coiled tail should register a TTM squeeze"


def test_rmv_gate_vetoes_below_pivot_only():
    """RMV semantics (benchmarked): while price is still BELOW the pivot the base should be
    quiet, so a loud tape is vetoed. AT/ABOVE the pivot a breakout IS a burst of movement —
    RMV reads high at exactly the moment a setup completes (the SMBC false-negative class) —
    so it must NOT veto there; structure alone decides."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    base_legs = [(100, 150), (10, 126), (8, 148), (10, 131.7), (8, 146),
                 (10, 135.8), (8, 144), (10, 136.8)]
    calm = _lin_series(base_legs + [(8, 143)])                    # quiet drift below pivot
    loud = _lin_series(base_legs + [(1, 143), (1, 137.5), (1, 143), (1, 137.5),
                                    (1, 143), (1, 137.5), (1, 142), (1, 138)])
    hot = _lin_series(base_legs + [(2, 150)])                     # breakout thrust AT pivot

    vc = detect_vcp(calm, float(calm["Close"].iloc[-1]), {}, thr=0.04)
    vl = detect_vcp(loud, float(loud["Close"].iloc[-1]), {}, thr=0.04)
    vh = detect_vcp(hot, float(hot["Close"].iloc[-1]), {}, thr=0.04)

    assert vc["is_vcp"] is True, vc["pattern_details"]
    # below the pivot with a whipsawing tape -> RMV (or the churned structure) must reject
    assert vl["is_vcp"] is False, (vl["rmv"], vl["pattern_details"])
    assert vl["rmv"] > vc["rmv"], (vl["rmv"], vc["rmv"])
    # at the pivot mid-breakout the RMV burst must NOT veto a valid structure
    assert vh["is_vcp"] is True, (vh["rmv"], vh["pattern_details"])
    assert vh["rmv"] > vc["rmv"], (vh["rmv"], vc["rmv"])
    assert vh["tier"] == "A", (vh["tier"], vh["tier_reason"])


def _ohlc_series(segments, start=100.0, band=0.01):
    """Like ``_lin_series`` but with real intrabar range: High/Low sit ±band around the
    close walk. Needed for adaptive-mode (thr=None) detector tests — H=L=C frames have no
    true range, so they'd false-trigger the dead-tape guard."""
    import pandas as pd
    closes = [start]
    for days, end in segments:
        s = closes[-1]
        closes += [s + (end - s) * k / days for k in range(1, days + 1)]
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=len(closes))
    c = pd.Series(closes, index=idx)
    return pd.DataFrame({"Open": c, "High": c * (1 + band), "Low": c * (1 - band),
                         "Close": c, "Volume": 1_000_000}, index=idx)


def test_vcp_multi_threshold_sees_quiet_taper_after_loud_history():
    """SMBC shape: a formerly-loud stock (±20% swings) whose base tapers 9→7→5.5→4.5%.
    The long-history threshold alone is calibrated to the loud past and cannot see the
    tight legs — the multi-threshold ladder must find them (tier A, adaptive mode)."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 150), (15, 115), (15, 150), (15, 122), (15, 152),
                       (10, 141.5), (8, 152), (8, 145.2), (8, 152.5),
                       (8, 147.2), (8, 153), (8, 149.2), (6, 156)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["is_vcp"] is True, (v["tier"], v["tier_reason"], v["pattern_details"])
    assert v["tier"] == "A", (v["tier"], v["tier_reason"])
    assert v["contraction_count"] >= 3, v["pattern_details"]


def test_vcp_finds_tight_final_leg_after_wide_start():
    """VRA shape: wide early pullbacks (~20%) ending in one tight final leg, price still
    below the pivot. A single history-wide threshold missed the final leg entirely."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 150), (12, 120), (12, 148), (10, 133), (10, 147),
                       (8, 137), (10, 145)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["is_vcp"] is True, (v["tier"], v["tier_reason"], v["pattern_details"])
    assert v["tier"] == "A", (v["tier"], v["tier_reason"])
    depths = [c["drawdown_pct"] for c in v["contractions"]]
    assert depths[-1] < 10.0, depths


def test_vcp_two_day_spike_is_not_a_base():
    """EQ shape: a straight run-up whose only 'pullback' is a violent 1-bar plunge and
    rebound. A 1-bar leg is a junk anchor, not a base — tier C, never a VCP."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(150, 250), (1, 225), (1, 248)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["is_vcp"] is False, v["pattern_details"]
    assert v["tier"] == "C", (v["tier"], v["tier_reason"])


def test_vcp_deal_pinned_stock_is_dead_tape():
    """Deal-stock shape: months of near-zero movement (an acquisition-arb zombie) cannot
    be a live setup — tier C with the dead-tape reason recorded."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 90), (20, 100)], band=0.01)
    import pandas as pd
    flat_idx = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=150)
    flat = pd.DataFrame({"Open": 100.2, "High": 100.25, "Low": 100.15, "Close": 100.2,
                         "Volume": 50_000}, index=flat_idx)
    df = pd.concat([df, flat])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["tier"] == "C", (v["tier"], v["tier_reason"])
    assert "Dead tape" in v["tier_reason"], v["tier_reason"]


def test_vcp_extended_breakout_is_watch_not_review():
    """Extended-breakout shape: a textbook base whose breakout already ran ~15% past the
    pivot is spent — a valid pattern, but tier B (watch), never tier A."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 150), (10, 138), (8, 149), (8, 141.8), (8, 150),
                       (8, 145.5), (15, 175)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["tier"] == "B", (v["tier"], v["tier_reason"])
    assert "extended" in v["tier_reason"].lower() or "past the pivot" in v["tier_reason"], \
        v["tier_reason"]


def test_vcp_benchmark_200_charts():
    """The 200-chart hand-labeled benchmark (see tests/vcp_labels.py for the blind
    protocol). Hard contracts:
      - never-miss: ZERO YES-labeled charts land in tier C;
      - shortlist:  every YES-labeled chart lands in tier A or B;
      - regression floor: tier A captures at least 45 of the 72 YES charts.
    Soft stats (tier sizes, precision) are printed for future tuning."""
    import pandas as pd
    from vcp_labels import LABELS, fixture_filename
    from src.stock_screener.cockpit.vcp import detect_vcp

    fdir = ROOT / "tests" / "fixtures" / "vcp_bench"
    assert fdir.exists(), "benchmark fixtures missing — were tests/fixtures committed?"

    tiers, misses = {}, []
    for t, lab in LABELS.items():
        df = pd.read_parquet(fdir / fixture_filename(t))
        r = detect_vcp(df, float(df["Close"].iloc[-1]), {})
        tiers[t] = r["tier"]
        if lab["label"] == "YES" and r["tier"] == "C":
            misses.append((t, r["tier_reason"]))

    yes = [t for t, v in LABELS.items() if v["label"] == "YES"]
    n_a = sum(1 for t in tiers if tiers[t] == "A")
    n_b = sum(1 for t in tiers if tiers[t] == "B")
    n_c = sum(1 for t in tiers if tiers[t] == "C")
    yes_a = sum(1 for t in yes if tiers[t] == "A")
    yes_b = sum(1 for t in yes if tiers[t] == "B")
    print(f"    benchmark: A={n_a} (YES {yes_a}, precision {yes_a / max(n_a, 1) * 100:.0f}%)"
          f"  B={n_b} (YES {yes_b})  C={n_c}  | YES total {len(yes)}")

    assert not misses, f"never-miss violated — YES charts in tier C: {misses}"
    assert all(tiers[t] in ("A", "B") for t in yes), "a YES chart left tier A/B"
    assert yes_a >= 45, f"tier-A recall regressed: only {yes_a}/{len(yes)} YES in A"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} cockpit tests passed")


if __name__ == "__main__":
    _run_all()
