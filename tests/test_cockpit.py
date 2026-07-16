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


def test_earnings_date_plumbing():
    """The next-earnings date flows yfinance-calendar -> fundamentals dict -> scan
    column/payload -> trade plan; the day-count helper handles past/None/garbage."""
    import datetime as _dt

    import pandas as pd
    from src.stock_screener.cockpit import data_feed
    from src.stock_screener.cockpit.trade import build_buy_plan

    # calendar parsing: dict shape (modern yfinance) — earliest of the 2-day window wins
    class _TkDict:
        calendar = {"Earnings Date": [_dt.date(2026, 8, 3), _dt.date(2026, 7, 30)]}

    assert data_feed._next_earnings_date(_TkDict()) == "2026-07-30"

    # DataFrame shape (older yfinance): an 'Earnings Date' row
    class _TkFrame:
        calendar = pd.DataFrame({0: [pd.Timestamp("2026-07-30")]},
                                index=["Earnings Date"])

    assert data_feed._next_earnings_date(_TkFrame()) == "2026-07-30"

    # empty / broken calendars -> None, never a raise
    class _TkEmpty:
        calendar = {}

    class _TkBoom:
        @property
        def calendar(self):
            raise RuntimeError("offline")

    assert data_feed._next_earnings_date(_TkEmpty()) is None
    assert data_feed._next_earnings_date(_TkBoom()) is None

    # day-count helper (today pinned so the test is deterministic)
    today = pd.Timestamp("2026-07-07")
    assert scan_mod._days_to_earnings({"next_earnings": "2026-07-17"}, today=today) == 10
    assert scan_mod._days_to_earnings({"next_earnings": "2026-07-04"}, today=today) == -3
    assert scan_mod._days_to_earnings({"next_earnings": None}, today=today) is None
    assert scan_mod._days_to_earnings(None, today=today) is None
    assert scan_mod._days_to_earnings({"next_earnings": "garbage"}, today=today) is None

    # through the funnel: the candidates column and the payload both carry the value
    prices, spy, _ = _synthetic_slice()
    soon = (pd.Timestamp.today().normalize() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    def fund_with_date(_t):
        return {"revenue_yoy": 40.0, "eps_yoy": 60.0, "eps_yoy_prev": 50.0,
                "margin_trend": 1.0, "operating_margin": 25.0, "next_earnings": soon}

    res = screen_universe(list(prices), prices, spy, get_fundamentals=fund_with_date,
                          cfg=ScanConfig(min_rs=0.0))
    assert len(res.candidates), "no candidates in the synthetic slice"
    assert "earnings_in" in res.candidates.columns
    # scan computes vs the real 'today'; tolerate the (theoretical) midnight rollover
    assert set(int(v) for v in res.candidates["earnings_in"]) <= {9, 10}
    t0 = res.candidates["ticker"].iloc[0]
    assert res.payloads[t0]["earnings_in"] in (9, 10)

    # and into the trade plan, untouched (build_buy_plan does no date math)
    pl = {t0: {"df": res.payloads[t0]["df"], "levels": res.payloads[t0]["levels"],
               "earnings_in": 5}}
    plan, _ = build_buy_plan([t0], pl, mode="shares", amount=1)
    assert plan and plan[0]["earnings_in"] == 5


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


def test_streamlit_app_renders_offline():
    """Execute app.py through Streamlit's AppTest with run_scan patched to a real,
    offline ScanResult (synthetic fixture). Verifies the whole UI render path —
    regime banner, table, selectbox, chart, Step-2/Step-4 panels — raises nothing."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_streamlit_app_renders_offline (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    import pandas as pd

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()

    # Fundamentals WITH a near-term earnings date, so the populated Step-2 panel
    # (earnings line + ⚠ flag) and the earnings_in table column render, not just
    # their n/a fallbacks.
    soon = (pd.Timestamp.today().normalize() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    def _fund(_t):
        return {"revenue_yoy": 40.0, "eps_yoy": 60.0, "eps_yoy_prev": 50.0,
                "margin_trend": 1.0, "operating_margin": 25.0, "next_earnings": soon}

    result = screen_universe(list(prices), prices, spy, get_fundamentals=_fund,
                             cfg=ScanConfig(min_rs=0.0))
    assert len(result.candidates) >= 1, "fixture must yield >=1 candidate for the app path"

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    # app.py does `from ...scan import run_scan`, resolved at script-exec time, so
    # patching the source attribute before .run() propagates into the app's namespace.
    # Redirect the persisted watchlist to a temp file so the test neither reads nor writes
    # the real data/cockpit/watchlist.json.
    # run_scan fake that actually DRIVES the progress callback — the live-crash pattern:
    # in-scan st.progress calls into an outside slot, then a second (memoized) rerun. Under
    # @st.cache_data this died with CacheReplayClosureError on the rerun (element replay
    # into a vanished block); the session-state memo must survive it, calling the scan once.
    calls = {"n": 0}

    def _fake_scan(*a, progress=None, **kw):
        calls["n"] += 1
        if progress:
            for i in (1, 20, 40):
                progress(i, 40, "SYN")
        return result

    with tempfile.TemporaryDirectory() as _tmp:
        with patch.object(scanmod, "run_scan", side_effect=_fake_scan), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
            at = AppTest.from_file(app_path, default_timeout=60)
            at.run()
            assert not at.exception, f"app raised: {at.exception}"
            at.run()                                     # rerun -> memo hit, no replay
    assert not at.exception, f"app raised on the memoized rerun: {at.exception}"
    assert calls["n"] == 1, f"scan should run once and memoize, ran {calls['n']}x"


def test_watchlist_export_helpers():
    """The watchlist CSV builders: the decision list keeps user order and never drops a
    picked ticker (stale ones survive as ticker-only rows); the OHLCV dump stacks every
    present name long-format with a Ticker column."""
    import pandas as pd
    from src.stock_screener.cockpit.export import (watchlist_list_csv,
                                                   watchlist_ohlcv_csv)

    cand = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"],
                         "tier": ["A", "B", "A"], "pivot": [10.0, 20.0, 30.0]})

    # list CSV takes ENTRIES (dicts and/or bare strings): order follows the watchlist
    # (BBB before AAA); a stale pick (ZZZ, not in the scan) still appears so nothing the
    # user chose is silently lost; the frozen-pivot metadata columns always ride along.
    ents = [{"ticker": "BBB", "judged_pivot": 19.5, "date_added": "2026-07-12",
             "pivot_source": "judged", "note": "n1"},
            "AAA",                                             # bare string still accepted
            {"ticker": "ZZZ"}]
    csv = watchlist_list_csv(cand, ents, columns=["ticker", "tier", "pivot"]).decode()
    lines = csv.strip().splitlines()
    assert lines[0] == "ticker,tier,pivot,judged_pivot,date_added,pivot_source,note"
    order = [ln.split(",")[0] for ln in lines[1:]]
    assert order == ["BBB", "AAA", "ZZZ"], order
    assert lines[1].startswith("BBB,B,20.0,19.5,2026-07-12,judged,n1")
    assert lines[2].split(",")[3] == ""                        # unfrozen -> empty judged_pivot

    # empty candidates -> ticker-only rows + (empty) metadata columns (never raises)
    empty = watchlist_list_csv(pd.DataFrame(), ["AAA", "BBB"]).decode()
    elines = empty.strip().splitlines()
    assert elines[0] == "ticker,judged_pivot,date_added,pivot_source,note"
    assert [ln.split(",")[0] for ln in elines[1:]] == ["AAA", "BBB"]

    # OHLCV CSV: two names stacked, Date + Ticker lead, present names only
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=5)
    mk = lambda base: pd.DataFrame(  # noqa: E731
        {"Open": base, "High": base + 1, "Low": base - 1, "Close": base,
         "Volume": 100}, index=idx)
    payloads = {"AAA": {"df": mk(10.0)}, "BBB": {"df": mk(20.0)}}
    ocsv = watchlist_ohlcv_csv(["AAA", "BBB", "ZZZ"], payloads).decode()
    olines = ocsv.strip().splitlines()
    assert olines[0].split(",")[:2] == ["Date", "Ticker"], olines[0]
    assert len(olines) == 1 + 2 * len(idx)                      # header + 5 bars × 2 names
    assert "ZZZ" not in ocsv                                    # absent name omitted
    assert ",AAA," in ocsv and ",BBB," in ocsv

    assert watchlist_ohlcv_csv([], payloads) == b""             # empty list -> empty bytes


def test_watchlist_persistence():
    """save_watchlist/load_watchlist round-trip ENTRY DICTS (ticker upper/strip, de-duped
    first-seen); a legacy string-array file migrates to unfrozen entries at load; load
    returns [] for missing/corrupt/non-list files."""
    import json as _json
    import tempfile
    from pathlib import Path as _P
    from src.stock_screener.cockpit.export import (save_watchlist, load_watchlist,
                                                   make_entry)

    with tempfile.TemporaryDirectory() as tmp:
        p = _P(tmp) / "sub" / "watchlist.json"                 # parent dir created on save
        assert load_watchlist(p) == []                         # missing file -> []

        # bare strings coerce to unfrozen entries: dedupe + upper + strip preserved
        save_watchlist(p, ["nvda", "MSFT", "nvda", " aapl "])
        assert p.exists()
        got = load_watchlist(p)
        assert [e["ticker"] for e in got] == ["NVDA", "MSFT", "AAPL"]
        assert all(e["judged_pivot"] is None and e["pivot_source"] is None for e in got)

        # entry dicts round-trip the frozen-pivot metadata (pivot rounded to cents)
        ents = [make_entry("ebay", 34.119, date_added="2026-07-12",
                           pivot_source="judged", note="clean base"),
                make_entry("BAP")]
        save_watchlist(p, ents)
        got2 = load_watchlist(p)
        assert got2[0] == {"ticker": "EBAY", "judged_pivot": 34.12,
                           "date_added": "2026-07-12", "pivot_source": "judged",
                           "note": "clean base"}
        assert got2[1]["ticker"] == "BAP" and got2[1]["judged_pivot"] is None

        save_watchlist(p, [])                                  # clearing persists an empty list
        assert load_watchlist(p) == []

        # LEGACY file on disk (raw string array) -> migrated in memory, file untouched
        p.write_text(_json.dumps(["EBAY", "BAP"]), encoding="utf-8")
        legacy = load_watchlist(p)
        assert [e["ticker"] for e in legacy] == ["EBAY", "BAP"]
        assert all(e["judged_pivot"] is None for e in legacy)

        p.write_text("{ not json", encoding="utf-8")           # corrupt -> [] (never raises)
        assert load_watchlist(p) == []
        p.write_text('{"a": 1}', encoding="utf-8")             # valid JSON, not a list -> []
        assert load_watchlist(p) == []


def test_watchlist_entry_normalization():
    """make_entry/_coerce_entry: ticker case/strip, float()-coerced pivot (np.float64 in,
    plain json-safe float out), bad pivots -> None, pivot_source only sticks alongside a
    valid pivot, junk elements dropped."""
    import numpy as np
    from src.stock_screener.cockpit.export import make_entry, _coerce_entry

    e = make_entry(" ebay ", np.float64(34.119), date_added="2026-07-12",
                   pivot_source="judged", note=None)
    assert e == {"ticker": "EBAY", "judged_pivot": 34.12, "date_added": "2026-07-12",
                 "pivot_source": "judged", "note": ""}
    assert type(e["judged_pivot"]) is float                    # not np.float64 (json-safe)

    assert make_entry("") is None and make_entry(None) is None
    for bad in (0, -3, "garbage", float("nan"), None):
        assert make_entry("AAA", bad)["judged_pivot"] is None, bad
    # pivot_source never sticks without a valid pivot, or with an unknown source name
    assert make_entry("AAA", None, pivot_source="judged")["pivot_source"] is None
    assert make_entry("AAA", 10.0, pivot_source="bogus")["pivot_source"] is None
    assert make_entry("AAA", 10.0, pivot_source="auto")["pivot_source"] == "auto"

    assert _coerce_entry("bap")["ticker"] == "BAP"             # legacy bare string
    assert _coerce_entry({"ticker": "x", "judged_pivot": 5})["judged_pivot"] == 5.0
    assert _coerce_entry(42) is None and _coerce_entry(None) is None


def test_watchlist_tickers_projection():
    """watchlist_tickers projects mixed dict/str input to ordered unique upper tickers."""
    from src.stock_screener.cockpit.export import watchlist_tickers
    mixed = [{"ticker": "EBAY", "judged_pivot": 34.12}, "bap", {"ticker": "ebay"},
             42, "", {"ticker": "PECO"}]
    assert watchlist_tickers(mixed) == ["EBAY", "BAP", "PECO"]
    assert watchlist_tickers([]) == [] and watchlist_tickers(None) == []


def test_watchlist_legacy_migration_roundtrip():
    """A legacy string-array file loads as unfrozen entries (the load itself never writes);
    saving what was loaded rewrites the file in the dict schema; re-loading is idempotent."""
    import json as _json
    import tempfile
    from pathlib import Path as _P
    from src.stock_screener.cockpit.export import load_watchlist, save_watchlist

    with tempfile.TemporaryDirectory() as tmp:
        p = _P(tmp) / "watchlist.json"
        p.write_text(_json.dumps(["EBAY", "BAP", "PECO", "EIX"]), encoding="utf-8")
        ents = load_watchlist(p)
        assert [e["ticker"] for e in ents] == ["EBAY", "BAP", "PECO", "EIX"]
        assert isinstance(_json.loads(p.read_text(encoding="utf-8"))[0], str), \
            "load must not rewrite the file"
        save_watchlist(p, ents)                                # first mutation-persist
        raw = _json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(raw[0], dict) and raw[0]["ticker"] == "EBAY"
        assert load_watchlist(p) == ents                       # idempotent


def test_parse_ticker_list():
    """The .txt-upload parser tokenizes on commas AND whitespace/newlines, upper-cases,
    drops blanks, and de-duplicates while preserving first-seen order."""
    from src.stock_screener.cockpit.export import parse_ticker_list

    assert parse_ticker_list("aapl, msft\nnvda,,  tsla ") == ["AAPL", "MSFT", "NVDA", "TSLA"]
    assert parse_ticker_list("MSFT,msft , AAPL") == ["MSFT", "AAPL"]   # case-insensitive dedupe
    assert parse_ticker_list("") == [] and parse_ticker_list("  , \n ") == []
    assert parse_ticker_list("BRK.B goog") == ["BRK.B", "GOOG"]        # dots kept, ws-split


def test_watchlist_add_button_and_download(monkeypatch=None):
    """Through the real app: clicking the ⭐ button adds the charted name to
    session_state['watchlist'], and the sidebar then exposes the two download buttons."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_watchlist_add_button_and_download (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    assert len(result.candidates) >= 1

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    # Redirect the persisted watchlist to a fresh temp file: the app neither reads the real
    # data/cockpit/watchlist.json nor clobbers it when the add click persists.
    with tempfile.TemporaryDirectory() as _tmp:
        with patch.object(scanmod, "run_scan", return_value=result), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
            at = AppTest.from_file(app_path, default_timeout=60)
            at.run()
            assert not at.exception, f"app raised: {at.exception}"
            # _wl() loads from the (nonexistent) temp file on first access -> starts empty
            assert list(at.session_state["watchlist"]) == [], "watchlist should start empty"

            toggle = [b for b in at.button if b.key == "wl_toggle"]
            assert toggle, "watchlist add/remove button missing"
            toggle[0].click().run()
            assert not at.exception, f"app raised after add: {at.exception}"

            wl = list(at.session_state["watchlist"])
            assert len(wl) == 1, f"expected 1 watchlisted entry, got {wl}"
            ent = wl[0]
            assert ent["ticker"] in result.payloads
            # end-to-end freeze: the ⭐ add froze the charted payload's pivot as the
            # judged trigger level, stamped today
            import datetime as _dt
            _pv = result.payloads[ent["ticker"]]["levels"]["pivot"]
            assert ent["judged_pivot"] == round(float(_pv), 2), (ent, _pv)
            assert ent["pivot_source"] == "judged"
            assert ent["date_added"] == _dt.date.today().isoformat()
            # the add persisted to the temp file -> loading it back returns the same entry
            from src.stock_screener.cockpit.export import load_watchlist
            assert load_watchlist(cache.WATCHLIST_JSON) == wl, "add did not persist to disk"
    # With a non-empty watchlist the sidebar builds both download buttons; that it reran
    # without raising (asserted above) means the CSV builders ran cleanly on real payloads.


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


def test_stop_is_valid():
    """A protective sell-stop is valid only strictly below the reference price."""
    from src.stock_screener.cockpit.trade import stop_is_valid
    assert stop_is_valid(92.0, 100.0) is True
    for bad in [(100.0, 100.0), (101.0, 100.0), (0.0, 100.0), (None, 100.0), (50.0, None)]:
        assert stop_is_valid(*bad) is False, bad

def test_submit_buy_plan_stop_logic():
    """submit_buy_plan against a fake Alpaca client: an already-held name becomes a GTC
    stop-only order that REPLACES its open stop; a fresh name becomes an OTO buy+stop;
    attach_stop=False yields a naked buy (and skips held names); an invalid stop (>= price)
    is skipped, no order. Cases D/E cover Minervini's never-lower-a-stop ratchet: an existing
    HIGHER stop is kept (no order), a LOWER one is replaced upward."""
    from src.stock_screener.cockpit import trade
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
    from alpaca.trading.enums import OrderSide, OrderClass, OrderType, TimeInForce

    class _Acct:
        equity = "100000"; cash = "100000"; account_number = "PA000123"

    class _Pos:
        def __init__(self, symbol, qty):
            self.symbol, self.qty = symbol, qty

    class _Asset:
        tradable = True

    class _Order:
        def __init__(self, oid, symbol, otype, stop_price=None):
            self.id, self.symbol, self.type = oid, symbol, otype
            self.side = OrderSide.SELL
            self.stop_price = stop_price

    class _Resp:
        def __init__(self, oid):
            self.id = oid

    class FakeClient:
        def __init__(self, positions=None, open_orders=None):
            self._positions = positions or {}       # {sym: qty_str}
            self._open = open_orders or {}          # {sym: [_Order, ...]}
            self.submitted, self.cancelled, self._n = [], [], 0

        def get_account(self):
            return _Acct()

        def get_all_positions(self):
            return [_Pos(s, q) for s, q in self._positions.items()]

        def get_asset(self, t):
            return _Asset()

        def get_orders(self, filter=None):
            out = []
            for s in (getattr(filter, "symbols", None) or []):
                out.extend(self._open.get(s, []))
            return out

        def cancel_order_by_id(self, oid):
            self.cancelled.append(str(oid))

        def submit_order(self, order_data=None):
            self.submitted.append(order_data)
            self._n += 1
            return _Resp(f"id-{self._n}")

    def _entry(t, shares, price, stop):
        return {"ticker": t, "shares": shares, "price": price, "pivot": price,
                "est_value": round(shares * price, 2), "extended": False, "stop_price": stop}

    def _run(plan, attach, fake):
        orig = trade._connect_paper
        trade._connect_paper = lambda: (fake, True)
        try:
            return trade.submit_buy_plan(plan, attach_stop=attach)
        finally:
            trade._connect_paper = orig

    # --- A: held name (40 sh, open stop of UNKNOWN price) + a fresh name, attach on -------
    # No readable stop_price -> can't ratchet -> replace with the new GTC stop.
    fa = FakeClient(positions={"HELD": "40"},
                    open_orders={"HELD": [_Order("old-1", "HELD", OrderType.STOP)]})
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
                    open_orders={"HELD": [_Order("hi-1", "HELD", OrderType.STOP,
                                                 stop_price=98.0)]})
    outD = _run([_entry("HELD", 10, 100.0, 95.0)], True, fd)["results"][0]
    assert outD["status"] == "stop_kept"
    assert outD["stop_price"] == 98.0                    # kept the higher existing stop
    assert not fd.submitted and not fd.cancelled         # nothing placed, nothing cancelled

    # --- E: held name with an existing LOWER stop -> RAISE (cancel old, place GTC at new) --
    fe = FakeClient(positions={"HELD": "40"},
                    open_orders={"HELD": [_Order("lo-1", "HELD", OrderType.STOP,
                                                 stop_price=90.0)]})
    outE = _run([_entry("HELD", 10, 100.0, 95.0)], True, fe)["results"][0]
    assert outE["status"] == "stop_only" and outE["stop_price"] == 95.0
    assert "lo-1" in fe.cancelled                        # replaced the lower stop
    ereq = [r for r in fe.submitted if isinstance(r, StopOrderRequest)][0]
    assert float(ereq.stop_price) == 95.0 and ereq.time_in_force == TimeInForce.GTC

    # --- F: held name, existing stop EQUAL to the new stop -> kept (no churn) --------------
    ff = FakeClient(positions={"HELD": "40"},
                    open_orders={"HELD": [_Order("eq-1", "HELD", OrderType.STOP,
                                                 stop_price=95.0)]})
    outF = _run([_entry("HELD", 10, 100.0, 95.0)], True, ff)["results"][0]
    assert outF["status"] == "stop_kept" and not ff.submitted and not ff.cancelled


def test_trade_plan_preview_renders_stop_controls():
    """With a trade plan seeded in session_state, the paper-trade preview renders the new
    attach-stop toggle and a per-ticker editable stop number_input (keyed by the build nonce),
    without raising — exercises the preview loop + stop_is_valid call path in app.py."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trade_plan_preview_renders_stop_controls (AppTest unavailable: {e})")
        return
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with patch.object(scanmod, "run_scan", return_value=result):
        at = AppTest.from_file(app_path, default_timeout=60)
        at.session_state["watchlist"] = [                # non-empty -> trade section renders
            {"ticker": "AAA", "judged_pivot": None, "date_added": None,
             "pivot_source": None, "note": ""}]
        at.session_state["trade_build_n"] = 1
        at.session_state["trade_mode"] = "Risk % to stop"     # exercise the risk-mode UI branch
        at.session_state["trade_plan"] = {
            # capped=True exercises the ⚠︎ capped flag + footnote; the valid stop drives the
            # live "risk to stop" caption.
            "plan": [{"ticker": "AAA", "shares": 100, "price": 100.0, "pivot": 100.0,
                      "est_value": 10000.0, "extended": False, "capped": True,
                      "stop_price": 92.5, "earnings_in": None}],
            "skipped": [],
            "account": {"account_number": "PA000123", "equity": 100000.0,
                        "using_dedicated": True},
            "build_ts": 1}
        at.run()
        assert not at.exception, f"app raised: {at.exception}"

    stops = [n for n in at.number_input if n.key == "stop_AAA_1"]
    assert stops, "per-ticker stop number_input did not render"
    assert stops[0].value == 92.5, f"stop should default to computed value, got {stops[0].value}"


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


# --------------------------------------------------------------------------- #
# Positions page (stop management)
# --------------------------------------------------------------------------- #
def _pos_fakes():
    """Build (Client, _Pos, _Order) fakes for the positions/re-arm tests. The Client's
    get_orders honors filter.symbols=None -> ALL open orders (the batched query rearm/fetch use);
    _Pos carries the alpaca-py Position P&L attrs; _Order carries symbol/type/stop_price."""
    from alpaca.trading.enums import OrderSide, OrderType

    class _Acct:
        equity = "50000"; cash = "10000"; account_number = "PA00SZOE"

    class _Pos:
        def __init__(self, symbol, qty, **kw):
            self.symbol, self.qty = symbol, str(qty)
            for k, v in kw.items():
                setattr(self, k, v)

    class _Order:
        def __init__(self, oid, symbol, stop_price, otype=None):
            self.id, self.symbol = oid, symbol
            self.type = otype or OrderType.STOP
            self.stop_price, self.side = stop_price, OrderSide.SELL

    class _Resp:
        def __init__(self, oid):
            self.id = oid

    class Client:
        def __init__(self, positions, open_orders=None):
            self._positions = positions
            self._open = open_orders or {}
            self.submitted, self.cancelled, self._n = [], [], 0

        def get_account(self):
            return _Acct()

        def get_all_positions(self):
            return list(self._positions)

        def get_orders(self, filter=None):
            syms = getattr(filter, "symbols", None)
            out = []
            if syms:
                for s in syms:
                    out.extend(self._open.get(s, []))
            else:                                     # batched query: every open order
                for lst in self._open.values():
                    out.extend(lst)
            return out

        def cancel_order_by_id(self, oid):
            self.cancelled.append(str(oid))

        def submit_order(self, order_data=None):
            self.submitted.append(order_data)
            self._n += 1
            return _Resp(f"id-{self._n}")

    return Client, _Pos, _Order


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

    orig_conn, orig_gmp = trade._connect_paper, data_feed.get_many_prices
    trade._connect_paper = lambda: (client, True)
    data_feed.get_many_prices = lambda syms, **kw: frames
    try:
        out = trade.fetch_positions()
    finally:
        trade._connect_paper, data_feed.get_many_prices = orig_conn, orig_gmp

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


def test_positions_page_renders():
    """The Positions page loads and renders with an offline (patched) fetch_positions — no
    network, no re-arm click (covered by the rearm unit test)."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_positions_page_renders (AppTest unavailable: {e})")
        return
    from unittest.mock import patch
    from src.stock_screener.cockpit import trade

    offline = {
        "account": {"account_number": "PA00SZOE", "equity": 50000.0, "cash": 10000.0,
                    "using_dedicated": True, "positions_count": 1, "total_unrealized_pl": 300.0},
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
    with patch.object(trade, "fetch_positions", return_value=offline):
        at = AppTest.from_file(page, default_timeout=60)
        at.run()
    assert not at.exception, f"positions page raised: {at.exception}"
    # The re-arm button only renders after the positions loop, so its presence proves the page
    # rendered end-to-end with the offline holding (and didn't st.stop() early).
    assert any("Re-arm" in str(getattr(b, "label", "")) for b in at.button), \
        "positions page did not render the re-arm control"


# --------------------------------------------------------------------------- #
# Trade journal ("know your numbers")
# --------------------------------------------------------------------------- #
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


def test_journal_page_renders():
    """The Journal page loads and renders its stats tiles + tables with an offline (patched)
    fetch_order_fills — no network."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_journal_page_renders (AppTest unavailable: {e})")
        return
    from unittest.mock import patch
    from src.stock_screener.cockpit import trade

    offline = {
        "account": {"account_number": "PA00SZOE", "equity": 50000.0, "cash": 10000.0,
                    "using_dedicated": True},
        "fills": [
            {"symbol": "AAA", "side": "buy", "qty": 10.0, "price": 100.0,
             "time": "2026-06-01T14:30:00Z", "order_id": "1",
             "client_order_id": "SEPAoto-AAA-1"},
            {"symbol": "AAA", "side": "sell", "qty": 10.0, "price": 111.0,
             "time": "2026-06-10T14:30:00Z", "order_id": "2",
             "client_order_id": "SEPAstop-AAA-2"},
            {"symbol": "BBB", "side": "buy", "qty": 5.0, "price": 20.0,
             "time": "2026-06-05T14:30:00Z", "order_id": "3",
             "client_order_id": "SEPAcockpit-BBB-3"},
        ],
    }
    page = str(ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "3_Journal.py")
    with patch.object(trade, "fetch_order_fills", return_value=offline):
        at = AppTest.from_file(page, default_timeout=60)
        at.run()
    assert not at.exception, f"journal page raised: {at.exception}"
    labels = [str(getattr(m, "label", "")) for m in at.metric]
    assert any("Batting" in x for x in labels), f"stats tiles missing: {labels}"
    # 1 closed winner out of 1 -> batting 100%; the open BBB episode must not enter the stats
    batting = [m for m in at.metric if "Batting" in str(getattr(m, "label", ""))][0]
    assert str(batting.value) == "100%", batting.value


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

    # (c) TODAY's bar is PROVISIONAL while the session runs — its close moving between
    # intraday fetches is NOT a split, so it must never re-baseline (the merge still takes
    # the newest bar); the SAME divergence on a settled prior day still re-baselines.
    tidx = pd.DatetimeIndex(list(pd.bdate_range(end=today - pd.Timedelta(days=3),
                                                periods=9)) + [today])
    base = _ohlcv(tidx, [100.0] * 10)
    moved_today = _ohlcv(pd.DatetimeIndex([today]), [107.0])       # +7% vs cached today-bar
    m, full = dfeed._merge_incremental(base, moved_today, "2y")
    assert full is False, "a moving TODAY bar must not trigger a re-baseline"
    assert float(m.loc[today, "Close"]) == 107.0                   # newest provisional bar wins
    moved_prior = _ohlcv(pd.DatetimeIndex([tidx[-2]]), [55.0])     # settled day -> real split
    _, full2 = dfeed._merge_incremental(base, moved_prior, "2y")
    assert full2 is True, "divergence on a settled day must still re-baseline"


# --------------------------------------------------------------------------- #
# Nightly EOD trigger check (triggers.py — pure)
# --------------------------------------------------------------------------- #
def _trigger_frame(end, closes, vols=None):
    """Daily OHLCV ending exactly at ``end`` for deterministic stale-bar checks."""
    import pandas as pd
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=len(closes))
    vols = vols if vols is not None else [1000] * len(closes)
    return pd.DataFrame({"Open": closes, "High": [c * 1.01 for c in closes],
                         "Low": [c * 0.99 for c in closes], "Close": closes,
                         "Volume": vols}, index=idx)


def test_check_triggers_pure():
    """The nightly gate: close above the FROZEN pivot on >=1.5x 50-day volume, today's bar
    only. Flat volume blocks a trigger, extended (> pivot*1.05) files under don't-chase
    even when price+volume cleared, a stale bar never fires, below-pivot is a watch, and
    the earnings flag + summary lists come through."""
    from src.stock_screener.cockpit.export import make_entry
    from src.stock_screener.cockpit.triggers import check_one, check_triggers

    TODAY = "2026-07-10"                                    # a Friday
    flat = [100.0] * 60
    spike_vol = [1000] * 59 + [3000]                        # 3x the prior 50-day average

    def E(t, pivot):
        return make_entry(t, pivot, date_added="2026-07-01", pivot_source="judged")

    # (a) close 100 > pivot 98 on 3x volume, bar dated today -> TRIGGERED. After the
    # close (16:30) the session is fully elapsed, so pace == the plain 50-day ratio.
    r = check_one(E("TRG", 98.0), _trigger_frame(TODAY, flat, spike_vol), None,
                  today=TODAY, now=f"{TODAY} 16:30")
    assert r["status"] == "triggered" and r["triggered"] is True, r
    assert r["close_above_pivot"] and r["volume_confirmed"] and not r["stale"]
    assert abs(r["volume_ratio_50"] - 3.0) < 1e-9
    assert abs(r["volume_pace"] - 3.0) < 1e-9            # elapsed=1.0 -> pace == ratio
    assert abs(r["pct_from_pivot"] - (100.0 / 98.0 - 1) * 100) < 0.01

    # intraday pace: at 12:45 half the 09:30-16:00 session has elapsed -> the same
    # volume reads at DOUBLE pace ("running hot for the time of day"); the gate is
    # untouched (still the plain ratio).
    r_mid = check_one(E("TRG", 98.0), _trigger_frame(TODAY, flat, spike_vol), None,
                      today=TODAY, now=f"{TODAY} 12:45")
    assert abs(r_mid["volume_pace"] - 6.0) < 1e-9        # 3.0 / 0.5
    assert r_mid["triggered"] is True                    # gate = actual ratio, not pace

    # (b) same but flat volume -> price cleared, volume didn't -> watch, not triggered
    r2 = check_one(E("NOV", 98.0), _trigger_frame(TODAY, flat), None, today=TODAY)
    assert r2["status"] == "watch" and r2["triggered"] is False
    assert r2["close_above_pivot"] is True and r2["volume_confirmed"] is False

    # (c) extended: close 100 vs pivot 90 (+11%) on volume -> don't chase beats triggered
    r3 = check_one(E("EXT", 90.0), _trigger_frame(TODAY, flat, spike_vol), None, today=TODAY)
    assert r3["status"] == "extended" and r3["extended"] is True

    # (d) stale bar (run date is the next business day) -> never fires, and pace is
    # meaningless off a stale bar -> None
    r4 = check_one(E("STL", 98.0), _trigger_frame(TODAY, flat, spike_vol), None,
                   today="2026-07-13", now="2026-07-13 12:00")
    assert r4["status"] == "stale" and r4["stale"] is True and r4["triggered"] is False
    assert r4["volume_pace"] is None

    # (e) below the pivot -> watch, negative distance
    r5 = check_one(E("BLW", 105.0), _trigger_frame(TODAY, flat), None, today=TODAY)
    assert r5["status"] == "watch" and r5["pct_from_pivot"] < 0

    # earnings flag rides in from the fundamentals dict (10 days out -> soon)
    r6 = check_one(E("ERN", 98.0), _trigger_frame(TODAY, flat),
                   {"next_earnings": "2026-07-20"}, today=TODAY)
    assert r6["earnings_in"] == 10 and r6["earnings_soon"] is True

    # full report: summary buckets by STATUS; missing frame -> no_data; spy note present
    entries = [E("TRG", 98.0), E("EXT", 90.0), E("BLW", 105.0),
               make_entry("GONE"), make_entry("UNF")]      # GONE: no frame; UNF: no pivot
    prices = {"TRG": _trigger_frame(TODAY, flat, spike_vol),
              "EXT": _trigger_frame(TODAY, flat, spike_vol),
              "BLW": _trigger_frame(TODAY, flat),
              "UNF": _trigger_frame(TODAY, flat)}
    spy = _trigger_frame(TODAY, [300 + i * 0.5 for i in range(260)])
    rep = check_triggers(entries, prices, spy=spy, today=TODAY, now=f"{TODAY} 11:00")
    s = rep["summary"]
    assert s["n"] == 5
    assert s["triggered"] == ["TRG"] and s["extended"] == ["EXT"]
    assert s["no_data"] == ["GONE"] and s["no_pivot"] == ["UNF"]
    assert rep["all_stale"] is False and rep["date"] == TODAY
    assert rep["spy"] is not None and "phase" in rep["spy"]
    assert rep["intraday"] is True                       # fresh bar + mid-session run

    # the same report generated after the close is NOT intraday (settled bar)
    rep_eod = check_triggers(entries, prices, spy=spy, today=TODAY, now=f"{TODAY} 16:30")
    assert rep_eod["intraday"] is False

    # all-stale run (holiday): report still builds, flagged, nothing triggered, and a
    # mid-session clock does NOT make a stale-bar report "intraday"
    rep2 = check_triggers([E("TRG", 98.0)], {"TRG": _trigger_frame(TODAY, flat, spike_vol)},
                          today="2026-07-13", now="2026-07-13 12:00")
    assert rep2["all_stale"] is True and rep2["summary"]["triggered"] == []
    assert rep2["intraday"] is False


def test_freeze_missing_pivots():
    """Freeze-on-first-sight: an unfrozen entry with a >=200-row frame gets the scan-chain
    pivot recorded (source 'auto', dated the run day) WITHOUT mutating the input; a short
    frame stays unfrozen (-> no_pivot at check time); a judged entry is never touched."""
    from src.stock_screener.cockpit.export import make_entry
    from src.stock_screener.cockpit.triggers import (check_one, compute_scan_pivot,
                                                     freeze_missing_pivots)

    TODAY = "2026-07-10"
    up = _trigger_frame(TODAY, [100.0 + i * 0.3 for i in range(260)])   # 260-row uptrend
    short = _trigger_frame(TODAY, [100.0] * 60)                          # < 200 rows
    prices = {"UP": up, "SHORT": short}

    expected = compute_scan_pivot(up)
    assert expected and 0 < expected <= float(up["High"].max()) * 1.05
    assert compute_scan_pivot(short) is None                # too short -> None, no raise
    assert compute_scan_pivot(None) is None

    entries = [make_entry("UP"), make_entry("SHORT"),
               make_entry("HELD", 55.5, date_added="2026-07-01", pivot_source="judged")]
    out, frozen = freeze_missing_pivots(entries, prices, today=TODAY)

    assert frozen == ["UP"]
    by = {e["ticker"]: e for e in out}
    assert by["UP"]["judged_pivot"] == round(expected, 2)
    assert by["UP"]["pivot_source"] == "auto" and by["UP"]["date_added"] == TODAY
    assert by["SHORT"]["judged_pivot"] is None              # couldn't compute -> unchanged
    assert by["HELD"] == entries[2]                         # judged entry untouched
    assert entries[0]["judged_pivot"] is None, "input entries must not be mutated"

    # a still-unfrozen entry evaluates to no_pivot (skipped), never a crash
    r = check_one(by["SHORT"], short, None, today=TODAY)
    assert r["status"] == "no_pivot" and r["triggered"] is False

    # REGRESSION (the EBAY 118.98-vs-111.86 bug): an auto-frozen pivot must equal the
    # pivot the APP shows for the same frame — compute_scan_pivot must replicate
    # screen_universe's chain INCLUDING detect_vcp; without the VCP arg, detect_breakout
    # yields no level and the pivot silently falls back to the (higher) 52-week high.
    prices_syn, spy_syn, _ = _synthetic_slice()
    res = screen_universe(list(prices_syn), prices_syn, spy_syn, get_fundamentals=None,
                          cfg=ScanConfig(min_rs=0.0))
    assert len(res.payloads), "no synthetic candidates to check"
    for t in list(res.payloads)[:5]:
        pl = res.payloads[t]
        got = compute_scan_pivot(pl["df"])
        assert got is not None and abs(got - pl["levels"]["pivot"]) < 1e-6, \
            (t, got, pl["levels"]["pivot"])


def test_trigger_report_roundtrip():
    """save_trigger_report/load_latest_trigger_report: dated files, newest parseable wins,
    a corrupt newest file falls back to the older one, missing dir -> None."""
    import tempfile
    from src.stock_screener.cockpit.triggers import (format_report,
                                                     load_latest_trigger_report,
                                                     save_trigger_report)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "triggers"
        assert load_latest_trigger_report(d) is None        # missing dir -> None
        r1 = {"schema": 1, "date": "2026-07-09", "names": [], "summary": {"n": 0}}
        r2 = {"schema": 1, "date": "2026-07-10", "names": [], "summary": {"n": 0}}
        p1 = save_trigger_report(r1, d)
        assert p1.name == "triggers_2026-07-09.json" and p1.exists()
        save_trigger_report(r2, d)
        assert load_latest_trigger_report(d)["date"] == "2026-07-10"
        # corrupt the newest -> the loader falls back to the older parseable file
        (d / "triggers_2026-07-11.json").write_text("{ not json", encoding="utf-8")
        assert load_latest_trigger_report(d)["date"] == "2026-07-10"
        # the console renderer never chokes on a minimal report (ASCII path)
        assert "TRIGGER CHECK" in format_report(r2)


def test_eod_trigger_cli_offline():
    """The CLI end-to-end, in-process and offline: patched data feed + temp watchlist/
    report paths. main() returns 0, writes the dated report, and AUTO-FREEZES the
    unfrozen entry's pivot back into watchlist.json; --no-write leaves both untouched."""
    import json as _json
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import cache, eod_trigger
    from src.stock_screener.cockpit.export import load_watchlist, save_watchlist

    TODAY = "2026-07-10"
    up = _trigger_frame(TODAY, [100.0 + i * 0.3 for i in range(260)])
    spy = _trigger_frame(TODAY, [300 + i * 0.5 for i in range(260)])

    def fake_prices(tickers, **kw):
        assert kw.get("max_age_days") == 0.0, "nightly run must use the top-up path"
        return {t: (spy if t == "SPY" else up) for t in tickers}

    with tempfile.TemporaryDirectory() as tmp:
        wl = Path(tmp) / "watchlist.json"
        trg = Path(tmp) / "triggers"
        save_watchlist(wl, ["UPUP"])                       # one legacy, unfrozen name
        with patch.object(cache, "WATCHLIST_JSON", wl), \
                patch.object(cache, "TRIGGERS_DIR", trg), \
                patch.object(eod_trigger.data_feed, "get_many_prices", fake_prices), \
                patch.object(eod_trigger.data_feed, "get_fundamentals",
                             lambda t, **kw: {"next_earnings": "2026-07-20"}):
            # --no-write: report printed only, nothing lands on disk
            assert eod_trigger.main(["--date", TODAY, "--no-write"]) == 0
            assert not trg.exists() or not list(trg.glob("*.json"))
            assert load_watchlist(wl)[0]["judged_pivot"] is None

            # real run: report written, pivot auto-frozen into the watchlist file
            assert eod_trigger.main(["--date", TODAY]) == 0
            rep = _json.loads((trg / f"triggers_{TODAY}.json").read_text(encoding="utf-8"))
            assert rep["date"] == TODAY and rep["summary"]["n"] == 1
            assert rep["summary"]["auto_frozen"] == ["UPUP"]
            assert rep["names"][0]["earnings_in"] == 10
            ent = load_watchlist(wl)[0]
            assert ent["judged_pivot"] and ent["pivot_source"] == "auto"
            assert rep["names"][0]["judged_pivot"] == ent["judged_pivot"]


def test_trigger_report_sidebar_renders():
    """The app's sidebar surfaces the latest trigger report (canned file in a temp
    TRIGGERS_DIR): the date and a triggered name appear in the rendered output."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trigger_report_sidebar_renders (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    from src.stock_screener.cockpit.triggers import save_trigger_report
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))

    canned = {
        "schema": 1, "date": "2026-07-10",
        "generated_at": "2026-07-10T14:31:05-04:00",       # -> "14:31" in the header
        "spy": {"phase": 2, "phase_name": "Stage 2 - Advancing", "trend": "Bullish"},
        "all_stale": False, "intraday": True,
        "names": [
            {"ticker": "TRGX", "status": "triggered", "judged_pivot": 34.12,
             "pivot_source": "judged", "close": 35.0, "volume_ratio_50": 2.1,
             "volume_pace": 2.8, "triggered": True, "earnings_soon": True,
             "earnings_in": 12},
            {"ticker": "AUTX", "status": "watch", "judged_pivot": 50.0,
             "pivot_source": "auto", "close": 48.0, "volume_ratio_50": 0.9},
            {"ticker": "NOPX", "status": "no_pivot"},      # sparse row must render too
        ],
        "summary": {"n": 3, "triggered": ["TRGX"], "extended": [], "stale": [],
                    "earnings_soon": ["TRGX"], "no_data": [], "no_pivot": ["NOPX"],
                    "auto_frozen": []},
    }
    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp:
        trg = Path(_tmp) / "triggers"
        save_trigger_report(canned, trg)
        with patch.object(scanmod, "run_scan", return_value=result), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", trg):
            at = AppTest.from_file(app_path, default_timeout=60)
            at.run()
    assert not at.exception, f"app raised: {at.exception}"
    rendered = " ".join(str(getattr(m, "value", ""))
                        for m in list(at.markdown) + list(getattr(at, "caption", [])))
    assert "2026-07-10" in rendered, "report date not rendered"
    assert "14:31" in rendered, "check time (from generated_at) not rendered"
    assert "TRGX" in rendered and "triggered" in rendered, "triggered name not rendered"
    assert "pace 2.8" in rendered, "intraday volume pace not rendered"


def test_get_many_prices_max_age_days():
    """The new ``max_age_days`` knob on get_many_prices: a 3-day-old parquet is served
    as-is under ``max_age_days=7`` (yfinance never touched), but leaves the gate under the
    1.0 default and goes through the incremental top-up — where a failed fetch degrades to
    the untouched cache instead of raising. Offline (mocked yfinance + temp PRICES_DIR)."""
    import os
    import tempfile
    import time as _time
    from unittest.mock import patch
    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=3),
                         periods=60)
    closes = [100.0 + i for i in range(60)]
    cached = pd.DataFrame({"Open": closes, "High": [c + 1 for c in closes],
                           "Low": [c - 1 for c in closes], "Close": closes,
                           "Volume": [1000] * 60}, index=idx)

    calls = []

    def fake_download(*a, **kw):
        calls.append(kw)
        return pd.DataFrame()                              # failed/empty fetch

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        with patch.object(dfeed, "PRICES_DIR", pdir), patch("yfinance.download",
                                                            fake_download):
            path = pdir / "AAPL.parquet"
            cached.to_parquet(path)
            old = _time.time() - 3 * 86400                 # mtime: 3 days old
            os.utime(path, (old, old))

            # relaxed gate: served straight from the cache, no network attempt
            got = dfeed.get_many_prices(["AAPL"], max_age_days=7.0, pause=0, retries=0)
            assert not calls, "cache hit must not touch yfinance"
            assert len(got["AAPL"]) == 60

            # default gate (1.0): leaves the gate -> incremental top-up attempted; the
            # empty fetch degrades to the untouched cache (no raise, no data loss)
            got2 = dfeed.get_many_prices(["AAPL"], pause=0, retries=0)
            assert calls, "default gate should attempt the incremental fetch"
            assert len(got2["AAPL"]) == 60
            assert float(got2["AAPL"]["Close"].iloc[-1]) == float(cached["Close"].iloc[-1])


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
    assert vh["tier"] == "A", (vh["tier"], vh["pattern_details"])


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
    assert v["is_vcp"] is True, (v["tier"], v["pattern_details"])
    assert v["tier"] == "A", (v["tier"], v["pattern_details"])
    assert v["contraction_count"] >= 3, v["pattern_details"]


def test_vcp_finds_tight_final_leg_after_wide_start():
    """VRA shape: wide early pullbacks (~20%) ending in one tight final leg, price still
    below the pivot. A single history-wide threshold missed the final leg entirely."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 150), (12, 120), (12, 148), (10, 133), (10, 147),
                       (8, 137), (10, 145)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["is_vcp"] is True, (v["tier"], v["pattern_details"])
    assert v["tier"] == "A", (v["tier"], v["pattern_details"])
    depths = [c["drawdown_pct"] for c in v["contractions"]]
    assert depths[-1] < 10.0, depths


def test_vcp_two_day_spike_is_not_a_base():
    """EQ shape: a straight run-up whose only 'pullback' is a violent 1-bar plunge and
    rebound. A 1-bar leg is a junk anchor, not a base — tier C, never a VCP."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(150, 250), (1, 225), (1, 248)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["is_vcp"] is False, v["pattern_details"]
    assert v["tier"] == "C", (v["tier"], v["pattern_details"])


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
    assert v["tier"] == "C", (v["tier"], v["pattern_details"])
    assert "Dead tape" in v["pattern_details"], v["pattern_details"]


def test_vcp_extended_breakout_is_watch_not_review():
    """Extended-breakout shape: a textbook base whose breakout already ran ~15% past the
    pivot is spent — a valid pattern, but tier B (watch), never tier A."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 150), (10, 138), (8, 149), (8, 141.8), (8, 150),
                       (8, 145.5), (15, 175)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["tier"] == "B", (v["tier"], v["pattern_details"])
    # extended = a VALID pattern demoted to watch (tier_reason text removed 2026-07-13; the
    # is_vcp True + tier B pair is what distinguishes "extended" from "still forming")
    assert v["is_vcp"] is True, (v["tier"], v["pattern_details"])


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
            misses.append((t, r["pattern_details"]))

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
