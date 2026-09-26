"""Cockpit tests — the yfinance data layer: universe, price cache, incremental top-ups, EDGAR/fundamentals.

Runs standalone (`python tests/cockpit/test_data_feed.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_data_feed_isolated_from_vendored_data_layer():
    """A fresh interpreter importing cockpit.data_feed AND cockpit.scan must reach NOTHING
    of the vendored package outside ``.screening``, and must not pull SQLAlchemy.

    Item 20: a guarded ``.screener`` import silently succeeded whenever SQLAlchemy was
    installed (it is a transitive dep of wrds), dragging the whole dead live-fetch layer
    into every cockpit process. Those modules have since been deleted outright, so this
    asserts the SHAPE rather than their old names — anything re-vendored under
    ``minervini_screener`` that is not a pure rule module has to fail here."""
    code = (
        "import sys\n"
        "import src.stock_screener.cockpit.data_feed\n"
        "import src.stock_screener.cockpit.scan\n"
        "V = 'src.stock_screener.minervini_screener'\n"
        "bad = [m for m in sys.modules\n"
        "       if (m.startswith(V + '.') and not m.startswith(V + '.screening'))\n"
        "       or m == 'sqlalchemy' or m.startswith('sqlalchemy.')]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, f"isolated import failed:\n{out.stdout}\n{out.stderr}"
    assert "OK" in out.stdout


def test_synthetic_fixture_isolated_from_engine_chain():
    """A fresh interpreter importing backtest_daily.synthetic_provider — the cockpit
    suites' offline price fixture, and therefore part of the runtime image — must NOT
    pull the backtest engine. ``metrics.py`` imports ``momentum_lib`` and
    ``ml_stock_prediction.backtest_lib``, two parked research tracks; while the package
    __init__ re-exported the engine, importing the fixture dragged both into the Pi's
    image to satisfy one test helper. This subprocess keeps the whitelist honest."""
    code = (
        "import sys\n"
        "import src.stock_screener.backtest_daily.synthetic_provider\n"
        "bad = [m for m in ('src.stock_screener.backtest_daily.engine',\n"
        "                   'src.stock_screener.backtest_daily.metrics',\n"
        "                   'src.stock_screener.momentum_lib',\n"
        "                   'src.ml_stock_prediction.backtest_lib')\n"
        "       if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, f"isolated import failed:\n{out.stdout}\n{out.stderr}"
    assert "OK" in out.stdout


def test_edgar_backfill_parsing():
    """The SEC EDGAR backfill: quarterly vs annual bucketing by filing duration, amended
    filings winning by 'filed' date, the revenue tag fallback chain, date-matched YoY
    math, FY EPS growth and the 3-quarter acceleration flag — against canned
    company-facts JSON, fully offline (mocked requests + temp EDGAR_DIR)."""
    import tempfile
    from unittest.mock import MagicMock, patch

    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    def q(end, val, filed="2026-01-01", days=91):
        start = (pd.Timestamp(end) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        return {"start": start, "end": end, "val": val, "filed": filed}

    # 9 quarters of diluted EPS; the newest quarter has an AMENDED duplicate (later
    # 'filed' wins). Two FY (annual, ~365d duration) rows.
    eps_entries = [
        q("2024-03-31", 1.00), q("2024-06-30", 1.00), q("2024-09-30", 1.00),
        q("2024-12-31", 1.00), q("2025-03-31", 1.05), q("2025-06-30", 1.15),
        q("2025-09-30", 1.30), q("2025-12-31", 1.50),
        q("2026-03-31", 1.20, filed="2026-04-20"),
        q("2026-03-31", 1.47, filed="2026-05-05"),          # amended -> wins
        q("2025-12-31", 4.90, days=365), q("2024-12-31", 4.00, days=365),  # FY rows
    ]
    # Revenue only under the FALLBACK tag (no 'Revenues' key at all).
    rev_entries = [q("2025-03-31", 100.0), q("2026-03-31", 130.0)]
    facts = {"facts": {"us-gaap": {
        "EarningsPerShareDiluted": {"units": {"USD/shares": eps_entries}},
        "RevenueFromContractWithCustomerExcludingAssessedTax":
            {"units": {"USD": rev_entries}},
    }}}
    cik_map = {"0": {"cik_str": 123456, "ticker": "TSTX", "title": "Test Co"}}

    def fake_get(url, headers=None, timeout=0):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = cik_map if "company_tickers" in url else facts
        return r

    today = pd.Timestamp("2026-06-01")
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(dfeed, "EDGAR_DIR", Path(tmp)), \
                patch("requests.get", fake_get), patch("time.sleep", lambda s: None):
            out = dfeed._edgar_backfill("TSTX", today=today)
            # cached: a second call must not refetch (poison the network to prove it)
            with patch("requests.get", side_effect=AssertionError("refetched!")):
                again = dfeed._edgar_backfill("TSTX", today=today)
    assert again == out

    # YoY math off the AMENDED 2026-03-31 value: 1.47 vs 1.05 -> +40%
    assert abs(out["eps_yoy"] - 40.0) < 1e-6, out
    assert out["eps_quarter_end"] == "2026-03-31"
    assert out["revenue_quarter_end"] == "2026-03-31"
    # prev quarter: 1.50 vs 1.00 -> +50%
    assert abs(out["eps_yoy_prev"] - 50.0) < 1e-6
    # 3q acceleration: +40% (amended) vs +50% -> NOT strictly accelerating
    assert out["eps_accel_3q"] is False
    # FY growth: 4.90 vs 4.00 -> +22.5%
    assert abs(out["eps_fy_yoy"] - 22.5) < 1e-6
    # revenue via the fallback tag: 130 vs 100 -> +30%
    assert abs(out["revenue_yoy"] - 30.0) < 1e-6
    assert out["revenue_yoy_prev"] is None                  # only one matched pair


def test_fundamentals_surprise_and_edgar_merge():
    """(a) _last_earnings_surprise reads the newest REPORTED Surprise(%) row (future NaN
    rows drop) with its date, and drops one older than SURPRISE_MAX_AGE_DAYS: yfinance
    0.2.65's earnings_dates stopped at May 2025, so the panel showed a 16-month-old
    surprise (§6.85); (b) get_fundamentals merges the EDGAR backfill: yfinance values WIN,
    EDGAR fills the Nones and adds its own keys, but a prior-quarter YoY only joins a
    yfinance YoY describing the same quarter; the merged dict is what gets cached;
    (c) a pre-surprise-era cache (missing the new key) triggers one upgrade refetch."""
    import json as _json
    import tempfile
    from unittest.mock import patch
    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    # (a) surprise parsing, incl. a future unreported row
    class _Tk:
        earnings_dates = pd.DataFrame(
            {"EPS Estimate": [1.0, 1.1, 1.2], "Surprise(%)": [4.0, 6.5, float("nan")]},
            index=pd.to_datetime(["2026-01-15", "2026-04-16", "2026-07-20"]))

    assert dfeed._last_earnings_surprise(_Tk(), today="2026-05-01") == ("2026-04-16", 6.5)
    assert dfeed._last_earnings_surprise(_Tk(), today="2026-09-01") == (None, None)

    class _TkNone:
        earnings_dates = None

    assert dfeed._last_earnings_surprise(_TkNone()) == (None, None)

    # (b) + (c) merge & cache behavior with both fetchers patched
    yf_dict = {"revenue_yoy": None, "eps_yoy": 33.0, "eps_yoy_prev": None,
               "operating_margin": 20.0, "next_earnings": "2026-08-01",
               "last_surprise_pct": 6.5}
    ed_dict = {"revenue_yoy": 12.0, "eps_yoy": 99.0, "eps_yoy_prev": 8.0,
               "eps_fy_yoy": 22.5, "eps_accel_3q": True}
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(dfeed, "FUNDAMENTALS_DIR", Path(tmp)), \
                patch.object(dfeed, "_fetch_fundamentals", lambda s: dict(yf_dict)), \
                patch.object(dfeed, "_edgar_backfill", lambda s: dict(ed_dict)):
            out = dfeed.get_fundamentals("TSTX")
            assert out["revenue_yoy"] == 12.0             # EDGAR fills the None
            assert out["eps_yoy"] == 33.0                 # yfinance wins when present
            assert out["eps_yoy_prev"] is None            # quarter unknown: not paired
            assert out["eps_fy_yoy"] == 22.5 and out["eps_accel_3q"] is True
            cached = _json.loads((Path(tmp) / "TSTX.json").read_text())
            assert cached == out                          # the MERGED dict is cached

            # (c) an old-schema cache (no last_surprise_pct) forces one refetch
            (Path(tmp) / "OLD.json").write_text(
                _json.dumps({"revenue_yoy": 1.0, "next_earnings": None}))
            out2 = dfeed.get_fundamentals("OLD")
            assert "last_surprise_pct" in out2 and out2["eps_fy_yoy"] == 22.5


def _facts_q(end, val, days=91, filed="2026-01-01"):
    import pandas as pd
    start = (pd.Timestamp(end) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    return {"start": start, "end": end, "val": val, "filed": filed}


def test_edgar_picks_freshest_tag():
    """§6.85: EDGAR took the first tag with ANY data. HALO's and GILD's `Revenues` stops at
    2020-12-31 while `RevenueFromContract…` runs to 2026, so their revenue YoY pair came
    from 2020. The tag whose newest period ends latest now wins."""
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    old = [_facts_q(e, v) for e, v in (("2019-12-31", 50.0), ("2020-12-31", 60.0))]
    new = [_facts_q(e, v) for e, v in (("2025-06-30", 100.0), ("2026-06-30", 125.0))]
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": old}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": new}},
    }}}
    q, _a = dfeed._edgar_series(facts, dfeed.EDGAR_REVENUE_TAGS, ("USD",),
                                today=pd.Timestamp("2026-09-25"))
    assert [v for _, v in q] == [100.0, 125.0], q


def test_edgar_ignores_proxy_statement_facts():
    """§6.86: ANET's DEF 14A pay-versus-performance table tags net income in millions
    (3511 for $3.5B). Filed after the 10-K, it won the latest-filed rule, and the derived
    fiscal Q4 came out at −$2.6B. Only 10-Q/10-K facts count; a fact without a form
    (older canned data) still does."""
    from src.stock_screener.cockpit import data_feed as dfeed

    ten_k = {**_facts_q("2025-12-31", 3.5114e9, days=364, filed="2026-02-17"), "form": "10-K"}
    proxy = {**_facts_q("2025-12-31", 3511.0, days=364, filed="2026-04-16"), "form": "DEF 14A"}
    q, a = dfeed._edgar_tag_series({"USD": [ten_k, proxy]}, ("USD",))
    assert a[0][1] == 3.5114e9 and q == []
    q, a = dfeed._edgar_tag_series({"USD": [_facts_q("2025-12-31", 7.0, days=364)]}, ("USD",))
    assert a[0][1] == 7.0


def test_edgar_stale_series_dropped():
    """§6.85: GILD's quarterly `EarningsPerShareDiluted` ends 2010-06-30; with no fresher
    tag the series is dropped rather than read as current. Annual facts have their own,
    longer limit."""
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    eps = [_facts_q("2010-03-31", 1.0), _facts_q("2010-06-30", 1.2),
           _facts_q("2025-12-31", 5.0, days=365), _facts_q("2024-12-31", 4.0, days=365)]
    facts = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": eps}}}}}
    q, a = dfeed._edgar_series(facts, dfeed.EDGAR_EPS_TAGS, ("USD/shares",),
                               today=pd.Timestamp("2026-09-25"))
    assert q == [] and [v for _, v in a] == [4.0, 5.0]
    q, a = dfeed._edgar_series(facts, dfeed.EDGAR_EPS_TAGS, ("USD/shares",),
                               today=pd.Timestamp("2028-01-01"))
    assert q == [] and a == []


def test_edgar_fills_fiscal_q4():
    """§6.85: companies file the fourth quarter only inside the 10-K's year, so EDGAR had
    no March quarter for MLAB (fiscal year ends in March), and '3 quarters running' spanned
    a missing one. Q4 = FY − (Q1+Q2+Q3) when exactly three quarters sit inside the year."""
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    T = pd.Timestamp
    quarterly = [(T("2025-06-30"), 1.0), (T("2025-09-30"), 2.0), (T("2025-12-31"), 3.0),
                 (T("2026-06-30"), 5.0)]
    annual = [(T("2026-03-31"), 10.0)]
    filled = dfeed._edgar_fill_q4(quarterly, annual)
    assert (T("2026-03-31"), 4.0) in filled and len(filled) == 5
    assert dfeed._consecutive(filled, 3) == [3.0, 4.0, 5.0]
    # only two quarters inside the year: no fill
    assert dfeed._edgar_fill_q4(quarterly[1:], annual) == quarterly[1:]
    # a Q4 already filed is kept, not derived
    have = quarterly[:3] + [(T("2026-03-31"), 9.0)]
    assert dfeed._edgar_fill_q4(have, annual) == have


def test_eps_accel_3q_needs_consecutive_quarters():
    """§6.85: `eps_accel_3q` compared the last three growth figures whatever their dates,
    so a missing quarter made it span half a year. It now needs three consecutive quarters,
    and a prior-quarter YoY must be the quarter immediately before."""
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    T = pd.Timestamp
    run = [(T("2025-12-31"), 10.0), (T("2026-03-31"), 20.0), (T("2026-06-30"), 30.0)]
    assert dfeed._consecutive(run, 3) == [10.0, 20.0, 30.0]
    gap = [(T("2025-09-30"), 10.0), (T("2026-03-31"), 20.0), (T("2026-06-30"), 30.0)]
    assert dfeed._consecutive(gap, 3) is None
    assert dfeed._consecutive(gap, 2) == [20.0, 30.0]
    assert dfeed._consecutive(run[:1], 2) is None


def test_edgar_code33():
    """§6.86 (audit Step-2 #2): Code 33 = EPS growth, sales growth and net margin each
    higher in each of the three consecutive quarters ending at the newest EPS quarter.
    A flat leg breaks it; a missing quarter makes it unknown; growth from a loss is not
    growth; two falling EPS growth steps set the deceleration warning."""
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    ends = pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31",
                           "2026-03-31", "2026-06-30"])

    def ser(vals):
        return list(zip(ends, vals))

    eps = ser([1.0, 1.0, 1.0, 1.1, 1.3, 1.6])          # growth +10, +30, +60 on 1.0
    # year-ago quarters: 2025-03..06 have no match, so growth starts at 2026-03-31;
    # extend with a year of history so three growth quarters exist
    hist = pd.to_datetime(["2024-09-30", "2024-12-31"])
    eps = list(zip(hist, [1.0, 1.0])) + eps
    rev = list(zip(hist, [100.0, 100.0])) + ser([100.0, 100.0, 100.0, 110, 125, 140])
    ni = list(zip(hist, [10.0, 10.0])) + ser([10.0, 10.0, 10.0, 12.0, 15.0, 18.9])
    out = dfeed._code33(eps, rev, ni)
    assert out["eps_g3"] == [10.0, 30.0, 60.0], out
    assert out["rev_g3"] == [10.0, 25.0, 40.0]
    assert out["margin3"] == [10.9, 12.0, 13.5]
    assert out["code33"] == {"eps": True, "sales": True, "margin": True, "all": True}
    assert out["eps_decel_2q"] is False

    flat = list(ni[:-1]) + [(ends[-1], 16.7)]           # margin 12.0 -> 11.9: not rising
    c = dfeed._code33(eps, rev, flat)["code33"]
    assert c["margin"] is False and c["all"] is False and c["eps"] is True

    gap = [p for p in rev if p[0] != ends[-2]]
    assert dfeed._code33(eps, gap, ni)["code33"] is None

    loss = [(e, -1.0 if e == hist[1] else v) for e, v in eps]   # 2024-12-31 was a loss
    assert dfeed._code33(loss, rev, ni)["code33"] is None

    slowing = list(eps[:-3]) + [(ends[-3], 1.6), (ends[-2], 1.3), (ends[-1], 1.1)]
    assert dfeed._code33(slowing, rev, ni)["eps_decel_2q"] is True
    assert dfeed._code33([], rev, ni)["code33"] is None


def test_edgar_annual_eps_runs():
    """§6.86: annual EPS up on the year, and up three years running, over consecutive
    fiscal years only."""
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    T = pd.Timestamp
    fy = [(T("2022-12-31"), 1.0), (T("2023-12-31"), 1.5), (T("2024-12-31"), 2.0),
          (T("2025-12-31"), 2.5)]
    assert dfeed._annual_runs(fy) == {"eps_fy_up": True, "eps_fy_up_3y": True}
    dip = fy[:2] + [(T("2024-12-31"), 1.2), (T("2025-12-31"), 2.5)]
    assert dfeed._annual_runs(dip) == {"eps_fy_up": True, "eps_fy_up_3y": False}
    missing = [fy[0], fy[1], fy[3]]
    assert dfeed._annual_runs(missing) == {"eps_fy_up": None, "eps_fy_up_3y": None}
    assert dfeed._annual_runs(fy[:1]) == {"eps_fy_up": None, "eps_fy_up_3y": None}


def test_edgar_last_report_picks_newest_202():
    """§6.87: the last earnings release is the newest 8-K carrying item 2.02, timed by its
    EDGAR acceptance in New York. Yahoo's past dates stop at May 2025 (§6.85). Other 8-Ks,
    10-Qs and 6-Ks don't count; a failed fetch reads None."""
    from unittest.mock import patch
    from src.stock_screener.cockpit import data_feed as dfeed

    recent = {"form": ["8-K", "10-Q", "8-K", "8-K", "6-K"],
              "items": ["5.02", "", "2.02,9.01", "2.02", "2.02"],
              "acceptanceDateTime": ["2026-09-01T12:00:00.000Z", "2026-08-07T20:00:00.000Z",
                                     "2026-08-06T20:08:16.000Z", "2026-05-11T12:30:00.000Z",
                                     "2026-09-10T12:00:00.000Z"]}
    with patch.object(dfeed, "_edgar_get_json", lambda url: {"filings": {"recent": recent}}):
        assert dfeed._edgar_last_report(1) == {"last_report": "2026-08-06",
                                               "last_report_time": "16:08"}
    with patch.object(dfeed, "_edgar_get_json", lambda url: None):
        assert dfeed._edgar_last_report(1) == {"last_report": None, "last_report_time": None}


def test_fundamentals_estimates_and_holders():
    """§6.88 (audit Step-2 #4): the audit said analyst revisions and fund ownership had no
    free source, but yfinance 0.2.65 returns both. The current-year estimate's change over
    30/90 days, and the institutions' share and count; a raising property, an empty frame
    or a zero base read None."""
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    class _Tk:
        eps_trend = pd.DataFrame(
            {"current": [2.24, 9.95], "7daysAgo": [2.73, 11.62], "30daysAgo": [2.73, 11.62],
             "60daysAgo": [2.73, 11.62], "90daysAgo": [1.26, 5.39]},
            index=pd.Index(["0q", "0y"], name="period"))
        major_holders = pd.DataFrame(
            {"Value": [0.0562, 0.96038, 1.01756, 235.0]},
            index=pd.Index(["insidersPercentHeld", "institutionsPercentHeld",
                            "institutionsFloatPercentHeld", "institutionsCount"],
                           name="Breakdown"))

    est = dfeed._estimate_revisions(_Tk())
    assert round(est["est_rev_90d"], 1) == round((9.95 / 5.39 - 1) * 100, 1)
    assert round(est["est_rev_30d"], 1) == round((9.95 / 11.62 - 1) * 100, 1)
    assert dfeed._institutional(_Tk()) == {"inst_pct": 96.0, "inst_count": 235}

    class _Raises:
        @property
        def eps_trend(self):
            raise RuntimeError("yahoo down")
        major_holders = pd.DataFrame()

    assert dfeed._estimate_revisions(_Raises()) == {"est_rev_30d": None, "est_rev_90d": None}
    assert dfeed._institutional(_Raises()) == {"inst_pct": None, "inst_count": None}

    class _Zero:
        eps_trend = pd.DataFrame({"current": [1.0], "30daysAgo": [0.0], "90daysAgo": [0.0]},
                                 index=["0q"])
    assert dfeed._estimate_revisions(_Zero()) == {"est_rev_30d": None, "est_rev_90d": None}


def test_inst_history_carried_forward():
    """§6.88: the fund count is a snapshot, so its trend is kept by carrying a dated list
    across refetches: appended only when the count changes, the last 8 kept."""
    from src.stock_screener.cockpit import data_feed as dfeed

    out = dfeed._carry_inst_history({"inst_count": 221}, None, today="2026-07-02")
    assert out["inst_history"] == [["2026-07-02", 221]]
    same = dfeed._carry_inst_history({"inst_count": 221}, out, today="2026-07-09")
    assert same["inst_history"] == [["2026-07-02", 221]]
    up = dfeed._carry_inst_history({"inst_count": 235}, same, today="2026-09-25")
    assert up["inst_history"] == [["2026-07-02", 221], ["2026-09-25", 235]]
    gone = dfeed._carry_inst_history({"inst_count": None}, up, today="2026-10-02")
    assert gone["inst_history"] == up["inst_history"]
    prior = {"inst_history": [[f"2026-0{m}-01", m] for m in range(1, 10)]}
    capped = dfeed._carry_inst_history({"inst_count": 99}, prior, today="2026-10-01")
    assert len(capped["inst_history"]) == 8 and capped["inst_history"][-1] == ["2026-10-01", 99]


def test_prev_pair_from_one_source():
    """§6.85: a yfinance YoY was paired with an EDGAR prior-quarter YoY from whatever
    quarter EDGAR had, so "EPS accelerating" could compare 2026 with 2010. The EDGAR prior
    joins only when both describe the same quarter; with no yfinance YoY, EDGAR supplies the
    whole pair."""
    from src.stock_screener.cockpit import data_feed as dfeed

    ed = {"eps_yoy": 50.0, "eps_yoy_prev": 20.0, "eps_quarter_end": "2026-03-31",
          "revenue_yoy": 30.0, "revenue_yoy_prev": 25.0, "revenue_quarter_end": "2026-03-31",
          "eps_fy_yoy": 12.0}
    same = dfeed._merge_edgar({"eps_yoy": 40.0, "eps_yoy_prev": None,
                               "eps_quarter_end": "2026-03-31"}, dict(ed))
    assert same["eps_yoy"] == 40.0 and same["eps_yoy_prev"] == 20.0
    ahead = dfeed._merge_edgar({"eps_yoy": 40.0, "eps_yoy_prev": None,
                                "eps_quarter_end": "2026-06-30"}, dict(ed))
    assert ahead["eps_yoy"] == 40.0 and ahead["eps_yoy_prev"] is None
    none = dfeed._merge_edgar({"revenue_yoy": None, "revenue_yoy_prev": None}, dict(ed))
    assert (none["revenue_yoy"], none["revenue_yoy_prev"], none["revenue_quarter_end"]) == \
        (30.0, 25.0, "2026-03-31")
    assert none["eps_fy_yoy"] == 12.0
    assert dfeed._merge_edgar({"eps_yoy": 1.0}, None) == {"eps_yoy": 1.0}


def test_fundamentals_refetch_after_report_date():
    """§6.85: a cache written on or before its `next_earnings` date is refetched once that
    date has passed, rather than serving the pre-report quarter for up to 7 days."""
    import json as _json
    import tempfile
    from unittest.mock import patch
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    written = pd.Timestamp.now().normalize()          # the file's mtime is "now"
    day = lambda n: (written + pd.Timedelta(days=n)).strftime("%Y-%m-%d")   # noqa: E731
    fresh = {"revenue_yoy": 99.0, "next_earnings": day(9), "last_surprise_pct": None}
    forced = []
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "TSTX.json"
        with patch.object(dfeed, "FUNDAMENTALS_DIR", Path(tmp)), \
                patch.object(dfeed, "_fetch_fundamentals", lambda s: dict(fresh)), \
                patch.object(dfeed, "_edgar_backfill",
                             lambda s, force=False: forced.append(force)):
            def cached(report):
                p.write_text(_json.dumps({"revenue_yoy": 1.0, "next_earnings": report,
                                          "last_surprise_pct": None, "est_rev_90d": None}))
            cached(day(1))
            assert dfeed.get_fundamentals("TSTX", today=day(1))["revenue_yoy"] == 1.0
            assert dfeed.get_fundamentals("TSTX", today=day(2))["revenue_yoy"] == 99.0
            # §6.87: EDGAR is refetched past its own cache, for the new release date
            assert forced == [True]
            cached(day(-1))                            # written after the report
            assert dfeed.get_fundamentals("TSTX", today=day(2))["revenue_yoy"] == 1.0
            cached(None)
            assert dfeed.get_fundamentals("TSTX", today=day(2))["revenue_yoy"] == 1.0


def test_fundamentals_refetch_old_step2_cache():
    """§6.90: on 2026-09-26 the Pi still showed F out of 4. Friday's screen ran before the
    eight-check code was promoted and rewrote ~450 passers' caches in the old schema, and a
    cache under 7 days old was served whatever it held, so F would have read ≤ 4 until
    ~Oct 2. A cache without `est_rev_90d` is now refetched at once; one holding it as None
    is served."""
    import json as _json
    import tempfile
    from unittest.mock import patch
    from src.stock_screener.cockpit import data_feed as dfeed

    fresh = {"revenue_yoy": 99.0, "next_earnings": None, "last_surprise_pct": None,
             "est_rev_90d": 7.0}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "TSTX.json"
        with patch.object(dfeed, "FUNDAMENTALS_DIR", Path(tmp)), \
                patch.object(dfeed, "_fetch_fundamentals", lambda s: dict(fresh)), \
                patch.object(dfeed, "_edgar_backfill", lambda s, force=False: None):
            p.write_text(_json.dumps({"revenue_yoy": 1.0, "next_earnings": None,
                                      "last_surprise_pct": None}))
            assert dfeed.get_fundamentals("TSTX")["revenue_yoy"] == 99.0
            p.write_text(_json.dumps({"revenue_yoy": 1.0, "next_earnings": None,
                                      "last_surprise_pct": None, "est_rev_90d": None}))
            assert dfeed.get_fundamentals("TSTX")["revenue_yoy"] == 1.0


def test_edgar_refetch_old_cache():
    """§6.90: a fresh EDGAR cache from before Code 33 and the release date is refetched,
    not served for the rest of its 7 days."""
    import json as _json
    import tempfile
    from unittest.mock import patch
    from src.stock_screener.cockpit import data_feed as dfeed

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "TSTX.json"
        with patch.object(dfeed, "EDGAR_DIR", Path(tmp)), \
                patch.object(dfeed, "_edgar_cik", lambda s: 1), \
                patch.object(dfeed, "_edgar_get_json", lambda url: {"facts": {}}):
            p.write_text(_json.dumps({"eps_yoy": 5.0}))
            out = dfeed._edgar_backfill("TSTX")
            assert "code33" in out and out["eps_yoy"] is None      # refetched
            p.write_text(_json.dumps({"eps_yoy": 5.0, "code33": None, "last_report": None}))
            assert dfeed._edgar_backfill("TSTX")["eps_yoy"] == 5.0   # served


def test_margin_aligns_num_and_den_quarters():
    """_margin / _margin_trend must pair numerator and denominator from the SAME quarter. When
    yfinance has the newest quarter's revenue but not yet its operating income (a common
    right-after-a-release state), the OI series ends one quarter earlier — aligning on the common
    index keeps the ratio honest instead of dividing OI(Q-1) by Rev(Q0) (a wrong margin and a
    possibly flipped trend sign)."""
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    qends = pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30",
                            "2025-12-31", "2026-03-31"])
    rev = pd.Series([100., 110., 120., 130., 200.], index=qends)   # newest quarter revenue jumps
    oi = pd.Series([18., 20., 24., 32.5], index=qends[:4])         # OI not yet reported for Q0

    # aligned: latest COMMON quarter is 2025-12-31 -> 32.5/130 = 25.0% (NOT 32.5/200 = 16.25%)
    assert abs(dfeed._margin(oi, rev) - 25.0) < 1e-9
    # aligned trend: 25.0% (Q-1) vs 20.0% (Q-2, 24/120) = +5.0pp expanding — the buggy
    # cross-quarter pairing would read it as CONTRACTING
    assert abs(dfeed._margin_trend(oi, rev) - 5.0) < 1e-9
    # degenerate inputs still yield None, never raise
    assert dfeed._margin(None, rev) is None
    assert dfeed._margin(oi, pd.Series(dtype=float)) is None
    assert dfeed._margin_trend(oi.iloc[:1], rev) is None           # only one common quarter


def test_yoy_is_date_matched():
    """_yoy / _yoy_prev compare against the entry ~a year earlier by DATE (330-400 days), so an
    extra/missing quarter can't misalign a fixed 4-step positional lag."""
    import pandas as pd
    from src.stock_screener.cockpit import data_feed as dfeed

    # six entries incl. an off-cycle 2025-11-15 restatement: a positional -5 lag lands on
    # 2025-09-30 (~9mo back, wrong); date-matching finds 2025-06-30 (365 days) -> +25%.
    idx = pd.to_datetime(["2025-06-30", "2025-09-30", "2025-11-15",
                          "2025-12-31", "2026-03-31", "2026-06-30"])
    s = pd.Series([100., 102., 103., 104., 106., 125.], index=idx)
    assert abs(dfeed._yoy(s) - 25.0) < 1e-9

    # too short to have a ~1-year-prior quarter -> None (not a bogus positional read)
    short = pd.Series([100., 110.],
                      index=pd.to_datetime(["2026-03-31", "2026-06-30"]))
    assert dfeed._yoy(short) is None


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
              "ABCDW|Some Warrant|Q|N|N|100|N|N\n"              # base+W (5-char) -> drop
              "ABCDU|Some SPAC Unit|Q|N|N|100|N|N\n"            # base+U (5-char) -> drop
              "ABCDR|Some SPAC Right|Q|N|N|100|N|N\n"           # base+R (5-char) -> drop
              "SNOW|Snowflake Inc - Common Stock|Q|N|N|100|N|N\n"   # 4-char, ends W -> KEEP
              "PLTR|Palantir Technologies - Common|Q|N|N|100|N|N\n" # 4-char, ends R -> KEEP
              "LULU|lululemon athletica - Common|Q|N|N|100|N|N\n"   # 4-char, ends U -> KEEP
              "U|Unity Software Inc - Common Stock|N|N|N|100|N|N\n"  # 1-char, ends U -> KEEP
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
    assert "TSTZ" not in syms, "test issue must be dropped"
    assert not {"ABCDW", "ABCDU", "ABCDR"} & set(syms), "base+W/R/U (5-char) warrants/units/rights must be dropped"
    # Regression: ordinary common stocks ending in W/R/U must NOT be dropped by the
    # warrant filter (the old (?:W|R|U)$ pattern silently removed all of these).
    for keep in ("SNOW", "PLTR", "LULU", "U"):
        assert keep in syms, f"{keep} is a common stock and must be kept"
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

    # (d) The cache's FINAL bar can be provisional too: an intraday scan persisted a
    # mid-session close for it on an EARLIER day. On the next open the settled fetch
    # differs on that bar only — NOT a split; the merge must adopt the settled bar
    # instead of re-baselining. (This was the new-day-open 2y avalanche: every name
    # that moved >SPLIT_TOL after the last intraday scan re-downloaded full history.)
    cdx = pd.bdate_range(end=today, periods=10)
    cdx = cdx[cdx < today]                                         # ends BEFORE today
    prov = _ohlcv(cdx, [100.0] * (len(cdx) - 1) + [107.0])         # final bar provisional
    settled = _ohlcv(cdx[-4:], [100.0] * 4)                        # settled overlap re-fetch
    m3, full3 = dfeed._merge_incremental(prov, settled, "2y")
    assert full3 is False, "a provisional FINAL bar must not trigger a re-baseline"
    assert float(m3.loc[cdx[-1], "Close"]) == 100.0                # settled bar adopted
    # ...but a real split still re-baselines: the OLDER overlap days diverge as well.
    halved = _ohlcv(cdx[-4:], [50.0] * 4)
    _, full4 = dfeed._merge_incremental(prov, halved, "2y")
    assert full4 is True, "divergence across settled prior days must still re-baseline"


def test_get_many_prices_progress_labels():
    """Download-transparency labels through the progress callback: 'SYM: cached (fresh)' for
    a fresh-cache serve, 'SYM: M/D/YYYY - M/D/YYYY' (missing-days range) for an incremental
    top-up, a single 'SYM: M/D/YYYY' when only today's bar needs refreshing, and
    'SYM: full history (2y)' for a cold name. Offline (empty fake yfinance)."""
    import tempfile
    from unittest.mock import patch
    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    def _ohlcv(idx, close):
        return pd.DataFrame({"Open": close, "High": [c + 1 for c in close],
                             "Low": [c - 1 for c in close], "Close": close,
                             "Volume": [1000] * len(idx)}, index=pd.DatetimeIndex(idx))

    today = pd.Timestamp.today().normalize()

    def _us(d):
        return f"{d.month}/{d.day}/{d.year}"

    labels = []

    def prog(done, total, label):
        labels.append(label)

    fake_empty = lambda *a, **kw: pd.DataFrame()          # noqa: E731 — download "fails"

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        with patch.object(dfeed, "PRICES_DIR", pdir), patch("yfinance.download", fake_empty):
            # (a) fresh cache (mtime = now) under max_age_days=7 -> served, no fetch
            fresh_idx = pd.bdate_range(end=today, periods=15)
            _ohlcv(fresh_idx, [100.0] * 15).to_parquet(pdir / "FRESH.parquet")
            labels.clear()
            dfeed.get_many_prices(["FRESH"], max_age_days=7.0, pause=0, retries=0,
                                  progress=prog)
            assert labels == ["FRESH: cached (fresh)"], labels

            # (b) cache with a gap -> incremental; label = the missing-days range
            gap_idx = pd.bdate_range(end=today - pd.Timedelta(days=4), periods=15)
            _ohlcv(gap_idx, [100.0] * 15).to_parquet(pdir / "GAPPY.parquet")
            last = gap_idx[-1]
            labels.clear()
            dfeed.get_many_prices(["GAPPY"], max_age_days=-1, pause=0, retries=0,
                                  progress=prog)
            exp = f"GAPPY: {_us(last + pd.Timedelta(days=1))} - {_us(today)}"
            assert labels == [exp], (labels, exp)

            # (c) cache already ends TODAY (provisional bar) -> single-date refresh label.
            # bdate_range ends on the last BUSINESS day, so on a weekend it can't produce a
            # today-dated bar — append one explicitly (the production scenario, a provisional
            # bar dated today, only exists on trading days; without this the test failed
            # every Saturday/Sunday on a range label instead).
            tody_idx = (fresh_idx if fresh_idx[-1] == today
                        else fresh_idx.append(pd.DatetimeIndex([today])))
            _ohlcv(tody_idx, [100.0] * len(tody_idx)).to_parquet(pdir / "TODY.parquet")
            labels.clear()
            dfeed.get_many_prices(["TODY"], max_age_days=-1, pause=0, retries=0,
                                  progress=prog)
            assert labels == [f"TODY: {_us(today)}"], labels

            # (d) no cache at all AND the download fails -> honest FAILED label (item 13:
            # a successful full fetch keeps the plain "full history (2y)" label)
            labels.clear()
            dfeed.get_many_prices(["COLD"], max_age_days=-1, pause=0, retries=0,
                                  progress=prog)
            assert labels == ["COLD: full history (2y) FAILED (no data)"], labels


def test_get_many_prices_full_fetch_serves_stale_cache():
    """Item 13a: a name routed to the FULL pass (gap > max_gap_days) whose download fails
    must fall back to its stale parquet instead of being silently dropped — without
    re-persisting it (the mtime must not enter the fresh-serve window). A truly cold name
    stays absent, with an honest label."""
    import tempfile
    from unittest.mock import patch
    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    today = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=today - pd.Timedelta(days=20), periods=30)   # gap 20 > max 10
    stale = pd.DataFrame({"Open": [100.0] * 30, "High": [101.0] * 30, "Low": [99.0] * 30,
                          "Close": [100.0] * 30, "Volume": [1000] * 30}, index=idx)
    labels = []

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        with patch.object(dfeed, "PRICES_DIR", pdir), \
                patch("yfinance.download", lambda *a, **kw: pd.DataFrame()):
            stale.to_parquet(pdir / "STALE.parquet")
            mtime = (pdir / "STALE.parquet").stat().st_mtime_ns
            got = dfeed.get_many_prices(["STALE", "COLD"], max_age_days=-1, pause=0,
                                        retries=0, progress=lambda d, t, s: labels.append(s))
        assert "STALE" in got and len(got["STALE"]) == 30
        assert float(got["STALE"]["Close"].iloc[-1]) == 100.0
        assert "COLD" not in got
        assert (pdir / "STALE.parquet").stat().st_mtime_ns == mtime, \
            "stale fallback must not re-persist (mtime would enter the fresh window)"
    by = {s.split(":")[0]: s for s in labels}
    assert "stale cache served" in by["STALE"]
    assert "FAILED (no data)" in by["COLD"]


def test_get_many_prices_retries_failed_subset():
    """Item 13b: when a batch download PARTIALLY fails (some tickers' frames missing),
    the missing subset gets ONE retry batch before any fallback — _download_batch's own
    retry only fires when the WHOLE batch is empty."""
    import tempfile
    from unittest.mock import patch
    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    today = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=today, periods=30)

    def _flat(close):
        return pd.DataFrame({"Open": close, "High": close, "Low": close,
                             "Close": close, "Volume": [1000] * len(idx)}, index=idx)

    good_multi = pd.concat({"GOOD": _flat([50.0] * 30)}, axis=1)    # group_by='ticker' shape
    badd_flat = _flat([70.0] * 30)
    calls = []

    def fake_dl(tickers, **kw):
        calls.append(list(tickers) if isinstance(tickers, (list, tuple)) else [tickers])
        return good_multi if len(calls) == 1 else badd_flat

    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(dfeed, "PRICES_DIR", Path(tmp)), \
                patch("yfinance.download", fake_dl):
            got = dfeed.get_many_prices(["GOOD", "BADD"], max_age_days=-1, pause=0,
                                        retries=1)
        assert len(calls) == 2, f"expected batch + one subset retry, got {calls}"
        assert calls[1] == ["BADD"], "the retry must target only the missing subset"
        assert "GOOD" in got and "BADD" in got
        assert float(got["BADD"]["Close"].iloc[-1]) == 70.0
        assert (Path(tmp) / "BADD.parquet").exists()     # retried data is persisted


def test_yf_download_lock_held_and_exclusive():
    """R2-2: every yf.download call happens WITH _YF_LOCK held (recording fake — never
    assert-raise inside it: _download_batch swallows exceptions and the test would pass
    vacuously), across all three paths: cold full fetch, incremental top-up, and the
    subset retry. Then an Event-sequenced two-thread run pins mutual exclusion."""
    import tempfile
    import threading as th
    from unittest.mock import patch

    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    today = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=today, periods=30)

    def _flat(close):
        return pd.DataFrame({"Open": close, "High": close, "Low": close,
                             "Close": close, "Volume": [1000] * len(idx)}, index=idx)

    good_multi = pd.concat({"GOOD": _flat([50.0] * 30)}, axis=1)
    badd_flat = _flat([70.0] * 30)
    lock_states = []

    def recording_dl(tickers, **kw):
        lock_states.append(dfeed._YF_LOCK.locked())
        tl = list(tickers) if isinstance(tickers, (list, tuple)) else [tickers]
        if "INCR" in tl:
            return pd.DataFrame()        # incremental attempt fails -> cache served
        # Full batch: GOOD present, BADD missing -> forces the subset-retry path,
        # which then gets the flat frame.
        return good_multi if "BADD" in tl and "GOOD" in tl else badd_flat

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        # An INCREMENTAL name: cache ending 3 business days back (gap <= max_gap_days).
        gap_idx = pd.bdate_range(end=today, periods=33)[:30]
        _flat([90.0] * 30).set_axis(gap_idx).to_parquet(pdir / "INCR.parquet")
        with patch.object(dfeed, "PRICES_DIR", pdir), \
                patch("yfinance.download", recording_dl):
            dfeed.get_many_prices(["GOOD", "BADD", "INCR"], max_age_days=-1, pause=0,
                                  retries=1)
    assert len(lock_states) >= 3, f"expected incr + full + subset-retry calls, got {len(lock_states)}"
    assert all(lock_states), "every yf.download must run under _YF_LOCK"

    # --- mutual exclusion: thread B's download can't start while A's is in flight ------
    started, release = th.Event(), th.Event()
    overlap = {"seen": False, "in_flight": False}
    guard = th.Lock()

    def blocking_dl(tickers, **kw):
        with guard:
            if overlap["in_flight"]:
                overlap["seen"] = True
            overlap["in_flight"] = True
        started.set()
        assert release.wait(10), "test deadlock"
        with guard:
            overlap["in_flight"] = False
        return pd.DataFrame()

    def _fetch(sym, tmp):
        with patch.object(dfeed, "PRICES_DIR", Path(tmp)):
            dfeed.get_many_prices([sym], max_age_days=-1, pause=0, retries=0)

    with tempfile.TemporaryDirectory() as tmp, patch("yfinance.download", blocking_dl):
        t1 = th.Thread(target=_fetch, args=("AAA", tmp), daemon=True)
        t1.start()
        assert started.wait(10)
        assert dfeed._YF_LOCK.locked(), "lock must be held during a download"
        t2 = th.Thread(target=_fetch, args=("BBB", tmp), daemon=True)
        t2.start()
        release.set()
        t1.join(10); t2.join(10)
    assert not overlap["seen"], "two yf.download calls overlapped despite _YF_LOCK"


def test_incremental_persist_requires_reaching_last_bar():
    """R2-5: the incremental persist fires only when the fetch reached the cache's
    newest bar. An overlap-only response (provider lag) must NOT rewrite the parquet —
    the rewrite re-stamps mtime, and a post-cutoff mtime would arm the settled-close
    gate on content lacking the settled bar. Same-day revisions (max == last, the ~16:30
    finalize) and newer-only responses still persist."""
    import tempfile
    from unittest.mock import patch

    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    today = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=today, periods=10)
    last = idx[-1]

    def _flat(ix, close):
        return pd.DataFrame({"Open": close, "High": close, "Low": close,
                             "Close": close, "Volume": [1000] * len(ix)}, index=ix)

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        path = pdir / "AAA.parquet"

        def _reset():
            _flat(idx, [100.0] * 10).to_parquet(path)
            return path.stat().st_mtime_ns

        with patch.object(dfeed, "PRICES_DIR", pdir):
            # (a) overlap-only response (every bar BEFORE last) -> NO persist
            m0 = _reset()
            with patch("yfinance.download",
                       lambda tk, **kw: _flat(idx[-6:-1], [100.0] * 5)):
                out = dfeed.get_many_prices(["AAA"], max_age_days=-1, pause=0, retries=0)
            assert path.stat().st_mtime_ns == m0, "overlap-only must not re-stamp mtime"
            assert len(out["AAA"]) == 10                     # frame still served intact

            # (b) same-day finalize (response reaches last, close revised) -> persists
            m0 = _reset()
            with patch("yfinance.download",
                       lambda tk, **kw: _flat(idx[-3:], [100.0, 100.0, 107.0])):
                dfeed.get_many_prices(["AAA"], max_age_days=-1, pause=0, retries=0)
            assert path.stat().st_mtime_ns != m0, "reaching the last bar must persist"
            assert float(pd.read_parquet(path)["Close"].iloc[-1]) == 107.0

            # (c) newer-only response (bars strictly after last) -> persists with them
            m0 = _reset()
            nxt = pd.bdate_range(start=last + pd.Timedelta(days=1), periods=2)
            with patch("yfinance.download", lambda tk, **kw: _flat(nxt, [111.0, 112.0])):
                dfeed.get_many_prices(["AAA"], max_age_days=-1, pause=0, retries=0)
            assert path.stat().st_mtime_ns != m0
            assert float(pd.read_parquet(path)["Close"].iloc[-1]) == 112.0


def test_settled_serve_requires_current_frame():
    """R2-5b: the settled-close gate is mtime-keyed, but the FRAME must be current too —
    a file written post-cutoff whose last bar predates the latest settled session (a
    lagging full-fetch response) falls through to the top-up instead of being served as
    settled; a frame ending at the last completed session settle-serves with zero
    network. no_session_since is patched with an epoch threshold so the test is
    deterministic at any wall-clock time."""
    import tempfile
    import time as _time
    from unittest.mock import patch

    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed, triggers as trg

    def _flat(ix, close):
        return pd.DataFrame({"Open": close, "High": close, "Low": close,
                             "Close": close, "Volume": [1000] * len(ix)}, index=ix)

    today = pd.Timestamp.today().normalize()
    cur_idx = pd.bdate_range(end=today, periods=10)          # ends the last bday
    stale_idx = pd.bdate_range(end=today, periods=15)[:10]   # ends 5 bdays back
    # Epochs within the last ~4 days count as "no session since" (mtime = now, and the
    # current frame's session end); anything older (the stale frame's end) fails.
    threshold = _time.time() - 4 * 86400
    calls, labels = [], []

    def fake_dl(tickers, **kw):
        calls.append(1)
        return pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        _flat(cur_idx, [100.0] * 10).to_parquet(pdir / "CUR.parquet")
        _flat(stale_idx, [100.0] * 10).to_parquet(pdir / "STALE.parquet")
        with patch.object(dfeed, "PRICES_DIR", pdir), \
                patch.object(trg, "no_session_since",
                             lambda ep, now=None: ep >= threshold), \
                patch("yfinance.download", fake_dl):
            dfeed.get_many_prices(["CUR"], max_age_days=0.0, pause=0, retries=0,
                                  progress=lambda d, t, s: labels.append(s))
            assert labels[-1] == "CUR: cached (settled close)"
            assert not calls, "a current settled frame must serve with zero network"
            dfeed.get_many_prices(["STALE"], max_age_days=0.0, pause=0, retries=0,
                                  progress=lambda d, t, s: labels.append(s))
    assert "settled" not in labels[-1], "short frame must fall through to the top-up"
    assert calls, "the fallthrough must actually attempt a fetch"


def test_get_many_prices_cache_only_when_network_busy():
    """allow_network=False serves every cached name AS-IS with zero yfinance calls —
    incremental candidates from their pre-pass frames, too-stale names straight from
    parquet — and a name with no cache at all comes back absent instead of triggering a
    download. The mode interactive pages use while the bulk pipeline holds _YF_LOCK."""
    import tempfile
    from unittest.mock import patch

    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    today = pd.Timestamp.today().normalize()

    def _flat(ix, close):
        return pd.DataFrame({"Open": close, "High": close, "Low": close,
                             "Close": close, "Volume": [1000] * len(ix)}, index=ix)

    def _no_net(*a, **kw):
        raise AssertionError("network touched in cache-only mode")

    gap_idx = pd.bdate_range(end=today, periods=13)[:10]     # ends 3 bdays back -> incr
    old_idx = pd.bdate_range(end=today - pd.Timedelta(days=40), periods=10)  # -> full
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        _flat(gap_idx, [50.0] * 10).to_parquet(pdir / "GAPPY.parquet")
        _flat(old_idx, [70.0] * 10).to_parquet(pdir / "OLDIE.parquet")
        with patch.object(dfeed, "PRICES_DIR", pdir), patch("yfinance.download", _no_net):
            out = dfeed.get_many_prices(["GAPPY", "OLDIE", "NOCACHE"], max_age_days=-1,
                                        pause=0, retries=0, allow_network=False)
    assert float(out["GAPPY"]["Close"].iloc[-1]) == 50.0     # incr frame, as-is
    assert float(out["OLDIE"]["Close"].iloc[-1]) == 70.0     # stale parquet, as-is
    assert "NOCACHE" not in out                              # absent, not downloaded


def test_fetch_positions_skips_network_when_pipeline_busy():
    """fetch_positions passes allow_network=False to the price pull whenever
    network_busy() reports the bulk pipeline holding _YF_LOCK — the Positions page
    (and its SELL controls) must never queue behind a multi-minute sweep. When the
    pipeline is free, the normal (network-allowed) pull is unchanged."""
    from src.stock_screener.cockpit import trade, data_feed

    Client, _Pos, _Order = _pos_fakes()
    positions = [_Pos("NMM", 716, avg_entry_price=79.5, current_price=79.0,
                      market_value=56564.0, cost_basis=56922.0, unrealized_pl=-358.0,
                      unrealized_plpc=-0.0063, lastday_price=78.3)]
    client = Client(positions, {})
    seen = {}

    def _gmp(syms, **kw):
        seen["allow_network"] = kw.get("allow_network", True)
        return {}

    orig = (trade._connect_paper, data_feed.get_many_prices,
            data_feed.get_fundamentals, data_feed.network_busy)
    trade._connect_paper = lambda: (client, True)
    data_feed.get_many_prices = _gmp
    data_feed.get_fundamentals = lambda t, **kw: None
    try:
        data_feed.network_busy = lambda: True
        out = trade.fetch_positions()
        assert seen["allow_network"] is False, "busy pipeline -> cache-only pull"
        assert out["positions"][0]["symbol"] == "NMM"        # page still renders
        assert out["positions"][0]["current_price"] == 79.0  # Alpaca price intact

        data_feed.network_busy = lambda: False
        trade.fetch_positions()
        assert seen["allow_network"] is True, "free pipeline -> normal pull"
    finally:
        (trade._connect_paper, data_feed.get_many_prices,
         data_feed.get_fundamentals, data_feed.network_busy) = orig


def test_get_many_prices_settled_cache_served():
    """§6.37: a parquet hours past the 30-min freshness window is still served AS-IS when
    no market session has elapsed since it was written (post-close/weekend), yfinance
    untouched — and goes through the normal top-up when a session HAS elapsed. The
    calendar predicate is patched for determinism; its truth table is pinned by
    ``test_no_session_since_calendar``."""
    import os
    import tempfile
    import time as _time
    from unittest.mock import patch
    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed, triggers as trg

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=60)
    closes = [100.0 + i for i in range(60)]
    cached = pd.DataFrame({"Open": closes, "High": [c + 1 for c in closes],
                           "Low": [c - 1 for c in closes], "Close": closes,
                           "Volume": [1000] * 60}, index=idx)
    calls = []

    def fake_download(*a, **kw):
        calls.append(kw)
        return pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        with patch.object(dfeed, "PRICES_DIR", pdir), patch("yfinance.download",
                                                            fake_download):
            path = pdir / "AAPL.parquet"
            cached.to_parquet(path)
            old = _time.time() - 3 * 3600                  # 3h old — far past 30 min
            os.utime(path, (old, old))

            # No session since the write -> served as-is even at max_age_days ~ 30 min
            with patch.object(trg, "no_session_since", lambda *a, **k: True):
                got = dfeed.get_many_prices(["AAPL"], max_age_days=30 / (24 * 60),
                                            pause=0, retries=0)
            assert not calls, "settled cache must not touch yfinance"
            assert len(got["AAPL"]) == 60

            # A session HAS elapsed -> normal top-up path fires (fetch attempted)
            with patch.object(trg, "no_session_since", lambda *a, **k: False):
                got2 = dfeed.get_many_prices(["AAPL"], max_age_days=30 / (24 * 60),
                                             pause=0, retries=0)
            assert calls, "elapsed session must leave the settled gate"
            assert len(got2["AAPL"]) == 60                 # empty fetch degrades to cache


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


def test_atomic_parquet_replace_and_tmp_cleanup():
    """_atomic_to_parquet writes a sibling .tmp then os.replace's it over the target: a
    failed write leaves the OLD file byte-intact (the torn-file -> silent-network-refetch
    hole this closes) and no .tmp behind; a successful write also leaves no .tmp."""
    import tempfile
    from unittest.mock import patch

    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    df = pd.DataFrame({"Close": [1.0, 2.0]},
                      index=pd.bdate_range("2026-01-05", periods=2))
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "AAA.parquet"

        # success: file lands, no .tmp siblings
        dfeed._atomic_to_parquet(df, target)
        assert target.exists()
        assert not list(Path(tmp).glob("*.tmp"))
        before = target.read_bytes()

        # failure mid-write: to_parquet writes garbage then raises -> target UNCHANGED,
        # no .tmp left behind
        def _boom(self, path, *a, **kw):
            Path(path).write_bytes(b"torn")
            raise OSError("disk full")

        with patch.object(pd.DataFrame, "to_parquet", _boom):
            try:
                dfeed._atomic_to_parquet(df, target)
            except OSError:
                pass                                     # callers swallow; either is fine
        assert target.read_bytes() == before, "failed write must leave the old file intact"
        assert not list(Path(tmp).glob("*.tmp")), "no .tmp litter after a failed write"


def test_get_many_prices_threaded_cache_reads():
    """The cache-read pre-pass runs in a thread pool: every fresh-cached name is served
    with NO network touch, the returned dict preserves the input symbol order (assembly
    happens after the join), and the progress counter is strictly monotonic 1..N under
    the emit lock (label ORDER may interleave — consumers are order-insensitive)."""
    import tempfile
    from unittest.mock import patch

    import pandas as pd

    from src.stock_screener.cockpit import data_feed as dfeed

    def _ohlcv(close):
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=10)
        return pd.DataFrame({"Open": close, "High": close, "Low": close,
                             "Close": close, "Volume": 1000.0}, index=idx)

    syms = [f"T{i:02d}" for i in range(40)]
    dones, labels = [], []

    def prog(done, total, label):
        dones.append(done)
        labels.append(label)

    def _no_net(*a, **kw):
        raise AssertionError("network touched on a fully warm cache")

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp)
        for i, s in enumerate(syms):
            _ohlcv([100.0 + i] * 10).to_parquet(pdir / f"{s}.parquet")
        with patch.object(dfeed, "PRICES_DIR", pdir), patch("yfinance.download", _no_net):
            out = dfeed.get_many_prices(syms, max_age_days=7.0, pause=0, retries=0,
                                        progress=prog)

    assert list(out.keys()) == syms, "assembly must preserve input symbol order"
    assert all(float(out[s]["Close"].iloc[-1]) == 100.0 + i
               for i, s in enumerate(syms)), "each name must get ITS OWN frame"
    assert sorted(dones) == list(range(1, 41)) and dones == sorted(dones), \
        f"progress counter must be strictly monotonic 1..40, got {dones[:5]}…"
    assert all("cached (fresh)" in lb for lb in labels)
    assert {lb.split(":")[0] for lb in labels} == set(syms)


def test_data_feed_logs_one_summary_line_per_sweep():
    """The sweep summary is the record parquet mtimes cannot give: an all-cached sweep is
    proof the box deliberately did NOT download, where an unchanged mtime is
    indistinguishable from a sweep that never ran. Failures name their symbols (capped)
    because that is the half worth acting on."""
    import tempfile
    import time as _time
    from unittest.mock import patch

    from src.stock_screener.cockpit import cache as cachemod
    from src.stock_screener.cockpit import data_feed, runlog

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "logs"
        with patch.object(cachemod, "LOGS_DIR", d):
            data_feed._log_fetch("sweep", 4213, 342, 3870, 1, 3871, ["FBYDP"],
                                 _time.time() - 12.0)
            text = runlog.log_path(dir_path=d).read_text(encoding="utf-8")
            # Must happen INSIDE the tempdir: the module-level handler still holds
            # today's log open, and Windows refuses to unlink an open file, so the
            # cleanup below raises WinError 32 and aborts the whole suite. POSIX does
            # not care, which is why the Pi gate and CI stayed green through this.
            runlog.release_files()
    assert "4213 requested" in text and "cached 342" in text, text
    assert "failed 1" in text, text
    assert "FBYDP" in text, "a failed name must be identified, not just counted"
    assert text.isascii(), "log output must stay ASCII (journald-safe)"



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "data_feed"))
