"""Synthetic-data tests for src/stock_screener/hunt (the weekend-hunt pipeline).

Runs as a plain script (`python tests/test_hunt.py`) or under pytest, repo style.
No disk pickle, no network: a tiny fake ScanResult is built in memory. The most
important assertions are the rule boundaries — buy zone at exactly 0% / +5%,
approach at -3%, volume confirmation at 1.5x, earnings block at 21 days — since
mis-remembered boundaries are precisely why this pipeline exists.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stock_screener.cockpit.advisories import volume_dryup  # noqa: E402
from src.stock_screener.hunt import pipeline as pl  # noqa: E402

PASSED = 0


def ok(name: str, cond: bool) -> None:
    global PASSED
    assert cond, f"FAIL  {name}"
    PASSED += 1
    print(f"  PASS  {name}")


# --------------------------------------------------------------------------- #
def _payload(pivot: float, close: float, *, last_vol_mult=0.8, breakout=False,
             earnings_in=60, n_bars=260) -> dict:
    """Minimal payload with a flat synthetic tape ending at `close`.

    Volume is FLAT except for the final bar, scaled by ``last_vol_mult`` — so the
    confirmation ratio the pipeline computes IS that multiplier, exactly. The payload
    deliberately carries no ``levels["volume_ratio"]``: the gate must compute its own
    from the frame, and reading a pre-baked field is what let the hunt drift onto a
    different window than the trigger job."""
    idx = pd.bdate_range("2025-08-01", periods=n_bars)
    c = np.linspace(close * 0.7, close, n_bars)
    vol = np.full(n_bars, 1e6)
    vol[-1] = 1e6 * last_vol_mult
    df = pd.DataFrame({"Open": c * 0.999, "High": c * 1.01, "Low": c * 0.99,
                       "Close": c, "Volume": vol}, index=idx)
    cons = [{"peak_date": idx[-40], "trough_date": idx[-35], "peak_price": close * 1.05,
             "trough_price": close * 0.95, "drawdown_pct": 9.5, "volume_ratio": 0.8,
             "duration_days": 5, "number": 1}]
    return {
        "df": df,
        "levels": {"pivot": pivot, "stop": pivot * 0.97, "breakout_today": breakout},
        "vcp": {"contractions": cons},
        "dryup": volume_dryup(df, cons),
        "step2": {"score": 2, "available": True,
                  "checks": {"revenue_growth": True, "eps_growth": True,
                             "eps_accelerating": False, "margin_expanding": False}},
        "fundamentals": {"revenue_yoy": 25.0, "eps_yoy": 30.0},
        "earnings_in": earnings_in,
    }


def _bundle():
    # AAA 2% above pivot / BBB 2% below / CCC rs 69 (filtered) / DDD confirmed
    # breakout / EEE earnings-blocked
    payloads = {
        "AAA": _payload(100.0, 102.0),
        "BBB": _payload(100.0, 98.0),
        "CCC": _payload(50.0, 50.0),
        "DDD": _payload(20.0, 20.4, last_vol_mult=1.50, breakout=True),
        "EEE": _payload(10.0, 10.1, earnings_in=21),
    }
    cand = pd.DataFrame([
        {"ticker": "AAA", "tier": "A", "rs": 95, "vcp_quality": 90.0, "fund_score": 2},
        {"ticker": "BBB", "tier": "A", "rs": 80, "vcp_quality": 85.0, "fund_score": 3},
        {"ticker": "CCC", "tier": "A", "rs": 69, "vcp_quality": 99.0, "fund_score": 4},
        {"ticker": "DDD", "tier": "A", "rs": 75, "vcp_quality": 80.0, "fund_score": 1},
        {"ticker": "EEE", "tier": "A", "rs": 72, "vcp_quality": 70.0, "fund_score": 0},
        {"ticker": "ZZZ", "tier": "B", "rs": 99, "vcp_quality": 95.0, "fund_score": 4},
    ])
    result = SimpleNamespace(candidates=cand, payloads=payloads,
                             regime={"regime": "RISK-ON"}, n_scanned=6, n_passed=6)
    return pl.ScanBundle(result, completed_wall=0.0, key=("test", 8))


# --------------------------------------------------------------------------- #
def test_rs_floor_and_tier():
    b = _bundle()
    got = list(pl.candidates(b)["ticker"])
    ok("tier B excluded", "ZZZ" not in got)
    ok("rs 69 excluded by floor", "CCC" not in got)
    ok("floor keeps the rest", got == ["AAA", "BBB", "DDD", "EEE"])
    ok("min_rs is 70", pl.MIN_RS == 70)


def test_bucket_boundaries():
    ok("0% is in the buy zone", pl.bucket(0.0) == "buy_zone")
    ok("+5.0% still in the buy zone", pl.bucket(5.0) == "buy_zone")
    ok("+5.01% is past entry", pl.bucket(5.01) == "past_entry")
    ok("-0.01% is approaching, not buy zone", pl.bucket(-0.01) == "approaching")
    ok("-3.0% still approaching", pl.bucket(-3.0) == "approaching")
    ok("-3.01% is below", pl.bucket(-3.01) == "below")


def test_diagnostics_and_gates():
    b = _bundle()
    cand = pl.candidates(b)
    diag = pl.diagnostics(b, cand)
    ok("one diag row per candidate", len(diag) == 4)
    a = diag[diag.ticker == "AAA"].iloc[0]
    ok("vs_pivot computed from close/pivot", abs(a["vs_pivot_pct"] - 2.0) < 0.01)
    # §6.75: the typical day comes from the frame (H/L = ±1% -> a 2% true range), not the
    # vcp payload, so a pickle written before the payload carried it reads the same; the
    # 3%-below-pivot stop is 1.5 ordinary days away.
    ok("typical day computed from df", abs(a["day_range"] - 2.0) < 0.01)
    ok("stop room = pivot-to-stop over the typical day", abs(a["stop_room"] - 1.5) < 0.05)
    # §6.80: Step-1 reads come from the scan row; a scan from before they existed reads
    # None, never a KeyError.
    ok("step-1 reads absent from an older scan read as None",
       a["rs_trend"] is None and a["sma200_m"] is None and a["depth_vs_spy"] is None
       and a["industry"] is None)
    # §6.86: Code 33 and the inventory flag come from the payload's fundamentals
    # §6.100: the dry-up read comes from the scan payload, as the scan stored it
    ok("dry-up read from the payload",
       a["dryup_ratio"] is not None and a["step3"].startswith("DU ")
       and "dryup" not in diag.columns and "shakeout" not in diag.columns)
    ok("step3_summary empty without reads", pl.step3_summary(None) == "")
    ok("step-2 reads absent from older fundamentals read as None",
       a["code33"] is None and a["inv_flag"] is None and a["earn_react"] is None)
    # §6.87/§6.100: the reaction is the one the scan stored in the payload
    b2 = _bundle()
    b2.result.payloads["AAA"]["reaction"] = {"day_pct": 1.8, "flag": None}
    a2 = pl.diagnostics(b2, pl.candidates(b2)).set_index("ticker").loc["AAA"]
    ok("earnings reaction read from the payload", a2["earn_react"] == 1.8
       and a2["earn_flag"] is None)
    # §6.81: ADV through the shared helper, and the liquidity ceiling beside it
    _df = b.result.payloads["AAA"]["df"]
    _adv = float((_df["Close"] * _df["Volume"]).tail(20).mean())
    ok("adv_musd is the 20-day dollar volume in $M", abs(a["adv_musd"] - _adv / 1e6) < 0.01)
    ok("max_order_usd is 2% of the 20-day dollar volume",
       abs(a["max_order_usd"] - 0.02 * _adv) < 1.0)

    verdicts = {t: {"ticker": t, "verdict": "PASS", "notes": ""} for t in diag.ticker}
    g = pl.gates(diag, verdicts, min_fund=0)
    ok("AAA in buy zone", any(r["ticker"] == "AAA" for r in g["buy_zone"]))
    ok("BBB approaching", any(r["ticker"] == "BBB" for r in g["approaching"]))
    ok("DDD volume-confirmed at exactly 1.50x",
       [r["ticker"] for r in g["volume_confirmed"]] == ["DDD"])
    ok("the ratio is the prior-50 average, today's bar excluded",
       abs(float(diag[diag.ticker == "DDD"].iloc[0]["volume_ratio"]) - 1.50) < 1e-9)
    ok("hunt confirms on the shared doctrine ratio, not its own",
       pl.VOL_CONFIRM_RATIO == 1.5)
    d2 = diag.copy()
    d2.loc[d2.ticker == "DDD", "vs_pivot_pct"] = 6.0
    g_ext = pl.gates(d2, verdicts, min_fund=0)
    ok("a breakout past the buy zone is not a confirmation (no chasing)",
       g_ext["volume_confirmed"] == []
       and any(r["ticker"] == "DDD" for r in g_ext["past_entry"]))
    ok("EEE blocked at exactly 21 days",
       [r["ticker"] for r in g["earnings_blocked"]] == ["EEE"])
    ok("blocked name appears in no bucket",
       all(r["ticker"] != "EEE" for k in ("buy_zone", "approaching", "below", "past_entry")
           for r in g[k]))

    g2 = pl.gates(diag, verdicts, min_fund=2)
    ok("min_fund=2 drops DDD (F=1) from buckets",
       all(r["ticker"] != "DDD" for k in ("buy_zone", "approaching") for r in g2[k]))

    verdicts["AAA"]["verdict"] = "FAIL"
    g3 = pl.gates(diag, verdicts)
    ok("FAIL names never reach a bucket",
       all(r["ticker"] != "AAA" for r in g3["buy_zone"]))


def test_verdict_bookkeeping():
    b = _bundle()
    diag = pl.diagnostics(b, pl.candidates(b))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "verdicts.csv"
        pl.append_verdicts(p, [{"ticker": "AAA", "verdict": "PASS", "notes": "x"},
                               {"ticker": "BBB", "verdict": "PASS-", "notes": "y"}])
        probs = pl.validate_verdicts(p, diag)
        ok("missing tickers reported", any("DDD" in q and "missing" in q for q in probs))
        pl.append_verdicts(p, [{"ticker": "DDD", "verdict": "FAIL", "notes": ""},
                               {"ticker": "EEE", "verdict": "FAIL", "notes": ""},
                               {"ticker": "EEE", "verdict": "FAIL", "notes": "dup"}])
        probs = pl.validate_verdicts(p, diag)
        ok("duplicate detected", any("EEE" in q and "2 verdicts" in q for q in probs))
        try:
            pl.append_verdicts(p, [{"ticker": "Q", "verdict": "BUY", "notes": ""}])
            ok("bad verdict rejected", False)
        except pl.HuntError:
            ok("bad verdict rejected", True)


def test_append_batch():
    """`append-verdicts --file` is how an unattended review records a batch: its tool
    allowlist cannot cover a free-form shell append. A rejected batch writes nothing, so
    re-running one after a crash cannot double-count a ticker."""
    b = _bundle()
    diag = pl.diagnostics(b, pl.candidates(b))
    with tempfile.TemporaryDirectory() as td:
        p, batch = Path(td) / "verdicts.csv", Path(td) / "verdicts_batch_01.csv"

        def rejected(text: str) -> bool:
            batch.write_text(text, encoding="utf-8")
            before = p.read_text(encoding="utf-8") if p.exists() else None
            try:
                pl.append_batch(p, batch, diag)
                return False
            except pl.HuntError:
                return before == (p.read_text(encoding="utf-8") if p.exists() else None)

        batch.write_text('ticker,verdict,notes\nAAA,PASS,"OK; tight, quiet"\n'
                         'BBB,PASS-,CAV:loose; y\n', encoding="utf-8")
        ok("batch appended", pl.append_batch(p, batch, diag) == 2)
        ok("a quoted comma survives the round trip",
           pl.read_verdicts(p)["AAA"]["notes"] == "OK; tight, quiet")
        ok("re-running a batch is rejected, nothing written",
           rejected("ticker,verdict,notes\nAAA,PASS,OK; again\nDDD,FAIL,DQ:no-base; z\n"))
        ok("a non-candidate is rejected", rejected("ticker,verdict,notes\nCCC,PASS,OK; x\n"))
        ok("a ticker repeated in the batch is rejected",
           rejected("ticker,verdict,notes\nDDD,PASS,OK; x\nDDD,FAIL,DQ:no-base; y\n"))
        ok("a bad verdict is rejected", rejected("ticker,verdict,notes\nDDD,BUY,OK; x\n"))
        ok("a note without a code is rejected", rejected("ticker,verdict,notes\nDDD,FAIL,z\n"))
        ok("a caveat on a FAIL is rejected",
           rejected("ticker,verdict,notes\nDDD,FAIL,CAV:loose; z\n"))
        ok("a batch without the header is rejected", rejected("DDD,PASS,OK; x\n"))
        ok("an empty batch is rejected", rejected(""))
        ok("a missing batch file is rejected",
           _raises(lambda: pl.append_batch(p, Path(td) / "absent.csv", diag)))
        batch.write_text("ticker,verdict,notes\nDDD,FAIL,DQ:expanding; z\n"
                         "EEE,FAIL,DQ:v-shape; w\n", encoding="utf-8")
        pl.append_batch(p, batch, diag)
        ok("batches add up to a clean validation", pl.validate_verdicts(p, diag) == [])


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except pl.HuntError:
        return True


def test_report_builds():
    import json
    from src.stock_screener.hunt.report import build_report
    b = _bundle()
    diag = pl.diagnostics(b, pl.candidates(b))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        diag.to_csv(d / "diagnostics.csv", index=False)
        pl.append_verdicts(d / "verdicts.csv",
                           [{"ticker": t, "verdict": v, "notes": "n"}
                            for t, v in [("AAA", "PASS"), ("BBB", "PASS"),
                                         ("DDD", "PASS-"), ("EEE", "FAIL")]])
        (d / "meta.json").write_text(json.dumps(
            {"scan_time": "2026-08-23 15:50", "regime": {"regime": "RISK-ON",
             "phase2_pct": 30.5}, "n_scanned": 6, "n_passed_template": 6,
             "n_tier_a": 5, "n_eligible": 4, "min_rs": 70}))
        out = build_report(d, min_fund=0)
        html = out.read_text(encoding="utf-8")
        # §6.89: the fixture's step2 holds the four older checks, so F reads out of 4
        ok("F out of the payload's check count", "</b>/4</td>" in html
           and "</b>/8</td>" not in html)
        eight = diag.copy()
        eight["f_max"] = 8
        eight.to_csv(d / "diagnostics.csv", index=False)
        ok("F out of 8 on a scan with the eight checks",
           "</b>/8</td>" in build_report(d, min_fund=6).read_text(encoding="utf-8"))
        diag.to_csv(d / "diagnostics.csv", index=False)
        for frag in ("Weekend Hunt", "In the buy zone", "Approaching pivot",
                     "Volume-confirmed", "Step-2 fundamentals", "Code&nbsp;33", "Full review",
                     "RS line", "Groups among PASS names", "AAA", "EEE"):
            ok(f"report contains {frag!r}", frag in html)


def test_narrative_markdown():
    """The reviewer's narrative.md is rendered by the report's own small Markdown
    renderer: no dependency the Pi image lacks, and the text is escaped first, so a
    verdict note cannot inject markup."""
    from src.stock_screener.hunt.report import md_to_html
    html = md_to_html("# Regime\n\nA **weak** *risk-on* tape with `MU` leading.\n\n"
                      "- first\n- second\n\n1. one\n2) two\n\n"
                      "| Ticker | F |\n|---|---|\n| MU | 6 |\n\n---\n\n"
                      "<script>x</script> & done")
    for frag in ("<h3>Regime</h3>", "<b>weak</b>", "<i>risk-on</i>", "<code>MU</code>",
                 "<ul><li>first</li><li>second</li></ul>", "<ol><li>one</li><li>two</li></ol>",
                 "<th>Ticker</th>", "<td>MU</td><td>6</td>", "<hr>",
                 "&lt;script&gt;x&lt;/script&gt; &amp; done"):
        ok(f"markdown renders {frag!r}", frag in html)
    ok("a lone asterisk is not emphasis", "<i>" not in md_to_html("5 * 3 = 15"))
    ok("empty narrative renders nothing", md_to_html("\n\n") == "")


def test_report_mirror():
    """report.html is the Friday run's deliverable: the narrative under "Reviewer's read",
    the sheets under their verdicts, and a copy of the page with its sheets in the docs
    folder. None of it needs matplotlib, so this runs on the deploy gate too."""
    import json
    from src.stock_screener.hunt import report as rp
    b = _bundle()
    # The report's watchlist audit MUST NOT depend on the watchlist on this machine's disk.
    real_watchlist = pl._watchlist_tickers
    pl._watchlist_tickers = lambda: ["AAA", "QQQ"]
    try:
        diag = pl.diagnostics(b, pl.candidates(b))
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "hunt"
            d.mkdir()
            diag.to_csv(d / "diagnostics.csv", index=False)
            pl.append_verdicts(d / "verdicts.csv",
                               [{"ticker": "AAA", "verdict": "PASS", "notes": "$2M ADV caps at $40k"},
                                {"ticker": "BBB", "verdict": "PASS", "notes": "n"},
                                {"ticker": "DDD", "verdict": "PASS-", "notes": "n"}])
            (d / "meta.json").write_text(json.dumps(
                {"scan_time": "2026-08-23 15:50", "regime": {"regime": "RISK-ON"},
                 "n_scanned": 6, "n_passed_template": 6, "n_tier_a": 5, "n_eligible": 4,
                 "min_rs": 70}))
            st = rp.load_state(d, 0)
            ok("the state's buckets come from pipeline.gates",
               [r["ticker"] for r in st.buckets["buy_zone"]] == ["AAA"]
               and [r["ticker"] for r in st.buckets["approaching"]] == ["BBB"])
            html = rp.build_report(d).read_text(encoding="utf-8")
            ok("no narrative, no sheets: neither section is rendered",
               "Reviewer&rsquo;s read" not in html and 'class="sheet"' not in html)

            charts = d / "charts"
            charts.mkdir()
            for name in ("sheet_001.png", "sheet_002.png"):
                (charts / name).write_bytes(b"not a real PNG; only its name matters")
            ok("no index, sheet count off the default size: sheets kept, tickers unknown",
               [tk for _, tk in rp.review_sheets(d, st.diag_rows)] == [None, None])
            (charts / "sheets.json").write_text(json.dumps(
                {"sheet_001.png": ["AAA", "BBB", "DDD"], "sheet_002.png": ["EEE"],
                 "sheet_003.png": ["GONE"]}))
            ok("the index names each sheet's tickers; a sheet missing on disk is dropped",
               [(p.name, tk) for p, tk in rp.review_sheets(d, st.diag_rows)]
               == [("sheet_001.png", ["AAA", "BBB", "DDD"]), ("sheet_002.png", ["EEE"])])
            (d / "narrative.md").write_text("# Read\n\nTwo **names** stand out.\n",
                                            encoding="utf-8")
            html = rp.build_report(d).read_text(encoding="utf-8")
            ok("report embeds the narrative and the sheets",
               "Reviewer&rsquo;s read" in html and "<b>names</b>" in html
               and 'src="charts/sheet_001.png"' in html and "Sheet 2 of 2" in html
               and html.count('class="sheet"') == 2)
            ok("an unreviewed name on a sheet reads UNREVIEWED",
               "UNREVIEWED" in html and "KeyError" not in html)

            docs = Path(td) / "docs" / "2026-08-23"
            (docs / "charts").mkdir(parents=True)
            (docs / "charts" / "sheet_009.png").write_bytes(b"old")
            page = rp.mirror_report(d, docs)
            ok("the docs copy is the page plus the sheets it shows, and nothing stale",
               page.read_text(encoding="utf-8") == html
               and sorted(p.name for p in (docs / "charts").iterdir())
               == ["sheet_001.png", "sheet_002.png"])
            try:
                rp.mirror_report(Path(td) / "nowhere", docs)
                ok("mirroring an unbuilt report raises", False)
            except FileNotFoundError:
                ok("mirroring an unbuilt report raises", True)
    finally:
        pl._watchlist_tickers = real_watchlist


def test_load_state_tolerates_gaps():
    """A hunt dir is read on the Pi by the Weekend Hunt page, so the loader MUST accept
    what earlier hunts wrote: an empty ``adv_musd`` cell (the pipeline's None for a dead
    tape) and a diagnostics file from before the eight-check columns existed."""
    import json
    from src.stock_screener.hunt import report as rp
    b = _bundle()
    diag = pl.diagnostics(b, pl.candidates(b))
    real_watchlist = pl._watchlist_tickers
    pl._watchlist_tickers = lambda: []
    try:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "meta.json").write_text(json.dumps({"scan_time": "2026-08-23 15:50",
                                                     "regime": {}, "min_rs": 70}))
            gappy = diag.copy()
            gappy.loc[gappy.ticker == "AAA", "adv_musd"] = None
            gappy.loc[gappy.ticker == "AAA", "max_order_usd"] = None
            gappy.to_csv(d / "diagnostics.csv", index=False)
            st = rp.load_state(d)
            row = next(r for r in st.diag_rows if r["ticker"] == "AAA")
            ok("an empty ADV cell reads as None", row["adv_musd"] is None
               and row["max_order_usd"] is None)
            old = diag.drop(columns=["f_max", "f_code33", "f_fy", "f_est", "f_react",
                                     "stop", "max_order_usd", "full_position_usd",
                                     "size_capped"])
            old.to_csv(d / "diagnostics.csv", index=False)
            st = rp.load_state(d)
            row = st.diag_rows[0]
            ok("a pre-eight-check hunt loads with defaults",
               row["f_max"] == 8 and row["f_code33"] == 0 and row["stop"] is None
               and row["size_capped"] is None and row["full_position_usd"] is None)
            html = rp.build_report(d).read_text(encoding="utf-8")
            ok("and still renders a report", "Weekend Hunt" in html)
    finally:
        pl._watchlist_tickers = real_watchlist


_RSS_YAHOO = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Yahoo</title>
<item><title>Acme wins a $2B contract</title><link>https://finance.yahoo.com/a1</link>
<pubDate>Tue, 06 Oct 2026 12:30:04 +0000</pubDate>
<description>&lt;p&gt;Acme &amp;amp; Co signed&lt;/p&gt; a multi-year deal.</description></item>
<item><title>Acme (ACME) moves higher</title><link>https://finance.yahoo.com/a2</link>
<pubDate>Mon, 05 Oct 2026 09:00:00 +0000</pubDate></item>
</channel></rss>"""
_RSS_GOOGLE = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Acme wins a $2B contract - Reuters</title><link>https://news.google.com/g1</link>
<pubDate>Tue, 06 Oct 2026 13:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source>
<description>&lt;a href="x"&gt;Acme wins&lt;/a&gt;</description></item>
<item><title>Why Acme could double - Motley Fool</title><link>https://news.google.com/g2</link>
<pubDate>not a date</pubDate><source url="https://fool.com">Motley Fool</source></item>
</channel></rss>"""


def test_news_feeds():
    """The catalyst read judges from ``news/<ticker>.json`` alone, so the fetcher MUST
    yield a file per ticker whatever the feeds do: a failing feed is an error line, a
    ticker with nothing stays an empty list, and duplicate titles across feeds collapse."""
    import json
    from src.stock_screener.hunt import news as nw
    y = nw.parse_feed(_RSS_YAHOO, "Yahoo Finance")
    ok("yahoo items parse with dates, links and clean summaries",
       [h["title"] for h in y] == ["Acme wins a $2B contract", "Acme (ACME) moves higher"]
       and y[0]["date"] == "2026-10-06" and y[0]["url"] == "https://finance.yahoo.com/a1"
       and y[0]["summary"] == "Acme & Co signed a multi-year deal."
       and y[0]["publisher"] == "Yahoo Finance")
    g = nw.parse_feed(_RSS_GOOGLE, "Google News")
    ok("google items take the publisher from <source> and drop the title suffix",
       g[0]["title"] == "Acme wins a $2B contract" and g[0]["publisher"] == "Reuters"
       and g[1]["date"] == "" and g[1]["publisher"] == "Motley Fool")
    ok("a non-RSS document parses to nothing", nw.parse_feed(b"<html>no</html>", "x") == []
       and nw.parse_feed(b"\x00garbage", "x") == [])

    def fake_fetch(url):
        if "ZZZ" in url:
            raise RuntimeError("feed down")
        return _RSS_GOOGLE if "google" in url else _RSS_YAHOO

    news = nw.fetch_news(["ACME", "ZZZ"], limit=2, fetch=fake_fetch)
    acme = news["ACME"]
    ok("two feeds merge newest first, de-duplicated by title, capped at the limit",
       [h["title"] for h in acme["headlines"]] == ["Acme wins a $2B contract",
                                                   "Acme (ACME) moves higher"]
       and acme["errors"] == [])
    ok("a ticker whose feeds fail keeps an empty list and the errors, no exception",
       news["ZZZ"]["headlines"] == [] and len(news["ZZZ"]["errors"]) == 2)
    with tempfile.TemporaryDirectory() as td:
        paths = nw.write_news(Path(td), news)
        ok("one json per ticker under news/",
           sorted(p.name for p in paths) == ["ACME.json", "ZZZ.json"]
           and json.loads((Path(td) / "news" / "ACME.json").read_text(encoding="utf-8"))
           ["headlines"][0]["title"] == "Acme wins a $2B contract")


def test_report_catalyst_line():
    """The catalyst read is a label beside a PASS name, never a gate: the report shows it
    under the row's notes when ``catalyst.json`` has it and ignores a malformed file."""
    import json
    from src.stock_screener.hunt import report as rp
    b = _bundle()
    diag = pl.diagnostics(b, pl.candidates(b))
    real_watchlist = pl._watchlist_tickers
    pl._watchlist_tickers = lambda: []
    try:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            diag.to_csv(d / "diagnostics.csv", index=False)
            pl.append_verdicts(d / "verdicts.csv",
                               [{"ticker": "AAA", "verdict": "PASS", "notes": "tight"}])
            (d / "meta.json").write_text(json.dumps({"scan_time": "2026-08-23 15:50",
                                                     "regime": {}, "min_rs": 70}))
            (d / "catalyst.json").write_text(json.dumps({
                "aaa": {"category": "contract", "sentiment": "positive",
                        "summary": "A $2B award <reported> Monday.",
                        "sources": [{"title": "t", "publisher": "Reuters",
                                     "date": "2026-10-06", "url": "https://r/1"}]},
                "BBB": "not a dict"}), encoding="utf-8")
            cats = rp.load_catalysts(d)
            ok("catalysts load keyed by upper-case ticker, non-dict entries dropped",
               list(cats) == ["AAA"] and cats["AAA"]["category"] == "contract")
            html = rp.build_report(d).read_text(encoding="utf-8")
            ok("the buy-zone row carries the catalyst line, escaped",
               "contract &middot; positive" in html and "&lt;reported&gt;" in html)
            (d / "catalyst.json").write_text("{not json", encoding="utf-8")
            ok("a malformed file reads as no catalysts and the report still builds",
               rp.load_catalysts(d) == {}
               and "contract &middot;" not in rp.build_report(d).read_text(encoding="utf-8"))
            ok("a missing file reads as no catalysts",
               rp.load_catalysts(Path(td) / "nowhere") == {})
    finally:
        pl._watchlist_tickers = real_watchlist


def test_equity_and_full_position():
    """A full position is 5% of the paper equity the Pi saves beside the scan (user
    decision, 2026-10-09). The reviewer had guessed an account of about $100k, so thin
    names were marked PASS- against the wrong size. An old or missing equity reads as
    unknown, never as a stale number."""
    import datetime as dt
    import json
    now = dt.datetime(2026, 10, 10, 9, 0)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "account.json"
        ok("no file: equity unknown", pl.load_equity(f, now) is None)
        f.write_text(json.dumps({"equity": 100000.0, "as_of": "2026-10-09T16:30:00"}))
        ok("a fresh file gives its equity", pl.load_equity(f, now) == 100000.0)
        f.write_text(json.dumps({"equity": 100000.0, "as_of": "2026-10-06T16:30:00"}))
        ok("an equity older than 3 days is unknown", pl.load_equity(f, now) is None)
        f.write_text(json.dumps({"equity": 0, "as_of": "2026-10-09T16:30:00"}))
        ok("a non-positive equity is unknown", pl.load_equity(f, now) is None)
        f.write_text("{oops")
        ok("an unreadable file is unknown", pl.load_equity(f, now) is None)

    b = _bundle()
    cand = pl.candidates(b)
    base = pl.diagnostics(b, cand).set_index("ticker")
    ok("no equity: no full position and no cap flag",
       base["full_position_usd"].isna().all() and base["size_capped"].isna().all())
    cap = float(base.loc["AAA", "max_order_usd"])
    for label, equity, want in (("below", cap / 0.05 - 2000, 0), ("at", cap / 0.05, 0),
                                ("above", cap / 0.05 + 2000, 1)):
        row = pl.diagnostics(b, cand, equity=equity).set_index("ticker").loc["AAA"]
        ok(f"a full position {label} the order ceiling: size_capped={want}",
           int(row["size_capped"]) == want
           and row["full_position_usd"] == round(0.05 * equity))


def test_note_codes():
    """Each verdict note opens with codes from a fixed list, so a verdict that changes
    from one week to the next shows which reason changed."""
    ok("a good PASS", pl.parse_note_codes("PASS", "OK; 12-8-4 legs") == [])
    ok("a good PASS- with two caveats under one prefix",
       pl.parse_note_codes("PASS-", "CAV:loose,no-dryup; 12-9-9 legs, DU 1.1x") == [])
    ok("each caveat with its own prefix", pl.parse_note_codes("PASS-", "CAV:loose,CAV:late") == [])
    ok("a good FAIL", pl.parse_note_codes("FAIL", "DQ:v-shape; straight up") == [])
    ok("no code", pl.parse_note_codes("PASS", "nice base") != [])
    ok("an empty note", pl.parse_note_codes("PASS", "") == ["no code"])
    ok("an unknown code", any("unknown" in q for q in pl.parse_note_codes("FAIL", "DQ:ugly; x")))
    ok("a caveat on a FAIL", any("belongs to PASS-" in q
                                 for q in pl.parse_note_codes("FAIL", "CAV:loose; x")))
    ok("a disqualifier on a PASS-", any("belongs to FAIL" in q
                                       for q in pl.parse_note_codes("PASS-", "DQ:no-base; x")))
    ok("two disqualifiers on a FAIL", any("exactly one" in q for q in
                                          pl.parse_note_codes("FAIL", "DQ:no-base,v-shape; x")))
    ok("text after the semicolon may hold commas and colons",
       pl.parse_note_codes("PASS-", "CAV:thin; ADV $2M, cap: $40k") == [])


def test_compare_verdicts():
    a = {"AAA": {"verdict": "PASS", "notes": "OK"}, "BBB": {"verdict": "PASS-", "notes": "x"},
         "CCC": {"verdict": "FAIL", "notes": "y"}, "DDD": {"verdict": "PASS", "notes": ""}}
    b = {"AAA": {"verdict": "PASS", "notes": "OK"}, "BBB": {"verdict": "FAIL", "notes": "z"},
         "CCC": {"verdict": "FAIL", "notes": "y"}, "EEE": {"verdict": "PASS", "notes": ""}}
    c = pl.compare_verdicts(a, b)
    ok("compare counts the shared names", c["shared"] == 3)
    ok("compare overall agreement", c["agree_pct"] == 66.7)
    ok("compare agreement where either said PASS- or FAIL",
       c["pass_minus_or_fail"] == {"n": 2, "agree_pct": 50.0})
    ok("compare lists the changed name with both verdicts",
       [(x["ticker"], x["a"], x["b"]) for x in c["changed"]] == [("BBB", "PASS-", "FAIL")])
    ok("nothing shared: agreement unknown", pl.compare_verdicts(a, {})["agree_pct"] is None)


# ---- the per-ticker reviewer ---------------------------------------------- #
def _form(**kw) -> dict:
    from src.stock_screener.hunt import reviewer as rv
    f = {"pattern": "vcp", "swings_match": "yes", "right_side": "orderly_tightening",
         "final_area": "tight", "volume": "drying_up", "pivot": "clear",
         "character": "quick_recoveries", "base_count": "first_or_second",
         "swings_seen": [], "sure": True}
    f.update(kw)
    f["evidence"] = {q: f"why {q}" for q in rv.FORM}
    return f


def _facts(**kw) -> dict:
    f = {"swings": [], "legs": [{"from": "A", "to": "B", "depth_pct": 20.0},
                                {"from": "C", "to": "D", "depth_pct": 10.0},
                                {"from": "E", "to": "F", "depth_pct": 5.0}],
         "leg_ratios": [0.5, 0.5], "dryup_ratio": 0.6, "closed_below_200_in_base": False}
    f.update(kw)
    return f


def test_reviewer_form_and_schema():
    from src.stock_screener.hunt import reviewer as rv
    sch = rv.schema()
    ok("the schema asks every form question plus swings, evidence and sure",
       set(sch["required"]) == set(rv.FORM) | {"swings_seen", "evidence", "sure"}
       and all(sch["properties"][q]["enum"] == list(a) for q, a in rv.FORM.items()))
    ok("a full answer validates", rv.validate_form(_form()) == [])
    ok("an answer off the list is caught", rv.validate_form(_form(volume="dry")) != [])
    bad = _form()
    del bad["evidence"]["pivot"]
    ok("missing evidence is caught", rv.validate_form(bad) != [])
    ok("a non-object is caught", rv.validate_form(None) == ["not an object"])
    ok("the prompt hash is stable", rv.prompt_hash() == rv.prompt_hash()
       and len(rv.prompt_hash()) == 12)


def test_reviewer_facts_hide_the_gates():
    """The reviewer judges the base only. RS, F, the quality score, the distance to the
    pivot, days to earnings and dollar volume decide nothing in the verdict, so they are
    never shown to it, and neither is the ticker."""
    from src.stock_screener.hunt import reviewer as rv
    b = _bundle()
    diag = pl.diagnostics(b, pl.candidates(b))
    row = diag.set_index("ticker").loc["AAA"].to_dict()
    facts = rv.fact_sheet(b.result.payloads["AAA"], row)
    ok("swings are lettered peak then trough",
       [(s["label"], s["kind"]) for s in facts["swings"]] == [("A", "peak"), ("B", "trough")])
    ok("legs carry depth, length and volume", facts["legs"][0]["depth_pct"] == 9.5
       and facts["legs"][0]["bars"] == 6 and facts["legs"][0]["volume_x50"] is not None)
    for key in ("rs", "fund", "q", "vs_pivot_pct", "earnings_in", "adv_musd", "max_order_usd",
                "size_capped", "pivot", "close"):
        ok(f"the facts leave out {key}", key not in facts)
    prompt = rv.build_prompt(Path("w.png"), Path("d.png"), facts)
    ok("the prompt names both images and not the ticker",
       "w.png" in prompt and "d.png" in prompt and "AAA" not in prompt)
    ok("the prompt shows the measured legs", "A->B  -9.5%" in prompt)
    ok("an unmeasured value is left out of the prompt",
       "Base depth over the market" not in prompt)


def test_verdict_table():
    from src.stock_screener.hunt import reviewer as rv
    v = rv.verdict_from_form
    ok("a clean form is PASS", v(_form(), _facts(), {})[:2] == ("PASS", ["OK"]))
    ok("closing below the 200-day fails whatever the form says",
       v(_form(), _facts(closed_below_200_in_base=True), {})[:2] == ("FAIL", ["DQ:below-200"]))
    ok("a stop inside one typical day is a bronco", v(_form(), _facts(), {"stop_room": 0.9})[1]
       == ["DQ:bronco"])
    ok("a stop of exactly one typical day is not", v(_form(), _facts(), {"stop_room": 1.0})[0]
       == "PASS")
    for kw, code in (({"pattern": "no_base"}, "DQ:no-base"),
                     ({"right_side": "wider_leg"}, "DQ:expanding"),
                     ({"right_side": "v_shape"}, "DQ:v-shape"),
                     ({"volume": "heavy_selling"}, "DQ:distribution"),
                     ({"character": "wild_swings"}, "DQ:bronco")):
        ok(f"{kw} fails with {code}", v(_form(**kw), _facts(), {})[:2] == ("FAIL", [code]))
    ok("several disqualifiers name the first in order",
       v(_form(pattern="no_base", volume="heavy_selling"), _facts(), {})[1] == ["DQ:no-base"])
    for kw, code in (({"final_area": "loose"}, "CAV:loose"),
                     ({"final_area": "not_formed"}, "CAV:pivot"),
                     ({"pivot": "several"}, "CAV:pivot"), ({"pivot": "none"}, "CAV:pivot"),
                     ({"volume": "mixed"}, "CAV:no-dryup"),
                     ({"base_count": "fourth_or_later"}, "CAV:late")):
        ok(f"{kw} is PASS- with {code}", v(_form(**kw), _facts(), {})[:2] == ("PASS-", [code]))
    ok("a dry-up ratio of exactly 1.0 is no dry-up",
       v(_form(), _facts(dryup_ratio=1.0), {})[1] == ["CAV:no-dryup"])
    ok("0.99 is a dry-up", v(_form(), _facts(dryup_ratio=0.99), {})[0] == "PASS")
    ok("a capped order size is thin", v(_form(), _facts(), {"size_capped": 1})[1] == ["CAV:thin"])
    ok("an unknown order size adds nothing", v(_form(), _facts(), {"size_capped": None})[0]
       == "PASS")
    ok("caveats add up, each once",
       v(_form(final_area="not_formed", pivot="none", volume="mixed"), _facts(dryup_ratio=1.2),
         {"size_capped": 1})[1] == ["CAV:pivot", "CAV:no-dryup", "CAV:thin"])
    ok("swings and character grinding change nothing",
       v(_form(swings_match="no", character="grinding"), _facts(), {})[0] == "PASS")


def test_reviewer_votes_and_checks():
    """Each question takes the majority of a ticker's answers, a three-way split takes the
    answer that weighs most against the base, and anything the user should look at is
    listed."""
    from src.stock_screener.hunt import reviewer as rv
    ok("two identical answers are settled", rv.settled([_form(), _form()]))
    ok("one differing question is not", not rv.settled([_form(), _form(pivot="several")]))
    ok("one answer is not", not rv.settled([_form()]))
    f, sure = rv.combine([_form(volume="mixed"), _form(volume="drying_up"),
                          _form(volume="mixed")])
    ok("the majority answer wins", f["volume"] == "mixed" and sure)
    f, _ = rv.combine([_form(volume="mixed"), _form(volume="drying_up"),
                       _form(volume="heavy_selling")])
    ok("a three-way split takes the worst answer", f["volume"] == "heavy_selling")
    ok("evidence comes from a vote that gave the chosen answer",
       f["evidence"]["volume"] == "why volume")

    d = rv.decide([_form()] * 3, _facts(), {})
    ok("unanimous and sure: nothing to check", d["needs_check"] == [] and d["verdict"] == "PASS")
    d = rv.decide([_form(), _form(), _form(right_side="v_shape")], _facts(), {})
    ok("one dissenting FAIL is listed as a split",
       d["verdict"] == "PASS" and d["needs_check"] == ["answers split FAIL/PASS"])
    d = rv.decide([_form(), _form(sure=False), _form()], _facts(), {})
    ok("one unsure answer is listed", "unsure" in d["needs_check"])
    d = rv.decide([_form()], _facts(), {}, wanted=3)
    ok("too few answers are listed", d["needs_check"] == ["only 1 of 3 answers"])
    def legs(*depths):
        return _facts(legs=[{"depth_pct": x} for x in depths])

    d = rv.decide([_form()] * 3, legs(20.0, 10.0, 12.0), {})
    ok("tightening against a measured leg 1.2x the one before is a conflict",
       any("clearly deeper than the one before" in c for c in d["needs_check"]))
    ok("a leg exactly 1.10x the one before is still tightening",
       rv.conflicts(_form(), legs(20.0, 10.0, 11.0)) == [])
    d = rv.decide([_form(right_side="wider_leg")] * 3, _facts(), {})
    ok("a wider leg against shrinking measured legs is a conflict",
       any("no measured leg is clearly deeper" in c for c in d["needs_check"]))
    ok("so is a wider leg against legs that barely shrink (0.99x)",
       rv.conflicts(_form(right_side="wider_leg"), legs(7.9, 7.8)) != [])
    ok("a wider leg against a leg 1.2x the one before is not",
       rv.conflicts(_form(right_side="wider_leg"), legs(10.0, 12.0)) == [])
    d = rv.decide([_form()] * 3, _facts(dryup_ratio=1.3), {})
    ok("drying up against a dry-up ratio over 1 is a conflict",
       any("1.30x its average" in c for c in d["needs_check"]))
    for fm, fc, row in ((_form(), _facts(), {}),
                        (_form(volume="mixed", pivot="several"), _facts(), {"size_capped": 1}),
                        (_form(right_side="v_shape"), _facts(), {}),
                        (_form(), _facts(closed_below_200_in_base=True), {})):
        d = rv.decide([fm, fm, dict(fm, sure=False)], fc, row)
        ok(f"the {d['verdict']} note passes the code check",
           pl.parse_note_codes(d["verdict"], d["notes"]) == [] and "Check: unsure" in d["notes"])


def test_reviewer_measures_named_swings():
    """When the reviewer says the lettered points are wrong, it names its own dates and
    the code measures the legs from them."""
    from src.stock_screener.hunt import reviewer as rv
    idx = pd.bdate_range("2026-06-01", periods=30)
    hi = np.full(30, 100.0)
    lo = np.full(30, 99.0)
    hi[5], lo[10], hi[15], lo[20] = 110.0, 88.0, 108.0, 99.0 - 1.0
    df = pd.DataFrame({"High": hi, "Low": lo, "Close": lo + 0.5}, index=idx)
    pts = [{"date": idx[4].strftime("%Y-%m-%d"), "kind": "peak"},
           {"date": idx[11].strftime("%Y-%m-%d"), "kind": "trough"},
           {"date": idx[15].strftime("%Y-%m-%d"), "kind": "peak"},
           {"date": idx[20].strftime("%Y-%m-%d"), "kind": "trough"},
           {"date": "not a date", "kind": "peak"}, {"date": "2030-01-01", "kind": "peak"}]
    legs = rv.measure_swings(df, pts)
    ok("a named date snaps to the extreme within two bars",
       legs[0]["peak_date"] == idx[5].strftime("%Y-%m-%d")
       and legs[0]["trough_date"] == idx[10].strftime("%Y-%m-%d"))
    ok("depths come from the snapped highs and lows",
       [lg["depth_pct"] for lg in legs] == [20.0, 9.3])
    ok("an unreadable or outside date is dropped", len(legs) == 2)
    form = _form(swings_match="no", swings_seen=pts[:4])
    ok("conflicts use the re-measured legs",
       rv.conflicts(form, _facts(legs=[{"depth_pct": 5.0}, {"depth_pct": 9.0}]), df) == [])


def _review_folder(td: Path):
    """A hunt folder with diagnostics and each ticker's review files, as review-charts
    leaves it (the images are stand-ins: only the caller would read them)."""
    import json
    from src.stock_screener.hunt import reviewer as rv
    b = _bundle()
    diag = pl.diagnostics(b, pl.candidates(b))
    for r in diag.to_dict("records"):
        weekly, daily, facts = rv.review_paths(td, r["ticker"])
        weekly.parent.mkdir(parents=True, exist_ok=True)
        weekly.write_bytes(b"png")
        daily.write_bytes(b"png")
        facts.write_text(json.dumps(rv.fact_sheet(b.result.payloads[r["ticker"]], r)))
    return diag


def test_run_review():
    """The review loop asks each ticker until it has its answers, records every answer
    as it arrives so a rerun resumes, and writes one coded verdict per ticker."""
    import json
    from src.stock_screener.hunt import reviewer as rv
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        diag = _review_folder(d)
        (d / "review" / "BBB_daily.png").unlink()
        ok("review refuses a folder missing a review chart",
           _raises(lambda: rv.run_review(d, diag, lambda p, s: {}, log=lambda m: None)))

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        diag = _review_folder(d)
        bbb = rv.review_paths(d, "BBB")[2]
        bbb.write_text(json.dumps({**json.loads(bbb.read_text()),
                                   "closed_below_200_in_base": True}))
        script = {"AAA": [_form(base_count="third"), _form(), _form()],
                  "DDD": [_form(), _form()],
                  "EEE": [None, {"pattern": "vcp"}, _form(), _form(base_count="third"),
                          None, None]}
        calls = []

        def caller(prompt, schema):
            t = re.search(r"(\w+)_weekly\.png", prompt).group(1)
            n = calls.count(t)
            calls.append(t)
            form = script[t][n]
            if form is None:
                return {"form": None, "error": "boom"}
            return {"form": form, "model": "m", "cost_usd": 0.1}

        out = rv.run_review(d, diag, caller, votes=3, jobs=1, log=lambda m: None)
        lines = (d / "review" / rv.VOTES_FILE).read_text().splitlines()
        verdicts = pl.read_verdicts(d / "verdicts.csv")
        ok("a name a measurement fails is not asked and FAILs with its code",
           "BBB" not in calls and out["measured"] == ["BBB"]
           and verdicts["BBB"]["verdict"] == "FAIL"
           and verdicts["BBB"]["notes"].startswith("DQ:below-200;"))
        ok("two answers that agree on every question settle a name",
           calls.count("DDD") == 2)
        ok("two answers that differ get a third", calls.count("AAA") == 3)
        ok("failed calls are retried once and recorded with their error",
           calls.count("EEE") == 6 and len(lines) == 9
           and sum(1 for x in lines if json.loads(x)["error"]) == 2)
        ok("a name short of an answer is listed for a check",
           out["needs_check"]["EEE"] == ["only 2 of 3 answers"])
        ok("a settled name is not short, and a measured contradiction is listed",
           out["needs_check"]["DDD"] == ["drying up, but the final area trades 1.01x its average"]
           and "AAA" not in out["needs_check"])
        ok("the counts include the measured FAIL",
           out["verdicts"] == {"PASS": 2, "PASS-": 1, "FAIL": 1})
        ok("one verdict per ticker, valid and coded",
           pl.validate_verdicts(d / "verdicts.csv", diag) == []
           and all(pl.parse_note_codes(v["verdict"], v["notes"]) == []
                   for v in verdicts.values()))
        form = rv.read_labels(d / "review" / rv.FORM_FILE)
        ok("form.csv holds the combined answers of the names asked",
           set(form) == {"AAA", "DDD", "EEE"} and form["AAA"]["base_count"] == "first_or_second")

        calls.clear()
        out2 = rv.run_review(d, diag, caller, votes=3, jobs=2, log=lambda m: None)
        ok("a rerun asks nothing more and writes no second verdict",
           calls == [] and pl.validate_verdicts(d / "verdicts.csv", diag) == []
           and out2["no_answer"] == [])

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        diag = _review_folder(d)
        out = rv.run_review(d, diag, lambda p, s: {"form": None, "error": "down"},
                            votes=2, jobs=2, log=lambda m: None)
        ok("no answers at all: no verdicts, every ticker listed",
           sorted(out["no_answer"]) == sorted(diag["ticker"])
           and pl.read_verdicts(d / "verdicts.csv") == {})


def test_labels_and_score():
    """Your own answers, in the same form, are how the reviewer is measured."""
    from src.stock_screener.hunt import reviewer as rv
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labels.csv"
        rv.write_label_template(path, ["AAA", "BBB"])
        ok("the template has a blank row per ticker",
           rv.read_labels(path) == {"AAA": {"ticker": "AAA"}, "BBB": {"ticker": "BBB"}})
        path.write_text("ticker,verdict,volume\nAAA,PASS,dry\n")
        ok("an answer off the list is refused", _raises(lambda: rv.read_labels(path)))
        path.write_text("ticker,verdict\nAAA,BUY\n")
        ok("a bad verdict is refused", _raises(lambda: rv.read_labels(path)))
    mine = {"A": {"verdict": "PASS", "volume": "drying_up"},
            "B": {"verdict": "FAIL", "volume": "mixed"},
            "C": {"verdict": "PASS-"}, "D": {"verdict": "PASS"}}
    theirs = {"A": {"verdict": "PASS", "volume": "drying_up"},
              "B": {"verdict": "PASS", "volume": "drying_up"},
              "C": {"verdict": "PASS-", "volume": "mixed"}, "D": {"verdict": "FAIL"}}
    sc = rv.score(mine, theirs)
    ok("verdict agreement over the names both labelled",
       sc["questions"]["verdict"] == {"n": 4, "agree_pct": 50.0})
    ok("a question counts only names both answered",
       sc["questions"]["volume"] == {"n": 2, "agree_pct": 50.0})
    ok("an unanswered question has no agreement",
       sc["questions"]["pivot"] == {"n": 0, "agree_pct": None})
    ok("false PASS and missed PASS are named",
       sc["verdict"]["false_pass"] == ["B"] and sc["verdict"]["missed_pass"] == ["D"])
    ok("the confusion table counts mine -> theirs",
       sc["verdict"]["confusion"]["FAIL"]["PASS"] == 1
       and sc["verdict"]["confusion"]["PASS"]["PASS"] == 1)
    ok("kappa is below the raw agreement", sc["verdict"]["kappa"] == 0.2)
    ok("perfect agreement is kappa 1", rv.score(mine, mine)["verdict"]["kappa"] == 1.0)


def test_review_images():
    """The review charts need matplotlib, which the Pi image doesn't carry: this test
    runs where the hunt runs."""
    try:
        from src.stock_screener.hunt import charts
    except ImportError:
        print("  SKIP  review images (no matplotlib)")
        return
    import json
    from src.stock_screener.hunt import reviewer as rv
    b = _bundle()
    diag = pl.diagnostics(b, pl.candidates(b))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        got = charts.render_review_images(b, diag.head(2), d)
        weekly, daily, facts = rv.review_paths(d, got[0])
        ok("each ticker gets a weekly and a daily PNG and its facts",
           got == ["AAA", "BBB"] and weekly.read_bytes()[:4] == b"\x89PNG"
           and daily.read_bytes()[:4] == b"\x89PNG"
           and json.loads(facts.read_text())["swings"][0]["label"] == "A")


if __name__ == "__main__":
    test_rs_floor_and_tier()
    test_bucket_boundaries()
    test_diagnostics_and_gates()
    test_verdict_bookkeeping()
    test_append_batch()
    test_equity_and_full_position()
    test_note_codes()
    test_compare_verdicts()
    test_reviewer_form_and_schema()
    test_reviewer_facts_hide_the_gates()
    test_verdict_table()
    test_reviewer_votes_and_checks()
    test_reviewer_measures_named_swings()
    test_run_review()
    test_labels_and_score()
    test_review_images()
    test_report_builds()
    test_narrative_markdown()
    test_report_mirror()
    test_load_state_tolerates_gaps()
    test_news_feeds()
    test_report_catalyst_line()
    print(f"\n{PASSED} hunt assertions passed.")
