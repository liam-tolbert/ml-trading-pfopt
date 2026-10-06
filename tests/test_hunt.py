"""Synthetic-data tests for src/stock_screener/hunt (the weekend-hunt pipeline).

Runs as a plain script (`python tests/test_hunt.py`) or under pytest, repo style.
No disk pickle, no network: a tiny fake ScanResult is built in memory. The most
important assertions are the rule boundaries — buy zone at exactly 0% / +5%,
approach at -3%, volume confirmation at 1.5x, earnings block at 21 days — since
mis-remembered boundaries are precisely why this pipeline exists.
"""
from __future__ import annotations

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

        batch.write_text('ticker,verdict,notes\nAAA,PASS,"tight, quiet"\nBBB,PASS-,y\n',
                         encoding="utf-8")
        ok("batch appended", pl.append_batch(p, batch, diag) == 2)
        ok("a quoted comma survives the round trip",
           pl.read_verdicts(p)["AAA"]["notes"] == "tight, quiet")
        ok("re-running a batch is rejected, nothing written",
           rejected("ticker,verdict,notes\nAAA,PASS,again\nDDD,FAIL,z\n"))
        ok("a non-candidate is rejected", rejected("ticker,verdict,notes\nCCC,PASS,x\n"))
        ok("a ticker repeated in the batch is rejected",
           rejected("ticker,verdict,notes\nDDD,PASS,x\nDDD,FAIL,y\n"))
        ok("a bad verdict is rejected", rejected("ticker,verdict,notes\nDDD,BUY,x\n"))
        ok("a batch without the header is rejected", rejected("DDD,PASS,x\n"))
        ok("an empty batch is rejected", rejected(""))
        ok("a missing batch file is rejected",
           _raises(lambda: pl.append_batch(p, Path(td) / "absent.csv", diag)))
        batch.write_text("ticker,verdict,notes\nDDD,FAIL,z\nEEE,FAIL,w\n", encoding="utf-8")
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
                                     "stop", "max_order_usd"])
            old.to_csv(d / "diagnostics.csv", index=False)
            st = rp.load_state(d)
            row = st.diag_rows[0]
            ok("a pre-eight-check hunt loads with defaults",
               row["f_max"] == 8 and row["f_code33"] == 0 and row["stop"] is None)
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


if __name__ == "__main__":
    test_rs_floor_and_tier()
    test_bucket_boundaries()
    test_diagnostics_and_gates()
    test_verdict_bookkeeping()
    test_append_batch()
    test_report_builds()
    test_narrative_markdown()
    test_report_mirror()
    test_load_state_tolerates_gaps()
    test_news_feeds()
    test_report_catalyst_line()
    print(f"\n{PASSED} hunt assertions passed.")
