"""Weekend Hunt page and its hunt_view module.

The page reads a hunt folder the hunt PC pushed, cycles the PASS names, and adds one to
the watchlist. These tests build such a folder from tests/test_hunt.py's synthetic bundle.
Nothing reaches the hunt PC: a request is a file the PC's poller reads over ssh.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from tests.cockpit._common import *  # noqa: F401,F403
from tests.cockpit._common import ROOT, _rendered_text

from src.stock_screener.cockpit import cache, hunt_view  # noqa: E402
from src.stock_screener.cockpit.export import load_watchlist  # noqa: E402
from src.stock_screener.hunt import pipeline as pl  # noqa: E402

from test_hunt import _bundle, _payload  # noqa: E402  (tests/ is on sys.path)

PAGE = ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "4_Weekend_Hunt.py"
HUNT_DATE = "2026-08-23"


class _HuntDir:
    """A temp ``data/cockpit/hunt`` with one dated folder, and the patches a page needs:
    HUNT_DIR, the hunt's watchlist read, the app's watchlist path and triggers dir."""

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        root = Path(self._td.name)
        self.root = root / "hunt"
        self.path = self.root / HUNT_DATE
        (self.path / "charts").mkdir(parents=True)
        (self.root / "logs").mkdir()
        self._patches = [patch.object(pl, "HUNT_DIR", self.root),
                         patch.object(pl, "_watchlist_tickers", lambda: ["AAA", "QQQ"]),
                         patch.object(cache, "WATCHLIST_JSON", root / "watchlist.json"),
                         patch.object(cache, "TRIGGERS_DIR", root / "triggers")]
        for p in self._patches:
            p.start()
        b = _bundle()
        diag = pl.diagnostics(b, pl.candidates(b))
        diag.to_csv(self.path / "diagnostics.csv", index=False)
        pl.append_verdicts(self.path / "verdicts.csv",
                           [{"ticker": t, "verdict": "PASS", "notes": f"{t} tight"}
                            for t in ("AAA", "BBB", "DDD", "EEE")])
        (self.path / "meta.json").write_text(json.dumps(
            {"scan_time": "2026-08-23 15:50", "regime": {"regime": "RISK-ON"},
             "n_scanned": 6, "n_passed_template": 6, "n_tier_a": 5, "n_eligible": 4,
             "min_rs": 70}))
        from PIL import Image                      # a real PNG: st.image decodes it
        Image.new("RGB", (8, 8), "white").save(self.path / "charts" / "sheet_001.png")
        (self.path / "charts" / "sheets.json").write_text(
            json.dumps({"sheet_001.png": ["AAA", "BBB", "DDD", "EEE"]}))
        (self.path / "narrative.md").write_text("# Read\n\nThin **breadth**.\n", encoding="utf-8")
        (self.path / "report.html").write_text(
            '<title>Weekend Hunt</title><p>hi</p><img src="charts/sheet_001.png" alt="s">',
            encoding="utf-8")
        (self.path / "catalyst.json").write_text(json.dumps({
            "AAA": {"category": "contract", "sentiment": "positive",
                    "summary": "A two-billion award.",
                    "sources": [{"title": "Acme wins", "publisher": "Reuters",
                                 "date": "2026-08-22", "url": "https://r/1"}]}}),
            encoding="utf-8")
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        self._td.cleanup()


# --------------------------------------------------------------------------- #
def test_hunt_dirs_skips_logs_and_incomplete():
    """The page lists dated folders that hold a finished hunt; ``logs/``, a half-pushed
    folder and a run still in progress (no report yet) are not hunts."""
    with _HuntDir() as h:
        (h.root / "2026-09-01").mkdir()
        (h.root / "2026-09-01" / "diagnostics.csv").write_text("x")
        (h.root / "2026-09-01" / "meta.json").write_text("{}")
        (h.root / "2026-07-01").mkdir()
        for f in hunt_view.FINISHED_FILES:
            (h.root / "2026-07-01" / f).write_text("x")
        names = [p.name for p in hunt_view.hunt_dirs()]
        assert names == [HUNT_DATE, "2026-07-01"], names
        assert hunt_view.latest_hunt_dir().name == HUNT_DATE
    assert hunt_view.hunt_dirs(Path("/nowhere/at/all")) == []


def test_ordered_picks_bucket_order_and_blocked_last():
    """Every PASS name, buckets in entry order, earnings-blocked names last and flagged."""
    with _HuntDir() as h:
        state, why = hunt_view.load_state_safe(h.path)
        assert state is not None, why
        picks = hunt_view.ordered_picks(state)
        assert [p.ticker for p in picks] == ["AAA", "DDD", "BBB", "EEE"], [p.ticker for p in picks]
        assert [p.bucket for p in picks] == ["buy_zone", "buy_zone", "approaching",
                                             "earnings_blocked"]
        assert picks[-1].blocked and picks[-1].earnings_in == 21
        assert picks[0].notes == "AAA tight" and not picks[0].blocked
        sheet = hunt_view.sheet_for(h.path, state, "DDD")
        assert sheet is not None and sheet[1:] == (1, 1, 2)
        assert hunt_view.panel_name(2) == "bottom-left" and hunt_view.panel_name(None)
        nums = hunt_view.entry_numbers(picks[0].row)
        assert nums["pivot"] == 100.0 and abs(nums["zone_hi"] - 105.0) < 1e-9
        assert nums["stop"] == 97.0 and nums["earnings_in"] == 60
        assert hunt_view.load_state_safe(h.root / "missing")[0] is None


def test_inline_report_images():
    with _HuntDir() as h:
        html = (h.path / "report.html").read_text(encoding="utf-8")
        out = hunt_view.inline_report_images(html, h.path, embed=True)
        assert 'src="data:image/png;base64,' in out and "<p>hi</p>" in out
        out = hunt_view.inline_report_images(html, h.path, embed=False)
        assert "<img" not in out and "<p>hi</p>" in out
        gone = html.replace("sheet_001", "sheet_009")
        assert "<img" not in hunt_view.inline_report_images(gone, h.path, embed=True)


def test_request_and_progress():
    """Start leaves a request file the PC's poller claims over ssh; the page's state comes
    from that file and the PC's status.json, request first. Two presses are one hunt."""
    from src.stock_screener.cockpit import hunt_request as hr
    with _HuntDir() as h:
        assert hunt_view.hunt_progress()["state"] == "idle"
        rec, created = hunt_view.request_hunt()
        assert created and rec["source"] == "cockpit" and rec["requested_at"]
        assert json.loads(hr.request_path().read_text(encoding="utf-8"))["source"] == "cockpit"
        rec2, created2 = hunt_view.request_hunt()
        assert not created2 and rec2 == rec
        assert hunt_view.hunt_progress()["state"] == "requested"
        # The PC claims it (removes the file) and reports.
        hr.request_path().unlink()
        hr._write(hr.status_path(), {"state": "running", "date": "2026-10-10",
                                     "message": "review attempt 1 of 2",
                                     "requested_at": rec["requested_at"],
                                     "updated_at": hr._now()})
        p = hunt_view.hunt_progress()
        assert p["state"] == "running" and p["message"] == "review attempt 1 of 2"
        # A running status the PC stopped updating hours ago is a dead run.
        hr._write(hr.status_path(), {"state": "running", "updated_at": "2020-01-01T00:00:00"})
        assert hunt_view.hunt_progress()["state"] == "error"
        hr.status_path().write_text("{not json", encoding="utf-8")
        assert hunt_view.hunt_progress()["state"] == "idle"
        # The Pi's Friday timer uses the CLI.
        assert hr.main(["write", "--source", "schedule"]) == 0
        assert hr.read_request()["source"] == "schedule"
        assert hr.main(["show"]) == 0
        assert not (h.root / "request.json.tmp").exists()


def test_add_to_watchlist_writes_hunt_pivot():
    with _HuntDir() as h:
        merged, added = hunt_view.add_to_watchlist(None, "AAA", 100.0, HUNT_DATE, "PASS", "tight")
        assert added and [e["ticker"] for e in merged] == ["AAA"]
        disk = load_watchlist(cache.WATCHLIST_JSON)
        assert disk[0]["judged_pivot"] == 100.0 and disk[0]["pivot_source"] == "judged"
        assert "weekend hunt 2026-08-23: PASS" in disk[0]["note"] and "tight" in disk[0]["note"]
        merged2, added2 = hunt_view.add_to_watchlist(None, "AAA", 101.0, HUNT_DATE, "PASS")
        assert not added2 and load_watchlist(cache.WATCHLIST_JSON)[0]["judged_pivot"] == 100.0
        # A session copy stays authoritative and gains the entry
        session = [{"ticker": "ZZZ", "judged_pivot": None}]
        merged3, added3 = hunt_view.add_to_watchlist(session, "BBB", 50.0, HUNT_DATE, "PASS")
        assert added3 and [e["ticker"] for e in merged3] == ["ZZZ", "BBB"]
        assert [e["ticker"] for e in load_watchlist(cache.WATCHLIST_JSON)] == ["ZZZ", "BBB"]
        assert session == [{"ticker": "ZZZ", "judged_pivot": None}], "the caller's list is not mutated"


def test_catalyst_helpers():
    with _HuntDir() as h:
        cats = hunt_view.load_catalysts(h.path)
        assert cats["AAA"]["category"] == "contract"
        assert hunt_view.load_catalysts(h.root / "nowhere") == {}
    links = hunt_view.manual_links("BRK-B")
    assert len(links) == 4 and all("BRK-B" in url for _, url in links)


# --------------------------------------------------------------------------- #
def _apptest():
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:                  # pragma: no cover
        print(f"  SKIP (AppTest unavailable: {e})")
        return None
    return AppTest.from_file(str(PAGE), default_timeout=60)


def test_hunt_page_empty_state():
    with _HuntDir() as h:
        import shutil
        shutil.rmtree(h.path)
        at = _apptest()
        if at is None:
            return
        at.run()
        assert not at.exception, at.exception
        assert any("No weekend hunt yet" in str(i.value) for i in at.info), \
            [str(i.value) for i in at.info]


def test_hunt_page_renders_and_steps():
    """The click-through: bucket order, wrap-around, the blocked name's warning, the hunt
    sheet when the name is not in the latest scan, and the catalyst section."""
    with _HuntDir():
        at = _apptest()
        if at is None:
            return
        at.run()
        assert not at.exception, at.exception
        text = _rendered_text(at)
        assert "1 of 4 · buy zone · **AAA**" in text, text[:600]
        assert "Hunt sheet 1 of 1: AAA is the top-left panel" in text
        assert "contract" in text and "A two-billion award." in text and "Acme wins" in text
        assert "Yahoo Finance news" in text and "SEC filings (8-K)" in text
        assert "The hunt's numbers" in text, "the from-row Step-2 panel"
        assert "**Pivot** 100.00" in text and "**stop** 97.00" in text
        at.button(key="hunt_next").click().run()
        assert "2 of 4 · buy zone · **DDD**" in _rendered_text(at)
        assert "Not read for this hunt" in _rendered_text(at)
        for _ in range(2):
            at.button(key="hunt_next").click().run()
        text = _rendered_text(at)
        assert "4 of 4 · earnings-blocked · **EEE**" in text
        assert any("Earnings in 21d" in str(w.value) for w in at.warning), "blocked warning"
        at.button(key="hunt_next").click().run()
        assert "1 of 4 · buy zone · **AAA**" in _rendered_text(at), "Next wraps"
        at.button(key="hunt_prev").click().run()
        assert "4 of 4 · earnings-blocked" in _rendered_text(at), "Prev wraps"
        assert "Thin **breadth**." in _rendered_text(at), "narrative tab rendered"


def test_hunt_page_chart_path_when_scanned():
    """A name in the latest scan gets the interactive chart and the shared Step-2 panel."""
    with _HuntDir():
        payload = _payload(100.0, 102.0)
        payload["fundamentals"].update({"revenue_qoq": 5.0, "eps_qoq": 7.0,
                                        "operating_margin": 21.0, "margin_trend": 1.5})
        with patch.object(hunt_view, "latest_payloads", return_value={"AAA": payload}):
            at = _apptest()
            if at is None:
                return
            at.run()
            assert not at.exception, at.exception
            text = _rendered_text(at)
            assert "**Rev:** +25.0% YoY · +5.0% QoQ" in text, text[:800]
            assert at.get("plotly_chart"), "the interactive chart"
            assert "Latest scan on the Pi" in text


def test_hunt_page_add_to_watchlist():
    with _HuntDir():
        at = _apptest()
        if at is None:
            return
        at.run()
        assert "⭐ Add AAA" in "".join(b.label for b in at.button), [b.label for b in at.button]
        at.button(key="hunt_wl_add").click().run()
        assert not at.exception, at.exception
        assert any("AAA added with the pivot frozen at 100.00" in str(s.value) for s in at.success), \
            [str(s.value) for s in at.success]
        assert "✓ AAA is on the watchlist" in _rendered_text(at)
        disk = load_watchlist(cache.WATCHLIST_JSON)
        assert [e["ticker"] for e in disk] == ["AAA"] and disk[0]["judged_pivot"] == 100.0
        assert at.session_state["watchlist"] if "watchlist" in at.session_state else True


def test_hunt_page_start_writes_request():
    """Start leaves the request and disables itself while one is waiting."""
    from src.stock_screener.cockpit import hunt_request as hr
    with _HuntDir():
        at = _apptest()
        if at is None:
            return
        at.run()
        at.button(key="hunt_start").click().run()
        assert not at.exception, at.exception
        assert hr.read_request()["source"] == "cockpit"
        assert any("Requested." in str(i.value) for i in at.info), [str(i.value) for i in at.info]
        assert any("Waiting for the hunt PC" in str(i.value) for i in at.info)
        assert at.button(key="hunt_start").disabled, "Start stays off while a request waits"


def test_hunt_page_status_poll_done():
    """The status line follows the PC's status.json; the first sight of a finished run
    reloads the page once, and a running one is shown with the PC's message."""
    from src.stock_screener.cockpit import hunt_request as hr
    with _HuntDir():
        hr._write(hr.status_path(), {"state": "done", "date": "2026-10-10",
                                     "message": "the hunt is on the Pi",
                                     "updated_at": "2026-10-10T18:20:00"})
        at = _apptest()
        if at is None:
            return
        at.run()
        assert not at.exception, at.exception
        assert at.session_state["hunt_seen"] == "2026-10-10T18:20:00"
        assert not at.button(key="hunt_start").disabled
        hr._write(hr.status_path(), {"state": "running", "date": "2026-10-10",
                                     "message": "review attempt 1 of 2",
                                     "updated_at": hr._now()})
        at = _apptest()
        at.run()
        assert any("Hunt running on the PC" in str(i.value)
                   and "review attempt 1 of 2" in str(i.value) for i in at.info)
        assert at.button(key="hunt_start").disabled


if __name__ == "__main__":
    from tests.cockpit._common import run_suite
    raise SystemExit(run_suite(globals(), "hunt page"))
