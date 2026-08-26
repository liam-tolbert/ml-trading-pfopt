"""Cockpit tests — the Streamlit surfaces, driven through AppTest.

Runs standalone (`python tests/cockpit/test_app.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


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
            # Price-phase labels carry the per-ticker download detail (drives the scrolling
            # download-log slot); the screening phase exercises the log's skip branch.
            for i in (1, 20, 40):
                progress(i, 40, "Prices · SYN: full history (2y)")
            for i in (1, 40):
                progress(i, 40, "Screening · SYN")
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


def test_filter_tweak_reuses_scan_memo():
    """Item 18: moving the Min RS slider re-filters the memoized result INSTANTLY — the
    scan body runs exactly once across all tweaks, including the toggle-back case the old
    one-slot memo re-scanned twice on."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_filter_tweak_reuses_scan_memo (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    n_all = len(result.candidates)
    calls = {"n": 0}

    def _fake_scan(*a, **kw):
        calls["n"] += 1
        return result

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp:
        with patch.object(scanmod, "run_scan", side_effect=_fake_scan), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
            at = AppTest.from_file(app_path, default_timeout=60)
            at.run()
            assert not at.exception, f"app raised: {at.exception}"
            rs = [s for s in at.slider if "RS" in str(getattr(s, "label", ""))]
            assert rs, "Min RS slider not found"
            rs[0].set_value(90).run()                     # tighten
            assert not at.exception, f"app raised on tighten: {at.exception}"
            n_tight = "".join(str(getattr(c, "value", "")) for c in at.caption)
            rs[0].set_value(0).run()                      # toggle back past the original
            assert not at.exception, f"app raised on loosen: {at.exception}"
            n_loose = "".join(str(getattr(c, "value", "")) for c in at.caption)
    assert calls["n"] == 1, f"filter tweaks must reuse the memo; scan ran {calls['n']}x"
    assert f"{n_all} after filters" in n_loose, n_loose   # min_rs=0 shows everything
    assert f"{n_all} after filters" not in n_tight        # min_rs=90 filtered some out


def test_full_redownload_button_forces_scan():
    """The Advanced '⟳ Full re-download' button re-runs the scan with force=True (full-
    history cache re-baseline); the initial scan — and by the same wiring the Re-scan
    button — never passes force (they use the incremental top-up)."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_full_redownload_button_forces_scan (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    forces = []

    def _fake_scan(*a, force=False, progress=None, **kw):
        forces.append(force)
        return result

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp:
        with patch.object(scanmod, "run_scan", side_effect=_fake_scan), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
            at = AppTest.from_file(app_path, default_timeout=60)
            at.run()
            assert not at.exception, f"app raised: {at.exception}"
            assert forces == [False], f"initial scan must not force, got {forces}"
            full = [b for b in at.button if b.key == "full_refetch"]
            assert full, "Advanced full re-download button not found"
            full[0].click().run()
            assert not at.exception, f"app raised after full re-download: {at.exception}"
    assert forces == [False, True], f"full re-download must re-scan with force=True, got {forces}"


def test_watchlist_picker_pills_sync():
    """The watchlist multiselect is CONTROLLED: its pills ARE the watchlist. The widget
    seeds from the list every run (a STALE name absent from the scan still shows and is
    removable — the chart toggle can never reach one); dismissing a pill removes the entry
    (persisted, plan invalidated, survivor's frozen pivot intact); picking a new option
    adds an unfrozen entry."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_watchlist_picker_pills_sync (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    from src.stock_screener.cockpit.export import load_watchlist
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    keep_t = result.candidates["ticker"].iloc[0]          # a real scanned name to keep
    add_t = result.candidates["ticker"].iloc[1]           # a scanned name to add later

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp:
        wl_path = Path(_tmp) / "watchlist.json"
        with patch.object(scanmod, "run_scan", return_value=result), \
                patch.object(cache, "WATCHLIST_JSON", wl_path), \
                patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
            at = AppTest.from_file(app_path, default_timeout=60)
            at.session_state["watchlist"] = [
                {"ticker": keep_t, "judged_pivot": 55.5, "date_added": "2026-07-20",
                 "pivot_source": "judged", "note": ""},
                {"ticker": "GONEX", "judged_pivot": 12.0, "date_added": "2026-07-20",
                 "pivot_source": "auto", "note": ""}]     # STALE: not in the scan at all
            at.session_state["trade_build_n"] = 1
            at.session_state["trade_plan"] = {"plan": [], "skipped": [],
                                              "account": {}, "build_ts": 1}
            at.run()
            assert not at.exception, f"app raised: {at.exception}"

            pick = [m for m in at.multiselect if m.key == "wl_picker"]
            assert pick, "watchlist picker not rendered"
            assert list(pick[0].value) == [keep_t, "GONEX"], \
                "pills must seed from the watchlist (stale names included)"
            assert "GONEX" in pick[0].options

            # Dismiss the stale pill -> removed everywhere, survivor's pivot intact.
            pick[0].set_value([keep_t]).run()
            assert not at.exception, f"app raised on pill dismiss: {at.exception}"
            tickers = [e["ticker"] for e in at.session_state["watchlist"]]
            assert tickers == [keep_t], f"GONEX should be gone, got {tickers}"
            assert at.session_state["watchlist"][0]["judged_pivot"] == 55.5
            assert "trade_plan" not in at.session_state    # removal invalidates the plan
            assert [e["ticker"] for e in load_watchlist(wl_path)] == [keep_t]

            # Pick a new option -> added as an unfrozen entry and persisted.
            pick = [m for m in at.multiselect if m.key == "wl_picker"]
            pick[0].set_value([keep_t, add_t]).run()
            assert not at.exception, f"app raised on pill add: {at.exception}"
            by = {e["ticker"]: e for e in at.session_state["watchlist"]}
            assert set(by) == {keep_t, add_t}
            assert by[add_t]["judged_pivot"] is None       # picker adds are unfrozen
            assert set(e["ticker"] for e in load_watchlist(wl_path)) == {keep_t, add_t}


def test_trigger_sidebar_chart_button():
    """Each trigger-report row carries a 📈 button: clicking it jumps the MAIN chart to
    that ticker (overriding the table's row selection for that run); a name outside the
    scan payloads still renders a (disabled) button instead of crashing."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trigger_sidebar_chart_button (AppTest unavailable: {e})")
        return
    import json as _json
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    assert len(result.candidates) >= 2, "need two candidates for a distinct jump target"
    # Put TWO real candidates in the report: the displayed table's default row is sort-
    # dependent, so the jump target is chosen at runtime as whichever one ISN'T default.
    t_a, t_b = result.candidates["ticker"].iloc[0], result.candidates["ticker"].iloc[1]

    rep = {"schema": 1, "date": "2026-07-22", "generated_at": "2026-07-22T16:31:00-04:00",
           "spy": None, "all_stale": False, "early_close": False, "intraday": False,
           "names": [
               {"ticker": t_a, "status": "watch", "judged_pivot": 55.5, "close": 50.0,
                "volume_ratio_50": 1.0},
               {"ticker": t_b, "status": "watch", "judged_pivot": 44.0, "close": 40.0,
                "volume_ratio_50": 1.0},
               {"ticker": "GONEX", "status": "untracked", "judged_pivot": 12.0}],
           "summary": {"n": 3, "triggered": [], "extended": [], "stale": [],
                       "untracked": ["GONEX"], "earnings_soon": [], "no_data": [],
                       "no_pivot": [], "auto_frozen": []}}

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp:
        trg = Path(_tmp) / "triggers"
        trg.mkdir(parents=True)
        (trg / "triggers_2026-07-22.json").write_text(_json.dumps(rep), encoding="utf-8")
        with patch.object(scanmod, "run_scan", return_value=result), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", trg):
            at = AppTest.from_file(app_path, default_timeout=60)
            at.run()
            assert not at.exception, f"app raised: {at.exception}"
            # The ⭐/✓ toggle label names the charted pick — learn the default, then jump
            # to the OTHER candidate so the change is observable.
            _toggle = [b for b in at.button if b.key == "wl_toggle"]
            assert _toggle, "chart toggle not rendered"
            jump_t = t_b if t_a in str(_toggle[0].label) else t_a
            # The out-of-scan name renders a button too (disabled, not a crash).
            assert [b for b in at.button if b.key == "trg_chart_GONEX"]

            jump = [b for b in at.button if b.key == f"trg_chart_{jump_t}"]
            assert jump, "per-row chart button not rendered"
            jump[0].click().run()
            assert not at.exception, f"app raised on chart jump: {at.exception}"
            _toggle = [b for b in at.button if b.key == "wl_toggle"]
            assert _toggle and jump_t in str(_toggle[0].label), \
                f"chart should have jumped to {jump_t}: {_toggle[0].label}"
            assert "chart_pick" not in at.session_state    # consumed (one-run override)


def test_trigger_check_now_button():
    """The 🔔 Check-triggers-now button runs the refresh_job pipeline in-process, writes
    the report to cache.TRIGGERS_DIR, and renders it in the SAME pass (the check runs
    before the report load). A failing check degrades to a warning, never a crash."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trigger_check_now_button (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache, refresh_job
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))

    rep = {"schema": 1, "date": "2026-07-27", "generated_at": "2026-07-27T10:31:00-04:00",
           "spy": None, "all_stale": False, "early_close": False, "intraday": True,
           "names": [{"ticker": "MANUX", "status": "watch", "judged_pivot": 55.5,
                      "close": 50.0, "volume_ratio_50": 1.0}],
           "summary": {"n": 1, "triggered": [], "extended": [], "stale": [],
                       "untracked": [], "earnings_soon": [], "no_data": [],
                       "no_pivot": [], "auto_frozen": []}}

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp:
        trg = Path(_tmp) / "triggers"                       # left EMPTY: no report yet
        with patch.object(scanmod, "run_scan", return_value=result), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", trg), \
                patch.object(refresh_job, "build_report", return_value=rep) as _br:
            at = AppTest.from_file(app_path, default_timeout=60)
            at.run()
            assert not at.exception, f"app raised: {at.exception}"
            # No report on disk: the empty-state caption points at the button.
            assert any("No trigger report yet" in str(c.value) for c in at.caption)
            btn = [b for b in at.button if b.key == "trigger_check_now"]
            assert btn, "Check-triggers-now button not rendered"

            btn[0].click().run()
            assert not at.exception, f"app raised on manual check: {at.exception}"
            assert _br.call_count == 1, "build_report should run exactly once per click"
            assert (trg / "triggers_2026-07-27.json").exists(), \
                "manual check should persist the dated report"
            # Rendered in the same pass: the canned name's row is in the panel.
            assert any("MANUX" in str(c.value) for c in at.caption), \
                "fresh report should render without another click/rerun"

            # Failure path: the pipeline raising surfaces a warning, not a crash, and the
            # last good report stays up.
            _br.side_effect = RuntimeError("yfinance down")
            _btn = [b for b in at.button if b.key == "trigger_check_now"]
            _btn[0].click().run()
            assert not at.exception, f"app raised on failing check: {at.exception}"
            assert any("Trigger check failed" in str(w.value) for w in at.warning)
            assert any("MANUX" in str(c.value) for c in at.caption)


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


def test_trade_plan_preview_renders_stop_controls():
    """With a trade plan seeded in session_state, the paper-trade preview renders the new
    attach-stop toggle and a per-ticker editable stop number_input (keyed by the build nonce),
    without raising — exercises the preview loop + stop_is_valid call path in app.py."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trade_plan_preview_renders_stop_controls (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import journal_cache, scan as scanmod, cache, trade
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    journal_cache.cached_fills.clear()          # process-global cache; keep this run offline
    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(scanmod, "run_scan", return_value=result), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"), \
            patch.object(trade, "fetch_order_fills",
                         side_effect=trade.TradeUnavailable("offline")):
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


def test_trade_panel_risk_guidance():
    """§6.51 progressive exposure at the point of sizing: in risk mode the panel shows
    the last-10 closed form ("2W/8L … pilot size") from the shared journal cache and a
    one-click "Use suggested" button that writes the risk widget and invalidates the
    built plan — never silently changing the user's input. A failed journal read renders
    "journal unavailable", never blocks the panel, and is memoized per session so an
    Alpaca outage costs exactly ONE fetch attempt."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trade_panel_risk_guidance (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import journal_cache, scan as scanmod, cache, trade
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")

    def canned_fills():
        # 10 tagged round trips: 2 winners (+10%) and 8 losers (-6%) -> pilot size.
        fills = []
        for i in range(10):
            sym = f"T{i:02d}"
            fills.append({"symbol": sym, "side": "buy", "qty": 10, "price": 100.0,
                          "time": f"2026-06-{i + 1:02d}T14:30:00Z", "order_id": f"b{i}",
                          "client_order_id": f"SEPAcockpit-{sym}-1"})
            fills.append({"symbol": sym, "side": "sell", "qty": 10,
                          "price": 110.0 if i < 2 else 94.0,
                          "time": f"2026-06-{i + 11:02d}T14:30:00Z", "order_id": f"s{i}",
                          "client_order_id": ""})
        return {"account": {"account_number": "PA1", "equity": 100000.0, "cash": 0.0,
                            "using_dedicated": True}, "fills": fills}

    def _seed(at):
        at.session_state["watchlist"] = [
            {"ticker": "AAA", "judged_pivot": None, "date_added": None,
             "pivot_source": None, "note": ""}]
        at.session_state["trade_mode"] = "Risk % to stop"

    # 1) guidance + one-click apply
    journal_cache.cached_fills.clear()
    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(scanmod, "run_scan", return_value=result), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"), \
            patch.object(trade, "fetch_order_fills", return_value=canned_fills()):
        at = AppTest.from_file(app_path, default_timeout=60)
        _seed(at)
        at.session_state["trade_plan"] = {"plan": [], "skipped": [], "account": {},
                                          "build_ts": 1}
        at.run()
        assert not at.exception, f"app raised: {at.exception}"
        rendered = " ".join(str(getattr(c, "value", "")) for c in at.caption)
        assert "2W/8L" in rendered and "pilot size" in rendered, rendered
        btns = [b for b in at.button if b.key == "risk_apply"]
        assert btns and "0.50%" in btns[0].label, "apply button missing or mislabeled"
        btns[0].click().run()
        assert at.session_state["trade_amt_risk"] == 0.5, "suggestion not applied"
        assert "trade_plan" not in at.session_state, "apply must invalidate the plan"

    # 2) journal down: caption degrades, panel lives, exactly ONE fetch attempt/session
    calls = {"n": 0}

    def _dead():
        calls["n"] += 1
        raise trade.TradeUnavailable("no creds")

    journal_cache.cached_fills.clear()
    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(scanmod, "run_scan", return_value=result), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"), \
            patch.object(trade, "fetch_order_fills", _dead):
        at = AppTest.from_file(app_path, default_timeout=60)
        _seed(at)
        at.run()
        assert not at.exception, f"app raised: {at.exception}"
        rendered = " ".join(str(getattr(c, "value", "")) for c in at.caption)
        assert "journal unavailable" in rendered, rendered
        at.run()                                        # rerun -> memoized failure
        assert calls["n"] == 1, f"outage must cost one attempt, got {calls['n']}"


def test_trade_plan_preview_marks_held_names():
    """Issue 6: build_buy_plan is holdings-blind, but submit sends NO buy for a held name (re-arm
    only). The preview must mark a held name 'already held … no buy', keep the buy line for a fresh
    name, and EXCLUDE the held name from the est-value total (which counts only executing buys)."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trade_plan_preview_marks_held_names (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(scanmod, "run_scan", return_value=result), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
        at = AppTest.from_file(app_path, default_timeout=60)
        at.session_state["watchlist"] = [
            {"ticker": "NEWX", "judged_pivot": None, "date_added": None,
             "pivot_source": None, "note": ""},
            {"ticker": "HELDX", "judged_pivot": None, "date_added": None,
             "pivot_source": None, "note": ""}]
        at.session_state["trade_build_n"] = 1
        at.session_state["trade_plan"] = {
            "plan": [
                {"ticker": "NEWX", "shares": 10, "price": 50.0, "pivot": 50.0,
                 "est_value": 500.0, "extended": False, "capped": False,
                 "stop_price": 46.0, "earnings_in": None},
                {"ticker": "HELDX", "shares": 20, "price": 100.0, "pivot": 100.0,
                 "est_value": 2000.0, "extended": False, "capped": False,
                 "stop_price": 92.0, "earnings_in": None}],
            "skipped": [],
            "account": {"account_number": "PA000123", "equity": 100000.0,
                        "using_dedicated": True},
            "held": {"HELDX": 20},                            # HELDX already held -> re-arm only
            "build_ts": 1}
        at.run()
        assert not at.exception, f"app raised: {at.exception}"

    rendered = " ".join(str(getattr(m, "value", ""))
                        for m in list(at.markdown) + list(getattr(at, "caption", [])))
    assert "already held (20 sh)" in rendered, rendered      # held name marked, no buy line
    assert "1 buy(s)" in rendered                            # only NEWX counts as a buy
    # total excludes the $2,000 held name (would be ~$2,500 if it were summed in)
    assert "$2,500" not in rendered, rendered


def test_trade_account_error_shown_with_empty_plan():
    """Item 16: a missing-credentials build produces an EMPTY plan — the account error must
    render anyway (it used to sit inside `if _plan:` and the user saw only 'No tradable
    orders', never the actionable message)."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trade_account_error_shown_with_empty_plan (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    _ERR = "No Alpaca credentials in .env"

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(scanmod, "run_scan", return_value=result), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
        at = AppTest.from_file(app_path, default_timeout=60)
        at.session_state["watchlist"] = [
            {"ticker": "NEWX", "judged_pivot": None, "date_added": None,
             "pivot_source": None, "note": ""}]
        at.session_state["trade_build_n"] = 1
        at.session_state["trade_plan"] = {"plan": [], "skipped": [],
                                          "account": {"error": _ERR}, "build_ts": 1}
        at.run()
        assert not at.exception, f"app raised: {at.exception}"

    warnings = " ".join(str(getattr(w, "value", "")) for w in at.warning)
    assert _ERR in warnings, f"credentials error must render with an empty plan: {warnings}"
    rendered = " ".join(str(getattr(m, "value", ""))
                        for m in list(at.markdown) + list(getattr(at, "caption", [])))
    assert "No tradable orders" in rendered      # the empty-plan caption still shows too


def test_trade_plan_invalidated_on_events():
    """Item 19: a built trade plan is a snapshot — a Re-scan, a watchlist mutation, or a
    sizing change must drop it (and trade_result). A plain rerun must NOT: the four
    seeded-plan tests rely on a plan surviving a render pass."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trade_plan_invalidated_on_events (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))

    def _seed(at):
        at.session_state["watchlist"] = [
            {"ticker": "NEWX", "judged_pivot": None, "date_added": None,
             "pivot_source": None, "note": ""}]
        at.session_state["trade_build_n"] = 1
        at.session_state["trade_plan"] = {
            "plan": [{"ticker": "NEWX", "shares": 10, "price": 50.0, "pivot": 50.0,
                      "est_value": 500.0, "extended": False, "capped": False,
                      "stop_price": 46.0, "earnings_in": None}],
            "skipped": [], "account": {"account_number": "PA000123", "equity": 100000.0,
                                       "using_dedicated": True}, "build_ts": 1}

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    # WATCHLIST_JSON MUST be patched: scenario (d) dismisses a watchlist pill, which
    # PERSISTS — an unpatched run mutates the user's real data/cockpit/watchlist.json
    # (the old Clear-watchlist variant of this scenario wiped it, once).
    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(scanmod, "run_scan", return_value=result), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
        at = AppTest.from_file(app_path, default_timeout=60)
        _seed(at)
        at.run()
        assert not at.exception, f"app raised: {at.exception}"
        at.run()                                          # (a) plain rerun -> plan SURVIVES
        assert "trade_plan" in at.session_state, "a plain rerun must not drop the plan"

        rescan = [b for b in at.button if b.key == "rescan"]
        assert rescan, "rescan button (key='rescan') not found"
        rescan[0].click().run()                           # (b) Re-scan -> plan gone
        assert "trade_plan" not in at.session_state, "Re-scan must invalidate the plan"

        _seed(at)
        at.run()
        sel = [s for s in at.selectbox if s.key == "trade_mode"]
        assert sel, "trade_mode selectbox not found"
        sel[0].select("$ per name").run()                 # (c) sizing change -> plan gone
        assert "trade_plan" not in at.session_state, "sizing change must invalidate the plan"

        _seed(at)
        at.run()
        pills = [m for m in at.multiselect if m.key == "wl_picker"]
        assert pills, "watchlist pills picker not found"
        pills[0].set_value([]).run()                      # (d) dismiss the pill -> plan gone
        assert "trade_plan" not in at.session_state, "watchlist edit must invalidate the plan"


def test_trade_plan_buy_checkboxes_filter_submit():
    """Per-buy include/exclude checkboxes: an earnings-flagged buy starts UNCHECKED (the
    ~21-day no-fly), a clean buy starts checked, the selected-count caption tracks the
    boxes, and Submit sends ONLY the checked buys (held names always pass through)."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trade_plan_buy_checkboxes_filter_submit (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, trade as tradmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    sent = {}

    def _fake_submit(plan, attach_stop=True):
        sent["tickers"] = [o["ticker"] for o in plan]
        return {"results": [{"ticker": o["ticker"], "status": "submitted", "detail": ""}
                            for o in plan],
                "account_number": "PA000123", "equity": 100000.0}

    def _entry(t, earnings_in=None):
        return {"ticker": t, "shares": 10, "price": 50.0, "pivot": 50.0,
                "est_value": 500.0, "extended": False, "capped": False,
                "stop_price": 46.0, "earnings_in": earnings_in}

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(scanmod, "run_scan", return_value=result), \
            patch.object(tradmod, "submit_buy_plan", _fake_submit), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
        at = AppTest.from_file(app_path, default_timeout=60)
        at.session_state["watchlist"] = [
            {"ticker": t, "judged_pivot": None, "date_added": None,
             "pivot_source": None, "note": ""} for t in ("CLEAN", "ERNS", "HELDX")]
        at.session_state["trade_build_n"] = 1
        at.session_state["trade_plan"] = {
            "plan": [_entry("CLEAN"), _entry("ERNS", earnings_in=10), _entry("HELDX")],
            "skipped": [],
            "account": {"account_number": "PA000123", "equity": 100000.0,
                        "using_dedicated": True},
            "held": {"HELDX": 20},
            "build_ts": 1}
        at.run()
        assert not at.exception, f"app raised: {at.exception}"

        boxes = {c.key: c for c in at.checkbox if str(c.key or "").startswith("buy_")}
        assert set(boxes) == {"buy_CLEAN_1", "buy_ERNS_1"}, \
            f"one checkbox per BUY row (held has none), got {sorted(boxes)}"
        assert boxes["buy_CLEAN_1"].value is True
        assert boxes["buy_ERNS_1"].value is False, "earnings-soon buy must start unchecked"
        rendered = _rendered_text(at)
        assert "1/2 buy(s) selected" in rendered, rendered

        submit = [b for b in at.button if b.key == "trade_submit"]
        assert submit, "submit button not found"
        submit[0].click().run()
        assert not at.exception, f"app raised on submit: {at.exception}"
    assert sent["tickers"] == ["CLEAN", "HELDX"], \
        f"submit must send checked buys + held only, got {sent.get('tickers')}"


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


def test_journal_page_renders():
    """The Journal page loads and renders its stats tiles + tables with an offline (patched)
    fetch_order_fills — no network."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_journal_page_renders (AppTest unavailable: {e})")
        return
    from unittest.mock import patch
    from src.stock_screener.cockpit import journal_cache, trade

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
    journal_cache.cached_fills.clear()          # process-global cache; take THIS patch's data
    with patch.object(trade, "fetch_order_fills", return_value=offline):
        at = AppTest.from_file(page, default_timeout=60)
        at.run()
    assert not at.exception, f"journal page raised: {at.exception}"
    labels = [str(getattr(m, "label", "")) for m in at.metric]
    assert any("Batting" in x for x in labels), f"stats tiles missing: {labels}"
    # 1 closed winner out of 1 -> batting 100%; the open BBB episode must not enter the stats
    batting = [m for m in at.metric if "Batting" in str(getattr(m, "label", ""))][0]
    assert str(batting.value) == "100%", batting.value


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
            {"ticker": "CRSX", "status": "crossed", "judged_pivot": 100.0,
             "pivot_source": "judged", "close": 100.8, "volume_ratio_50": 0.7,
             "crossed": True},                             # §6.36 quiet drift above pivot
            {"ticker": "PBKX", "status": "pullback", "judged_pivot": 100.0,
             "pivot_source": "judged", "close": 100.8, "volume_ratio_50": 0.6,
             "pullback": True, "crossed_earlier": True},   # §6.50 secondary-entry setup
            {"ticker": "NOPX", "status": "no_pivot"},      # sparse row must render too
        ],
        "summary": {"n": 5, "triggered": ["TRGX"], "pullback": ["PBKX"],
                    "crossed": ["CRSX"], "extended": [],
                    "stale": [], "earnings_soon": ["TRGX"], "no_data": [],
                    "no_pivot": ["NOPX"], "auto_frozen": []},
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
    assert "CRSX" in rendered and "crossed" in rendered, "crossed name not rendered"
    assert "quiet drift" in rendered, "crossed explainer caption not rendered"
    assert "PBKX" in rendered and "pullback" in rendered, "pullback name not rendered"
    assert "secondary entry" in rendered, "pullback explainer caption not rendered"


def test_freeze_warning_post_breakout():
    """§6.19/§6.36 freeze-time warning: when the charted name already trades ABOVE the
    pivot the ⭐/📌 buttons would freeze, the Step-3 panel warns that the armed trigger
    may never re-fire (post-breakout freeze — the PECO case); a name still below its
    pivot renders no such warning."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_freeze_warning_post_breakout (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    assert len(result.candidates) >= 1, "fixture must yield >=1 candidate"

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")

    def _render(pivot_vs_close: float) -> str:
        # Reposition EVERY payload's app pivot relative to its own last close, so the
        # assertion holds regardless of which table row the app picks for the chart.
        for pl in result.payloads.values():
            last_close = float(pl["df"]["Close"].iloc[-1])
            pl["levels"]["pivot"] = round(last_close * pivot_vs_close, 2)
        with tempfile.TemporaryDirectory() as _tmp:
            with patch.object(scanmod, "run_scan", return_value=result), \
                    patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                    patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
                at = AppTest.from_file(app_path, default_timeout=60)
                at.run()
        assert not at.exception, f"app raised: {at.exception}"
        return " ".join(str(getattr(c, "value", "")) for c in at.caption)

    caps = _render(0.95)                                 # price ABOVE pivot -> warning
    assert "post-breakout" in caps, "freeze warning missing for a name above its pivot"
    caps2 = _render(1.05)                                # price BELOW pivot -> silent
    assert "post-breakout" not in caps2, "freeze warning must not fire below the pivot"


def test_app_status_line_renders_scan_and_price_asof():
    """The stale-while-refresh status fragment labels the SCAN timestamp as such once a
    result exists, and the latest()-first flow still runs the scan exactly once per session
    (the store is inert under AppTest — per-session isolation intact).

    It says "scan", never "data as of": the table is the last SCREEN, which only Re-scan
    advances, while prices move on cockpit-refresh's schedule. One timestamp for both
    reported the older of the two as if it were both, which made a minutes-old price cache
    look days stale."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_app_status_line_renders_data_as_of (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, cache
    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    calls = {"n": 0}

    def _fake_scan(*a, **kw):
        calls["n"] += 1
        return result

    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")
    with tempfile.TemporaryDirectory() as _tmp:
        with patch.object(scanmod, "run_scan", side_effect=_fake_scan), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
            at = AppTest.from_file(app_path, default_timeout=60)
            at.run()
            assert not at.exception, f"app raised: {at.exception}"
            caps = "".join(str(getattr(c, "value", "")) for c in at.caption)
            assert "scan " in caps, f"status line must label the scan time: {caps[:200]}"
            assert "data as of" not in caps, \
                f"the ambiguous one-timestamp caption is gone: {caps[:200]}"
            at.run()                                     # rerun -> same result, no rescan
            assert not at.exception, f"app raised on rerun: {at.exception}"
    assert calls["n"] == 1, f"scan must run once per session, ran {calls['n']}x"


def test_trade_panel_gate_blocks_buys():
    """#23 in the panel: a Build-time CLOSED gate renders the red caption; buy rows are
    stamped gate_blocked at Submit while held re-arms still flow (button enabled when
    held rows exist); with ONLY buys the Submit button is disabled outright. Plans
    without a gate key (older sessions / seeded tests) change nothing."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trade_panel_gate_blocks_buys (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod, trade as tradmod, cache

    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))
    sent = {}

    def _fake_submit(plan, attach_stop=True):
        sent["plan"] = plan
        return {"results": [{"ticker": o["ticker"], "status": "submitted", "detail": ""}
                            for o in plan],
                "account_number": "PA000123", "equity": 100000.0}

    def _entry(t):
        return {"ticker": t, "shares": 10, "price": 50.0, "pivot": 50.0,
                "est_value": 500.0, "extended": False, "capped": False,
                "stop_price": 46.0, "earnings_in": None}

    _gate_closed = {"open": False,
                    "reason": "newest position below breakeven: BBB",
                    "probe_size_factor": 1.0, "consecutive_losses": 0}
    _wl = [{"ticker": t, "judged_pivot": None, "date_added": None,
            "pivot_source": None, "note": ""} for t in ("CLEAN", "HELDX")]
    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")

    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(scanmod, "run_scan", return_value=result), \
            patch.object(tradmod, "submit_buy_plan", _fake_submit), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
        # 1) gate closed, buys + a held row: caption renders, submit stays enabled for
        #    the re-arm, and the sent plan carries gate_blocked on the buy only.
        at = AppTest.from_file(app_path, default_timeout=60)
        at.session_state["watchlist"] = list(_wl)
        at.session_state["trade_build_n"] = 1
        at.session_state["trade_plan"] = {
            "plan": [_entry("CLEAN"), _entry("HELDX")], "skipped": [],
            "account": {"account_number": "PA000123", "equity": 100000.0,
                        "using_dedicated": True},
            "held": {"HELDX": 20}, "build_ts": 1, "gate": _gate_closed}
        at.run()
        assert not at.exception, f"app raised: {at.exception}"
        rendered = _rendered_text(at)
        assert "Exposure gate closed" in rendered, "gate caption missing"
        submit = [b for b in at.button if b.key == "trade_submit"]
        assert submit and not submit[0].disabled, \
            "submit must stay enabled while a held row needs its re-arm"
        submit[0].click().run()
        assert not at.exception, f"app raised on submit: {at.exception}"
        by = {o["ticker"]: o for o in sent["plan"]}
        assert by["CLEAN"].get("gate_blocked") is True
        assert by["HELDX"].get("gate_blocked") is False and by["HELDX"]["rearm_only"]

        # 2) gate closed, buys only: submit disabled outright.
        at = AppTest.from_file(app_path, default_timeout=60)
        at.session_state["watchlist"] = [dict(_wl[0])]
        at.session_state["trade_build_n"] = 1
        at.session_state["trade_plan"] = {
            "plan": [_entry("CLEAN")], "skipped": [],
            "account": {"account_number": "PA000123", "equity": 100000.0,
                        "using_dedicated": True},
            "held": {}, "build_ts": 1, "gate": _gate_closed}
        at.run()
        assert not at.exception, f"app raised: {at.exception}"
        submit = [b for b in at.button if b.key == "trade_submit"]
        assert submit and submit[0].disabled, \
            "submit must be disabled when the payload is exclusively gate-blocked buys"



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "app"))
