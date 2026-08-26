"""Cockpit tests — armed entries — the pre-authorized next-open buy.

Runs standalone (`python tests/cockpit/test_entries.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_build_entry_plan_filters_and_coercion():
    """#24 arming filters: only genuine limit buys arm (rearm_only/stop_only/zero-share
    rows silently drop; a missing limit or a stop at/above the limit refuses with a
    note); numpy scalars coerce to plain JSON types; panel order is preserved (the
    executor's walk order IS the ranking)."""
    import numpy as np
    from src.stock_screener.cockpit import entries

    rows = [
        {"ticker": "AAA", "shares": np.int64(10), "price": np.float64(50.0),
         "pivot": np.float64(50.0), "limit_price": np.float64(52.5),
         "stop_price": np.float64(46.0), "est_value": np.float64(525.0),
         "earnings_in": np.int64(30)},
        {"ticker": "HELD", "shares": 5, "rearm_only": True, "limit_price": 52.5,
         "stop_price": 46.0},
        {"ticker": "ZERO", "shares": 0, "stop_only": True, "limit_price": 52.5,
         "stop_price": 46.0},
        {"ticker": "MKT", "shares": 5, "price": 50.0, "limit_price": None,
         "stop_price": 46.0},
        {"ticker": "BADSTOP", "shares": 5, "price": 50.0, "limit_price": 52.0,
         "stop_price": 60.0},
        {"ticker": "BBB", "shares": 7, "price": 40.0, "pivot": 40.0,
         "limit_price": 42.0, "stop_price": 37.0, "est_value": 294.0,
         "earnings_in": None},
    ]
    plan = entries.build_entry_plan(rows, today="2026-08-20")
    assert [r["ticker"] for r in plan["rows"]] == ["AAA", "BBB"]
    r0 = plan["rows"][0]
    assert type(r0["shares"]) is int and type(r0["limit_price"]) is float
    assert r0["earnings_in"] == 30 and type(r0["earnings_in"]) is int
    assert r0["status"] == "armed"
    assert any("MKT" in n for n in plan["notes"])
    assert any("BADSTOP" in n for n in plan["notes"])
    import json as _json
    _json.dumps(plan)                     # numpy leakage would raise here
    assert plan["date"] == "2026-08-20"


def test_entry_plan_persistence_freshness_and_disarm():
    """#24 plan files + the weekend-safe freshness rule (NOT the sells version): a
    Friday/Saturday/Sunday plan executes Monday — exactly one business day inside
    (date, today]; same-day and two-session-old plans refuse. Atomic dated save,
    ``before=``, corrupt-skip, and disarm flipping only armed rows."""
    import tempfile
    from src.stock_screener.cockpit import entries

    def plan_for(date):
        return {"date": date, "generated_at": "x",
                "rows": [{"ticker": "AAA", "shares": 5, "price": 50.0, "pivot": 50.0,
                          "limit_price": 52.5, "stop_price": 46.0, "est_value": 262.5,
                          "earnings_in": None, "status": "armed", "detail": ""}],
                "notes": [], "executed_at": None}

    # 2026-08: 20=Thu, 21=Fri, 22=Sat, 23=Sun, 24=Mon, 25=Tue
    assert entries.plan_is_current(plan_for("2026-08-20"), today="2026-08-21")
    assert entries.plan_is_current(plan_for("2026-08-21"), today="2026-08-24")
    assert entries.plan_is_current(plan_for("2026-08-22"), today="2026-08-24")
    assert entries.plan_is_current(plan_for("2026-08-23"), today="2026-08-24")
    assert not entries.plan_is_current(plan_for("2026-08-21"), today="2026-08-21")
    assert not entries.plan_is_current(plan_for("2026-08-21"), today="2026-08-25")
    assert not entries.plan_is_current(plan_for("2026-08-22"), today="2026-08-25")

    with tempfile.TemporaryDirectory() as tmp:
        entries.save_entry_plan(plan_for("2026-08-21"), tmp)
        entries.save_entry_plan(plan_for("2026-08-22"), tmp)
        (Path(tmp) / "entry_plan_2026-08-23.json").write_text("{corrupt",
                                                              encoding="utf-8")
        assert entries.load_latest_entry_plan(tmp)["date"] == "2026-08-22"
        assert entries.load_latest_entry_plan(tmp, before="2026-08-22")["date"] == \
            "2026-08-21"
        assert entries.load_latest_entry_plan(tmp, before="2026-08-21") is None
        p = entries.load_latest_entry_plan(tmp)
        assert entries.disarm_row(p, "AAA") is True
        assert p["rows"][0]["status"] == "disarmed"
        assert entries.disarm_row(p, "AAA") is False
        assert entries.disarm_row(p, "ZZZ") is False


def test_execute_entry_plan_matrix():
    """#24 executor: AUTOBUY gate (ships dark), stale refused, gate closed AND gate
    unknown both fail CLOSED, already-held and skipped results move to the NEXT armed
    row, only a submitted result consumes the one-per-day bullet, failed stops the
    walk (later rows stay armed), reruns submit nothing."""
    from src.stock_screener.cockpit import entries

    def row(t):
        return {"ticker": t, "shares": 5, "price": 50.0, "pivot": 50.0,
                "limit_price": 52.5, "stop_price": 46.0, "est_value": 262.5,
                "earnings_in": None, "status": "armed", "detail": ""}

    def mkplan(*ts, date="2026-08-19"):
        return {"date": date, "generated_at": "x", "rows": [row(t) for t in ts],
                "notes": [], "executed_at": None}

    OPEN = {"open": True, "reason": "ok", "probe_size_factor": 1.0,
            "consecutive_losses": 0}
    SHUT = {"open": False, "reason": "newest red", "probe_size_factor": 1.0,
            "consecutive_losses": 0}
    calls = []

    def submit(r):
        calls.append(r["ticker"])
        if r["ticker"] == "FAIL":
            return {"status": "failed", "detail": "rejected"}
        if r["ticker"] == "PEND":
            return {"status": "skipped", "detail": "pending buy"}
        return {"status": "submitted", "detail": "ok"}

    s = entries.execute_entry_plan(mkplan("AAA"), submit=submit, gate=OPEN,
                                   held_by_symbol={}, today="2026-08-20",
                                   enabled=False)
    assert s["status"] == "disabled" and not calls
    s = entries.execute_entry_plan(mkplan("AAA"), submit=submit, gate=OPEN,
                                   held_by_symbol={}, today="2026-08-21", enabled=True)
    assert s["status"] == "stale" and not calls
    s = entries.execute_entry_plan(mkplan("AAA"), submit=submit, gate=SHUT,
                                   held_by_symbol={}, today="2026-08-20", enabled=True)
    assert s["status"] == "gate_closed" and not calls
    s = entries.execute_entry_plan(mkplan("AAA"), submit=submit, gate=None,
                                   held_by_symbol={}, today="2026-08-20", enabled=True)
    assert s["status"] == "gate_closed" and not calls

    plan = mkplan("HELD", "DIS", "PEND", "AAA", "BBB")
    plan["rows"][1]["status"] = "disarmed"
    s = entries.execute_entry_plan(plan, submit=submit, gate=OPEN,
                                   held_by_symbol={"HELD": 10}, today="2026-08-20",
                                   enabled=True)
    assert calls == ["PEND", "AAA"], f"unexpected submits: {calls}"
    assert s["submitted"] == ["AAA"] and s["disarmed"] == ["DIS"]
    assert set(s["skipped"]) == {"HELD", "PEND", "BBB"}
    by = {r["ticker"]: r for r in plan["rows"]}
    assert by["HELD"]["status"] == "skipped" and "held" in by["HELD"]["detail"]
    assert by["AAA"]["status"] == "submitted"
    assert by["BBB"]["status"] == "skipped" and "one entry per day" in by["BBB"]["detail"]
    assert s["status"] == "ok" and plan["executed_at"]

    calls.clear()
    s2 = entries.execute_entry_plan(plan, submit=submit, gate=OPEN,
                                    held_by_symbol={}, today="2026-08-20",
                                    enabled=True)
    assert calls == [] and s2["submitted"] == []

    calls.clear()
    plan = mkplan("FAIL", "AAA")
    s = entries.execute_entry_plan(plan, submit=submit, gate=OPEN,
                                   held_by_symbol={}, today="2026-08-20", enabled=True)
    assert calls == ["FAIL"] and s["status"] == "failed"
    assert plan["rows"][1]["status"] == "armed", "the walk must stop on a failure"


def test_trade_panel_arm_and_disarm():
    """#24 arming UI: a LIMIT plan's Arm button writes tonight's entry plan from the
    session-edited widgets (checked buys only, held rows excluded); the armed section
    renders and its Disarm button rewrites the file (armed -> disarmed); a MARKET
    plan's Arm button is disabled (a market row would buy the open blind)."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_trade_panel_arm_and_disarm (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import entries, scan as scanmod, cache

    prices, spy, _ = _synthetic_slice()
    result = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                             cfg=ScanConfig(min_rs=0.0))

    def _entry(t, limit=52.5):
        return {"ticker": t, "shares": 10, "price": 50.0, "pivot": 50.0,
                "est_value": 525.0, "extended": False, "capped": False,
                "stop_price": 46.0, "limit_price": limit, "earnings_in": None}

    _wl = [{"ticker": t, "judged_pivot": None, "date_added": None,
            "pivot_source": None, "note": ""} for t in ("CLEAN", "HELDX")]
    _acct = {"account_number": "PA000123", "equity": 100000.0, "using_dedicated": True}
    app_path = str(ROOT / "src" / "stock_screener" / "cockpit" / "app.py")

    with tempfile.TemporaryDirectory() as _tmp, \
            patch.object(scanmod, "run_scan", return_value=result), \
            patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
            patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
        trg = Path(_tmp) / "triggers"
        at = AppTest.from_file(app_path, default_timeout=60)
        at.session_state["watchlist"] = list(_wl)
        at.session_state["trade_build_n"] = 1
        at.session_state["trade_plan"] = {
            "plan": [_entry("CLEAN"), _entry("HELDX")], "skipped": [],
            "account": dict(_acct), "held": {"HELDX": 20},
            "build_ts": 1, "order_type": "limit"}
        at.run()
        assert not at.exception, f"app raised: {at.exception}"
        arm = [b for b in at.button if b.key == "trade_arm"]
        assert arm and not arm[0].disabled, "limit plan must offer Arm"

        # Edit the widgets, then arm — the plan file must carry the EDITED values.
        [n for n in at.number_input if n.key == "lim_CLEAN_1"][0].set_value(53.0)
        [n for n in at.number_input if n.key == "stop_CLEAN_1"][0].set_value(47.0)
        arm[0].click().run()
        assert not at.exception, f"app raised on arm: {at.exception}"
        saved = entries.load_latest_entry_plan(trg)
        assert saved and [r["ticker"] for r in saved["rows"]] == ["CLEAN"], \
            "held rows must not arm"
        assert saved["rows"][0]["limit_price"] == 53.0
        assert saved["rows"][0]["stop_price"] == 47.0
        assert saved["rows"][0]["status"] == "armed"
        rendered = _rendered_text(at)
        assert "Armed for next open" in rendered, "armed section missing after arming"

        # Disarm rewrites the file.
        db = [b for b in at.button if str(b.key or "").startswith("disarm_CLEAN")]
        assert db, "disarm button missing"
        db[0].click()
        at.run()
        assert not at.exception, f"app raised on disarm: {at.exception}"
        saved = entries.load_latest_entry_plan(trg)
        assert saved["rows"][0]["status"] == "disarmed"

        # Market plan: Arm disabled.
        at2 = AppTest.from_file(app_path, default_timeout=60)
        at2.session_state["watchlist"] = [dict(_wl[0])]
        at2.session_state["trade_build_n"] = 1
        at2.session_state["trade_plan"] = {
            "plan": [{**_entry("CLEAN"), "limit_price": None}], "skipped": [],
            "account": dict(_acct), "held": {}, "build_ts": 1, "order_type": "market"}
        at2.run()
        assert not at2.exception, f"app raised: {at2.exception}"
        arm2 = [b for b in at2.button if b.key == "trade_arm"]
        assert arm2 and arm2[0].disabled, "market plan must not be armable"



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "entries"))
