"""Cockpit tests — display-only SEPA advisories (stop room, base count, post-breakout
reads, the tape).

Runs standalone (`python -m tests.cockpit.test_advisories`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def _bars(closes, *, spread=0.01, volume=1e6, start="2026-01-02"):
    """OHLCV with Open at the prior close and High/Low ``spread`` beyond the bar's body, so
    an up day closes in the upper half of its range and a down day in the lower half.
    ``volume`` is a scalar or one value per bar."""
    import numpy as np
    import pandas as pd
    c = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(c))
    o = np.r_[c[0], c[:-1]]
    v = (np.asarray(volume, dtype=float) if np.ndim(volume)
         else np.full(len(c), float(volume)))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + spread),
                         "Low": np.minimum(o, c) * (1 - spread), "Close": c,
                         "Volume": v}, index=idx)


def _breakout(post, post_vol=None, entry_vol=1e6):
    """51 bars rising 80 -> 100 (the entry bar is the last of them, day 0), then ``post``.
    Returns (frame, entry_date)."""
    import numpy as np
    pre = list(np.linspace(80.0, 100.0, 51))
    vols = [1e6] * 50 + [entry_vol] + list(post_vol or [1e6] * len(post))
    df = _bars(pre + list(post), volume=vols)
    return df, df.index[50]


def test_typical_day_range_and_stop_room():
    """§6.75 (audit #5): the typical day is the MEDIAN true range over ~2 months (one gap
    day must not make a quiet stock look wild), and a stop fewer than STOP_ROOM_MIN_DAYS
    ordinary days below the fill is flagged — the book's bucking bronco."""
    from src.stock_screener.cockpit import advisories
    from src.stock_screener.cockpit.doctrine import STOP_ROOM_MIN_DAYS

    flat = _bars([100.0] * 60)                          # H/L ±1% -> a 2% true range
    dr = advisories.typical_day_range(flat)
    assert abs(dr - 0.02) < 1e-9, dr
    gappy = flat.copy()
    gappy.iloc[-5, gappy.columns.get_loc("High")] = 130.0          # one wild day
    assert abs(advisories.typical_day_range(gappy) - 0.02) < 1e-9, "median, not mean"
    assert advisories.typical_day_range(flat.iloc[:20]) is None     # too few bars

    tight = advisories.stop_room(0.02, 97.0, 100.0)                 # 3% stop, 2%/day
    assert abs(tight["room_days"] - 1.5) < 1e-9 and tight["warn"] is True
    assert STOP_ROOM_MIN_DAYS == 2.0
    roomy = advisories.stop_room(0.02, 92.5, 100.0)                 # 7.5% stop
    assert abs(roomy["room_days"] - 3.75) < 1e-9 and roomy["warn"] is False
    for bad in ((None, 92.5, 100.0), (0.02, 101.0, 100.0), (0.02, None, 100.0), (0, 95, 100)):
        assert advisories.stop_room(*bad) is None, bad
    assert advisories.stop_room_text(tight, 0.02).startswith("⚠ stop 1.5 typical days")
    assert advisories.stop_room_text(None, 0.02) == ""


def test_vcp_median_tr_exposed():
    """§6.75: detect_vcp exports the typical day it already measured for the dead-tape
    gate, on the adaptive path, the pinned-threshold path and the dead-tape exclusion
    alike; the scan row carries it as `day_range`, and a trade-plan row as
    `day_range_pct`."""
    import pandas as pd
    from src.stock_screener.cockpit.trade import build_buy_plan
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _bars([100.0 + (i % 7) for i in range(120)])
    adaptive = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    pinned = detect_vcp(df, float(df["Close"].iloc[-1]), {}, thr=0.04)
    assert adaptive["median_tr_pct"] and adaptive["median_tr_pct"] == pinned["median_tr_pct"]
    dead = detect_vcp(_bars([50.0] * 120, spread=0.002), 50.0, {})
    assert dead["tier"] == "C" and "Dead tape" in dead["pattern_details"]
    assert abs(dead["median_tr_pct"] - 0.4) < 1e-9
    assert detect_vcp(df.iloc[:30], 100.0, {})["median_tr_pct"] is None

    prices, spy, _ = _synthetic_slice()
    res = screen_universe(list(prices), prices, spy, get_fundamentals=None,
                          cfg=ScanConfig(min_rs=0.0))
    assert "day_range" in res.candidates.columns
    t = res.candidates["ticker"].iloc[0]
    assert res.candidates["day_range"].iloc[0] == res.payloads[t]["vcp"]["median_tr_pct"]
    plan, _ = build_buy_plan([t], res.payloads, mode="shares", amount=1)
    assert plan[0]["day_range_pct"] == res.payloads[t]["vcp"]["median_tr_pct"]
    assert isinstance(res.candidates, pd.DataFrame)


def test_bar_is_provisional():
    """§6.77: today's bar is provisional until the session settles (~16:05 ET, ~13:05 on
    an early close); any earlier bar never is."""
    from src.stock_screener.cockpit.triggers import bar_is_provisional

    assert bar_is_provisional("2026-08-12", now="2026-08-12 11:00") is True
    assert bar_is_provisional("2026-08-12", now="2026-08-12 16:30") is False
    assert bar_is_provisional("2026-08-11", now="2026-08-12 11:00") is False
    assert bar_is_provisional("2026-11-27", now="2026-11-27 12:30") is True   # half day
    assert bar_is_provisional("2026-11-27", now="2026-11-27 13:30") is False
    assert bar_is_provisional("not a date", now="2026-08-12 11:00") is False


def test_sma20_warn_window():
    """§6.77 (audit #6): a fresh breakout should hold its 20-day line for the first ~month.
    A close under it inside POST_BREAKOUT_DAYS is a violation (the note names the latest
    day and how many); the same close on day 25 is not; a flat tape (close == the line)
    never trips it."""
    from src.stock_screener.cockpit import advisories

    df, e = _breakout([101.0, 102.0, 95.0, 97.0, 99.0])
    r = advisories.post_breakout_read(df, e, avg_entry=100.0, today=df.index[-1])
    assert r["day_n"] == 5
    assert r["sma20"] == "closed below the 20-day line on day 4 (2×)", r["sma20"]

    late, e2 = _breakout([100.5 + 0.5 * k for k in range(24)] + [95.0])
    r2 = advisories.post_breakout_read(late, e2, avg_entry=100.0, today=late.index[-1])
    assert r2["day_n"] == 25 and r2["sma20"] is None, r2

    flat = _bars([100.0] * 60)
    r3 = advisories.post_breakout_read(flat, flat.index[40], avg_entry=100.0,
                                       today=flat.index[-1])
    assert r3["violations"] == [] and r3["follow_through"] == [], r3


def test_post_breakout_drops_provisional_bar():
    """§6.77: a live read ignores today's bar while the session is open — its "close" is
    the latest print, and an intraday dip under the 20-day line is not a close under it.
    After the settle, or with an explicit `today` (tests, --date replays), it counts."""
    from src.stock_screener.cockpit import advisories

    df, e = _breakout([101.0, 102.0, 95.0])                   # the dip is the LAST bar
    day = df.index[-1].strftime("%Y-%m-%d")
    live = advisories.post_breakout_read(df, e, now=f"{day} 11:00")
    assert live["provisional_dropped"] is True and live["day_n"] == 2
    assert live["sma20"] is None, live
    settled = advisories.post_breakout_read(df, e, now=f"{day} 16:30")
    assert settled["provisional_dropped"] is False and settled["sma20"], settled
    replay = advisories.post_breakout_read(df, e, today=day, now=f"{day} 11:00")
    assert replay["provisional_dropped"] is False and replay["sma20"], replay


def test_post_breakout_violations_each_fire():
    """§6.77 (audit #7): each of Minervini's post-breakout violations, on a frame built to
    show it — a light-volume breakout then a heavy down day, lower lows with nobody
    stepping in, more down closes / lower-half closes than up / upper-half, a close under
    the 50-day on heavy volume, a +5% gain given back."""
    from src.stock_screener.cockpit import advisories

    # light breakout (0.9x) then a heavy (2.0x) down day
    df, e = _breakout([101.0, 102.0, 95.0, 97.0, 99.0],
                      post_vol=[1e6, 1e6, 2e6, 1e6, 1e6], entry_vol=0.9e6)
    v = advisories.post_breakout_read(df, e, avg_entry=100.0, today=df.index[-1])["violations"]
    assert any(x.startswith("light-volume breakout (0.9×) then a heavy down day (day 3, 2.0×")
               for x in v), v

    # drifting lower on ordinary volume
    df, e = _breakout([101.0, 100.5, 100.0, 99.5, 99.0])
    v = advisories.post_breakout_read(df, e, avg_entry=100.0, today=df.index[-1])["violations"]
    assert "3 lower lows in a row (to day 5)" in v, v
    assert "more down closes than up (4 vs 1)" in v, v
    assert any(x.startswith("more closes in the lower half") for x in v), v

    # a +7% gain handed back, and the 50-day broken on heavy volume
    df, e = _breakout([104.0, 106.0, 100.0, 99.5])
    v = advisories.post_breakout_read(df, e, avg_entry=100.0, below_sma50=True,
                                      volume_ratio=1.8, today=df.index[-1])["violations"]
    assert "gave back a +7% gain" in v, v
    assert "closed below the 50-day on heavy volume" in v, v


def test_post_breakout_follow_through():
    """§6.77: the positive side — up closes on rising volume, 3 of the first 4 and 6 of
    the first 8 sessions up, more upper-half closes, and dips recovered within 2 days."""
    from src.stock_screener.cockpit import advisories

    ups = [101.0 + k for k in range(8)]
    df, e = _breakout(ups, post_vol=[1.1e6 + 1e5 * k for k in range(8)])
    f = advisories.post_breakout_read(df, e, avg_entry=100.0,
                                      today=df.index[-1])["follow_through"]
    assert "8 up close(s) on rising volume" in f, f
    assert "3 of the first 4 sessions up" in f and "6 of the first 8 sessions up" in f, f
    assert "more upper-half closes (8 vs 0)" in f, f

    df, e = _breakout([102.0, 101.0, 103.0, 104.0])            # one dip, bought back
    f = advisories.post_breakout_read(df, e, avg_entry=100.0,
                                      today=df.index[-1])["follow_through"]
    assert "every dip recovered within 2 sessions" in f, f


def test_violations_switch_and_plan_notes():
    """§6.77: violations are P1 WARNINGS — they reach the evening plan's notes and never
    an order — until doctrine.VIOLATIONS_CAN_FAIL is switched on, when
    VIOLATION_FAIL_COUNT of them fail P1 (and a P1 fail plans a full exit). The switch is
    read at call time."""
    from unittest.mock import patch
    from src.stock_screener.cockpit import doctrine, sells
    from src.stock_screener.cockpit.trade import sell_pillars

    df, e = _breakout([101.0, 100.5, 100.0, 99.5, 99.0])
    pos = {"symbol": "AAA", "qty": 10, "last_close": 99.0, "gain_pct": -0.01, "df": df,
           "template_criteria": 8, "earnings_in": None, "avg_entry": 100.0}
    today = df.index[-1].strftime("%Y-%m-%d")

    off = sell_pillars(pos, entry_date=e, today=today)
    assert off["P1"]["status"] == "warn" and "violations:" in off["P1"]["detail"], off["P1"]
    plan = sells.build_sell_plan([pos], {"AAA": off}, today=today)
    assert not plan["orders"], "a warning must never plan an order"
    assert any("violations:" in n for n in plan["notes"]), plan["notes"]

    assert doctrine.VIOLATIONS_CAN_FAIL is False, "ships off"
    with patch.object(doctrine, "VIOLATIONS_CAN_FAIL", True):
        on = sell_pillars(pos, entry_date=e, today=today)
    assert on["P1"]["status"] == "fail" and "post-breakout violations" in on["P1"]["detail"]
    plan = sells.build_sell_plan([pos], {"AAA": on}, today=today)
    assert [o["symbol"] for o in plan["orders"]] == ["AAA"]


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "advisories"))
