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


def test_regime_tier_prefix():
    """§6.78 (audit #8): tiers by label PREFIX. "TRANSITIONAL / Uncertain" contains "on",
    and the old substring test painted it green (risk-on); a Weak/Mixed risk-on is not a
    strong tape either."""
    from src.stock_screener.cockpit.advisories import regime_tier

    assert regime_tier("RISK-ON (Strong)") == "strong"
    assert regime_tier("RISK-ON (Moderate)") == "strong"
    assert regime_tier("RISK-ON (Weak) / Mixed") == "weak"
    assert regime_tier("TRANSITIONAL / Uncertain") == "weak"
    assert regime_tier("RISK-OFF") == "off"
    assert regime_tier(None) == "unknown" and regime_tier("") == "unknown"


def test_weak_market_advice():
    """§6.78: in a weak or risk-off tape the book tightens (5-6% stops, 10-12% profits,
    smaller size) — stated beside the plan's own numbers, never applied. Silent in a
    strong or unknown tape; falls back to the trigger report's SPY-only read."""
    from src.stock_screener.cockpit import advisories

    strong = {"regime": "RISK-ON (Strong)", "should_generate_buys": True}
    assert advisories.weak_market_advice(strong, stop_pct=0.075) is None
    assert advisories.weak_market_advice(None, None) is None

    txt = advisories.weak_market_advice({"regime": "TRANSITIONAL / Uncertain"},
                                        stop_pct=0.075, target_pct=0.25, risk_pct=1.0)
    assert txt.startswith("TRANSITIONAL / Uncertain")
    assert "stops 5–6% below the buy (plan: 7.5%)" in txt, txt
    assert "profits taken at 10–12% (plan target +25%)" in txt, txt
    assert "(plan: 1.00% risk per trade)" in txt, txt

    assert advisories.weak_market_advice(None, {"trend": "Bullish"}) is None
    assert advisories.weak_market_advice(None, {"trend": "Bearish"}).startswith("SPY Bearish")
    assert advisories.in_weak_take_profit_band(0.11)
    assert not advisories.in_weak_take_profit_band(0.08)
    assert not advisories.in_weak_take_profit_band(None)


def test_spy_confirm_streak():
    """§6.78 (audit #9): the re-entry lag counts consecutive settled sessions, newest
    first, with SPY in Stage 1-2, stopping at REGIME_CONFIRM_DAYS (satisfied) — so SPY is
    classified at most 15 times. A Stage 3/4 session breaks the streak."""
    from unittest.mock import patch
    import pandas as pd
    from src.stock_screener.cockpit import advisories
    from src.stock_screener.cockpit.doctrine import REGIME_CONFIRM_DAYS
    from src.stock_screener.minervini_screener import screening

    spy = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2025-01-01", periods=300))
    calls = []

    def fake_phase(phases_by_age):
        def _f(sub, cp):
            age = len(spy) - len(sub)                     # 0 = today, 1 = yesterday ...
            calls.append(age)
            return {"phase": phases_by_age(age)}
        return _f

    with patch.object(screening, "classify_phase", fake_phase(lambda a: 2 if a < 6 else 4)):
        r = advisories.spy_confirm_streak(spy)
    assert r == {"streak": 6, "satisfied": False, "phase_now": 2,
                 "breadth": False, "partial": False}, r

    calls.clear()
    with patch.object(screening, "classify_phase", fake_phase(lambda a: 1 if a % 2 else 2)):
        r = advisories.spy_confirm_streak(spy)
    assert r["streak"] == REGIME_CONFIRM_DAYS and r["satisfied"] is True
    assert len(calls) == REGIME_CONFIRM_DAYS, "classified more often than the lag needs"

    with patch.object(screening, "classify_phase", fake_phase(lambda a: 4)):
        r = advisories.spy_confirm_streak(spy)
    assert r["streak"] == 0 and r["phase_now"] == 4
    assert advisories.spy_confirm_streak(spy.iloc[:150]) is None


def test_spy_confirm_streak_with_breadth():
    """§6.82 (audit Step-1 #8): with a breadth history, a session counts toward the
    re-entry lag only when SPY was in Stage 1-2 AND at least 15% of the universe was in
    Stage 2, as the backtest required. A session with no breadth row counts on SPY alone
    and marks the read partial. A sub-15% session breaks the streak."""
    from unittest.mock import patch
    import pandas as pd
    from src.stock_screener.cockpit import advisories
    from src.stock_screener.cockpit.doctrine import BREADTH_MIN_PHASE2, REGIME_CONFIRM_DAYS
    from src.stock_screener.minervini_screener import screening

    assert BREADTH_MIN_PHASE2 == 15.0, "MUST match should_generate_signals' default"
    spy = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2025-01-01", periods=300))
    days = [d.strftime("%Y-%m-%d") for d in spy.index]
    stage2 = lambda sub, cp: {"phase": 2}                       # noqa: E731

    # every session has ample breadth -> the full lag, with breadth, not partial
    full = {d: 22.0 for d in days}
    with patch.object(screening, "classify_phase", stage2):
        r = advisories.spy_confirm_streak(spy, phase2_by_date=full)
    assert r["streak"] == REGIME_CONFIRM_DAYS and r["satisfied"] is True
    assert r["breadth"] is True and r["partial"] is False

    # the session 3 back had 12% breadth -> the streak stops at 3
    thin = {**full, days[-4]: 12.0}
    with patch.object(screening, "classify_phase", stage2):
        r = advisories.spy_confirm_streak(spy, phase2_by_date=thin)
    assert r["streak"] == 3 and r["satisfied"] is False and r["partial"] is False

    # a history that starts 5 sessions ago: older sessions count on SPY alone, partial
    recent = {d: 22.0 for d in days[-5:]}
    with patch.object(screening, "classify_phase", stage2):
        r = advisories.spy_confirm_streak(spy, phase2_by_date=recent)
    assert r["streak"] == REGIME_CONFIRM_DAYS and r["partial"] is True

    # no map at all: SPY only
    with patch.object(screening, "classify_phase", stage2):
        r = advisories.spy_confirm_streak(spy)
    assert r["breadth"] is False and r["partial"] is False


def test_market_turn_transition_only():
    """§6.78: the market turn fires on the evening SPY ENTERS Stage 4 — not every evening
    it stays there (a "reduce" rule would halve the book night after night). With no prior
    plan to date it against, Stage 4 is reported unconfirmed, never as a turn."""
    from src.stock_screener.cockpit.advisories import market_turn

    s4, s2 = {"phase": 4, "trend": "Bearish"}, {"phase": 2, "trend": "Bullish"}
    t = market_turn(s4, {"spy_phase": 2})
    assert t["turn"] is True and t["stage4"] is True and t["unconfirmed"] is False
    stay = market_turn(s4, {"spy_phase": 4})
    assert stay["turn"] is False and stay["stage4"] is True and stay["unconfirmed"] is False
    first = market_turn(s4, None, has_prior_plan=False)
    assert first["turn"] is False and first["unconfirmed"] is True
    old = market_turn(s4, None, has_prior_plan=True)      # a plan from before "market"
    assert old["turn"] is False and old["unconfirmed"] is True
    assert market_turn(s2, {"spy_phase": 4})["turn"] is False
    assert market_turn(None, {"spy_phase": 2}) == {"spy_phase": None, "stage4": False,
                                                    "turn": False, "unconfirmed": False}


def test_depth_vs_market():
    """§6.83 (audit Step-1 #6): the base's deepest fall from a running high against the
    market's over the same dates, from the first contraction's peak. 2.3× is fine; 4×
    is flagged (the books: avoid > 2.5–3× the market); a market that barely moved, no
    contractions, or a window with under two bars read None."""
    import pandas as pd
    from src.stock_screener.cockpit import advisories

    idx = pd.bdate_range(end="2026-06-30", periods=60)

    def frame(path):
        c = pd.Series(path, index=idx, dtype=float)
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6})

    # stock: 80 -> 100 by bar 20, -23% to 77 by bar 40, back to 95; SPY: 300 -> 270 -> 300
    stock = ([80 + i for i in range(21)] + [100 - 23 * (i + 1) / 20 for i in range(20)]
             + [77 + 18 * (i + 1) / 19 for i in range(19)])
    spy = ([300.0] * 21 + [300 - 30 * (i + 1) / 20 for i in range(20)]
           + [270 + 30 * (i + 1) / 19 for i in range(19)])
    cons = [{"peak_date": idx[20], "trough_date": idx[40], "peak_price": 100.0,
             "trough_price": 77.0, "drawdown_pct": 23.0}]
    d = advisories.depth_vs_market(frame(stock), frame(spy), cons)
    assert d == {"depth_pct": 23.0, "spy_depth_pct": 10.0, "ratio": 2.3, "flag": False}, d
    text = advisories.depth_vs_market_text(d)
    assert "23%" in text and "2.3×" in text

    deep = stock[:21] + [100 - 40 * (i + 1) / 20 for i in range(20)] + [60.0] * 19
    d4 = advisories.depth_vs_market(frame(deep), frame(spy), cons)
    assert d4["ratio"] == 4.0 and d4["flag"] is True
    assert advisories.depth_vs_market_text(d4).startswith("⚠️")

    # the market barely moved -> no ratio; no contractions -> None; a window with < 2 bars
    assert advisories.depth_vs_market(frame(stock), frame([300.0] * 60), cons) is None
    assert advisories.depth_vs_market(frame(stock), frame(spy), []) is None
    late = [{"peak_date": idx[-1], "trough_date": idx[-1]}]
    assert advisories.depth_vs_market(frame(stock), frame(spy), late) is None
    assert advisories.depth_vs_market_text(None) == ""

    # a tz-aware stock index against a naive SPY still aligns on dates
    tz = frame(stock).tz_localize("America/New_York")
    assert advisories.depth_vs_market(tz, frame(spy), cons)["ratio"] == 2.3


def _reaction_frame(moves=None, n=80):
    """A flat 100.0 tape of 1,000-share days from 2026-07-01, with ``moves`` =
    ``{date: (open, close, volume)}`` overriding single bars."""
    import pandas as pd
    idx = pd.bdate_range("2026-07-01", periods=n)
    df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
                       "Volume": 1000.0}, index=idx)
    for d, (o, c, v) in (moves or {}).items():
        df.loc[pd.Timestamp(d), ["Open", "Close", "Volume"]] = [o, c, v]
    return df


def test_earnings_reaction_session_timing():
    """§6.87 (audit Step-2 #3): the reaction session is the release day's bar for a
    release before the open or during the session, and the next bar for one at or after
    16:00 New York time: a Friday-evening release reacts on Monday."""
    from src.stock_screener.cockpit import advisories

    df = _reaction_frame({"2026-09-11": (100.0, 97.0, 3000.0),      # Friday
                          "2026-09-14": (94.0, 92.0, 3000.0)})      # Monday
    after = advisories.earnings_reaction(df, "2026-09-11", "16:05")
    assert after["date"] == "2026-09-14"
    assert after["day_pct"] == round((92.0 / 97.0 - 1) * 100, 1)
    before = advisories.earnings_reaction(df, "2026-09-11", "08:00")
    assert before["date"] == "2026-09-11" and before["day_pct"] == -3.0
    assert advisories.earnings_reaction(df, "2026-09-11", "11:30")["date"] == "2026-09-11"
    assert advisories.earnings_reaction(df, "2026-09-11")["date"] == "2026-09-11"
    # the reaction bar hasn't printed yet: an after-close release on the last bar
    last = df.index[-1].strftime("%Y-%m-%d")
    assert advisories.earnings_reaction(df, last, "16:10") is None


def test_earnings_reaction_flags():
    """§6.87: −6% on 2× volume is a hard drop; −6% on 1.2× is not; +7% on 2× is strong. A
    release outside the frame, or over 100 days before its last bar, reads None."""
    from src.stock_screener.cockpit import advisories

    df = _reaction_frame({"2026-09-15": (95.0, 94.0, 2000.0)})
    r = advisories.earnings_reaction(df, "2026-09-15", "07:00")
    assert r["flag"] == "hard_drop" and r["vol_ratio"] == 2.0 and r["gap_pct"] == -5.0
    assert r["since_pct"] == round((100.0 / 94.0 - 1) * 100, 1)
    text = advisories.earnings_reaction_text(r, "2026-09-15", "07:00")
    assert text.startswith("**Last report** 2026-09-15 (before the open): -6.0% on 2.0×")
    assert "⚠️ a hard drop" in text
    light = _reaction_frame({"2026-09-15": (95.0, 94.0, 1200.0)})
    assert advisories.earnings_reaction(light, "2026-09-15", "07:00")["flag"] is None
    up = _reaction_frame({"2026-09-15": (104.0, 107.0, 2000.0)})
    assert advisories.earnings_reaction(up, "2026-09-15", "07:00")["flag"] == "strong"
    assert advisories.earnings_reaction(df, "2026-01-02", "07:00") is None
    long = _reaction_frame({"2026-07-06": (95.0, 94.0, 2000.0)}, n=160)
    assert advisories.earnings_reaction(long, "2026-07-06", "07:00") is None
    assert advisories.earnings_reaction(df, None) is None
    assert advisories.earnings_reaction_text(None) == ""


def test_volume_dryup():
    """§6.91 (audit Step-3 #1): the final tight area's volume against the 50-day average
    before it, and its near-silent days. A breakout close ends the window, so a name that
    already broke out is judged on its base; zero-volume bars are data gaps, not quiet
    days."""
    from src.stock_screener.cockpit import advisories

    df = _reaction_frame(n=80)
    peak = df.index[60]
    cons = [{"peak_date": peak, "peak_price": 105.0, "trough_price": 98.0}]
    df.loc[df.index[60:], "Volume"] = 400.0
    r = advisories.volume_dryup(df, cons)
    assert r["verdict"] == "dry" and r["avg_ratio"] == 0.4 and r["quiet_days"] == 20
    assert r["window_bars"] == 20 and len(r["quiet_dates"]) == 20
    assert "Volume dry-up ✅" in advisories.volume_dryup_text(r)

    loud = _reaction_frame(n=80)
    loud.loc[loud.index[60:], "Volume"] = 1500.0
    assert advisories.volume_dryup(loud, cons)["verdict"] == "none"
    assert advisories.volume_dryup_text(advisories.volume_dryup(loud, cons)).startswith("⚠️")

    half = _reaction_frame(n=80)
    half.loc[half.index[60:], "Volume"] = 900.0          # below average, no quiet day
    assert advisories.volume_dryup(half, cons)["verdict"] == "partial"

    broke = df.copy()
    broke.loc[broke.index[75:], ["Close", "Volume"]] = [110.0, 5000.0]
    b = advisories.volume_dryup(broke, cons)
    assert b["window_bars"] == 15 and b["verdict"] == "dry"

    gaps = df.copy()
    gaps.loc[gaps.index[62:66], "Volume"] = 0.0
    assert advisories.volume_dryup(gaps, cons)["quiet_days"] == 16

    assert advisories.volume_dryup(df, []) is None
    short = [{"peak_date": df.index[-2], "peak_price": 105.0}]
    assert advisories.volume_dryup(df, short) is None
    marks = advisories.step3_marks({"dryup": r})
    assert len(marks) == 20 and marks[0]["pane"] == "volume"
    assert advisories.step3_marks({}) == [] and advisories.step3_marks(None) == []


def test_shakeouts():
    """§6.92 (audit Step-3 #3): an undercut of a base low that closes back above it within 3
    bars is a shakeout (counting the undercut bar as 0); one that stays below is a lower
    low; one too recent to tell is open. One bar under two lows counts once."""
    from src.stock_screener.cockpit import advisories

    cons = [{"peak_date": "2026-08-03", "trough_date": "2026-08-26", "trough_price": 95.0},
            {"peak_date": "2026-09-01", "trough_date": "2026-09-15", "trough_price": 97.0}]

    def frame(lows=None, closes=None):
        df = _reaction_frame(n=80)
        for d, v in (lows or {}).items():
            df.loc[d, "Low"] = v
        for d, v in (closes or {}).items():
            df.loc[d, "Close"] = v
        return df

    same = advisories.shakeouts(frame({"2026-09-18": 96.5}), cons)
    e = same["latest"]
    assert same["shakeouts"] == 1 and e["status"] == "shakeout" and e["recovered_in"] == 0
    assert e["ref_low"] == 97.0 and e["undercut_pct"] == round((1 - 96.5 / 97) * 100, 1)
    later = advisories.shakeouts(frame({"2026-09-18": 96.5},
                                       {"2026-09-18": 96.8, "2026-09-21": 96.9,
                                        "2026-09-22": 96.9}), cons)
    assert later["latest"]["recovered_in"] == 3
    stays = {d: 96.0 for d in ("2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23",
                               "2026-09-24")}
    broken = advisories.shakeouts(frame({"2026-09-18": 95.5}, stays), cons)
    assert broken["broken"] == 1 and broken["latest"]["status"] == "broken"
    # one bar under both lows counts once, against the higher (97)
    both = advisories.shakeouts(frame({"2026-09-18": 94.0}, stays), cons)
    assert len(both["events"]) == 1 and both["events"][0]["ref_low"] == 97.0
    last = frame().index[-1]
    fresh = advisories.shakeouts(frame({last: 96.0}, {last: 96.0}), cons)
    assert fresh["latest"]["status"] == "open"
    clean = advisories.shakeouts(frame(), cons)
    assert clean["events"] == []
    assert advisories.shakeouts(frame(), cons[:1]) is None


def test_v_recovery():
    """§6.93 (audit Step-3 #4): the right side's pace against the decline. A 20% base that
    fell for 30 sessions and got back within 5% of its high in 5 is a V (6×); the same base
    recovering over 29 sessions is not; a shallow base is never a V; a base that hasn't
    recovered has no right side yet."""
    from src.stock_screener.cockpit import advisories

    def vframe(recover_at, floor=84.0, low_min=80.0):
        df = _reaction_frame(n=80)
        df.loc[df.index[21:], ["Close", "Low"]] = [floor + 1.0, floor]
        df.loc[df.index[50], "Low"] = low_min
        if recover_at is not None:
            df.loc[df.index[recover_at:], "Close"] = 96.0
        return df

    cons = [{"peak_date": _reaction_frame(n=80).index[20], "peak_price": 100.0}]
    fast = advisories.v_recovery(vframe(55), cons)
    assert fast == {"depth_pct": 20.0, "left_bars": 30, "right_bars": 5, "speed": 6.0,
                    "v_flag": True}, fast
    slow = advisories.v_recovery(vframe(79), cons)
    assert slow["right_bars"] == 29 and slow["v_flag"] is False
    shallow = advisories.v_recovery(vframe(55, floor=93.0, low_min=92.0), cons)
    assert shallow["depth_pct"] == 8.0 and shallow["v_flag"] is False
    assert advisories.v_recovery(vframe(None), cons) is None
    assert advisories.v_recovery(vframe(55), []) is None


def test_book_tightening():
    """§6.94 (audit Step-3 #2, #6): the books' rule of thumb, each dip about half the one
    before, shown beside the detector's looser rule; the base length is shown, not
    judged."""
    from src.stock_screener.cockpit import advisories

    legs = lambda *d: [{"drawdown_pct": x} for x in d]              # noqa: E731
    half = advisories.book_tightening(legs(24.0, 13.0, 6.0), 6.14)
    assert half == {"depths": [24.0, 13.0, 6.0], "ratios": [0.54, 0.46], "book_tight": True,
                    "base_weeks": 6.1}
    assert advisories.book_tightening_text(half) == (
        "Dips 24% → 13% → 6% (each 0.54×, 0.46× the last): the books' halving ✅ · "
        "base 6.1 weeks (books: ≥ 3)")
    loose = advisories.book_tightening(legs(20.0, 18.0, 16.0))
    assert loose["book_tight"] is False and loose["base_weeks"] is None
    assert "looser than the books' halving" in advisories.book_tightening_text(loose)
    assert advisories.book_tightening(legs(0.0, 5.0))["book_tight"] is False
    assert advisories.book_tightening(legs(20.0)) is None
    assert advisories.book_tightening_text(None) == ""


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "advisories"))
