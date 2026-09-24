"""Cockpit tests — display-only SEPA advisories (stop room, base count, post-breakout
reads, the tape).

Runs standalone (`python -m tests.cockpit.test_advisories`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def _bars(closes, *, spread=0.01, volume=1e6, start="2026-01-02"):
    """OHLCV with High/Low ±``spread`` around each close and Open at the prior close."""
    import numpy as np
    import pandas as pd
    c = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(c))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + spread),
                         "Low": np.minimum(o, c) * (1 - spread), "Close": c,
                         "Volume": np.full(len(c), float(volume))}, index=idx)


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


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "advisories"))
