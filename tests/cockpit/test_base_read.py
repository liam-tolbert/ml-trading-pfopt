"""Cockpit tests — the detector's descriptive reads (swings.py, vcp._reads) and the eight
base answers computed from them (base_read.py).

Runs standalone (`python -m tests.cockpit.test_base_read`) or as part of the full gate
(`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_tight_area_states():
    """The final area is read from the last bars alone: tight from 5 bars inside the
    range, loose under that, not formed until 3 bars (the trough's included) follow the
    last trough."""
    import numpy as np
    from src.stock_screener.cockpit.swings import tight_area

    kw = dict(max_range=0.08, min_bars=3, tight_bars=5, max_bars=10)
    close = np.full(20, 100.0)
    high, low = close + 2.0, close - 2.0                 # every bar spans 4%
    ok = tight_area(high, low, close, 20, 5, **kw)
    assert ok == ("tight", 10, 102.0), ok
    for t in (19, 18):
        assert tight_area(high, low, close, 20, t, **kw)[0] == "not_formed", t
    assert tight_area(high, low, close, 20, 17, **kw)[0] == "tight"
    assert tight_area(high, low, close, 20, None, **kw) == ("not_formed", 0, None)

    wide_high = high.copy()
    wide_high[15] = 110.0                                # 5 bars back the range opens to 12%
    st, bars, top = tight_area(wide_high, low, close, 20, 5, **kw)
    assert (st, bars, top) == ("loose", 4, 102.0), (st, bars, top)

    jumpy_low = low.copy()
    jumpy_low[18] = 90.0                                 # even the last 3 bars span 12%
    st, bars, top = tight_area(high, jumpy_low, close, 20, 5, **kw)
    assert st == "loose" and bars == 1 and top == 102.0, (st, bars, top)


def test_vcp_reads_describe_the_base():
    """detect_vcp returns the base's top, its final area and one ZigZag pass over the
    base, without changing its tier."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _lin_series([(100, 150), (10, 126), (8, 148), (10, 131.7), (8, 146),
                      (10, 135.8), (8, 144), (10, 136.8), (8, 143)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {}, thr=0.04)
    assert v["tier"] == "A" and v["pivot_price"] == 150.0, (v["tier"], v["pivot_price"])
    assert v["final_area"] == "tight" and v["area_high"] == 143.0, (v["final_area"], v["area_high"])
    assert v["breakout_bar_date"] is None
    assert [round(g["drawdown_pct"]) for g in v["zz_legs"]] == [16, 11, 7, 5], v["zz_legs"]
    assert v["swings_disagree"] is False
    assert v["first_leg_pct"] == v["contractions"][0]["drawdown_pct"]
    assert v["final_leg_pct"] == v["contractions"][-1]["drawdown_pct"]
    assert len(v["recoveries"]) == 4 and all(f > 0 and r > 0 for f, r in v["recoveries"])
    assert v["v_shape"]["flag"] is False

    ext = _ohlc_series([(60, 150), (10, 138), (8, 149), (8, 141.8), (8, 150),
                        (8, 145.5), (15, 175)])
    e = detect_vcp(ext, float(ext["Close"].iloc[-1]), {})
    assert e["tier"] == "B" and e["breakout_bar_date"] is not None, e["pattern_details"]
    assert e["breakout_bar_date"] < ext.index[-1]
    assert e["area_high"] is not None and e["area_high"] < e["pivot_price"] * 1.02


def test_vcp_reads_flag_an_earlier_wider_swing():
    """A swing deeper than the one before it, earlier than the newest two, is what the
    selected legs can leave out: the read flags it for the user. A wider newest swing is
    the form's own answer, not this flag."""
    from src.stock_screener.cockpit.vcp import _reads

    def reads(df, legs):
        h, lo, c = (df[k].to_numpy(dtype=float) for k in ("High", "Low", "Close"))
        return _reads(h, lo, c, df.index, [(p, t, d, 100.0) for p, t, d in legs], 100.0, 0.035)

    df = _lin_series([(40, 100), (6, 90), (6, 100), (6, 95), (6, 100), (6, 88),
                      (6, 100), (6, 96), (6, 99)], start=80.0)
    r = reads(df, [(40, 46, 10.0), (76, 82, 4.0)])
    assert [round(g["drawdown_pct"]) for g in r["zz_legs"]] == [10, 5, 12, 4], r["zz_legs"]
    assert r["swings_disagree"] is True
    calm = _lin_series([(40, 100), (6, 90), (6, 100), (6, 95), (6, 100), (6, 96),
                        (6, 99)], start=80.0)
    assert reads(calm, [(40, 46, 10.0), (64, 70, 4.0)])["swings_disagree"] is False
    assert reads(calm, [])["zz_legs"] == []


def test_v_shape_read_on_benchmark():
    """§6.109: the V read, pre-registered before it ran: it would ship as an answer only if
    it fired on >= 7 of the 13 V-labelled charts and on <= 25% of the YES charts. It
    fired on 0 of 13. Those labels are retired as a check (§6.110), so the counts are only
    printed; the switch stays off until a check on the user's own hunt labels passes."""
    from vcp_labels import LABELS
    from src.stock_screener.cockpit import base_read
    from tests.cockpit.test_vcp import _bench_results

    v_named = ("UTI", "WLFC", "PRLD", "AGX", "CGEM", "DSGN", "EPC", "KB", "LINE", "MPC",
               "RLAY", "SRRK", "VTRS")
    bench = _bench_results()
    flagged = {t for t, (_, r) in bench.items() if (r.get("v_shape") or {}).get("flag")}
    yes = {t for t in LABELS if LABELS[t]["label"] == "YES"}
    print(f"    v-shape (report only): V-labelled {len(flagged & set(v_named))}/{len(v_named)}  "
          f"YES {len(flagged & yes)}/{len(yes)}  "
          f"other NO {len(flagged - yes - set(v_named))}/{len(LABELS) - len(yes) - len(v_named)}")
    assert base_read.V_SHAPE_ENABLED is False


def test_typical_day_window_matches_dead_tape():
    """advisories' typical day and the detector's dead-tape check read the same window,
    so the stop room and the barely-moving rule describe the same days."""
    from src.stock_screener.cockpit import advisories, vcp
    assert advisories.TYPICAL_DAY_BARS == vcp.DEAD_TAPE_BARS


def _vcp(**kw) -> dict:
    v = {"contraction_count": 3, "zz_legs": [{"drawdown_pct": 20.0}, {"drawdown_pct": 10.0},
                                             {"drawdown_pct": 5.0}],
         "swings_disagree": False, "final_area": "tight", "area_bars": 7,
         "pivot_price": 100.0, "area_high": 98.0, "median_tr_pct": 2.5,
         "recoveries": [(5, 6), (4, 5)], "v_shape": {"flag": False}}
    v.update(kw)
    return v


_FACTS = {"heavy_down_days_in_base": 1, "up_down_volume_in_base": 1.3, "dryup_ratio": 0.6,
          "closed_below_200_in_base": False, "legs": []}


def _verdict(vcp=None, facts=None, row=None):
    from src.stock_screener.cockpit.base_read import read_form
    from src.stock_screener.hunt.reviewer import validate_form, verdict_from_form
    facts = {**_FACTS, **(facts or {})}
    row = {"stop_room": 3.0, **(row or {})}
    form = read_form(_vcp(**(vcp or {})), facts, row)
    checks = form.pop("checks")
    assert validate_form(form) == [], validate_form(form)
    return form, verdict_from_form(form, facts, row)[:2], checks


def test_read_form_answers_from_numbers():
    """Each answer comes from a number, and the reviewer's verdict table turns the
    answers into the same codes it gives Claude's."""
    form, verdict, checks = _verdict()
    assert verdict == ("PASS", ["OK"]) and checks == [], (verdict, form)
    assert form["base_count"] == "cant_tell" and form["sure"] is True

    cases = (
        ({"vcp": {"contraction_count": 1}}, ("FAIL", ["DQ:no-base"])),
        ({"vcp": {"zz_legs": [{"drawdown_pct": 5.0}, {"drawdown_pct": 5.6}]}},
         ("FAIL", ["DQ:expanding"])),
        ({"facts": {"heavy_down_days_in_base": 3, "up_down_volume_in_base": 0.9}},
         ("FAIL", ["DQ:distribution"])),
        ({"vcp": {"median_tr_pct": 8.0}}, ("FAIL", ["DQ:bronco"])),
        ({"vcp": {"final_area": "loose"}}, ("PASS-", ["CAV:loose"])),
        ({"vcp": {"final_area": "not_formed", "area_high": None}}, ("PASS-", ["CAV:pivot"])),
        ({"vcp": {"area_high": 94.0}}, ("PASS-", ["CAV:pivot"])),
        ({"facts": {"dryup_ratio": 1.0}}, ("PASS-", ["CAV:no-dryup"])),
        ({"facts": {"dryup_ratio": None}}, ("PASS-", ["CAV:no-dryup"])),
    )
    for kw, want in cases:
        assert _verdict(**kw)[1] == want, (kw, _verdict(**kw)[1])


def test_read_form_boundaries():
    """The boundaries sit where the constants say: a swing exactly 1.10x the one before is
    not wider, 3 heavy down days need more volume on down days than up days, a base top
    under 5% over the area high is still one level."""
    form, _, _ = _verdict(vcp={"zz_legs": [{"drawdown_pct": 5.0}, {"drawdown_pct": 5.5}]})
    assert form["right_side"] == "orderly_tightening"
    form, _, _ = _verdict(facts={"heavy_down_days_in_base": 3, "up_down_volume_in_base": 1.0})
    assert form["volume"] == "drying_up"
    form, _, _ = _verdict(vcp={"area_high": 95.5})
    assert form["pivot"] == "clear"
    form, _, _ = _verdict(vcp={"median_tr_pct": 7.99}, row={"stop_room": 0.99})
    assert form["character"] == "wild_swings"


def test_read_form_leaves_unknowns_and_flags():
    """Grinding and a disagreeing swing change no verdict; the disagreement is a check
    for the user. A V answer needs the switch the pre-registered check left off."""
    from src.stock_screener.cockpit import base_read

    form, verdict, _ = _verdict(vcp={"recoveries": [(2, 5), (2, 5), (4, 5)]})
    assert form["character"] == "grinding" and verdict[0] == "PASS"
    form, verdict, checks = _verdict(vcp={"swings_disagree": True})
    assert form["swings_match"] == "no" and verdict[0] == "PASS" and len(checks) == 1
    v = {"v_shape": {"flag": True, "depth_pct": 20.0, "fall_bars": 10, "rise_bars": 4}}
    assert _verdict(vcp=v)[0]["right_side"] == "orderly_tightening"
    base_read.V_SHAPE_ENABLED = True
    try:
        assert _verdict(vcp=v)[1] == ("FAIL", ["DQ:v-shape"])
    finally:
        base_read.V_SHAPE_ENABLED = False
    form, _, _ = _verdict(vcp={"zz_legs": [], "recoveries": []})
    assert form["right_side"] == "orderly_tightening" and form["character"] == "quick_recoveries"


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "base read"))
