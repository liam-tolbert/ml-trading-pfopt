"""Cockpit tests — trigger evaluation (pure) plus the refresh job that feeds it.

Runs standalone (`python tests/cockpit/test_triggers.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_volume_ratio_excludes_the_bar_it_measures():
    """The shared confirmation read: last bar over the mean of the PRIOR window.

    50 flat bars of 1000 then a 3000 spike is exactly 3.0. Including the spike in its own
    average would give 2.96, and the error grows with the spike — precisely the bars a
    breakout gate exists to judge. Too little history is None, never a confident ratio off
    a partial window."""
    from src.stock_screener.cockpit.indicators import prior_volume_average, volume_ratio
    import pandas as pd

    v = [1000.0] * 50 + [3000.0]
    df = pd.DataFrame({"Volume": v})
    assert abs(volume_ratio(df, 50) - 3.0) < 1e-9
    assert volume_ratio(pd.DataFrame({"Volume": v[:50]}), 50) is None, "50 bars is not enough"
    assert volume_ratio(pd.DataFrame({"Close": [1.0] * 60}), 50) is None, "no Volume column"

    avg = prior_volume_average(pd.Series(v), 50)
    assert abs(float(avg.iloc[-1]) - 1000.0) < 1e-9, "the spike must not be in its own average"
    assert pd.isna(avg.iloc[49]), "NaN until a full window exists"


def test_check_triggers_pure():
    """The nightly gate: close above the FROZEN pivot on >=1.5x 50-day volume, today's bar
    only. Flat volume blocks a trigger, extended (> pivot*1.05) files under don't-chase
    even when price+volume cleared, a stale bar never fires, below-pivot is a watch, and
    the earnings flag + summary lists come through."""
    from src.stock_screener.cockpit.export import make_entry
    from src.stock_screener.cockpit.triggers import STATUSES, check_one, check_triggers

    TODAY = "2026-07-10"                                    # a Friday
    flat = [100.0] * 60
    spike_vol = [1000] * 59 + [3000]                        # 3x the prior 50-day average

    def E(t, pivot):
        return make_entry(t, pivot, date_added="2026-07-01", pivot_source="judged")

    # (a) close 100 > pivot 98 on 3x volume, bar dated today -> TRIGGERED. After the
    # close (16:30) the session is fully elapsed, so pace == the plain 50-day ratio.
    r = check_one(E("TRG", 98.0), _trigger_frame(TODAY, flat, spike_vol), None,
                  today=TODAY, now=f"{TODAY} 16:30")
    assert r["status"] == "triggered" and r["triggered"] is True, r
    assert r["close_above_pivot"] and r["volume_confirmed"] and not r["stale"]
    assert abs(r["volume_ratio_50"] - 3.0) < 1e-9
    assert abs(r["volume_pace"] - 3.0) < 1e-9            # elapsed=1.0 -> pace == ratio
    assert abs(r["pct_from_pivot"] - (100.0 / 98.0 - 1) * 100) < 0.01

    # intraday pace: at 12:45 half the 09:30-16:00 session has elapsed -> the same
    # volume reads at DOUBLE pace ("running hot for the time of day"); the gate is
    # untouched (still the plain ratio).
    r_mid = check_one(E("TRG", 98.0), _trigger_frame(TODAY, flat, spike_vol), None,
                      today=TODAY, now=f"{TODAY} 12:45")
    assert abs(r_mid["volume_pace"] - 6.0) < 1e-9        # 3.0 / 0.5
    assert r_mid["triggered"] is True                    # gate = actual ratio, not pace

    # (b) same but flat volume -> price cleared, volume didn't -> CROSSED (§6.19's quiet
    # drift, built §6.36): loud in the report, distinct from a below-pivot "watch", and
    # still NOT triggered — a volume-less cross is never a buy signal.
    r2 = check_one(E("NOV", 98.0), _trigger_frame(TODAY, flat), None, today=TODAY)
    assert r2["status"] == "crossed" and r2["triggered"] is False, r2
    assert r2["close_above_pivot"] is True and r2["volume_confirmed"] is False
    assert r2["crossed"] is True

    # (c) extended: close 100 vs pivot 90 (+11%) on volume -> don't chase beats triggered
    r3 = check_one(E("EXT", 90.0), _trigger_frame(TODAY, flat, spike_vol), None, today=TODAY)
    assert r3["status"] == "extended" and r3["extended"] is True

    # (d) stale bar (run date is the next business day) -> never fires, and pace is
    # meaningless off a stale bar -> None. The raw `crossed` boolean is also False
    # (R2-10: symmetry with `triggered` — a stale bar above the pivot is no live cross).
    r4 = check_one(E("STL", 98.0), _trigger_frame(TODAY, flat, spike_vol), None,
                   today="2026-07-13", now="2026-07-13 12:00")
    assert r4["status"] == "stale" and r4["stale"] is True and r4["triggered"] is False
    assert r4["crossed"] is False
    assert r4["volume_pace"] is None
    # (d2) the R2-10 tooth: stale + FLAT volume (the quiet cross shape) — the raw
    # crossed boolean must read False, not just be hidden by status precedence
    r4b = check_one(E("STL2", 98.0), _trigger_frame(TODAY, flat), None,
                    today="2026-07-13", now="2026-07-13 12:00")
    assert r4b["status"] == "stale" and r4b["close_above_pivot"] is True
    assert r4b["crossed"] is False, "a stale bar above the pivot is not a live cross"

    # (e) below the pivot -> watch, negative distance, and NOT crossed
    r5 = check_one(E("BLW", 105.0), _trigger_frame(TODAY, flat), None, today=TODAY)
    assert r5["status"] == "watch" and r5["pct_from_pivot"] < 0
    assert r5["crossed"] is False

    # earnings flag rides in from the fundamentals dict (10 days out -> soon)
    r6 = check_one(E("ERN", 98.0), _trigger_frame(TODAY, flat),
                   {"next_earnings": "2026-07-20"}, today=TODAY)
    assert r6["earnings_in"] == 10 and r6["earnings_soon"] is True

    # full report: summary buckets by STATUS; missing frame -> no_data; spy note present
    entries = [E("TRG", 98.0), E("NOV", 98.0), E("EXT", 90.0), E("BLW", 105.0),
               make_entry("GONE"), make_entry("UNF")]      # GONE: no frame; UNF: no pivot
    prices = {"TRG": _trigger_frame(TODAY, flat, spike_vol),
              "NOV": _trigger_frame(TODAY, flat),
              "EXT": _trigger_frame(TODAY, flat, spike_vol),
              "BLW": _trigger_frame(TODAY, flat),
              "UNF": _trigger_frame(TODAY, flat)}
    spy = _trigger_frame(TODAY, [300 + i * 0.5 for i in range(260)])
    rep = check_triggers(entries, prices, spy=spy, today=TODAY, now=f"{TODAY} 11:00")
    s = rep["summary"]
    assert s["n"] == 6
    assert s["triggered"] == ["TRG"] and s["extended"] == ["EXT"]
    assert s["crossed"] == ["NOV"]
    assert s["no_data"] == ["GONE"] and s["no_pivot"] == ["UNF"]
    assert rep["all_stale"] is False and rep["date"] == TODAY
    assert rep["spy"] is not None and "phase" in rep["spy"]
    assert rep["intraday"] is True                       # fresh bar + mid-session run
    # Item 29: STATUSES is the status vocabulary — every emitted status must come from it
    # (any future status must register itself there; "crossed" did in §6.36).
    assert all(n["status"] in STATUSES for n in rep["names"]), rep["names"]

    # the same report generated after the close is NOT intraday (settled bar)
    rep_eod = check_triggers(entries, prices, spy=spy, today=TODAY, now=f"{TODAY} 16:30")
    assert rep_eod["intraday"] is False

    # all-stale run (holiday): report still builds, flagged, nothing triggered, and a
    # mid-session clock does NOT make a stale-bar report "intraday"
    rep2 = check_triggers([E("TRG", 98.0)], {"TRG": _trigger_frame(TODAY, flat, spike_vol)},
                          today="2026-07-13", now="2026-07-13 12:00")
    assert rep2["all_stale"] is True and rep2["summary"]["triggered"] == []
    assert rep2["intraday"] is False


def test_pullback_trigger():
    """§6.50 pullback status (the low-risk secondary entry after a crossed-without-volume
    breakout, e.g. the 127/208 crossed names in the 8/9 hunt): a prior settled close beat
    the band top (pivot*1.02) on/after date_added, today's close is back within ±2% of
    the frozen pivot, and volume is dry (session-normalized pace <= 0.8x). Stateless —
    derived from the daily frame, no watchlist schema change. Precedence: triggered
    outranks pullback outranks crossed; the raw booleans stay authoritative. The dry
    read uses volume_pace so a morning's mechanically-low raw ratio can't false-fire
    intraday (the same frame that is NOT dry at 11:00 is dry at 16:30)."""
    from src.stock_screener.cockpit.export import make_entry
    from src.stock_screener.cockpit.triggers import (
        STATUSES, check_one, check_triggers, format_report)

    TODAY = "2026-07-10"                                    # a Friday
    PIVOT = 100.0

    def E(t, date_added="2026-07-01"):
        return make_entry(t, PIVOT, date_added=date_added, pivot_source="judged")

    def F(today_close, today_vol, spike_close=103.0, spike_at=55):
        closes = [98.0] * 60
        closes[spike_at] = spike_close                      # the earlier cross
        closes[-1] = today_close
        vols = [1000] * 59 + [today_vol]
        return _trigger_frame(TODAY, closes, vols)

    SETTLED = f"{TODAY} 16:30"

    # (a) crossed at 103 days ago, today 101 (+1%) on 0.7x volume, settled -> PULLBACK.
    # The raw crossed boolean survives (close above pivot, no volume confirm) — status
    # precedence, not the boolean, decides the display.
    r = check_one(E("PBK"), F(101.0, 700), None, today=TODAY, now=SETTLED)
    assert r["status"] == "pullback" and r["pullback"] is True, r
    assert r["crossed_earlier"] is True and r["crossed"] is True
    assert r["triggered"] is False

    # (b) slight undercut (-1%) still counts — Minervini's secondary entry tolerates a
    # small poke below the pivot; crossed is False here (close not above pivot).
    rb = check_one(E("PBK"), F(99.0, 700), None, today=TODAY, now=SETTLED)
    assert rb["status"] == "pullback" and rb["crossed"] is False

    # (c) same shape on 1.5x volume -> triggered outranks, and 1.5x is not dry
    rc = check_one(E("PBK"), F(101.0, 1500), None, today=TODAY, now=SETTLED)
    assert rc["status"] == "triggered" and rc["pullback"] is False

    # (d) ordinary volume (1.0x): in-band but not dry -> stays crossed
    rd = check_one(E("PBK"), F(101.0, 1000), None, today=TODAY, now=SETTLED)
    assert rd["status"] == "crossed" and rd["pullback"] is False

    # (e) prior high never beat the band top (101.5 < 102) -> "retraced" is not real
    re_ = check_one(E("PBK"), F(101.0, 700, spike_close=101.5), None,
                    today=TODAY, now=SETTLED)
    assert re_["status"] == "crossed" and re_["crossed_earlier"] is False

    # (f) the cross predates date_added -> a re-freeze reset the clock -> excluded
    rf = check_one(E("PBK"), F(101.0, 700, spike_at=5), None, today=TODAY, now=SETTLED)
    assert rf["status"] == "crossed" and rf["crossed_earlier"] is False

    # (g) stale bar -> no pullback alert (no live read at all)
    rg = check_one(E("PBK"), F(101.0, 700), None,
                   today="2026-07-13", now="2026-07-13 16:30")
    assert rg["status"] == "stale" and rg["pullback"] is False

    # (h) -3% is a failing base, not a pullback -> watch
    rh = check_one(E("PBK"), F(97.0, 700), None, today=TODAY, now=SETTLED)
    assert rh["status"] == "watch" and rh["pullback"] is False

    # (i) intraday guard: at 11:00 the same 0.7x bar paces ~3x for the time of day ->
    # NOT dry -> no morning misfire; the settled run reads it dry (case a).
    ri = check_one(E("PBK"), F(101.0, 700), None, today=TODAY, now=f"{TODAY} 11:00")
    assert ri["status"] == "crossed" and ri["pullback"] is False

    # report level: summary bucket, registered vocabulary, ASCII explainer
    assert "pullback" in STATUSES
    rep = check_triggers([E("PBK")], {"PBK": F(101.0, 700)},
                         today=TODAY, now=SETTLED)
    assert rep["summary"]["pullback"] == ["PBK"]
    txt = format_report(rep)
    assert "PULLBACK (" in txt and "PBK" in txt
    txt.encode("ascii")                                     # cp1252 log stays safe


def test_template_chain_helper():
    """§6.52's template_chain extraction: byte-identical criteria to the inline
    classify_phase -> full-frame SMA-200 -> validate chain it replaced in scan/triggers/
    positions, and None (not a raise) under classify_phase's 200-row floor."""
    from src.stock_screener.cockpit.scan import template_chain
    from src.stock_screener.minervini_screener.screening import (
        calculate_sma, classify_phase, validate_minervini_trend_template)

    df = _trigger_frame("2026-07-10", [100.0 + i * 0.3 for i in range(260)])
    cp = float(df["Close"].iloc[-1])
    chain = template_chain(df, cp)
    assert chain is not None
    tmpl, phase_info = chain
    ref = validate_minervini_trend_template(cp, classify_phase(df, cp),
                                            calculate_sma(df["Close"], 200))
    assert tmpl["criteria_passed"] == ref["criteria_passed"]
    assert tmpl["criteria_details"] == ref["criteria_details"]
    assert phase_info.get("phase") is not None

    # close defaults to the frame's last close
    dflt = template_chain(df)
    assert dflt is not None and dflt[0]["criteria_passed"] == tmpl["criteria_passed"]

    assert template_chain(df.iloc[-150:]) is None           # short frame -> None
    assert template_chain(None) is None


def test_early_close_calendar_and_gate():
    """Item 17: NYSE half days (July 3 Mon-Thu, day after Thanksgiving, Dec 24 Mon-Thu).
    The session clock shortens to 09:30-13:00 (pace divisor + intraday cutoff), and the
    volume GATE is scaled x(390/210) so a heavy half session can still confirm — while the
    reported ratio stays raw. Normal days are byte-identical to before."""
    from src.stock_screener.cockpit.export import make_entry
    from src.stock_screener.cockpit.triggers import (
        _early_close, _intraday_cutoff_min, _session_len_min, check_one, check_triggers,
        format_report)

    # Calendar truth table (Fri Jul 3 / Dec 24 = OBSERVED full holidays, not half days).
    for d, exp in (("2025-07-03", True), ("2026-07-03", False), ("2026-11-27", True),
                   ("2025-11-28", True), ("2026-12-24", True), ("2027-12-24", False),
                   ("2026-07-10", False), ("2026-11-26", False)):
        assert _early_close(d) is exp, (d, exp)
    assert _session_len_min("2026-11-27") == 210 and _session_len_min("2026-07-10") == 390
    assert _intraday_cutoff_min("2026-11-27") == 13 * 60 + 5
    assert _intraday_cutoff_min("2026-07-10") == 16 * 60 + 5

    HALF = "2026-11-27"                                  # day after Thanksgiving (Friday)
    flat = [100.0] * 60
    even_vol = [1000] * 60                               # today's vol == prior-50 avg -> raw 1.0
    E = make_entry("TRG", 98.0, date_added="2026-11-01", pivot_source="judged")

    # Settled half day: raw 1.0 < 1.5, but scaled 1.0 x 390/210 = 1.86 >= 1.5 -> TRIGGERED.
    r = check_one(E, _trigger_frame(HALF, flat, even_vol), None,
                  today=HALF, now=f"{HALF} 13:30")
    assert r["early_close"] is True
    assert abs(r["volume_ratio_50"] - 1.0) < 1e-9        # displayed ratio stays RAW
    assert abs(r["volume_ratio_50_scaled"] - 1.86) < 0.01
    assert r["volume_confirmed"] is True and r["triggered"] is True, r
    # ...and the same tape on a NORMAL Friday fails the (unscaled) gate — above the pivot
    # without volume that's a §6.36 "crossed", not a trigger.
    r_norm = check_one(E, _trigger_frame("2026-07-10", flat, even_vol), None,
                       today="2026-07-10", now="2026-07-10 16:30")
    assert r_norm["early_close"] is False and r_norm["volume_ratio_50_scaled"] is None
    assert r_norm["volume_confirmed"] is False and r_norm["status"] == "crossed"
    # A genuinely quiet half day (raw 0.7 -> scaled 1.3) still fails the gate.
    low_vol = [1000] * 59 + [700]
    r_low = check_one(E, _trigger_frame(HALF, flat, low_vol), None,
                      today=HALF, now=f"{HALF} 13:30")
    assert r_low["volume_confirmed"] is False

    # Pace divisor uses the 210-min session: at 11:15, 105 min in -> elapsed 0.5 -> pace 2x.
    r_pace = check_one(E, _trigger_frame(HALF, flat, even_vol), None,
                       today=HALF, now=f"{HALF} 11:15")
    assert abs(r_pace["volume_pace"] - 2.0) < 1e-9

    # Report-level: 13:30 on a half day is SETTLED (cutoff 13:05), 12:00 is intraday;
    # the report carries early_close and format_report prints the scaled-gate note.
    prices = {"TRG": _trigger_frame(HALF, flat, even_vol)}
    rep = check_triggers([E], prices, today=HALF, now=f"{HALF} 13:30")
    assert rep["early_close"] is True and rep["intraday"] is False
    rep2 = check_triggers([E], prices, today=HALF, now=f"{HALF} 12:00")
    assert rep2["intraday"] is True
    txt = format_report(rep)
    assert "early close (1:00 pm ET)" in txt and "x1.86" in txt

    # A STALE bar on a half day never scales/fires (Wednesday's bar on Friday's run).
    r_stale = check_one(E, _trigger_frame("2026-11-25", flat, even_vol), None,
                        today=HALF, now=f"{HALF} 13:30")
    assert r_stale["stale"] is True and r_stale["triggered"] is False
    assert r_stale["volume_ratio_50_scaled"] is None


def test_breakout_wrapper_fires_prior_bar_highs():
    """Item 23: the vendored Base/Pivot branches are unreachable (their windowed highs
    include the current bar while callers pass that bar's close). The cockpit wrapper
    takes the highs over the PRIOR bars, so a genuine close above the old base/pivot high
    fires with that level — while the VCP branch's precedence and the phase-1/2 gate are
    preserved, and the vendored function itself stays byte-untouched."""
    import pandas as pd
    from src.stock_screener.minervini_screener.screening import detect_breakout
    from src.stock_screener.cockpit.scan import detect_breakout_prior_high

    def _frame(closes):
        idx = pd.bdate_range(end="2026-07-17", periods=len(closes))
        return pd.DataFrame({"Open": closes, "High": [c * 1.005 for c in closes],
                             "Low": [c * 0.995 for c in closes], "Close": closes,
                             "Volume": [1000] * len(closes)}, index=idx)

    phase2 = {"phase": 2, "sma_50": None}

    # (a) Fresh 60-day closing high: 95-flat base, final close 100. The VENDORED call
    # cannot fire (its base high includes today's 100); the wrapper fires Base Breakout
    # at the prior 60-bar high.
    base = _frame([95.0] * 79 + [100.0])
    vend = detect_breakout(base, 100.0, phase2, None)
    assert not vend.get("is_breakout"), "vendored Base branch must still be unreachable"
    got = got_base = detect_breakout_prior_high(base, 100.0, phase2, None)
    assert got["is_breakout"] and got["breakout_type"] == "Base Breakout"
    assert got["breakout_level"] == 95.0

    # (b) Pivot case: an older 105 high sits inside the 60-bar window, the last 20 bars
    # base at 95, close 100 -> above the 20-day pivot high but below the base high.
    piv = _frame([95.0] * 30 + [105.0] + [95.0] * 48 + [100.0])
    got2 = detect_breakout_prior_high(piv, 100.0, phase2, None)
    assert got2["is_breakout"] and got2["breakout_type"] == "Pivot Breakout"
    assert got2["breakout_level"] == 95.0

    # (c) VCP precedence: a VCP breakout from the vendored call rides through untouched.
    vcp = {"is_vcp": True, "contraction_count": 3,
           "contractions": [{"peak_price": 98.0}]}
    got3 = detect_breakout_prior_high(base, 100.0, phase2, vcp)
    assert got3["is_breakout"] and str(got3["breakout_type"]).startswith("VCP")
    assert got3["breakout_level"] == 98.0

    # (d) The phase gate holds: same fresh high in phase 3 -> no breakout invented.
    got4 = detect_breakout_prior_high(base, 100.0, {"phase": 3}, None)
    assert not got4.get("is_breakout")

    # (e) No breakout when the close is under the prior highs -> vendored result verbatim.
    quiet = _frame([95.0] * 80)
    assert detect_breakout_prior_high(quiet, 95.0, phase2, None) == \
        detect_breakout(quiet, 95.0, phase2, None)

    assert got_base.get("volume_ratio") is not None      # vendored volume fields ride along


def test_untracked_watchlist_names():
    """A watchlist name that no longer passes the 8/8 trend template is UNTRACKED: it
    stays on the list, but its trigger is NOT evaluated (a close above the frozen pivot
    on huge volume must NOT fire), it files under summary['untracked'], outranks 'stale',
    and the console report explains it. A template-passing name is unaffected, and short
    (<200-row) frames fail open (keep evaluating — the existing flat-frame tests pin
    that side)."""
    from src.stock_screener.cockpit.export import make_entry
    from src.stock_screener.cockpit.triggers import check_one, check_triggers, format_report

    TODAY = "2026-07-10"
    E = make_entry("BRKN", 130.0, date_added="2026-07-01", pivot_source="judged")
    # 260-bar DOWNTREND: close under its falling SMAs -> fails the template. Frozen pivot
    # 130 sits BELOW the last close (150) and volume spikes 3x — a would-be trigger.
    down = _trigger_frame(TODAY, [280.0 - i * 0.5 for i in range(260)],
                          [1000] * 259 + [3000])
    r = check_one(E, down, None, today=TODAY, now=f"{TODAY} 16:30")
    assert r["untracked"] is True and r["status"] == "untracked", r
    assert r["triggered"] is False, "an untracked name's trigger must NOT be evaluated"

    # Untracked outranks stale: the same broken-down frame on a later run date.
    r2 = check_one(E, down, None, today="2026-07-13", now="2026-07-13 12:00")
    assert r2["status"] == "untracked" and r2["stale"] is True

    # Control: a template-passing uptrend with the same would-be trigger still fires.
    # (Pivot within 5% of the 177.70 close — further out would file as "extended".)
    up = _trigger_frame(TODAY, [100.0 + i * 0.3 for i in range(260)],
                        [1000] * 259 + [3000])
    E_up = make_entry("UPUP", 176.0, date_added="2026-07-01", pivot_source="judged")
    r3 = check_one(E_up, up, None, today=TODAY, now=f"{TODAY} 16:30")
    assert r3["untracked"] is False and r3["triggered"] is True, r3

    rep = check_triggers([E, E_up], {"BRKN": down, "UPUP": up},
                         today=TODAY, now=f"{TODAY} 16:30")
    assert rep["summary"]["untracked"] == ["BRKN"]
    assert rep["summary"]["triggered"] == ["UPUP"]
    txt = format_report(rep)
    assert "UNTRACKED" in txt and "BRKN" in txt


def test_freeze_missing_pivots():
    """Freeze-on-first-sight: an unfrozen entry with a >=200-row frame gets the scan-chain
    pivot recorded (source 'auto', dated the run day) WITHOUT mutating the input; a short
    frame stays unfrozen (-> no_pivot at check time); a judged entry is never touched."""
    from src.stock_screener.cockpit.export import make_entry
    from src.stock_screener.cockpit.triggers import (check_one, compute_scan_pivot,
                                                     freeze_missing_pivots)

    TODAY = "2026-07-10"
    up = _trigger_frame(TODAY, [100.0 + i * 0.3 for i in range(260)])   # 260-row uptrend
    short = _trigger_frame(TODAY, [100.0] * 60)                          # < 200 rows
    prices = {"UP": up, "SHORT": short}

    expected = compute_scan_pivot(up)
    assert expected and 0 < expected <= float(up["High"].max()) * 1.05
    assert compute_scan_pivot(short) is None                # too short -> None, no raise
    assert compute_scan_pivot(None) is None

    entries = [make_entry("UP"), make_entry("SHORT"),
               make_entry("HELD", 55.5, date_added="2026-07-01", pivot_source="judged")]
    out, frozen = freeze_missing_pivots(entries, prices, today=TODAY)

    assert frozen == ["UP"]
    by = {e["ticker"]: e for e in out}
    assert by["UP"]["judged_pivot"] == round(expected, 2)
    assert by["UP"]["pivot_source"] == "auto" and by["UP"]["date_added"] == TODAY
    assert by["SHORT"]["judged_pivot"] is None              # couldn't compute -> unchanged
    assert by["HELD"] == entries[2]                         # judged entry untouched
    assert entries[0]["judged_pivot"] is None, "input entries must not be mutated"

    # a still-unfrozen entry evaluates to no_pivot (skipped), never a crash
    r = check_one(by["SHORT"], short, None, today=TODAY)
    assert r["status"] == "no_pivot" and r["triggered"] is False

    # REGRESSION (the EBAY 118.98-vs-111.86 bug): an auto-frozen pivot must equal the
    # pivot the APP shows for the same frame — compute_scan_pivot must replicate
    # screen_universe's chain INCLUDING detect_vcp; without the VCP arg, detect_breakout
    # yields no level and the pivot silently falls back to the (higher) 52-week high.
    prices_syn, spy_syn, _ = _synthetic_slice()
    res = screen_universe(list(prices_syn), prices_syn, spy_syn, get_fundamentals=None,
                          cfg=ScanConfig(min_rs=0.0))
    assert len(res.payloads), "no synthetic candidates to check"
    for t in list(res.payloads)[:5]:
        pl = res.payloads[t]
        got = compute_scan_pivot(pl["df"])
        assert got is not None and abs(got - pl["levels"]["pivot"]) < 1e-6, \
            (t, got, pl["levels"]["pivot"])


def test_trigger_report_roundtrip():
    """save_trigger_report/load_latest_trigger_report: dated files, newest parseable wins,
    a corrupt newest file falls back to the older one, missing dir -> None."""
    import tempfile
    from src.stock_screener.cockpit.triggers import (format_report,
                                                     load_latest_trigger_report,
                                                     save_trigger_report)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "triggers"
        assert load_latest_trigger_report(d) is None        # missing dir -> None
        r1 = {"schema": 1, "date": "2026-07-09", "names": [], "summary": {"n": 0}}
        r2 = {"schema": 1, "date": "2026-07-10", "names": [], "summary": {"n": 0}}
        p1 = save_trigger_report(r1, d)
        assert p1.name == "triggers_2026-07-09.json" and p1.exists()
        save_trigger_report(r2, d)
        assert load_latest_trigger_report(d)["date"] == "2026-07-10"
        # corrupt the newest -> the loader falls back to the older parseable file
        (d / "triggers_2026-07-11.json").write_text("{ not json", encoding="utf-8")
        assert load_latest_trigger_report(d)["date"] == "2026-07-10"
        # the console renderer never chokes on a minimal report (ASCII path)
        assert "TRIGGER CHECK" in format_report(r2)


def test_refresh_job_offline():
    """The CLI end-to-end, in-process and offline: patched data feed + temp watchlist/
    report paths. main() returns 0, writes the dated report, and AUTO-FREEZES the
    unfrozen entry's pivot back into watchlist.json; --no-write leaves both untouched."""
    import json as _json
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import cache, refresh_job
    from src.stock_screener.cockpit.export import load_watchlist, save_watchlist

    TODAY = "2026-07-10"
    up = _trigger_frame(TODAY, [100.0 + i * 0.3 for i in range(260)])
    spy = _trigger_frame(TODAY, [300 + i * 0.5 for i in range(260)])

    def fake_prices(tickers, **kw):
        assert kw.get("max_age_days") == refresh_job.REFRESH_MAX_AGE_DAYS, \
            "scheduled runs must pass the freshness floor, not 0.0"
        return {t: (spy if t == "SPY" else up) for t in tickers}

    with tempfile.TemporaryDirectory() as tmp:
        wl = Path(tmp) / "watchlist.json"
        trg = Path(tmp) / "triggers"
        save_watchlist(wl, ["UPUP"])                       # one legacy, unfrozen name
        with patch.object(cache, "WATCHLIST_JSON", wl), \
                patch.object(cache, "TRIGGERS_DIR", trg), \
                patch.object(refresh_job.trade, "position_symbols", lambda: []), \
                patch.object(refresh_job.data_feed, "get_universe",
                             lambda *a, **kw: ["UPUP", "OTHER"]), \
                patch.object(refresh_job.data_feed, "get_many_prices", fake_prices), \
                patch.object(refresh_job.data_feed, "get_fundamentals",
                             lambda t, **kw: {"next_earnings": "2026-07-20"}):
            # --no-write: report printed only, nothing lands on disk
            assert refresh_job.main(["--date", TODAY, "--no-write"]) == 0
            assert not trg.exists() or not list(trg.glob("*.json"))
            assert load_watchlist(wl)[0]["judged_pivot"] is None

            # real run: report written, pivot auto-frozen into the watchlist file
            assert refresh_job.main(["--date", TODAY]) == 0
            rep = _json.loads((trg / f"triggers_{TODAY}.json").read_text(encoding="utf-8"))
            assert rep["date"] == TODAY and rep["summary"]["n"] == 1
            assert rep["summary"]["auto_frozen"] == ["UPUP"]
            assert rep["names"][0]["earnings_in"] == 10
            ent = load_watchlist(wl)[0]
            assert ent["judged_pivot"] and ent["pivot_source"] == "auto"
            assert rep["names"][0]["judged_pivot"] == ent["judged_pivot"]


def test_refresh_job_merges_concurrent_watchlist_edit():
    """Item-11 race, trigger side: an app save landing DURING the trigger's price fetch
    (remove GONE, 📌-freeze UPUP, add NEWB) must survive the auto-freeze write-back —
    disk membership and the user's judged pivot win; the trigger's auto pivot lands only
    on the still-unfrozen KEEP; NEWB stays for next run."""
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import cache, refresh_job
    from src.stock_screener.cockpit.export import load_watchlist, make_entry, save_watchlist

    TODAY = "2026-07-10"
    up = _trigger_frame(TODAY, [100.0 + i * 0.3 for i in range(260)])
    spy = _trigger_frame(TODAY, [300 + i * 0.5 for i in range(260)])

    with tempfile.TemporaryDirectory() as tmp:
        wl = Path(tmp) / "watchlist.json"
        trg = Path(tmp) / "triggers"
        save_watchlist(wl, ["UPUP", "GONE", "KEEP"])       # all unfrozen at trigger load

        def fake_prices(tickers, **kw):
            # Simulate the app writing mid-fetch: the trigger has already loaded the
            # 3-name list above; the file now says otherwise.
            save_watchlist(wl, [make_entry("UPUP", 55.5, date_added=TODAY,
                                           pivot_source="judged"),
                                make_entry("KEEP"), make_entry("NEWB")])
            return {t: (spy if t == "SPY" else up) for t in tickers}

        with patch.object(cache, "WATCHLIST_JSON", wl), \
                patch.object(cache, "TRIGGERS_DIR", trg), \
                patch.object(refresh_job, "refresh_targets", lambda scope: []), \
                patch.object(refresh_job.data_feed, "get_many_prices", fake_prices), \
                patch.object(refresh_job.data_feed, "get_fundamentals", lambda t, **kw: None):
            # Empty targets on purpose: the race under test is "the app saves WHILE the
            # trigger's own price fetch is in flight, after it loaded the list". With a
            # pre-pass running first, fake_prices would fire its simulated write BEFORE
            # build_report ever reads watchlist.json and the race would evaporate.
            assert refresh_job.main(["--date", TODAY]) == 0

        by = {e["ticker"]: e for e in load_watchlist(wl)}
        assert set(by) == {"UPUP", "KEEP", "NEWB"}, "app's membership must win"
        assert by["UPUP"]["judged_pivot"] == 55.5           # user's 📌 beats the auto freeze
        assert by["UPUP"]["pivot_source"] == "judged"
        assert by["KEEP"]["judged_pivot"] and by["KEEP"]["pivot_source"] == "auto"
        assert by["NEWB"]["judged_pivot"] is None           # unseen by this run; next time


def test_no_session_since_calendar():
    """§6.37 settled-cache calendar: no-session-between predicate under a pinned now.
    A cache written after the settled close is current all evening / over the weekend /
    through pre-open; the 16:00-16:05 settle window, any session overlap, and early-close
    (13:05) days behave; a future mtime is trivially current."""
    import pandas as pd
    from src.stock_screener.cockpit.triggers import no_session_since

    def ep(s):                                            # ET wall time -> epoch seconds
        return pd.Timestamp(s, tz="America/New_York").timestamp()

    # Tuesday evening: cache 16:35, now 20:00 -> current (the user-reported case)
    assert no_session_since(ep("2026-08-04 16:35"), now="2026-08-04 20:00") is True
    # inside the 16:00-16:05 settle window -> possibly provisional volume -> NOT current
    assert no_session_since(ep("2026-08-04 16:02"), now="2026-08-04 20:00") is False
    # weekend: Friday 17:00 cache is current Sunday and Monday pre-open, stale mid-session
    assert no_session_since(ep("2026-07-31 17:00"), now="2026-08-02 12:00") is True
    assert no_session_since(ep("2026-07-31 17:00"), now="2026-08-03 08:00") is True
    assert no_session_since(ep("2026-07-31 17:00"), now="2026-08-03 10:00") is False
    # same-day intraday gap -> the 30-min window governs, not this predicate
    assert no_session_since(ep("2026-08-04 14:00"), now="2026-08-04 15:00") is False
    # early close (day after Thanksgiving): cutoff 13:05 — 13:30 cache current, 12:59 not
    assert no_session_since(ep("2026-11-27 13:30"), now="2026-11-27 18:00") is True
    assert no_session_since(ep("2026-11-27 12:59"), now="2026-11-27 18:00") is False
    # future mtime (clock skew) -> nothing newer can exist -> current
    assert no_session_since(ep("2026-08-04 21:00"), now="2026-08-04 20:00") is True


def test_save_trigger_report_atomic():
    """R2-7: save_trigger_report writes tmp + os.replace — a failed write leaves the
    previous day-file byte-intact and no .tmp behind (the app button and the scheduled
    job hit the same file from two processes; an in-place truncate could interleave
    into invalid JSON that the loader silently skips all weekend)."""
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import triggers as trg

    rep = {"date": "2026-08-10", "names": [], "summary": {"n": 0}}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        path = trg.save_trigger_report(rep, dir_path=d)
        assert path.exists() and not list(d.glob("*.tmp"))
        before = path.read_bytes()

        def _boom(obj, f, **kw):
            f.write("torn")
            raise OSError("disk full")

        with patch.object(trg.json, "dump", _boom):
            try:
                trg.save_trigger_report({**rep, "summary": {"n": 9}}, dir_path=d)
            except OSError:
                pass
        assert path.read_bytes() == before, "failed write must leave the old file intact"
        assert not list(d.glob("*.tmp")), "no .tmp litter after a failed write"


def test_refresh_scope_watchlist_unions_held_names():
    """The intraday scope is the watchlist UNION open positions, because those two drift:
    a held name can fall off the watchlist, and the Positions page + sell pillars go blind
    on a position they cannot price. Held names ALREADY watchlisted are not duplicated.
    An unreachable broker degrades to watchlist-only — never an empty refresh."""
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import cache as cachemod
    from src.stock_screener.cockpit import refresh_job
    from src.stock_screener.cockpit.export import save_watchlist

    with tempfile.TemporaryDirectory() as tmp:
        wl = Path(tmp) / "watchlist.json"
        save_watchlist(wl, ["AAA", "BBB"])
        with patch.object(cachemod, "WATCHLIST_JSON", wl):
            with patch.object(refresh_job.trade, "position_symbols",
                              lambda: ["BBB", "ZZZ"]):
                got = refresh_job.refresh_targets(refresh_job.SCOPE_WATCHLIST)
            assert sorted(got) == ["AAA", "BBB", "ZZZ"], got
            assert got.count("BBB") == 1, f"held+watchlisted must not duplicate: {got}"

            def boom():
                raise RuntimeError("no credentials")

            with patch.object(refresh_job.trade, "position_symbols", boom):
                degraded = refresh_job.refresh_targets(refresh_job.SCOPE_WATCHLIST)
            assert sorted(degraded) == ["AAA", "BBB"], \
                f"broker failure must degrade to watchlist-only, got {degraded}"

            with patch.object(refresh_job.data_feed, "get_universe",
                              lambda *a, **kw: ["U1", "U2", "U3"]):
                uni = refresh_job.refresh_targets(refresh_job.SCOPE_UNIVERSE)
            assert uni == ["U1", "U2", "U3"], uni


def test_refresh_max_age_floor_leaves_margin_under_the_cadence():
    """The floor must be strictly positive — 0.0 (the old value) makes _classify_cached's
    freshness branch unreachable, since no existing file is ever <= 0 days old, which is
    why every intraday sweep on 2026-08-26 reported `cached 0` and re-downloaded ~4,100
    names.

    It must also stay under HALF the 30-minute cadence. Age is measured from a file's
    WRITE time, so the gap to the next fire is the interval minus the run's duration minus
    AccuracySec; a floor near the full interval leaves a sub-minute margin, and one slow
    run would make the next scheduled fire serve its own last sweep and skip refreshing."""
    from src.stock_screener.cockpit import refresh_job

    cadence_days = 30.0 / (24.0 * 60.0)
    assert 0.0 < refresh_job.REFRESH_MAX_AGE_DAYS <= cadence_days / 2.0, \
        refresh_job.REFRESH_MAX_AGE_DAYS



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "triggers"))
