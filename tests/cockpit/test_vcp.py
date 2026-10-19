"""Cockpit tests — the cockpit VCP detector and its volatility indicators (RMV, BBWP, squeeze).

Runs standalone (`python tests/cockpit/test_vcp.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_vcp_detects_tightening_bases():
    """The cockpit detector flags genuine widest-first tightening bases — including the
    shallow ones the vendored detector missed (cc=0)."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    # Textbook VCP: rise, then pullbacks ~16→11→7→5% into a tight base near the high.
    # thr pinned to 0.04 so the adaptive default doesn't shift the deterministic pivot count.
    df = _lin_series([(100, 150), (10, 126), (8, 148), (10, 131.7), (8, 146),
                      (10, 135.8), (8, 144), (10, 136.8), (8, 143)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {}, thr=0.04)
    assert v["is_vcp"] is True, v["pattern_details"]
    assert v["contraction_count"] == 4, v["contraction_count"]
    depths = [c["drawdown_pct"] for c in v["contractions"]]
    assert depths == sorted(depths, reverse=True) and depths[0] > 12 and depths[-1] < 8, depths
    assert v["contractions"][-1]["trough_date"] == max(c["trough_date"] for c in v["contractions"])

    # Shallow tight base (~8→5%) that the vendored detector reported as cc=0. The final leg is
    # longer/calmer than the advance into it, so RMV bottoms in the base (the strict vol gate).
    df2 = _lin_series([(60, 130), (8, 119.6), (8, 128), (12, 121.6), (16, 127)])
    v2 = detect_vcp(df2, float(df2["Close"].iloc[-1]), {}, thr=0.04)
    assert v2["is_vcp"] is True and v2["contraction_count"] >= 2, v2["pattern_details"]


def test_vcp_rejects_noise_and_nobase():
    """Choppy names far from their high and straight-up no-base movers are NOT VCPs, and
    the contraction count stays bounded (no 22-swing noise tail)."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    # Chop that fades ~30% off its high -> not near the high -> not a VCP.
    choppy = _lin_series([(40, 150), (8, 138), (6, 146), (8, 128), (6, 138),
                          (8, 118), (6, 128), (8, 108), (6, 116), (8, 105)])
    v = detect_vcp(choppy, float(choppy["Close"].iloc[-1]), {})
    assert v["is_vcp"] is False and v["contraction_count"] <= 6, v["contraction_count"]

    # Straight-up, no pullback -> no completed peak->trough -> cc 0, not a VCP.
    up = _lin_series([(150, 250)])
    v2 = detect_vcp(up, float(up["Close"].iloc[-1]), {})
    assert v2["is_vcp"] is False and v2["contraction_count"] <= 1, v2["contraction_count"]


def test_vcp_base_does_not_span_a_breakout():
    """The base must be a single consolidation under a flat top: contractions from an OLD
    base (before a big advance) must not be stitched into the current one (the DVA bug)."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    # Base A near ~100 (8%, 5.6%), a +22% advance to ~120, then base B near ~120 (8%, 5%).
    df = _lin_series([(40, 100), (8, 92), (8, 99), (8, 93.5), (8, 98),
                      (10, 120), (8, 110.4), (8, 118), (8, 112), (8, 117)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {}, thr=0.04)
    # Only base B is the current base — base A's ~100 peaks are excluded by the flat-top rule.
    assert v["contraction_count"] == 2, v["contraction_count"]
    assert all(c["peak_price"] > 108 for c in v["contractions"]), \
        [round(c["peak_price"]) for c in v["contractions"]]
    assert v["is_vcp"] is True, v["pattern_details"]


def test_vcp_base_does_not_span_a_collapse():
    """A > MAX_DEPTH_PCT collapse (dropped by the depth filter) that sits BETWEEN two shallow
    legs is a base boundary — the walk-back must not stitch the legs on either side of it into
    one 'base' (their peaks are ~equal, so the flat-top rule alone can't catch it). Pre-fix this
    reported 2 contractions / a spannable base straddling a 40% crash."""
    from src.stock_screener.cockpit.vcp import detect_vcp, MAX_DEPTH_PCT

    # rise → C1 (100→90, 10%) → recover → CRASH (100→60, 40%, dropped) → recover →
    # C3 (100→92, 8%) → up to 99 (near the high). C1 and C3 peaks are both ~100 (flat top).
    df = _lin_series([(40, 100), (8, 90), (8, 100), (8, 60), (12, 100), (8, 92), (8, 99)],
                     start=70.0)
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {}, thr=0.04)
    # the base stops at the crash: only the recent shallow leg survives, so it's not a 2-leg VCP
    assert v["contraction_count"] == 1, v["contraction_count"]
    assert v["is_vcp"] is False, v["pattern_details"]
    # and the > 35% collapse is never reported as part of the base
    assert all(c["drawdown_pct"] <= MAX_DEPTH_PCT for c in v["contractions"]), \
        [c["drawdown_pct"] for c in v["contractions"]]


def test_bbwp_and_squeeze_indicators():
    """BBWP stays in 0-100 and reads low (a squeeze) as a wide base coils tight; TTM squeeze
    returns a bool Series that is on in the tight tail."""
    from src.stock_screener.cockpit.indicators import (
        bollinger_bandwidth_percentile, ttm_squeeze)

    df = _range_frame()
    bbwp = bollinger_bandwidth_percentile(df).dropna()
    assert len(bbwp), "BBWP produced no values"
    assert float(bbwp.min()) >= 0.0 and float(bbwp.max()) <= 100.0, (bbwp.min(), bbwp.max())
    assert float(bbwp.iloc[-1]) < 40.0, float(bbwp.iloc[-1])   # coiled tail = low percentile

    sq = ttm_squeeze(df)
    assert sq.dtype == bool and len(sq) == len(df)
    assert bool(sq.iloc[-1]) is True, "coiled tail should register a TTM squeeze"


def test_bbwp_last_matches_series():
    """Item 21: the scalar BBWP fast path equals the full rolling series' FINAL row for
    every history length; sub-warm-up lengths yield None on both paths (band needs 20
    bars, the percentile needs 20 band values -> first output at row 39)."""
    import numpy as np
    import pandas as pd
    from src.stock_screener.cockpit.indicators import (
        bollinger_bandwidth_percentile, bollinger_bandwidth_percentile_last)

    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 1, 400))
    full = pd.DataFrame({"Close": closes},
                        index=pd.bdate_range(end="2026-07-17", periods=400))

    for ln in (400, 165, 60, 45):                        # > lookback, partial, minimal
        df = full.tail(ln)
        last = float(bollinger_bandwidth_percentile(df).iloc[-1])
        scalar = bollinger_bandwidth_percentile_last(df)
        assert scalar is not None and abs(scalar - last) < 1e-12, (ln, scalar, last)
    for ln in (30, 38):                                  # series all-NaN band -> both None
        assert bollinger_bandwidth_percentile(full.tail(ln)).dropna().empty
        assert bollinger_bandwidth_percentile_last(full.tail(ln)) is None
    assert bollinger_bandwidth_percentile_last(full.tail(10)) is None     # < period


def test_scan_rmv_display_reuses_vcp():
    """Item 22: the scan's Step-4 RMV reuses the value detect_vcp already computed (the
    last-bar RMV window is ~60 trailing bars, a subset of the detector's 325-bar base, so
    the reads are equal) — and NEVER surfaces the _empty() sentinel rmv=100.0: a
    dead-tape/short frame gets a real read computed from the full frame."""
    import numpy as np
    import pandas as pd
    from src.stock_screener.cockpit.indicators import relative_measured_volatility
    from src.stock_screener.cockpit.scan import _rmv_display

    # Real detection result -> reuse the detector's (1-dp rounded) value; df untouched.
    assert _rmv_display(None, {"zz_threshold": 0.04, "rmv": 31.4}) == 31.4

    rng = np.random.default_rng(3)
    closes = 100 + np.cumsum(rng.normal(0, 1, 400))
    df = pd.DataFrame({"High": closes * 1.01, "Low": closes * 0.99, "Close": closes},
                      index=pd.bdate_range(end="2026-07-17", periods=400))
    real = float(relative_measured_volatility(df).dropna().iloc[-1])
    got = _rmv_display(df, {"zz_threshold": None, "rmv": 100.0})   # the _empty sentinel
    assert got is not None and abs(got - real) < 1e-12
    # The window argument that makes the reuse safe: full-frame RMV last value == the
    # 325-bar base's last value (rolling windows only look back ~60 bars).
    base_last = float(relative_measured_volatility(df.tail(325)).dropna().iloc[-1])
    assert abs(real - base_last) < 1e-9


def test_rmv_gate_vetoes_below_pivot_only():
    """RMV semantics (benchmarked): while price is still BELOW the pivot the base should be
    quiet, so a loud tape is vetoed. AT/ABOVE the pivot a breakout IS a burst of movement —
    RMV reads high at exactly the moment a setup completes (the SMBC false-negative class) —
    so it must NOT veto there; structure alone decides."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    base_legs = [(100, 150), (10, 126), (8, 148), (10, 131.7), (8, 146),
                 (10, 135.8), (8, 144), (10, 136.8)]
    calm = _lin_series(base_legs + [(8, 143)])                    # quiet drift below pivot
    loud = _lin_series(base_legs + [(1, 143), (1, 137.5), (1, 143), (1, 137.5),
                                    (1, 143), (1, 137.5), (1, 142), (1, 138)])
    hot = _lin_series(base_legs + [(2, 150)])                     # breakout thrust AT pivot

    vc = detect_vcp(calm, float(calm["Close"].iloc[-1]), {}, thr=0.04)
    vl = detect_vcp(loud, float(loud["Close"].iloc[-1]), {}, thr=0.04)
    vh = detect_vcp(hot, float(hot["Close"].iloc[-1]), {}, thr=0.04)

    assert vc["is_vcp"] is True, vc["pattern_details"]
    # below the pivot with a whipsawing tape -> RMV (or the churned structure) must reject
    assert vl["is_vcp"] is False, (vl["rmv"], vl["pattern_details"])
    assert vl["rmv"] > vc["rmv"], (vl["rmv"], vc["rmv"])
    # at the pivot mid-breakout the RMV burst must NOT veto a valid structure
    assert vh["is_vcp"] is True, (vh["rmv"], vh["pattern_details"])
    assert vh["rmv"] > vc["rmv"], (vh["rmv"], vc["rmv"])
    assert vh["tier"] == "A", (vh["tier"], vh["pattern_details"])


def test_vcp_sees_quiet_taper_after_loud_history():
    """SMBC shape: a formerly-loud stock (±20% swings) whose base tapers 9→7→5.5→4.5%.
    A swing filter sized to the loud past cannot see the tight legs; the detector must
    find them (tier A, default mode)."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 150), (15, 115), (15, 150), (15, 122), (15, 152),
                       (10, 141.5), (8, 152), (8, 145.2), (8, 152.5),
                       (8, 147.2), (8, 153), (8, 149.2), (6, 156)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["is_vcp"] is True, (v["tier"], v["pattern_details"])
    assert v["tier"] == "A", (v["tier"], v["pattern_details"])
    assert v["contraction_count"] >= 3, v["pattern_details"]


def test_vcp_finds_tight_final_leg_after_wide_start():
    """VRA shape: wide early pullbacks (~20%) ending in one tight final leg, price still
    below the pivot. A single history-wide threshold missed the final leg entirely."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 150), (12, 120), (12, 148), (10, 133), (10, 147),
                       (8, 137), (10, 145)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["is_vcp"] is True, (v["tier"], v["pattern_details"])
    assert v["tier"] == "A", (v["tier"], v["pattern_details"])
    depths = [c["drawdown_pct"] for c in v["contractions"]]
    assert depths[-1] < 10.0, depths


def test_vcp_two_day_spike_is_not_a_base():
    """EQ shape: a straight run-up whose only 'pullback' is a violent 1-bar plunge and
    rebound. A 1-bar leg is a junk anchor, not a base — tier C, never a VCP."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(150, 250), (1, 225), (1, 248)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["is_vcp"] is False, v["pattern_details"]
    assert v["tier"] == "C", (v["tier"], v["pattern_details"])


def test_vcp_deal_pinned_stock_is_dead_tape():
    """Deal-stock shape: months of near-zero movement (an acquisition-arb zombie) cannot
    be a live setup — tier C with the dead-tape reason recorded."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 90), (20, 100)], band=0.01)
    import pandas as pd
    flat_idx = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=150)
    flat = pd.DataFrame({"Open": 100.2, "High": 100.25, "Low": 100.15, "Close": 100.2,
                         "Volume": 50_000}, index=flat_idx)
    df = pd.concat([df, flat])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["tier"] == "C", (v["tier"], v["pattern_details"])
    assert v["median_tr_pct"] is not None and v["median_tr_pct"] < 1.0, v


def test_vcp_extended_breakout_is_watch_not_review():
    """Extended-breakout shape: a textbook base whose breakout already ran ~15% past the
    pivot is spent — a valid pattern, but tier B (watch), never tier A."""
    from src.stock_screener.cockpit.vcp import detect_vcp

    df = _ohlc_series([(60, 150), (10, 138), (8, 149), (8, 141.8), (8, 150),
                       (8, 145.5), (15, 175)])
    v = detect_vcp(df, float(df["Close"].iloc[-1]), {})
    assert v["tier"] == "B", (v["tier"], v["pattern_details"])
    # extended = a VALID pattern demoted to watch (tier_reason text removed 2026-07-13; the
    # is_vcp True + tier B pair is what distinguishes "extended" from "still forming")
    assert v["is_vcp"] is True, (v["tier"], v["pattern_details"])


_BENCH: dict = {}


def _bench_results() -> dict:
    """``{ticker: (df, detect_vcp result)}`` over the 200 benchmark fixtures, computed once
    per process for the benchmark and the Step-3 cross-tab."""
    if not _BENCH:
        import time
        import pandas as pd
        from vcp_labels import LABELS, fixture_filename
        from src.stock_screener.cockpit.vcp import detect_vcp

        fdir = ROOT / "tests" / "fixtures" / "vcp_bench"
        assert fdir.exists(), "benchmark fixtures missing — were tests/fixtures committed?"
        frames = {t: pd.read_parquet(fdir / fixture_filename(t)) for t in LABELS}
        t0 = time.perf_counter()
        for t, df in frames.items():
            _BENCH[t] = (df, detect_vcp(df, float(df["Close"].iloc[-1]), {}))
        ms = (time.perf_counter() - t0) * 1000.0 / len(frames)
        print(f"    detect_vcp: {ms:.1f} ms per chart over {len(frames)} charts")
    return _BENCH


def test_vcp_benchmark_report():
    """The 200 charts in tests/vcp_labels.py, as a printed report, not a gate. The user
    retired their labels as the check of the detector (HANDOFF_HISTORY §6.110); the user's
    own judgments of real hunt charts replace them (HANDOFF §12). What is asserted is the
    code's own contract: every result, tier C included, carries the typical day the stop
    is judged by (§6.75), and only a dead tape, no pullbacks or a stale base is tier C."""
    from collections import Counter
    from vcp_labels import LABELS

    tiers = {}
    for t in LABELS:
        df, r = _bench_results()[t]
        tiers[t] = r["tier"]
        assert r.get("median_tr_pct") is not None, (t, r["tier"], r["pattern_details"])
        if r["tier"] == "C":
            why = r["pattern_details"]
            assert (why.startswith(("Barely moving", "No pullbacks", "Insufficient"))
                    or "weeks old" in why), (t, why)
    tab = Counter((LABELS[t]["label"], tiers[t]) for t in LABELS)
    print("    labelled charts by tier (report only): " + "  ".join(
        f"{tier}: YES {tab[('YES', tier)]} NO {tab[('NO', tier)]}" for tier in "ABC"))


def test_step3_reads_on_benchmark():
    """The dry-up read over the 200 charts, printed for the ledger. Its pre-registered
    named-chart check (§6.91) was made against the earlier detector's legs and those
    labels; both are retired (§6.110), so nothing here is asserted. The read itself is
    tested on synthetic frames in test_advisories."""
    from collections import Counter
    from vcp_labels import LABELS
    from src.stock_screener.cockpit import advisories

    bench = _bench_results()

    def kind(d):
        if d is None:
            return "n/a"
        below, quiet = d["avg_ratio"] < 1.0, d["quiet_days"] >= 1
        return "dry" if below and quiet else "partial" if below or quiet else "none"

    dry = {t: kind(advisories.volume_dryup(df, r["contractions"]))
           for t, (df, r) in bench.items()}
    tab = Counter((LABELS[t]["label"], d) for t, d in dry.items())
    print("    step3 dry-up: " + "  ".join(
        f"{lab} " + " ".join(f"{v}={tab[(lab, v)]}" for v in ("dry", "partial", "none", "n/a"))
        for lab in ("YES", "NO")))


def test_zigzag_fast_parity():
    """_zigzag_pivots (plain-float hot path) returns EXACTLY the reference
    implementation's pivots — same indices, prices, kinds — across all 200 benchmark
    fixtures at a threshold grid around the one detect_vcp uses, plus seeded
    random walks. This is the ship-gate for the vcp.py fast path."""
    import numpy as np
    import pandas as pd
    from vcp_labels import LABELS, fixture_filename

    from src.stock_screener.cockpit.vcp import _zigzag_pivots, _zigzag_pivots_ref

    thresholds = (0.02, 0.03, 0.05, 0.08, 0.12)
    fdir = ROOT / "tests" / "fixtures" / "vcp_bench"
    checked = 0
    for t in LABELS:
        df = pd.read_parquet(fdir / fixture_filename(t))
        high = df["High"].to_numpy()[-325:]
        low = df["Low"].to_numpy()[-325:]
        for thr in thresholds:
            assert _zigzag_pivots(high, low, thr) == _zigzag_pivots_ref(high, low, thr), \
                (t, thr)
            checked += 1
    rng = np.random.default_rng(0)
    for _ in range(25):
        mid = 100.0 + np.cumsum(rng.normal(0, 1.5, 325))
        high = mid + rng.uniform(0.0, 2.0, 325)
        low = mid - rng.uniform(0.0, 2.0, 325)
        for thr in thresholds:
            assert _zigzag_pivots(high, low, thr) == _zigzag_pivots_ref(high, low, thr)
            checked += 1
    assert checked >= 1000, checked



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "vcp"))
