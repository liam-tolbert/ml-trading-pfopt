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
    return {
        "df": df,
        "levels": {"pivot": pivot, "stop": pivot * 0.97, "breakout_today": breakout},
        "vcp": {"contractions": [
            {"peak_date": idx[-40], "trough_date": idx[-35], "peak_price": close * 1.05,
             "trough_price": close * 0.95, "drawdown_pct": 9.5, "volume_ratio": 0.8,
             "duration_days": 5, "number": 1}]},
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
        for frag in ("Weekend Hunt", "In the buy zone", "Approaching pivot",
                     "Volume-confirmed", "Step-2 fundamentals", "Full review",
                     "AAA", "EEE"):
            ok(f"report contains {frag!r}", frag in html)


if __name__ == "__main__":
    test_rs_floor_and_tier()
    test_bucket_boundaries()
    test_diagnostics_and_gates()
    test_verdict_bookkeeping()
    test_report_builds()
    print(f"\n{PASSED} hunt assertions passed.")
