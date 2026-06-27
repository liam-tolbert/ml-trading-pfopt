"""Market-regime gate — wraps the vendored benchmark functions.

Uses the cheap *vectorized* phase for breadth (a coarse 15% gate, the one place we
spend approximation) and the vendored ``analyze_spy_trend`` on the SPY <=t slice.
"""
from __future__ import annotations

import pandas as pd

from src.stock_screener.minervini_screener.screening import (
    analyze_spy_trend,
    calculate_market_breadth,
    should_generate_signals,
)


def assess(t, cache, members_t, spy_slice, cfg):
    """Return (gate_dict, spy_analysis, breadth). gate has should_generate_buys,
    should_generate_sells, risk_off."""
    if spy_slice is None or len(spy_slice) < cfg.min_history_rows:
        return ({"should_generate_buys": False, "should_generate_sells": True,
                 "risk_off": False}, {"phase": 0}, {})
    cp = float(spy_slice["Close"].iloc[-1])
    spy_an = analyze_spy_trend(spy_slice, cp)
    phase_results = [{"phase": cache.vectorized_phase(p, t)} for p in members_t]
    breadth = calculate_market_breadth(phase_results)
    gate = should_generate_signals(spy_an, breadth, min_phase2_pct=cfg.min_phase2_pct)
    gate["risk_off"] = (spy_an.get("phase") == 4)
    return gate, spy_an, breadth
