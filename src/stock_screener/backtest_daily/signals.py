"""The ONLY module that builds <=t slices and calls the vendored rule functions.

Centralizing every vendored call here keeps the leak contract auditable: each
function takes ``(permno, t, cache, ...)``, gets the leak-safe slice from the cache,
and treats its last row as "now" — exactly what the vendored functions assume.
"""
from __future__ import annotations

import pandas as pd

from src.stock_screener.minervini_screener.screening import (
    classify_phase,
    detect_vcp_pattern,
    calculate_relative_strength,
    score_buy_signal,
    score_sell_signal,
)


def evaluate_buy(permno, t, cache, spy_slice, fundamentals, cfg):
    """Return the vendored buy-signal dict if this name is a buy at t, else None."""
    df = cache.ohlcv_upto(permno, t)
    if df is None or len(df) < cfg.min_history_rows:
        return None
    cp = float(df["Close"].iloc[-1])
    if pd.isna(cp):
        return None
    if cfg.min_price and cache.raw_price(permno, t) < cfg.min_price:   # Minervini price floor
        return None
    phase_info = classify_phase(df, cp)
    if phase_info.get("phase") != 2:                 # cheap early-out (scorer re-checks)
        return None
    vcp = detect_vcp_pattern(df, cp, phase_info)
    rs = calculate_relative_strength(df["Close"], spy_slice["Close"])
    sig = score_buy_signal(str(int(permno)), df, cp, phase_info, rs,
                           fundamentals=fundamentals, vcp_data=vcp)
    if not sig.get("is_buy"):
        return None
    sig["permno"] = int(permno)
    sig["phase"] = 2
    return sig


def evaluate_sell(permno, t, cache, spy_slice, prev_phase, cfg):
    """Return the vendored sell-signal dict for a held name (phase 3/4 fires)."""
    df = cache.ohlcv_upto(permno, t)
    if df is None or len(df) < cfg.min_history_rows:
        return None
    cp = float(df["Close"].iloc[-1])
    if pd.isna(cp):
        return None
    phase_info = classify_phase(df, cp)
    rs = calculate_relative_strength(df["Close"], spy_slice["Close"])
    sig = score_sell_signal(str(int(permno)), df, cp, phase_info, rs,
                            previous_phase=prev_phase)
    sig["permno"] = int(permno)
    sig.setdefault("phase", phase_info.get("phase"))
    return sig
