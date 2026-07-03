"""Scan orchestration — wire the pure Minervini rule functions into a SEPA funnel.

``screen_universe`` is dependency-injected (you pass it price frames + an optional
fundamentals callable), so it runs deterministically offline in tests. ``run_scan``
is the live convenience wrapper that pulls data via ``data_feed`` first.

Funnel:
  Step 1  validate_minervini_trend_template   -> HARD gate (the candidate list)
  +       trailing-return percentile           -> RS rating (display / optional filter)
  Step 2  fundamentals summary                 -> highlight (badges + pass-count)
  Step 3  detect_vcp (cockpit ZigZag detector) -> hint for the chart (not a gate by default)
  Step 4  detect_breakout + calculate_stop_loss-> advisory entry levels
  Regime  analyze_spy_trend + market_breadth   -> environment banner
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.stock_screener.minervini_screener.screening import (
    analyze_spy_trend,
    calculate_market_breadth,
    calculate_sma,
    classify_phase,
    detect_breakout,
    should_generate_signals,
    validate_minervini_trend_template,
)
from src.stock_screener.minervini_screener.screening import calculate_stop_loss
from .indicators import (relative_measured_volatility,
                         bollinger_bandwidth_percentile, ttm_squeeze)
# Cockpit VCP detector replaces the vendored detect_vcp_pattern (which starves strong
# uptrends → cc=0 for ~84% of candidates on full_us). Same dict schema = drop-in.
from .vcp import detect_vcp


@dataclass
class ScanConfig:
    min_criteria: int = 8          # Step-1 gate: require all 8 trend-template criteria
    min_history_rows: int = 200    # classify_phase needs >= 200 rows
    rs_period: int = 126           # ~6 months, for the RS rating percentile
    min_rs: float = 0.0            # 0 = off; else require RS rating >= this (1-99)
    require_vcp: bool = False       # if True, only keep names with a valid VCP
    min_fundamental_score: int = 0  # keep names with >= this many Step-2 checks passed


@dataclass
class ScanResult:
    candidates: pd.DataFrame                       # one row per passer, sorted
    payloads: Dict[str, dict]                      # ticker -> chart/detail payload
    regime: dict                                   # banner: regime/breadth/should_buy
    n_scanned: int = 0
    n_passed: int = 0
    errors: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
def _rs_ratings(prices: Dict[str, pd.DataFrame], period: int) -> Dict[str, int]:
    """IBD-style RS rating: percentile-rank trailing `period`-day return across the
    whole scanned universe, mapped to 1-99."""
    rets = {}
    for t, df in prices.items():
        c = df["Close"]
        if len(c) > period and c.iloc[-1 - period] > 0:
            rets[t] = float(c.iloc[-1] / c.iloc[-1 - period] - 1.0)
    if not rets:
        return {}
    pr = pd.Series(rets).rank(pct=True) * 99.0
    return {t: int(round(v)) for t, v in pr.items()}


def _step2_summary(f: Optional[dict]) -> dict:
    """SEPA Step 2 as a set of pass/fail checks + a 0-4 pass-count. YoY preferred,
    QoQ used as a fallback when yfinance exposes too few quarters for YoY."""
    if not f:
        return {"score": 0, "checks": {}, "available": False}
    rev = f.get("revenue_yoy")
    rev = rev if rev is not None else f.get("revenue_qoq")
    eps = f.get("eps_yoy")
    eps = eps if eps is not None else f.get("eps_qoq")
    checks = {
        "revenue_growth": rev is not None and rev >= 20.0,
        "eps_growth": eps is not None and eps >= 20.0,
        "eps_accelerating": (f.get("eps_yoy") is not None
                             and f.get("eps_yoy_prev") is not None
                             and f["eps_yoy"] >= f["eps_yoy_prev"]),
        "margin_expanding": f.get("margin_trend") is not None and f["margin_trend"] >= 0.0,
    }
    return {"score": int(sum(checks.values())), "checks": checks, "available": True}


def _entry_levels(cp: float, breakout: dict, stop: Optional[float],
                  phase_info: dict) -> dict:
    """SEPA Step 4 advisory levels. Pivot = the breakout/base level if detected, else
    the 52-week high (the line a breakout would clear)."""
    pivot = breakout.get("breakout_level")
    if not pivot or pivot <= 0:
        pivot = phase_info.get("week_52_high") or cp
    stop_price = stop if (stop and stop > 0 and stop < pivot) else pivot * 0.925
    pct_to_pivot = ((pivot - cp) / cp * 100.0) if cp else None
    return {
        "pivot": float(pivot),
        "buy_zone": (float(pivot), float(pivot) * 1.05),   # no chasing > +5%
        "stop": float(stop_price),                          # ~7-8% below pivot
        "target": float(pivot) * 1.225,                     # ~20-25% objective
        "breakout_today": bool(breakout.get("is_breakout")),
        "volume_ratio": float(breakout.get("volume_ratio", 1.0) or 1.0),
        "volume_confirmed": bool(breakout.get("volume_confirmed", False)),  # vol >= 1.5x avg
        "pct_to_pivot": pct_to_pivot,
    }


def screen_universe(tickers: List[str], prices: Dict[str, pd.DataFrame],
                    spy: pd.DataFrame,
                    get_fundamentals: Optional[Callable[[str], Optional[dict]]] = None,
                    cfg: Optional[ScanConfig] = None) -> ScanResult:
    """Run the SEPA funnel over already-fetched price frames (deterministic/offline).

    ``prices``: {ticker -> daily OHLCV}. ``spy``: SPY daily OHLCV.
    ``get_fundamentals``: optional callable run only on Step-1 passers (cheap).
    """
    cfg = cfg or ScanConfig()
    errors: List[str] = []
    spy_cp = float(spy["Close"].iloc[-1])
    spy_analysis = analyze_spy_trend(spy, spy_cp)
    rs_rating = _rs_ratings(prices, cfg.rs_period)

    phase_results: List[dict] = []
    rows: List[dict] = []
    payloads: Dict[str, dict] = {}

    for t in tickers:
        df = prices.get(t)
        if df is None or len(df) < cfg.min_history_rows:
            continue
        cp = float(df["Close"].iloc[-1])
        if not np.isfinite(cp):
            continue
        try:
            phase_info = classify_phase(df, cp)
            phase_results.append({"ticker": t, "phase": phase_info.get("phase", 0)})

            sma200 = calculate_sma(df["Close"], 200)
            tmpl = validate_minervini_trend_template(cp, phase_info, sma200)
            if tmpl.get("criteria_passed", 0) < cfg.min_criteria:
                continue

            rsr = rs_rating.get(t)
            if cfg.min_rs and (rsr is None or rsr < cfg.min_rs):
                continue

            vcp = detect_vcp(df, cp, phase_info)
            if cfg.require_vcp and not vcp.get("is_vcp"):
                continue

            breakout = detect_breakout(df, cp, phase_info, vcp)
            stop = calculate_stop_loss(df, cp, phase_info, phase_info.get("phase", 2))

            fund = get_fundamentals(t) if get_fundamentals else None
            s2 = _step2_summary(fund)
            if s2["score"] < cfg.min_fundamental_score:
                continue

            levels = _entry_levels(cp, breakout, stop, phase_info)
            # RMV (Relative Measured Volatility): advisory base-tightness read for Step 4.
            # Does NOT feed the pivot/stop/target math — the cockpit shows hints, it doesn't
            # move the levels for you.
            rmv_series = relative_measured_volatility(df).dropna()
            levels["rmv"] = float(rmv_series.iloc[-1]) if len(rmv_series) else None
            # BBWP + TTM squeeze: the Bollinger-side volatility read, a cross-check on RMV
            # (also advisory). BBWP low = a Bollinger squeeze; squeeze True = bands inside
            # the Keltner channel (a coiled spring).
            bbwp_series = bollinger_bandwidth_percentile(df).dropna()
            levels["bbwp"] = float(bbwp_series.iloc[-1]) if len(bbwp_series) else None
            sq = ttm_squeeze(df)
            levels["squeeze"] = bool(sq.iloc[-1]) if len(sq) else False
            # "squeeze fired": the base was coiled within the last ~6 bars but is expanding
            # (off) now — the volatility-EXPANSION side of a breakout (vs. the tight base).
            prior = sq.iloc[-6:-1] if len(sq) >= 2 else sq.iloc[:0]
            levels["squeeze_released"] = bool(len(prior) and bool(prior.any())
                                              and not levels["squeeze"])
            rows.append({
                "ticker": t,
                "price": round(cp, 2),
                "rs": rsr,
                "criteria": tmpl.get("criteria_passed"),
                "fund_score": s2["score"],
                "rev_yoy": _fmt(fund and fund.get("revenue_yoy")),
                "eps_yoy": _fmt(fund and fund.get("eps_yoy")),
                "op_margin": _fmt(fund and fund.get("operating_margin")),
                "vcp": bool(vcp.get("is_vcp")),
                "num_contractions": int(vcp.get("contraction_count", 0) or 0),
                "vcp_quality": round(float(vcp.get("vcp_quality", 0) or 0), 0),
                "breakout_today": levels["breakout_today"],
                "vol_confirmed": levels["volume_confirmed"],
                "pct_to_pivot": _fmt(levels["pct_to_pivot"]),
                "pivot": round(levels["pivot"], 2),
                "stop": round(levels["stop"], 2),
                "target": round(levels["target"], 2),
            })
            payloads[t] = {
                "df": df, "phase_info": phase_info, "vcp": vcp,
                "breakout": breakout, "levels": levels, "fundamentals": fund,
                "step2": s2, "rs": rsr, "template": tmpl,
            }
        except Exception as e:                                  # never let one name kill the scan
            errors.append(f"{t}: {e}")
            continue

    breadth = calculate_market_breadth(phase_results)
    sig = should_generate_signals(spy_analysis, breadth)
    regime = {
        "regime": sig.get("regime"),
        "should_generate_buys": sig.get("should_generate_buys"),
        "phase2_pct": breadth.get("phase_2_pct", 0.0),
        "breadth_quality": breadth.get("breadth_quality"),
        "spy_phase": spy_analysis.get("phase"),
        "spy_trend": spy_analysis.get("trend"),
        "reasons": sig.get("reasons", []),
    }

    if rows:
        cand = pd.DataFrame(rows).sort_values(
            ["fund_score", "rs"], ascending=[False, False],
            na_position="last").reset_index(drop=True)
    else:
        cand = pd.DataFrame()
    return ScanResult(candidates=cand, payloads=payloads, regime=regime,
                      n_scanned=len(phase_results), n_passed=len(rows), errors=errors)


def _fmt(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), 1)


# --------------------------------------------------------------------------- #
def run_scan(universe: str = "sp500", cfg: Optional[ScanConfig] = None,
             max_workers: int = 6, force: bool = False,
             progress: Optional[Callable[[int, int, str], None]] = None) -> ScanResult:
    """Live wrapper: fetch via data_feed, then screen. Fundamentals are fetched lazily
    inside the funnel (only for Step-1 passers)."""
    from . import data_feed
    tickers = data_feed.get_universe(universe, force=force)
    spy = data_feed.get_spy(force=force)
    if spy is None or len(spy) < 200:
        raise RuntimeError("Could not fetch SPY benchmark data (needed for RS/regime).")
    prices = data_feed.get_many_prices(tickers, max_workers=max_workers,
                                       force=force, progress=progress)
    return screen_universe(list(prices.keys()), prices, spy,
                           get_fundamentals=data_feed.get_fundamentals, cfg=cfg)
