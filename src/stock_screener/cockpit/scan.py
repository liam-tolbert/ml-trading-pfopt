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
# Cockpit VCP detector: the vendored detect_vcp_pattern starves strong uptrends
# (cc=0 for ~84% of candidates on full_us). Same dict schema = drop-in.
from .vcp import detect_vcp

# Minervini's stop is measured from the pivot: 7-8% ideal, 10% hard max. Floors the advisory
# stop this far below the pivot so a price-anchored engine stop can't breach max-loss.
MAX_STOP_FROM_PIVOT = 0.10


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
    """IBD-style RS rating, 1-99: percentile-rank a WEIGHTED multi-horizon return blend
    across the whole scanned universe.

    Horizons derive from ``period`` (the ~6-mo leg, 126 trading days by default): 3-mo =
    period/2 at DOUBLE weight (IBD's recent-strength emphasis), then 6-mo, 9-mo, 12-mo at
    single weight — the classic ``2·r3 + r6 + r9 + r12`` blend. The weighted MEAN (not the
    raw sum) is ranked so young listings compete on the legs they have instead of dropping
    out. Inclusion still requires >= ``period`` bars; for full-history names the mean is the
    IBD sum / 5, ranking identically."""
    horizons = ((max(1, period // 2), 2.0), (period, 1.0),
                (period * 3 // 2, 1.0), (period * 2, 1.0))
    scores = {}
    for t, df in prices.items():
        c = df["Close"]
        if len(c) <= period or c.iloc[-1 - period] <= 0:
            continue                                     # the 6-mo leg is mandatory
        num = den = 0.0
        for look, w in horizons:
            if len(c) > look and c.iloc[-1 - look] > 0:
                num += w * float(c.iloc[-1] / c.iloc[-1 - look] - 1.0)
                den += w
        if den > 0:
            scores[t] = num / den
    if not scores:
        return {}
    pr = pd.Series(scores).rank(pct=True) * 99.0
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


def _days_to_earnings(f: Optional[dict],
                      today: Optional[pd.Timestamp] = None) -> Optional[int]:
    """Calendar days until the next scheduled earnings report, from the fundamentals
    dict's ``next_earnings`` ('YYYY-MM-DD'). Negative = the (cached) date has passed,
    i.e. the company just reported; None = no date known. ``today`` is overridable so
    tests stay deterministic."""
    d = (f or {}).get("next_earnings")
    if not d:
        return None
    try:
        today = (today if today is not None else pd.Timestamp.today()).normalize()
        return int((pd.Timestamp(d).normalize() - today).days)
    except Exception:
        return None


def _entry_levels(cp: float, breakout: dict, stop: Optional[float],
                  phase_info: dict) -> dict:
    """SEPA Step 4 advisory levels. Pivot = the breakout/base level if detected, else the
    52-week high (the line a breakout would clear).

    The stop is measured from the PIVOT (the intended buy point): Minervini's 7-8% ideal, 10%
    hard max. The engine's ``calculate_stop_loss`` anchors to the *current* price and swing-low/
    50-SMA support, which for a name below its pivot can sit well past 10% below it. We floor the
    advisory stop at ``MAX_STOP_FROM_PIVOT`` below the pivot so the max-loss rule holds (a tighter
    engine stop is kept; ``stop_clamped`` records whether the floor bound). Advisory only — never
    moves a real order."""
    pivot = breakout.get("breakout_level")
    # A '50 SMA Breakout' level IS the 50-day SMA (a routine pullback-to-50-day recovery), not a
    # base pivot — anchoring the buy zone/stop/target (and the frozen trigger level) to it is
    # wrong. Ignore it and fall through to the 52-week high the strategy defines as the pivot.
    if breakout.get("breakout_type") == "50 SMA Breakout":
        pivot = None
    if not pivot or pivot <= 0:
        pivot = phase_info.get("week_52_high") or cp
    raw_stop = stop if (stop and stop > 0 and stop < pivot) else pivot * 0.925
    floor = pivot * (1.0 - MAX_STOP_FROM_PIVOT)             # 10% below pivot = the hard max
    stop_price = max(raw_stop, floor)
    stop_clamped = raw_stop < floor
    pct_to_pivot = ((pivot - cp) / cp * 100.0) if cp else None
    return {
        "pivot": float(pivot),
        "buy_zone": (float(pivot), float(pivot) * 1.05),   # no chasing > +5%
        "stop": float(stop_price),                          # 7-8% below pivot, 10% hard floor
        "stop_pct_from_pivot": ((pivot - stop_price) / pivot * 100.0) if pivot else None,
        "stop_clamped": bool(stop_clamped),
        "target": float(pivot) * 1.25,                      # +25% objective (user-locked)
        "breakout_today": bool(breakout.get("is_breakout")),
        "volume_ratio": float(breakout.get("volume_ratio", 1.0) or 1.0),
        "volume_confirmed": bool(breakout.get("volume_confirmed", False)),  # vol >= 1.5x avg
        "pct_to_pivot": pct_to_pivot,
    }


def screen_universe(tickers: List[str], prices: Dict[str, pd.DataFrame],
                    spy: pd.DataFrame,
                    get_fundamentals: Optional[Callable[[str], Optional[dict]]] = None,
                    cfg: Optional[ScanConfig] = None,
                    progress: Optional[Callable[[int, int, str], None]] = None) -> ScanResult:
    """Run the SEPA funnel over already-fetched price frames (deterministic/offline).

    ``prices``: {ticker -> daily OHLCV}. ``spy``: SPY daily OHLCV.
    ``get_fundamentals``: optional callable run only on Step-1 passers (cheap).
    ``progress``: optional ``(done, total, ticker)`` callback, called once per name —
    on a warm cache this loop (phase/VCP detection) is the multi-minute part of a scan.
    """
    cfg = cfg or ScanConfig()
    errors: List[str] = []
    spy_cp = float(spy["Close"].iloc[-1])
    spy_analysis = analyze_spy_trend(spy, spy_cp)
    rs_rating = _rs_ratings(prices, cfg.rs_period)

    phase_results: List[dict] = []
    rows: List[dict] = []
    payloads: Dict[str, dict] = {}

    for _i, t in enumerate(tickers, 1):
        if progress:
            progress(_i, len(tickers), t)
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
            earnings_in = _days_to_earnings(fund)

            levels = _entry_levels(cp, breakout, stop, phase_info)
            # RMV (Relative Measured Volatility): advisory base-tightness read for Step 4.
            # Advisory only — does NOT feed the pivot/stop/target math.
            rmv_series = relative_measured_volatility(df).dropna()
            levels["rmv"] = float(rmv_series.iloc[-1]) if len(rmv_series) else None
            # BBWP + TTM squeeze: Bollinger-side volatility read, an advisory cross-check on RMV.
            # BBWP low = a Bollinger squeeze; squeeze True = bands inside the Keltner channel.
            bbwp_series = bollinger_bandwidth_percentile(df).dropna()
            levels["bbwp"] = float(bbwp_series.iloc[-1]) if len(bbwp_series) else None
            sq = ttm_squeeze(df)
            levels["squeeze"] = bool(sq.iloc[-1]) if len(sq) else False
            # "squeeze fired": coiled within the last ~6 bars but expanding now — the
            # volatility-EXPANSION side of a breakout.
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
                "earnings_in": earnings_in,
                "tier": vcp.get("tier", "B"),
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
                "earnings_in": earnings_in,
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
        # Review order: tier A first (the shortlist), then quality within a tier — the
        # recall-first workflow is "walk the A block, glance at B, trust C's reasons".
        cand = pd.DataFrame(rows).sort_values(
            ["tier", "vcp_quality", "fund_score", "rs"],
            ascending=[True, False, False, False],
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
    inside the funnel (only for Step-1 passers).

    Prices always go through the cheap incremental top-up (``max_age_days=0.0``): a cache
    that already has today's bar re-fetches just the latest bars; a cache last written on
    an earlier day fetches only the missing days; only cold names (or a genuine
    split/dividend re-baseline) pay the full 2y download. ``force=True`` is the explicit
    full-re-download escape hatch (the app's Advanced ⟳ button)."""
    from . import data_feed
    tickers = data_feed.get_universe(universe, force=force)
    spy = data_feed.get_spy(force=force, max_age_days=0.0)
    if spy is None or len(spy) < 200:
        raise RuntimeError("Could not fetch SPY benchmark data (needed for RS/regime).")
    # Two sequential progress phases share one (done, total, label) callback; the label
    # prefix tells the UI which phase the bar is in.
    _p_fetch = (None if progress is None
                else lambda d, t, s: progress(d, t, f"Prices · {s}"))
    _p_screen = (None if progress is None
                 else lambda d, t, s: progress(d, t, f"Screening · {s}"))
    prices = data_feed.get_many_prices(tickers, max_workers=max_workers,
                                       force=force, max_age_days=0.0, progress=_p_fetch)
    return screen_universe(list(prices.keys()), prices, spy,
                           get_fundamentals=data_feed.get_fundamentals, cfg=cfg,
                           progress=_p_screen)
