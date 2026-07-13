"""Nightly EOD trigger check — pure logic (no Streamlit, no network; data via arguments).

The weekend hunt builds the watchlist; the daily job (HANDOFF §6.11) is answering ONE
question per name after the close: **did it close above its frozen pivot on ≥1.5× average
volume?** :func:`check_triggers` answers it from already-fetched frames; the CLI wrapper
(``eod_trigger.py``) does the fetching, report persistence, and scheduling glue.

Pivots are FROZEN on the watchlist entry (``judged_pivot`` — see ``export.py``): the
detected pivot drifts with every scan, so a trigger against a recomputed level would move
under your feet. An entry that arrives unfrozen is frozen ON FIRST SIGHT here
(:func:`freeze_missing_pivots`, ``pivot_source="auto"``) and checked in the same run; the
📌 button in the app overrides with the level you judged (``pivot_source="judged"``).

Volume gate: last volume / prior 50-day average ≥ 1.5 — Minervini's confirmation standard,
the same 50-day read the positions page uses (``trade.HEAVY_VOL_RATIO``). The scan's
"Vol OK" badge is a 20-day read (``detect_breakout``); that number is reported as context
(``volume_ratio_20``) but never gates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.stock_screener.minervini_screener.screening import (
    analyze_spy_trend, calculate_stop_loss, classify_phase, detect_breakout)
from . import cache
from .export import make_entry
from .scan import _days_to_earnings, _entry_levels
from .vcp import detect_vcp

TRIGGER_VOL_RATIO = 1.5     # Minervini's breakout confirmation (mirrors trade.HEAVY_VOL_RATIO)
VOL_AVG_DAYS = 50           # ...vs the 50-day average volume, EXCLUDING today's bar
VOL_CONTEXT_DAYS = 20       # the scan's window — reported as context, never the gate
EXTENDED_PCT = 0.05         # close > pivot * 1.05 = past the buy zone ("don't chase")
EARNINGS_SOON_DAYS = 21     # mirror the app's ⚠ earnings window
MIN_ROWS_FOR_PIVOT = 200    # classify_phase needs >= 200 rows to compute a pivot

REPORT_SCHEMA = 1
STATUSES = ("no_data", "no_pivot", "stale", "extended", "triggered", "watch")


def _today_et(today=None) -> pd.Timestamp:
    """The run's trading date: 'today' in New York, tz-naive (price indexes are naive)."""
    if today is not None:
        return pd.Timestamp(today).normalize()
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


def _volume_ratio(df: pd.DataFrame, window: int) -> Optional[float]:
    """Last bar's volume vs the mean of the PRIOR ``window`` bars (excluding the last).
    None when there's no Volume column, too little history, or a non-positive mean."""
    try:
        v = df["Volume"]
        if len(v) < window + 1:
            return None
        avg = float(v.iloc[-(window + 1):-1].mean())
        return float(v.iloc[-1]) / avg if avg > 0 else None
    except Exception:
        return None


def compute_scan_pivot(df: Optional[pd.DataFrame]) -> Optional[float]:
    """Recompute the APP pivot for one frame — the EXACT chain the scan uses
    (classify_phase -> detect_vcp -> detect_breakout -> calculate_stop_loss ->
    _entry_levels, scan.py's screen_universe loop). The VCP result MUST be passed into
    detect_breakout: without it there is no VCP-peak breakout level and _entry_levels
    silently falls back to the 52-week high — a different, usually higher pivot than the
    one on the chart (live case: EBAY 118.98 vs the app's 111.86). None when the frame is
    missing/short (< MIN_ROWS_FOR_PIVOT) or the chain errors — never raises."""
    if df is None or len(df) < MIN_ROWS_FOR_PIVOT:
        return None
    try:
        cp = float(df["Close"].iloc[-1])
        phase_info = classify_phase(df, cp)
        vcp = detect_vcp(df, cp, phase_info)
        breakout = detect_breakout(df, cp, phase_info, vcp)
        stop = calculate_stop_loss(df, cp, phase_info, phase_info.get("phase", 2))
        pivot = _entry_levels(cp, breakout, stop, phase_info).get("pivot")
        return float(pivot) if pivot and pivot > 0 else None
    except Exception:
        return None


def freeze_missing_pivots(entries: Sequence[dict], prices: Dict[str, pd.DataFrame],
                          today=None) -> Tuple[List[dict], List[str]]:
    """Freeze-on-first-sight: every unfrozen entry with a computable pivot gets it
    recorded (``pivot_source="auto"``, ``date_added`` = the run date) so the trigger level
    stops drifting from tonight on. Pure — returns (updated entry COPIES, tickers frozen
    this run); the caller persists. Entries that can't be computed (no/short frame, chain
    error) come back unchanged and retry next run."""
    run_date = _today_et(today).strftime("%Y-%m-%d")
    out: List[dict] = []
    frozen: List[str] = []
    for e in entries or []:
        ent = dict(e) if isinstance(e, dict) else {"ticker": str(e).strip().upper()}
        t = ent.get("ticker")
        if t and not ent.get("judged_pivot"):
            pivot = compute_scan_pivot(prices.get(t))
            if pivot:
                fresh = make_entry(t, pivot, date_added=run_date, pivot_source="auto",
                                   note=ent.get("note", ""))
                if fresh and fresh["judged_pivot"]:
                    ent = fresh
                    frozen.append(t)
        out.append(ent)
    return out, frozen


def check_one(entry: dict, df: Optional[pd.DataFrame], fund: Optional[dict], *,
              today=None) -> dict:
    """Evaluate ONE watchlist entry against its (already-refreshed) daily frame.

    Returns the per-name report dict (see ``check_triggers``). ``status`` is a display
    convenience with precedence no_data -> no_pivot -> stale -> extended -> triggered ->
    watch; the booleans stay authoritative. ``triggered`` requires close above the frozen
    pivot AND the 50-day volume gate AND a bar dated today (a Friday bar must not re-fire
    on a Monday-holiday run)."""
    t = _today_et(today)
    out = {
        "ticker": entry.get("ticker"), "status": "no_data",
        "judged_pivot": entry.get("judged_pivot"),
        "pivot_source": entry.get("pivot_source"),
        "date_added": entry.get("date_added"), "note": entry.get("note", ""),
        "close": None, "last_bar": None, "stale": None, "close_above_pivot": None,
        "volume": None, "volume_ratio_50": None, "volume_ratio_20": None,
        "volume_confirmed": None, "triggered": False, "extended": None,
        "pct_from_pivot": None, "earnings_in": None, "earnings_soon": None, "error": None,
    }
    try:
        out["earnings_in"] = _days_to_earnings(fund, today=t)
        ei = out["earnings_in"]
        out["earnings_soon"] = (ei is not None and 0 <= ei <= EARNINGS_SOON_DAYS)

        if df is None or not len(df):
            return out                                    # status stays no_data
        close = float(df["Close"].iloc[-1])
        last_bar = pd.Timestamp(df.index[-1]).normalize()
        out["close"] = round(close, 2)
        out["last_bar"] = last_bar.strftime("%Y-%m-%d")
        out["stale"] = bool(last_bar != t)
        out["volume"] = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else None
        out["volume_ratio_50"] = _volume_ratio(df, VOL_AVG_DAYS)
        out["volume_ratio_20"] = _volume_ratio(df, VOL_CONTEXT_DAYS)

        pivot = entry.get("judged_pivot")
        if not pivot or pivot <= 0:
            out["status"] = "no_pivot"                    # unfrozen and couldn't auto-freeze
            return out

        vr = out["volume_ratio_50"]
        out["close_above_pivot"] = bool(close > pivot)
        out["volume_confirmed"] = bool(vr is not None and vr >= TRIGGER_VOL_RATIO)
        out["extended"] = bool(close > pivot * (1.0 + EXTENDED_PCT))
        out["pct_from_pivot"] = round((close / pivot - 1.0) * 100.0, 2)
        out["triggered"] = bool(out["close_above_pivot"] and out["volume_confirmed"]
                                and not out["stale"])

        out["status"] = ("stale" if out["stale"]
                         else "extended" if out["extended"]
                         else "triggered" if out["triggered"]
                         else "watch")
    except Exception as e:                                # per-name failures never abort the run
        out["error"] = str(e)
    return out


def check_triggers(entries: Sequence[dict], prices: Dict[str, pd.DataFrame],
                   fundamentals: Optional[Callable[[str], Optional[dict]]] = None,
                   spy: Optional[pd.DataFrame] = None, today=None) -> dict:
    """Build the full nightly report (pure, deterministic under a pinned ``today``).

    ``fundamentals`` is an optional per-ticker callable (the CLI passes a cached
    ``data_feed.get_fundamentals``); its failures count as "no earnings info", never
    fatal. ``spy`` (if given, >= 200 rows) yields an ``analyze_spy_trend`` note — SPY-only
    by design: the app banner's full regime needs universe breadth a nightly check
    shouldn't pay for."""
    t = _today_et(today)
    names: List[dict] = []
    for e in entries or []:
        if not isinstance(e, dict) or not e.get("ticker"):
            continue
        fund = None
        if fundamentals is not None:
            try:
                fund = fundamentals(e["ticker"])
            except Exception:
                fund = None
        names.append(check_one(e, prices.get(e["ticker"]), fund, today=t))

    spy_note = None
    if spy is not None and len(spy) >= MIN_ROWS_FOR_PIVOT:
        try:
            a = analyze_spy_trend(spy, float(spy["Close"].iloc[-1]))
            spy_note = {"phase": a.get("phase"), "phase_name": a.get("phase_name"),
                        "trend": a.get("trend")}
        except Exception:
            spy_note = None

    with_bars = [n for n in names if n["status"] != "no_data"]
    summary = {
        "n": len(names),
        # by STATUS, not the raw booleans: an extended name that also cleared price+volume
        # keeps triggered=True in its row, but the summary files it under "don't chase".
        "triggered": [n["ticker"] for n in names if n["status"] == "triggered"],
        "extended": [n["ticker"] for n in names if n["status"] == "extended"],
        "stale": [n["ticker"] for n in names if n.get("stale")],
        "earnings_soon": [n["ticker"] for n in names if n.get("earnings_soon")],
        "no_data": [n["ticker"] for n in names if n["status"] == "no_data"],
        "no_pivot": [n["ticker"] for n in names if n["status"] == "no_pivot"],
        "auto_frozen": [],                                # the CLI fills this in
    }
    return {
        "schema": REPORT_SCHEMA,
        "date": t.strftime("%Y-%m-%d"),
        "generated_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "spy": spy_note,
        "all_stale": bool(with_bars) and all(n["stale"] for n in with_bars),
        "names": names,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Report persistence — dated JSON files, newest wins (track_portfolio's snapshot idiom).
# --------------------------------------------------------------------------- #
def save_trigger_report(report: dict, dir_path=None) -> Path:
    """Write ``triggers_YYYY-MM-DD.json`` (same-day rerun overwrites = idempotent).
    ``dir_path`` defaults to ``cache.TRIGGERS_DIR`` read at CALL time (patchable)."""
    d = Path(dir_path if dir_path is not None else cache.TRIGGERS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"triggers_{report.get('date', 'undated')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def load_latest_trigger_report(dir_path=None) -> Optional[dict]:
    """Newest parseable ``triggers_*.json`` in the directory, or None (missing dir, no
    files, all corrupt). One corrupt file can't blind the app — we walk newest-first.
    Never raises."""
    try:
        d = Path(dir_path if dir_path is not None else cache.TRIGGERS_DIR)
        for path in sorted(d.glob("triggers_*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    except Exception:
        pass
    return None


def format_report(report: dict) -> str:
    """ASCII-only console rendering (the .bat wrapper appends stdout to a cp1252 log —
    no emoji here; icons live in the Streamlit surface)."""
    lines: List[str] = []
    spy = report.get("spy") or {}
    spy_s = (f"SPY: {spy.get('trend', '?')} (phase {spy.get('phase', '?')})"
             if spy else "SPY: n/a")
    lines.append(f"EOD TRIGGER CHECK  {report.get('date', '?')}   {spy_s}")
    names = report.get("names", [])
    if not names:
        lines.append("watchlist is empty -- nothing to check.")
    else:
        lines.append(f"{'TICKER':<8}{'CLOSE':>9}{'PIVOT':>9}  {'SRC':<7}{'%FROM':>7}"
                     f"{'VOL50':>7}  {'STATUS':<10}{'EARNINGS':<10}")
        for n in names:
            piv = n.get("judged_pivot")
            vr = n.get("volume_ratio_50")
            pct = n.get("pct_from_pivot")
            ei = n.get("earnings_in")
            earn = ("-" if ei is None
                    else f"in {ei}d" + (" !" if n.get("earnings_soon") else ""))
            lines.append(
                f"{n.get('ticker', '?'):<8}"
                f"{(f'{n['close']:.2f}' if n.get('close') is not None else '-'):>9}"
                f"{(f'{piv:.2f}' if piv else '-'):>9}  "
                f"{(n.get('pivot_source') or '-'):<7}"
                f"{(f'{pct:+.1f}%' if pct is not None else '-'):>7}"
                f"{(f'{vr:.1f}x' if vr is not None else '-'):>7}  "
                f"{n.get('status', '?').upper():<10}{earn:<10}"
                + (f"  ERR: {n['error']}" if n.get("error") else ""))
    s = report.get("summary", {})
    lines.append(f"summary: {len(s.get('triggered', []))} triggered, "
                 f"{len(s.get('extended', []))} extended, {len(s.get('stale', []))} stale, "
                 f"{len(s.get('no_pivot', []))} without a pivot, of {s.get('n', 0)}")
    if s.get("auto_frozen"):
        lines.append(f"auto-froze pivots (first sight): {', '.join(s['auto_frozen'])}"
                     " -- chart + pin (freeze button in the app) to override with your"
                     " judged level")
    if report.get("all_stale"):
        lines.append("NOTE: no new bar on the report date (weekend/holiday?) -- "
                     "no trigger can fire from a stale bar.")
    return "\n".join(lines)
