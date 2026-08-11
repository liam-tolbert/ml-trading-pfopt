"""Watchlist trigger check — pure logic (no Streamlit, no network; data via arguments).

The weekend hunt builds the watchlist; the scheduled job (every 30 minutes during market
hours) answers ONE question per name: **is it above its frozen pivot on ≥1.5× average
volume?** Intraday runs read the live provisional bar (flagged ``intraday``; ``volume_pace``
says whether volume is running hot for the time of day); the last run of the day (~16:30)
sees the settled close. :func:`check_triggers` answers from already-fetched frames; the CLI
wrapper (``eod_trigger.py``) does the fetching, report persistence, and scheduling glue.

Pivots are FROZEN on the watchlist entry (``judged_pivot`` — see ``export.py``): the
detected pivot drifts with every scan, so a trigger against a recomputed level would move
under your feet. An entry that arrives unfrozen is frozen ON FIRST SIGHT here
(:func:`freeze_missing_pivots`, ``pivot_source="auto"``) and checked in the same run; the
📌 button in the app overrides with the level you judged (``pivot_source="judged"``).

Volume gate: last volume / prior 50-day average ≥ 1.5 — Minervini's confirmation standard,
the same 50-day read the positions page uses (``trade.HEAVY_VOL_RATIO``). The scan's
"Vol OK" badge is a 20-day read (``detect_breakout``); reported as context
(``volume_ratio_20``) but never gates.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.stock_screener.minervini_screener.screening import (
    analyze_spy_trend, calculate_sma, calculate_stop_loss, classify_phase,
    validate_minervini_trend_template)
from . import cache
from .export import make_entry
from .scan import _days_to_earnings, _entry_levels, detect_breakout_prior_high
from .vcp import detect_vcp

TRIGGER_VOL_RATIO = 1.5     # Minervini's breakout confirmation (mirrors trade.HEAVY_VOL_RATIO)
VOL_AVG_DAYS = 50           # ...vs the 50-day average volume, EXCLUDING today's bar
VOL_CONTEXT_DAYS = 20       # the scan's window — reported as context, never the gate
EXTENDED_PCT = 0.05         # close > pivot * 1.05 = past the buy zone ("don't chase")
EARNINGS_SOON_DAYS = 21     # mirror the app's ⚠ earnings window
MIN_ROWS_FOR_PIVOT = 200    # classify_phase needs >= 200 rows to compute a pivot
TEMPLATE_CRITERIA = 8       # mirror ScanConfig.min_criteria — the scan table's hard gate

# Intraday half-hourly runs: the session clock for the volume-pace read and the
# provisional-bar flag.
SESSION_OPEN_MIN = 9 * 60 + 30    # 09:30 ET, in minutes-of-day
SESSION_LEN_MIN = 390             # 09:30 -> 16:00
PACE_MIN_ELAPSED = 0.1            # clip: dividing by the first few minutes explodes the pace
INTRADAY_CUTOFF_MIN = 16 * 60 + 5  # runs before ~16:05 ET see a provisional bar
EARLY_CLOSE_LEN_MIN = 210         # NYSE half days close 13:00 -> a 09:30-13:00 session
EARLY_CLOSE_CUTOFF_MIN = 13 * 60 + 5  # ...and the bar settles ~13:05 on those days

REPORT_SCHEMA = 1
# The full per-name status vocabulary, pinned by the test suite — a new status must be
# registered here or the report test fails. "untracked" = the name no longer passes the
# 8/8 trend template (it left the scan table): kept on the watchlist, but its trigger is
# NOT evaluated until it re-qualifies. "crossed" (§6.19 open item, built §6.36) = above
# the frozen pivot WITHOUT volume confirmation — the quiet drift the volume gate will
# never fire on (PECO). Rendered loud so "it left without me" stops looking identical to
# "still basing"; NOT a buy signal.
STATUSES = ("no_data", "untracked", "no_pivot", "stale", "extended", "triggered",
            "crossed", "watch")


def _today_et(today=None) -> pd.Timestamp:
    """The run's trading date: 'today' in New York, tz-naive (price indexes are naive)."""
    if today is not None:
        return pd.Timestamp(today).normalize()
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


def _now_et(now=None) -> pd.Timestamp:
    """The run's wall clock in New York, tz-naive — pinned in tests like ``today``."""
    if now is not None:
        return pd.Timestamp(now)
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None)


def _early_close(date) -> bool:
    """NYSE recurring 1:00 pm early closes: July 3 (when Mon-Thu), the day after
    Thanksgiving (4th Thursday of November), and December 24 (when Mon-Thu).

    A FRIDAY Jul 3 / Dec 24 is the OBSERVED full holiday (Jul 4 / Dec 25 falls on a
    Saturday), so the Mon-Thu rule excludes exactly the right years; weekend dates aren't
    sessions at all. One-off special closes (e.g. days of mourning) are not modeled."""
    d = pd.Timestamp(date)
    if (d.month == 7 and d.day == 3) or (d.month == 12 and d.day == 24):
        return d.weekday() <= 3                          # Mon..Thu
    if d.month == 11:
        first_thu = 1 + (3 - pd.Timestamp(year=d.year, month=11, day=1).weekday()) % 7
        return d.day == first_thu + 21 + 1               # day after the 4th Thursday
    return False


def _session_len_min(date) -> int:
    return EARLY_CLOSE_LEN_MIN if _early_close(date) else SESSION_LEN_MIN


def _intraday_cutoff_min(date) -> int:
    return EARLY_CLOSE_CUTOFF_MIN if _early_close(date) else INTRADAY_CUTOFF_MIN


def _session_elapsed(now: pd.Timestamp) -> float:
    """Fraction of the trading session elapsed at ``now`` (09:30-16:00, or 09:30-13:00 on
    an early-close half day): 0.0 pre-open, 1.0 at/after the close, clipped to
    >= PACE_MIN_ELAPSED once trading has begun (the first minutes would otherwise explode
    the pace ratio)."""
    mins = now.hour * 60 + now.minute - SESSION_OPEN_MIN
    if mins <= 0:
        return 0.0
    return min(1.0, max(PACE_MIN_ELAPSED, mins / _session_len_min(now.normalize())))


def no_session_since(mtime_epoch: float, now=None) -> bool:
    """True when NO market-session time falls between ``mtime_epoch`` (a cache file's
    write time, epoch seconds) and ``now`` — no new price data can exist, so the cache
    is still CURRENT regardless of wall-clock age. Written after the settled close
    (~16:05 ET; ~13:05 on an early close) it stays good all evening, over the weekend,
    and through Monday pre-open. Sessions are weekday 09:30 → the settled-bar cutoff —
    the 16:00-16:05 settle window counts as session time, so a cache written at 16:02
    (possibly provisional volume) correctly reads stale. Full-market holidays are NOT
    modeled (same stance as ``_early_close``): a holiday weekday counts as a session,
    which fails SAFE — a needless cheap top-up, never a stale serve."""
    try:
        m = (pd.Timestamp(mtime_epoch, unit="s", tz="UTC")
             .tz_convert("America/New_York").tz_localize(None))
    except Exception:
        return False
    n = _now_et(now)
    if m >= n:
        return True
    day = m.normalize()
    while day <= n.normalize():
        if day.weekday() < 5:                             # Mon-Fri
            s_open = day + pd.Timedelta(minutes=SESSION_OPEN_MIN)
            s_end = day + pd.Timedelta(minutes=_intraday_cutoff_min(day))
            if max(s_open, m) < min(s_end, n):            # (m, n) overlaps the session
                return False
        day += pd.Timedelta(days=1)
    return True


def frame_settled_current(last_bar_date, now=None) -> bool:
    """True when a frame ENDING at ``last_bar_date`` already contains every bar that can
    exist: that bar's session has settled and no later session has started (evenings /
    weekends / pre-open). The settled-close gate's content-side companion (R2-5b): the
    file mtime says WHEN it was written; this says whether what's INSIDE is actually the
    latest settled data — a lagging provider response persisted post-cutoff would
    otherwise serve a short frame as "settled" for the whole no-session window. Routes
    through the module-global ``no_session_since`` (tests patch it there). Never raises;
    any error reads as not-current (the cheap top-up decides)."""
    try:
        day = pd.Timestamp(last_bar_date).normalize()
        end = day + pd.Timedelta(minutes=_intraday_cutoff_min(day))
        end_epoch = end.tz_localize("America/New_York").timestamp()
        return no_session_since(end_epoch, now=now)
    except Exception:
        return False


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
    """Recompute the APP pivot for one frame — the EXACT chain the scan uses (classify_phase
    -> detect_vcp -> detect_breakout -> calculate_stop_loss -> _entry_levels). The VCP result
    MUST be passed into detect_breakout: without it there's no VCP-peak breakout level and
    _entry_levels silently falls back to the 52-week high — a different, usually higher pivot
    than the chart's. None when the frame is missing/short (< MIN_ROWS_FOR_PIVOT) or the chain
    errors — never raises."""
    if df is None or len(df) < MIN_ROWS_FOR_PIVOT:
        return None
    try:
        cp = float(df["Close"].iloc[-1])
        phase_info = classify_phase(df, cp)
        vcp = detect_vcp(df, cp, phase_info)
        breakout = detect_breakout_prior_high(df, cp, phase_info, vcp)
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
              today=None, now=None) -> dict:
    """Evaluate ONE watchlist entry against its (already-refreshed) daily frame.

    Returns the per-name report dict (see ``check_triggers``). ``status`` is a display
    convenience with precedence no_data -> untracked -> no_pivot -> stale -> extended ->
    triggered -> crossed -> watch; the booleans stay authoritative. ``triggered`` requires close above the frozen
    pivot AND the 50-day volume gate AND a bar dated today (a Friday bar must not re-fire
    on a Monday-holiday run). ``crossed`` = close above the pivot WITHOUT the volume
    confirm — the quiet drift the trigger can't fire on (a name frozen post-breakout may
    sit here forever); informational, never a buy signal. ``volume_pace`` (intraday context, NEVER the gate) is the
    50-day ratio divided by the fraction of the session elapsed at ``now`` — "is volume
    running hot for this time of day?"; equals the plain ratio after the close."""
    t = _today_et(today)
    out = {
        "ticker": entry.get("ticker"), "status": "no_data",
        "judged_pivot": entry.get("judged_pivot"),
        "pivot_source": entry.get("pivot_source"),
        "date_added": entry.get("date_added"), "note": entry.get("note", ""),
        "close": None, "last_bar": None, "stale": None, "close_above_pivot": None,
        "volume": None, "volume_ratio_50": None, "volume_ratio_20": None,
        "volume_ratio_50_scaled": None, "early_close": bool(_early_close(t)),
        "volume_pace": None, "volume_confirmed": None, "triggered": False,
        "crossed": None, "untracked": False,
        "extended": None, "pct_from_pivot": None, "earnings_in": None,
        "earnings_soon": None, "error": None,
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
        if out["volume_ratio_50"] is not None and out["stale"] is False:
            frac = _session_elapsed(_now_et(now))
            if frac > 0:                                  # pre-open -> no pace read
                out["volume_pace"] = round(out["volume_ratio_50"] / frac, 2)

        # A name that no longer passes the trend template has LEFT the scan table — keep
        # it on the watchlist but don't evaluate its trigger (a "breakout" on a
        # broken-down base is noise, not a Minervini entry). Judged on the name's own
        # frame so the headless half-hourly job needs no universe scan; frames too short
        # to judge (< MIN_ROWS_FOR_PIVOT — the scan couldn't table them either) and any
        # template-chain error fail OPEN (keep evaluating, never blind the check).
        if len(df) >= MIN_ROWS_FOR_PIVOT:
            try:
                phase_info = classify_phase(df, close)
                sma200 = calculate_sma(df["Close"], 200)
                tmpl = validate_minervini_trend_template(close, phase_info, sma200)
                out["untracked"] = bool(
                    tmpl.get("criteria_passed", 0) < TEMPLATE_CRITERIA)
            except Exception:
                out["untracked"] = False
        if out["untracked"]:
            out["status"] = "untracked"
            return out

        pivot = entry.get("judged_pivot")
        if not pivot or pivot <= 0:
            out["status"] = "no_pivot"                    # unfrozen and couldn't auto-freeze
            return out

        vr = out["volume_ratio_50"]
        gate = vr
        if vr is not None and out["stale"] is False and out["early_close"]:
            # A half session's volume vs full-day averages understates ~1.86x — scale the
            # GATE so a genuinely heavy half day can still confirm (user decision); the
            # displayed ratio stays RAW, the scaled value is reported alongside.
            gate = vr * (SESSION_LEN_MIN / _session_len_min(t))
            out["volume_ratio_50_scaled"] = round(gate, 2)
        out["close_above_pivot"] = bool(close > pivot)
        out["volume_confirmed"] = bool(gate is not None and gate >= TRIGGER_VOL_RATIO)
        out["extended"] = bool(close > pivot * (1.0 + EXTENDED_PCT))
        out["pct_from_pivot"] = round((close / pivot - 1.0) * 100.0, 2)
        out["triggered"] = bool(out["close_above_pivot"] and out["volume_confirmed"]
                                and not out["stale"])
        # `not stale` for symmetry with `triggered` (R2-10): a stale Friday bar above
        # the pivot is no more a live cross than it is a live trigger — the status
        # precedence hid this, but the raw boolean is documented as authoritative.
        out["crossed"] = bool(out["close_above_pivot"] and not out["volume_confirmed"]
                              and not out["stale"])

        out["status"] = ("stale" if out["stale"]
                         else "extended" if out["extended"]
                         else "triggered" if out["triggered"]
                         else "crossed" if out["crossed"]
                         else "watch")
    except Exception as e:                                # per-name failures never abort the run
        out["error"] = str(e)
    return out


def check_triggers(entries: Sequence[dict], prices: Dict[str, pd.DataFrame],
                   fundamentals: Optional[Callable[[str], Optional[dict]]] = None,
                   spy: Optional[pd.DataFrame] = None, today=None, now=None) -> dict:
    """Build the full report (pure, deterministic under pinned ``today``/``now``).

    ``fundamentals`` is an optional per-ticker callable (the CLI passes a cached
    ``data_feed.get_fundamentals``); its failures count as "no earnings info", never
    fatal. ``spy`` (if given, >= 200 rows) yields an ``analyze_spy_trend`` note — SPY-only
    by design: the app banner's full regime needs universe breadth a scheduled check
    shouldn't pay for. The report's ``intraday`` flag marks runs made on a live session
    bar (fresh bar + before ~16:05 ET) — close/volume are then provisional."""
    t = _today_et(today)
    now_t = _now_et(now)
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
        names.append(check_one(e, prices.get(e["ticker"]), fund, today=t, now=now_t))

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
        "crossed": [n["ticker"] for n in names if n["status"] == "crossed"],
        "extended": [n["ticker"] for n in names if n["status"] == "extended"],
        "stale": [n["ticker"] for n in names if n.get("stale")],
        "untracked": [n["ticker"] for n in names if n["status"] == "untracked"],
        "earnings_soon": [n["ticker"] for n in names if n.get("earnings_soon")],
        "no_data": [n["ticker"] for n in names if n["status"] == "no_data"],
        "no_pivot": [n["ticker"] for n in names if n["status"] == "no_pivot"],
        "auto_frozen": [],                                # the CLI fills this in
    }
    fresh = [n for n in names if n.get("stale") is False]
    return {
        "schema": REPORT_SCHEMA,
        "date": t.strftime("%Y-%m-%d"),
        "generated_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "spy": spy_note,
        "all_stale": bool(with_bars) and all(n["stale"] for n in with_bars),
        "early_close": bool(_early_close(t)),
        # A live session bar: close/volume are provisional until ~16:05 ET (~13:05 on an
        # early-close half day).
        "intraday": bool(fresh
                         and now_t.hour * 60 + now_t.minute < _intraday_cutoff_min(t)),
        "names": names,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Report persistence — dated JSON files, newest wins (track_portfolio's snapshot idiom).
# --------------------------------------------------------------------------- #
def save_trigger_report(report: dict, dir_path=None) -> Path:
    """Write ``triggers_YYYY-MM-DD.json`` (same-day rerun overwrites = idempotent).
    ``dir_path`` defaults to ``cache.TRIGGERS_DIR`` read at CALL time (patchable).

    Atomic (R2-7): tmp + ``os.replace``, mirroring data_feed's ``_atomic_to_parquet`` —
    the app's 🔔 button and the half-hourly scheduled job run in SEPARATE processes and
    can hit the same day-file; an in-place truncate-write could interleave into invalid
    JSON, which the loader silently skips, serving the PREVIOUS day all weekend if the
    ~16:30 settled report was the casualty."""
    d = Path(dir_path if dir_path is not None else cache.TRIGGERS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"triggers_{report.get('date', 'undated')}.json"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
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
    hm = str(report.get("generated_at", ""))[11:16]      # ISO -> HH:MM (crude, best-effort)
    lines.append(f"TRIGGER CHECK  {report.get('date', '?')}"
                 + (f" {hm}" if hm else "") + f"   {spy_s}")
    names = report.get("names", [])
    if not names:
        lines.append("watchlist is empty -- nothing to check.")
    else:
        lines.append(f"{'TICKER':<8}{'CLOSE':>9}{'PIVOT':>9}  {'SRC':<7}{'%FROM':>7}"
                     f"{'VOL50':>7}{'PACE':>7}  {'STATUS':<10}{'EARNINGS':<10}")
        for n in names:
            piv = n.get("judged_pivot")
            vr = n.get("volume_ratio_50")
            pace = n.get("volume_pace")
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
                f"{(f'{vr:.1f}x' if vr is not None else '-'):>7}"
                f"{(f'{pace:.1f}x' if pace is not None else '-'):>7}  "
                f"{n.get('status', '?').upper():<10}{earn:<10}"
                + (f"  ERR: {n['error']}" if n.get("error") else ""))
    s = report.get("summary", {})
    lines.append(f"summary: {len(s.get('triggered', []))} triggered, "
                 f"{len(s.get('crossed', []))} crossed, "
                 f"{len(s.get('extended', []))} extended, {len(s.get('stale', []))} stale, "
                 f"{len(s.get('no_pivot', []))} without a pivot, of {s.get('n', 0)}")
    if s.get("crossed"):
        lines.append(f"CROSSED ({', '.join(s['crossed'])}): above the frozen pivot WITHOUT "
                     f"volume confirmation -- a quiet drift is not a buy; wait for a "
                     f">={TRIGGER_VOL_RATIO}x volume close, or plan a pullback/secondary "
                     "entry off the pivot.")
    if s.get("untracked"):
        lines.append(f"UNTRACKED ({', '.join(s['untracked'])}): no longer passes the "
                     f"{TEMPLATE_CRITERIA}/8 trend template -- kept on the watchlist, but "
                     "the trigger is NOT evaluated until it re-qualifies.")
    if s.get("auto_frozen"):
        lines.append(f"auto-froze pivots (first sight): {', '.join(s['auto_frozen'])}"
                     " -- chart + pin (freeze button in the app) to override with your"
                     " judged level")
    if report.get("all_stale"):
        lines.append("NOTE: no new bar on the report date (weekend/holiday?) -- "
                     "no trigger can fire from a stale bar.")
    if report.get("early_close"):
        lines.append("NOTE: early close (1:00 pm ET) -- volume gate scaled x"
                     f"{SESSION_LEN_MIN / EARLY_CLOSE_LEN_MIN:.2f} for the short session; "
                     "VOL50 shows the raw ratio.")
    if report.get("intraday"):
        settle = "~13:00" if report.get("early_close") else "~16:00"
        lines.append(f"NOTE: intraday bar -- close/volume are provisional until {settle} ET; "
                     "PACE = volume so far vs expected by this time of day.")
    return "\n".join(lines)
