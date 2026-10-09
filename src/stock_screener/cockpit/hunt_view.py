"""What the Weekend Hunt page reads and does, without Streamlit, so it is unit-testable.

A hunt result is a folder ``data/cockpit/hunt/<date>/`` the hunt PC pushes to the Pi
(diagnostics.csv, verdicts.csv, meta.json, narrative.md, catalyst.json, report.html,
charts/). This module finds those folders, orders the PASS names for the click-through,
leaves a request for the hunt PC and reads its status (``hunt_request``), and adds a
name to the watchlist.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from src.stock_screener.cockpit import cache, hunt_request
from src.stock_screener.cockpit.export import (load_watchlist, make_entry, merge_frozen_pivots,
                                                save_watchlist, watchlist_tickers)
from src.stock_screener.hunt import pipeline as pl
from src.stock_screener.hunt.report import (HuntState, load_catalysts,  # noqa: F401 (re-export)
                                            load_state, review_sheets)

HUNT_DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BUCKET_ORDER = ("buy_zone", "approaching", "below", "past_entry")
BUCKET_LABEL = {"buy_zone": "buy zone", "approaching": "approaching",
                "below": "below pivot", "past_entry": "past entry",
                "earnings_blocked": "earnings-blocked"}
_PANEL_NAMES = ("top-left", "top-right", "bottom-left", "bottom-right")
_IMG_RX = re.compile(r'<img\s[^>]*src="charts/([^"]+)"[^>]*>')


# ---- hunt folders ----------------------------------------------------------- #
def hunt_root() -> Path:
    """``pipeline.HUNT_DIR`` read at call time, so tests can patch it."""
    return pl.HUNT_DIR


FINISHED_FILES = ("diagnostics.csv", "meta.json", "report.html")


def hunt_dirs(root: Optional[Path] = None) -> List[Path]:
    """Dated hunt folders holding a finished hunt (``FINISHED_FILES``), newest first.
    The ``logs/`` sibling, a half-pushed folder and, on the hunt PC itself, a run still
    in progress are skipped."""
    root = root or hunt_root()
    if not root.is_dir():
        return []
    out = [p for p in root.iterdir()
           if p.is_dir() and HUNT_DATE_RX.match(p.name)
           and all((p / f).exists() for f in FINISHED_FILES)]
    return sorted(out, key=lambda p: p.name, reverse=True)


def latest_hunt_dir(root: Optional[Path] = None) -> Optional[Path]:
    dirs = hunt_dirs(root)
    return dirs[0] if dirs else None


def load_state_safe(hunt_path: Path) -> Tuple[Optional[HuntState], str]:
    """``(state, "")`` or ``(None, reason)``. Never raises: a page must not die on a
    folder a crashed run left behind."""
    try:
        return load_state(hunt_path), ""
    except Exception as e:                # the reason is shown to the user
        return None, f"{hunt_path.name}: {type(e).__name__}: {e}"


# ---- the click-through ------------------------------------------------------ #
@dataclass
class Pick:
    ticker: str
    bucket: str            # a BUCKET_ORDER key, or "earnings_blocked"
    row: dict              # the diagnostics row (load_state's coercion)
    verdict: str
    notes: str
    blocked: bool
    earnings_in: Optional[int]


def _earnings_in(row: dict) -> Optional[int]:
    try:
        return int(float(row.get("earnings_in")))
    except (TypeError, ValueError):
        return None


def ordered_picks(state: HuntState) -> List[Pick]:
    """Every PASS name, bucket by bucket (the gates already sort each by quality), then
    the earnings-blocked PASS names, soonest report first."""
    def pick(row, bucket, blocked):
        v = state.verdicts.get(row["ticker"]) or {}
        return Pick(row["ticker"], bucket, row, v.get("verdict", "PASS"),
                    v.get("notes", ""), blocked, _earnings_in(row))
    out = [pick(r, b, False) for b in BUCKET_ORDER for r in state.buckets.get(b, [])]
    blocked = sorted(state.blocked, key=lambda r: _earnings_in(r) or 0)
    out += [pick(r, "earnings_blocked", True) for r in blocked]
    return out


def sheet_for(hunt_path: Path, state: HuntState,
              ticker: str) -> Optional[Tuple[Path, int, int, Optional[int]]]:
    """``(png, sheet_no, n_sheets, panel_index)`` of the review sheet showing ``ticker``;
    the panel index is None when the sheet's contents are unknown; None without a sheet."""
    sheets = review_sheets(hunt_path, state.diag_rows)
    for i, (png, tickers) in enumerate(sheets, start=1):
        if tickers and ticker in tickers:
            return png, i, len(sheets), tickers.index(ticker)
    return None


def sheet_panel(png: Path, j: Optional[int]) -> Optional[bytes]:
    """The quarter of a 2×2 review sheet that holds panel ``j``, as PNG bytes. None when
    the panel is unknown or the image cannot be read; the caller shows the whole sheet."""
    if j is None or not 0 <= j < len(_PANEL_NAMES):
        return None
    try:
        import io
        from PIL import Image
        with Image.open(png) as im:
            w, h = im.size
            x0, y0 = (j % 2) * w // 2, (j // 2) * h // 2
            buf = io.BytesIO()
            im.crop((x0, y0, x0 + w // 2, y0 + h // 2)).save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return None


def panel_name(j: Optional[int]) -> str:
    """Where a name sits on its 2×2 review sheet (``charts.render_sheets`` fills it
    row-major)."""
    if j is None:
        return "one of the panels"
    return _PANEL_NAMES[j] if j < len(_PANEL_NAMES) else f"panel {j + 1}"


def entry_numbers(row: dict) -> dict:
    """The Step-4 numbers from a diagnostics row. Missing cells read as None."""
    def num(k):
        try:
            return float(row.get(k))
        except (TypeError, ValueError):
            return None
    pivot = num("pivot")
    return {"pivot": pivot, "close": num("close"), "vs_pivot_pct": num("vs_pivot_pct"),
            "zone_hi": None if pivot is None else pivot * (1 + pl.BUY_ZONE_MAX_PCT / 100.0),
            "stop": num("stop"), "adv_musd": num("adv_musd"),
            "max_order_usd": num("max_order_usd"), "earnings_in": _earnings_in(row),
            "dist_days": num("dist_days"), "rs": num("rs"), "q": num("q"),
            "fund": num("fund"), "f_max": num("f_max") or 8,
            "depths": str(row.get("depths") or ""), "step3": str(row.get("step3") or "")}


PIVOT_DRIFT_PCT = 0.5        # a smaller move is rounding, not a new pivot


def hunt_levels(row: dict, scan_levels: Optional[dict] = None) -> dict:
    """The chart's levels for a hunt pick: the pivot the hunt judged, with its stop, buy
    zone and +25% target. The detector re-anchors the pivot as bars arrive (a new high
    starts a new base), so the latest scan's levels can differ from what the verdict and
    the watchlist entry refer to; the chart MUST show the hunt's. Keys the hunt does not
    record (breakout flags, volume ratio) come from ``scan_levels`` when given."""
    nums = entry_numbers(row)
    out = dict(scan_levels or {})
    piv = nums["pivot"]
    if piv is None:
        return out
    out.update({"pivot": piv, "buy_zone": (piv, nums["zone_hi"]), "target": piv * 1.25,
                "stop": nums["stop"] if nums["stop"] is not None else out.get("stop")})
    return out


def pivot_drift(row: dict, scan_levels: Optional[dict]) -> Optional[Tuple[float, float]]:
    """``(today's pivot, % vs the hunt's)`` when the latest scan detects a pivot at least
    ``PIVOT_DRIFT_PCT`` away from the hunt's; None otherwise."""
    hunt_piv = entry_numbers(row)["pivot"]
    try:
        now = float((scan_levels or {}).get("pivot"))
    except (TypeError, ValueError):
        return None
    if not hunt_piv or not now:
        return None
    pct = (now / hunt_piv - 1.0) * 100.0
    return (now, pct) if abs(pct) >= PIVOT_DRIFT_PCT else None


def latest_payloads() -> Dict[str, dict]:
    """The latest scan's payloads by ticker, or ``{}``. Never raises and never waits: a
    cold start, the test harness, or a worker error all read as "no scan"."""
    try:
        from src.stock_screener.cockpit import scan_worker
        res = scan_worker.get_worker().latest()
        return dict(getattr(res, "payloads", None) or {})
    except Exception:
        return {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def inline_report_images(html: str, hunt_path: Path, embed: bool = True) -> str:
    """``report.html`` for an iframe, which cannot resolve ``charts/…``: the sheet images
    become data URIs, or are dropped when ``embed`` is False (a dozen full-size sheets is
    a heavy page on a Pi). A sheet missing on disk is dropped either way."""
    def sub(m):
        png = hunt_path / "charts" / m.group(1)
        if not embed or not png.exists():
            return ""
        data = base64.b64encode(png.read_bytes()).decode("ascii")
        return m.group(0).replace(f'src="charts/{m.group(1)}"',
                                  f'src="data:image/png;base64,{data}"')
    return _IMG_RX.sub(sub, html)


# ---- the catalyst ----------------------------------------------------------- #
def manual_links(ticker: str) -> List[Tuple[str, str]]:
    """Where to look further by hand: ``(label, url)`` pairs."""
    t = quote(ticker)
    return [
        ("Yahoo Finance news", f"https://finance.yahoo.com/quote/{t}/news"),
        ("SEC filings (8-K)",
         f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={t}&type=8-K"),
        ("Finviz", f"https://finviz.com/quote.ashx?t={t}"),
        ("Google News", f"https://news.google.com/search?q={t}+stock"),
    ]


# ---- asking the hunt PC ----------------------------------------------------- #
def request_hunt() -> Tuple[dict, bool]:
    """Leave a request for the PC's poller; ``(request, created)``. Nothing connects to
    the PC: it reads the file over ssh when it is awake."""
    return hunt_request.write_request("cockpit")


def hunt_progress() -> dict:
    """``hunt_request.progress()``: the request and the PC's status, as one state."""
    return hunt_request.progress()


# ---- the watchlist ---------------------------------------------------------- #
def hunt_note(hunt_date: str, verdict: str, notes: str) -> str:
    return f"weekend hunt {hunt_date}: {verdict}" + (f" — {notes}" if notes else "")


def add_to_watchlist(session_entries: Optional[list], ticker: str, pivot, hunt_date: str,
                     verdict: str, notes: str = "") -> Tuple[List[dict], bool]:
    """Add ``ticker`` with the hunt's pivot frozen as the judged pivot.

    The session's copy, when the scan page has one, is authoritative for membership, so
    the entry is added there and merged with the disk copy before saving, the way the
    scan page persists. Returns ``(merged entries, added)``; ``added`` is False when the
    name was already present, and nothing is written then."""
    import datetime as _dt
    base = list(session_entries) if session_entries is not None else load_watchlist(cache.WATCHLIST_JSON)
    if ticker in watchlist_tickers(base):
        return base, False
    entry = make_entry(ticker, pivot, _dt.date.today().isoformat(), "judged",
                       hunt_note(hunt_date, verdict, notes))
    if entry is None:
        return base, False
    merged = merge_frozen_pivots(base + [entry], load_watchlist(cache.WATCHLIST_JSON))
    save_watchlist(cache.WATCHLIST_JSON, merged)
    return merged, True
