"""Pure helpers for the cockpit watchlist (no Streamlit — unit-testable).

The watchlist is an ordered list of ENTRY DICTS — ``{ticker, judged_pivot, date_added,
pivot_source, note}`` — persisted to JSON. ``judged_pivot`` is the FROZEN trigger level:
the detected pivot drifts with every scan, so the level is recorded once (user-judged,
``pivot_source="judged"``, or auto-frozen by the EOD check, ``pivot_source="auto"``) and
stays put. ``date_added`` stamps the current pivot decision. Legacy files (a bare JSON array
of ticker strings) migrate to unfrozen entries at read time; rewritten on first mutation.

Also here: the two CSV builders (decision list + long-format OHLCV dump) and the .txt
ticker-list parser.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

PIVOT_SOURCES = ("judged", "auto")


def make_entry(ticker, judged_pivot=None, date_added=None, pivot_source=None,
               note: str = "") -> Optional[dict]:
    """Normalize one watchlist entry; returns ``None`` when the ticker is blank (invalid).

    The pivot is coerced with ``float()`` (an ``np.float64`` would make ``json.dumps`` raise
    inside :func:`save_watchlist`'s swallow-all — a silent no-persist) and rounded to cents;
    non-positive/NaN/garbage pivots become ``None``. ``pivot_source`` is kept only when it
    names a known source AND a pivot is set — an unfrozen entry carries ``pivot_source=None``.

    Tickers adopt the yfinance dash convention (``BRK.B`` → ``BRK-B``, mirroring
    ``data_feed.normalize`` without importing it) so watchlist entries always match the
    scan-payload keys — a dotted .txt-upload add would otherwise never find its data. This
    single choke point also heals legacy dotted ``watchlist.json`` entries at load time.
    """
    sym = str(ticker or "").strip().upper().replace(".", "-")
    if not sym:
        return None
    pivot = None
    try:
        p = float(judged_pivot)
        if p > 0:                                       # NaN fails this test too
            pivot = round(p, 2)
    except (TypeError, ValueError):
        pivot = None
    src = pivot_source if (pivot is not None and pivot_source in PIVOT_SOURCES) else None
    return {"ticker": sym, "judged_pivot": pivot,
            "date_added": str(date_added) if date_added else None,
            "pivot_source": src, "note": str(note or "")}


def _coerce_entry(obj) -> Optional[dict]:
    """A bare string is a legacy unfrozen entry; a dict goes through :func:`make_entry`;
    anything else is dropped (``None``)."""
    if isinstance(obj, dict):
        return make_entry(obj.get("ticker"), obj.get("judged_pivot"),
                          obj.get("date_added"), obj.get("pivot_source"),
                          obj.get("note", ""))
    if isinstance(obj, str):
        return make_entry(obj)
    return None


def _coerce_entries(entries) -> List[dict]:
    """Normalize + de-dupe (by ticker, first wins) a mixed dict/str sequence."""
    out: List[dict] = []
    seen: set = set()
    for e in entries or []:
        ent = _coerce_entry(e)
        if ent and ent["ticker"] not in seen:
            seen.add(ent["ticker"])
            out.append(ent)
    return out


def watchlist_tickers(entries: Sequence) -> List[str]:
    """Ordered, de-duped ticker projection of a watchlist. Tolerates mixed dict/str input
    (a half-migrated caller or an old test seed never crashes a consumer)."""
    return [e["ticker"] for e in _coerce_entries(entries)]


def save_watchlist(path, entries: Sequence) -> None:
    """Persist the watchlist as a JSON array of entry dicts (strings are coerced to
    unfrozen entries). The write is ATOMIC: serialized to a pid-suffixed sibling temp
    file, then ``os.replace``'d over the target — a crash mid-write can never leave a
    truncated file for :func:`load_watchlist` to silently read back as ``[]``, and the
    app/eod_trigger writers can't see each other's half-written bytes. Best-effort: a
    failure is swallowed (the in-session list stays authoritative) and the existing
    file is left intact."""
    tmp = None
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")  # unique per process — no collisions
        tmp.write_text(json.dumps(_coerce_entries(entries)), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        except Exception:
            pass


def load_watchlist(path) -> List[dict]:
    """Load persisted watchlist entries, de-duped in first-seen order. A legacy file (a bare
    JSON array of ticker strings) is migrated to unfrozen entry dicts in memory only.
    Returns ``[]`` when the file is missing, unreadable, corrupt, or not a JSON list. Never
    raises, never writes."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return _coerce_entries(data)


def merge_frozen_pivots(primary: Sequence, donor: Sequence) -> List[dict]:
    """Lost-update-safe merge of two watchlist copies (pure — no I/O).

    ``primary`` is authoritative for membership, order, notes, and any pivot it has
    frozen itself; the ONLY thing taken from ``donor`` is a frozen pivot
    (``judged_pivot``/``date_added``/``pivot_source``) for a primary entry that is
    still unfrozen. Both sides go through the usual entry coercion.

    Both directions of the app <-> eod_trigger race use it just before saving:
    the app persists ``merge(session, disk)`` so its stale session copy can't clobber
    pivots the half-hourly trigger job froze meanwhile; the trigger persists
    ``merge(disk_now, frozen_copies)`` so entries the user removed or 📌-re-froze
    during its slow fetch window stay removed/judged, and its auto pivots land only
    on entries still unfrozen on disk.
    """
    out = _coerce_entries(primary)
    frozen = {e["ticker"]: e for e in _coerce_entries(donor)
              if e["judged_pivot"] is not None}
    for ent in out:
        d = frozen.get(ent["ticker"])
        if ent["judged_pivot"] is None and d is not None:
            ent["judged_pivot"] = d["judged_pivot"]
            ent["date_added"] = d["date_added"]
            ent["pivot_source"] = d["pivot_source"]
    return out


def parse_ticker_list(text: str) -> List[str]:
    """Parse an uploaded ticker list: split on commas AND any whitespace/newlines,
    upper-case, drop blanks, and de-duplicate while keeping first-seen order.

    So ``"aapl, msft\\nnvda,, tsla"`` -> ``["AAPL", "MSFT", "NVDA", "TSLA"]``.
    """
    seen: dict = {}                                      # ordered set (py3.7+ dict order)
    for token in (text or "").replace(",", " ").split():
        sym = token.strip().upper()
        if sym:
            seen.setdefault(sym, None)
    return list(seen)


def watchlist_list_csv(candidates: Optional[pd.DataFrame], entries: Sequence,
                       columns: Optional[Sequence[str]] = None) -> bytes:
    """The shortlist with its decision columns, in watchlist order, plus the frozen-pivot
    metadata columns (``judged_pivot``, ``date_added``, ``pivot_source``, ``note`` —
    empty for unfrozen entries). ``entries`` may be entry dicts and/or bare tickers.

    Names absent from ``candidates`` still appear as a ticker-only row, so nothing the user
    picked is dropped. Exported as ``judged_pivot`` because ``candidates`` already carries a
    scan ``pivot`` column — the two are different numbers by design.
    """
    ents = _coerce_entries(entries)
    tickers = [e["ticker"] for e in ents]
    meta = {e["ticker"]: e for e in ents}

    if candidates is None or len(candidates) == 0 or "ticker" not in candidates.columns:
        rows = pd.DataFrame({"ticker": tickers})
    else:
        # One row per watchlist entry IN watchlist order: reindex over ALL tickers so a stale
        # (not-in-scan) name stays IN PLACE as a ticker-only NaN row, instead of being appended
        # at the end (which broke the documented "in the order you added them" ordering).
        rows = (candidates.drop_duplicates("ticker").set_index("ticker")
                .reindex(tickers).reset_index())
        if columns:
            rows = rows[[c for c in columns if c in rows.columns]]

    for col in ("judged_pivot", "date_added", "pivot_source", "note"):
        rows[col] = [meta.get(t, {}).get(col) for t in rows["ticker"]]
    return rows.to_csv(index=False).encode("utf-8")


def watchlist_ohlcv_csv(tickers: Sequence[str], payloads: Dict[str, dict]) -> bytes:
    """Long-format daily OHLCV for every watchlisted name present in ``payloads``,
    stacked with leading Date + Ticker columns. Returns ``b""`` if none are present."""
    frames: List[pd.DataFrame] = []
    for t in dict.fromkeys(tickers):
        payload = payloads.get(t)
        df = payload.get("df") if payload else None
        if df is None or len(df) == 0:
            continue
        d = df.reset_index()
        d = d.rename(columns={d.columns[0]: "Date"})      # the former index (a DatetimeIndex)
        d.insert(1, "Ticker", t)
        frames.append(d)
    if not frames:
        return b""
    return pd.concat(frames, ignore_index=True).to_csv(index=False).encode("utf-8")
