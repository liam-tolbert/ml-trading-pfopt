"""Pure helpers for the cockpit watchlist. No Streamlit, so they are unit-testable.

The watchlist is an ordered list of entry dicts, ``{ticker, judged_pivot, date_added,
pivot_source, note}``, persisted to JSON. ``judged_pivot`` is the frozen trigger level. The
detected pivot drifts with every scan, so the level is recorded once and stays put: judged
by the user (``pivot_source="judged"``) or auto-frozen by the trigger check
(``pivot_source="auto"``). ``date_added`` stamps the current pivot decision. A legacy file
(a bare JSON array of ticker strings) reads as unfrozen entries and is rewritten on the
next save.

Also here: the two CSV builders (decision list and long-format OHLCV dump) and the .txt
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
    """Normalize one watchlist entry. Returns ``None`` when the ticker is blank.

    The pivot is coerced with ``float()`` and rounded to cents. A numpy scalar such as
    ``np.int64`` or ``np.float32`` would make ``json.dumps`` raise inside
    :func:`save_watchlist`, which swallows the error and persists nothing. Non-positive, NaN
    or unparseable pivots become ``None``. ``pivot_source`` is kept only when it names a
    known source and a pivot is set; an unfrozen entry carries ``pivot_source=None``.

    Tickers take the yfinance dash form (``BRK.B`` → ``BRK-B``), mirroring
    ``data_feed.normalize`` without importing it, so entries match the scan-payload keys. A
    dotted ticker from a .txt upload would never find its data. Every entry passes through
    here, so dotted ``watchlist.json`` entries are healed at load time too.
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
    """The watchlist's tickers, ordered and de-duplicated. Accepts mixed dict/str input, so
    a caller passing bare tickers never crashes a consumer."""
    return [e["ticker"] for e in _coerce_entries(entries)]


def save_watchlist(path, entries: Sequence) -> None:
    """Persist the watchlist as a JSON array of entry dicts. Strings become unfrozen entries.

    The write is atomic: a pid-suffixed sibling temp file, then ``os.replace`` over the
    target. A crash mid-write cannot leave a truncated file that :func:`load_watchlist`
    reads back as ``[]``, and the app and refresh_job never see each other's partial
    writes. Best-effort: a failure is swallowed and the existing file is left intact. The
    in-session list stays authoritative."""
    tmp = None
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")  # per process: writers can't collide
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
    """Lost-update-safe merge of two watchlist copies. Pure, no I/O.

    ``primary`` is authoritative for membership, order, notes, and any pivot it has
    frozen itself. The only thing taken from ``donor`` is a frozen pivot
    (``judged_pivot``/``date_added``/``pivot_source``) for a primary entry that is
    still unfrozen. Both sides go through the usual entry coercion.

    The app and refresh_job both merge just before saving. The app persists
    ``merge(session, disk)``, so its stale session copy can't clobber pivots the
    half-hourly trigger job froze meanwhile. The trigger job persists
    ``merge(disk_now, frozen_copies)``, so entries the user removed or 📌-re-froze during
    its slow fetch stay removed or judged. Its auto pivots land only on entries still
    unfrozen on disk.
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
        # Reindex over all tickers so a name missing from the scan keeps its watchlist place
        # as a ticker-only NaN row. Appended at the end, it would break the watchlist order.
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
