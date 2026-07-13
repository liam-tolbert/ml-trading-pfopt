"""Pure helpers for the cockpit watchlist (no Streamlit — unit-testable).

The watchlist is an ordered list of ENTRY DICTS — ``{ticker, judged_pivot, date_added,
pivot_source, note}`` — persisted to JSON so it survives between app runs. ``judged_pivot``
is the FROZEN trigger level: the detected pivot drifts with every scan, so the level in
force is recorded once (when the user judges it on the chart, ``pivot_source="judged"``,
or by the nightly EOD trigger check on first sight, ``pivot_source="auto"``) and stays put.
``date_added`` is the date of the entry's current pivot decision (add / re-freeze /
auto-freeze all stamp it). Legacy files holding a bare JSON array of ticker strings load
as unfrozen entries (migration is read-time; the file is rewritten on the first mutation).

Also here: the two downloadable CSV builders (decision list + long-format OHLCV dump) and
the .txt ticker-list parser.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

WATCHLIST_KEYS = ("ticker", "judged_pivot", "date_added", "pivot_source", "note")
PIVOT_SOURCES = ("judged", "auto")


def make_entry(ticker, judged_pivot=None, date_added=None, pivot_source=None,
               note: str = "") -> Optional[dict]:
    """Normalize one watchlist entry; returns ``None`` when the ticker is blank (invalid).

    The pivot is coerced with ``float()`` (an ``np.float64`` would make ``json.dumps``
    raise inside :func:`save_watchlist`'s swallow-all — a silent no-persist) and rounded
    to cents; non-positive/NaN/garbage pivots become ``None``. ``pivot_source`` is kept
    only when it names a known source AND a pivot is actually set — an unfrozen entry
    always carries ``pivot_source=None``.
    """
    sym = str(ticker or "").strip().upper()
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
    unfrozen entries). Best-effort: a write failure is swallowed (the in-session list
    stays authoritative)."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_coerce_entries(entries)), encoding="utf-8")
    except Exception:
        pass


def load_watchlist(path) -> List[dict]:
    """Load persisted watchlist entries, de-duped in first-seen order. A legacy file (a
    bare JSON array of ticker strings) is MIGRATED element-wise to unfrozen entry dicts —
    in memory only; the file is rewritten on the first mutation. Returns ``[]`` when the
    file is missing, unreadable, corrupt, or not a JSON list — a bad file never breaks
    app startup. Never raises, never writes."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return _coerce_entries(data)


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

    Names absent from ``candidates`` (e.g. a stale entry from another universe) still
    appear as a ticker-only row, so nothing the user picked is silently dropped. The
    frozen level is exported as ``judged_pivot`` because ``candidates`` already carries a
    scan ``pivot`` column — the two are different numbers by design.
    """
    ents = _coerce_entries(entries)
    tickers = [e["ticker"] for e in ents]
    meta = {e["ticker"]: e for e in ents}

    if candidates is None or len(candidates) == 0 or "ticker" not in candidates.columns:
        rows = pd.DataFrame({"ticker": tickers})
    else:
        present = [t for t in tickers if t in set(candidates["ticker"])]
        if present:
            rows = (candidates[candidates["ticker"].isin(present)]
                    .set_index("ticker").reindex(present).reset_index())
            if columns:
                rows = rows[[c for c in columns if c in rows.columns]]
        else:
            rows = pd.DataFrame({"ticker": tickers})
        missing = [t for t in tickers if t not in present]
        if missing and "ticker" in rows.columns:
            rows = pd.concat([rows, pd.DataFrame({"ticker": missing})], ignore_index=True)

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
