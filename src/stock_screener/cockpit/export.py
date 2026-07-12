"""Pure helpers for the cockpit watchlist export (no Streamlit — unit-testable).

The watchlist is just an ordered list of tickers the user has clicked together. These
build the two downloadable CSVs: a decision list (the tickers plus their scan columns,
in the user's chosen order) and a long-format OHLCV dump for those names — plus
save/load persistence so the list survives between app runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


def save_watchlist(path, tickers: Sequence[str]) -> None:
    """Persist the watchlist (an ordered, de-duped, upper-cased ticker list) to a JSON file.
    Best-effort: a write failure is swallowed (the in-session list stays authoritative)."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        seen = list(dict.fromkeys(
            s for s in (str(t).strip().upper() for t in tickers) if s))
        p.write_text(json.dumps(seen), encoding="utf-8")
    except Exception:
        pass


def load_watchlist(path) -> List[str]:
    """Load a persisted watchlist ticker list, de-duped in first-seen order. Returns ``[]``
    when the file is missing, unreadable, corrupt, or not a JSON list of strings — so a bad
    file never breaks app startup."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[str] = []
    for t in data:
        s = str(t).strip().upper()
        if s and s not in out:
            out.append(s)
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


def watchlist_list_csv(candidates: Optional[pd.DataFrame], tickers: Sequence[str],
                       columns: Optional[Sequence[str]] = None) -> bytes:
    """The shortlist with its decision columns, in ``tickers`` order.

    Names absent from ``candidates`` (e.g. a stale entry from another universe) still
    appear as a ticker-only row, so nothing the user picked is silently dropped.
    """
    tickers = list(dict.fromkeys(tickers))               # dedupe, keep order
    if candidates is None or len(candidates) == 0 or "ticker" not in candidates.columns:
        return pd.DataFrame({"ticker": tickers}).to_csv(index=False).encode("utf-8")

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
