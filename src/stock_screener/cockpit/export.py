"""Pure helpers for the cockpit watchlist export (no Streamlit — unit-testable).

The watchlist is just an ordered list of tickers the user has clicked together. These
build the two downloadable CSVs: a decision list (the tickers plus their scan columns,
in the user's chosen order) and a long-format OHLCV dump for those names.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd


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
