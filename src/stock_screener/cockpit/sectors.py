"""Sector and industry per symbol, from yfinance ``Ticker.info``, cached in one JSON.

The books read stocks by group: most big winners move with their industry, and a group
breaks together. The universe list carries symbols only, so the labels come from Yahoo.
A company's industry rarely changes, so a label is kept ``SECTOR_MAX_AGE_DAYS``; a failed
or empty fetch is remembered for ``SECTOR_RETRY_DAYS`` so a nightly screen doesn't ask
again. Only names that pass the trend template and held names are looked up.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from . import cache

SECTOR_MAX_AGE_DAYS = 180
SECTOR_RETRY_DAYS = 7

# The scan thread and a page's positions read can both write the file in one process.
_LOCK = threading.Lock()


def _path(path) -> Path:
    return Path(path) if path is not None else cache.SECTORS_JSON


def read_all(path=None) -> Dict[str, dict]:
    """The whole cache; ``{}`` when the file is missing or unreadable."""
    try:
        d = json.loads(_path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data: Dict[str, dict], path=None) -> None:
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def _fresh(ent: Optional[dict], now: float) -> bool:
    if not isinstance(ent, dict) or not ent.get("fetched"):
        return False
    days = SECTOR_MAX_AGE_DAYS if ent.get("industry") else SECTOR_RETRY_DAYS
    return now - float(ent["fetched"]) < days * 86400.0


def _clean(v) -> Optional[str]:
    s = str(v).strip() if v is not None else ""
    return s or None


def get_sector(symbol: str, tk=None, *, allow_network: bool = True, path=None,
               now: Optional[float] = None) -> dict:
    """``{sector, industry, fetched}`` for ``symbol``, both labels None when unknown.

    A fresh cache entry is returned as is. Otherwise, with ``allow_network``, ``tk.info``
    is read (``tk`` defaults to ``yfinance.Ticker(symbol)``) and the result, a miss
    included, is cached. Without network a stale entry is returned rather than nothing.
    Never raises."""
    sym = str(symbol or "").strip().upper()
    t = time.time() if now is None else float(now)
    empty = {"sector": None, "industry": None, "fetched": None}
    if not sym:
        return empty
    with _LOCK:
        ent = read_all(path).get(sym)
    if _fresh(ent, t) or not allow_network:
        return {**empty, **ent} if isinstance(ent, dict) else empty
    try:
        if tk is None:
            import yfinance as yf
            tk = yf.Ticker(sym)
        info = tk.info or {}
        new = {"sector": _clean(info.get("sector")),
               "industry": _clean(info.get("industry")), "fetched": t}
    except Exception:
        new = {**empty, "fetched": t}
    try:
        with _LOCK:
            data = read_all(path)
            data[sym] = new
            _write(data, path)
    except Exception:
        pass
    return new
