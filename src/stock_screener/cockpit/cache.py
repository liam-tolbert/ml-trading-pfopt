"""On-disk cache locations + staleness helpers for the cockpit.

Everything lives under ``data/cockpit/`` (gitignored like the rest of ``data/``).
Prices are cached one parquet per ticker so the daily scan is fast after the first
run; staleness is measured by file mtime in days.
"""
from __future__ import annotations

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "cockpit"
PRICES_DIR = CACHE_DIR / "prices"
FUNDAMENTALS_DIR = CACHE_DIR / "fundamentals"
EDGAR_DIR = CACHE_DIR / "edgar"                 # SEC XBRL backfill cache (per-ticker JSON)
WATCHLIST_JSON = CACHE_DIR / "watchlist.json"   # persisted watchlist (entry dicts, across sessions)
TRIGGERS_DIR = CACHE_DIR / "triggers"           # trigger reports + run log (half-hourly)
TICKERS_TXT = ROOT / "data" / "tickers.txt"


def ensure_dirs() -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    TRIGGERS_DIR.mkdir(parents=True, exist_ok=True)


def age_days(path: Path) -> float:
    """Age of a file in days, or +inf if it does not exist."""
    try:
        return (time.time() - path.stat().st_mtime) / 86400.0
    except (FileNotFoundError, OSError):
        return float("inf")
