"""On-disk cache locations and staleness helpers for the cockpit.

Everything lives under ``data/cockpit/``, gitignored like the rest of ``data/``. Prices
are cached one parquet per ticker, so the daily scan is fast after the first run.
Staleness is file mtime age in days.
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
TRIGGERS_DIR = CACHE_DIR / "triggers"           # trigger reports + entry/sell plans
LOGS_DIR = CACHE_DIR / "logs"                   # dated run logs, pruned by runlog.RETENTION_DAYS
LAST_SCAN_PKL = CACHE_DIR / "last_scan.pkl"     # newest completed ScanResult (scan_worker)
BREADTH_CSV = CACHE_DIR / "breadth.csv"         # one row per settled session (breadth_store)

# MUST be bumped whenever the persisted scan dict changes shape. It lives here, not in
# scan_worker, because the weekend hunt reads the same pickle. With two copies, the writer
# could be bumped while the reader still accepted the old shape.
SCAN_PERSIST_VERSION = 1


def ensure_dirs() -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    TRIGGERS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def age_days(path: Path) -> float:
    """Age of a file in days, or +inf if it does not exist."""
    try:
        return (time.time() - path.stat().st_mtime) / 86400.0
    except (FileNotFoundError, OSError):
        return float("inf")
