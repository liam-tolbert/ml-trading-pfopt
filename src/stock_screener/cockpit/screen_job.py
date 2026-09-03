"""Scheduled universe screen — step 2 of ``cockpit-eod.timer`` (16:20 ET weekdays).

    python src/stock_screener/cockpit/screen_job.py [--universe full_us] [--min-criteria 8]

Runs the full SEPA funnel over the scan universe — 8/8 trend template, RS percentile, Step-2
fundamentals for passers, VCP tiers, entry levels — and publishes the result through the
process-wide store, which persists it to ``last_scan.pkl``. That file IS the scan table: the
app renders it and only ever rewrites it from an explicit Re-scan.

**Why this exists.** Screening used to ride along on a thread inside the Streamlit container
(an in-app scheduler thread). That thread was invisible to ``systemctl list-timers`` and died with
the container, so a deploy landing after its slot silently cost that day's screen. It was
removed when price refreshing moved to systemd; this job is the other half — the piece that
advances the CANDIDATE LIST rather than the price cache.

**Ordering matters, and is now structural.** This is the second ``ExecStart`` of the
``cockpit-eod`` oneshot unit: systemd starts it only once step 1
(``refresh_job.py --scope universe``) has exited 0, however long that took, and skips it
entirely if the sweep failed. With the sweep done every read here is served from cache
(``_cache_settled``: no session has elapsed, so no new bar can exist) and the run costs CPU
only — no network. Run it BEFORE the sweep and it screens yesterday's bars; run it DURING
one and it re-fetches what the sweep has not reached yet while both containers hit
yfinance — which is exactly what a 5-minute timer gap produced on 2026-08-28.

Screening only: this places no orders and touches no watchlist state.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:                       # so `from src.X import ...` resolves
    sys.path.insert(0, str(ROOT))

from src.stock_screener.cockpit import runlog, scan  # noqa: E402
from src.stock_screener.cockpit.scan_worker import (  # noqa: E402
    _STORE, DEFAULT_MIN_CRITERIA, DEFAULT_UNIVERSE)

_LOG = runlog.get_logger("screen")


def run_screen(universe: str = DEFAULT_UNIVERSE,
               min_criteria: int = DEFAULT_MIN_CRITERIA, store=None) -> dict:
    """Screen ``universe`` and publish the result. Returns a small summary dict.

    Published under the SAME key the app reads — ``(universe, min_criteria)`` — or the app
    would never see it. ``store.put`` persists atomically (tmp + ``os.replace``), and the
    app's ResultStore re-reads ``last_scan.pkl`` when its mtime advances, which is what
    makes a result written from THIS one-shot container visible to the long-running
    Streamlit process."""
    store = _STORE if store is None else store
    _LOG.info("screen starting: %s, min_criteria=%d", universe, min_criteria)
    t0 = time.time()
    res = scan.run_scan(universe=universe, cfg=scan.ScanConfig(min_criteria=min_criteria))
    store.put((universe, min_criteria), res)
    # NB: `candidates` is a DataFrame -- never `or []` it. A DataFrame has no truth value,
    # so `df or []` raises ValueError and (as on 2026-08-28) killed the job AFTER store.put
    # had already persisted a perfectly good scan: the table was fine, the unit reported
    # failure. len() alone is safe on both a frame and None-guarded default.
    cand = getattr(res, "candidates", None)
    out = {"scanned": getattr(res, "n_scanned", None),
           "passed": getattr(res, "n_passed", None),
           "candidates": 0 if cand is None else int(len(cand)),
           "errors": len(getattr(res, "errors", None) or []),
           "elapsed": round(time.time() - t0, 1)}
    _LOG.info("screen done: %s scanned, %s passed 8/8, %d candidates, %d errors, %.1fs",
              out["scanned"], out["passed"], out["candidates"], out["errors"],
              out["elapsed"])
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Scheduled universe screen — rebuilds the "
                                             "scan table (places no orders).")
    ap.add_argument("--universe", default=DEFAULT_UNIVERSE)
    ap.add_argument("--min-criteria", type=int, default=DEFAULT_MIN_CRITERIA)
    args = ap.parse_args(argv)
    try:
        out = run_screen(args.universe, args.min_criteria)
        print(f"screened {out['scanned']} -> {out['passed']} pass 8/8 -> "
              f"{out['candidates']} candidates in {out['elapsed']}s "
              f"({out['errors']} errors)")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
