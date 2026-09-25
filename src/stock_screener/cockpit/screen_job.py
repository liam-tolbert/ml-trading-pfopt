"""Scheduled universe screen: step 2 of ``cockpit-eod.timer`` (16:20 ET weekdays).

    python src/stock_screener/cockpit/screen_job.py [--universe full_us] [--min-criteria 8]

Runs the full SEPA funnel over the scan universe: the 8/8 trend template, RS percentile,
Step-2 fundamentals for passers, VCP tiers and entry levels. The result is published
through the process-wide store, which persists it to ``last_scan.pkl``. That file is the
scan table: the app renders it and rewrites it only on an explicit Re-scan.

Screening runs under systemd, not on a thread inside the Streamlit container. Such a
thread is invisible to ``systemctl list-timers`` and dies with the container, so a deploy
landing after its slot costs that day's screen. ``refresh_job.py`` advances the price
cache; this job advances the candidate list.

This is the second ``ExecStart`` of the ``cockpit-eod`` oneshot unit. systemd starts it
only after step 1 (``refresh_job.py --scope universe``) exits 0, however long that takes,
and skips it if the sweep failed. After the sweep every read here is served from cache
(``_cache_settled``: no session has elapsed, so no new bar can exist), so the run costs
CPU only. It MUST NOT run before or during the sweep. Before, it screens yesterday's bars.
During, it re-fetches what the sweep has not reached yet while both containers hit
yfinance.

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

from src.stock_screener.cockpit import breadth_store, runlog, scan  # noqa: E402
from src.stock_screener.cockpit.scan_worker import (  # noqa: E402
    _STORE, DEFAULT_MIN_CRITERIA, DEFAULT_UNIVERSE)

_LOG = runlog.get_logger("screen")


def run_screen(universe: str = DEFAULT_UNIVERSE,
               min_criteria: int = DEFAULT_MIN_CRITERIA, store=None) -> dict:
    """Screen ``universe`` and publish the result. Returns ``{scanned, passed, candidates,
    errors, elapsed}``; a scan failure raises.

    The key MUST be the one the app reads, ``(universe, min_criteria)``, or the app never
    sees the result. ``store.put`` persists atomically (tmp + ``os.replace``). The app's
    ResultStore re-reads ``last_scan.pkl`` when its mtime advances: that is how a result
    from this one-shot container reaches the long-running Streamlit process."""
    store = _STORE if store is None else store
    _LOG.info("screen starting: %s, min_criteria=%d", universe, min_criteria)
    t0 = time.time()
    res = scan.run_scan(universe=universe, cfg=scan.ScanConfig(min_criteria=min_criteria))
    store.put((universe, min_criteria), res)
    # `candidates` is a DataFrame and MUST NOT be `or []`-ed: it has no truth value, so
    # that raises ValueError. The unit would then fail after store.put had already
    # persisted a good scan.
    cand = getattr(res, "candidates", None)
    out = {"scanned": getattr(res, "n_scanned", None),
           "passed": getattr(res, "n_passed", None),
           "candidates": 0 if cand is None else int(len(cand)),
           "errors": len(getattr(res, "errors", None) or []),
           "elapsed": round(time.time() - t0, 1)}
    # The breadth history gets one settled row per scheduled screen. Only this job
    # appends: an in-app scan mid-session would write a provisional row.
    reg = getattr(res, "regime", None) or {}
    if reg.get("session"):
        try:
            breadth_store.append({"date": reg["session"], "n_scanned": out["scanned"],
                                  "phase2_pct": reg.get("phase2_pct"),
                                  "new_highs": reg.get("new_highs"),
                                  "new_lows": reg.get("new_lows")})
        except Exception as e:
            _LOG.warning("breadth row not written: %s", e)
    _LOG.info("screen done: %s scanned, %s passed 8/8, %d candidates, %d errors, "
              "NH/NL %s/%s, %.1fs", out["scanned"], out["passed"], out["candidates"],
              out["errors"], reg.get("new_highs"), reg.get("new_lows"), out["elapsed"])
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
