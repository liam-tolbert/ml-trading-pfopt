"""Scheduled universe screen: step 2 of ``cockpit-eod.timer`` (16:20 ET weekdays).

    python src/stock_screener/cockpit/screen_job.py [--universe full_us] [--min-criteria 8]

Runs the full chain of SEPA filters over the scan universe: all 8 Trend Template checks,
RS percentile, Step-2 fundamentals for the stocks that pass, VCP tiers and entry levels.
The result goes into the store shared by the whole process (``scan_worker._STORE``),
which saves it to ``last_scan.pkl``. That file is the scan table: the app shows it and
rewrites it only on an explicit Re-scan.

Screening runs under systemd, not on a thread inside the Streamlit container. Such a
thread is invisible to ``systemctl list-timers`` and dies with the container, so a deploy
that restarts the container after the scheduled time loses that day's screen.
``refresh_job.py`` updates the saved prices; this job updates the candidate list.

This is the second ``ExecStart`` of the ``cockpit-eod`` oneshot unit. systemd starts it
only after step 1 (``refresh_job.py --scope universe``) exits 0, however long that takes,
and skips it if step 1 failed. After step 1 every read here comes from the saved files
(``_cache_settled``: no trading day has passed, so no new daily bar can exist), so the
run uses CPU only. It MUST NOT run before or during step 1. Before, it screens
yesterday's bars. During, it re-downloads what step 1 has not reached yet while both
containers call yfinance at once.

Screening only: this places no orders and touches no watchlist state.
After the screen it saves the paper account's equity for the weekend hunt (``write_account``).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:                       # so `from src.X import ...` resolves
    sys.path.insert(0, str(ROOT))

from src.stock_screener.cockpit import breadth_store, cache, runlog, scan  # noqa: E402
from src.stock_screener.cockpit.scan_worker import (  # noqa: E402
    _STORE, DEFAULT_MIN_CRITERIA, DEFAULT_UNIVERSE)

_LOG = runlog.get_logger("screen")


def run_screen(universe: str = DEFAULT_UNIVERSE,
               min_criteria: int = DEFAULT_MIN_CRITERIA, store=None) -> dict:
    """Screen ``universe`` and publish the result. Returns ``{scanned, passed, candidates,
    errors, elapsed}``; a scan failure raises.

    The key MUST be the one the app reads, ``(universe, min_criteria)``, or the app never
    sees the result. ``store.put`` writes a temp file and renames it with
    ``os.replace``, so no reader sees a half-written file. The app's ResultStore
    re-reads ``last_scan.pkl`` when the file's modified time changes: that is how a
    result from this short-lived container reaches the Streamlit process, which keeps
    running."""
    store = _STORE if store is None else store
    _LOG.info("screen starting: %s, min_criteria=%d", universe, min_criteria)
    t0 = time.time()
    res = scan.run_scan(universe=universe, cfg=scan.ScanConfig(min_criteria=min_criteria))
    store.put((universe, min_criteria), res)
    # `candidates` is a DataFrame and MUST NOT go through `or []`: it has no truth value,
    # so that raises ValueError. The unit would then fail after store.put already
    # saved a good scan.
    cand = getattr(res, "candidates", None)
    out = {"scanned": getattr(res, "n_scanned", None),
           "passed": getattr(res, "n_passed", None),
           "candidates": 0 if cand is None else int(len(cand)),
           "errors": len(getattr(res, "errors", None) or []),
           "elapsed": round(time.time() - t0, 1)}
    # The breadth history (the share of stocks in stage 2, new highs, new lows) gets one
    # row per scheduled screen, from settled closes. Only this job adds rows: an in-app
    # scan during market hours would write a row from prices that may still change.
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


def write_account(path: Optional[Path] = None) -> bool:
    """Write the paper account's equity to ``path`` (default ``cache.ACCOUNT_JSON``) as
    ``{"equity": float, "as_of": ISO timestamp}``. The hunt PC copies it with the scan and
    sizes a full position from it.

    Returns False, leaving any existing file as it was, when the account cannot be read.
    That MUST NOT fail the screen."""
    path = cache.ACCOUNT_JSON if path is None else path
    try:
        from src.stock_screener.cockpit import trade      # alpaca-py stays out of the import
        equity = float(trade.fetch_account_summary()["equity"])
    except Exception as e:
        _LOG.warning("account equity not written: %s", e)
        return False
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"equity": equity,
                               "as_of": _dt.datetime.now().isoformat(timespec="seconds")}),
                   encoding="utf-8")
    os.replace(tmp, path)
    return True


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
    write_account()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
