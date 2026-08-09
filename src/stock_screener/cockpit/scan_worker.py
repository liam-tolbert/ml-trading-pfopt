"""Background scan worker — the universe scan in a daemon thread, so it (a) kicks off as
soon as ANY cockpit page loads and (b) survives page switches.

Streamlit cancels the running *script* whenever the user navigates or interacts, so a scan
executed inline dies with the run that started it: the old cockpit both started the
multi-minute cold scan only when the scan page rendered, and lost the whole fetch if you
clicked another page mid-download. Here the scan runs in a plain daemon thread that never
touches Streamlit APIs — a page switch kills the script run, not the thread — and the scan
page polls ``snapshot()`` for live progress until the result lands.

One worker per browser session (``get_worker()`` keeps it in ``st.session_state``), so
AppTest sessions stay isolated and a browser refresh starts clean. The actual ``run_scan``
call is serialized process-wide (``_SCAN_SERIAL``) so two sessions of the LAN app (laptop +
phone) can't race yfinance or the CSV price caches with duplicate concurrent downloads.

``scan.run_scan`` is resolved at call time inside the thread, so a test's
``patch.object(scan, "run_scan", ...)`` is honored exactly like the old inline call.
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from collections import deque
from typing import Optional, Tuple

# The app scans the full US common-stock universe with the full 8/8 trend template —
# app.py and the non-scan pages' warm-up must agree on these or they'd start two scans.
DEFAULT_UNIVERSE = "full_us"
DEFAULT_MIN_CRITERIA = 8

_SCAN_SERIAL = threading.Lock()    # process-wide: one real scan at a time, ever

_LOG_TAIL = 14                     # visible download-log window (same as the old UI deque)
_PRICE_PREFIX = "Prices · "        # run_scan's fetch-phase progress label prefix


class ScanWorker:
    """State machine: idle → running → done|error, re-armed by ``request_rescan``.

    Every mutable field sits behind ``_lock``; the thread writes only plain Python state
    (never Streamlit elements), which is what makes it immune to script-run cancellation.
    """

    def __init__(self, universe: str = DEFAULT_UNIVERSE,
                 min_criteria: int = DEFAULT_MIN_CRITERIA) -> None:
        self.universe = universe
        self.min_criteria = min_criteria
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._generation = 1            # request_rescan bumps it → new key → new run
        self._pending_force = False     # full-2y-re-download flag for the NEXT run
        self._status = "idle"           # idle | running | done | error
        self._result = None
        self._result_key = None         # the key the current result/error belongs to
        self._error: Optional[str] = None
        self._progress: Tuple[int, int, str] = (0, 0, "starting")
        self._log: deque = deque(maxlen=_LOG_TAIL)
        self._started_at = float("-inf")   # monotonic clock at thread start (anchors wait())

    def _key(self):
        return (self.universe, int(self.min_criteria), int(self._generation))

    # ---- API for script runs ---------------------------------------------- #
    def ensure_started(self) -> None:
        """Start a scan for the current key unless one is running or already landed.
        A failed run does NOT auto-retry (its error sticks to the key — an auto-retry
        would hammer yfinance in a rerun loop); the page's Retry button goes through
        ``request_rescan`` for a fresh key."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._status in ("done", "error") and self._result_key == self._key():
                return
            self._start_locked()

    def request_rescan(self, force: bool = False) -> None:
        """Invalidate the current result (new generation) and start a fresh run.
        ``force=True`` = the Advanced full 2-year re-download. A mid-flight run can't be
        cancelled (yfinance has no abort) — the bumped generation makes its result land
        stale, and the page's ``ensure_started`` starts the fresh run when it finishes."""
        with self._lock:
            self._generation += 1
            self._pending_force = self._pending_force or force
            if self._thread is None or not self._thread.is_alive():
                self._start_locked()

    def result_if_ready(self):
        with self._lock:
            ready = self._status == "done" and self._result_key == self._key()
            return self._result if ready else None

    def wait(self, grace: float = 3.0):
        """Poll for the result, but only within ``grace`` seconds OF THE RUN'S START:
        a run that began moments ago (fresh cache / test fake) completes without the
        page ever flashing the progress view, while a rerun during a long cold scan
        falls straight through to it. Returns the ScanResult or None."""
        while True:
            res = self.result_if_ready()
            if res is not None:
                return res
            with self._lock:
                erred = self._status == "error" and self._result_key == self._key()
                remaining = grace - (time.monotonic() - self._started_at)
            if erred or remaining <= 0:
                return None
            time.sleep(min(0.05, remaining))

    def snapshot(self) -> dict:
        with self._lock:
            done, total, label = self._progress
            return {"status": self._status, "done": done, "total": total, "label": label,
                    "log": list(self._log), "error": self._error}

    # ---- internals --------------------------------------------------------- #
    def _start_locked(self) -> None:
        key, force = self._key(), self._pending_force
        self._pending_force = False
        self._status = "running"
        self._error = None
        self._progress = (0, 0, "starting")
        self._log.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, args=(key, force),
                                        name="sepa-scan", daemon=True)
        self._thread.start()

    def _on_progress(self, done: int, total: int, label: str) -> None:
        with self._lock:
            self._progress = (int(done), int(total), str(label))
            if str(label).startswith(_PRICE_PREFIX):
                self._log.append(f"Downloading {str(label)[len(_PRICE_PREFIX):]}")

    def _run(self, key, force: bool) -> None:
        try:
            with _SCAN_SERIAL:
                from . import scan              # attribute lookup at call time → patchable
                res = scan.run_scan(universe=key[0],
                                    cfg=scan.ScanConfig(min_criteria=key[1]),
                                    force=force, progress=self._on_progress)
            with self._lock:
                self._status, self._result, self._result_key = "done", res, key
        except Exception:
            with self._lock:
                self._status, self._result_key = "error", key
                self._error = traceback.format_exc()


def get_worker() -> ScanWorker:
    """The session's worker, created on first touch (any page)."""
    import streamlit as st
    w = st.session_state.get("_scan_worker")
    if not isinstance(w, ScanWorker):
        w = ScanWorker()
        st.session_state["_scan_worker"] = w
    return w


def autostart() -> None:
    """Kick the default scan from a NON-scan page, so it's warming while the user reads
    Positions/Journal/Guide. Inert under the test harness: page AppTests don't patch
    ``run_scan``, so a background scan there would hit the real network — the
    ``streamlit.testing`` import is the tell that we're inside one."""
    if "streamlit.testing.v1" in sys.modules:
        return
    try:
        get_worker().ensure_started()
    except Exception:
        pass                                    # warm-up is best-effort, never a page crash
