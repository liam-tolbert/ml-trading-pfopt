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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# The app scans the full US common-stock universe with the full 8/8 trend template —
# app.py and the non-scan pages' warm-up must agree on these or they'd start two scans.
DEFAULT_UNIVERSE = "full_us"
DEFAULT_MIN_CRITERIA = 8

_SCAN_SERIAL = threading.Lock()    # process-wide: one real scan at a time, ever

# The one-per-this-many-seconds cap on BACKGROUND refreshes (user spec: at most one
# every half hour, process-wide). Explicit Re-scan / full-re-download always bypass.
REFRESH_TTL_SECONDS = 30 * 60


def _testing() -> bool:
    """True inside the AppTest harness. Page/app AppTests patch ``scan.run_scan`` per
    test and rely on per-session isolation — the process-wide store must stay inert
    there or results would leak across tests (the tell is sticky for the whole test
    process, so unit tests inject a store explicitly instead)."""
    return "streamlit.testing.v1" in sys.modules


@dataclass
class StoreEntry:
    result: object          # ScanResult — treated as IMMUTABLE by every consumer
    completed_wall: float   # time.time()      — "data as of HH:MM" display
    completed_mono: float   # time.monotonic() — staleness / throttle / adopt ordering


class ResultStore:
    """Process-wide last-completed-scan store, keyed ``(universe, min_criteria)`` —
    deliberately NOT by generation (generations are per-session identities; the store is
    process identity). Streamlit-free and clock-injectable so unit tests run without a
    browser session or real sleeps."""

    def __init__(self, ttl: float = REFRESH_TTL_SECONDS, clock=time.monotonic) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[tuple, StoreEntry] = {}
        self._last_claim: Dict[tuple, float] = {}
        self.ttl = ttl
        self._clock = clock

    def get(self, key) -> Optional[StoreEntry]:
        with self._lock:
            return self._entries.get(key)

    def put(self, key, result) -> StoreEntry:
        ent = StoreEntry(result, time.time(), self._clock())
        with self._lock:
            self._entries[key] = ent
        return ent

    def try_claim_refresh(self, key) -> bool:
        """Atomically claim the one-per-TTL background-refresh slot: True iff an entry
        exists, it is at least ``ttl`` old, and nothing claimed within the last ``ttl``.
        The claim is recorded ON START (not completion), so a slow or FAILED refresh
        can't retry-loop — the next attempt opens a full TTL after this one."""
        with self._lock:
            ent = self._entries.get(key)
            if ent is None:
                return False
            now = self._clock()
            if now - ent.completed_mono < self.ttl:
                return False
            last = self._last_claim.get(key)
            if last is not None and now - last < self.ttl:
                return False
            self._last_claim[key] = now
            return True

    def stamp_claim(self, key) -> None:
        """Record a claim without conditions — a MANUAL rescan counts against the
        background throttle, so an auto-refresh doesn't fire minutes after one."""
        with self._lock:
            self._last_claim[key] = self._clock()


_STORE = ResultStore()             # the production singleton (inert under AppTest)

_LOG_TAIL = 14                     # visible download-log window (same as the old UI deque)
_PRICE_PREFIX = "Prices · "        # run_scan's fetch-phase progress label prefix

# What the progress bar calls each phase. "cache" is a price label whose detail says
# "cached" — a zero-network serve used to render as "Downloading SYM: cached (…)", which
# read as a full re-download to the user. The label STRINGS from data_feed stay untouched
# (they're pinned by tests); only this classification/rendering layer changes.
_PHASE_LABELS = {"cache": "Reading cache", "fetch": "Downloading", "screen": "Screening"}


class ScanWorker:
    """State machine: idle → running → done|error, re-armed by ``request_rescan``.

    Every mutable field sits behind ``_lock``; the thread writes only plain Python state
    (never Streamlit elements), which is what makes it immune to script-run cancellation.
    """

    def __init__(self, universe: str = DEFAULT_UNIVERSE,
                 min_criteria: int = DEFAULT_MIN_CRITERIA,
                 store: Optional[ResultStore] = None) -> None:
        self.universe = universe
        self.min_criteria = min_criteria
        self._store = store             # explicit injection (unit tests) beats _STORE
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._generation = 1            # request_rescan bumps it → new key → new run
        self._pending_force = False     # full-2y-re-download flag for the NEXT run
        self._status = "idle"           # idle | running | done | error
        self._result = None
        self._result_key = None         # the key the current result/error belongs to
        self._completed_at: Optional[float] = None   # wall clock of current _result
        self._error: Optional[str] = None
        self._progress: Tuple[int, int, str] = (0, 0, "starting")
        self._phase = "fetch"              # cache | fetch | screen (see _PHASE_LABELS)
        self._log: deque = deque(maxlen=_LOG_TAIL)
        self._started_at = float("-inf")   # monotonic clock at thread start (anchors wait())

    def _key(self):
        return (self.universe, int(self.min_criteria), int(self._generation))

    def _key2(self):
        return (self.universe, int(self.min_criteria))    # the store's key (no generation)

    def _store_or_none(self) -> Optional[ResultStore]:
        if self._store is not None:     # explicit injection always wins (unit tests)
            return self._store
        return None if _testing() else _STORE             # AppTests: store inert

    # ---- API for script runs ---------------------------------------------- #
    def ensure_started(self) -> None:
        """Adopt the newest store result, then start a scan for the current key unless
        one is running or already landed. A failed run does NOT auto-retry (its error
        sticks to the key — an auto-retry would hammer yfinance in a rerun loop); the
        page's Retry button goes through ``request_rescan`` for a fresh key. When the
        adopted/held result is older than ``REFRESH_TTL_SECONDS`` and the store grants
        the one-per-TTL claim, a BACKGROUND refresh starts under a fresh generation —
        ``latest()`` keeps serving the stale result meanwhile."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            store = self._store_or_none()
            if store is not None:
                ent = store.get(self._key2())
                if ent is not None and (self._completed_at is None
                                        or ent.completed_wall > self._completed_at):
                    self._adopt_locked(ent)               # another session's scan landed
            if self._status in ("done", "error") and self._result_key == self._key():
                # The TTL claim also covers the ERROR state: a failed background refresh
                # retries once the window reopens (claim-on-start spaced the attempts).
                # A cold-start error can't loop here — no store entry, claim denied —
                # so the no-auto-retry rule for true failures still holds.
                if store is not None and store.try_claim_refresh(self._key2()):
                    self._generation += 1                 # refresh runs under a fresh key
                    self._start_locked(adopt_ok=True)
                return
            self._start_locked(adopt_ok=True)

    def _adopt_locked(self, ent: StoreEntry) -> None:
        """Take a store entry as this session's done result (caller holds ``_lock``)."""
        self._status, self._result = "done", ent.result
        self._completed_at = ent.completed_wall
        self._result_key = self._key()      # reads as "done for the current key"
        self._error = None

    def request_rescan(self, force: bool = False) -> None:
        """Invalidate the current result (new generation) and start a fresh run.
        ``force=True`` = the Advanced full 2-year re-download. A mid-flight run can't be
        cancelled (yfinance has no abort) — the bumped generation makes its result land
        stale, and the page's ``ensure_started`` starts the fresh run when it finishes.
        A user-forced run is never satisfied by someone else's result (``adopt_ok``
        False), and it stamps the store's refresh claim so an automatic background
        refresh doesn't fire minutes after a manual one."""
        with self._lock:
            self._generation += 1
            self._pending_force = self._pending_force or force
            store = self._store_or_none()
            if store is not None:
                store.stamp_claim(self._key2())
            if self._thread is None or not self._thread.is_alive():
                self._start_locked(adopt_ok=False)

    def result_if_ready(self):
        with self._lock:
            ready = self._status == "done" and self._result_key == self._key()
            return self._result if ready else None

    def latest(self):
        """Newest result this session can serve RIGHT NOW, without waiting: the
        current-key done result, else — with the store active — whatever ``_result``
        holds even mid-refresh (stale-while-refresh; possibly adopted from another
        session). Returns None on a true cold start, and ALSO under the AppTest tell
        while a run is in flight: the store is inert there, so the app falls through to
        ``wait()`` and blocks for the run's own result — the deterministic flow the
        AppTests (memo-hit counts, forces ordering) are built on. Do not "simplify"
        this divergence away."""
        with self._lock:
            if self._status == "done" and self._result_key == self._key():
                return self._result
            if self._store_or_none() is None:
                return None
            return self._result

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
                    "phase": self._phase,
                    "phase_label": _PHASE_LABELS.get(self._phase, "Working"),
                    "as_of": self._completed_at,          # wall clock of current _result
                    "log": list(self._log), "error": self._error}

    # ---- internals --------------------------------------------------------- #
    def _start_locked(self, adopt_ok: bool = True) -> None:
        key, force = self._key(), self._pending_force
        self._pending_force = False
        self._status = "running"
        self._error = None
        self._progress = (0, 0, "starting")
        self._phase = "fetch"
        self._log.clear()
        self._started_at = time.monotonic()
        # _started_at doubles as the run's birth time for _run's adopt check — captured
        # BEFORE the serial-lock wait, or a result landing while we queue wouldn't count.
        self._thread = threading.Thread(target=self._run,
                                        args=(key, force, adopt_ok, self._started_at),
                                        name="sepa-scan", daemon=True)
        self._thread.start()

    def _on_progress(self, done: int, total: int, label: str) -> None:
        label = str(label)
        if label.startswith(_PRICE_PREFIX):
            detail = label[len(_PRICE_PREFIX):]
            # A cache serve is not a download — keep it off the download log entirely and
            # let the bar's phase text ("Reading cache") carry it.
            phase = "cache" if "cached" in detail else "fetch"
            log_line = None if phase == "cache" else f"Downloading {detail}"
        else:
            phase, log_line = "screen", None
        with self._lock:
            self._progress = (int(done), int(total), label)
            self._phase = phase
            if log_line:
                self._log.append(log_line)

    def _run(self, key, force: bool, adopt_ok: bool, run_started: float) -> None:
        try:
            with _SCAN_SERIAL:
                store = self._store_or_none()
                ent = store.get(key[:2]) if store is not None else None
                if adopt_ok and ent is not None and ent.completed_mono >= run_started:
                    # Another session's scan landed while we queued on the serial lock —
                    # adopt it instead of re-scanning (dedups the two-sessions cold start).
                    res, completed_wall = ent.result, ent.completed_wall
                else:
                    from . import scan          # attribute lookup at call time → patchable
                    res = scan.run_scan(universe=key[0],
                                        cfg=scan.ScanConfig(min_criteria=key[1]),
                                        force=force, progress=self._on_progress)
                    completed_wall = (store.put(key[:2], res).completed_wall
                                      if store is not None else time.time())
            with self._lock:
                self._status, self._result, self._result_key = "done", res, key
                self._completed_at = completed_wall
        except Exception:
            with self._lock:
                self._status, self._result_key = "error", key
                self._error = traceback.format_exc()
                # _result deliberately kept: a failed REFRESH keeps serving the stale
                # result via latest(); only the error banner/status line changes.


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
