"""Background scan worker: the universe scan runs in a daemon thread, so it starts as soon
as any cockpit page loads and survives page switches.

Streamlit cancels the running *script* whenever the user navigates or interacts, so a scan
run inline dies with the run that started it. The thread never touches Streamlit APIs: a
page switch kills the script run, not the thread. The scan page polls ``snapshot()`` for
live progress until the result lands.

One worker per browser session (``get_worker()`` keeps it in ``st.session_state``), so
AppTest sessions stay isolated and a browser refresh starts clean. The ``run_scan`` call is
serialized process-wide (``_SCAN_SERIAL``), so two sessions of the LAN app (laptop and
phone) can't race yfinance or the CSV price caches with duplicate downloads.

This process schedules nothing, and page interaction never refreshes: entering or clicking
a page serves the stored result and downloads nothing. Price freshness belongs to
``cockpit-refresh.timer`` (09:30, :00/:30 to 15:30, 16:10 ET), which tops up the watchlist
plus held names, and to ``cockpit-eod.timer`` (16:20 ET), which sweeps the whole universe
and then screens it. Both run from one-shot containers, so they are visible in
``systemctl list-timers`` and cannot die with this container. The only in-process network
paths are the true cold start and the explicit Re-scan / full-re-download buttons.

``scan.run_scan`` MUST be resolved at call time inside the thread, so a test's
``patch.object(scan, "run_scan", ...)`` is honored.
"""
from __future__ import annotations

import os
import pickle
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .cache import (LAST_SCAN_PKL as _LAST_SCAN_PKL,
                    SCAN_PERSIST_VERSION as _PERSIST_VERSION)

# The full US common-stock universe on the full 8/8 trend template. app.py and the
# non-scan pages' warm-up MUST agree on these, or they start two scans.
DEFAULT_UNIVERSE = "full_us"
DEFAULT_MIN_CRITERIA = 8

_SCAN_SERIAL = threading.Lock()    # process-wide: one real scan at a time

# The last completed ScanResult is pickled so a server restart serves it instantly; the
# store itself is process memory. ``get`` re-reads the file whenever its mtime advances.
# Any load failure (missing, corrupt, old shape, different key) falls back to a cold
# scan. The path and version come from cache.py because the weekend hunt reads this file.


def _testing() -> bool:
    """True inside the AppTest harness. Page and app AppTests patch ``scan.run_scan`` per
    test and rely on per-session isolation, so the process-wide store MUST stay inert
    there, or results leak across tests. The tell is sticky for the whole test process,
    so unit tests inject a store explicitly."""
    return "streamlit.testing.v1" in sys.modules


@dataclass
class StoreEntry:
    result: object          # ScanResult; consumers MUST treat it as immutable
    completed_wall: float   # time.time()      — "data as of HH:MM" display
    completed_mono: float   # time.monotonic() — staleness / throttle / adopt ordering


class ResultStore:
    """Process-wide store of the last completed scan, keyed ``(universe, min_criteria)``.
    Not keyed by generation: generations are per-session, the store is per-process.
    Streamlit-free and clock-injectable, so unit tests need no browser session or real
    sleeps."""

    def __init__(self, clock=time.monotonic, persist_path=None) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[tuple, StoreEntry] = {}
        self._clock = clock
        self._persist_path = persist_path       # None (unit tests) = no disk I/O at all
        self._seen_mtime = 0.0                  # newest last_scan.pkl mtime already adopted

    def get(self, key) -> Optional[StoreEntry]:
        with self._lock:
            self._sync_locked(key)
            return self._entries.get(key)

    def _sync_locked(self, key) -> None:
        """Adopt the persisted scan for ``key`` when the file's mtime has advanced and its
        scan is newer than the one held (caller holds ``_lock``).

        This store is process memory, but ``last_scan.pkl`` is shared state:
        ``cockpit-eod`` (step 2) rewrites it from a one-shot container while the Streamlit
        process runs for days. Freshness MUST be decided by mtime, not by whether the file
        was ever loaded. A load-once guard hides every scheduled screen until the app
        restarts.

        Re-reading this store's own ``put`` is harmless. The disk copy is adopted only when
        its ``completed_wall`` is strictly newer, so an in-memory entry, which carries a real
        ``completed_mono``, is never replaced by its own -inf copy."""
        if self._persist_path is None:
            return
        try:
            mtime = self._persist_path.stat().st_mtime
        except OSError:
            return
        if mtime <= self._seen_mtime:
            return
        self._seen_mtime = mtime                # even on failure: don't re-read a bad file
        loaded = self._read_locked(key)
        cur = self._entries.get(key)
        if loaded is not None and (cur is None
                                   or loaded.completed_wall > cur.completed_wall):
            self._entries[key] = loaded

    def _read_locked(self, key) -> Optional[StoreEntry]:
        """Read the pickled scan for ``key`` (caller holds ``_lock``); no mutation. Returns
        a StoreEntry, or None when the file is unreadable, another version or another
        key.

        ``completed_wall`` is the original scan time. ``completed_mono`` is -inf, because
        monotonic clocks don't survive a restart. -inf also keeps an in-flight run's
        adopt-while-queued check from adopting it, so a run that has already started does
        its own scan."""
        try:
            with open(self._persist_path, "rb") as f:
                d = pickle.load(f)
            if (isinstance(d, dict) and d.get("version") == _PERSIST_VERSION
                    and tuple(d.get("key") or ()) == tuple(key)
                    and d.get("result") is not None):
                return StoreEntry(d["result"], float(d.get("completed_wall") or 0.0),
                                  float("-inf"))
        except Exception:
            pass
        return None

    def put(self, key, result) -> StoreEntry:
        ent = StoreEntry(result, time.time(), self._clock())
        with self._lock:
            self._entries[key] = ent
        if self._persist_path is not None:
            self._persist(key, ent)             # outside the lock — pickling isn't cheap
        return ent

    def _persist(self, key, ent: StoreEntry) -> None:
        """Pickle ``ent`` atomically (tmp + ``os.replace``), best effort. Never raises: a
        failed persist costs one cold scan after the next restart. ``completed_mono`` is
        not persisted because it is process-relative."""
        tmp = None
        try:
            path = self._persist_path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"version": _PERSIST_VERSION, "key": tuple(key),
                             "result": ent.result,
                             "completed_wall": ent.completed_wall}, f)
            os.replace(tmp, path)
        except Exception:
            pass
        finally:
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def now(self) -> float:
        """The store's clock. Run birth times for the adopt check MUST come from here, never
        from raw ``time.monotonic()``, so an injected test clock is never compared with
        real time."""
        return self._clock()


# The production singleton, inert under AppTest. It survives page and session churn in
# memory, and a server restart via the last-scan pickle.
_STORE = ResultStore(persist_path=_LAST_SCAN_PKL)


_PRICE_PREFIX = "Prices · "        # run_scan's fetch-phase progress label prefix

# What the progress bar calls each phase. "cache" is a price label whose detail says
# "cached": a zero-network serve MUST NOT read as "Downloading". data_feed's label
# strings are pinned by tests, so the phase is classified here instead.
_PHASE_LABELS = {"cache": "Reading cache", "fetch": "Downloading", "screen": "Screening"}


class ScanWorker:
    """State machine: idle → running → done|error, re-armed by ``request_rescan``.

    Every mutable field sits behind ``_lock``. The thread MUST write only plain Python
    state, never Streamlit elements: that is what makes it immune to script-run
    cancellation.
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
        one is running or has already landed.

        A failed run MUST NOT auto-retry: its error sticks to the key, because a retry
        would hammer yfinance in a rerun loop. The page's Retry button goes through
        ``request_rescan`` for a fresh key. Page interaction never starts a background
        refresh; the only network paths are the true cold start (no result anywhere) and
        an explicit Re-scan."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._pending_force:
                # A forced run outranks adoption. Another session's result MUST NOT
                # satisfy it, and a flag left armed would fire in a later run as a
                # surprise full re-download.
                self._start_locked(adopt_ok=False)
                return
            store = self._store_or_none()
            if store is not None:
                ent = store.get(self._key2())
                if ent is not None and (self._completed_at is None
                                        or ent.completed_wall > self._completed_at):
                    self._adopt_locked(ent)               # a newer scan landed
            if self._status in ("done", "error") and self._result_key == self._key():
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
        ``force=True`` is the Advanced full 2-year re-download.

        A mid-flight run can't be cancelled; yfinance has no abort. The bumped generation
        makes its result land stale, and the page's ``ensure_started`` starts the fresh
        run when it finishes. A run started here never adopts another session's result
        (``adopt_ok`` False)."""
        with self._lock:
            self._generation += 1
            self._pending_force = self._pending_force or force
            if self._thread is None or not self._thread.is_alive():
                self._start_locked(adopt_ok=False)

    def result_if_ready(self):
        with self._lock:
            ready = self._status == "done" and self._result_key == self._key()
            return self._result if ready else None

    def latest(self):
        """The newest result this session can serve now, without waiting.

        That is the current-key done result, else, with the store active, whatever
        ``_result`` holds, even mid-refresh (stale-while-refresh, possibly adopted from
        another session). None on a true cold start. Under the AppTest tell it is also
        None until the current key is done: the store is inert there, so the app falls
        through to ``wait()`` and blocks for the run's own result. The AppTests (memo-hit
        counts, force ordering) are built on that flow, so this divergence MUST stay."""
        with self._lock:
            if self._status == "done" and self._result_key == self._key():
                return self._result
            if self._store_or_none() is None:
                return None
            return self._result

    def wait(self, grace: float = 3.0):
        """Poll for the result until ``grace`` seconds after the run's start. Returns the
        ScanResult, or None on an error or once the grace has run out.

        A run that began moments ago (fresh cache, test fake) completes without the page
        flashing the progress view. A rerun during a long cold scan falls straight
        through to it."""
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
                    "error": self._error}

    # ---- internals --------------------------------------------------------- #
    def _start_locked(self, adopt_ok: bool = True) -> None:
        key, force = self._key(), self._pending_force
        self._pending_force = False
        self._status = "running"
        self._error = None
        self._progress = (0, 0, "starting")
        self._phase = "fetch"
        self._started_at = time.monotonic()          # wait()'s grace anchor: REAL clock
        # The adopt-check birth time comes from the store's clock, captured before the
        # serial-lock wait: a result landing while the run queues MUST count. The
        # resolved store rides along as a thread arg, so _run can't re-resolve a
        # different store than the clock came from.
        store = self._store_or_none()
        run_started = store.now() if store is not None else self._started_at
        self._thread = threading.Thread(target=self._run,
                                        args=(key, force, adopt_ok, run_started, store),
                                        name="sepa-scan", daemon=True)
        self._thread.start()

    def _on_progress(self, done: int, total: int, label: str) -> None:
        label = str(label)
        if label.startswith(_PRICE_PREFIX):
            phase = "cache" if "cached" in label[len(_PRICE_PREFIX):] else "fetch"
        else:
            phase = "screen"
        with self._lock:
            self._progress = (int(done), int(total), label)
            self._phase = phase

    def _run(self, key, force: bool, adopt_ok: bool, run_started: float, store) -> None:
        try:
            with _SCAN_SERIAL:
                ent = store.get(key[:2]) if store is not None else None
                if adopt_ok and ent is not None and ent.completed_mono >= run_started:
                    # Another session's scan landed while this run queued on the serial
                    # lock: adopt it instead of scanning twice.
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
                # _result is kept: a failed refresh keeps serving the stale result via
                # latest(), and only the error banner changes.


def get_worker() -> ScanWorker:
    """The session's worker, created on first touch (any page)."""
    import streamlit as st
    w = st.session_state.get("_scan_worker")
    if not isinstance(w, ScanWorker):
        w = ScanWorker()
        st.session_state["_scan_worker"] = w
    return w


def autostart() -> None:
    """Start the default scan from a non-scan page, so it warms while the user reads
    Positions, Journal or the Guide. Best effort; never raises. Inert under the test
    harness, detected by the ``streamlit.testing`` import: page AppTests don't patch
    ``run_scan``, so a background scan there would hit the real network."""
    if "streamlit.testing.v1" in sys.modules:
        return
    try:
        get_worker().ensure_started()
    except Exception:
        pass                                    # warm-up is best-effort, never a page crash
