"""Cockpit tests — the background scan worker — result store, adoption, persistence.

Runs standalone (`python tests/cockpit/test_scan_worker.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_scan_store_publish_and_adopt():
    """A completed scan publishes into the injected ResultStore; a SECOND worker (a new
    session) adopts the stored result in ensure_started without calling run_scan, and
    latest()/result_if_ready serve it immediately."""
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod
    from src.stock_screener.cockpit.scan_worker import ResultStore, ScanWorker

    store = ResultStore()
    calls = {"n": 0}

    def fake(*a, **kw):
        calls["n"] += 1
        return {"scan": calls["n"]}

    with patch.object(scanmod, "run_scan", side_effect=fake):
        w1 = ScanWorker(store=store)
        w1.ensure_started()
        assert w1.wait(grace=10.0) == {"scan": 1} and calls["n"] == 1
        assert store.get(w1._key2()).result == {"scan": 1}

        w2 = ScanWorker(store=store)                 # a fresh "session"
        w2.ensure_started()                          # adopts — no thread, no scan
        assert calls["n"] == 1, "adoption must not re-run the scan"
        assert w2.latest() == {"scan": 1} and w2.result_if_ready() == {"scan": 1}
        assert w2.snapshot()["as_of"] == store.get(w2._key2()).completed_wall


def test_scan_store_serial_adopt_dedups_cold_start():
    """Two sessions starting cold simultaneously: the second queues on the process-wide
    scan serializer and, once inside, finds the first's result already in the store
    (completed after its own run started) — it ADOPTS instead of re-scanning. Exactly
    one real scan for two cold sessions."""
    import threading as th
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod
    from src.stock_screener.cockpit.scan_worker import ResultStore, ScanWorker

    store = ResultStore()
    started, release = th.Event(), th.Event()
    calls = {"n": 0}

    def fake(*a, **kw):
        calls["n"] += 1
        started.set()
        assert release.wait(10), "test deadlock"
        return {"scan": calls["n"]}

    with patch.object(scanmod, "run_scan", side_effect=fake):
        w1, w2 = ScanWorker(store=store), ScanWorker(store=store)
        w1.ensure_started()
        assert started.wait(10)                      # w1 holds _SCAN_SERIAL inside fake
        w2.ensure_started()                          # queues behind the serial lock
        release.set()
        assert w1.wait(grace=10.0) == {"scan": 1}
        assert w2.wait(grace=10.0) == {"scan": 1}, "w2 must adopt w1's result"
    assert calls["n"] == 1, f"two cold sessions must cost ONE scan, ran {calls['n']}"


def test_worker_serves_stale_while_refreshing():
    """While a refresh is mid-flight, latest() keeps serving the previous result (the
    stale-while-refresh contract the UI is built on); result_if_ready stays None for the
    new key until the refresh lands; a FAILED refresh keeps the stale result serving."""
    import threading as th
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod
    from src.stock_screener.cockpit.scan_worker import ResultStore, ScanWorker

    store = ResultStore()
    started, release = th.Event(), th.Event()
    behavior = {"raise": False}
    calls = {"n": 0}

    def fake(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            started.set()
            assert release.wait(10), "test deadlock"
            if behavior["raise"]:
                raise RuntimeError("refresh boom")
        return {"scan": calls["n"]}

    with patch.object(scanmod, "run_scan", side_effect=fake):
        w = ScanWorker(store=store)
        w.ensure_started()
        assert w.wait(grace=10.0) == {"scan": 1}

        w.request_rescan()                           # manual refresh, blocks in fake
        assert started.wait(10)
        assert w.result_if_ready() is None           # new key not ready…
        assert w.latest() == {"scan": 1}             # …but the stale result still serves
        release.set()
        assert w.wait(grace=10.0) == {"scan": 2}
        assert store.get(w._key2()).result == {"scan": 2}

        started.clear(); release.clear()
        behavior["raise"] = True                     # now a FAILING refresh
        w.request_rescan()
        assert started.wait(10)
        release.set()
        w._thread.join(10)
        assert w.snapshot()["status"] == "error"
        assert w.latest() == {"scan": 2}, "failed refresh must keep serving stale"
        assert store.get(w._key2()).result == {"scan": 2}


def test_rescan_always_runs_and_no_interaction_refresh():
    """A manual rescan always re-scans (never satisfied by the store's fresh entry); a
    FAILED rescan keeps serving the stale result; and ensure_started NEVER starts a
    background refresh on interaction — not on a fresh entry, not on an old one, not
    after an error. Recovery paths are the Retry button and the half-hourly
    cockpit-refresh job, which tops prices up in a separate process entirely."""
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod
    from src.stock_screener.cockpit.scan_worker import ResultStore, ScanWorker

    store = ResultStore()
    calls = {"n": 0}
    fail = {"on": False}

    def fake(*a, **kw):
        calls["n"] += 1
        if fail["on"]:
            raise RuntimeError("boom")
        return {"scan": calls["n"]}

    with patch.object(scanmod, "run_scan", side_effect=fake):
        w = ScanWorker(store=store)
        w.ensure_started()
        assert w.wait(grace=10.0) == {"scan": 1} and calls["n"] == 1

        w.ensure_started()                           # done result -> nothing starts
        assert calls["n"] == 1, "interaction must never refresh a done result"

        fail["on"] = True
        w.request_rescan()                           # manual -> runs (and fails)
        w._thread.join(10)
        assert calls["n"] == 2 and w.snapshot()["status"] == "error"
        assert w.latest() == {"scan": 1}, "failed rescan keeps serving stale"

        fail["on"] = False
        w.ensure_started()                           # error key -> no auto-retry, ever
        w._thread.join(10)
        assert calls["n"] == 2, "interaction must never retry a failed run"


def test_scan_store_persists_and_reloads():
    """A completed scan is pickled (atomically) and a NEW store — a server restart —
    serves it on the first get: same result, ORIGINAL completed_wall (the 'data as of'
    display), and completed_mono = -inf so an in-flight run's adopt check can never be
    satisfied by it. ensure_started serves the loaded entry WITHOUT starting any
    network refresh (cockpit-refresh.timer owns price freshness now — a restart must not
    trigger a download per §6.56). Corrupt / wrong-version / wrong-key pickles fail
    open to a cold start."""
    import pickle as _pickle
    import tempfile

    from src.stock_screener.cockpit.scan_worker import (
        _PERSIST_VERSION, ResultStore, ScanWorker)

    key = ("full_us", 8)
    with tempfile.TemporaryDirectory() as tmp:
        pkl = Path(tmp) / "last_scan.pkl"

        # put -> pickle on disk, no tmp litter
        s1 = ResultStore(persist_path=pkl)
        ent1 = s1.put(key, {"scan": "persisted"})
        assert pkl.exists() and not list(Path(tmp).glob("*.tmp"))

        # "restart": a fresh store loads it lazily on the first miss
        s2 = ResultStore(persist_path=pkl)
        ent2 = s2.get(key)
        assert ent2 is not None and ent2.result == {"scan": "persisted"}
        assert ent2.completed_wall == ent1.completed_wall     # original scan time kept
        assert ent2.completed_mono == float("-inf")           # never adoptable mid-run

        # the worker end-to-end: a fresh session adopts the loaded entry instantly and
        # starts NOTHING (run_scan is deliberately unpatched — a network refresh here
        # would be the exact page-entry download that §6.56 removed).
        s3 = ResultStore(persist_path=pkl)
        w = ScanWorker(store=s3)
        assert w.latest() is None                             # nothing adopted yet
        w.ensure_started()
        assert w._thread is None, "page entry must never start a network refresh"
        assert w.latest() == {"scan": "persisted"}
        assert w.result_if_ready() == {"scan": "persisted"}

        # wrong key -> ignored (cold start)
        s4 = ResultStore(persist_path=pkl)
        assert s4.get(("sp500", 8)) is None

        # wrong version -> ignored
        pkl.write_bytes(_pickle.dumps({"version": _PERSIST_VERSION + 1, "key": key,
                                       "result": {"scan": "old-shape"},
                                       "completed_wall": 1.0}))
        s5 = ResultStore(persist_path=pkl)
        assert s5.get(key) is None

        # corrupt bytes -> ignored, no raise, and only ONE load attempt per process
        pkl.write_bytes(b"not a pickle")
        s6 = ResultStore(persist_path=pkl)
        assert s6.get(key) is None
        assert s6.get(key) is None                            # no retry loop

        # no file at all -> plain cold start
        pkl.unlink()
        s7 = ResultStore(persist_path=pkl)
        assert s7.get(key) is None

    # persist_path=None (every other unit test): zero disk I/O, still fully functional
    s8 = ResultStore()
    s8.put(key, {"scan": 1})
    assert s8.get(key).result == {"scan": 1}


def test_pending_force_survives_adoption():
    """R2-6: a Full-re-download armed while another run is in flight must RUN once the
    thread dies — adoption of another session's newer store result must not swallow it
    (and it must not detonate later inside a scheduled refresh). The forced run's result wins;
    the flag is consumed."""
    import threading as th
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod
    from src.stock_screener.cockpit.scan_worker import ResultStore, ScanWorker

    store = ResultStore()
    started, release = th.Event(), th.Event()
    forces = []

    def fake(*a, force=False, **kw):
        forces.append(force)
        if len(forces) == 1:
            started.set()
            assert release.wait(10), "test deadlock"
        return {"scan": len(forces), "force": force}

    with patch.object(scanmod, "run_scan", side_effect=fake):
        w = ScanWorker(store=store)
        w.ensure_started()
        assert started.wait(10)                      # run 1 in flight
        w.request_rescan(force=True)                 # thread alive -> force ARMS only
        assert w._pending_force is True
        release.set()
        w._thread.join(10)                           # run 1 lands (stale generation)

        # Another session's scan lands NEWER than ours — the adoption bait. Bump its
        # wall clock explicitly: a time.time() resolution tie would otherwise let the
        # old (buggy) code pass by accident.
        ent = store.put(w._key2(), {"scan": "other-session"})
        ent.completed_wall = (w.snapshot()["as_of"] or 0) + 60

        w.ensure_started()                           # must START the forced run…
        w._thread.join(10)
    assert forces == [False, True], f"forced run must execute, got {forces}"
    assert w.result_if_ready() == {"scan": 2, "force": True}, \
        "a forced run is never satisfied by someone else's result"
    assert w._pending_force is False


def test_worker_clock_coherence_queued_adopts():
    """R2-8 direction 2: with the store clock PINNED at 5.0 (far below real monotonic —
    the OLD mixed-clock check wrongly re-scanned on every machine), a worker queued
    behind the serial lock still adopts the result that landed while it waited (also
    exercises the >= equality case, since the clock never advances)."""
    import threading as th
    from unittest.mock import patch

    from src.stock_screener.cockpit import scan as scanmod
    from src.stock_screener.cockpit.scan_worker import ResultStore, ScanWorker

    store = ResultStore(clock=lambda: 5.0)
    started, release = th.Event(), th.Event()
    calls = {"n": 0}

    def fake(*a, **kw):
        calls["n"] += 1
        started.set()
        assert release.wait(10), "test deadlock"
        return {"scan": calls["n"]}

    with patch.object(scanmod, "run_scan", side_effect=fake):
        w1, w2 = ScanWorker(store=store), ScanWorker(store=store)
        w1.ensure_started()
        assert started.wait(10)                      # w1 holds _SCAN_SERIAL inside fake
        w2.ensure_started()                          # queues behind the serial lock
        release.set()
        assert w1.wait(grace=10.0) == {"scan": 1}
        assert w2.wait(grace=10.0) == {"scan": 1}, "w2 must adopt w1's result"
    assert calls["n"] == 1, f"one scan for two queued cold sessions, ran {calls['n']}"


def test_scan_worker_progress_classification():
    """ScanWorker._on_progress classifies labels into the phases the status line
    renders: 'Prices · SYM: cached (…)' -> 'cache' (a zero-network pass must not read
    as 'Downloading'), other price labels -> 'fetch', screening labels -> 'screen'.
    snapshot() exposes phase + phase_label + counts. (The per-name download log was
    removed 2026-08-11 with the full-page progress view.) data_feed label strings are
    untouched."""
    from src.stock_screener.cockpit.scan_worker import ScanWorker

    w = ScanWorker()
    w._on_progress(1, 10, "Prices · AAPL: cached (fresh)")
    s = w.snapshot()
    assert s["phase"] == "cache" and s["phase_label"] == "Reading cache"

    w._on_progress(2, 10, "Prices · MSFT: cached (settled close)")
    assert w.snapshot()["phase"] == "cache"

    w._on_progress(3, 10, "Prices · NVDA: 8/7/2026 - 8/9/2026")
    s = w.snapshot()
    assert s["phase"] == "fetch" and s["phase_label"] == "Downloading"

    w._on_progress(4, 10, "Prices · COLD: full history (2y)")
    assert w.snapshot()["phase"] == "fetch"

    w._on_progress(5, 10, "Screening · AAPL")
    s = w.snapshot()
    assert s["phase"] == "screen" and s["phase_label"] == "Screening"
    assert s["done"] == 5 and s["total"] == 10
    assert "log" not in s, "the download log was removed — snapshot must not carry one"



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "scan_worker"))
