"""A weekend-hunt request and its status: two JSON files under ``data/cockpit/hunt/``.

Nothing on the LAN connects to the hunt PC. The cockpit (the Start button) or the Pi's
Friday timer writes ``request.json``; the PC's poller, which already reaches the Pi over
ssh, claims it (an atomic rename on the Pi), runs the hunt, and writes ``status.json``
back as the run goes: ``claimed`` → ``running`` → ``done`` or ``error``. The finished
folder arrives beside them. This module is the Pi side; ``scripts/hunt/hunt_poller.ps1``
is the PC side.

    python -m src.stock_screener.cockpit.hunt_request write --source schedule
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Optional, Tuple

from src.stock_screener.hunt import pipeline as pl

REQUEST_FILE = "request.json"
STATUS_FILE = "status.json"
ACTIVE_STATES = ("claimed", "running")
STALE_RUN_S = 4 * 3600        # a "running" status older than this is a dead run


def request_path() -> Path:
    return pl.HUNT_DIR / REQUEST_FILE      # pipeline.HUNT_DIR read at call time: patchable


def status_path() -> Path:
    return pl.HUNT_DIR / STATUS_FILE


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _read(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(path: Path, data: dict) -> None:
    """Atomic: a sibling temp file, then ``os.replace``, so the poller's ssh read never
    sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_request() -> Optional[dict]:
    return _read(request_path())


def read_status() -> Optional[dict]:
    return _read(status_path())


def write_request(source: str = "cockpit") -> Tuple[dict, bool]:
    """Ask for a hunt. Returns ``(request, created)``; a request already waiting is
    returned unchanged with ``created`` False, so two presses are one hunt."""
    existing = read_request()
    if existing:
        return existing, False
    rec = {"source": source, "requested_at": _now(), "date": _dt.date.today().isoformat()}
    _write(request_path(), rec)
    return rec, True


def _age_s(stamp: Optional[str]) -> Optional[float]:
    try:
        return (_dt.datetime.now() - _dt.datetime.fromisoformat(stamp)).total_seconds()
    except (TypeError, ValueError):
        return None


def progress() -> dict:
    """What the page shows: ``{state, message, requested_at, updated_at, date}``.

    ``state`` is ``requested`` (waiting for the PC to claim it), ``claimed`` or
    ``running`` (the PC's own words), ``done`` or ``error`` (the last run's end), or
    ``idle``. A request outranks a status, because the PC removes the request when it
    claims it. A ``running`` older than ``STALE_RUN_S`` reads as ``error``: the PC died
    mid-run and will never report."""
    req = read_request()
    if req:
        return {"state": "requested", "message": "", "requested_at": req.get("requested_at"),
                "updated_at": None, "date": req.get("date")}
    st = read_status()
    if not st:
        return {"state": "idle", "message": "", "requested_at": None, "updated_at": None,
                "date": None}
    state = str(st.get("state") or "idle")
    age = _age_s(st.get("updated_at"))
    if state in ACTIVE_STATES and age is not None and age > STALE_RUN_S:
        state = "error"
        st = {**st, "message": "the hunt PC stopped reporting mid-run; check its log"}
    return {"state": state, "message": str(st.get("message") or ""),
            "requested_at": st.get("requested_at"), "updated_at": st.get("updated_at"),
            "date": st.get("date")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hunt_request", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("write", help="ask the hunt PC for a hunt (no-op when one is waiting)")
    p.add_argument("--source", default="schedule")
    sub.add_parser("show", help="print the request and status files")
    args = ap.parse_args(argv)
    if args.cmd == "write":
        rec, created = write_request(args.source)
        print(json.dumps({"created": created, **rec}))
    else:
        print(json.dumps({"request": read_request(), "status": read_status(),
                          "progress": progress()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
