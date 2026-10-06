"""The hunt PC's API: lets the Pi's cockpit start a weekend hunt and watch it finish.

    python scripts/hunt/hunt_api.py          # scripts/hunt/hunt_api.ps1 is the logon task

Standard library only. Routes, all JSON:

    GET  /health        -> {ok, host, time}                       no token
    POST /hunt/start    -> {state: "running", date, started}      409 while one runs
    GET  /hunt/status   -> {state: idle|running|done|error, date, started, finished,
                            message, pushed, log_tail}

``POST /hunt/start`` launches ``weekend_hunt.ps1 -NoSleep`` detached: the PC is awake
and someone pressed a button, so the run's sleep rule does not apply. The script pulls
the Pi's scan, reviews, builds the report and pushes the folder back; ``done`` means its
exit code was 0, which includes the push. One run at a time; the Friday task is not
tracked here, so a button press during it is refused only if this API started it.

Configuration from the environment: ``HUNT_API_TOKEN`` (required: every call but
/health MUST carry ``Authorization: Bearer <token>``), ``HUNT_API_BIND`` (default
``0.0.0.0:8765``; the Windows firewall rule MUST limit it to the LAN). Nothing in a
request is interpreted beyond the route, so there is no input to inject.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "hunt" / "weekend_hunt.ps1"
HUNT_DIR = ROOT / "data" / "cockpit" / "hunt"
LOG_DIR = HUNT_DIR / "logs"
API_LOG = LOG_DIR / "api.log"


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _log(line: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(API_LOG, "a", encoding="utf-8") as f:
            f.write(f"{_now()}  {line}\n")
    except OSError:
        pass


def launch_hunt() -> subprocess.Popen:
    """Start the hunt script in its own hidden console, its output to the day's api_run
    log. It MUST get a console (CREATE_NO_WINDOW): without one powershell.exe exits 0 at
    once and runs nothing."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = open(LOG_DIR / f"api_run_{_dt.date.today().isoformat()}.log", "ab")
    flags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
             | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
         "-NoSleep"],
        cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        creationflags=flags)


class Runs:
    """The single run slot: at most one hunt started from here at a time."""

    def __init__(self, launcher: Callable[[], subprocess.Popen] = launch_hunt,
                 hunt_dir: Path = HUNT_DIR):
        self._lock = threading.Lock()
        self._launch = launcher
        self._hunt_dir = hunt_dir
        self.proc: Optional[subprocess.Popen] = None
        self.date: Optional[str] = None
        self.started: Optional[str] = None
        self.finished: Optional[str] = None
        self.code: Optional[int] = None

    def _poll(self) -> None:
        if self.proc is not None and self.code is None:
            rc = self.proc.poll()
            if rc is not None:
                self.code, self.finished = int(rc), _now()
                _log(f"run {self.date} exited {rc}")

    def start(self) -> Optional[dict]:
        """Launch a run; None when one is already running."""
        with self._lock:
            self._poll()
            if self.proc is not None and self.code is None:
                return None
            self.date = _dt.date.today().isoformat()
            self.started, self.finished, self.code = _now(), None, None
            self.proc = self._launch()
            _log(f"run {self.date} started (pid {self.proc.pid})")
            return {"state": "running", "date": self.date, "started": self.started}

    def status(self) -> dict:
        with self._lock:
            self._poll()
            if self.proc is None:
                return {"state": "idle", "date": None, "started": None, "finished": None,
                        "message": "", "pushed": False, "log_tail": []}
            state = "running" if self.code is None else ("done" if self.code == 0 else "error")
            return {"state": state, "date": self.date, "started": self.started,
                    "finished": self.finished, "message": self._message(state),
                    "pushed": self.code == 0, "log_tail": self._log_tail()}

    def _log_tail(self, n: int = 3) -> list:
        try:
            lines = (LOG_DIR / f"{self.date}.log").read_text(encoding="utf-8").splitlines()
            return [ln[:160] for ln in lines[-n:]]
        except OSError:
            return []

    def _message(self, state: str) -> str:
        if state == "error":
            try:
                return (self._hunt_dir / self.date / "FAILED.txt").read_text(
                    encoding="utf-8").splitlines()[0][:200]
            except (OSError, IndexError):
                return f"exit code {self.code}"
        tail = self._log_tail(1)
        return tail[0].split("  ", 1)[-1] if tail else ""


def make_handler(runs: Runs, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "hunt-api/1"

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            return self.headers.get("Authorization", "") == f"Bearer {token}"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "host": socket.gethostname(), "time": _now()})
            elif not self._authorized():
                self._send(401, {"message": "bad or missing token"})
            elif self.path == "/hunt/status":
                self._send(200, runs.status())
            else:
                self._send(404, {"message": "no such route"})

        def do_POST(self) -> None:
            if not self._authorized():
                self._send(401, {"message": "bad or missing token"})
            elif self.path == "/hunt/start":
                info = runs.start()
                if info is None:
                    self._send(409, {"message": "a hunt is already running", **runs.status()})
                else:
                    self._send(200, info)
            else:
                self._send(404, {"message": "no such route"})

        def log_message(self, fmt, *args) -> None:     # to the file, not stderr
            _log(f"{self.client_address[0]} {fmt % args}")

    return Handler


def serve(bind: str, token: str, runs: Optional[Runs] = None) -> ThreadingHTTPServer:
    host, _, port = bind.rpartition(":")
    srv = ThreadingHTTPServer((host or "0.0.0.0", int(port)), make_handler(runs or Runs(), token))
    return srv


def main() -> int:
    token = os.environ.get("HUNT_API_TOKEN", "").strip()
    if not token:
        print("hunt_api: HUNT_API_TOKEN is not set; refusing to start", file=sys.stderr)
        return 2
    bind = os.environ.get("HUNT_API_BIND", "0.0.0.0:8765")
    srv = serve(bind, token)
    _log(f"listening on {bind}")
    print(f"hunt_api: listening on {bind}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
