"""The hunt PC's API (scripts/hunt/hunt_api.py), driven over a loopback socket with a fake
launcher. Runs as a plain script on the laptop; it is not part of the deploy gate (the
API never ships to the Pi).

    python tests/test_hunt_api.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "hunt"))

import hunt_api  # noqa: E402

PASSED = 0


def ok(name: str, cond: bool) -> None:
    global PASSED
    assert cond, f"FAIL  {name}"
    PASSED += 1
    print(f"  PASS  {name}")


class _FakeProc:
    """A launched run the test ends by setting ``rc``."""
    pid = 4242

    def __init__(self):
        self.rc = None

    def poll(self):
        return self.rc


def _call(base: str, method: str, path: str, token=None):
    req = urllib.request.Request(base + path, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def test_api():
    procs = []

    def launcher():
        p = _FakeProc()
        procs.append(p)
        return p

    with tempfile.TemporaryDirectory() as td:
        hunt_dir = Path(td)
        runs = hunt_api.Runs(launcher=launcher, hunt_dir=hunt_dir)
        srv = hunt_api.serve("127.0.0.1:0", "secret", runs)
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            code, body = _call(base, "GET", "/health")
            ok("health needs no token", code == 200 and body["ok"] is True)
            ok("status without a token is 401", _call(base, "GET", "/hunt/status")[0] == 401)
            ok("start with a wrong token is 401",
               _call(base, "POST", "/hunt/start", "nope")[0] == 401)
            ok("unknown routes are 404",
               _call(base, "GET", "/hunt/nothing", "secret")[0] == 404
               and _call(base, "POST", "/hunt/status", "secret")[0] == 404)
            code, body = _call(base, "GET", "/hunt/status", "secret")
            ok("idle before any run", code == 200 and body["state"] == "idle")
            code, body = _call(base, "POST", "/hunt/start", "secret")
            ok("start launches a run", code == 200 and body["state"] == "running"
               and len(procs) == 1 and body["date"])
            date = body["date"]
            code, body = _call(base, "POST", "/hunt/start", "secret")
            ok("a second start while running is 409, no second launch",
               code == 409 and len(procs) == 1 and body["state"] == "running")
            code, body = _call(base, "GET", "/hunt/status", "secret")
            ok("status reports running", body["state"] == "running" and body["pushed"] is False)
            (hunt_dir / date).mkdir()
            (hunt_dir / date / "FAILED.txt").write_text("The weekend hunt did not finish: x\n")
            procs[0].rc = 1
            code, body = _call(base, "GET", "/hunt/status", "secret")
            ok("a non-zero exit is error, with FAILED.txt's first line",
               body["state"] == "error" and body["message"].startswith("The weekend hunt")
               and body["finished"] and body["pushed"] is False)
            code, body = _call(base, "POST", "/hunt/start", "secret")
            ok("after an error a new run may start", code == 200 and len(procs) == 2)
            procs[1].rc = 0
            code, body = _call(base, "GET", "/hunt/status", "secret")
            ok("a zero exit is done and pushed", body["state"] == "done" and body["pushed"] is True)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    test_api()
    print(f"\n{PASSED} hunt api assertions passed.")
