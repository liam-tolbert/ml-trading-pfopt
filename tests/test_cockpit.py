"""SEPA cockpit test gate — runs every suite under tests/cockpit/.

This FILENAME is the contract: deploy/deploy.sh and .github/workflows/tests.yml both invoke
`python tests/test_cockpit.py`, and .dockerignore un-ignores it by name. The tests
themselves live in tests/cockpit/test_<category>.py and each runs standalone too:

    python tests/test_cockpit.py          # the whole gate
    python tests/cockpit/test_vcp.py      # one category

Imports are EXPLICIT, never a directory glob. A category file missing from the image has to
fail loudly as an ImportError and block the deploy; discovery would quietly run fewer tests
and still report green, which is exactly the failure mode HANDOFF §6.61 records.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.cockpit import (  # noqa: E402
    test_app,
    test_data_feed,
    test_entries,
    test_journal,
    test_positions,
    test_runlog,
    test_scan,
    test_scan_worker,
    test_trade,
    test_triggers,
    test_vcp,
    test_watchlist)
from tests.cockpit._common import run_suite  # noqa: E402

SUITES = (
    test_data_feed,
    test_scan,
    test_vcp,
    test_watchlist,
    test_trade,
    test_positions,
    test_entries,
    test_triggers,
    test_app,
    test_scan_worker,
    test_journal,
    test_runlog)


def _run_all() -> int:
    """Every suite's tests in one pass, so the gate prints ONE total the way it always has.

    A name colliding across suites is fatal rather than silently shadowed: the collected
    namespace is a dict, so the second definition would otherwise replace the first and the
    count would still look right."""
    ns = {}
    for mod in SUITES:
        for k, v in vars(mod).items():
            if k.startswith("test_") and callable(v):
                if k in ns:
                    raise SystemExit(f"FATAL: duplicate test name across suites: {k}")
                ns[k] = v
    return run_suite(ns)


if __name__ == "__main__":
    raise SystemExit(_run_all())
