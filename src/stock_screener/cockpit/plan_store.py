"""Shared storage for the dated entry and sell plans.

``entries.py`` and ``sells.py`` mirror each other: one arms buys for the next open, the
other plans exits for it. Their storage MUST live here once: the day-file naming, the
atomic write, the newest-first loader and the env gate. Copies drift. The trading logic
stays in the two callers, where the rules differ.

Both plan kinds live in ``cache.TRIGGERS_DIR`` beside the trigger reports. It MUST be
read at call time, never captured at import: the test suite patches it to keep AppTests
away from real state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from src.stock_screener.cockpit import cache

_TRUTHY = {"1", "true", "yes", "on"}


def env_enabled(name: str, env: Optional[dict] = None) -> bool:
    """True when env var ``name`` is truthy (1/true/yes/on). Both executors ship dark, so
    anything unset or unparseable MUST read False."""
    e = os.environ if env is None else env
    return str(e.get(name, "")).strip().lower() in _TRUTHY


def today_iso(today=None) -> str:
    """Today's ET calendar date as ISO, or ``today``'s date when given. Plans are named and
    aged by trading day, so the date MUST come from the market's clock, not the host's."""
    import pandas as pd
    if today is None:
        t = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    else:
        t = pd.Timestamp(today).normalize()
    return t.date().isoformat()


def plan_path(prefix: str, date_iso: str, dir_path=None) -> Path:
    d = Path(dir_path if dir_path is not None else cache.TRIGGERS_DIR)
    return d / f"{prefix}_{date_iso}.json"


def save_plan(prefix: str, plan: dict, dir_path=None) -> Path:
    """Write ``plan`` to its day-file atomically (tmp + ``os.replace``); returns the path.

    Three processes write one day-file: the building CLI, a page's disarm/veto click and
    the morning executor. An in-place write can interleave into JSON the loader skips,
    which reads as "no plan today": a silently missed exit. The ``finally`` removes the
    temp file when serialization fails; the error still propagates."""
    path = plan_path(prefix, plan["date"], dir_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def load_latest_plan(prefix: str, dir_path=None, *,
                     before: Optional[str] = None) -> Optional[dict]:
    """The newest parseable ``<prefix>_*.json`` holding a dict, or None.

    ``before`` (ISO date) skips plans dated on or after it. The evening sell planner uses
    it to read an earlier day's plan for the P2 streak, not its own same-day output.

    Never raises. The walk skips anything unreadable, so one corrupt file can't blind the
    executor."""
    try:
        d = Path(dir_path if dir_path is not None else cache.TRIGGERS_DIR)
        for path in sorted(d.glob(f"{prefix}_*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                if before is not None and str(data.get("date", "")) >= before:
                    continue
                return data
            except Exception:
                continue
    except Exception:
        pass
    return None


def flip_status(plan: dict, *, items_key: str, id_key: str, ident: str,
                from_status: str, to_status: str) -> bool:
    """Move the ``ident`` row from ``from_status`` to ``to_status`` in place; True if
    anything changed.

    This is the overnight veto or disarm. Only a row still in ``from_status`` moves, so a
    click can cancel a pending row but never resurrect a submitted or failed one."""
    changed = False
    for item in plan.get(items_key, []):
        if item.get(id_key) == ident and item.get("status") == from_status:
            item["status"] = to_status
            changed = True
    return changed
