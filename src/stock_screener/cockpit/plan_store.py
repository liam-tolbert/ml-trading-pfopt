"""Shared storage for the dated entry/sell plans — the mechanics both planners repeat.

``entries.py`` and ``sells.py`` are deliberate mirrors of each other (arm a buy for the
next open / plan an exit for the next open), and their *storage* halves had been copied
verbatim: same day-file naming, same atomic write, same newest-first loader, same env
gate. The copies had already begun to drift — only one of them cleaned up its temp file
on a failed write — which is the failure mode this module exists to end. The trading
logic stays in the two callers, where the rules genuinely differ.

Both plan kinds live in ``cache.TRIGGERS_DIR`` beside the trigger reports, and
``cache.TRIGGERS_DIR`` is read at CALL time, never captured at import: the test suite
patches it to keep AppTests away from real state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from src.stock_screener.cockpit import cache

_TRUTHY = {"1", "true", "yes", "on"}


def env_enabled(name: str, env: Optional[dict] = None) -> bool:
    """Is this automation armed? Both executors ship dark and stay off until the env
    var is set, so the default answer must be False for anything unparseable."""
    e = os.environ if env is None else env
    return str(e.get(name, "")).strip().lower() in _TRUTHY


def today_iso(today=None) -> str:
    """ET calendar date as ISO. Plans are named and aged by TRADING day, so the date has
    to come from the market's clock, not the host's."""
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
    """Atomic write (tmp + ``os.replace``).

    The building CLI, a page's disarm/veto click and the morning executor are three
    separate processes writing one day-file; an in-place truncate-write can interleave
    into JSON the loader then silently skips, which would look exactly like "no plan
    today" — a silently missed exit. The ``finally`` matters as much: a failed
    serialization used to leave a stray ``.tmp`` behind in the trigger directory."""
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
    """Newest parseable ``<prefix>_*.json``, or None. ``before`` (ISO date) skips plans
    dated >= it, which is how the evening sell planner reads YESTERDAY's snapshot for the
    P2 streak instead of its own earlier output on a same-day rerun.

    Never raises: one corrupt file must not blind the executor, so the walk continues
    newest-first past anything unreadable."""
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
    """Move one row's status in place; True if anything changed.

    The overnight veto/disarm. Guarded on ``from_status`` so a click can only ever
    cancel something still pending — never resurrect a submitted or failed row."""
    changed = False
    for item in plan.get(items_key, []):
        if item.get(id_key) == ident and item.get("status") == from_status:
            item["status"] = to_status
            changed = True
    return changed
