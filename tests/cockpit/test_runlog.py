"""Cockpit tests — dated run logs under data/cockpit/logs/ and their retention.

Runs standalone (`python tests/cockpit/test_runlog.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_runlog_prune_expires_at_fourteen_days():
    """Retention counts today as age 0, so RETENTION_DAYS files survive (today plus the 13
    before it) and a log exactly RETENTION_DAYS old is gone. The boundary IS the contract,
    so it is pinned from both sides — an off-by-one here silently keeps or drops a day."""
    import datetime as dt
    import tempfile

    from src.stock_screener.cockpit import runlog

    today = dt.date(2026, 8, 24)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        made = {}
        for age in (0, 1, 13, 14, 30):
            p = d / f"cockpit_{(today - dt.timedelta(days=age)).isoformat()}.log"
            p.write_text("x", encoding="utf-8")
            made[age] = p
        removed = runlog.prune_logs(today=today, dir_path=d)
        for age in (0, 1, 13):
            assert made[age].exists(), f"log aged {age}d is inside the window, must survive"
        assert not made[14].exists(), "a log aged exactly 14d must expire"
        assert not made[30].exists(), "an ancient log must expire"
        assert set(removed) == {made[14], made[30]}, removed


def test_runlog_prune_never_touches_foreign_files():
    """Only ``cockpit_<iso-date>.log`` is ever unlinked. Pruning runs unattended inside
    ordinary logging calls, and the directory sits next to hand-dropped files — the dead
    ``eod_trigger.log`` was exactly such a file — so the matcher must be exact."""
    import datetime as dt
    import tempfile

    from src.stock_screener.cockpit import runlog

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        keep = [d / "eod_trigger.log", d / "cockpit_not-a-date.log",
                d / "cockpit_2026-01-01.log.bak", d / "notes.txt"]
        for p in keep:
            p.write_text("x", encoding="utf-8")
        expired = d / "cockpit_2026-01-01.log"
        expired.write_text("x", encoding="utf-8")

        runlog.prune_logs(today=dt.date(2026, 8, 24), dir_path=d)

        assert not expired.exists(), "the dated log was past retention and should be gone"
        for p in keep:
            assert p.exists(), f"pruner deleted a file it does not own: {p.name}"


def test_runlog_handler_writes_dated_file_and_rolls():
    """One file per local date, reopened when the date changes. This is what lets the app,
    the one-shot trigger container and the CLIs share the directory with no rotation race:
    nobody ever renames a file another process holds open."""
    import datetime as dt
    import logging
    import tempfile
    from unittest.mock import patch

    from src.stock_screener.cockpit import cache as cachemod
    from src.stock_screener.cockpit import runlog

    day1, day2 = dt.date(2026, 8, 24), dt.date(2026, 8, 25)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "logs"
        with patch.object(cachemod, "LOGS_DIR", d):
            handler = runlog.DatedFileHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            with patch.object(runlog, "_today", lambda: day1):
                handler.emit(_log_record("first"))
            with patch.object(runlog, "_today", lambda: day2):
                handler.emit(_log_record("second"))
            handler.close()
            first = (d / f"cockpit_{day1.isoformat()}.log").read_text(encoding="utf-8")
            second = (d / f"cockpit_{day2.isoformat()}.log").read_text(encoding="utf-8")
    assert first.strip() == "first", first
    assert second.strip() == "second", second


def test_runlog_logger_is_isolated_from_the_stdlib_root():
    """``propagate=False``. Streamlit configures the stdlib root logger, so propagation
    would print every cockpit record a second time in ``docker logs``."""
    import logging

    from src.stock_screener.cockpit import runlog

    log = runlog.get_logger("prices")
    root = logging.getLogger(runlog._ROOT_NAME)
    assert log.name == "cockpit.prices", log.name
    assert root.propagate is False, "cockpit records must not reach the stdlib root"
    assert any(isinstance(h, runlog.DatedFileHandler) for h in root.handlers), root.handlers



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "runlog"))
