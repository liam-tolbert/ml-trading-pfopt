"""Run logging for the cockpit: dated files under ``data/cockpit/logs/``, pruned at 14 days.

Three kinds of process write here: the long-lived Streamlit app, the one-shot refresh and
EOD containers, and the morning buy/sell CLIs. ``TimedRotatingFileHandler`` MUST NOT be
used: its rename-on-rollover races between processes, and two containers rolling the same
file silently lose records. A dated filename needs no rollover. Every process appends to
``cockpit_<today>.log`` (``O_APPEND`` keeps interleaved line writes whole), and retention
prunes files whose date has left the window. The trigger reports next door use the same
convention, so the two read as one dated trail.

Records also go to stdout, so ``docker logs cockpit-app`` and journald see everything. The
file is the durable copy, not the only one.

This module MUST NOT import pandas. It is on the import path of every cockpit process, and
the handler is called from inside fetch loops.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
import sys
import threading
from pathlib import Path
from typing import List, Optional

from . import cache

RETENTION_DAYS = 14                 # a log this many days old is expired (today = age 0)
LOG_PREFIX = "cockpit_"
LOG_SUFFIX = ".log"
LOG_GLOB = f"{LOG_PREFIX}*{LOG_SUFFIX}"
# The pruner deletes only files of this exact shape. A stray .log dropped in the
# directory by hand is left alone.
_NAME_RE = re.compile(rf"^{re.escape(LOG_PREFIX)}(\d{{4}}-\d{{2}}-\d{{2}})"
                      rf"{re.escape(LOG_SUFFIX)}$")

_ROOT_NAME = "cockpit"
_FORMAT = "%(asctime)s  %(levelname)-7s %(name)s  %(message)s"


def _today() -> _dt.date:
    return _dt.datetime.now().astimezone().date()


def _logs_dir(dir_path=None) -> Path:
    """``dir_path``, else ``cache.LOGS_DIR``. It MUST be resolved on every call, never
    captured at import: the test suite patches ``cache.LOGS_DIR`` to keep runs away from
    real state."""
    return Path(dir_path if dir_path is not None else cache.LOGS_DIR)


def log_path(day: Optional[_dt.date] = None, dir_path=None) -> Path:
    return _logs_dir(dir_path) / f"{LOG_PREFIX}{(day or _today()).isoformat()}{LOG_SUFFIX}"


def prune_logs(retention_days: int = RETENTION_DAYS, today: Optional[_dt.date] = None,
               dir_path=None) -> List[Path]:
    """Delete dated logs at or past ``retention_days`` of age; return the removed paths.

    Age is in whole days with today at 0, so ``RETENTION_DAYS`` leaves exactly that many
    dated files: today plus the 13 before it. Never raises. A log that can't be parsed or
    unlinked is skipped: pruning is a side effect of ordinary logging and MUST NOT take a
    scan down with it."""
    t = today or _today()
    removed: List[Path] = []
    try:
        for path in sorted(_logs_dir(dir_path).glob(LOG_GLOB)):
            m = _NAME_RE.match(path.name)
            if not m:
                continue
            try:
                day = _dt.date.fromisoformat(m.group(1))
            except ValueError:
                continue
            if (t - day).days >= retention_days:
                try:
                    path.unlink()
                    removed.append(path)
                except OSError:
                    continue
    except OSError:
        pass
    return removed


class _Formatter(logging.Formatter):
    """ISO date, 12-hour clock and zone: ``2026-08-24 05:27:28 PM EDT``.

    A person reads these lines, so the clock is 12-hour. The date stays ISO: it sorts and
    matches the log filenames. The containers set ``TZ=America/New_York`` (compose and
    Dockerfile), so local time is market time. The zone stays on the line so a box whose
    TZ has drifted shows it, instead of writing times that can't be placed."""

    def formatTime(self, record, datefmt=None):        # noqa: N802  (stdlib spelling)
        t = _dt.datetime.fromtimestamp(record.created).astimezone()
        # %Z (EDT) is empty on a box with no tz database; fall back to the numeric offset.
        zone = t.strftime("%Z") or t.strftime("%z")
        return f"{t:%Y-%m-%d %I:%M:%S %p} {zone}".rstrip()


class DatedFileHandler(logging.Handler):
    """Append to ``cockpit_<today>.log``, re-opening when the date (or the configured
    directory) changes and pruning expired logs on each such roll.

    ``logging`` already serializes ``emit`` behind the handler lock, so the stream swap
    needs no lock of its own. A file that can't be opened degrades to ``handleError``:
    stdout still has the record, and a full or read-only card never costs a scan."""

    def __init__(self, retention_days: int = RETENTION_DAYS) -> None:
        super().__init__()
        self.retention_days = retention_days
        self._stream = None
        self._key = None                # (day, resolved dir) the open stream belongs to

    def release_file(self) -> None:
        """Close the open file without tearing the handler down. It re-opens lazily on
        the next record, so this is safe to call at any time.

        Windows can't unlink an open file, so a test's ``TemporaryDirectory`` cleanup
        fails with ``WinError 32`` without this. POSIX allows the unlink.

        MUST NOT be named ``release``: ``Handler.handle`` calls that after every emit to
        free the handler lock. Overriding it leaves the lock held, and every other thread
        that logs blocks forever."""
        try:
            if self._stream is not None:
                self._stream.close()
        except OSError:
            pass
        finally:
            self._stream = None
            self._key = None

    def _ensure_stream(self) -> None:
        key = (_today(), _logs_dir())
        if self._stream is not None and key == self._key:
            return
        self.release_file()
        day, directory = key
        directory.mkdir(parents=True, exist_ok=True)
        self._stream = open(log_path(day), "a", encoding="utf-8")
        self._key = key
        prune_logs(self.retention_days, today=day)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_stream()
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()        # one-shot containers die without an atexit flush
        except Exception:               # noqa: BLE001  (logging MUST NOT raise upward)
            self.handleError(record)

    def close(self) -> None:
        self.release_file()
        super().close()


_configured = False
_configure_lock = threading.Lock()


def _configure() -> None:
    global _configured
    with _configure_lock:
        if _configured:
            return
        root = logging.getLogger(_ROOT_NAME)
        root.setLevel(logging.INFO)
        # MUST NOT propagate: Streamlit configures the stdlib root, so every cockpit
        # record would print twice in `docker logs`.
        root.propagate = False
        fmt = _Formatter(_FORMAT)
        fh = DatedFileHandler()
        fh.setFormatter(fmt)
        root.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
        _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """The ``cockpit`` logger, or a ``cockpit.<name>`` child. Configured once per process."""
    _configure()
    return logging.getLogger(_ROOT_NAME if not name else f"{_ROOT_NAME}.{name}")


def release_files() -> None:
    """Close every open dated-log file; each re-opens lazily on the next record.

    Callers MUST call this before deleting a directory that holds a log. Windows can't
    unlink an open file, so a test's ``TemporaryDirectory`` cleanup raises ``WinError 32``
    and aborts the suite. Harmless elsewhere."""
    for h in logging.getLogger(_ROOT_NAME).handlers:
        if isinstance(h, DatedFileHandler):
            h.release_file()
