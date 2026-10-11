"""
Central logging for the dashboard.

Every module calls ``get_logger(__name__)``. Records go to two places:
  1. the console (visible in Streamlit Cloud -> Manage app -> Logs)
  2. ``logs/app.log`` (rotating, 1 MB x 3 files)

Line format (timestamps in SGT, the same clock as the dashboard):
  2026-10-11 17:14:02 SGT | WARNING | market_data.fetch_daily | sid=a1b2 | USDTWD=X: 0 rows

Streamlit re-runs the whole script on every click, so handlers are attached
exactly once per process (guarded by a flag) - otherwise every rerun would
duplicate every log line.
"""
from __future__ import annotations

import contextvars
import functools
import logging
import time
import uuid
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_NAME = "fxdash"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "app.log"
FORMAT = "%(asctime)s SGT | %(levelname)-7s | %(where)s | sid=%(sid)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
SLOW_MS = 1000  # log_step reports durations above this at INFO, below at DEBUG

# Session id of the Streamlit session currently running in this thread/context.
_SID: contextvars.ContextVar[str] = contextvars.ContextVar("fxdash_sid", default="-")


class _ContextFilter(logging.Filter):
    """Adds the session id and a short module name to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.sid = _SID.get()
        if not hasattr(record, "where"):  # log_step passes the decorated function's name explicitly
            name = record.name
            short = name[len(ROOT_NAME) + 1:] if name.startswith(ROOT_NAME + ".") else name
            record.where = f"{short}.{record.funcName}"
        return True


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root 'fxdash' logger once per process. Safe to call on every rerun."""
    root = logging.getLogger(ROOT_NAME)
    if getattr(root, "_fxdash_configured", False):
        return root

    root.setLevel(level)
    root.propagate = False
    formatter = logging.Formatter(FORMAT, DATEFMT)
    formatter.converter = lambda secs: time.gmtime(secs + 8 * 3600)  # SGT (UTC+8, no DST), as shown in the app
    context_filter = _ContextFilter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(context_filter)
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)
    except OSError as exc:  # read-only filesystem etc. -> console only
        root.warning("File logging disabled (%s); console only.", exc)

    root._fxdash_configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str) -> logging.Logger:
    """Module logger, e.g. get_logger(__name__) -> 'fxdash.market_data'."""
    setup_logging()
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def set_level(level_name: str) -> None:
    """Switch between INFO and DEBUG at runtime (Diagnostics toggle)."""
    logging.getLogger(ROOT_NAME).setLevel(getattr(logging, level_name.upper(), logging.INFO))


def set_session_id(sid: str) -> None:
    """Called at the top of every script run so log lines can be filtered per user session."""
    _SID.set(sid)


def new_error_id() -> str:
    """Short, searchable id shown to the user and written next to the traceback."""
    return uuid.uuid4().hex[:6].upper()


def log_step(func=None, *, name: str | None = None):
    """
    Decorator: logs duration of a function (INFO if slow, else DEBUG) and, on failure, the full traceback
    with an error id (attached to the exception as ``exc.err_id``) before re-raising.
    Nested decorated calls re-use the first error id instead of logging twice.
    """

    def decorator(f):
        step = name or f"{f.__module__}.{f.__qualname__}"
        log = get_logger(f.__module__)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            log.debug("start %s", step, extra={"where": step})
            try:
                result = f(*args, **kwargs)
            except Exception as exc:
                if not hasattr(exc, "err_id"):
                    err_id = new_error_id()
                    try:
                        exc.err_id = err_id  # type: ignore[attr-defined]
                    except AttributeError:
                        pass
                    log.exception("[%s] %s failed: %s", err_id, step, exc, extra={"where": step})
                raise
            ms = (time.perf_counter() - start) * 1000
            # slow steps at INFO so they stand out; routine fast ones at DEBUG to keep the log readable
            log.log(logging.INFO if ms >= SLOW_MS else logging.DEBUG, "%s done in %.0f ms", step, ms,
                    extra={"where": step})
            return result

        return wrapper

    return decorator(func) if func is not None else decorator


def tail_log(n: int = 50, min_level: str = "DEBUG") -> list[str]:
    """Last ``n`` lines of app.log at or above ``min_level`` (for the Diagnostics panel)."""
    if not LOG_FILE.exists():
        return []
    order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    allowed = set(order[order.index(min_level.upper()):]) if min_level.upper() in order else set(order)
    keep: deque[str] = deque(maxlen=n)
    record_kept = False
    with LOG_FILE.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split("|")
            is_record = len(parts) > 2 and line[:4].isdigit()
            if is_record:
                record_kept = parts[1].strip() in allowed
                if record_kept:
                    keep.append(line.rstrip("\n"))
            elif record_kept:  # traceback continuation lines of a kept record
                keep.append(line.rstrip("\n"))
    return list(keep)


def find_in_log(text: str, max_lines: int = 300) -> list[str]:
    """Every log record containing ``text`` (e.g. an error id) plus its traceback lines."""
    if not text or not LOG_FILE.exists():
        return []
    out: list[str] = []
    capturing = False
    with LOG_FILE.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            is_record = len(line.split("|")) > 2 and line[:4].isdigit()
            if is_record:
                capturing = text.lower() in line.lower()
            if capturing:
                out.append(line.rstrip("\n"))
    return out[-max_lines:]


def read_log_bytes() -> bytes:
    """Full current log file for the download button."""
    return LOG_FILE.read_bytes() if LOG_FILE.exists() else b""
