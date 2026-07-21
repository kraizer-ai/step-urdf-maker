"""Persistent diagnostics for the console-free desktop application.

``STEP_URDF_Maker.bat`` starts Python with ``pythonw.exe``.  That is pleasant
for normal use, but it also means an exception (or a native OCCT/VTK failure)
has no console on which to leave useful evidence.  This module installs the
process-wide hooks used by both the GUI and the isolated STEP worker.

The primary log intentionally lives beside the application.  A temporary-file
fallback is used only when that location is not writable, so diagnostics must
never become a reason that the application itself cannot start.
"""

from __future__ import annotations

import faulthandler
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import TracebackType
from typing import Any


LOGGER_NAME = "urdf_maker"
LOG_FILENAME = "STEP_URDF_Maker.log"

_lock = threading.RLock()
_fault_stream: Any | None = None
_active_log_path: Path | None = None
_qt_handler_installed = False

_original_sys_excepthook = sys.excepthook
_original_threading_excepthook = getattr(threading, "excepthook", None)
_original_unraisablehook = getattr(sys, "unraisablehook", None)


def application_directory() -> Path:
    """Return the folder in which user-visible application files reside."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Source and editable installs keep this module at
    # ``<application>/urdf_maker/runtime_log.py``.
    return Path(__file__).resolve().parent.parent


def default_log_path() -> Path:
    return application_directory() / LOG_FILENAME


def _open_fault_stream(path: Path):
    # faulthandler writes bytes directly to the file descriptor.  Opening the
    # stream in binary append mode also prevents locale-dependent encoding
    # failures while the interpreter is already handling a fatal error.
    return path.open("ab", buffering=0)


def _resolve_writable_log_path(requested: str | Path | None) -> Path:
    desired = Path(requested).expanduser().resolve() if requested else default_log_path()
    try:
        desired.parent.mkdir(parents=True, exist_ok=True)
        with desired.open("ab"):
            pass
        return desired
    except OSError:
        fallback = Path(tempfile.gettempdir()).resolve() / LOG_FILENAME
        fallback.parent.mkdir(parents=True, exist_ok=True)
        with fallback.open("ab"):
            pass
        return fallback


def _log_uncaught(
    where: str,
    exc_type: type[BaseException],
    exc_value: BaseException,
    traceback: TracebackType | None,
) -> None:
    logging.getLogger(LOGGER_NAME).critical(
        "Unhandled exception (%s)",
        where,
        exc_info=(exc_type, exc_value, traceback),
    )


def _sys_exception_hook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    traceback: TracebackType | None,
) -> None:
    _log_uncaught("main thread", exc_type, exc_value, traceback)
    if _original_sys_excepthook not in (None, _sys_exception_hook):
        _original_sys_excepthook(exc_type, exc_value, traceback)


def _thread_exception_hook(args: Any) -> None:
    thread_name = getattr(getattr(args, "thread", None), "name", "unknown")
    _log_uncaught(
        f"thread {thread_name}",
        args.exc_type,
        args.exc_value,
        args.exc_traceback,
    )
    if _original_threading_excepthook not in (None, _thread_exception_hook):
        _original_threading_excepthook(args)


def _unraisable_hook(unraisable: Any) -> None:
    object_repr = repr(getattr(unraisable, "object", None))
    error_message = getattr(unraisable, "err_msg", None) or "unraisable exception"
    logging.getLogger(LOGGER_NAME).error(
        "%s in %s",
        error_message,
        object_repr,
        exc_info=(
            unraisable.exc_type,
            unraisable.exc_value,
            unraisable.exc_traceback,
        ),
    )
    if _original_unraisablehook not in (None, _unraisable_hook):
        _original_unraisablehook(unraisable)


def install_qt_message_logging() -> bool:
    """Route Qt warnings/errors to the application log when PySide is present."""

    global _qt_handler_installed
    with _lock:
        if _qt_handler_installed:
            return True
        try:
            from PySide6.QtCore import QtMsgType, qInstallMessageHandler
        except (ImportError, OSError):
            return False

        levels = {
            QtMsgType.QtDebugMsg: logging.DEBUG,
            QtMsgType.QtInfoMsg: logging.INFO,
            QtMsgType.QtWarningMsg: logging.WARNING,
            QtMsgType.QtCriticalMsg: logging.ERROR,
            QtMsgType.QtFatalMsg: logging.CRITICAL,
        }

        def qt_message_handler(message_type: Any, context: Any, message: str) -> None:
            category = getattr(context, "category", None) or "qt"
            location = ""
            file_name = getattr(context, "file", None)
            line_number = getattr(context, "line", 0)
            if file_name:
                location = f" ({file_name}:{line_number})"
            logging.getLogger(LOGGER_NAME).log(
                levels.get(message_type, logging.WARNING),
                "Qt[%s]%s: %s",
                category,
                location,
                message,
            )

        # Qt keeps the Python callable alive after installation.
        qInstallMessageHandler(qt_message_handler)
        _qt_handler_installed = True
        return True


def install_runtime_diagnostics(
    log_path: str | Path | None = None,
    *,
    install_qt: bool = True,
) -> Path:
    """Install append logging, fatal-error dumps, and global exception hooks.

    The function is idempotent and returns the path actually used.  If the app
    folder is not writable, that path can therefore point at the OS temporary
    directory instead of :func:`default_log_path`.
    """

    global _active_log_path, _fault_stream
    destination = _resolve_writable_log_path(log_path)
    with _lock:
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if _active_log_path != destination:
            for handler in list(logger.handlers):
                if getattr(handler, "_step_urdf_runtime_handler", False):
                    logger.removeHandler(handler)
                    handler.close()
            handler = logging.FileHandler(destination, mode="a", encoding="utf-8")
            handler._step_urdf_runtime_handler = True  # type: ignore[attr-defined]
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s [%(process)d:%(threadName)s] %(message)s"
                )
            )
            logger.addHandler(handler)

            if _fault_stream is not None:
                try:
                    faulthandler.disable()
                finally:
                    _fault_stream.close()
            _fault_stream = _open_fault_stream(destination)
            try:
                faulthandler.enable(file=_fault_stream, all_threads=True)
            except (RuntimeError, OSError):
                # Some embedded interpreters do not expose a usable file
                # descriptor.  Regular exception logging still remains active.
                logger.exception("Could not enable faulthandler")
            _active_log_path = destination

        sys.excepthook = _sys_exception_hook
        if hasattr(threading, "excepthook"):
            threading.excepthook = _thread_exception_hook
        if hasattr(sys, "unraisablehook"):
            sys.unraisablehook = _unraisable_hook

        logger.info(
            "Runtime diagnostics active (Python %s, executable=%s, cwd=%s)",
            sys.version.split()[0],
            sys.executable,
            os.getcwd(),
        )

    if install_qt:
        install_qt_message_logging()
    return destination


__all__ = [
    "LOG_FILENAME",
    "LOGGER_NAME",
    "application_directory",
    "default_log_path",
    "install_qt_message_logging",
    "install_runtime_diagnostics",
]
