"""Keep native DLL resolution inside the application's Python environment.

The project runtime is a Conda prefix, but the desktop launcher deliberately
does not require an activated shell.  On Windows that means ``Library\\bin``
is otherwise absent from ``PATH`` and a worker can accidentally load DLLs from
an already-active base Conda environment.  Native packages such as NumPy,
OpenCascade, Qt, and VTK must all resolve against this prefix instead.
"""

from __future__ import annotations

from collections.abc import MutableMapping
import ctypes
import os
from pathlib import Path
import sys
from typing import Any


_DLL_DIRECTORY_HANDLES: dict[str, Any] = {}
_SYSTEM_OPENGL_HANDLE: Any | None = None


def preload_system_opengl() -> Path | None:
    """Prefer the vendor OpenGL driver over Conda's bundled LLVMpipe DLL.

    The Windows Conda VTK stack includes a Mesa ``opengl32.dll`` beside its
    native dependencies. If that DLL wins resolution, Mesa may fall back to
    the CPU-only LLVMpipe renderer even when a capable GPU is installed.
    Loading the system OpenGL dispatcher by absolute path before importing VTK
    lets Windows reuse it for VTK and reach the installed AMD/NVIDIA/Intel ICD.
    """

    global _SYSTEM_OPENGL_HANDLE
    if os.name != "nt":
        return None
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    system_opengl = system_root / "System32" / "opengl32.dll"
    if _SYSTEM_OPENGL_HANDLE is not None:
        return system_opengl
    if not system_opengl.is_file():
        return None
    try:
        _SYSTEM_OPENGL_HANDLE = ctypes.WinDLL(str(system_opengl))
    except (AttributeError, OSError):
        return None
    return system_opengl


def runtime_binary_directories(prefix: str | Path | None = None) -> tuple[Path, ...]:
    """Return existing executable/DLL directories for a Python prefix."""

    root = Path(prefix or sys.prefix).expanduser().resolve()
    candidates = (
        root,
        root / "Library" / "mingw-w64" / "bin",
        root / "Library" / "usr" / "bin",
        root / "Library" / "bin",
        root / "Scripts",
        root / "bin",
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def prepend_runtime_paths(
    environment: MutableMapping[str, str] | None = None,
    *,
    prefix: str | Path | None = None,
) -> tuple[Path, ...]:
    """Prepend the runtime prefix to one process environment's ``PATH``."""

    target = os.environ if environment is None else environment
    root = Path(prefix or sys.prefix).expanduser().resolve()
    runtime_entries = runtime_binary_directories(root)
    existing = target.get("PATH", "").split(os.pathsep)
    combined: list[str] = []
    seen: set[str] = set()
    for raw in [*(str(path) for path in runtime_entries), *existing]:
        value = str(raw).strip().strip('"')
        if not value:
            continue
        key = os.path.normcase(os.path.abspath(os.path.expandvars(value)))
        if key in seen:
            continue
        seen.add(key)
        combined.append(value)
    target["PATH"] = os.pathsep.join(combined)
    target["PYTHONNOUSERSITE"] = "1"
    if (root / "conda-meta").is_dir():
        target["CONDA_PREFIX"] = str(root)
    return runtime_entries


def prepare_current_process(prefix: str | Path | None = None) -> tuple[Path, ...]:
    """Prepare ``PATH`` and Windows' explicit DLL search directories."""

    # This must happen before any VTK OpenGL DLL is imported.
    preload_system_opengl()
    entries = prepend_runtime_paths(prefix=prefix)
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in entries:
            key = os.path.normcase(str(directory))
            if key in _DLL_DIRECTORY_HANDLES:
                continue
            try:
                _DLL_DIRECTORY_HANDLES[key] = os.add_dll_directory(str(directory))
            except OSError:
                # PATH remains a valid fallback when one optional directory
                # cannot be registered explicitly.
                continue
    return entries


def console_python_executable(executable: str | Path | None = None) -> str:
    """Use ``python.exe`` for hidden workers even when the GUI uses pythonw."""

    current = Path(executable or sys.executable).expanduser().resolve()
    if os.name == "nt" and current.name.casefold() == "pythonw.exe":
        console = current.with_name("python.exe")
        if console.is_file():
            return str(console)
    return str(current)


__all__ = [
    "console_python_executable",
    "preload_system_opengl",
    "prepare_current_process",
    "prepend_runtime_paths",
    "runtime_binary_directories",
]
