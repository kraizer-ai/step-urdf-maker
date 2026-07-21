from __future__ import annotations

import os
from pathlib import Path

from urdf_maker.runtime_env import (
    console_python_executable,
    preload_system_opengl,
    prepend_runtime_paths,
    runtime_binary_directories,
)


def test_runtime_paths_are_prepended_and_deduplicated(tmp_path: Path) -> None:
    prefix = tmp_path / "runtime"
    library_bin = prefix / "Library" / "bin"
    scripts = prefix / "Scripts"
    library_bin.mkdir(parents=True)
    scripts.mkdir()
    (prefix / "conda-meta").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    environment = {
        "PATH": os.pathsep.join((str(external), str(library_bin))),
        "CONDA_PREFIX": "wrong-prefix",
    }

    entries = prepend_runtime_paths(environment, prefix=prefix)

    assert entries == runtime_binary_directories(prefix)
    path_entries = environment["PATH"].split(os.pathsep)
    assert path_entries[:3] == [str(prefix.resolve()), str(library_bin), str(scripts)]
    assert path_entries.count(str(library_bin)) == 1
    assert environment["CONDA_PREFIX"] == str(prefix.resolve())
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_console_python_replaces_pythonw_on_windows(tmp_path: Path) -> None:
    pythonw = tmp_path / "pythonw.exe"
    python = tmp_path / "python.exe"
    pythonw.touch()
    python.touch()

    actual = console_python_executable(pythonw)

    expected = python.resolve() if os.name == "nt" else pythonw.resolve()
    assert Path(actual) == expected


def test_system_opengl_preload_is_platform_safe() -> None:
    loaded = preload_system_opengl()

    if os.name == "nt":
        assert loaded is not None
        assert loaded.name.casefold() == "opengl32.dll"
        assert loaded.parent.name.casefold() == "system32"
    else:
        assert loaded is None
