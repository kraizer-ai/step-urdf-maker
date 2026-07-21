from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np
import pytest

from urdf_maker.runtime_log import (
    LOGGER_NAME,
    default_log_path,
    install_runtime_diagnostics,
)
from urdf_maker.step_subprocess import (
    IsolatedStepLoadError,
    _native_exit_hint,
    _worker_command,
    load_step_project_isolated,
    run_step_import_isolated,
)


def _write_box_step(path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone


def test_runtime_diagnostics_append_to_requested_log(tmp_path: Path) -> None:
    log_path = tmp_path / "STEP_URDF_Maker.log"

    actual = install_runtime_diagnostics(log_path, install_qt=False)
    logging.getLogger(LOGGER_NAME).warning("runtime-log-test-marker")
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()

    assert actual == log_path.resolve()
    contents = log_path.read_text(encoding="utf-8")
    assert "Runtime diagnostics active" in contents
    assert "runtime-log-test-marker" in contents
    assert default_log_path().name == "STEP_URDF_Maker.log"


def test_missing_native_procedure_has_recovery_hint() -> None:
    hint = _native_exit_hint(0xC06D007F)
    assert "setup.ps1" in hint
    assert _native_exit_hint(2) == ""


def test_frozen_worker_relaunches_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / name for name in ("request", "result", "error", "stage")]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "STEP_URDF_Maker.exe"))

    command = _worker_command(*paths)

    assert command == [
        sys.executable,
        "--step-worker",
        *(str(path) for path in paths),
    ]


def test_isolated_worker_returns_robot_project(tmp_path: Path) -> None:
    source = tmp_path / "box.step"
    _write_box_step(source)

    outcome = run_step_import_isolated(source, timeout=60)

    assert outcome.diagnostics.returncode == 0
    assert outcome.diagnostics.elapsed_seconds > 0.0
    assert "MESH_READY" in outcome.diagnostics.worker_log
    assert "RESULT_READY" in outcome.diagnostics.worker_log
    assert outcome.project.source_path == str(source.resolve())
    assert outcome.project.source_kind == "step"
    assert len(outcome.project.parts) == 1
    assert outcome.project.metadata["isolated_step_import"]["parallel"] is False
    part = next(iter(outcome.project.parts.values()))
    np.testing.assert_allclose(part.bounds[0], (0.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(part.bounds[1], (0.01, 0.02, 0.03), atol=1e-9)

    # The convenience function is the API consumed by MainWindow's loader.
    project = load_step_project_isolated(source, timeout=60)
    assert len(project.parts) == 1


def test_isolated_worker_reports_python_import_error(tmp_path: Path) -> None:
    source = tmp_path / "invalid.step"
    source.write_text("this is not STEP data", encoding="utf-8")

    with pytest.raises(IsolatedStepLoadError) as caught:
        run_step_import_isolated(source, timeout=60)

    assert caught.value.returncode == 2
    assert "StepLoadError" in str(caught.value)
    assert "PYTHON_ERROR" in caught.value.worker_log
    assert str(source.resolve()) in caught.value.worker_log


def test_isolated_worker_timeout_is_terminated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "slow.step"
    source.write_text("placeholder", encoding="utf-8")

    def slow_command(*_args: object) -> list[str]:
        return [sys.executable, "-c", "import time; time.sleep(5)"]

    monkeypatch.setattr("urdf_maker.step_subprocess._worker_command", slow_command)
    with pytest.raises(IsolatedStepLoadError, match="timed out") as caught:
        run_step_import_isolated(source, timeout=0.05)

    assert caught.value.returncode is not None
