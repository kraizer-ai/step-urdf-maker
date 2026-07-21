"""Crash-isolated STEP import for the desktop application.

Open Cascade is native C++ code.  A malformed model, DLL mismatch, or driver
interaction can therefore terminate the interpreter without raising a Python
exception.  Importing in a short-lived child process keeps the Qt/VTK GUI alive
and turns such failures into a normal, reportable process exit code.

The parent and child belong to the same local installation and communicate
through a newly-created private temporary directory.  A pickle is appropriate
for this trusted same-machine boundary; callers must never replace the worker
result with an untrusted file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from time import perf_counter
import traceback
from typing import Any, Sequence

from .runtime_env import (
    console_python_executable,
    prepare_current_process,
    prepend_runtime_paths,
)

# ``model`` imports NumPy, so establish the correct native DLL search path
# before importing it in both the GUI and worker processes.
prepare_current_process()

from .model import RobotProject
from .runtime_log import LOGGER_NAME, install_runtime_diagnostics


PROTOCOL_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 900.0


class IsolatedStepLoadError(RuntimeError):
    """Raised when an isolated STEP worker fails or violates the protocol."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        worker_log: str = "",
    ) -> None:
        self.returncode = returncode
        self.worker_log = worker_log
        details = message.rstrip()
        if worker_log.strip():
            details += "\n\nWorker stage log:\n" + worker_log.strip()
        super().__init__(details)


@dataclass(frozen=True)
class StepWorkerDiagnostics:
    """Non-geometric details returned alongside a successful import."""

    returncode: int
    elapsed_seconds: float
    worker_log: str
    stdout: str = ""
    stderr: str = ""


@dataclass
class IsolatedStepResult:
    project: RobotProject
    diagnostics: StepWorkerDiagnostics


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_pickle(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_stage(path: Path, message: str) -> None:
    line = message.replace("\r", " ").replace("\n", " ").strip()
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(line + "\n")
    logging.getLogger(LOGGER_NAME).info("STEP worker: %s", line)


def _read_text(path: Path, *, max_characters: int = 12_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_characters:
        return text
    return "... (earlier output omitted) ...\n" + text[-max_characters:]


def _format_exit_code(returncode: int) -> str:
    if returncode < 0:
        return f"{returncode} (signal {-returncode})"
    if os.name == "nt" or returncode > 255:
        return f"{returncode} (0x{returncode & 0xFFFFFFFF:08X})"
    return str(returncode)


def _native_exit_hint(returncode: int) -> str:
    if returncode & 0xFFFFFFFF == 0xC06D007F:
        return (
            "Windows 네이티브 DLL에서 필요한 함수를 찾지 못했습니다. "
            "모든 STEP URDF Maker 창을 닫고 setup.ps1을 다시 실행한 뒤 재시도하세요."
        )
    return ""


def _worker_main(
    request_path: Path,
    result_path: Path,
    error_path: Path,
    stage_path: Path,
) -> int:
    """Run the native import in the child.  This function is CLI-only."""

    install_runtime_diagnostics(install_qt=False)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if int(request.get("protocol", -1)) != PROTOCOL_VERSION:
            raise ValueError("Unsupported STEP worker protocol")

        source = Path(request["source_path"]).expanduser().resolve()
        linear_deflection = float(request["linear_deflection"])
        angular_deflection = float(request["angular_deflection"])
        prefer_xcaf = bool(request["prefer_xcaf"])
        _append_stage(stage_path, f"START source={source}")
        _append_stage(
            stage_path,
            "OPTIONS "
            f"linear_deflection={linear_deflection:g} "
            f"angular_deflection={angular_deflection:g} "
            f"prefer_xcaf={prefer_xcaf} parallel=False",
        )

        # Importing here (rather than at module load time) guarantees the GUI
        # process never loads OCCT merely by importing this helper module.
        from .step_loader import load_step

        last_progress = (-1, -1, "")

        def progress(done: int, total: int, name: str) -> None:
            nonlocal last_progress
            current = (int(done), int(total), str(name))
            if current != last_progress:
                _append_stage(stage_path, f"PROGRESS {done}/{total} {name}")
                last_progress = current

        loaded = load_step(
            source,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            parallel=False,
            prefer_xcaf=prefer_xcaf,
            progress=progress,
        )
        triangle_count = sum(len(part.triangles) for part in loaded.parts)
        _append_stage(
            stage_path,
            f"MESH_READY parts={len(loaded.parts)} triangles={triangle_count}",
        )
        project = loaded.to_robot_project()
        envelope = {
            "protocol": PROTOCOL_VERSION,
            "project": project,
        }
        _atomic_write_pickle(result_path, envelope)
        _append_stage(stage_path, f"RESULT_READY bytes={result_path.stat().st_size}")
        return 0
    except BaseException as exc:
        # SystemExit/KeyboardInterrupt are captured too: the parent should see
        # one structured error instead of only a mysterious exit status.
        report = {
            "protocol": PROTOCOL_VERSION,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            _atomic_write_json(error_path, report)
            _append_stage(
                stage_path,
                f"PYTHON_ERROR {type(exc).__name__}: {str(exc) or '<no message>'}",
            )
        except BaseException:
            logging.getLogger(LOGGER_NAME).exception(
                "STEP worker could not write its structured error report"
            )
        logging.getLogger(LOGGER_NAME).exception("STEP worker failed")
        return 2


def _worker_command(
    request_path: Path,
    result_path: Path,
    error_path: Path,
    stage_path: Path,
) -> list[str]:
    return [
        console_python_executable(),
        "-m",
        "urdf_maker.step_subprocess",
        "--worker",
        str(request_path),
        str(result_path),
        str(error_path),
        str(stage_path),
    ]


def _structured_error(error_path: Path) -> str:
    try:
        payload = json.loads(error_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ""
    exception_type = str(payload.get("exception_type") or "Error")
    message = str(payload.get("message") or "STEP import failed")
    trace = str(payload.get("traceback") or "").strip()
    summary = f"{exception_type}: {message}"
    if trace:
        summary += "\n" + trace
    return summary


def run_step_import_isolated(
    path: str | Path,
    *,
    linear_deflection: float = 0.001,
    angular_deflection: float = 0.35,
    prefer_xcaf: bool = True,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> IsolatedStepResult:
    """Load a STEP file in a child process and return its project plus diagnostics.

    No OCCT module is imported in the parent. On Windows the child is created
    with ``CREATE_NO_WINDOW`` and uses the sibling ``python.exe`` even when the
    GUI itself was started with ``pythonw.exe``.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.casefold() not in {".step", ".stp"}:
        raise IsolatedStepLoadError(f"Not a STEP file: {source.name}")
    linear_deflection = float(linear_deflection)
    angular_deflection = float(angular_deflection)
    timeout = float(timeout)
    if linear_deflection <= 0.0:
        raise ValueError("linear_deflection must be positive")
    if angular_deflection <= 0.0:
        raise ValueError("angular_deflection must be positive")
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")

    started = perf_counter()
    logger = logging.getLogger(LOGGER_NAME)
    with tempfile.TemporaryDirectory(prefix="step_urdf_import_") as raw_directory:
        directory = Path(raw_directory)
        request_path = directory / "request.json"
        result_path = directory / "result.pickle"
        error_path = directory / "error.json"
        stage_path = directory / "worker-stage.log"
        stdout_path = directory / "stdout.log"
        stderr_path = directory / "stderr.log"
        _atomic_write_json(
            request_path,
            {
                "protocol": PROTOCOL_VERSION,
                "source_path": str(source),
                "linear_deflection": linear_deflection,
                "angular_deflection": angular_deflection,
                "prefer_xcaf": bool(prefer_xcaf),
            },
        )

        command = _worker_command(request_path, result_path, error_path, stage_path)
        environment = os.environ.copy()
        prepend_runtime_paths(environment)
        environment.setdefault("PYTHONUTF8", "1")
        # A second safety belt in addition to BRepMesh's parallel=False.  These
        # settings affect only the short-lived worker, never the Qt/VTK process.
        environment["OMP_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        environment["NUMEXPR_NUM_THREADS"] = "1"
        creation_flags = 0
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        logger.info("Starting isolated STEP worker for %s", source)
        with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                env=environment,
                creationflags=creation_flags,
                close_fds=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                elapsed = perf_counter() - started
                worker_log = _read_text(stage_path)
                logger.error(
                    "Isolated STEP worker timed out after %.1f seconds\n%s",
                    elapsed,
                    worker_log,
                )
                raise IsolatedStepLoadError(
                    f"STEP import timed out after {elapsed:.1f} seconds.",
                    returncode=process.returncode,
                    worker_log=worker_log,
                ) from None

        elapsed = perf_counter() - started
        worker_log = _read_text(stage_path)
        stdout = _read_text(stdout_path)
        stderr = _read_text(stderr_path)
        if returncode != 0:
            structured = _structured_error(error_path)
            exit_text = _format_exit_code(returncode)
            message = f"Isolated STEP worker exited with code {exit_text}."
            hint = _native_exit_hint(returncode)
            if hint:
                message += "\n" + hint
            if structured:
                message += "\n" + structured
            if stderr.strip():
                message += "\nWorker stderr:\n" + stderr.strip()
            logger.error("%s\n%s", message, worker_log)
            raise IsolatedStepLoadError(
                message,
                returncode=returncode,
                worker_log=worker_log,
            )
        if not result_path.is_file():
            raise IsolatedStepLoadError(
                "STEP worker reported success but produced no result file.",
                returncode=returncode,
                worker_log=worker_log,
            )

        try:
            with result_path.open("rb") as stream:
                envelope = pickle.load(stream)
        except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError) as exc:
            raise IsolatedStepLoadError(
                f"Could not read STEP worker result: {exc}",
                returncode=returncode,
                worker_log=worker_log,
            ) from exc
        if not isinstance(envelope, dict) or envelope.get("protocol") != PROTOCOL_VERSION:
            raise IsolatedStepLoadError(
                "STEP worker returned an incompatible result protocol.",
                returncode=returncode,
                worker_log=worker_log,
            )
        project = envelope.get("project")
        if not isinstance(project, RobotProject):
            raise IsolatedStepLoadError(
                "STEP worker result did not contain a RobotProject.",
                returncode=returncode,
                worker_log=worker_log,
            )

        diagnostics = StepWorkerDiagnostics(
            returncode=returncode,
            elapsed_seconds=elapsed,
            worker_log=worker_log,
            stdout=stdout,
            stderr=stderr,
        )
        project.metadata["isolated_step_import"] = {
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "parallel": False,
        }
        logger.info(
            "Isolated STEP worker completed in %.2f seconds (%d parts)",
            elapsed,
            len(project.parts),
        )
        return IsolatedStepResult(project=project, diagnostics=diagnostics)


def load_step_project_isolated(path: str | Path, **kwargs: Any) -> RobotProject:
    """GUI-friendly wrapper returning only the imported :class:`RobotProject`."""

    return run_step_import_isolated(path, **kwargs).project


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal isolated STEP import worker")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("request", nargs="?")
    parser.add_argument("result", nargs="?")
    parser.add_argument("error", nargs="?")
    parser.add_argument("stage", nargs="?")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.worker or not all((args.request, args.result, args.error, args.stage)):
        raise SystemExit("This module is an internal STEP import worker.")
    return _worker_main(
        Path(args.request),
        Path(args.result),
        Path(args.error),
        Path(args.stage),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "IsolatedStepLoadError",
    "IsolatedStepResult",
    "PROTOCOL_VERSION",
    "StepWorkerDiagnostics",
    "load_step_project_isolated",
    "run_step_import_isolated",
]
