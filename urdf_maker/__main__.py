from __future__ import annotations

import sys
from pathlib import Path
import traceback


def main() -> int:
    """Dispatch the desktop app and frozen-only maintenance entry points."""

    arguments = sys.argv[1:]
    if arguments[:1] == ["--step-worker"]:
        # A PyInstaller app has no sibling Python interpreter.  The isolated
        # STEP loader therefore starts this executable again and reaches the
        # same worker implementation used by ``python -m`` in a source tree.
        from urdf_maker.step_subprocess import main as worker_main

        return worker_main(["--worker", *arguments[1:]])
    if arguments == ["--portable-smoke-test"]:
        # Keep CI smoke tests independent of an X server/display while still
        # proving that all three large native runtimes were bundled.
        diagnostics = Path(sys.executable).resolve().parent / "portable-smoke-test.txt"

        def record(message: str) -> None:
            with diagnostics.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

        diagnostics.write_text("START\n", encoding="utf-8")
        try:
            from urdf_maker.runtime_env import prepare_current_process

            prepare_current_process()
            record("runtime environment: OK")
            import PySide6.QtWidgets  # noqa: F401

            record("PySide6: OK")
            import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

            record("VTK: OK")
            import OCP.STEPControl  # noqa: F401

            record("OCP: OK")
        except BaseException:
            record(traceback.format_exc())
            return 1
        return 0

    from urdf_maker.app import main as application_main

    return application_main()


if __name__ == "__main__":
    raise SystemExit(main())
