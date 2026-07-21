from __future__ import annotations

import os
import sys
from pathlib import Path

from .runtime_env import prepare_current_process

# This must run before importing Qt/VTK so their dependent DLLs come from the
# project prefix even when the user launches without ``conda activate``.
prepare_current_process()

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from .runtime_log import install_runtime_diagnostics


def _configure_application(app: QApplication) -> None:
    QCoreApplication.setOrganizationName("AIK")
    QCoreApplication.setApplicationName("STEP URDF Maker")
    app.setApplicationDisplayName("STEP URDF Maker")
    app.setStyle("Fusion")

    font = QFont(app.font())
    if os.name == "nt":
        font.setFamilies(["Malgun Gothic", "Segoe UI"])
    font.setPointSize(10)
    app.setFont(font)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    install_runtime_diagnostics()
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(args)
    _configure_application(app)

    from .ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    if len(args) > 1:
        source = Path(args[1]).expanduser()
        if source.exists():
            window.open_path(source)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
