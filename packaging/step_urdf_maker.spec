# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(SPECPATH).parent


datas = []
binaries = []
hiddenimports = [
    "cadquery_ocp_proxy",
    "OCP.OCP",
    "OCP.BRep",
    "OCP.BRepMesh",
    "OCP.IFSelect",
    "OCP.Quantity",
    "OCP.STEPCAFControl",
    "OCP.STEPControl",
    "OCP.TCollection",
    "OCP.TColStd",
    "OCP.TDF",
    "OCP.TDataStd",
    "OCP.TDocStd",
    "OCP.TopAbs",
    "OCP.TopExp",
    "OCP.TopLoc",
    "OCP.TopoDS",
    "OCP.XCAFApp",
    "OCP.XCAFDoc",
]

# OCP exposes hundreds of Python namespaces from one native extension and
# imports the namespaces needed by STEP loading lazily. Static analysis alone
# cannot see them. Its wheel also keeps OCCT libraries in a sibling *.libs
# directory, which must retain that relative layout in the portable bundle.
ocp_spec = importlib.util.find_spec("OCP")
if ocp_spec is None or ocp_spec.origin is None:
    raise RuntimeError("OCP is not installed in the build environment")
site_packages = Path(ocp_spec.origin).parent.parent
for library_dir in site_packages.glob("cadquery_ocp*.libs"):
    for library in library_dir.rglob("*"):
        if library.is_file():
            relative_parent = library.relative_to(library_dir).parent
            destination = Path(library_dir.name) / relative_parent
            binaries.append((str(library), str(destination)))

# Conda's PySide6 build places its Qt and ABI DLLs in Library/bin while the pip
# wheel keeps them beside the Python extensions.  CI uses wheels, but accepting
# both layouts lets maintainers produce a verified Windows build locally too.
conda_binary_dir = Path(sys.prefix) / "Library" / "bin"
if conda_binary_dir.is_dir():
    for pattern in ("Qt6*.dll", "pyside6*.dll", "shiboken6*.dll"):
        for library in conda_binary_dir.glob(pattern):
            binaries.append((str(library), "."))

a = Analysis(
    [str(ROOT / "urdf_maker" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="STEP_URDF_Maker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="STEP_URDF_Maker",
)
