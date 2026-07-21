from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "step_urdf_maker.spec"
VERSION = "0.1.4"


def platform_slug() -> str:
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "windows-x64"
    if sys.platform.startswith("linux") and machine in {"amd64", "x86_64"}:
        return "linux-x64"
    if sys.platform.startswith("linux") and machine in {"aarch64", "arm64"}:
        return "linux-arm64"
    raise RuntimeError(f"Unsupported portable target: {sys.platform}/{machine}")


def copy_distribution_files(bundle: Path) -> None:
    shutil.copy2(ROOT / "README.md", bundle / "README.md")
    portable_readme = bundle / "README_PORTABLE.txt"
    portable_readme.write_text(
        "STEP URDF Maker portable build\n"
        "================================\n\n"
        "Windows: STEP_URDF_Maker.exe 를 더블클릭하세요.\n"
        "Linux:   ./STEP_URDF_Maker.sh 를 실행하세요.\n\n"
        "Python, Conda 설치는 필요하지 않습니다. Linux는 Ubuntu 24.04 이상과 "
        "OpenGL 지원 데스크톱 환경이 필요합니다.\n",
        encoding="utf-8",
    )

    examples = bundle / "examples"
    examples.mkdir(exist_ok=True)
    for source in (
        ROOT / "data" / "wsr-0002898 (simplified extended) 2025-11-23.STEP",
        ROOT / "data" / "wsr-0002898_simplified_extended_2025-11-23.urdfmaker.json",
    ):
        if source.is_file():
            shutil.copy2(source, examples / source.name)

    if sys.platform.startswith("linux"):
        launcher = bundle / "STEP_URDF_Maker.sh"
        launcher.write_text(
            "#!/usr/bin/env sh\n"
            "set -eu\n"
            "APP_DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
            "exec \"$APP_DIR/STEP_URDF_Maker\" \"$@\"\n",
            encoding="utf-8",
            newline="\n",
        )
        launcher.chmod(
            launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )


def smoke_test(bundle: Path) -> None:
    executable = bundle / (
        "STEP_URDF_Maker.exe" if sys.platform == "win32" else "STEP_URDF_Maker"
    )
    diagnostics = bundle / "portable-smoke-test.txt"
    completed = subprocess.run(
        [str(executable), "--portable-smoke-test"],
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = (
            diagnostics.read_text(encoding="utf-8", errors="replace")
            if diagnostics.is_file()
            else ""
        )
        raise RuntimeError(
            f"Portable smoke test failed with code {completed.returncode}.\n{detail}"
        )
    diagnostics.unlink(missing_ok=True)


def make_archive(bundle: Path, output_dir: Path, slug: str) -> Path:
    archive_base = f"step-urdf-maker-{VERSION}-{slug}"
    if sys.platform == "win32":
        archive = output_dir / f"{archive_base}.zip"
        with zipfile.ZipFile(
            archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as stream:
            for path in bundle.rglob("*"):
                if path.is_file():
                    stream.write(path, Path(archive_base) / path.relative_to(bundle))
    else:
        archive = output_dir / f"{archive_base}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=9) as stream:
            stream.add(bundle, arcname=archive_base)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the native portable archive")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release")
    parser.add_argument("--skip-smoke-test", action="store_true")
    args = parser.parse_args()

    slug = platform_slug()
    build_dir = ROOT / "build" / f"pyinstaller-{slug}"
    dist_dir = ROOT / "dist" / slug
    build_environment = os.environ.copy()
    if sys.platform == "win32":
        conda_bin = Path(sys.prefix) / "Library" / "bin"
        if conda_bin.is_dir():
            build_environment["PATH"] = (
                str(conda_bin)
                + os.pathsep
                + build_environment.get("PATH", "")
            )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--workpath",
            str(build_dir),
            "--distpath",
            str(dist_dir),
            str(SPEC),
        ],
        cwd=ROOT,
        env=build_environment,
        check=True,
    )

    bundle = dist_dir / "STEP_URDF_Maker"
    copy_distribution_files(bundle)
    if not args.skip_smoke_test:
        smoke_test(bundle)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = make_archive(bundle, output_dir, slug)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
