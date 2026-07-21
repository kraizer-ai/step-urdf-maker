@echo off
setlocal
cd /d "%~dp0"

set "APP_PYTHONW=%~dp0.venv\pythonw.exe"
set "APP_ENV=%~dp0.venv"
if not exist "%APP_PYTHONW%" (
    echo STEP URDF Maker runtime was not found.
    echo Run setup.ps1 once from PowerShell, then try again.
    echo.
    pause
    exit /b 1
)

rem Reproduce the essential DLL search order of `conda activate` without
rem requiring the user to open an activated terminal first.
set "PATH=%APP_ENV%;%APP_ENV%\Library\mingw-w64\bin;%APP_ENV%\Library\usr\bin;%APP_ENV%\Library\bin;%APP_ENV%\Scripts;%APP_ENV%\bin;%PATH%"
set "CONDA_PREFIX=%APP_ENV%"
set "PYTHONNOUSERSITE=1"
start "STEP URDF Maker" "%APP_PYTHONW%" -m urdf_maker %*
endlocal
