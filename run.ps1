$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$runtimePython = Join-Path $projectRoot ".venv\python.exe"
if (-not (Test-Path -LiteralPath $runtimePython)) {
    throw "실행 환경이 없습니다. 먼저 .\setup.ps1 을 실행하세요."
}
$runtimePrefix = Join-Path $projectRoot ".venv"
$runtimePaths = @(
    $runtimePrefix
    (Join-Path $runtimePrefix "Library\mingw-w64\bin")
    (Join-Path $runtimePrefix "Library\usr\bin")
    (Join-Path $runtimePrefix "Library\bin")
    (Join-Path $runtimePrefix "Scripts")
    (Join-Path $runtimePrefix "bin")
) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
$env:PATH = (($runtimePaths + @($env:PATH)) -join [System.IO.Path]::PathSeparator)
$env:CONDA_PREFIX = $runtimePrefix
$env:PYTHONNOUSERSITE = "1"
& $runtimePython -m urdf_maker @args
