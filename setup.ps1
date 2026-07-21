$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$environmentPath = Join-Path $projectRoot ".venv"
$environmentFile = Join-Path $projectRoot "environment.yml"

if (Test-Path -LiteralPath (Join-Path $environmentPath "conda-meta\history")) {
    $existingPython = Join-Path $environmentPath "python.exe"
    if (Test-Path -LiteralPath $existingPython) {
        # The original cadquery-ocp wheel bundles OCCT's VTK bridge. Keeping it
        # beside the viewport VTK can cause a native DLL procedure conflict.
        & $existingPython -m pip uninstall -y cadquery-ocp | Out-Null
    }
    conda env update --prefix $environmentPath --file $environmentFile --prune
} else {
    conda env create --prefix $environmentPath --file $environmentFile
}

Write-Host "설치 완료. .\run.ps1 로 실행하세요."
