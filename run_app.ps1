Param(
    [switch]$NoInstall,
    [switch]$Upgrade
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$venvPath = Join-Path $RepoRoot '.venv'
$pythonExe = Join-Path $venvPath 'Scripts/python.exe'

function Get-SystemPython {
    # Prefer an on-PATH 'python' executable, fallback to the py launcher.
    $pythonCmd = Get-Command -Name python -ErrorAction SilentlyContinue
    if ($pythonCmd) { return 'python' }
    $pyCmd = Get-Command -Name py -ErrorAction SilentlyContinue
    if ($pyCmd) { return 'py -3' }
    return $null
}

$venvCreated = $false
if (-not (Test-Path $venvPath)) {
    Write-Host 'Creating virtual environment (.venv)...' -ForegroundColor Cyan
    $sysPy = Get-SystemPython
    if (-not $sysPy) {
        Write-Host "No 'python' or 'py' launcher found. Please install Python 3 and ensure it's on PATH." -ForegroundColor Red
        exit 1
    }
    try {
        if ($sysPy -eq 'python') {
            & python -m venv .venv
        } else {
            # 'py -3' may include args; invoke via the program name + args
            & py -3 -m venv .venv
        }
        $venvCreated = $true
    } catch {
        Write-Host "Failed to create virtual environment with ${sysPy}: ${_}" -ForegroundColor Yellow
        # Try the alternate launcher if available
        if ($sysPy -eq 'py -3') {
            $alt = Get-Command -Name python -ErrorAction SilentlyContinue
            if ($alt) {
                try { & python -m venv .venv; $venvCreated = $true } catch { }
            }
        } else {
            $alt = Get-Command -Name py -ErrorAction SilentlyContinue
            if ($alt) {
                try { & py -3 -m venv .venv; $venvCreated = $true } catch { }
            }
        }
    }
}

# Recompute pythonExe path (in case venv was created just now)
$pythonExe = Join-Path $venvPath 'Scripts/python.exe'

if (-not (Test-Path $pythonExe)) {
    Write-Host 'Python executable not found in venv.' -ForegroundColor Yellow
    $sysPy = Get-SystemPython
    if ($sysPy) {
        Write-Host "Attempting to continue using system Python launcher: $sysPy" -ForegroundColor Cyan
    } else {
        Write-Host "No usable Python found (neither .venv nor system). Aborting." -ForegroundColor Red
        exit 1
    }
}

# If the venv python exists but is a broken launcher (points to missing home), try to run a trivial command to verify it.
function Test-PythonExecutable($exe) {
    try {
        & $exe -c "import sys; print('ok')" | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (Test-Path $pythonExe) {
    if (-not (Test-PythonExecutable $pythonExe)) {
        Write-Host "Detected .venv python is not executable or points to a missing interpreter. Recreating .venv..." -ForegroundColor Yellow
        try {
            Remove-Item -Recurse -Force $venvPath -ErrorAction SilentlyContinue
        } catch {}
        $sysPy = Get-SystemPython
        if (-not $sysPy) { Write-Host "No system Python available to recreate venv. Aborting." -ForegroundColor Red; exit 1 }
        try {
            if ($sysPy -eq 'python') { & python -m venv .venv } else { & py -3 -m venv .venv }
        } catch {
            Write-Host "Failed to recreate venv with ${sysPy}: ${_}" -ForegroundColor Red
            exit 1
        }
        $pythonExe = Join-Path $venvPath 'Scripts/python.exe'
        if (-not (Test-Path $pythonExe)) { Write-Host "Recreated venv but python executable still missing. Aborting." -ForegroundColor Red; exit 1 }
    }
}

if (-not $NoInstall) {
    if (Test-Path 'requirements.txt') {
        Write-Host 'Installing dependencies...' -ForegroundColor Cyan
        $pipArgs = @('install')
        if ($Upgrade) { $pipArgs += '--upgrade' }
        $pipArgs += '-r'; $pipArgs += 'requirements.txt'
        if (Test-Path $pythonExe) {
            & $pythonExe -m pip $pipArgs
        } else {
            # Fall back to system launcher if available
            $sysPy = Get-SystemPython
            if ($sysPy -eq 'python') { & python -m pip $pipArgs } elseif ($sysPy -eq 'py -3') { & py -3 -m pip $pipArgs } else { Write-Host 'Could not run pip: no python available.' -ForegroundColor Red; exit 1 }
        }
    }
}

Write-Host 'Launching app...' -ForegroundColor Green
if (Test-Path $pythonExe) {
    & $pythonExe app.py
} else {
    $sysPy = Get-SystemPython
    if ($sysPy -eq 'python') { & python app.py }
    elseif ($sysPy -eq 'py -3') { & py -3 app.py }
    else { Write-Host 'Failed to launch: no python executable available.' -ForegroundColor Red; exit 1 }
}
