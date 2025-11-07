#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Builds a portable executable of the Mood Palette Generator app

.DESCRIPTION
    This script creates a standalone, portable executable using PyInstaller.
    The resulting executable can be run on Windows machines without requiring Python installation.

.PARAMETER Clean
    Clean the build directories before building

.PARAMETER SkipInstall
    Skip installing build dependencies

.PARAMETER Debug
    Create a debug build with console output

.PARAMETER OneFile
    Create a single executable file (slower startup but more portable)

.EXAMPLE
    .\build_portable.ps1
    .\build_portable.ps1 -Clean -Debug
    .\build_portable.ps1 -OneFile
#>

Param(
    [switch]$Clean,
    [switch]$SkipInstall,
    [switch]$Debug,
    [switch]$OneFile
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "=== Mood Palette Generator Portable Build ===" -ForegroundColor Cyan

# Activate virtual environment if it exists
$venvPath = Join-Path $RepoRoot '.venv'
$pythonExe = Join-Path $venvPath 'Scripts/python.exe'

function Get-WorkingPython {
    # Test venv python first
    if (Test-Path $pythonExe) {
        try {
            & $pythonExe -c "import sys; print('ok')" 2>$null | Out-Null
            return $pythonExe
        } catch {}
    }
    
    # Try system python commands
    foreach ($cmd in @("python", "py", "py -3")) {
        try {
            & $cmd -c "import sys; print('ok')" 2>$null | Out-Null
            return $cmd
        } catch {}
    }
    return $null
}

$workingPython = Get-WorkingPython
if ($workingPython) {
    if ($workingPython -eq $pythonExe) {
        Write-Host "Using virtual environment: $venvPath" -ForegroundColor Green
        $env:PATH = "$(Join-Path $venvPath 'Scripts');$env:PATH"
    } else {
        Write-Host "Using system Python: $workingPython" -ForegroundColor Yellow
        $pythonExe = $workingPython
    }
} else {
    Write-Host "No working Python installation found!" -ForegroundColor Red
    exit 1
}

# Clean build directories if requested
if ($Clean) {
    Write-Host "Cleaning build directories..." -ForegroundColor Yellow
    Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "dist" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "*.spec" -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "app_portable.spec" }
    Remove-Item -Path "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
}

# Install build dependencies
if (-not $SkipInstall) {
    Write-Host "Installing build dependencies..." -ForegroundColor Cyan
    try {
        & $pythonExe -m pip install --upgrade pip
        & $pythonExe -m pip install -r requirements.txt
        & $pythonExe -m pip install -r requirements-build.txt
    } catch {
        Write-Host "Failed to install dependencies: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Verify PyInstaller is available
try {
    & $pythonExe -m PyInstaller --version | Out-Null
    Write-Host "PyInstaller is ready" -ForegroundColor Green
} catch {
    Write-Host "PyInstaller not found. Please install it with: pip install pyinstaller" -ForegroundColor Red
    exit 1
}

# Prepare spec file
$specFile = "app_portable.spec"
if (-not (Test-Path $specFile)) {
    Write-Host "Spec file not found: $specFile" -ForegroundColor Red
    exit 1
}

# Modify spec file for different build types
$specContent = Get-Content $specFile -Raw

if ($Debug) {
    Write-Host "Building DEBUG version (with console)" -ForegroundColor Yellow
    $specContent = $specContent -replace "console=True", "console=True"
    $specContent = $specContent -replace "debug=False", "debug=True"
} else {
    Write-Host "Building RELEASE version" -ForegroundColor Green
    $specContent = $specContent -replace "console=True", "console=False"
    $specContent = $specContent -replace "debug=True", "debug=False"
}

if ($OneFile) {
    Write-Host "Creating ONE-FILE executable (slower startup)" -ForegroundColor Yellow
    # Modify the spec to create a one-file build
    $specContent = $specContent -replace "exclude_binaries=True", "exclude_binaries=False"
    $specContent = $specContent -replace "COLLECT\(", "# COLLECT("
    $buildSpecFile = "app_portable_onefile.spec"
} else {
    Write-Host "Creating DIRECTORY-based executable (faster startup)" -ForegroundColor Yellow
    $buildSpecFile = $specFile
}

# Write modified spec if needed
if ($OneFile) {
    Set-Content -Path $buildSpecFile -Value $specContent
}

# Run PyInstaller
Write-Host "Running PyInstaller..." -ForegroundColor Cyan
try {
    $buildArgs = @(
        "--clean",
        "--noconfirm"
    )
    
    if ($OneFile -and -not $buildSpecFile.EndsWith("_onefile.spec")) {
        $buildArgs += "--onefile"
        $buildArgs += "app.py"
    } else {
        $buildArgs += $buildSpecFile
    }
    
    & $pythonExe -m PyInstaller @buildArgs
    
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
    
    Write-Host "Build completed successfully!" -ForegroundColor Green
    
} catch {
    Write-Host "Build failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Check output
$distPath = Join-Path $RepoRoot "dist"
if (Test-Path $distPath) {
    Write-Host "Build output location: $distPath" -ForegroundColor Green
    
    # List contents
    Write-Host "Contents:" -ForegroundColor Cyan
    Get-ChildItem $distPath | ForEach-Object {
        Write-Host "  $($_.Name)" -ForegroundColor White
    }
    
    # Find executable
    $exePath = Get-ChildItem $distPath -Recurse -Filter "*.exe" | Select-Object -First 1
    if ($exePath) {
        Write-Host "Executable: $($exePath.FullName)" -ForegroundColor Green
        
        # Get size info
        $sizeGB = [math]::Round($exePath.Length / 1GB, 2)
        $sizeMB = [math]::Round($exePath.Length / 1MB, 2)
        Write-Host "Size: $sizeMB MB" -ForegroundColor Yellow
        
        # Create launcher script
        $launcherPath = Join-Path $distPath "Launch_MoodPalette.ps1"
        $launcherContent = @"
#!/usr/bin/env pwsh
# Mood Palette Generator Launcher
# This script helps launch the portable app with proper configuration

`$ErrorActionPreference = 'Stop'
`$AppDir = Split-Path -Parent `$MyInvocation.MyCommand.Path

Write-Host "=== Mood Palette Generator (Portable) ===" -ForegroundColor Cyan
Write-Host "Starting application..." -ForegroundColor Green

# Set working directory to app directory
Set-Location `$AppDir

# Find the executable
`$exeFile = Get-ChildItem -Path `$AppDir -Recurse -Filter "*.exe" | Select-Object -First 1

if (-not `$exeFile) {
    Write-Host "Executable not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

try {
    # Launch the app
    & `$exeFile.FullName
} catch {
    Write-Host "Failed to start application: `$(`$_.Exception.Message)" -ForegroundColor Red
    Write-Host "Try running the executable directly: `$(`$exeFile.FullName)" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}
"@
        Set-Content -Path $launcherPath -Value $launcherContent
        Write-Host "Created launcher: $launcherPath" -ForegroundColor Green
        
    } else {
        Write-Host "Warning: No executable found in output directory" -ForegroundColor Yellow
    }
    
} else {
    Write-Host "Warning: dist directory not found" -ForegroundColor Yellow
}

# Cleanup temporary files
if ($OneFile -and (Test-Path $buildSpecFile) -and $buildSpecFile -ne $specFile) {
    Remove-Item $buildSpecFile -Force
}

Write-Host "=== Build Complete ===" -ForegroundColor Cyan
Write-Host "To distribute your portable app:" -ForegroundColor Yellow
Write-Host "1. Copy the entire 'dist' folder to the target machine" -ForegroundColor White
Write-Host "2. Run the executable or use the launcher script" -ForegroundColor White
Write-Host "3. The app will work without Python installation" -ForegroundColor White