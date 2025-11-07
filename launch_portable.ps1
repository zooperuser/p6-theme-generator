#!/usr/bin/env pwsh
<#!
.SYNOPSIS
    Portable Image Palette Generator Launcher

.DESCRIPTION
    This launcher script helps run the portable Image Palette Generator.
    It sets up the environment and handles common issues.

.PARAMETER Debug
    Run in debug mode with console output

.PARAMETER ConfigDir
    Specify a custom configuration directory
#>

Param(
    [switch]$Debug,
    [string]$ConfigDir = ""
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Set console title
$Host.UI.RawUI.WindowTitle = "Image Palette Generator (Portable)"

Write-Host "=== Image Palette Generator (Portable) ===" -ForegroundColor Cyan
Write-Host "Version: Portable Build" -ForegroundColor Yellow
Write-Host "Starting application..." -ForegroundColor Green

# Set working directory to script directory
Set-Location $ScriptDir

# Find the executable
$exeFile = $null
$searchPaths = @(
    $ScriptDir,
    (Join-Path $ScriptDir "MoodPaletteGenerator"),
    (Join-Path $ScriptDir "dist"),
    (Join-Path $ScriptDir "dist\MoodPaletteGenerator")
)

foreach ($path in $searchPaths) {
    if (Test-Path $path) {
        $found = Get-ChildItem -Path $path -Filter "*.exe" -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*Mood*" -or $_.Name -like "*app*" } | Select-Object -First 1
        if ($found) {
            $exeFile = $found
            Write-Host "Found executable: $($found.FullName)" -ForegroundColor Green
            break
        }
    }
}

if (-not $exeFile) {
    Write-Host "Executable not found in any of these locations:" -ForegroundColor Red
    foreach ($path in $searchPaths) {
        Write-Host "  $path" -ForegroundColor Yellow
    }
    
    # Try to find any exe file as fallback
    $anyExe = Get-ChildItem -Path $ScriptDir -Recurse -Filter "*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($anyExe) {
        Write-Host "Found alternative executable: $($anyExe.FullName)" -ForegroundColor Yellow
        $response = Read-Host "Use this executable? (y/N)"
        if ($response -eq 'y' -or $response -eq 'Y') {
            $exeFile = $anyExe
        }
    }
    
    if (-not $exeFile) {
        Write-Host "No executable found. Please ensure the portable app was built correctly." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Set up environment variables for the portable app
if ($ConfigDir) {
    $env:APPDATA_OVERRIDE = $ConfigDir
    Write-Host "Using custom config directory: $ConfigDir" -ForegroundColor Yellow
} else {
    # Create a portable data directory
    $portableDataDir = Join-Path $ScriptDir "PortableData"
    if (-not (Test-Path $portableDataDir)) {
        New-Item -ItemType Directory -Path $portableDataDir -Force | Out-Null
    }
    $env:APPDATA_OVERRIDE = $portableDataDir
    Write-Host "Using portable data directory: $portableDataDir" -ForegroundColor Green
}

# Set LM Studio defaults for portable mode
if (-not $env:LM_STUDIO_BASE_URL) {
    $env:LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
}

# Create a log file for this session
$logFile = Join-Path $env:APPDATA_OVERRIDE "image_palette_log.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Function to write to log
function Write-Log {
    param($Message, $Level = "INFO")
    $logEntry = "[$timestamp] [$Level] $Message"
    Add-Content -Path $logFile -Value $logEntry -ErrorAction SilentlyContinue
}

Write-Log "Starting Image Palette Generator Portable"
Write-Log "Executable: $($exeFile.FullName)"
Write-Log "Working Directory: $ScriptDir"
Write-Log "Data Directory: $env:APPDATA_OVERRIDE"

try {
    # Launch the app
    Write-Host "Launching application..." -ForegroundColor Cyan
    Write-Host "Data will be stored in: $env:APPDATA_OVERRIDE" -ForegroundColor Yellow
    Write-Host "Log file: $logFile" -ForegroundColor Yellow
    
    if ($Debug) {
        Write-Host "Running in DEBUG mode" -ForegroundColor Yellow
        Write-Host "Executable path: $($exeFile.FullName)" -ForegroundColor Gray
        Write-Host "Environment variables:" -ForegroundColor Gray
        Write-Host "  LM_STUDIO_BASE_URL = $env:LM_STUDIO_BASE_URL" -ForegroundColor Gray
        Write-Host "  APPDATA_OVERRIDE = $env:APPDATA_OVERRIDE" -ForegroundColor Gray
    }
    
    Write-Log "Launching executable"
    
    # Start the process and wait
    $process = Start-Process -FilePath $exeFile.FullName -WorkingDirectory $ScriptDir -PassThru -NoNewWindow:$(-not $Debug)
    
    if ($process) {
        Write-Host "Application started (PID: $($process.Id))" -ForegroundColor Green
        Write-Log "Application started successfully (PID: $($process.Id))"
        
        if (-not $Debug) {
            Write-Host "Application is running in the background." -ForegroundColor Green
            Write-Host "Check your web browser for the Gradio interface (usually http://127.0.0.1:7860)" -ForegroundColor Yellow
            Write-Host "Press Ctrl+C to stop this launcher (the app will continue running)" -ForegroundColor Gray
        }
        
        # Wait for process if in debug mode, otherwise just monitor
        if ($Debug) {
            $process.WaitForExit()
            Write-Log "Application exited with code: $($process.ExitCode)"
        } else {
            # Give it a moment to start up
            Start-Sleep -Seconds 3
            if (-not $process.HasExited) {
                Write-Host "Application appears to be running successfully!" -ForegroundColor Green
            } else {
                Write-Host "Application exited quickly. Check the log for details." -ForegroundColor Yellow
                Write-Log "Application exited quickly with code: $($process.ExitCode)"
            }
        }
    }
    
} catch {
    $errorMsg = "Failed to start application: $($_.Exception.Message)"
    Write-Host $errorMsg -ForegroundColor Red
    Write-Log $errorMsg "ERROR"
    
    Write-Host "`nTroubleshooting tips:" -ForegroundColor Yellow
    Write-Host "1. Ensure all files from the 'dist' folder are present" -ForegroundColor White
    Write-Host "2. Try running the executable directly: $($exeFile.FullName)" -ForegroundColor White
    Write-Host "3. Check Windows Defender or antivirus settings" -ForegroundColor White
    Write-Host "4. Ensure you have proper permissions to run executables" -ForegroundColor White
    Write-Host "5. Check the log file: $logFile" -ForegroundColor White
    
    Read-Host "`nPress Enter to exit"
}

Write-Log "Launcher session ended"