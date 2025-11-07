"""
ZIP-based Portable App Creator
Creates a simple portable solution using ZIP app format
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_zip_portable():
    """Create a ZIP-based portable application."""
    
    print("Creating ZIP-based portable app...")
    
    # Configuration
    current_dir = Path.cwd()
    dist_dir = current_dir / "dist_zip"
    
    # Clean and create
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()
    
    # Step 1: Create app bundle ZIP
    print("Creating app bundle...")
    app_zip = dist_dir / "mood_palette_app.pyz"
    
    with zipfile.ZipFile(app_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add main app file as __main__.py so it can be run with python -m
        with open('app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        # Modify the app content to be more portable
        portable_content = f'''#!/usr/bin/env python3
"""
Mood Palette Generator - Portable Version
Entry point for ZIP-based portable application
"""

{app_content}
'''
        zf.writestr('__main__.py', portable_content)
        
        # Add any additional files if needed
        if os.path.exists('requirements.txt'):
            zf.write('requirements.txt', 'requirements.txt')
    
    print(f"Created app bundle: {app_zip}")
    
    # Step 2: Create Windows launcher batch file
    launcher_bat = dist_dir / "MoodPaletteGenerator_ZIP.bat"
    launcher_bat.write_text(f'''@echo off
echo Mood Palette Generator - ZIP Portable Version
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or later from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Using Python %PYTHON_VERSION%

REM Install required packages if not available
echo Checking/Installing required packages...
python -m pip install --quiet gradio>=5.0.0 pillow numpy pandas requests openai python-dotenv fastapi uvicorn

if errorlevel 1 (
    echo Warning: Some packages might not have installed correctly
    echo The app may still work with existing packages
    echo.
)

REM Run the app
echo Starting Mood Palette Generator...
echo.
python "%~dp0mood_palette_app.pyz"

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit.
    pause >nul
)
''')
    
    # Step 3: Create PowerShell launcher
    launcher_ps1 = dist_dir / "MoodPaletteGenerator_ZIP.ps1"
    launcher_ps1.write_text(f'''# Mood Palette Generator - ZIP Portable Version
Write-Host "Mood Palette Generator - ZIP Portable Version" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Check if Python is available
try {{
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {{
        throw "Python not found"
    }}
    Write-Host "Using: $pythonVersion" -ForegroundColor Cyan
}} catch {{
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.8 or later from:"
    Write-Host "https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Make sure to check 'Add Python to PATH' during installation"
    Read-Host "Press Enter to exit"
    exit 1
}}

# Install packages
Write-Host "Checking/Installing required packages..." -ForegroundColor Yellow
try {{
    python -m pip install --quiet gradio>=5.0.0 pillow numpy pandas requests openai python-dotenv fastapi uvicorn
    if ($LASTEXITCODE -ne 0) {{
        Write-Host "Warning: Some packages might not have installed correctly" -ForegroundColor Yellow
        Write-Host "The app may still work with existing packages" -ForegroundColor Yellow
    }}
}} catch {{
    Write-Host "Warning: Package installation had issues, continuing anyway..." -ForegroundColor Yellow
}}

# Run the app
Write-Host "Starting Mood Palette Generator..." -ForegroundColor Green
Write-Host ""

$appPath = Join-Path $PSScriptRoot "mood_palette_app.pyz"
try {{
    python $appPath
}} catch {{
    Write-Host "Error running application: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}}
''')
    
    # Step 4: Create README
    readme = dist_dir / "README.txt"
    readme.write_text(f'''Mood Palette Generator - ZIP Portable Version
============================================

This is a lightweight portable version that requires Python to be installed
on the target system but automatically installs all Python dependencies.

REQUIREMENTS:
- Python 3.8 or later installed on the system
- Internet connection (for first run to install packages)

TO RUN:
1. Double-click: MoodPaletteGenerator_ZIP.bat
   OR
2. Run: MoodPaletteGenerator_ZIP.ps1 (PowerShell)

The launcher will:
- Check if Python is installed
- Install required packages automatically
- Start the Mood Palette Generator

ADVANTAGES:
- Very small size (under 1 MB)  
- No Python environment conflicts
- Uses system Python installation
- Automatic dependency management
- Works on any system with Python

FIRST RUN:
- May take a few minutes to install packages
- Requires internet connection
- Subsequent runs are much faster

For systems without Python, recommend installing Python first:
https://www.python.org/downloads/

Make sure to check "Add Python to PATH" during Python installation.

Technical Details:
- Uses ZIP application format (.pyz)
- Installs packages to user's Python environment
- No compilation required
- Cross-platform compatible (with appropriate launcher)
''')
    
    # Step 5: Create Linux launcher (bonus)
    launcher_sh = dist_dir / "mood_palette_generator.sh"
    launcher_sh.write_text(f'''#!/bin/bash
echo "Mood Palette Generator - ZIP Portable Version"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed"
        echo "Please install Python 3.8 or later"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Using: $($PYTHON_CMD --version)"

# Install packages
echo "Checking/Installing required packages..."
$PYTHON_CMD -m pip install --quiet --user gradio>=5.0.0 pillow numpy pandas requests openai python-dotenv fastapi uvicorn

# Run the app
echo "Starting Mood Palette Generator..."
echo ""
$PYTHON_CMD "$(dirname "$0")/mood_palette_app.pyz"
''')
    launcher_sh.chmod(0o755)  # Make executable
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
    size_kb = total_size / 1024
    
    print(f"""
✅ ZIP-based portable app created!

📁 Location: {dist_dir}
📦 Size: {size_kb:.1f} KB (very lightweight!)
🚀 Windows: MoodPaletteGenerator_ZIP.bat
🚀 PowerShell: MoodPaletteGenerator_ZIP.ps1
🚀 Linux/Mac: mood_palette_generator.sh

This approach:
- Requires Python on target system
- Auto-installs all dependencies
- Very small download size
- Most reliable compatibility
- Works across platforms

Perfect for users who have Python installed!
""")
    
    return dist_dir

if __name__ == "__main__":
    create_zip_portable()