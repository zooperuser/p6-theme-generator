"""
Simple Portable App Creator - Direct Python Embedding
Creates a portable app by copying Python and packages directly
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def create_simple_portable():
    """Create a simple portable Python bundle."""
    
    print("Creating Simple Portable Python Bundle...")
    
    # Configuration
    app_name = "MoodPalette_Simple_Portable"
    current_dir = Path.cwd()
    dist_dir = current_dir / "dist_simple"
    app_dir = dist_dir / app_name
    
    # Clean and create
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    app_dir.mkdir(parents=True)
    
    # Get current virtual environment python
    venv_dir = current_dir / ".venv"
    if os.name == 'nt':  # Windows
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_scripts = venv_dir / "Scripts"
        venv_lib = venv_dir / "Lib" / "site-packages"
    
    print(f"Using Python: {venv_python}")
    print(f"Using packages from: {venv_lib}")
    
    # Step 1: Copy Python executable and essential DLLs
    print("Copying Python runtime...")
    
    # Copy python.exe
    if venv_python.exists():
        shutil.copy2(venv_python, app_dir / "python.exe")
    
    # Copy essential DLLs from main Python installation
    main_python_dir = Path(sys.executable).parent
    essential_files = [
        f"python{sys.version_info.major}{sys.version_info.minor}.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll"
    ]
    
    for dll in essential_files:
        dll_path = main_python_dir / dll
        if dll_path.exists():
            shutil.copy2(dll_path, app_dir / dll.name)
            print(f"Copied {dll}")
    
    # Copy DLLs directory
    dlls_src = main_python_dir / "DLLs"
    if dlls_src.exists():
        dlls_dst = app_dir / "DLLs"
        shutil.copytree(dlls_src, dlls_dst)
        print("Copied DLLs directory")
    
    # Step 2: Copy site-packages (only what we need)
    print("Copying required packages...")
    packages_dir = app_dir / "Lib" / "site-packages"
    packages_dir.mkdir(parents=True)
    
    # Copy specific packages we need
    required_packages = [
        "gradio*", "PIL*", "numpy*", "pandas*", "fastapi*", "starlette*", 
        "uvicorn*", "pydantic*", "jinja2*", "httpx*", "anyio*", "sniffio*",
        "safehttpx*", "groovy*", "aiofiles*", "python_multipart*", 
        "websockets*", "orjson*", "brotli*", "markupsafe*", "click*",
        "h11*", "httpcore*", "certifi*", "idna*", "charset_normalizer*",
        "urllib3*", "requests*", "typing_extensions*", "pyyaml*", "packaging*",
        "filelock*", "huggingface_hub*", "tqdm*", "fsspec*", "semantic_version*"
    ]
    
    if venv_lib.exists():
        for item in venv_lib.iterdir():
            for pattern in required_packages:
                package_name = pattern.rstrip("*")
                if item.name.lower().startswith(package_name.lower()):
                    dst = packages_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst, ignore_dangling_symlinks=True)
                    else:
                        shutil.copy2(item, dst)
                    print(f"Copied package: {item.name}")
                    break
    
    # Step 3: Copy app files
    print("Copying app files...")
    shutil.copy2("app.py", app_dir / "app.py")
    
    # Step 4: Create simple launcher
    launcher_bat = app_dir / f"{app_name}.bat"
    launcher_bat.write_text(f"""@echo off
cd /d "%~dp0"

echo Starting Mood Palette Generator...
echo Working Directory: %CD%

REM Set Python path to include our packages
set PYTHONPATH=%~dp0;%~dp0\\Lib\\site-packages

REM Disable bytecode generation to avoid permission issues  
set PYTHONDONTWRITEBYTECODE=1

REM Run the app
"%~dp0\\python.exe" "%~dp0\\app.py"

REM Keep console open on error
if errorlevel 1 (
    echo.
    echo Error occurred. Press any key to exit.
    pause >nul
)
""")
    
    # PowerShell launcher
    launcher_ps1 = app_dir / f"{app_name}.ps1"
    launcher_ps1.write_text(f"""# Simple Portable Mood Palette Generator Launcher
$ErrorActionPreference = "Stop"

Write-Host "Starting Mood Palette Generator..." -ForegroundColor Green

# Get script directory
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $AppDir

# Set Python environment
$env:PYTHONPATH = "$AppDir;$AppDir\\Lib\\site-packages"
$env:PYTHONDONTWRITEBYTECODE = "1"

Write-Host "App Directory: $AppDir" -ForegroundColor Cyan
Write-Host "Python Path: $env:PYTHONPATH" -ForegroundColor Cyan

try {{
    Write-Host "Launching app..." -ForegroundColor Yellow
    & "$AppDir\\python.exe" "$AppDir\\app.py"
}} catch {{
    Write-Error "Failed to start: $_"
    Read-Host "Press Enter to exit"
}}
""")
    
    # Create README
    readme = app_dir / "README.txt"
    readme.write_text(f"""Mood Palette Generator - Simple Portable Version
================================================

This is a simple portable version that includes Python and all dependencies.

To run:
- Double-click: {app_name}.bat
- Or run: {app_name}.ps1

The app will start and open in your web browser automatically.

Size: Approximately 350-400 MB
Compatibility: Windows 10+ (64-bit)

No Python installation required on target machines.

Troubleshooting:
- If you get permission errors, try running as administrator
- If DLL errors occur, install Visual C++ Redistributable 2019+
- Check Windows Defender isn't blocking the executable

For support, see the main project documentation.
""")
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in app_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print(f"""
✅ Simple portable app created!

📁 Location: {app_dir}
📦 Size: {size_mb:.1f} MB
🚀 Run: {app_name}.bat or {app_name}.ps1

This version uses direct Python embedding without virtual environments.
""")
    
    return app_dir

if __name__ == "__main__":
    create_simple_portable()