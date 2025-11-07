"""
Portable Python Environment Creator for Mood Palette Generator
Creates a self-contained Python environment that can run anywhere
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def create_portable_python_app():
    """Create a portable Python application bundle."""
    
    # Configuration
    app_name = "MoodPaletteGenerator_Portable"
    current_dir = Path.cwd()
    dist_dir = current_dir / "dist_portable"
    app_dir = dist_dir / app_name
    
    print(f"Creating portable Python app: {app_name}")
    
    # Clean previous build
    if dist_dir.exists():
        print("Cleaning previous build...")
        shutil.rmtree(dist_dir)
    
    # Create directories
    app_dir.mkdir(parents=True, exist_ok=True)
    python_dir = app_dir / "python"
    app_files_dir = app_dir / "app"
    
    print("Creating directory structure...")
    python_dir.mkdir(exist_ok=True)
    app_files_dir.mkdir(exist_ok=True)
    
    # Step 1: Create a minimal virtual environment
    print("Creating minimal Python environment...")
    venv_path = current_dir / ".venv_portable"
    
    # Create fresh venv for portable use
    if venv_path.exists():
        shutil.rmtree(venv_path)
    
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    
    # Get paths
    if os.name == 'nt':  # Windows
        venv_python = venv_path / "Scripts" / "python.exe"
        venv_pip = venv_path / "Scripts" / "pip.exe"
    else:  # Unix-like
        venv_python = venv_path / "bin" / "python"
        venv_pip = venv_path / "bin" / "pip"
    
    # Step 2: Install only required packages
    print("Installing required packages...")
    required_packages = [
        "gradio>=5.0.0",
        "pillow",
        "numpy",
        "pandas",
        "requests",
        "openai",
        "python-dotenv",
        "uvicorn",
        "fastapi",
    ]
    
    for package in required_packages:
        print(f"Installing {package}...")
        subprocess.run([str(venv_pip), "install", package], check=True)
    
    # Step 3: Copy Python runtime
    print("Copying Python runtime...")
    if os.name == 'nt':  # Windows
        # Copy essential Python files
        python_exe_src = venv_path / "Scripts" / "python.exe"
        python_exe_dst = python_dir / "python.exe"
        shutil.copy2(python_exe_src, python_exe_dst)
        
        # Copy DLLs directory
        dlls_src = Path(sys.executable).parent / "DLLs"
        if dlls_src.exists():
            dlls_dst = python_dir / "DLLs"
            shutil.copytree(dlls_src, dlls_dst, dirs_exist_ok=True)
        
        # Copy base Python standard library, then overlay venv site-packages
        base_lib_src = Path(sys.base_prefix) / "Lib"
        lib_dst = python_dir / "Lib"
        if base_lib_src.exists():
            shutil.copytree(base_lib_src, lib_dst, dirs_exist_ok=True)
        # Ensure site-packages from venv is included
        venv_site_packages = venv_path / "Lib" / "site-packages"
        if venv_site_packages.exists():
            shutil.copytree(venv_site_packages, lib_dst / "site-packages", dirs_exist_ok=True)

        # Copy pyvenv.cfg from venv root so python.exe can locate its stdlib
        venv_cfg = venv_path / "pyvenv.cfg"
        if venv_cfg.exists():
            shutil.copy2(venv_cfg, python_dir / "pyvenv.cfg")

        # Copy pythonXY.dll from base Python (required by python.exe on Windows)
        dll_name = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        # Try a few common locations in order
        candidate_dll_paths = [
            Path(sys.base_prefix) / dll_name,
            Path(sys.prefix) / dll_name,
            Path(sys.executable).parent / dll_name,
        ]
        for cand in candidate_dll_paths:
            if cand.exists():
                shutil.copy2(cand, python_dir / dll_name)
                break
    
    # Step 4: Copy application files
    print("Copying application files...")
    app_files = [
        "app.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file in app_files:
        src = current_dir / file
        if src.exists():
            dst = app_files_dir / file
            shutil.copy2(src, dst)
    
    # Step 5: Create launcher scripts
    print("Creating launcher scripts...")
    
    # Windows batch launcher
    launcher_bat = app_dir / f"{app_name}.bat"
    launcher_bat.write_text(f"""@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0\\app;%~dp0\\python\\Lib\\site-packages
set PYTHONHOME=%~dp0\\python
"%~dp0\\python\\python.exe" "%~dp0\\app\\app.py"
pause
""")
    
    # PowerShell launcher
    launcher_ps1 = app_dir / f"{app_name}.ps1"
    launcher_ps1.write_text(f"""# Mood Palette Generator Portable Launcher
$ErrorActionPreference = "Stop"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Set environment
$env:PYTHONPATH = "$ScriptDir\\app;$ScriptDir\\python\\Lib\\site-packages"
$env:PYTHONHOME = "$ScriptDir\\python"

# Launch app
Write-Host "Starting Mood Palette Generator..."
Write-Host "Script Directory: $ScriptDir"

try {{
    & "$ScriptDir\\python\\python.exe" "$ScriptDir\\app\\app.py"
}} catch {{
    Write-Error "Failed to start application: $_"
    Read-Host "Press Enter to exit"
}}
""")
    
    # Create simple executable launcher
    launcher_exe_py = app_dir / "launcher.py"
    launcher_exe_py.write_text("""#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory of this launcher
    launcher_dir = Path(__file__).parent
    
    # Set up environment
    python_exe = launcher_dir / "python" / "python.exe"
    app_script = launcher_dir / "app" / "app.py"
    
    # Set Python path
    python_path = f"{launcher_dir / 'app'};{launcher_dir / 'python' / 'Lib' / 'site-packages'}"
    env = os.environ.copy()
    env['PYTHONPATH'] = python_path
    env['PYTHONHOME'] = str(launcher_dir / 'python')
    
    print("Starting Mood Palette Generator...")
    print(f"Python: {python_exe}")
    print(f"App: {app_script}")
    
    # Launch the app
    try:
        subprocess.run([str(python_exe), str(app_script)], env=env, check=True)
    except Exception as e:
        print(f"Error starting app: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
""")
    
    # Step 6: Create README
    readme = app_dir / "README.txt"
    readme.write_text(f"""Mood Palette Generator - Portable Version
========================================

This is a portable version of the Mood Palette Generator that includes
a complete Python environment and all dependencies.

To run the application:

Windows:
1. Double-click {app_name}.bat
   OR
2. Right-click {app_name}.ps1 and select "Run with PowerShell"

The application will start automatically and open in your web browser.

Requirements:
- Windows 10 or later
- No Python installation required
- No additional dependencies required

Folder Structure:
- python/     - Portable Python runtime
- app/        - Application files
- {app_name}.bat - Windows launcher
- {app_name}.ps1 - PowerShell launcher

For support or issues, please refer to the main project documentation.
""")
    
    # Step 7: Clean up temporary venv
    print("Cleaning up...")
    if venv_path.exists():
        shutil.rmtree(venv_path)
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in app_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print(f"""
✅ Portable Python app created successfully!

📁 Location: {app_dir}
📦 Size: {size_mb:.1f} MB
🚀 Launcher: {app_name}.bat or {app_name}.ps1

To distribute:
1. Copy the entire '{app_name}' folder
2. Users can run the .bat or .ps1 file to start the app
3. No Python installation required on target machines
""")
    
    return app_dir

if __name__ == "__main__":
    create_portable_python_app()