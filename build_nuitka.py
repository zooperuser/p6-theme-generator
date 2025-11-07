"""
Nuitka Compiler Script for Mood Palette Generator
Creates a true native executable using Nuitka compiler
"""

import subprocess
import sys
from pathlib import Path

def build_with_nuitka():
    """Build the app using Nuitka compiler."""
    
    print("Building Mood Palette Generator with Nuitka...")
    
    # Nuitka command
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",  # Create standalone executable
        "--follow-imports",  # Follow all imports
        "--include-data-files=.env=.env",  # Include .env if exists
        "--windows-console-mode=attach",  # Show console on Windows
        "--output-dir=dist_nuitka",  # Output directory
        "--output-filename=MoodPaletteGenerator_Nuitka.exe",  # Output filename
        "--remove-output",  # Clean previous builds
        
        # Package includes
        "--include-package=gradio",
        "--include-package=gradio_client", 
        "--include-package=PIL",
        "--include-package=numpy",
        "--include-package=pandas",
        "--include-package=fastapi",
        "--include-package=starlette",
        "--include-package=uvicorn",
        "--include-package=pydantic",
        "--include-package=jinja2",
        "--include-package=httpx",
        "--include-package=anyio",
        "--include-package=safehttpx",
        "--include-package=groovy",
        "--include-package=aiofiles",
        "--include-package=websockets",
        "--include-package=orjson",
        
        # Module includes for dynamic imports
        "--include-module=gradio.components",
        "--include-module=gradio.blocks",
        "--include-module=gradio.interface",
        "--include-module=gradio.themes",
        "--include-module=PIL.Image",
        "--include-module=PIL.ImageOps",
        
        # Disable some optimizations that might break dynamic loading
        "--no-prefer-source-code",
        
        "app.py"  # Main script
    ]
    
    print("Running Nuitka with command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Nuitka build completed successfully!")
        
        # Check output
        dist_dir = Path("dist_nuitka")
        if dist_dir.exists():
            exe_files = list(dist_dir.rglob("*.exe"))
            if exe_files:
                exe_path = exe_files[0]
                print(f"📁 Executable created: {exe_path}")
                print(f"📦 Size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            else:
                print("⚠️ No executable found in output directory")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Nuitka build failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = build_with_nuitka()
    if success:
        print("""
🎉 Build completed! 

To test: 
cd dist_nuitka
./MoodPaletteGenerator_Nuitka.exe

Nuitka creates true native executables that should handle 
dynamic imports better than PyInstaller.
        """)
    else:
        print("Build failed. Check the error messages above.")