"""
cx_Freeze setup script for Image Palette Generator
Alternative approach to PyInstaller for creating portable executable
"""

import sys
import os
from cx_Freeze import setup, Executable

# Get the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Include files and directories
include_files = []

# Build options
build_exe_options = {
    "packages": [
        "gradio", 
        "gradio_client",
        "PIL", 
        "numpy",
        "uvicorn",
        "fastapi",
        "starlette",
        "httpx",
        "anyio",
        "sniffio",
        "safehttpx",
        "groovy",
        "jinja2",
        "aiofiles",
        "python_multipart",
        "websockets",
        "orjson",
        "pydantic",
        "typing_extensions",
        "certifi",
        "charset_normalizer",
        "idna",
        "urllib3",
        "requests",
        "click",
        "h11",
        "markupsafe",
    ],
    "excludes": [
        # Exclude heavy packages to reduce size
        "matplotlib",
        "scipy",
        "tensorflow",
        "torch",
        "torchvision",
        "jupyterlab",
        "notebook",
        "IPython",
        "pytest",
        "setuptools",
        "distutils",
        "email",
        "html",
        "http.client",
        "urllib.parse",
        "xml"
    ],
    "include_files": include_files,
    "build_exe": os.path.join(current_dir, "dist_cx_freeze"),
    "optimize": 2,
    "include_msvcrt": True,
}

# Target executable
target = Executable(
    script="app.py",
    base=None,  # Use None for console app, "Win32GUI" for windowed
    target_name="ImagePaletteGenerator_CXFreeze.exe",
    icon=None
)

setup(
    name="Image Palette Generator",
    version="1.0.0",
    description="Image-based color palette generator",
    author="Image Palette Team",
    options={"build_exe": build_exe_options},
    executables=[target]
)