# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Image Palette Generator
Creates a portable executable with all dependencies bundled
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the current directory
block_cipher = None
current_dir = os.path.abspath('.')

# Collect data files from various packages
datas = []

# Gradio data files
try:
    gradio_datas = collect_data_files('gradio')
    datas.extend(gradio_datas)
except:
    pass

# Gradio client data files
try:
    gradio_client_datas = collect_data_files('gradio_client')
    datas.extend(gradio_client_datas)
except:
    pass

# Transformers data files (for BLIP fallback)
try:
    transformers_datas = collect_data_files('transformers')
    datas.extend(transformers_datas)
except:
    pass

# Tokenizers data files
try:
    tokenizers_datas = collect_data_files('tokenizers')
    datas.extend(tokenizers_datas)
except:
    pass

# Hugging Face Hub data files
try:
    hf_hub_datas = collect_data_files('huggingface_hub')
    datas.extend(hf_hub_datas)
except:
    pass

# PIL/Pillow data files
try:
    pil_datas = collect_data_files('PIL')
    datas.extend(pil_datas)
except:
    pass

# SafeHTTPX data files
try:
    safehttpx_datas = collect_data_files('safehttpx')
    datas.extend(safehttpx_datas)
except:
    pass

# Groovy data files
try:
    groovy_datas = collect_data_files('groovy')
    datas.extend(groovy_datas)
except:
    pass

# Additional Gradio and related data files
try:
    # Collect any other common packages that might have data files
    for pkg in ['fastapi', 'starlette', 'uvicorn', 'pydantic', 'httpx', 'anyio', 'sniffio', 'groovy']:
        try:
            pkg_datas = collect_data_files(pkg)
            datas.extend(pkg_datas)
        except:
            pass
except:
    pass

# Collect hidden imports
hiddenimports = []

# Gradio hidden imports
try:
    gradio_modules = collect_submodules('gradio')
    hiddenimports.extend(gradio_modules)
except:
    pass

# SafeHTTPX hidden imports
try:
    safehttpx_modules = collect_submodules('safehttpx')
    hiddenimports.extend(safehttpx_modules)
except:
    pass

# Groovy hidden imports
try:
    groovy_modules = collect_submodules('groovy')
    hiddenimports.extend(groovy_modules)
except:
    pass

# Common ML/AI hidden imports
hiddenimports.extend([
    'transformers',
    'transformers.models',
    'transformers.models.blip',
    'transformers.models.blip.modeling_blip',
    'transformers.models.blip.configuration_blip',
    'transformers.models.blip.processing_blip',
    'transformers.pipelines',
    'transformers.pipelines.image_to_text',
    'torch',
    'torchvision',
    'numpy',
    'PIL',
    'PIL.Image',
    'PIL.ImageOps',
    'PIL.ImageEnhance',
    'json',
    'urllib',
    'urllib.request',
    'base64',
    'io',
    'warnings',
    'collections',
    'datetime',
    'os',
    're',
    'typing',
    # Additional imports for Gradio and HTTP handling
    'safehttpx',
    'safehttpx._version',
    'groovy',
    'fastapi',
    'starlette',
    'uvicorn',
    'httpx',
    'anyio',
    'sniffio',
])

# Remove duplicates
hiddenimports = list(set(hiddenimports))

a = Analysis(
    ['app.py'],
    pathex=[current_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude some heavy packages we don't need
        'matplotlib',
        'scipy',
        'sklearn',
        'tensorflow',
        'keras',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate entries
a.datas = list(set(a.datas))

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MoodPaletteGenerator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console for release
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MoodPaletteGenerator',
)