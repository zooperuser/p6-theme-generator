# Mood Palette Generator - Portable Version

## Overview
This directory contains the portable version build system for the Mood Palette Generator. The portable version allows users to run the application without installing Python or any dependencies.

## Building the Portable Version

### Prerequisites
1. Windows 10/11 (for Windows portable builds)
2. Python 3.8 or later installed
3. Virtual environment set up (run `.\run_app.ps1` first)

### Build Process

#### Option 1: Quick Build (Recommended)
```powershell
# Run the automated build script
.\build_portable.ps1
```

#### Option 2: Custom Build Options
```powershell
# Clean build with debug console
.\build_portable.ps1 -Clean -Debug

# Create a single-file executable (slower startup, more portable)
.\build_portable.ps1 -OneFile

# Skip dependency installation (if already installed)
.\build_portable.ps1 -SkipInstall
```

### Build Output
The build process creates:
- `dist/` folder containing the portable application
- `MoodPaletteGenerator.exe` (the main executable)
- All required dependencies bundled together
- `Launch_MoodPalette.ps1` (launcher script)

## Distribution

### What to Distribute
Copy the entire `dist/` folder to the target machine. The folder structure should be:
```
MoodPaletteGenerator/
├── MoodPaletteGenerator.exe
├── Launch_MoodPalette.ps1
├── launch_portable.ps1
├── _internal/ (PyInstaller bundled files)
└── PortableData/ (created on first run)
```

### Running the Portable App

#### Method 1: Direct Execution
```powershell
# Navigate to the dist folder and run
.\MoodPaletteGenerator.exe
```

#### Method 2: Using the Launcher (Recommended)
```powershell
# Use the included launcher script
.\launch_portable.ps1

# Or with debug mode
.\launch_portable.ps1 -Debug
```

### Features of the Portable Version

#### Self-Contained
- No Python installation required
- All dependencies bundled
- Works on clean Windows systems

#### Portable Data Storage
- Configuration and logs stored in `PortableData/` folder
- Prompts log saved locally
- No registry modifications

#### LM Studio Integration
- Automatically detects LM Studio on `http://127.0.0.1:1234/v1`
- Can be configured via environment variables
- Supports both embedding and vision models

### Troubleshooting

#### Common Issues

1. **Antivirus Detection**
   - Some antivirus software may flag PyInstaller executables
   - Add the executable to antivirus exceptions
   - This is a false positive common with PyInstaller

2. **Slow Startup**
   - First launch may be slower as Python initializes
   - Subsequent launches should be faster
   - Consider using directory-based build instead of one-file

3. **Missing Models**
   - Ensure LM Studio is running and models are loaded
   - Check the LM Studio server endpoint (default: http://127.0.0.1:1234/v1)
   - Use the model selection dropdown in the app

4. **Permission Issues**
   - Run as administrator if needed
   - Ensure the folder has write permissions for PortableData

#### Debug Mode
```powershell
# Run with console output for debugging
.\launch_portable.ps1 -Debug
```

#### Log Files
Check `PortableData/mood_palette_log.txt` for detailed error messages.

### Build Configuration

#### Customizing the Build
Edit `app_portable.spec` to customize:
- Hidden imports for additional packages
- Data files to include
- Executable properties (icon, version info)
- Exclusions to reduce size

#### Size Optimization
- The build excludes heavy packages like matplotlib, scipy, tensorflow
- Uses UPX compression (if available)
- Remove unused transformers models to reduce size

#### Fixed Issues
- **SafeHTTPX Missing Files**: Added comprehensive data file collection for `safehttpx` and related packages
- **Gradio Dependencies**: Includes all necessary Gradio client data files
- **Build Size**: Approximately 18.4 MB (includes all dependencies)

### Environment Variables
The portable version respects these environment variables:
- `LM_STUDIO_BASE_URL`: LM Studio endpoint (default: http://127.0.0.1:1234/v1)
- `EMBEDDING_MODEL_ID`: Default embedding model name
- `TEXT_MODEL_ID`: Default text/vision model name
- `APPDATA_OVERRIDE`: Custom data directory (set by launcher)

### Performance Notes
- **Directory Build**: Faster startup, larger folder size
- **One-File Build**: Slower startup, single executable file
- **First Run**: May take longer due to PyInstaller extraction
- **ML Models**: Loading transformers models adds startup time

### Security Considerations
- The portable app runs with the same permissions as the user
- No network access except to configured LM Studio endpoint
- All data stored locally in PortableData folder
- No system modifications or registry entries

## Advanced Usage

### Custom Configuration Directory
```powershell
.\launch_portable.ps1 -ConfigDir "C:\MyCustomPath"
```

### Batch Deployment
For enterprise deployment, you can:
1. Build once on a development machine
2. Copy the dist folder to network share
3. Create batch scripts for multiple users
4. Set organization-specific LM Studio endpoints

### Integration with Other Tools
The portable executable can be:
- Called from batch scripts
- Integrated into larger applications
- Deployed via software management systems
- Run as a service (with additional configuration)