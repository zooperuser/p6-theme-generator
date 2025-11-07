# Image Palette Generator: Quick Instructions

This project generates palettes from uploaded images (random image pixels + mirrored variant).
Install dependencies in a virtual environment and run the app locally.

---

Quick Run / Shortcut Instructions (Windows)

Option 1: PowerShell script

  1. Right-click `run_app.ps1` > Run with PowerShell (or from a PowerShell terminal: `./run_app.ps1`).
  2. Script auto-creates `.venv`, installs dependencies (unless you pass `-NoInstall`), then launches `app.py`.
  3. To force dependency upgrade: `./run_app.ps1 -Upgrade`.

Execution Policy Note: If blocked, run (once, as Admin):
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Option 2: CMD batch file

  1. Double-click `run_app.cmd` in Explorer, or run it from `cmd.exe`.
  2. It creates `.venv` if missing, installs dependencies, and starts the app.

Option 3: Desktop Shortcut

  1. Right-click `run_app.cmd` > Send to > Desktop (create shortcut). (Simplest: no policy issues.)
     OR create a shortcut to PowerShell for richer flags:
     Target: `powershell.exe -NoLogo -NoExit -ExecutionPolicy Bypass -File "D:\20250905_01_Mood Pallete\run_app.ps1"`
  2. (Optional) Change the icon: Right-click shortcut > Properties > Change Icon.
  3. Double-click the shortcut to launch the app.

One-liner (PowerShell) without activating env (from repo root):

  .\.venv\Scripts\python.exe app.py

If `.venv` does not exist yet:

  py -3 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt; .\.venv\Scripts\python.exe app.py

One-liner (CMD):

  .venv\Scripts\python.exe app.py

Create venv + install + run (CMD):

  py -3 -m venv .venv && .venv\Scripts\python.exe -m pip install -r requirements.txt && .venv\Scripts\python.exe app.py

Optional: Pin to Taskbar

  1. Create a shortcut as above (CMD version recommended).
  2. Right-click shortcut > Pin to taskbar.

Troubleshooting

- If Gradio port already in use, edit `ui.launch(server_port=7861)` inside `app.py` (or pick any free port).
- If clipboard copy doesn’t work, ensure browser permissions allow clipboard access.
- If you see `h11 LocalProtocolError: Too little data for declared Content-Length`, it's a brotli compression quirk.
  The app includes a middleware that disables brotli by stripping `br` from `Accept-Encoding`.
  Make sure you're using the pinned Gradio version from `requirements.txt` and restart the app.
