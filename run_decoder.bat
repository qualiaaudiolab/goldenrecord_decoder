@echo off
rem ---------------------------------------------------------------------------
rem  Golden Record Decoder - start from source on Windows.
rem  First run: creates a private Python environment (.venv) and installs the
rem  required packages. Later runs start the app directly.
rem  Any file dropped onto this .bat is opened in the app.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Setting up Python environment, this happens only once...
    python -m venv .venv || (
        echo.
        echo Python was not found. Install Python 3.10 or newer from python.org
        echo and tick "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e . || (
        echo Installation failed - see messages above.
        pause
        exit /b 1
    )
)

start "" ".venv\Scripts\pythonw.exe" -m grdecoder %*
