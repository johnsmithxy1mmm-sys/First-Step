@echo off
REM ==========================================================================
REM  Run the "Memorandums" app from source (no .exe build).
REM  First run installs dependencies, then opens the window.
REM  (The app window itself is fully in Russian.)
REM ==========================================================================

cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 goto nopython

python -c "import playwright" >nul 2>&1
if errorlevel 1 (
  echo First-time setup: installing dependencies, please wait...
  python -m pip install -r requirements.txt
  python -m playwright install chromium
)

start "" pythonw app.py
exit /b 0

:nopython
echo.
echo Python was not found.
echo Install Python 3.11+ from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation.
echo.
pause
exit /b 1
