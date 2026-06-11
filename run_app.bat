@echo off
REM ==========================================================================
REM  Run the "Memorandums" app from source (no .exe build).
REM  First run installs dependencies, then opens the window.
REM  (The app window itself is fully in Russian.)
REM ==========================================================================

cd /d "%~dp0"

REM Clear proxy variables for this session: a leftover SOCKS proxy makes pip
REM fail with "Missing dependencies for SOCKS support". Does NOT change system.
set ALL_PROXY=
set HTTP_PROXY=
set HTTPS_PROXY=
set all_proxy=
set http_proxy=
set https_proxy=

where python >nul 2>&1
if errorlevel 1 goto nopython

python -c "import playwright" >nul 2>&1
if errorlevel 1 (
  echo First-time setup: installing dependencies, please wait...
  python -m pip install -r requirements.txt
  if errorlevel 1 goto failed
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

:failed
echo.
echo Dependency install failed. See the messages above.
echo If you see a proxy/SOCKS or connection error, you may be behind a
echo required proxy: run "python -m pip install pysocks" once, then retry.
echo.
pause
exit /b 1
