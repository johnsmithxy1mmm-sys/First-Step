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
if not errorlevel 1 goto launch

echo First-time setup: installing dependencies, please wait...
python -m pip install --no-index --find-links vendor pysocks >nul 2>&1
python -m pip install -r requirements.txt
if errorlevel 1 echo     ...retrying with direct connection (no proxy)...
if errorlevel 1 python -m pip install --proxy "" -r requirements.txt
if errorlevel 1 goto failed
python -m playwright install chromium

:launch
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
echo If the error mentions connection or proxy, check your internet and retry.
echo As a last resort install Python 3.12 instead of 3.14.
echo.
pause
exit /b 1
