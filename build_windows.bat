@echo off
REM ==========================================================================
REM  Build the "Memorandums" app into a Windows .exe.
REM  Result: dist\Memorandums\Memorandums.exe
REM  Requires Python 3.11+ and an internet connection.
REM  (The built app window is fully in Russian.)
REM ==========================================================================

cd /d "%~dp0"

echo.
echo === Building the "Memorandums" application ===
echo.

where python >nul 2>&1
if errorlevel 1 goto nopython

echo [1/4] Enabling proxy support (offline)...
python -m pip install --no-index --find-links vendor pysocks >nul 2>&1

echo [2/4] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 echo     ...retrying with direct connection (no proxy)...
if errorlevel 1 python -m pip install --proxy "" -r requirements.txt
if errorlevel 1 goto failed

python -m pip install pyinstaller
if errorlevel 1 python -m pip install --proxy "" pyinstaller
if errorlevel 1 goto failed

echo.
echo [3/4] Building .exe (takes 1-3 minutes)...
python -m PyInstaller --noconfirm first_step.spec
if errorlevel 1 goto failed

echo.
echo [4/4] Done. Application: dist\Memorandums\Memorandums.exe
echo.
echo On first PDF build the app downloads the render engine
echo (Chromium, ~150 MB) once - internet required.
echo You can move the dist\Memorandums folder to another PC.
echo.
pause
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
echo Install/build failed. See the messages above.
echo If the error mentions connection or proxy, check your internet
echo and try again. As a last resort install Python 3.12 instead of 3.14.
echo.
pause
exit /b 1
