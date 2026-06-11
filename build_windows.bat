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

echo [1/3] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
if errorlevel 1 goto failed

echo.
echo [2/3] Building .exe (takes 1-3 minutes)...
python -m PyInstaller --noconfirm first_step.spec
if errorlevel 1 goto failed

echo.
echo [3/3] Done. Application: dist\Memorandums\Memorandums.exe
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
echo Build failed. See the messages above.
echo.
pause
exit /b 1
