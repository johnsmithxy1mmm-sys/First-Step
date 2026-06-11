@echo off
chcp 65001 >nul
REM ==========================================================================
REM  Запуск приложения «Меморандумы» из исходников (без сборки .exe).
REM  Первый запуск ставит зависимости; дальше открывает окно сразу.
REM ==========================================================================

python --version >nul 2>&1
if errorlevel 1 (
  echo [ОШИБКА] Python не найден. Установите Python 3.11+ с python.org
  echo          и при установке отметьте «Add python.exe to PATH».
  pause
  exit /b 1
)

REM Ставим зависимости только если ещё не стоят (быстрая проверка одного пакета).
python -c "import playwright" >nul 2>&1
if errorlevel 1 (
  echo Первичная настройка: устанавливаю зависимости...
  python -m pip install -r requirements.txt
  python -m playwright install chromium
)

start "" pythonw app.py
