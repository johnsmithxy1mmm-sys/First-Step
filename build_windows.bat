@echo off
chcp 65001 >nul
REM ==========================================================================
REM  Сборка приложения «Меморандумы» в .exe (Windows).
REM  Двойной клик по этому файлу — и в папке dist\Меморандумы появится
REM  Меморандумы.exe. Нужен установленный Python 3.11+ и интернет.
REM ==========================================================================

echo.
echo === Сборка приложения «Меморандумы» ===
echo.

REM 1. Проверка Python.
python --version >nul 2>&1
if errorlevel 1 (
  echo [ОШИБКА] Python не найден. Установите Python 3.11+ с python.org
  echo          и при установке отметьте «Add python.exe to PATH».
  pause
  exit /b 1
)

REM 2. Зависимости проекта + PyInstaller.
echo [1/4] Устанавливаю зависимости...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
python -m pip install pyinstaller
if errorlevel 1 ( echo [ОШИБКА] Не удалось установить зависимости. & pause & exit /b 1 )

REM 3. Сборка .exe по спецификации.
echo.
echo [2/4] Собираю .exe (это занимает 1-3 минуты)...
python -m PyInstaller --noconfirm first_step.spec
if errorlevel 1 ( echo [ОШИБКА] Сборка не удалась. & pause & exit /b 1 )

REM 4. Движок рендера (Chromium) — скачается при первом запуске приложения,
REM    либо можно скачать заранее сейчас.
echo.
echo [3/4] Готово. Приложение: dist\Меморандумы\Меморандумы.exe
echo.
echo [4/4] При первом формировании PDF приложение один раз скачает
echo       движок рендера (Chromium, ~150 МБ) — нужен интернет.
echo.
echo Можно перенести папку dist\Меморандумы целиком на другой компьютер.
echo.
pause
