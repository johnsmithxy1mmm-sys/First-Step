# PyInstaller-спецификация сборки графического приложения в .exe.
#
# Запуск (на Windows):  pyinstaller --noconfirm first_step.spec
# Проще — двойной клик по build_windows.bat (он всё сделает сам).
#
# Собирается папка dist/Memorandums/ с файлом Memorandums.exe.
# Движок Chromium НЕ кладётся внутрь: при первом запуске приложение
# скачивает его один раз (нужен интернет) — так же, как CLI-режим.

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = []
binaries = []
hiddenimports = []

# Ресурсы отчёта: шрифты, бренд-CSS, шаблон — кладём в том же дереве путей.
datas += [
    ("report/assets/brand.css", "report/assets"),
    ("report/assets/fonts/Lora.ttf", "report/assets/fonts"),
    ("report/assets/fonts/Lora-Italic.ttf", "report/assets/fonts"),
    ("report/assets/fonts/OFL.txt", "report/assets/fonts"),
    ("report/templates/memorandum.html.j2", "report/templates"),
    ("objects/park.yaml", "objects"),
    ("objects/polyana.yaml", "objects"),
]

# Playwright тащим целиком (включая node-драйвер) — нужен для рендера PDF.
for pkg in ("playwright", "matplotlib"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Pydantic v2 использует скомпилированное ядро — подстрахуемся хидден-импортами.
hiddenimports += ["pydantic", "pydantic_core", "numpy", "yaml", "jinja2"]


a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Memorandums",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # без чёрного окна консоли
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Memorandums",
)
