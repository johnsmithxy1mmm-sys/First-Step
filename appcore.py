"""Ядро графического приложения — без привязки к интерфейсу.

Содержит чистые операции, которыми пользуется окно (app.py): валидация
данных формы, гарантия наличия браузера для рендера, сборка PDF и открытие
готового файла. Вынесено отдельно, чтобы логику можно было тестировать без
запуска GUI.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import yaml

from config.loader import ConfigError, validate_config
from config.schema import ObjectConfig
from report.builder import build_pdf

# Варианты рекомендации для выпадающего списка (код + русская подпись).
REC_OPTIONS = ["ACQUIRE — Приобретать", "HOLD — Удерживать", "AVOID — Воздержаться"]


def slug(name: str) -> str:
    """Имя файла из названия объекта (буквы/цифры/дефис, нижний регистр)."""
    base = re.sub(r"[^\w\-]+", "-", name, flags=re.UNICODE).strip("-").lower()
    return base or "memo"


def config_to_flat(raw: dict) -> dict:
    """Преобразует YAML-конфиг объекта в «плоский» словарь полей формы.

    Поддерживает оба способа оценки выхода (прирост цены / cap rate) и
    подбирает соответствующий стохастический диапазон для блока Монте-Карло.
    """
    o, p, op = raw.get("object", {}), raw.get("purchase", {}), raw.get("operations", {})
    ex, asm = raw.get("exit", {}), raw.get("assumptions", {})
    mc = raw.get("monte_carlo", {})
    ranges = mc.get("ranges", {})
    adv = raw.get("advisor", {})
    inst = p.get("installment", {})

    growth = ex.get("exit_price_growth_pct")
    if growth:
        method, exit_value = "growth", growth
        er = ranges.get("exit_price_growth_pct", {})
    else:
        method, exit_value = "cap", ex.get("exit_cap_rate_pct", "")
        er = ranges.get("exit_cap_rate_pct", {})

    occ = ranges.get("occupancy_pct", {})
    grow = ranges.get("rental_growth_pct", {})
    rec_code = adv.get("recommendation", "HOLD")
    rec = next((x for x in REC_OPTIONS if x.startswith(rec_code)), REC_OPTIONS[1])

    return {
        "name": o.get("name", ""), "developer": o.get("developer", ""),
        "zone": o.get("zone", ""), "type": o.get("type", ""), "area_m2": o.get("area_m2", ""),
        "price_rub": p.get("price_rub", ""), "down_payment_pct": inst.get("down_payment_pct", ""),
        "schedule": [(t.get("month"), t.get("pct")) for t in inst.get("schedule", [])],
        "rental_rate_rub_month": op.get("rental_rate_rub_month", ""),
        "occupancy_pct": op.get("occupancy_pct", ""), "opex_pct_of_revenue": op.get("opex_pct_of_revenue", ""),
        "rental_growth_pct": op.get("rental_growth_pct", ""), "rent_start_month": op.get("rent_start_month", 0),
        "hold_years": ex.get("hold_years", ""), "exit_method": method, "exit_value": exit_value,
        "selling_cost_pct": ex.get("selling_cost_pct", ""),
        "discount_rate_pct": asm.get("discount_rate_pct", ""), "target_irr_pct": asm.get("target_irr_pct", ""),
        "iterations": mc.get("iterations", 10000), "seed": mc.get("seed", 42),
        "occ_low": occ.get("low", ""), "occ_mode": occ.get("mode", ""), "occ_high": occ.get("high", ""),
        "grow_mean": grow.get("mean", ""), "grow_sd": grow.get("sd", ""),
        "exitr_low": er.get("low", ""), "exitr_mode": er.get("mode", ""), "exitr_high": er.get("high", ""),
        "recommendation": rec, "thesis": adv.get("thesis", ""), "exit_scenario": adv.get("exit_scenario", ""),
        "advisor_name": adv.get("name", "Антон Перфилов"),
        "advisor_title": adv.get("title", "Independent Real Estate Counsel"),
    }


def app_dir() -> Path:
    """Каталог приложения.

    В собранном PyInstaller-режиме это папка с .exe (sys.frozen), иначе —
    корень проекта. Используется как база для выходной папки out/.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def config_from_form(data: dict) -> ObjectConfig:
    """Проверяет словарь, собранный формой, и возвращает валидную конфигурацию.

    Бросает ConfigError с понятным русским текстом, если что-то не так.
    """
    return validate_config(data)


def config_to_yaml(data: dict) -> str:
    """Сериализует словарь полей в читаемый YAML (для кнопки «Сохранить»)."""
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def yaml_to_dict(path: str | Path) -> dict:
    """Читает YAML-файл в словарь (для кнопки «Загрузить»)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Файл пуст или имеет неверную структуру.")
    return raw


def _chromium_available() -> bool:
    """Пытается запустить headless-Chromium; True, если получилось."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
        return True
    except Exception:
        return False


def ensure_chromium(log: Optional[Callable[[str], None]] = None) -> bool:
    """Гарантирует наличие браузера; при отсутствии — скачивает его один раз.

    Возвращает True, если браузер готов. Загрузка требует интернета и
    выполняется только при первом запуске (как и в CLI-режиме).
    """
    def say(msg: str) -> None:
        if log:
            log(msg)

    if _chromium_available():
        return True

    say("Первый запуск: загружаю движок рендера (Chromium, ~150 МБ)…")
    try:
        if getattr(sys, "frozen", False):
            # В собранном .exe нет «python -m playwright»: зовём node-драйвер,
            # который Playwright кладёт рядом с собой (включён в сборку).
            from playwright._impl._driver import compute_driver_executable
            driver = compute_driver_executable()
            subprocess.run([*map(str, driver), "install", "chromium"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
                           check=True)
    except Exception as exc:  # noqa: BLE001
        say(f"Не удалось загрузить браузер: {exc}")
        return False

    ok = _chromium_available()
    say("Движок рендера готов." if ok else "Браузер всё ещё недоступен.")
    return ok


def generate(data: dict, out_path: str | Path, date: Optional[_dt.date] = None,
             log: Optional[Callable[[str], None]] = None) -> Path:
    """Полный цикл: валидация формы → гарантия браузера → сборка PDF.

    Возвращает путь к готовому PDF. Любая проблема со входом поднимается как
    ConfigError с русским текстом.
    """
    def say(msg: str) -> None:
        if log:
            log(msg)

    cfg = config_from_form(data)  # ConfigError при кривом вводе
    say("Данные приняты, считаю модель…")

    if not ensure_chromium(log):
        raise RuntimeError(
            "Движок рендера (Chromium) недоступен. Проверьте подключение к интернету "
            "и повторите — при первом запуске он скачивается один раз."
        )

    say("Собираю PDF…")
    result = build_pdf(cfg, out_path, date or _dt.date.today())
    say(f"Готово: {result}")
    return result


def open_file(path: str | Path) -> None:
    """Открывает файл в системном приложении (Windows / macOS / Linux)."""
    path = str(path)
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)
