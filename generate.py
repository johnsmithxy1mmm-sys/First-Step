#!/usr/bin/env python3
"""Конвейер: один YAML-конфиг объекта → один PDF-меморандум.

    python generate.py objects/park.yaml
    # -> out/park-memo.pdf

Опции:
    --out PATH     путь к выходному PDF (по умолчанию out/<имя>-memo.pdf)
    --date YYYY-MM-DD  дата меморандума (по умолчанию сегодня);
                       фиксируйте её для байт-в-байт воспроизводимого PDF
    --seed N       переопределить сид Монте-Карло из конфига
    --html         дополнительно сохранить промежуточный HTML рядом с PDF
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

from config.loader import ConfigError, load_config
from report.builder import build_pdf, render_html


def _parse_date(value: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(value)
    except ValueError:
        raise SystemExit(f"Ошибка: дата «{value}» должна быть в формате ГГГГ-ММ-ДД.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Сборка инвестиционного меморандума (PDF) из YAML-конфига объекта.")
    parser.add_argument("config", help="путь к YAML-конфигу объекта, напр. objects/park.yaml")
    parser.add_argument("--out", help="путь к выходному PDF")
    parser.add_argument("--date", help="дата меморандума, ГГГГ-ММ-ДД (по умолчанию сегодня)")
    parser.add_argument("--seed", type=int, help="переопределить сид Монте-Карло")
    parser.add_argument("--html", action="store_true", help="сохранить также промежуточный HTML")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    date = _parse_date(args.date) if args.date else _dt.date.today()

    # Загрузка и валидация конфига — ошибки на русском, без трейсбэков.
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.seed is not None:
        cfg.monte_carlo.seed = args.seed

    out_path = Path(args.out) if args.out else Path("out") / f"{config_path.stem}-memo.pdf"

    if args.html:
        html_path = out_path.with_suffix(".html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_html(cfg, date), encoding="utf-8")
        print(f"HTML:  {html_path}")

    result = build_pdf(cfg, out_path, date)
    print(f"Готово: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
