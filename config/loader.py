"""Загрузка и валидация YAML-конфига объекта.

Превращает путь к YAML в провалидированную модель ObjectConfig и в готовый
к расчёту DCFInputs. Ошибки парсинга и валидации перехватываются и подаются
читаемым русским текстом — без трейсбэков pydantic в лицо.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from finance.dcf import DCFInputs

from .schema import ObjectConfig


class ConfigError(Exception):
    """Понятная ошибка конфига для показа пользователю."""


def _format_validation_error(exc: ValidationError) -> str:
    """Собирает человекочитаемое сообщение из ошибок pydantic."""
    lines = ["Конфиг объекта не прошёл проверку:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(корень)"
        msg = err["msg"]
        # Убираем технический префикс «Value error, ».
        msg = msg.replace("Value error, ", "")
        lines.append(f"  • {loc}: {msg}")
    return "\n".join(lines)


def load_config(path: str | Path) -> ObjectConfig:
    """Читает YAML и валидирует его в ObjectConfig."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Файл объекта не найден: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Не удалось разобрать YAML «{path}»: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Файл «{path}» пуст или имеет неверную структуру (ожидался словарь полей).")

    return validate_config(raw)


def validate_config(raw: dict) -> ObjectConfig:
    """Валидирует словарь полей в ObjectConfig с понятными ошибками на русском.

    Используется и при загрузке YAML, и графическим приложением (форма → словарь).
    """
    try:
        return ObjectConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def to_dcf_inputs(cfg: ObjectConfig) -> DCFInputs:
    """Собирает DCFInputs из провалидированного конфига."""
    return DCFInputs(
        price_rub=cfg.purchase.price_rub,
        schedule=[{"month": t.month, "pct": t.pct} for t in cfg.purchase.installment.schedule],
        rental_rate_rub_month=cfg.operations.rental_rate_rub_month,
        occupancy_pct=cfg.operations.occupancy_pct,
        opex_pct_of_revenue=cfg.operations.opex_pct_of_revenue,
        rental_growth_pct=cfg.operations.rental_growth_pct,
        rent_start_month=cfg.operations.rent_start_month,
        hold_years=cfg.exit.hold_years,
        exit_cap_rate_pct=cfg.exit.exit_cap_rate_pct if cfg.exit.exit_cap_rate_pct is not None else 0.0,
        selling_cost_pct=cfg.exit.selling_cost_pct,
        exit_price_growth_pct=cfg.exit.exit_price_growth_pct,
        discount_rate_pct=cfg.assumptions.discount_rate_pct,
    )
