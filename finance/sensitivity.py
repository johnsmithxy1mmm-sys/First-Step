"""Анализ чувствительности (торнадо-диаграмма).

Для каждого стохастического входа считаем IRR в двух крайних точках его
диапазона (low/high), удерживая остальные входы на базовом значении. Размах
IRR между этими точками — вклад допущения в неопределённость результата.
Допущения сортируются по убыванию размаха — это и есть «торнадо».
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List

from .dcf import DCFInputs, build_monthly_cash_flows, irr
from .monte_carlo import _VARIABLE_FIELDS


# Человекочитаемые подписи допущений для диаграммы.
_LABELS = {
    "occupancy_pct": "Загрузка",
    "rental_growth_pct": "Рост ставки аренды",
    "exit_cap_rate_pct": "Cap rate на выходе",
    "exit_price_growth_pct": "Рост цены объекта",
    "rental_rate_rub_month": "Ставка аренды",
    "opex_pct_of_revenue": "Доля OPEX",
    "selling_cost_pct": "Издержки продажи",
    "discount_rate_pct": "Ставка дисконтирования",
}


@dataclass
class TornadoBar:
    name: str          # машинное имя входа
    label: str         # подпись для диаграммы
    low_value: float   # значение входа в нижней точке
    high_value: float  # значение входа в верхней точке
    irr_low: float     # IRR при low_value (годовой, %)
    irr_high: float    # IRR при high_value (годовой, %)
    swing: float       # размах |irr_high - irr_low|


def _endpoints(spec: dict):
    """Возвращает (low, high) крайние точки диапазона для чувствительности."""
    dist = spec["dist"]
    if dist in ("triangular", "uniform"):
        return spec["low"], spec["high"]
    if dist == "normal":
        # ±2 стандартных отклонения охватывают ~95 % массы распределения.
        return spec["mean"] - 2 * spec["sd"], spec["mean"] + 2 * spec["sd"]
    raise ValueError(f"Неизвестное распределение «{dist}».")


def _irr_annual_for(inp: DCFInputs) -> float:
    cash_flows, *_ = build_monthly_cash_flows(inp)
    monthly = irr(cash_flows)
    if monthly != monthly:
        return float("nan")
    return ((1.0 + monthly) ** 12 - 1.0) * 100.0


def run_sensitivity(base: DCFInputs, ranges: Dict[str, dict]) -> List[TornadoBar]:
    """Считает торнадо-вклад каждого допущения, отсортированный по размаху."""
    bars: List[TornadoBar] = []
    for name, spec in ranges.items():
        if name not in _VARIABLE_FIELDS:
            continue
        low, high = _endpoints(spec)
        field = _VARIABLE_FIELDS[name]

        irr_low = _irr_annual_for(replace(base, **{field: low}))
        irr_high = _irr_annual_for(replace(base, **{field: high}))

        swing = abs(irr_high - irr_low)
        bars.append(TornadoBar(
            name=name,
            label=_LABELS.get(name, name),
            low_value=low,
            high_value=high,
            irr_low=irr_low,
            irr_high=irr_high,
            swing=swing,
        ))
    return sorted(bars, key=lambda b: b.swing, reverse=True)
