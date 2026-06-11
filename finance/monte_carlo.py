"""Монте-Карло симуляция распределения IRR.

Стохастические входы задаются в конфиге (блок monte_carlo.ranges). Поддержаны
распределения: triangular, normal, uniform. На каждой итерации сэмплируем
значения, пересобираем DCF и берём годовой IRR. Сид фиксируется для
воспроизводимости (можно переопределить из CLI).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np

from .dcf import DCFInputs, build_monthly_cash_flows, irr


# Поля DCFInputs, которые модель умеет варьировать стохастически.
# Имя во входе -> имя поля в DCFInputs.
_VARIABLE_FIELDS = {
    "occupancy_pct": "occupancy_pct",
    "rental_growth_pct": "rental_growth_pct",
    "exit_cap_rate_pct": "exit_cap_rate_pct",
    "exit_price_growth_pct": "exit_price_growth_pct",
    "rental_rate_rub_month": "rental_rate_rub_month",
    "opex_pct_of_revenue": "opex_pct_of_revenue",
    "selling_cost_pct": "selling_cost_pct",
    "discount_rate_pct": "discount_rate_pct",
}


@dataclass
class MonteCarloResult:
    irr_samples: np.ndarray   # массив годовых IRR (в процентах), без NaN
    iterations: int
    p5: float
    p50: float
    p95: float
    mean: float
    prob_below_target: float  # доля исходов с IRR < target_irr, в процентах
    prob_below_zero: float    # доля исходов с IRR < 0, в процентах
    target_irr_pct: float


def _sample(rng: np.random.Generator, spec: dict) -> float:
    """Один сэмпл из распределения по спецификации диапазона."""
    dist = spec["dist"]
    if dist == "triangular":
        return float(rng.triangular(spec["low"], spec["mode"], spec["high"]))
    if dist == "normal":
        return float(rng.normal(spec["mean"], spec["sd"]))
    if dist == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    raise ValueError(f"Неизвестное распределение «{dist}». Допустимы: triangular, normal, uniform.")


def _irr_annual_for(inp: DCFInputs) -> float:
    """Годовой IRR (в процентах) для набора входов; NaN если не локализован."""
    cash_flows, *_ = build_monthly_cash_flows(inp)
    monthly = irr(cash_flows)
    if monthly != monthly:  # NaN
        return float("nan")
    return ((1.0 + monthly) ** 12 - 1.0) * 100.0


def run_monte_carlo(base: DCFInputs, ranges: Dict[str, dict],
                    iterations: int, target_irr_pct: float,
                    seed: int = 42) -> MonteCarloResult:
    """Прогоняет N итераций, варьируя поля из ranges вокруг базовых значений."""
    rng = np.random.default_rng(seed)

    # Заранее проверим, что все варьируемые поля поддерживаются.
    for name in ranges:
        if name not in _VARIABLE_FIELDS:
            raise ValueError(
                f"Поле «{name}» нельзя варьировать в Монте-Карло. "
                f"Допустимы: {', '.join(sorted(_VARIABLE_FIELDS))}."
            )

    samples: List[float] = []
    for _ in range(iterations):
        overrides = {}
        for name, spec in ranges.items():
            value = _sample(rng, spec)
            # Загрузка не может быть < 0 или > 100; ставки — не отрицательны.
            if name in ("occupancy_pct",):
                value = float(np.clip(value, 0.0, 100.0))
            elif name in ("exit_cap_rate_pct", "rental_rate_rub_month",
                          "opex_pct_of_revenue", "selling_cost_pct", "discount_rate_pct",
                          "exit_price_growth_pct"):
                value = max(value, 1e-6)
            overrides[_VARIABLE_FIELDS[name]] = value
        trial = replace(base, **overrides)
        irr_val = _irr_annual_for(trial)
        if irr_val == irr_val:  # не NaN
            samples.append(irr_val)

    arr = np.array(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("Монте-Карло не дал ни одного валидного IRR — проверьте диапазоны входов.")

    return MonteCarloResult(
        irr_samples=arr,
        iterations=iterations,
        p5=float(np.percentile(arr, 5)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        mean=float(np.mean(arr)),
        prob_below_target=float(np.mean(arr < target_irr_pct) * 100.0),
        prob_below_zero=float(np.mean(arr < 0.0) * 100.0),
        target_irr_pct=target_irr_pct,
    )
