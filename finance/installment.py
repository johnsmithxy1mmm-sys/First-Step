"""График рассрочки: преобразование долей платежей в денежные потоки по месяцам.

Рассрочка задаётся списком траншей вида {month, pct}, где pct — доля от полной
цены объекта. Транши формируют ОТРИЦАТЕЛЬНЫЕ потоки (отток средств инвестора)
в указанные месяцы. Тайминг этих оттоков напрямую влияет на IRR, поэтому он
моделируется по месяцам, а не «одной суммой на входе».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class InstallmentTranche:
    """Один транш рассрочки."""

    month: int  # порядковый месяц от момента сделки (0 = первый платёж)
    pct: float  # доля от полной цены, в процентах
    amount_rub: float  # абсолютная сумма транша в рублях


def build_installment(price_rub: float, schedule: List[dict]) -> List[InstallmentTranche]:
    """Разворачивает доли рассрочки в абсолютные суммы по месяцам.

    schedule — список словарей {month, pct}. Суммы считаются от полной цены
    объекта price_rub. Доли должны в сумме давать 100 % (проверяется на этапе
    валидации конфига, здесь — доверяем входу, но он уже провалидирован).
    """
    tranches = []
    for entry in schedule:
        month = int(entry["month"])
        pct = float(entry["pct"])
        amount = price_rub * pct / 100.0
        tranches.append(InstallmentTranche(month=month, pct=pct, amount_rub=amount))
    # Сортируем по месяцу — потоки должны идти в хронологическом порядке.
    return sorted(tranches, key=lambda t: t.month)


def schedule_pct_sum(schedule: List[dict]) -> float:
    """Сумма долей платежей в процентах — для проверки сходимости к 100 %."""
    return sum(float(entry["pct"]) for entry in schedule)
