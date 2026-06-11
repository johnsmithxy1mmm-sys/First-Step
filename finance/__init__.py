"""Финансовый движок: детерминированная математика DCF + Монте-Карло.

Чистые функции, покрытые тестами. Никаких побочных эффектов и зашитых чисел —
все параметры приходят из конфига объекта.
"""

from .dcf import DCFInputs, DCFResult, compute_dcf, irr, npv
from .installment import build_installment, schedule_pct_sum
from .monte_carlo import MonteCarloResult, run_monte_carlo
from .sensitivity import TornadoBar, run_sensitivity

__all__ = [
    "DCFInputs",
    "DCFResult",
    "compute_dcf",
    "irr",
    "npv",
    "build_installment",
    "schedule_pct_sum",
    "MonteCarloResult",
    "run_monte_carlo",
    "TornadoBar",
    "run_sensitivity",
]
