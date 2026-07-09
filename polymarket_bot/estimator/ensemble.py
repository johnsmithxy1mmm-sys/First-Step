"""Ансамбль: взвешенное геометрическое среднее сигналов с confidence-весами.

Рыночная цена всегда участвует как якорный сигнал с весом
market_anchor_confidence — чтобы перевесить рынок, сигналы должны быть
одновременно уверенными и расходиться с ценой. Это защита от
favorite-longshot bias: по умолчанию мы верим рынку.
"""

from __future__ import annotations

import math

from ..models import Candidate, Estimate, Signal

P_FLOOR = 1e-4
P_CEIL = 0.99


def combine(candidate: Candidate, signals: list[Signal],
            market_anchor_confidence: float) -> Estimate:
    anchor = Signal(
        name="market",
        p_est=candidate.p_mkt,
        confidence=market_anchor_confidence,
        rationale="рыночная цена как байесовский якорь",
    )
    active = [anchor] + [s for s in signals if s.p_est is not None and s.confidence > 0]

    total_weight = sum(s.confidence for s in active)
    log_p = sum(
        s.confidence * math.log(min(max(s.p_est, P_FLOOR), P_CEIL))  # type: ignore[arg-type]
        for s in active
    )
    p_est = math.exp(log_p / total_weight) if total_weight > 0 else candidate.p_mkt
    p_est = min(max(p_est, P_FLOOR), P_CEIL)

    return Estimate(
        candidate=candidate,
        p_mkt=candidate.p_mkt,
        p_est=p_est,
        signals=active,
    )
