"""Сигнал моментума: информированный поток в хвосте.

Резкий рост цены дешёвого исхода на фоне всплеска объёма — признак того,
что кто-то с информацией накапливает позицию до новости. Сигнал слабый
(низкая confidence) и только усиливающий: падение цены мы не шортим.
"""

from __future__ import annotations

from ..models import Candidate, Signal


class MomentumSignal:
    name = "momentum"

    # Насколько сильно моментум может поднять оценку относительно рынка.
    MAX_BOOST = 1.8

    def evaluate(self, candidate: Candidate) -> Signal | None:
        m = candidate.market
        if candidate.p_mkt <= 0 or candidate.outcome_index not in (0, 1):
            return None

        # oneDayPriceChange в Gamma задан для первого исхода; для No знак обратный.
        change = m.one_day_price_change if candidate.outcome_index == 0 else -m.one_day_price_change
        if change <= 0:
            return None

        rel_move = change / candidate.p_mkt              # +0.01 к цене 0.02 = +50%
        vol_ratio = m.volume_24h_usd / m.volume_usd if m.volume_usd > 0 else 0.0
        vol_spike = min(vol_ratio / 0.05, 1.0)           # 5% оборота за сутки = максимум

        strength = min(rel_move, 1.0) * vol_spike
        if strength < 0.1:
            return None

        boost = 1.0 + (self.MAX_BOOST - 1.0) * strength
        p_est = min(candidate.p_mkt * boost, 0.999)
        return Signal(
            name=self.name,
            p_est=p_est,
            confidence=0.15 + 0.25 * strength,           # 0.15..0.40
            rationale=f"price +{change:.3f} ({rel_move * 100:.0f}% от p_mkt), "
                      f"vol24h/vol={vol_ratio:.3f} -> boost x{boost:.2f}",
        )
