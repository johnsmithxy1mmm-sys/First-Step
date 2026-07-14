"""Momentum signal: informed flow in the tail.

A sharp price rise of a cheap outcome amid a volume spike suggests someone
with information is accumulating ahead of news. The signal is weak (low
confidence) and only additive: we do not short a price drop.
"""

from __future__ import annotations

from ..models import Candidate, Signal


class MomentumSignal:
    name = "momentum"

    # How much momentum can lift the estimate relative to the market.
    MAX_BOOST = 1.8

    def evaluate(self, candidate: Candidate) -> Signal | None:
        m = candidate.market
        if candidate.p_mkt <= 0 or candidate.outcome_index not in (0, 1):
            return None

        # oneDayPriceChange in Gamma is for the first outcome; for No the sign flips.
        change = m.one_day_price_change if candidate.outcome_index == 0 else -m.one_day_price_change
        if change <= 0:
            return None

        rel_move = change / candidate.p_mkt              # +0.01 on a 0.02 price = +50%
        vol_ratio = m.volume_24h_usd / m.volume_usd if m.volume_usd > 0 else 0.0
        vol_spike = min(vol_ratio / 0.05, 1.0)           # 5% turnover in a day = max

        strength = min(rel_move, 1.0) * vol_spike
        if strength < 0.1:
            return None

        boost = 1.0 + (self.MAX_BOOST - 1.0) * strength
        p_est = min(candidate.p_mkt * boost, 0.999)
        return Signal(
            name=self.name,
            p_est=p_est,
            confidence=0.15 + 0.25 * strength,           # 0.15..0.40
            rationale=f"price +{change:.3f} ({rel_move * 100:.0f}% of p_mkt), "
                      f"vol24h/vol={vol_ratio:.3f} -> boost x{boost:.2f}",
        )
