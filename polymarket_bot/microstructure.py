"""Microstructure primitives for MM 2.0 (pure, testable, no I/O).

  RealizedVol      — EWMA of midpoint volatility per market (Avellaneda-Stoikov
                     spread/skew both scale with it).
  queue_ahead_usd  — dollar size resting ahead of our price on our side.
  fill_probability — rough P(fill) from queue position and recent flow.
"""

from __future__ import annotations

import math


class RealizedVol:
    """Per-market EWMA of midpoint log-return volatility (per update)."""

    def __init__(self, alpha: float = 0.10):
        self._alpha = alpha
        self._var: dict[str, float] = {}
        self._last_mid: dict[str, float] = {}

    def update(self, market_id: str, mid: float) -> float:
        """Feed a new midpoint; returns the current sigma (stddev of returns)."""
        prev = self._last_mid.get(market_id)
        self._last_mid[market_id] = mid
        if prev is None or prev <= 0 or mid <= 0:
            return math.sqrt(self._var.get(market_id, 0.0))
        ret = math.log(mid / prev)
        var = self._var.get(market_id, 0.0)
        var = (1 - self._alpha) * var + self._alpha * ret * ret
        self._var[market_id] = var
        return math.sqrt(var)

    def sigma(self, market_id: str) -> float:
        return math.sqrt(self._var.get(market_id, 0.0))


def queue_ahead_usd(levels: list, price: float, side: str) -> float:
    """Dollar size resting ahead of a maker order at `price`.

    side 'BUY': other bids at price >= ours are ahead (better/equal price).
    side 'SELL': other asks at price <= ours are ahead.
    """
    total = 0.0
    for level in levels:
        if side == "BUY" and level.price >= price:
            total += level.price * level.size
        elif side == "SELL" and level.price <= price:
            total += level.price * level.size
    return total


def fill_probability(queue_ahead: float, our_size_usd: float,
                     recent_volume_usd: float) -> float:
    """Rough P(our order fills) before we requote.

    Flow must clear the queue ahead plus part of ours. More flow, less queue,
    smaller order -> higher probability. Bounded (0, 1).
    """
    needed = queue_ahead + 0.5 * max(our_size_usd, 1e-9)
    if recent_volume_usd <= 0:
        return 0.0
    return 1.0 - math.exp(-recent_volume_usd / max(needed, 1e-9))
